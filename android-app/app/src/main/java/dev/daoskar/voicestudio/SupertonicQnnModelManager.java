package dev.daoskar.voicestudio;

import android.content.Context;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import java.util.Locale;

final class SupertonicQnnModelManager {
    static final String BUNDLE_URL =
            "https://github.com/daoskar/VoiceStudio/releases/download/android-qnn-models-latest/" +
            "VoiceStudio-Supertonic3-QNN-QDQ.tar.gz";
    static final String CHECKSUM_URL =
            BUNDLE_URL + ".sha256";
    static final String VERSION = "qnn-u16u8-4014d02";

    interface Listener {
        void onProgress(String message);
        void onDone(File modelRoot);
        void onError(Throwable error);
    }

    private SupertonicQnnModelManager() {}

    static File root(Context context) {
        return new File(context.getFilesDir(), "models/supertonic3-qnn/" + VERSION);
    }

    static boolean isInstalled(Context context) {
        File root = root(context);
        String[] required = {
                "onnx/duration_predictor.onnx",
                "onnx/text_encoder.onnx",
                "onnx/vector_estimator.onnx",
                "onnx/vocoder.onnx",
                "onnx/tts.json",
                "onnx/unicode_indexer.json",
                "qnn-model-manifest.json"
        };
        for (String rel : required) {
            File file = new File(root, rel);
            if (!file.isFile() || file.length() == 0) return false;
        }
        return true;
    }

    static void downloadAsync(Context context, Listener listener) {
        new Thread(() -> {
            File parent = new File(context.getFilesDir(), "models/supertonic3-qnn");
            File bundle = new File(parent, VERSION + ".tar.gz.part");
            File tempRoot = new File(parent, VERSION + ".tmp");
            File checksumFile = new File(parent, VERSION + ".sha256.part");
            File finalRoot = root(context);

            try {
                if (isInstalled(context)) {
                    listener.onDone(finalRoot);
                    return;
                }

                if (!parent.exists() && !parent.mkdirs()) {
                    throw new IllegalStateException("Cannot create " + parent);
                }

                deleteRecursive(tempRoot);
                if (!tempRoot.mkdirs()) {
                    throw new IllegalStateException("Cannot create " + tempRoot);
                }

                listener.onProgress("Downloading QNN U16/U8 model bundle...");
                download(BUNDLE_URL, bundle, listener);

                listener.onProgress("Downloading checksum...");
                download(CHECKSUM_URL, checksumFile, null);

                listener.onProgress("Verifying SHA-256...");
                String expected = readExpectedChecksum(checksumFile);
                String actual = sha256(bundle);
                if (!expected.equalsIgnoreCase(actual)) {
                    throw new IllegalStateException(
                            "QNN model SHA-256 mismatch: expected=" +
                            expected + " actual=" + actual
                    );
                }

                listener.onProgress("Extracting QNN models...");
                extractTarGz(bundle, tempRoot);

                File manifest = new File(tempRoot, "qnn-model-manifest.json");
                if (!manifest.isFile()) {
                    throw new IllegalStateException("QNN bundle manifest missing");
                }

                deleteRecursive(finalRoot);
                if (!tempRoot.renameTo(finalRoot)) {
                    throw new IllegalStateException("Cannot finalize QNN model directory");
                }

                bundle.delete();
                checksumFile.delete();
                listener.onDone(finalRoot);
            } catch (Throwable error) {
                bundle.delete();
                checksumFile.delete();
                deleteRecursive(tempRoot);
                listener.onError(error);
            }
        }, "supertonic-qnn-download").start();
    }

    private static void download(
            String urlString,
            File out,
            Listener listener
    ) throws Exception {
        HttpURLConnection conn = (HttpURLConnection) new URL(urlString).openConnection();
        conn.setConnectTimeout(20000);
        conn.setReadTimeout(120000);
        conn.setInstanceFollowRedirects(true);
        conn.setRequestProperty("User-Agent", "VoiceStudio-Android/0.1");

        int code = conn.getResponseCode();
        if (code < 200 || code >= 300) {
            throw new IllegalStateException("HTTP " + code + " for QNN model bundle");
        }

        long total = conn.getContentLengthLong();
        long read = 0L;
        long nextReport = 0L;

        try (InputStream in = new BufferedInputStream(conn.getInputStream());
             FileOutputStream fos = new FileOutputStream(out, false)) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) >= 0) {
                fos.write(buffer, 0, n);
                read += n;
                if (listener != null && read >= nextReport) {
                    if (total > 0) {
                        int pct = (int) Math.min(100L, read * 100L / total);
                        listener.onProgress("Downloading QNN models: " + pct + "%");
                    } else {
                        listener.onProgress(
                                String.format(Locale.US, "Downloading QNN models: %.1f MB",
                                        read / 1048576.0)
                        );
                    }
                    nextReport = read + 8L * 1024L * 1024L;
                }
            }
            fos.getFD().sync();
        } finally {
            conn.disconnect();
        }
    }

    private static String readExpectedChecksum(File file) throws Exception {
        byte[] data;
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            data = in.readAllBytes();
        }
        String text = new String(data, java.nio.charset.StandardCharsets.UTF_8).trim();
        if (text.isEmpty()) {
            throw new IllegalStateException("Empty QNN checksum file");
        }
        String token = text.split("\\s+")[0].trim();
        if (!token.matches("[0-9a-fA-F]{64}")) {
            throw new IllegalStateException("Invalid QNN checksum file: " + text);
        }
        return token.toLowerCase(Locale.US);
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) >= 0) {
                digest.update(buffer, 0, n);
            }
        }
        StringBuilder out = new StringBuilder();
        for (byte b : digest.digest()) {
            out.append(String.format(Locale.US, "%02x", b & 0xff));
        }
        return out.toString();
    }

    private static void extractTarGz(File archive, File destination) throws Exception {
        String destinationPath = destination.getCanonicalPath() + File.separator;

        try (
                InputStream fin = new BufferedInputStream(new FileInputStream(archive));
                GzipCompressorInputStream gzip = new GzipCompressorInputStream(fin);
                TarArchiveInputStream tar = new TarArchiveInputStream(gzip)
        ) {
            TarArchiveEntry entry;
            byte[] buffer = new byte[1024 * 1024];

            while ((entry = tar.getNextTarEntry()) != null) {
                String name = entry.getName();
                while (name.startsWith("./")) name = name.substring(2);
                if (name.isEmpty()) continue;

                File out = new File(destination, name);
                String canonical = out.getCanonicalPath();
                if (!canonical.startsWith(destinationPath)) {
                    throw new SecurityException("Unsafe archive path: " + name);
                }

                if (entry.isDirectory()) {
                    if (!out.exists() && !out.mkdirs()) {
                        throw new IllegalStateException("Cannot create " + out);
                    }
                    continue;
                }

                File parent = out.getParentFile();
                if (parent != null && !parent.exists() && !parent.mkdirs()) {
                    throw new IllegalStateException("Cannot create " + parent);
                }

                try (FileOutputStream fos = new FileOutputStream(out, false)) {
                    int n;
                    while ((n = tar.read(buffer)) >= 0) {
                        fos.write(buffer, 0, n);
                    }
                }
            }
        }
    }

    private static void deleteRecursive(File file) {
        if (!file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) deleteRecursive(child);
            }
        }
        file.delete();
    }
}
