package dev.daoskar.voicestudio;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

final class SupertonicModelManager {
    interface Listener {
        void onProgress(String message);
        void onDone(File modelRoot);
        void onError(Throwable error);
    }

    private SupertonicModelManager() {}

    static File root(Context context) {
        return new File(context.getFilesDir(), "models/supertonic3/" + SupertonicModelManifest.REVISION);
    }

    static boolean isInstalled(Context context) {
        File root = root(context);
        for (String path : SupertonicModelManifest.FILES) {
            File f = new File(root, path);
            if (!f.isFile() || f.length() == 0) return false;
        }
        return true;
    }

    static void downloadAsync(Context context, Listener listener) {
        new Thread(() -> {
            try {
                File root = root(context);
                for (int i = 0; i < SupertonicModelManifest.FILES.size(); i++) {
                    String rel = SupertonicModelManifest.FILES.get(i);
                    File out = new File(root, rel);
                    if (out.isFile() && out.length() > 0) {
                        listener.onProgress("Model " + (i + 1) + "/" + SupertonicModelManifest.FILES.size() + ": " + rel + " (cached)");
                        continue;
                    }
                    File parent = out.getParentFile();
                    if (parent != null && !parent.exists() && !parent.mkdirs()) {
                        throw new IllegalStateException("Cannot create " + parent);
                    }
                    File tmp = new File(out.getAbsolutePath() + ".part");
                    listener.onProgress("Downloading " + (i + 1) + "/" + SupertonicModelManifest.FILES.size() + ": " + rel);
                    download(SupertonicModelManifest.BASE + rel, tmp);
                    if (!tmp.renameTo(out)) {
                        throw new IllegalStateException("Cannot finalize " + out);
                    }
                }
                listener.onDone(root);
            } catch (Throwable t) {
                listener.onError(t);
            }
        }, "supertonic-download").start();
    }

    private static void download(String urlString, File out) throws Exception {
        HttpURLConnection conn = (HttpURLConnection) new URL(urlString).openConnection();
        conn.setConnectTimeout(20000);
        conn.setReadTimeout(60000);
        conn.setInstanceFollowRedirects(true);
        conn.setRequestProperty("User-Agent", "VoiceStudio-Android/0.1");
        int code = conn.getResponseCode();
        if (code < 200 || code >= 300) {
            throw new IllegalStateException("HTTP " + code + " for " + urlString);
        }
        try (InputStream in = new BufferedInputStream(conn.getInputStream());
             FileOutputStream fos = new FileOutputStream(out, false)) {
            byte[] buf = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buf)) >= 0) {
                fos.write(buf, 0, n);
            }
            fos.getFD().sync();
        } finally {
            conn.disconnect();
        }
    }
}
