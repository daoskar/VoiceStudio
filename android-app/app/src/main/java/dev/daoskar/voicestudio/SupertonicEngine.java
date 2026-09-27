package dev.daoskar.voicestudio;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

final class SupertonicEngine implements AutoCloseable {
    private static final int STATIC_TEXT_LENGTH = 128;
    private static final int STATIC_LATENT_LENGTH = 128;
    interface Listener {
        void onStatus(String message);
    }

    private final OrtEnvironment env;
    private final OrtSession durationSession;
    private final OrtSession textEncoderSession;
    private final OrtSession vectorSession;
    private final OrtSession vocoderSession;
    private String backendSummary;
    private boolean profileCollected;
    private final long[] unicodeIndexer;
    private final OnnxTensor styleTtl;
    private final OnnxTensor styleDp;
    private final int sampleRate;
    private final int baseChunkSize;
    private final int compressFactor;
    private final int latentDim;
    private final boolean staticQnnMode;

    private SupertonicEngine(
            OrtEnvironment env,
            OrtSession durationSession,
            OrtSession textEncoderSession,
            OrtSession vectorSession,
            OrtSession vocoderSession,
            String backendSummary,
            long[] unicodeIndexer,
            OnnxTensor styleTtl,
            OnnxTensor styleDp,
            int sampleRate,
            int baseChunkSize,
            int compressFactor,
            int latentDim,
            boolean staticQnnMode
    ) {
        this.env = env;
        this.durationSession = durationSession;
        this.textEncoderSession = textEncoderSession;
        this.vectorSession = vectorSession;
        this.vocoderSession = vocoderSession;
        this.backendSummary = backendSummary;
        this.profileCollected = false;
        this.unicodeIndexer = unicodeIndexer;
        this.styleTtl = styleTtl;
        this.styleDp = styleDp;
        this.sampleRate = sampleRate;
        this.baseChunkSize = baseChunkSize;
        this.compressFactor = compressFactor;
        this.latentDim = latentDim;
        this.staticQnnMode = staticQnnMode;
    }

    static SupertonicEngine load(
            File root,
            File fp32ConditioningRoot,
            String voice,
            boolean useQnnStatic,
            Listener listener
    ) throws Exception {
        listener.onStatus("Loading Supertonic-3 ONNX sessions...");
        File onnx = new File(root, "onnx");
        File conditioningOnnx = useQnnStatic && fp32ConditioningRoot != null
                ? new File(fp32ConditioningRoot, "onnx")
                : onnx;
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        OrtSession dp;
        OrtSession te;
        OrtSession ve;
        OrtSession voc;
        String backendSummary;
        String backendDetails;

        if (useQnnStatic) {
            listener.onStatus(QnnRuntime.registrationStatus(env));

            Map<String, Long> textDims = new HashMap<>();
            textDims.put("batch_size", 1L);
            textDims.put("text_length", (long) STATIC_TEXT_LENGTH);

            Map<String, Long> vectorDims = new HashMap<>();
            vectorDims.put("batch_size", 1L);
            vectorDims.put("text_length", (long) STATIC_TEXT_LENGTH);
            vectorDims.put("latent_length", (long) STATIC_LATENT_LENGTH);

            Map<String, Long> vocoderDims = new HashMap<>();
            vocoderDims.put("batch_size", 1L);
            vocoderDims.put("latent_length", (long) STATIC_LATENT_LENGTH);

            listener.onStatus("Language-safe hybrid: FP32 conditioning + QDQ/INT8 synthesis");
            try (OrtSession.SessionOptions conditioningOptions = new OrtSession.SessionOptions()) {
                conditioningOptions.setIntraOpNumThreads(
                        Math.max(2, Runtime.getRuntime().availableProcessors() / 2)
                );
                dp = env.createSession(
                        new File(conditioningOnnx, "duration_predictor.onnx").getAbsolutePath(),
                        conditioningOptions
                );
                te = env.createSession(
                        new File(conditioningOnnx, "text_encoder.onnx").getAbsolutePath(),
                        conditioningOptions
                );
            }

            QnnRuntime.SessionResult veResult = QnnRuntime.createSession(
                    env,
                    new File(onnx, "vector_estimator.onnx").getAbsolutePath(),
                    "vector_estimator",
                    vectorDims,
                    listener
            );
            QnnRuntime.SessionResult vocResult = QnnRuntime.createSession(
                    env,
                    new File(onnx, "vocoder.onnx").getAbsolutePath(),
                    "vocoder",
                    vocoderDims,
                    listener
            );

            ve = veResult.session;
            voc = vocResult.session;

            backendSummary =
                    "DP=CPU-FP32, TE=CPU-FP32, VE=" + veResult.backend +
                    ", VOC=" + vocResult.backend;

            backendDetails =
                    "Model set: language-safe hybrid" +
                    "\nDP: FP32 CPU conditioning" +
                    "\nTE: FP32 CPU conditioning" +
                    "\nVE: " + veResult.detail +
                    "\nVOC: " + vocResult.detail +
                    "\n" + describeDims("DP", dp) +
                    "\n" + describeDims("TE", te) +
                    "\n" + describeDims("VE", ve) +
                    "\n" + describeDims("VOC", voc) +
                    "\nStatic HTP buckets: text=" + STATIC_TEXT_LENGTH +
                    ", latent=" + STATIC_LATENT_LENGTH;
        } else {
            listener.onStatus("Using dynamic FP32 CPU fallback");
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(
                        Math.max(2, Runtime.getRuntime().availableProcessors() / 2)
                );
                dp = env.createSession(
                        new File(onnx, "duration_predictor.onnx").getAbsolutePath(),
                        options
                );
                te = env.createSession(
                        new File(onnx, "text_encoder.onnx").getAbsolutePath(),
                        options
                );
                ve = env.createSession(
                        new File(onnx, "vector_estimator.onnx").getAbsolutePath(),
                        options
                );
                voc = env.createSession(
                        new File(onnx, "vocoder.onnx").getAbsolutePath(),
                        options
                );
            }

            backendSummary = "DP=CPU, TE=CPU, VE=CPU, VOC=CPU";
            backendDetails =
                    "Model set: FP32 dynamic CPU" +
                    "\n" + describeDims("DP", dp) +
                    "\n" + describeDims("TE", te) +
                    "\n" + describeDims("VE", ve) +
                    "\n" + describeDims("VOC", voc);
        }

        JSONObject cfg = new JSONObject(readText(new File(onnx, "tts.json")));
        JSONObject ae = cfg.getJSONObject("ae");
        JSONObject ttl = cfg.getJSONObject("ttl");
        int sr = ae.getInt("sample_rate");
        int base = ae.getInt("base_chunk_size");
        int comp = ttl.getInt("chunk_compress_factor");
        int ldim = ttl.getInt("latent_dim");

        long[] indexer = jsonLongArray(
                new JSONArray(readText(new File(onnx, "unicode_indexer.json")))
        );

        JSONObject voiceJson = new JSONObject(
                readText(new File(root, "voice_styles/" + voice + ".json"))
        );
        OnnxTensor ttlStyle = styleTensor(env, voiceJson.getJSONObject("style_ttl"));
        OnnxTensor dpStyle = styleTensor(env, voiceJson.getJSONObject("style_dp"));

        listener.onStatus("Supertonic-3 loaded");
        listener.onStatus("Backends: " + backendSummary + "\n" + backendDetails);

        return new SupertonicEngine(
                env,
                dp,
                te,
                ve,
                voc,
                backendSummary + "\n" + backendDetails,
                indexer,
                ttlStyle,
                dpStyle,
                sr,
                base,
                comp,
                ldim,
                useQnnStatic
        );
    }

    float[] synthesize(String rawText, String language, int steps, float speed, Listener listener) throws Exception {
        String text = preprocess(rawText, language);
        listener.onStatus("Tokenizing...");
        int[] cps = text.codePoints().toArray();
        final int actualTextLength = cps.length;

        long[][] ids;
        float[][][] textMask;

        if (staticQnnMode) {
            if (cps.length > STATIC_TEXT_LENGTH) {
                throw new IllegalArgumentException(
                        "Text is too long for HTP bucket: " + cps.length +
                        " > " + STATIC_TEXT_LENGTH + " characters"
                );
            }
            ids = tokenizePadded(cps, STATIC_TEXT_LENGTH);
            textMask = new float[1][1][STATIC_TEXT_LENGTH];
            Arrays.fill(textMask[0][0], 0, actualTextLength, 1.0f);
        } else {
            ids = tokenizePadded(cps, actualTextLength);
            textMask = new float[1][1][actualTextLength];
            Arrays.fill(textMask[0][0], 1.0f);
        }

        try (
                OnnxTensor idsTensor = longTensor(ids);
                OnnxTensor textMaskTensor = floatTensor(textMask)
        ) {
            listener.onStatus("Predicting duration...");
            Map<String, OnnxTensor> dpInputs = new HashMap<>();
            dpInputs.put("text_ids", idsTensor);
            dpInputs.put("style_dp", styleDp);
            dpInputs.put("text_mask", textMaskTensor);

            float duration;
            try (OrtSession.Result result = durationSession.run(dpInputs)) {
                duration = firstFloat(result.get(0).getValue()) / Math.max(0.7f, Math.min(2.0f, speed));
            }
            if (!Float.isFinite(duration) || duration <= 0f) {
                throw new IllegalStateException("Invalid predicted duration: " + duration);
            }

            listener.onStatus("Encoding text...");
            Map<String, OnnxTensor> teInputs = new HashMap<>();
            teInputs.put("text_ids", idsTensor);
            teInputs.put("style_ttl", styleTtl);
            teInputs.put("text_mask", textMaskTensor);

            try (OrtSession.Result textResult = textEncoderSession.run(teInputs)) {
                OnnxTensor textEmb = (OnnxTensor) textResult.get(0);

                int chunk = baseChunkSize * compressFactor;
                long wavLength = Math.max(1L, (long) (duration * sampleRate));
                int actualLatentLength = (int) ((wavLength + chunk - 1) / chunk);
                int runtimeLatentLength = actualLatentLength;

                if (staticQnnMode) {
                    if (actualLatentLength > STATIC_LATENT_LENGTH) {
                        throw new IllegalArgumentException(
                                "Audio is too long for HTP bucket: latent " +
                                actualLatentLength + " > " + STATIC_LATENT_LENGTH
                        );
                    }
                    runtimeLatentLength = STATIC_LATENT_LENGTH;
                }

                int channels = latentDim * compressFactor;
                float[][][] latent = randomLatent(channels, runtimeLatentLength);
                float[][][] latentMask = new float[1][1][runtimeLatentLength];

                if (staticQnnMode) {
                    Arrays.fill(latentMask[0][0], 0, actualLatentLength, 1.0f);
                } else {
                    Arrays.fill(latentMask[0][0], 1.0f);
                }

                int totalSteps = Math.max(2, Math.min(12, steps));
                try (OnnxTensor totalStepTensor = OnnxTensor.createTensor(env, new float[]{totalSteps})) {
                    for (int step = 0; step < totalSteps; step++) {
                        listener.onStatus("Denoising " + (step + 1) + "/" + totalSteps + "...");
                        try (
                                OnnxTensor latentTensor = floatTensor(latent);
                                OnnxTensor latentMaskTensor = floatTensor(latentMask);
                                OnnxTensor maskTensor = floatTensor(textMask);
                                OnnxTensor currentStep = OnnxTensor.createTensor(env, new float[]{step})
                        ) {
                            Map<String, OnnxTensor> inputs = new HashMap<>();
                            inputs.put("noisy_latent", latentTensor);
                            inputs.put("text_emb", textEmb);
                            inputs.put("style_ttl", styleTtl);
                            inputs.put("latent_mask", latentMaskTensor);
                            inputs.put("text_mask", maskTensor);
                            inputs.put("current_step", currentStep);
                            inputs.put("total_step", totalStepTensor);
                            try (OrtSession.Result out = vectorSession.run(inputs)) {
                                latent = toFloat3(out.get(0).getValue());
                            }
                        }
                    }
                }

                listener.onStatus("Vocoding...");
                try (OnnxTensor finalLatent = floatTensor(latent)) {
                    Map<String, OnnxTensor> vocInputs = new HashMap<>();
                    vocInputs.put("latent", finalLatent);
                    try (OrtSession.Result out = vocoderSession.run(vocInputs)) {
                        float[] wav = flattenAudio(out.get(0).getValue());
                        int expected = Math.min(wav.length, Math.max(1, (int) (duration * sampleRate)));
                        float[] result = Arrays.copyOf(wav, expected);
                        collectProfilesOnce();
                        return result;
                    }
                }
            }
        }
    }

    void play(float[] wav) {
        short[] pcm = new short[wav.length];
        for (int i = 0; i < wav.length; i++) {
            float v = Math.max(-1f, Math.min(1f, wav[i]));
            pcm[i] = (short) Math.round(v * 32767f);
        }
        AudioTrack track = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build())
                .setAudioFormat(new AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build())
                .setTransferMode(AudioTrack.MODE_STATIC)
                .setBufferSizeInBytes(pcm.length * 2)
                .build();
        track.write(pcm, 0, pcm.length);
        track.setNotificationMarkerPosition(pcm.length);
        track.setPlaybackPositionUpdateListener(new AudioTrack.OnPlaybackPositionUpdateListener() {
            @Override public void onMarkerReached(AudioTrack audioTrack) {
                audioTrack.release();
            }
            @Override public void onPeriodicNotification(AudioTrack audioTrack) {}
        });
        track.play();
    }

    int getSampleRate() {
        return sampleRate;
    }

    String getBackendSummary() {
        return backendSummary;
    }

    private void collectProfilesOnce() {
        if (profileCollected || !staticQnnMode) return;
        profileCollected = true;

        String dpProfile = QnnRuntime.readOrtProfile(durationSession);
        String teProfile = QnnRuntime.readOrtProfile(textEncoderSession);
        String veProfile = QnnRuntime.readOrtProfile(vectorSession);
        String vocProfile = QnnRuntime.readOrtProfile(vocoderSession);

        backendSummary = backendSummary +
                "\nDP profile: " + dpProfile +
                "\nTE profile: " + teProfile +
                "\nVE profile: " + veProfile +
                "\nVOC profile: " + vocProfile;
    }

    private static String describeDims(String label, OrtSession session) {
        StringBuilder out = new StringBuilder(label).append(" dims:");
        try {
            for (Map.Entry<String, NodeInfo> entry : session.getInputInfo().entrySet()) {
                if (!(entry.getValue().getInfo() instanceof TensorInfo)) continue;
                TensorInfo info = (TensorInfo) entry.getValue().getInfo();
                out.append(" ").append(entry.getKey()).append("=")
                        .append(java.util.Arrays.toString(info.getShape()))
                        .append("/")
                        .append(java.util.Arrays.toString(info.getDimensionNames()));
            }
            return out.toString();
        } catch (Throwable error) {
            return label + " dims: <failed " + error.getClass().getSimpleName() + ">";
        }
    }

    private String preprocess(String raw, String language) {
        String lang = language == null ? "na" : language.trim().toLowerCase();
        String t = Normalizer.normalize(raw == null ? "" : raw.trim(), Normalizer.Form.NFKD);
        t = t.replace('’', '\'').replace('“', '"').replace('”', '"').replace('—', '-');
        t = t.replaceAll("\\s+", " ");
        if (t.isEmpty()) throw new IllegalArgumentException("Enter text first");
        if (!t.matches(".*[.!?;:]$")) t += ".";
        if ("na".equals(lang)) return t;
        return "<" + lang + ">" + t + "</" + lang + ">";
    }

    private long[][] tokenizePadded(int[] cps, int paddedLength) {
        long[][] ids = new long[1][paddedLength];
        for (int i = 0; i < cps.length; i++) {
            int cp = cps[i];
            if (cp < 0 || cp >= unicodeIndexer.length) {
                throw new IllegalArgumentException("Unsupported character U+" + Integer.toHexString(cp));
            }
            ids[0][i] = unicodeIndexer[cp];
        }
        return ids;
    }

    private float[][][] randomLatent(int channels, int length) {
        Random random = new Random(0x5354);
        float[][][] out = new float[1][channels][length];
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < length; t++) {
                double u1 = Math.max(1e-12, random.nextDouble());
                double u2 = random.nextDouble();
                out[0][c][t] = (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
            }
        }
        return out;
    }

    private OnnxTensor longTensor(long[][] value) throws OrtException {
        int rows = value.length, cols = value[0].length;
        long[] flat = new long[rows * cols];
        int n = 0;
        for (long[] row : value) for (long v : row) flat[n++] = v;
        return OnnxTensor.createTensor(env, LongBuffer.wrap(flat), new long[]{rows, cols});
    }

    private OnnxTensor floatTensor(float[][][] value) throws OrtException {
        int a = value.length, b = value[0].length, c = value[0][0].length;
        float[] flat = new float[a * b * c];
        int n = 0;
        for (float[][] x : value) for (float[] y : x) for (float v : y) flat[n++] = v;
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{a, b, c});
    }

    private static OnnxTensor styleTensor(OrtEnvironment env, JSONObject obj) throws Exception {
        JSONArray dimsJson = obj.getJSONArray("dims");
        long[] dims = new long[dimsJson.length()];
        int total = 1;
        for (int i = 0; i < dims.length; i++) {
            dims[i] = dimsJson.getLong(i);
            total *= (int) dims[i];
        }
        float[] flat = new float[total];
        int[] cursor = {0};
        flattenJson(obj.get("data"), flat, cursor);
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), dims);
    }

    private static void flattenJson(Object value, float[] out, int[] cursor) throws Exception {
        if (value instanceof JSONArray) {
            JSONArray a = (JSONArray) value;
            for (int i = 0; i < a.length(); i++) flattenJson(a.get(i), out, cursor);
        } else if (value instanceof Number) {
            out[cursor[0]++] = ((Number) value).floatValue();
        }
    }

    private static float firstFloat(Object value) {
        if (value instanceof float[]) return ((float[]) value)[0];
        if (value instanceof float[][]) return ((float[][]) value)[0][0];
        if (value instanceof float[][][]) return ((float[][][]) value)[0][0][0];
        throw new IllegalStateException("Unexpected duration output: " + value.getClass());
    }

    private static float[][][] toFloat3(Object value) {
        if (value instanceof float[][][]) return (float[][][]) value;
        throw new IllegalStateException("Unexpected vector output: " + value.getClass());
    }

    private static float[] flattenAudio(Object value) {
        if (value instanceof float[]) return (float[]) value;
        if (value instanceof float[][]) {
            float[][] a = (float[][]) value;
            int n = 0;
            for (float[] x : a) n += x.length;
            float[] out = new float[n];
            int p = 0;
            for (float[] x : a) {
                System.arraycopy(x, 0, out, p, x.length);
                p += x.length;
            }
            return out;
        }
        if (value instanceof float[][][]) {
            float[][][] a = (float[][][]) value;
            int n = 0;
            for (float[][] x : a) for (float[] y : x) n += y.length;
            float[] out = new float[n];
            int p = 0;
            for (float[][] x : a) for (float[] y : x) {
                System.arraycopy(y, 0, out, p, y.length);
                p += y.length;
            }
            return out;
        }
        throw new IllegalStateException("Unexpected vocoder output: " + value.getClass());
    }

    private static String readText(File file) throws Exception {
        try (FileInputStream in = new FileInputStream(file)) {
            byte[] data = new byte[(int) file.length()];
            int pos = 0;
            while (pos < data.length) {
                int n = in.read(data, pos, data.length - pos);
                if (n < 0) break;
                pos += n;
            }
            return new String(data, 0, pos, StandardCharsets.UTF_8);
        }
    }

    private static long[] jsonLongArray(JSONArray array) throws Exception {
        long[] out = new long[array.length()];
        for (int i = 0; i < out.length; i++) out[i] = array.getLong(i);
        return out;
    }

    @Override
    public void close() throws Exception {
        styleTtl.close();
        styleDp.close();
        durationSession.close();
        textEncoderSession.close();
        vectorSession.close();
        vocoderSession.close();
    }
}
