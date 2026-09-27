package dev.daoskar.voicestudio;

import ai.onnxruntime.OrtEpDevice;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtLoggingLevel;

import android.os.Process;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class QnnRuntime {
    static final class SessionResult {
        final OrtSession session;
        final String backend;
        final String detail;

        SessionResult(OrtSession session, String backend, String detail) {
            this.session = session;
            this.backend = backend;
            this.detail = detail;
        }
    }

    private static final String REGISTRATION_NAME = "QNNExecutionProvider";
    private static final String LIBRARY_NAME = "libonnxruntime_providers_qnn.so";
    private static boolean registrationAttempted;
    private static boolean registered;
    private static String registrationError;

    private QnnRuntime() {}

    static synchronized String registrationStatus(OrtEnvironment env) {
        ensureRegistered(env);
        if (registered) return "QNN plugin: registered";
        return "QNN plugin: unavailable" +
                (registrationError == null ? "" : " (" + registrationError + ")");
    }

    static SessionResult createSession(
            OrtEnvironment env,
            String modelPath,
            String modelName,
            Map<String, Long> symbolicDims,
            SupertonicEngine.Listener listener
    ) throws Exception {
        ensureRegistered(env);

        if (registered) {
            try {
                List<OrtEpDevice> devices = qnnDevices(env);
                if (!devices.isEmpty()) {
                    try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                        options.setIntraOpNumThreads(1);
                        options.setLoggerId("VoiceStudio-QNN-" + modelName);
                        options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
                        options.setSessionLogVerbosityLevel(4);
                        File traceDir = new File(
                                new File(modelPath).getParentFile(),
                                "qnn-trace-" + modelName
                        );
                        resetTraceDir(traceDir);

                        Map<String, String> providerOptions = new HashMap<>();
                        providerOptions.put("backend_type", "htp");
                        providerOptions.put("enable_htp_fp16_precision", "1");

                        options.addExecutionProvider(devices, providerOptions);
                        options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
                        for (Map.Entry<String, Long> dim : symbolicDims.entrySet()) {
                            options.setSymbolicDimensionValue(dim.getKey(), dim.getValue());
                        }
                        options.enableProfiling(
                                new File(
                                        new File(modelPath).getParentFile(),
                                        "ort-profile-" + modelName + "-"
                                ).getAbsolutePath()
                        );

                        listener.onStatus(modelName + ": compiling hybrid QNN/HTP + CPU...");
                        OrtSession session = env.createSession(modelPath, options);

                        String detail = "strict minimal HTP session created";
                        listener.onStatus(modelName + ": " + detail);
                        return new SessionResult(session, "QNN/HTP", detail);
                    }
                } else {
                    String reason = "registered plugin exposed no QNN EP device; EPs=" + epNames(env);
                    listener.onStatus(modelName + ": " + reason + "; CPU fallback");
                    return cpuSession(env, modelPath, symbolicDims, reason);
                }
            } catch (Throwable qnnError) {
                String reason = shortMessage(qnnError);
                String qnnLog = captureOwnOrtLog();
                String detail = "strict HTP reject: " + reason +
                        (qnnLog.isEmpty() ? "" : "\nQNN verbose: " + qnnLog);
                listener.onStatus(modelName + ": " + detail + "; CPU fallback");
                return cpuSession(env, modelPath, symbolicDims, detail);
            }
        } else {
            String reason = "plugin registration failed" +
                    (registrationError == null ? "" : ": " + registrationError);
            listener.onStatus(modelName + ": " + reason + "; CPU fallback");
            return cpuSession(env, modelPath, symbolicDims, reason);
        }
    }

    private static SessionResult cpuSession(
            OrtEnvironment env,
            String modelPath,
            Map<String, Long> symbolicDims,
            String reason
    ) throws Exception {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(
                    Math.max(2, Runtime.getRuntime().availableProcessors() / 2)
            );
            for (Map.Entry<String, Long> dim : symbolicDims.entrySet()) {
                options.setSymbolicDimensionValue(dim.getKey(), dim.getValue());
            }
            return new SessionResult(env.createSession(modelPath, options), "CPU", reason);
        }
    }

    private static String epNames(OrtEnvironment env) {
        try {
            StringBuilder out = new StringBuilder();
            for (OrtEpDevice device : env.getEpDevices()) {
                if (out.length() > 0) out.append(",");
                out.append(device.getEpName());
            }
            return out.length() == 0 ? "<none>" : out.toString();
        } catch (Throwable error) {
            return "<enumeration failed: " + shortMessage(error) + ">";
        }
    }

    private static void resetTraceDir(File dir) {
        if (dir.isDirectory()) {
            File[] files = dir.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) file.delete();
                }
            }
        } else {
            dir.mkdirs();
        }
    }

    private static String readTraceSummary(File dir) {
        try {
            File[] files = dir.listFiles((d, name) -> name.endsWith(".json") && name.contains("_op_trace"));
            if (files == null || files.length == 0) {
                File fallback = new File(dir, "qnn_op_trace.json");
                if (fallback.isFile()) files = new File[]{fallback};
            }
            if (files == null || files.length == 0) {
                return "QNN trace unavailable";
            }

            String json = readText(files[0]);
            JSONObject root = new JSONObject(json);
            JSONObject summary = root.optJSONObject("summary");
            if (summary == null) return "QNN trace has no summary";

            int supported = summary.optInt("supported_nodes", 0);
            int unsupported = summary.optInt("unsupported_nodes", 0);
            int total = summary.optInt("total_onnx_nodes", supported + unsupported);
            int qnnOps = summary.optInt("total_qnn_ops", 0);

            StringBuilder out = new StringBuilder();
            out.append("QNN nodes: ").append(supported).append("/").append(total)
                    .append(", unsupported: ").append(unsupported)
                    .append(", QNN ops: ").append(qnnOps);

            JSONArray rejected = root.optJSONArray("unsupported_nodes");
            if (rejected != null && rejected.length() > 0) {
                out.append("; first unsupported: ");
                int limit = Math.min(2, rejected.length());
                for (int i = 0; i < limit; i++) {
                    if (i > 0) out.append(" | ");
                    JSONObject node = rejected.optJSONObject(i);
                    if (node == null) continue;
                    out.append(node.optString("op_type", "?"));
                    String reason = node.optString("reason", "");
                    if (!reason.isEmpty()) {
                        if (reason.length() > 90) reason = reason.substring(0, 90);
                        out.append(" (").append(reason).append(")");
                    }
                }
            }
            return out.toString();
        } catch (Throwable error) {
            return "QNN trace parse failed: " + shortMessage(error);
        }
    }

    static String readOrtProfile(OrtSession session) {
        try {
            String path = session.endProfiling();
            if (path == null || path.isBlank()) return "ORT profile path unavailable";
            File file = new File(path);
            if (!file.isFile()) return "ORT profile missing: " + path;

            JSONArray events = new JSONArray(readText(file));
            int qnnEvents = 0;
            int cpuEvents = 0;
            long qnnDur = 0L;
            long cpuDur = 0L;

            for (int i = 0; i < events.length(); i++) {
                JSONObject event = events.optJSONObject(i);
                if (event == null) continue;
                JSONObject args = event.optJSONObject("args");
                if (args == null) continue;
                String provider = args.optString("provider", "");
                long dur = event.optLong("dur", 0L);

                if ("QNNExecutionProvider".equals(provider)) {
                    qnnEvents++;
                    qnnDur += dur;
                } else if ("CPUExecutionProvider".equals(provider)) {
                    cpuEvents++;
                    cpuDur += dur;
                }
            }

            long total = qnnDur + cpuDur;
            double qnnPct = total > 0 ? (100.0 * qnnDur / total) : 0.0;
            return "ORT profile: QNN events=" + qnnEvents +
                    ", CPU events=" + cpuEvents +
                    ", QNN dur=" + qnnDur + "us" +
                    ", CPU dur=" + cpuDur + "us" +
                    ", QNN share=" + String.format(java.util.Locale.US, "%.1f%%", qnnPct);
        } catch (Throwable error) {
            return "ORT profile failed: " + shortMessage(error);
        }
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

    private static synchronized void ensureRegistered(OrtEnvironment env) {
        if (registrationAttempted) return;
        registrationAttempted = true;
        try {
            env.registerExecutionProviderLibrary(REGISTRATION_NAME, LIBRARY_NAME);
            registered = true;
        } catch (Throwable error) {
            registered = false;
            registrationError = shortMessage(error);
        }
    }

    private static List<OrtEpDevice> qnnDevices(OrtEnvironment env) throws OrtException {
        List<OrtEpDevice> out = new ArrayList<>();
        for (OrtEpDevice device : env.getEpDevices()) {
            if (REGISTRATION_NAME.equals(device.getEpName())) {
                out.add(device);
            }
        }
        return out;
    }

    private static String captureOwnOrtLog() {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "logcat",
                    "--pid=" + Process.myPid(),
                    "-d",
                    "-v", "brief"
            );
            pb.redirectErrorStream(true);
            java.lang.Process p = pb.start();
            java.io.BufferedReader reader = new java.io.BufferedReader(
                    new java.io.InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8)
            );
            java.util.ArrayList<String> matches = new java.util.ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                String lower = line.toLowerCase(java.util.Locale.US);
                boolean useful =
                        lower.contains("getcapability") ||
                        lower.contains("unsupported") ||
                        lower.contains("not supported") ||
                        lower.contains("assigned to") ||
                        lower.contains("partition") ||
                        lower.contains("cpuexecutionprovider") ||
                        lower.contains("verifyeachnodeisassignedtoanep") ||
                        lower.contains("qnnexecutionprovider") ||
                        lower.contains("qnn ep");
                if (!useful) continue;

                String trimmed = line.trim();
                if (trimmed.length() > 260) trimmed = trimmed.substring(0, 260);
                matches.add(trimmed);
                if (matches.size() > 40) matches.remove(0);
            }
            reader.close();
            p.destroy();

            int from = Math.max(0, matches.size() - 16);
            StringBuilder out = new StringBuilder();
            for (int i = from; i < matches.size(); i++) {
                if (out.length() > 0) out.append(" | ");
                out.append(matches.get(i));
            }
            return out.toString();
        } catch (Throwable ignored) {
            return "";
        }
    }

    private static String shortMessage(Throwable error) {
        String msg = error.getMessage();
        if (msg == null || msg.isBlank()) return error.getClass().getSimpleName();
        msg = msg.replace('\n', ' ').replace('\r', ' ');
        return msg.length() > 180 ? msg.substring(0, 180) : msg;
    }
}
