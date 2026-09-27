package dev.daoskar.voicestudio;

import ai.onnxruntime.OrtEpDevice;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

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
            SupertonicEngine.Listener listener
    ) throws Exception {
        ensureRegistered(env);

        if (registered) {
            try {
                List<OrtEpDevice> devices = qnnDevices(env);
                if (!devices.isEmpty()) {
                    try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                        options.setIntraOpNumThreads(1);
                        Map<String, String> providerOptions = new HashMap<>();
                        providerOptions.put("backend_type", "htp");
                        options.addExecutionProvider(devices, providerOptions);
                        options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
                        listener.onStatus(modelName + ": compiling for QNN/HTP...");
                        OrtSession session = env.createSession(modelPath, options);
                        listener.onStatus(modelName + ": QNN/HTP ready");
                        return new SessionResult(session, "QNN/HTP", "strict HTP session created");
                    }
                } else {
                    String reason = "registered plugin exposed no QNN EP device; EPs=" + epNames(env);
                    listener.onStatus(modelName + ": " + reason + "; CPU fallback");
                    return cpuSession(env, modelPath, reason);
                }
            } catch (Throwable qnnError) {
                String reason = shortMessage(qnnError);
                listener.onStatus(modelName + ": QNN/HTP rejected (" +
                        reason + "); CPU fallback");
                return cpuSession(env, modelPath, reason);
            }
        } else {
            String reason = "plugin registration failed" +
                    (registrationError == null ? "" : ": " + registrationError);
            listener.onStatus(modelName + ": " + reason + "; CPU fallback");
            return cpuSession(env, modelPath, reason);
        }
    }

    private static SessionResult cpuSession(
            OrtEnvironment env,
            String modelPath,
            String reason
    ) throws Exception {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(
                    Math.max(2, Runtime.getRuntime().availableProcessors() / 2)
            );
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

    private static String shortMessage(Throwable error) {
        String msg = error.getMessage();
        if (msg == null || msg.isBlank()) return error.getClass().getSimpleName();
        msg = msg.replace('\n', ' ').replace('\r', ' ');
        return msg.length() > 180 ? msg.substring(0, 180) : msg;
    }
}
