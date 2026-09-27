package dev.daoskar.voicestudio;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

final class InferenceRuntime {
    private InferenceRuntime() {}

    static String probeNnapi() {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.addNnapi();
            return "available";
        } catch (OrtException | UnsatisfiedLinkError | RuntimeException error) {
            String name = error.getClass().getSimpleName();
            String message = error.getMessage();
            if (message == null || message.isBlank()) {
                return "unavailable (" + name + ")";
            }
            return "unavailable (" + name + ": " + message + ")";
        }
    }
}
