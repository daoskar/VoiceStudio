package dev.daoskar.voicestudio;

import java.util.List;

final class SupertonicModelManifest {
    static final String REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b";
    static final String BASE = "https://huggingface.co/Supertone/supertonic-3/resolve/" + REVISION + "/";

    static final List<String> FILES = List.of(
            "onnx/duration_predictor.onnx",
            "onnx/text_encoder.onnx",
            "onnx/vector_estimator.onnx",
            "onnx/vocoder.onnx",
            "onnx/tts.json",
            "onnx/unicode_indexer.json",
            "voice_styles/M1.json",
            "voice_styles/M3.json",
            "voice_styles/M4.json",
            "voice_styles/M5.json",
            "voice_styles/F3.json",
            "voice_styles/F4.json",
            "voice_styles/F5.json",
            "LICENSE"
    );

    private SupertonicModelManifest() {}
}
