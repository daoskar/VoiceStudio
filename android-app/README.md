# VoiceStudio Android

This directory contains the Android/ARM64 port of VoiceStudio.

## Current milestone

- Native Android application shell
- ARM64-only packaging
- microphone permission
- ONNX Runtime Android bundled
- NNAPI as the first mobile hardware acceleration path
- CI release APK

## Qualcomm NPU plan

Direct HTP/NPU execution will use ONNX Runtime's QNN Execution Provider.

The Android QNN provider requires a custom ONNX Runtime build with the Qualcomm
AI Runtime/QNN SDK using `--use_qnn static_lib`. The resulting runtime and
permitted QNN libraries will live under `app/src/main/jniLibs/arm64-v8a/`.

Do not commit Qualcomm SDK redistributables unless their license permits it.

## Local build

Install Android SDK 35 + JDK 17 and run:

```bash
cd android-app
gradle :app:assembleDebug
```
