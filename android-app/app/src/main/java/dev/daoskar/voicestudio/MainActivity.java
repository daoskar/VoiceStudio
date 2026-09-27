package dev.daoskar.voicestudio;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public final class MainActivity extends AppCompatActivity {
    private static final int REQ_AUDIO = 41;
    private TextView status;
    private final Object engineLock = new Object();
    private SupertonicEngine cachedEngine;
    private String cachedVoice;
    private String cachedMode;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(28), dp(20), dp(28));
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        scroll.addView(root, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = new TextView(this);
        title.setText("VoiceStudio Android");
        title.setTextSize(28f);
        root.addView(title, matchWrap());

        TextView subtitle = new TextView(this);
        subtitle.setText("Local-first mobile runtime • ARM64 • ONNX Runtime");
        subtitle.setTextSize(15f);
        subtitle.setPadding(0, dp(6), 0, dp(24));
        root.addView(subtitle, matchWrap());

        status = new TextView(this);
        status.setTextSize(15f);
        status.setText(buildStatus());
        root.addView(status, matchWrap());

        Button mic = new Button(this);
        mic.setText("Enable microphone");
        mic.setOnClickListener(v -> ensureMicPermission());
        LinearLayout.LayoutParams buttonParams = matchWrap();
        buttonParams.topMargin = dp(20);
        root.addView(mic, buttonParams);

        Button model = new Button(this);
        model.setText(SupertonicModelManager.isInstalled(this) ? "SUPERTONIC-3 MODEL READY" : "DOWNLOAD SUPERTONIC-3 MODEL");
        model.setOnClickListener(v -> {
            if (SupertonicModelManager.isInstalled(this)) {
                status.setText(buildStatus() + "\nSupertonic-3: ready");
                return;
            }
            model.setEnabled(false);
            SupertonicModelManager.downloadAsync(this, new SupertonicModelManager.Listener() {
                @Override
                public void onProgress(String message) {
                    runOnUiThread(() -> status.setText(buildStatus() + "\n" + message));
                }

                @Override
                public void onDone(java.io.File modelRoot) {
                    runOnUiThread(() -> {
                        model.setEnabled(true);
                        model.setText("SUPERTONIC-3 MODEL READY");
                        status.setText(buildStatus() + "\nSupertonic-3 ready: " + modelRoot.getAbsolutePath());
                    });
                }

                @Override
                public void onError(Throwable error) {
                    runOnUiThread(() -> {
                        model.setEnabled(true);
                        model.setText("RETRY SUPERTONIC-3 DOWNLOAD");
                        status.setText(buildStatus() + "\nModel download failed: " + error);
                    });
                }
            });
        });
        LinearLayout.LayoutParams modelParams = matchWrap();
        modelParams.topMargin = dp(12);
        root.addView(model, modelParams);

        Button qnnModel = new Button(this);
        qnnModel.setText(
                SupertonicQnnModelManager.isInstalled(this)
                        ? "QNN/INT8 MODEL READY"
                        : "DOWNLOAD QNN/INT8 MODEL"
        );
        qnnModel.setOnClickListener(v -> {
            if (SupertonicQnnModelManager.isInstalled(this)) {
                status.setText(buildStatus() + "\nQNN/INT8 Supertonic-3: ready");
                return;
            }

            qnnModel.setEnabled(false);
            SupertonicQnnModelManager.downloadAsync(
                    this,
                    new SupertonicQnnModelManager.Listener() {
                        @Override
                        public void onProgress(String message) {
                            runOnUiThread(() ->
                                    status.setText(buildStatus() + "\n" + message)
                            );
                        }

                        @Override
                        public void onDone(java.io.File modelRoot) {
                            runOnUiThread(() -> {
                                clearCachedEngine();
                                qnnModel.setEnabled(true);
                                qnnModel.setText("QNN/INT8 MODEL READY");
                                status.setText(
                                        buildStatus() +
                                        "\nQNN/INT8 models ready: " +
                                        modelRoot.getAbsolutePath()
                                );
                            });
                        }

                        @Override
                        public void onError(Throwable error) {
                            runOnUiThread(() -> {
                                qnnModel.setEnabled(true);
                                qnnModel.setText("RETRY QNN/INT8 DOWNLOAD");
                                status.setText(
                                        buildStatus() +
                                        "\nQNN model download failed: " +
                                        error.getClass().getSimpleName() +
                                        ": " + error.getMessage()
                                );
                            });
                        }
                    }
            );
        });
        LinearLayout.LayoutParams qnnParams = matchWrap();
        qnnParams.topMargin = dp(8);
        root.addView(qnnModel, qnnParams);

        TextView langLabel = new TextView(this);
        langLabel.setText("Language");
        root.addView(langLabel, matchWrap());

        String[] languageCodes = {
                "pl","en","de","fr","es","it","pt","cs","sk","uk","ru",
                "ja","ko","ar","bg","da","el","et","fi","hi","hr","hu",
                "id","lt","lv","nl","ro","sl","sv","tr","vi","na"
        };
        Spinner languageSpinner = new Spinner(this);
        languageSpinner.setAdapter(new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_dropdown_item, languageCodes));
        root.addView(languageSpinner, matchWrap());

        TextView voiceLabel = new TextView(this);
        voiceLabel.setText("Voice");
        root.addView(voiceLabel, matchWrap());

        String[] voices = {"M1","M3","M4","M5","F3","F4","F5"};
        Spinner voiceSpinner = new Spinner(this);
        voiceSpinner.setAdapter(new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_dropdown_item, voices));
        root.addView(voiceSpinner, matchWrap());

        TextView stepsLabel = new TextView(this);
        stepsLabel.setText("Quality steps");
        root.addView(stepsLabel, matchWrap());

        Integer[] stepOptions = {4,6,8,10,12};
        Spinner stepsSpinner = new Spinner(this);
        stepsSpinner.setAdapter(new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_dropdown_item, stepOptions));
        stepsSpinner.setSelection(1);
        root.addView(stepsSpinner, matchWrap());

        EditText ttsText = new EditText(this);
        ttsText.setHint("Text to speak (English test)");
        ttsText.setText("Hello from VoiceStudio on Android.");
        ttsText.setMinLines(2);
        root.addView(ttsText, matchWrap());

        Button generate = new Button(this);
        generate.setText("GENERATE & PLAY");
        generate.setOnClickListener(v -> {
            if (!SupertonicModelManager.isInstalled(this)
                    && !SupertonicQnnModelManager.isInstalled(this)) {
                status.setText(
                        buildStatus() +
                        "\nInstall either the FP32 or QNN/INT8 Supertonic-3 model first"
                );
                return;
            }
            final String text = ttsText.getText().toString();
            final String language = (String) languageSpinner.getSelectedItem();
            final String voice = (String) voiceSpinner.getSelectedItem();
            final int steps = (Integer) stepsSpinner.getSelectedItem();
            generate.setEnabled(false);
            model.setEnabled(false);
            qnnModel.setEnabled(false);
            new Thread(() -> {
                try {
                    long loadStarted = System.currentTimeMillis();
                    SupertonicEngine engine = getOrLoadEngine(
                            voice,
                            message -> runOnUiThread(() -> status.setText(buildStatus() + "\n" + message))
                    );
                    long loadElapsed = System.currentTimeMillis() - loadStarted;

                    long inferenceStarted = System.currentTimeMillis();
                    float[] wav = engine.synthesize(
                            text,
                            language,
                            steps,
                            1.0f,
                            message -> runOnUiThread(() -> status.setText(buildStatus() + "\n" + message))
                    );
                    long inferenceElapsed = System.currentTimeMillis() - inferenceStarted;

                    double audioSeconds = wav.length / (double) engine.getSampleRate();
                    double generationSeconds = inferenceElapsed / 1000.0;
                    double rtf = generationSeconds / Math.max(0.001, audioSeconds);
                    runOnUiThread(() -> {
                        status.setText(buildStatus()
                                + "\nGenerated " + wav.length + " samples"
                                + "\nAudio: " + String.format(java.util.Locale.US, "%.2f s", audioSeconds)
                                + "\nModel load/cache: " + loadElapsed + " ms"
                                + "\nInference: " + String.format(java.util.Locale.US, "%.3f s", generationSeconds)
                                + "\nRTF: " + String.format(java.util.Locale.US, "%.3f", rtf)
                                + "\nBackends: " + engine.getBackendSummary()
                                + "\nLanguage: " + language + "  Voice: " + voice + "  Steps: " + steps);
                        engine.play(wav);
                    });
                } catch (Throwable error) {
                    runOnUiThread(() -> status.setText(
                            buildStatus() + "\nTTS failed: " + error.getClass().getSimpleName() + ": " + error.getMessage()
                    ));
                } finally {
                    runOnUiThread(() -> {
                        generate.setEnabled(true);
                        model.setEnabled(true);
                        qnnModel.setEnabled(true);
                    });
                }
            }, "supertonic-generate").start();
        });
        LinearLayout.LayoutParams generateParams = matchWrap();
        generateParams.topMargin = dp(12);
        root.addView(generate, generateParams);

        TextView note = new TextView(this);
        note.setText(
                "\nRuntime status\n" +
                "• ONNX Runtime Android is bundled.\n" +
                "• NNAPI is the first hardware-acceleration path.\n" +
                "• Qualcomm QNN/HTP runtime is bundled.\n" +
                "• FP32 model = dynamic CPU fallback.\n" +
                "• QDQ/INT8 model = static QNN/HTP path.\n" +
                "• ONNX sessions stay warm between generations.\n" +
                "• ORT profiling reports real QNN vs CPU execution.\n"
        );
        note.setTextSize(14f);
        root.addView(note, matchWrap());

        setContentView(scroll);
    }

    private SupertonicEngine getOrLoadEngine(
            String voice,
            SupertonicEngine.Listener listener
    ) throws Exception {
        synchronized (engineLock) {
            boolean useQnn = SupertonicQnnModelManager.isInstalled(this);
            String mode = useQnn ? "QDQ-QNN" : "FP32-CPU";

            if (cachedEngine != null
                    && voice.equals(cachedVoice)
                    && mode.equals(cachedMode)) {
                listener.onStatus("Supertonic-3 warm cache: " + voice + " / " + mode);
                return cachedEngine;
            }

            clearCachedEngineLocked();

            java.io.File root = useQnn
                    ? SupertonicQnnModelManager.root(this)
                    : SupertonicModelManager.root(this);

            try {
                cachedEngine = SupertonicEngine.load(
                        root,
                        voice,
                        useQnn,
                        listener
                );
                cachedVoice = voice;
                cachedMode = mode;
                return cachedEngine;
            } catch (Throwable qnnError) {
                if (useQnn && SupertonicModelManager.isInstalled(this)) {
                    listener.onStatus(
                            "QNN model load failed (" +
                            qnnError.getClass().getSimpleName() +
                            ": " + qnnError.getMessage() +
                            "); switching to FP32 CPU fallback"
                    );
                    clearCachedEngineLocked();
                    cachedEngine = SupertonicEngine.load(
                            SupertonicModelManager.root(this),
                            voice,
                            false,
                            listener
                    );
                    cachedVoice = voice;
                    cachedMode = "FP32-CPU";
                    return cachedEngine;
                }
                if (qnnError instanceof Exception) throw (Exception) qnnError;
                throw new RuntimeException(qnnError);
            }
        }
    }

    private void clearCachedEngine() {
        synchronized (engineLock) {
            clearCachedEngineLocked();
        }
    }

    private void clearCachedEngineLocked() {
        if (cachedEngine != null) {
            try {
                cachedEngine.close();
            } catch (Exception ignored) {
            }
            cachedEngine = null;
            cachedVoice = null;
            cachedMode = null;
        }
    }

    @Override
    protected void onDestroy() {
        clearCachedEngine();
        super.onDestroy();
    }

    private String buildStatus() {
        String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : Build.HARDWARE;
        return "Device: " + Build.MANUFACTURER + " " + Build.MODEL +
                "\nSoC: " + soc +
                "\nAndroid API: " + Build.VERSION.SDK_INT +
                "\nABI: " + Build.SUPPORTED_ABIS[0] +
                "\nNNAPI provider: " + InferenceRuntime.probeNnapi() +
                "\n" + NnapiNativeProbe.probe() +
                "\nAcceleration target: QNN/HTP → CPU fallback" +
                "\nFP32 model: " + (SupertonicModelManager.isInstalled(this) ? "ready" : "not installed") +
                "\nQNN/INT8 model: " + (SupertonicQnnModelManager.isInstalled(this) ? "ready" : "not installed") +
                "\nMicrophone: " + (hasMicPermission() ? "ready" : "permission required");
    }

    private void ensureMicPermission() {
        if (hasMicPermission()) {
            status.setText(buildStatus());
            return;
        }
        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.RECORD_AUDIO},
                REQ_AUDIO
        );
    }

    private boolean hasMicPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_AUDIO) {
            status.setText(buildStatus());
        }
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
