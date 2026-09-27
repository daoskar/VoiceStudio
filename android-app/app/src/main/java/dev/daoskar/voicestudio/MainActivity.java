package dev.daoskar.voicestudio;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public final class MainActivity extends AppCompatActivity {
    private static final int REQ_AUDIO = 41;
    private TextView status;

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

        TextView note = new TextView(this);
        note.setText(
                "\nRuntime status\n" +
                "• ONNX Runtime Android is bundled.\n" +
                "• NNAPI is the first hardware-acceleration path.\n" +
                "• Qualcomm QNN/HTP backend is staged for a custom ORT build.\n" +
                "• VoiceStudio model integration is the next milestone.\n"
        );
        note.setTextSize(14f);
        root.addView(note, matchWrap());

        setContentView(scroll);
    }

    private String buildStatus() {
        String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : Build.HARDWARE;
        return "Device: " + Build.MANUFACTURER + " " + Build.MODEL +
                "\nSoC: " + soc +
                "\nAndroid API: " + Build.VERSION.SDK_INT +
                "\nABI: " + Build.SUPPORTED_ABIS[0] +
                "\nNNAPI provider: " + InferenceRuntime.probeNnapi() +
                "\nAcceleration target: NNAPI → QNN/HTP → CPU fallback" +
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
