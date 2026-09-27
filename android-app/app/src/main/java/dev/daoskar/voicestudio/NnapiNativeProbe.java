package dev.daoskar.voicestudio;

final class NnapiNativeProbe {
    private static boolean loaded;

    static {
        try {
            System.loadLibrary("voicestudio_nnapi_probe");
            loaded = true;
        } catch (UnsatisfiedLinkError error) {
            loaded = false;
        }
    }

    private NnapiNativeProbe() {}

    private static native String listDevices();

    static String probe() {
        if (!loaded) {
            return "NNAPI native probe unavailable";
        }
        try {
            return listDevices();
        } catch (RuntimeException error) {
            return "NNAPI native probe failed: " + error.getClass().getSimpleName();
        }
    }
}
