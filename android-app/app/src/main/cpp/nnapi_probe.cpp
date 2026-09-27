#include <jni.h>
#include <android/NeuralNetworks.h>
#include <sstream>
#include <string>

extern "C"
JNIEXPORT jstring JNICALL
Java_dev_daoskar_voicestudio_NnapiNativeProbe_listDevices(JNIEnv* env, jclass) {
#if __ANDROID_API__ >= 29
    uint32_t count = 0;
    int result = ANeuralNetworks_getDeviceCount(&count);
    if (result != ANEURALNETWORKS_NO_ERROR) {
        std::string msg = "NNAPI device enumeration failed: " + std::to_string(result);
        return env->NewStringUTF(msg.c_str());
    }

    std::ostringstream out;
    out << "NNAPI devices: " << count;
    for (uint32_t i = 0; i < count; ++i) {
        ANeuralNetworksDevice* device = nullptr;
        if (ANeuralNetworks_getDevice(i, &device) != ANEURALNETWORKS_NO_ERROR || !device) {
            out << "\n#" << i << ": <unavailable>";
            continue;
        }

        const char* name = nullptr;
        const char* version = nullptr;
        int32_t type = -1;

        ANeuralNetworksDevice_getName(device, &name);
        ANeuralNetworksDevice_getVersion(device, &version);
        ANeuralNetworksDevice_getType(device, &type);

        const char* typeName = "UNKNOWN";
        switch (type) {
            case ANEURALNETWORKS_DEVICE_CPU: typeName = "CPU"; break;
            case ANEURALNETWORKS_DEVICE_GPU: typeName = "GPU"; break;
            case ANEURALNETWORKS_DEVICE_ACCELERATOR: typeName = "ACCELERATOR"; break;
            case ANEURALNETWORKS_DEVICE_OTHER: typeName = "OTHER"; break;
        }

        out << "\n#" << i << ": "
            << (name ? name : "<unnamed>")
            << " [" << typeName << "]"
            << " driver=" << (version ? version : "?");
    }

    std::string text = out.str();
    return env->NewStringUTF(text.c_str());
#else
    return env->NewStringUTF("NNAPI device enumeration requires Android 10+");
#endif
}
