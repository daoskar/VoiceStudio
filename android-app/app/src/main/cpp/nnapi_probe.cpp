#include <jni.h>
#include <android/NeuralNetworks.h>
#include <dlfcn.h>
#include <sstream>
#include <string>
#include <vector>

static std::string qnnProbe() {
    const std::vector<const char*> libs = {
        "libQnnHtp.so",
        "libQnnSystem.so",
        "libQnnCpu.so",
        "libQnnGpu.so",
        "libQnnHtpPrepare.so"
    };

    std::ostringstream out;
    out << "QNN libraries:";
    for (const char* lib : libs) {
        dlerror();
        void* handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            out << "\n" << lib << ": accessible";
            dlclose(handle);
        } else {
            const char* err = dlerror();
            out << "\n" << lib << ": blocked/missing";
            if (err) {
                std::string msg(err);
                if (msg.size() > 120) msg.resize(120);
                out << " (" << msg << ")";
            }
        }
    }
    return out.str();
}

extern "C"
JNIEXPORT jstring JNICALL
Java_dev_daoskar_voicestudio_NnapiNativeProbe_listDevices(JNIEnv* env, jclass) {
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

    out << "\n" << qnnProbe();

    std::string text = out.str();
    return env->NewStringUTF(text.c_str());
}
