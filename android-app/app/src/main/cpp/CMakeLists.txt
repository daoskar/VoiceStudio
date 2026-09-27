cmake_minimum_required(VERSION 3.22.1)
project(voicestudio_android)

add_library(
    voicestudio_nnapi_probe
    SHARED
    nnapi_probe.cpp
)

find_library(log-lib log)
find_library(neuralnetworks-lib neuralnetworks)
find_library(dl-lib dl)

target_link_libraries(
    voicestudio_nnapi_probe
    ${log-lib}
    ${neuralnetworks-lib}
    ${dl-lib}
)
