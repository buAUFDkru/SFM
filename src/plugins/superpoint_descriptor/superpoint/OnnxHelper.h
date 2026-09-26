#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    int type;
};



class OnnxHelper {
public:

    static void GetModelInfo(Ort::Session& session, std::vector<TensorInfo> &inputInfos, std::vector<TensorInfo> &outputInfos);

    static void PrintModelInfo(const std::vector<TensorInfo> &inputInfos, const std::vector<TensorInfo> &outputInfos);

    // Convert a UTF-8 path to the ORTCHAR_T buffer expected by Ort::Session.
    // On Windows ORTCHAR_T is wchar_t, so a plain char-by-char copy (as done in xfeatc)
    // only works for pure ASCII paths and breaks on non-ASCII ones.
    static void Str2Ort(const std::string& modelFilePath, std::vector<ORTCHAR_T>& modelFileOrt);

    template <typename T>
    static Ort::Value CreateTensor(const std::vector<std::int64_t>& shape, T* data, size_t dataLength, bool isCpu) {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        return Ort::Value::CreateTensor<T>(memoryInfo, data, dataLength, shape.data(), shape.size());
    }
};
