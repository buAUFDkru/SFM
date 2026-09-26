#include "OnnxHelper.h"

#ifdef _WIN32
#include <windows.h>
#endif


void OnnxHelper::GetModelInfo(Ort::Session &session, std::vector<TensorInfo> &inputInfos,
                              std::vector<TensorInfo> &outputInfos) {
    inputInfos.clear();
    outputInfos.clear();
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    // 插件集成：静默上游控制台输出（每次加载模型都会打印，噪音过大）
    // std::cout << "Number of inputs = " << numInputNodes << std::endl;
    for (size_t i = 0; i < numInputNodes; ++i) {
        std::string name(session.GetInputNameAllocated(i, allocator).get());
        auto inputTypeInfo = session.GetInputTypeInfo(i);
        std::vector<int64_t> inShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        inputInfos.emplace_back(TensorInfo{name, inShape, inputTypeInfo.GetONNXType()});
    }

    size_t numOutputNodes = session.GetOutputCount();
    // std::cout << "Number of outputs = " << numOutputNodes << std::endl;
    for (size_t i = 0; i < numOutputNodes; ++i) {
        std::string name(session.GetOutputNameAllocated(i, allocator).get());
        auto outputTypeInfo = session.GetOutputTypeInfo(i);
        std::vector<int64_t> outShape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        outputInfos.emplace_back(TensorInfo{name, outShape, outputTypeInfo.GetONNXType()});
    }
}


void OnnxHelper::PrintModelInfo(const std::vector<TensorInfo> &inputInfos, const std::vector<TensorInfo> &outputInfos) {
    // Iterate over shape.size() instead of a hard-coded 4: outputs may be 3-D (e.g. (1,H,W)).
    auto printOne = [](const TensorInfo &info) {
        std::cout << "Name: " << info.name << ", Shape: [";
        for (size_t i = 0; i < info.shape.size(); ++i) {
            std::cout << info.shape[i] << (i + 1 == info.shape.size() ? "" : ", ");
        }
        std::cout << "], Type: " << info.type << std::endl;
    };

    std::cout << "Input Infos:" << std::endl;
    for (const auto &info : inputInfos) printOne(info);
    std::cout << "Output Infos:" << std::endl;
    for (const auto &info : outputInfos) printOne(info);
}


void OnnxHelper::Str2Ort(const std::string& modelFilePath, std::vector<ORTCHAR_T>& modelFileOrt) {
#ifdef _WIN32
    // ORTCHAR_T == wchar_t on Windows: do a real UTF-8 -> UTF-16 conversion.
    int needed = MultiByteToWideChar(CP_UTF8, 0, modelFilePath.c_str(), (int)modelFilePath.size(), nullptr, 0);
    modelFileOrt.assign((size_t)needed + 1, ORTCHAR_T('\0'));
    MultiByteToWideChar(CP_UTF8, 0, modelFilePath.c_str(), (int)modelFilePath.size(),
                        modelFileOrt.data(), needed);
    modelFileOrt[needed] = ORTCHAR_T('\0');
#else
    modelFileOrt.reserve(modelFilePath.size() + 1);
    modelFileOrt.assign(modelFilePath.begin(), modelFilePath.end());
    modelFileOrt.push_back(ORTCHAR_T('\0'));
#endif
}
