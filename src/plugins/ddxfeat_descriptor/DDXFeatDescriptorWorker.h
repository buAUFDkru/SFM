#ifndef DDXFEATDESCRIPTORWORKER_H
#define DDXFEATDESCRIPTORWORKER_H

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include "plugin/IDescriptorPlugin.h"

/**
 * @brief DD-XFeat 描述符 Worker（基于 ONNX Runtime 推理）。
 *
 * DD-XFeat 与 XFeat 仅卷积神经网络主干不同，后处理逻辑完全一致。
 * 构造时从 params 读取参数并持有；extract() 根据输入图像高度选择最匹配的
 * 外部 ONNX 模型完成推理与后处理。逐线程 / 任务独占，禁止跨线程共享使用。
 *
 * 模型来源：扫描 QCoreApplication::applicationDirPath()/models 目录下的
 * ddxfeat{H}x{W}.onnx 文件，按文件名解析输入高/宽，建立 高度→路径 映射。
 */
class DDXFeatDescriptorWorker : public IDescriptorWorker
{
public:
    explicit DDXFeatDescriptorWorker(const QJsonObject &params);
    ~DDXFeatDescriptorWorker() override;

    FeatureResult extract(const ImageEntry &image) override;

private:
    /// 根据图像高度选择最匹配的 ONNX 模型并加载
    bool loadModelForHeight(int imageHeight);
    /// 扫描 models 目录下所有 ddxfeat*.onnx 文件，建立高度→文件路径映射
    void scanAvailableModels();
    /// 对单张灰度图执行 DD-XFeat 推理并填充结果
    bool detectAndCompute(const cv::Mat &image, FeatureResult &outResult);

    // ── ONNX Runtime 相关 ──
    std::unique_ptr<Ort::Env> m_ortEnv;
    std::unique_ptr<Ort::Session> m_ortSession;
    std::vector<const char *> m_inputNames;
    std::vector<const char *> m_outputNames;
    std::vector<int64_t> m_inputShape;              ///< 当前模型输入shape [1,C,H,W]
    std::vector<std::string> m_inputNameStorage;    ///< 持久化输入名
    std::vector<std::string> m_outputNameStorage;   ///< 持久化输出名

    int m_modelH = 0;     ///< 当前模型输入高度
    int m_modelW = 0;     ///< 当前模型输入宽度
    int m_modelHd8 = 0;   ///< H/8
    int m_modelWd8 = 0;   ///< W/8

    // ── 可用模型列表：模型输入高度 → 文件路径 ──
    QMap<int, QString> m_availableModels;

    // 算法参数
    float m_scoreThreshold = 0.05f;
    int m_nmsKernelSize = 5;
    int m_maxFeatures = 8192;

    // ── 内部辅助函数（与 XFeat 一致，参考 xfeatc-main 实现） ──
    struct ScoredPoint {
        int x;
        int y;
        float score;
    };

    void softmaxScore(float *score, int h, int w, int c);
    void flattenScore(float *src, float *dst);
    void nms(const cv::Mat &scores, float scoreThresh, int kernelSize,
             std::vector<ScoredPoint> &points);
    void interpDescriptor(const float *descMat, float *descriptor, float ptx, float pty);
};

#endif // DDXFEATDESCRIPTORWORKER_H
