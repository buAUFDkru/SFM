#ifndef SUPERPOINTDESCRIPTORWORKER_H
#define SUPERPOINTDESCRIPTORWORKER_H

#include <QJsonObject>
#include <QString>
#include <memory>

#include <opencv2/core.hpp>

#include "plugin/IDescriptorPlugin.h"
#include "SuperPoint.h"

/**
 * @brief SuperPoint 描述符 Worker（基于 ONNX Runtime 推理）。
 *
 * 构造时从 params 读取参数并持有；首次 extract() 懒加载 ONNX 模型
 * （applicationDirPath()/models/ 下第一个 superpoint*.onnx，动态 shape），
 * 推理与后处理（softmax/NMS/描述子采样）复用上游 SuperPoint 实现。
 * 逐线程 / 任务独占，禁止跨线程共享使用。
 */
class SuperPointDescriptorWorker : public IDescriptorWorker
{
public:
    explicit SuperPointDescriptorWorker(const QJsonObject &params);

    FeatureResult extract(const ImageEntry &image) override;

private:
    /// 懒加载 ONNX 模型；成功返回 true，之后 m_superPoint 可用
    bool ensureModelLoaded();

    QString m_modelPath;
    bool m_modelScanned = false;
    std::unique_ptr<SuperPoint> m_superPoint;   ///< 上游推理实现（session + 后处理）

    // ── SuperPoint::Config 参数 ──
    int   m_nmsKernelSize = 9;      ///< NMS 窗口尺寸（2*nms_radius+1），SuperPoint 默认 9
    float m_scoreThresh   = 0.005f; ///< 检测阈值
    int   m_maxFeatures   = 8192;   ///< 最大特征点数量，<=0 表示不限制
    int   m_padMultiple   = 32;     ///< 图像 padding 到该值倍数（须为 8 的倍数）
    int   m_intraThreads  = 4;      ///< ONNX Runtime intra-op 线程数
    int   m_nmsMode       = 2;      ///< NMS 策略：2=官方 batched 1=dilate 0=xfeatc 掩码
};

#endif // SUPERPOINTDESCRIPTORWORKER_H
