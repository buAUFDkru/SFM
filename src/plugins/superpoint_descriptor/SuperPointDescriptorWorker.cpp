#include "SuperPointDescriptorWorker.h"

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFileInfo>

SuperPointDescriptorWorker::SuperPointDescriptorWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("nmsKernelSize")))
        m_nmsKernelSize = params.value(QStringLiteral("nmsKernelSize")).toInt(9);
    if (params.contains(QStringLiteral("scoreThreshold")))
        m_scoreThresh = static_cast<float>(
            params.value(QStringLiteral("scoreThreshold")).toDouble(0.005));
    if (params.contains(QStringLiteral("maxFeatures")))
        m_maxFeatures = params.value(QStringLiteral("maxFeatures")).toInt(8192);
    if (params.contains(QStringLiteral("padMultiple")))
        m_padMultiple = params.value(QStringLiteral("padMultiple")).toInt(32);
    if (params.contains(QStringLiteral("intraThreads")))
        m_intraThreads = params.value(QStringLiteral("intraThreads")).toInt(4);
    if (params.contains(QStringLiteral("nmsMode")))
        m_nmsMode = params.value(QStringLiteral("nmsMode")).toInt(2);
}

FeatureResult SuperPointDescriptorWorker::extract(const ImageEntry &image)
{
    if (image.imageGray.empty())
        return {};

    if (!ensureModelLoaded()) {
        qWarning() << "[SuperPoint] 模型不可用，返回空特征";
        return {};
    }

    std::vector<cv::KeyPoint> keys;
    cv::Mat descs;
    try {
        m_superPoint->DetectAndCompute(image.imageGray, keys, descs);
    } catch (const std::exception &e) {
        qWarning() << "[SuperPoint] 推理失败:" << e.what();
        return {};
    }

    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;
    r.keypoints = std::move(keys);
    r.descriptors = descs;   // N x 256, CV_32F, L2 归一化
    r.elapsedMs = 0.0;
    return r;
}

bool SuperPointDescriptorWorker::ensureModelLoaded()
{
    if (m_modelScanned)
        return m_superPoint != nullptr;
    m_modelScanned = true;

    // 扫描可执行程序目录下的 models 目录，取第一个 superpoint*.onnx（动态 shape）
    QDir modelsDir(QCoreApplication::applicationDirPath() + QStringLiteral("/models"));
    const QStringList files = modelsDir.entryList({ QStringLiteral("superpoint*.onnx") },
                                                  QDir::Files, QDir::Name);
    if (files.isEmpty()) {
        qWarning() << "[SuperPoint] 警告: 未在" << modelsDir.absolutePath()
                   << "下找到任何 superpoint*.onnx 模型文件";
        return false;
    }
    m_modelPath = modelsDir.filePath(files.first());

    SuperPoint::Config cfg;
    cfg.nmsKernelSize = m_nmsKernelSize;
    cfg.scoreThresh   = m_scoreThresh;
    cfg.maxKeypoints  = m_maxFeatures;
    cfg.padMultiple   = m_padMultiple;
    cfg.intraThreads  = m_intraThreads;
    cfg.nmsMode       = m_nmsMode;

    try {
        m_superPoint = std::make_unique<SuperPoint>(m_modelPath.toStdString(), cfg);
    } catch (const std::exception &e) {
        qWarning() << "[SuperPoint] 模型加载失败:" << m_modelPath << "-" << e.what();
        return false;
    }
    return true;
}
