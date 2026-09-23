#include "KazeDescriptorWorker.h"

#include <opencv2/features2d.hpp>

KazeDescriptorWorker::KazeDescriptorWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("extended")))
        m_extended = params.value(QStringLiteral("extended")).toBool(false);
    if (params.contains(QStringLiteral("upright")))
        m_upright = params.value(QStringLiteral("upright")).toBool(false);
    if (params.contains(QStringLiteral("threshold")))
        m_threshold = params.value(QStringLiteral("threshold")).toDouble(0.001);
    if (params.contains(QStringLiteral("nOctaves")))
        m_nOctaves = params.value(QStringLiteral("nOctaves")).toInt(4);
    if (params.contains(QStringLiteral("nOctaveLayers")))
        m_nOctaveLayers = params.value(QStringLiteral("nOctaveLayers")).toInt(4);
    if (params.contains(QStringLiteral("diffusivity")))
        m_diffusivity = params.value(QStringLiteral("diffusivity")).toInt(1);
    if (params.contains(QStringLiteral("maxFeatures")))
        m_maxFeatures = params.value(QStringLiteral("maxFeatures")).toInt(0);
}

FeatureResult KazeDescriptorWorker::extract(const ImageEntry &image)
{
    if (image.imageGray.empty())
        return {};

    auto kaze = cv::KAZE::create(m_extended, m_upright, m_threshold,
                                 m_nOctaves, m_nOctaveLayers,
                                 static_cast<cv::KAZE::DiffusivityType>(m_diffusivity));

    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;
    kaze->detectAndCompute(image.imageGray, cv::noArray(),
                           r.keypoints, r.descriptors);

    // 限制特征点最大数量
    if (m_maxFeatures > 0 &&
        static_cast<int>(r.keypoints.size()) > m_maxFeatures) {
        r.keypoints.resize(m_maxFeatures);
        r.descriptors = r.descriptors.rowRange(0, m_maxFeatures).clone();
    }

    r.elapsedMs = 0.0;
    return r;
}