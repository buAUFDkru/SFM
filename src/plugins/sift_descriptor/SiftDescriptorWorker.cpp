#include "SiftDescriptorWorker.h"

#include <opencv2/features2d.hpp>

SiftDescriptorWorker::SiftDescriptorWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("nOctaveLayers")))
        m_nOctaveLayers = params.value(QStringLiteral("nOctaveLayers")).toInt(3);
    if (params.contains(QStringLiteral("contrastThreshold")))
        m_contrastThreshold = params.value(QStringLiteral("contrastThreshold")).toDouble(0.04);
    if (params.contains(QStringLiteral("edgeThreshold")))
        m_edgeThreshold = params.value(QStringLiteral("edgeThreshold")).toDouble(10.0);
    if (params.contains(QStringLiteral("sigma")))
        m_sigma = params.value(QStringLiteral("sigma")).toDouble(1.6);
    if (params.contains(QStringLiteral("maxFeatures")))
        m_maxFeatures = params.value(QStringLiteral("maxFeatures")).toInt(8192);
}

FeatureResult SiftDescriptorWorker::extract(const ImageEntry &image)
{
    if (image.imageGray.empty())
        return {};

    auto sift = cv::SIFT::create(m_maxFeatures, m_nOctaveLayers,
                                 m_contrastThreshold, m_edgeThreshold, m_sigma);

    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;
    sift->detectAndCompute(image.imageGray, cv::noArray(),
                           r.keypoints, r.descriptors);
    r.elapsedMs = 0.0;
    return r;
}