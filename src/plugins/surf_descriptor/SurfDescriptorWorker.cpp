#include "SurfDescriptorWorker.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

SurfDescriptorWorker::SurfDescriptorWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("hessianThreshold")))
        m_hessianThreshold = params.value(QStringLiteral("hessianThreshold")).toDouble(100.0);
    if (params.contains(QStringLiteral("nOctaves")))
        m_nOctaves = params.value(QStringLiteral("nOctaves")).toInt(4);
    if (params.contains(QStringLiteral("nOctaveLayers")))
        m_nOctaveLayers = params.value(QStringLiteral("nOctaveLayers")).toInt(3);
    if (params.contains(QStringLiteral("extended")))
        m_extended = params.value(QStringLiteral("extended")).toBool(false);
    if (params.contains(QStringLiteral("upright")))
        m_upright = params.value(QStringLiteral("upright")).toBool(false);
    if (params.contains(QStringLiteral("maxFeatures")))
        m_maxFeatures = params.value(QStringLiteral("maxFeatures")).toInt(0);
}

FeatureResult SurfDescriptorWorker::extract(const ImageEntry &image)
{
    if (image.imageGray.empty())
        return {};

    auto surf = cv::xfeatures2d::SURF::create(m_hessianThreshold, m_nOctaves,
                                              m_nOctaveLayers, m_extended, m_upright);

    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;
    surf->detectAndCompute(image.imageGray, cv::noArray(),
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