#include "SampleDescriptorWorker.h"

#include <opencv2/core.hpp>

#include <algorithm>

SampleDescriptorWorker::SampleDescriptorWorker(const QJsonObject &params)
{
    m_scale = params.contains("scale") ? params.value("scale").toDouble(1.0) : 1.0;
}

FeatureResult SampleDescriptorWorker::extract(const ImageEntry &image)
{
    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;

    // 确定性伪特征：点数 = rows/10，位置沿 x 轴均匀取点
    const int n = std::max(0, image.imageGray.rows / 10);
    const int spacing = image.imageGray.rows / std::max(1, n);
    r.keypoints.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        r.keypoints.emplace_back(static_cast<float>(i * spacing * m_scale),
                                 0.0f, 1.0f);
    }

    // 确定性描述子矩阵：N x 8，填充常值
    r.descriptors = cv::Mat(n, 8, CV_32F, cv::Scalar(0.5f));

    r.elapsedMs = 0.0;
    return r;
}