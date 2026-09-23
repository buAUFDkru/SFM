#include "BruteForceMatcherWorker.h"

#include <opencv2/features2d.hpp>

BruteForceMatcherWorker::BruteForceMatcherWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("normType")))
        m_normType = params.value(QStringLiteral("normType")).toInt(-1);
    if (params.contains(QStringLiteral("crossCheck")))
        m_crossCheck = params.value(QStringLiteral("crossCheck")).toBool(false);
}

int BruteForceMatcherWorker::autoNormType(const cv::Mat &desc) const
{
    if (m_normType >= 0)
        return m_normType;
    // 根据描述符数据类型自动选择距离度量
    // CV_8U  → 二进制描述符(ORB 等) → Hamming 距离
    // 其他   → 浮点描述符(SIFT/SURF 等) → L2 距离
    return (desc.type() == CV_8U) ? cv::NORM_HAMMING : cv::NORM_L2;
}

bool BruteForceMatcherWorker::knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                           std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                                           int k)
{
    outKnnMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    auto bfMatcher = cv::BFMatcher::create(autoNormType(desc1), m_crossCheck);
    bfMatcher->knnMatch(desc1, desc2, outKnnMatches, k);
    return !outKnnMatches.empty();
}

bool BruteForceMatcherWorker::matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                        std::vector<cv::DMatch> &outMatches)
{
    outMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    auto bfMatcher = cv::BFMatcher::create(autoNormType(desc1), m_crossCheck);
    bfMatcher->match(desc1, desc2, outMatches);
    return !outMatches.empty();
}