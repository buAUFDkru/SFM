#include "FlannMatcherWorker.h"

#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

namespace {

cv::Ptr<cv::FlannBasedMatcher> makeFlannMatcher(int trees, int descType)
{
    if (descType == CV_8U) {
        // 二进制描述符（ORB 等）→ LSH 索引
        return cv::makePtr<cv::FlannBasedMatcher>(
            cv::makePtr<cv::flann::LshIndexParams>(20, 10, 2));
    }
    // 浮点描述符（SIFT/SURF 等）→ KD 树索引
    return cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(trees));
}

} // namespace

FlannMatcherWorker::FlannMatcherWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("trees")))
        m_trees = params.value(QStringLiteral("trees")).toInt(4);
    if (params.contains(QStringLiteral("checks")))
        m_checks = params.value(QStringLiteral("checks")).toInt(50);
}

bool FlannMatcherWorker::knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                      std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                                      int k)
{
    outKnnMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    auto flannMatcher = makeFlannMatcher(m_trees, desc1.type());
    flannMatcher->knnMatch(desc1, desc2, outKnnMatches, k);
    return !outKnnMatches.empty();
}

bool FlannMatcherWorker::matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                   std::vector<cv::DMatch> &outMatches)
{
    outMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    auto flannMatcher = makeFlannMatcher(m_trees, desc1.type());
    flannMatcher->match(desc1, desc2, outMatches);
    return !outMatches.empty();
}