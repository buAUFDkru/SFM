#include "SampleMatcherWorker.h"

SampleMatcherWorker::SampleMatcherWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("fixedDistance")))
        m_fixedDistance = params.value(QStringLiteral("fixedDistance")).toDouble(0.5);
}

bool SampleMatcherWorker::knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                       std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                                       int k)
{
    outKnnMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    const int rows1 = desc1.rows;
    const int rows2 = desc2.rows;
    const int perQuery = std::min(std::max(k, 1), rows2);

    outKnnMatches.reserve(static_cast<size_t>(rows1));
    for (int i = 0; i < rows1; ++i) {
        std::vector<cv::DMatch> candidates;
        candidates.reserve(static_cast<size_t>(perQuery));
        for (int j = 0; j < perQuery; ++j) {
            candidates.emplace_back(i, j,
                                    static_cast<float>(m_fixedDistance * (j + 1)));
        }
        outKnnMatches.push_back(std::move(candidates));
    }
    return true;
}

bool SampleMatcherWorker::matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                    std::vector<cv::DMatch> &outMatches)
{
    outMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    const int n = std::min(desc1.rows, desc2.rows);
    outMatches.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        outMatches.emplace_back(i, i, static_cast<float>(m_fixedDistance));
    return true;
}
