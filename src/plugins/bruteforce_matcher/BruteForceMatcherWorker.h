#ifndef BRUTEFORCEMATCHERWORKER_H
#define BRUTEFORCEMATCHERWORKER_H

#include <QJsonObject>

#include <opencv2/core.hpp>

#include "plugin/IMatcherPlugin.h"

/**
 * @brief BruteForce 匹配 Worker（低层两两匹配接口）。
 *
 * 构造时从 params 读取参数并持有。逐线程 / 任务独占，禁止跨线程共享使用。
 */
class BruteForceMatcherWorker : public IMatcherWorker
{
public:
    explicit BruteForceMatcherWorker(const QJsonObject &params);

    bool knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                      std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                      int k) override;
    bool matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                   std::vector<cv::DMatch> &outMatches) override;

private:
    /// 根据 normType 与描述符类型选择距离度量
    int autoNormType(const cv::Mat &desc) const;

    int m_normType = -1;
    bool m_crossCheck = false;
};

#endif // BRUTEFORCEMATCHERWORKER_H