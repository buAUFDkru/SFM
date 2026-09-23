#ifndef FLANNMATCHERWORKER_H
#define FLANNMATCHERWORKER_H

#include <QJsonObject>

#include "plugin/IMatcherPlugin.h"

/**
 * @brief FLANN 匹配 Worker（低层两两匹配接口）。
 *
 * 构造时从 params 读取参数并持有。逐线程 / 任务独占，禁止跨线程共享使用。
 */
class FlannMatcherWorker : public IMatcherWorker
{
public:
    explicit FlannMatcherWorker(const QJsonObject &params);

    bool knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                      std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                      int k) override;
    bool matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                   std::vector<cv::DMatch> &outMatches) override;

private:
    int m_trees = 4;
    int m_checks = 50; // 预留：保持与 sfm2.0 输出一致，未在调用中使用
};

#endif // FLANNMATCHERWORKER_H