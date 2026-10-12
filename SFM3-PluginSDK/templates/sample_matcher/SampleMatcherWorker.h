#ifndef SAMPLEMATCHERWORKER_H
#define SAMPLEMATCHERWORKER_H

#include <QJsonObject>

#include "plugin/IMatcherPlugin.h"

/**
 * @brief 示例匹配 Worker（确定性伪匹配，用于验证加载与接口约定）。
 *
 * knnMatchPair：desc1 第 i 行与 desc2 的前 min(k, desc2.rows) 行依次匹配，
 *               距离为 fixedDistance*(j+1)（第 1 候选最近，可通过 Lowe 比率测试）。
 * matchPair   ：desc1 第 i 行恒定匹配 desc2 第 i 行，距离为 fixedDistance。
 *
 * 不读取描述子内容，输出完全确定，适合框架测试与插件开发参考。
 * 逐线程 / 任务独占，禁止跨线程共享使用。
 */
class SampleMatcherWorker : public IMatcherWorker
{
public:
    explicit SampleMatcherWorker(const QJsonObject &params);

    bool knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                      std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                      int k) override;
    bool matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                   std::vector<cv::DMatch> &outMatches) override;

private:
    double m_fixedDistance = 0.5;
};

#endif // SAMPLEMATCHERWORKER_H
