#ifndef IMATCHERPLUGIN_H
#define IMATCHERPLUGIN_H

#include <QJsonObject>
#include <memory>

#include "ISfmPlugin.h"
#include "PluginTypes.h"

/**
 * @brief 有状态匹配 Worker（低层算法接口，与 sfm2.0 IMatcher 语义一致）。
 *
 * 只负责两两描述符的 KNN 匹配 / 单最佳匹配；候选对选择、Lowe 比率、
 * 几何验证(GRIC)与传递性匹配等编排逻辑驻留在 core 管线段（P3），
 * 保证插件输出与 sfm2.0 完全一致。逐线程 / 任务独占，禁止跨线程共享。
 */
class IMatcherWorker
{
public:
    virtual ~IMatcherWorker() = default;

    /// 两两 KNN 匹配；调用方在其上做 Lowe 比率测试
    virtual bool knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                              std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                              int k) = 0;

    /// 两两单最佳匹配
    virtual bool matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                           std::vector<cv::DMatch> &outMatches) = 0;
};

/**
 * @brief 匹配插件（无状态工厂）。
 */
class IMatcherPlugin : public ISfmPlugin
{
    Q_OBJECT
public:
    explicit IMatcherPlugin(QObject *parent = nullptr) : ISfmPlugin(parent) {}

    virtual std::unique_ptr<IMatcherWorker> createWorker(const QJsonObject &params) = 0;
};

Q_DECLARE_INTERFACE(IMatcherPlugin, "com.sfm3.matcher/1.0")

#endif // IMATCHERPLUGIN_H