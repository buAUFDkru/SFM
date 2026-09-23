#ifndef IMATCHERPLUGIN_H
#define IMATCHERPLUGIN_H

#include <QJsonObject>
#include <memory>

#include "ISfmPlugin.h"
#include "PluginTypes.h"

/**
 * @brief 有状态匹配 Worker：内部持有独立的特征索引实例。
 *
 * 逐线程 / 任务独占，禁止跨线程共享。
 */
class IMatcherWorker
{
public:
    virtual ~IMatcherWorker() = default;

    virtual void buildIndex(const std::vector<FeatureResult> &features) = 0; ///< 单线程
    virtual std::vector<MatchPairResult> matchPairs() = 0;                    ///< 单线程
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