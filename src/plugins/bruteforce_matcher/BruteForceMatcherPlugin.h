#ifndef BRUTEFORCEMATCHERPLUGIN_H
#define BRUTEFORCEMATCHERPLUGIN_H

#include <memory>

#include "plugin/IMatcherPlugin.h"

/**
 * @brief BruteForce 匹配器插件（无状态工厂）。
 *
 * 依据 params 校验并创建独立的 BruteForceMatcherWorker，工厂自身不持有算法对象。
 */
class BruteForceMatcherPlugin : public IMatcherPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "com.sfm3.matcher/1.0" FILE "plugin.json")
    Q_INTERFACES(IMatcherPlugin)

public:
    QString name() const override { return QStringLiteral("BruteForce"); }
    QString category() const override { return QStringLiteral("matcher"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于暴力穷举(Brute-Force)的描述符匹配器");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IMatcherWorker> createWorker(const QJsonObject &params) override;
};

#endif // BRUTEFORCEMATCHERPLUGIN_H