#ifndef SAMPLEMATCHERPLUGIN_H
#define SAMPLEMATCHERPLUGIN_H

#include <memory>

#include "plugin/IMatcherPlugin.h"

/**
 * @brief 示例匹配器插件（无状态工厂，DLL 插件开发模板）。
 *
 * 演示匹配器插件的三件套：paramSchema() / validateParams() / createWorker()。
 * 注意职责边界：仅实现低层两两匹配，Lowe 比率、几何验证(GRIC)与
 * 传递性匹配等编排逻辑属于 core 管线段，禁止写入插件。
 */
class SampleMatcherPlugin : public IMatcherPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "com.sfm3.matcher/1.0" FILE "plugin.json")
    Q_INTERFACES(IMatcherPlugin)

public:
    QString name() const override { return QStringLiteral("SampleMatcher"); }
    QString category() const override { return QStringLiteral("matcher"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("示例匹配器（确定性伪匹配，仅用于插件框架验证与开发参考）");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IMatcherWorker> createWorker(const QJsonObject &params) override;
};

#endif // SAMPLEMATCHERPLUGIN_H
