#ifndef SAMPLEDESCRIPTORPLUGIN_H
#define SAMPLEDESCRIPTORPLUGIN_H

#include <memory>

#include "plugin/IDescriptorPlugin.h"

/**
 * @brief P1 示例描述符插件（无状态工厂）。
 *
 * 仅用于验证：DLL 加载链路、参数协议读取、工厂 -> Worker 的并发一致性。
 */
class SampleDescriptorPlugin : public IDescriptorPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "com.sfm3.descriptor/1.0" FILE "plugin.json")
    Q_INTERFACES(IDescriptorPlugin)

public:
    QString name() const override { return QStringLiteral("SampleDescriptor"); }
    QString category() const override { return QStringLiteral("descriptor"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("P1 sample deterministic descriptor plugin (dummy output)");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IDescriptorWorker> createWorker(const QJsonObject &params) override;
};

#endif // SAMPLEDESCRIPTORPLUGIN_H