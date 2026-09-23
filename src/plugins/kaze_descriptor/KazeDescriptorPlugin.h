#ifndef KAZEDESCRIPTORPLUGIN_H
#define KAZEDESCRIPTORPLUGIN_H

#include <memory>

#include "plugin/IDescriptorPlugin.h"

/**
 * @brief KAZE 描述符插件（无状态工厂）。
 *
 * 依据 params 校验并创建独立的 KazeDescriptorWorker，工厂自身不持有算法对象。
 */
class KazeDescriptorPlugin : public IDescriptorPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "com.sfm3.descriptor/1.0" FILE "plugin.json")
    Q_INTERFACES(IDescriptorPlugin)

public:
    QString name() const override { return QStringLiteral("KAZE"); }
    QString category() const override { return QStringLiteral("descriptor"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于非线性尺度空间的 KAZE 关键点检测与描述子提取");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IDescriptorWorker> createWorker(const QJsonObject &params) override;
};

#endif // KAZEDESCRIPTORPLUGIN_H