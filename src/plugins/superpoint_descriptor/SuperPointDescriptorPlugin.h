#ifndef SUPERPOINTDESCRIPTORPLUGIN_H
#define SUPERPOINTDESCRIPTORPLUGIN_H

#include <memory>

#include "plugin/IDescriptorPlugin.h"

/**
 * @brief SuperPoint 描述符插件（无状态工厂，ONNX Runtime 推理）。
 *
 * 依据 params 校验并创建独立的 SuperPointDescriptorWorker，工厂自身不持有
 * 模型与算法对象。模型文件（superpoint*.onnx）部署在 applicationDirPath()/models。
 */
class SuperPointDescriptorPlugin : public IDescriptorPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "com.sfm3.descriptor/1.0" FILE "plugin.json")
    Q_INTERFACES(IDescriptorPlugin)

public:
    QString name() const override { return QStringLiteral("SuperPoint"); }
    QString category() const override { return QStringLiteral("descriptor"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于 SuperPoint 深度学习特征的关键点检测与 256 维描述子提取(ONNX Runtime 推理)");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IDescriptorWorker> createWorker(const QJsonObject &params) override;
};

#endif // SUPERPOINTDESCRIPTORPLUGIN_H
