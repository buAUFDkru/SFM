#ifndef DDXFEATDESCRIPTORPLUGIN_H
#define DDXFEATDESCRIPTORPLUGIN_H

#include <memory>

#include "plugin/IDescriptorPlugin.h"

/**
 * @brief DD-XFeat 描述符插件（无状态工厂）。
 *
 * DD-XFeat 为 XFeat 的改进卷积神经网络版本：仅 CNN 主干不同，
 * 后处理逻辑与 XFeat 完全一致（softmax → NMS → 描述符插值）。
 * 依据 params 校验并创建独立的 DDXFeatDescriptorWorker，
 * 工厂自身不持有算法对象。
 */
class DDXFeatDescriptorPlugin : public IDescriptorPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "com.sfm3.descriptor/1.0" FILE "plugin.json")
    Q_INTERFACES(IDescriptorPlugin)

public:
    QString name() const override { return QStringLiteral("DD-XFeat"); }
    QString category() const override { return QStringLiteral("descriptor"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("XFeat 改进卷积网络版关键点检测与描述子提取(ONNX Runtime 推理)");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IDescriptorWorker> createWorker(const QJsonObject &params) override;
};

#endif // DDXFEATDESCRIPTORPLUGIN_H
