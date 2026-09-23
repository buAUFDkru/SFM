#include "XFeatDescriptorPlugin.h"

#include <QDebug>

#include "XFeatDescriptorWorker.h"

QList<ParamSchema> XFeatDescriptorPlugin::paramSchema() const
{
    ParamSchema scoreThreshold;
    scoreThreshold.name = QStringLiteral("scoreThreshold");
    scoreThreshold.type = ParamSchema::Type::Double;
    scoreThreshold.defaultValue = 0.05;
    scoreThreshold.min = 0.001;
    scoreThreshold.max = 0.5;
    scoreThreshold.group = QStringLiteral("XFeat");
    scoreThreshold.tooltip = QStringLiteral("关键点得分阈值");

    ParamSchema nmsKernelSize;
    nmsKernelSize.name = QStringLiteral("nmsKernelSize");
    nmsKernelSize.type = ParamSchema::Type::Int;
    nmsKernelSize.defaultValue = 5;
    nmsKernelSize.min = 3;
    nmsKernelSize.max = 9;
    nmsKernelSize.group = QStringLiteral("XFeat");
    nmsKernelSize.tooltip = QStringLiteral("NMS 窗口大小，应为奇数");

    ParamSchema maxFeatures;
    maxFeatures.name = QStringLiteral("maxFeatures");
    maxFeatures.type = ParamSchema::Type::Int;
    maxFeatures.defaultValue = 8192;
    maxFeatures.min = 0;
    maxFeatures.group = QStringLiteral("XFeat");
    maxFeatures.tooltip = QStringLiteral("最大特征点数量，0 表示不限制");

    return { scoreThreshold, nmsKernelSize, maxFeatures };
}

bool XFeatDescriptorPlugin::validateParams(const QJsonObject &params,
                                           QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("nmsKernelSize") ||
            key == QStringLiteral("maxFeatures")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("scoreThreshold")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 double").arg(key);
                return false;
            }
        } else {
            if (errorMsg)
                *errorMsg = QStringLiteral("未知参数 %1").arg(key);
            return false;
        }
    }
    return true;
}

std::unique_ptr<IDescriptorWorker> XFeatDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[XFeat] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<XFeatDescriptorWorker>(params);
}