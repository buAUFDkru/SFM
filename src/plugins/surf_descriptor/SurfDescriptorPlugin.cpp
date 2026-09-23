#include "SurfDescriptorPlugin.h"

#include <QDebug>

#include "SurfDescriptorWorker.h"

QList<ParamSchema> SurfDescriptorPlugin::paramSchema() const
{
    ParamSchema hessianThreshold;
    hessianThreshold.name = QStringLiteral("hessianThreshold");
    hessianThreshold.type = ParamSchema::Type::Double;
    hessianThreshold.defaultValue = 100.0;
    hessianThreshold.min = 1.0;
    hessianThreshold.max = 100000.0;
    hessianThreshold.group = QStringLiteral("SURF");
    hessianThreshold.tooltip = QStringLiteral("Hessian 响应阈值，越大提取的点越少");

    ParamSchema nOctaves;
    nOctaves.name = QStringLiteral("nOctaves");
    nOctaves.type = ParamSchema::Type::Int;
    nOctaves.defaultValue = 4;
    nOctaves.min = 1;
    nOctaves.max = 16;
    nOctaves.group = QStringLiteral("SURF");

    ParamSchema nOctaveLayers;
    nOctaveLayers.name = QStringLiteral("nOctaveLayers");
    nOctaveLayers.type = ParamSchema::Type::Int;
    nOctaveLayers.defaultValue = 3;
    nOctaveLayers.min = 1;
    nOctaveLayers.max = 16;
    nOctaveLayers.group = QStringLiteral("SURF");

    ParamSchema extended;
    extended.name = QStringLiteral("extended");
    extended.type = ParamSchema::Type::Bool;
    extended.defaultValue = false;
    extended.group = QStringLiteral("SURF");
    extended.tooltip = QStringLiteral("是否使用扩展(128维)描述子");

    ParamSchema upright;
    upright.name = QStringLiteral("upright");
    upright.type = ParamSchema::Type::Bool;
    upright.defaultValue = false;
    upright.group = QStringLiteral("SURF");
    upright.tooltip = QStringLiteral("是否使用无旋转 Upright 描述子");

    ParamSchema maxFeatures;
    maxFeatures.name = QStringLiteral("maxFeatures");
    maxFeatures.type = ParamSchema::Type::Int;
    maxFeatures.defaultValue = 0;
    maxFeatures.min = 0;
    maxFeatures.group = QStringLiteral("SURF");
    maxFeatures.tooltip = QStringLiteral("最大特征点数量，0 表示不限制");

    return { hessianThreshold, nOctaves, nOctaveLayers, extended, upright, maxFeatures };
}

bool SurfDescriptorPlugin::validateParams(const QJsonObject &params,
                                          QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("nOctaves") ||
            key == QStringLiteral("nOctaveLayers") ||
            key == QStringLiteral("maxFeatures")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("hessianThreshold")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 double").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("extended") ||
                   key == QStringLiteral("upright")) {
            if (!params.value(key).isBool()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 bool").arg(key);
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

std::unique_ptr<IDescriptorWorker> SurfDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[SURF] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<SurfDescriptorWorker>(params);
}