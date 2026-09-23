#include "KazeDescriptorPlugin.h"

#include <QDebug>

#include "KazeDescriptorWorker.h"

QList<ParamSchema> KazeDescriptorPlugin::paramSchema() const
{
    ParamSchema extended;
    extended.name = QStringLiteral("extended");
    extended.type = ParamSchema::Type::Bool;
    extended.defaultValue = false;
    extended.group = QStringLiteral("KAZE");
    extended.tooltip = QStringLiteral("是否使用扩展(128维)描述子");

    ParamSchema upright;
    upright.name = QStringLiteral("upright");
    upright.type = ParamSchema::Type::Bool;
    upright.defaultValue = false;
    upright.group = QStringLiteral("KAZE");
    upright.tooltip = QStringLiteral("是否使用无旋转 Upright 描述子");

    ParamSchema threshold;
    threshold.name = QStringLiteral("threshold");
    threshold.type = ParamSchema::Type::Double;
    threshold.defaultValue = 0.001;
    threshold.min = 0.0;
    threshold.group = QStringLiteral("KAZE");
    threshold.tooltip = QStringLiteral("检测响应阈值");

    ParamSchema nOctaves;
    nOctaves.name = QStringLiteral("nOctaves");
    nOctaves.type = ParamSchema::Type::Int;
    nOctaves.defaultValue = 4;
    nOctaves.min = 1;
    nOctaves.max = 16;
    nOctaves.group = QStringLiteral("KAZE");

    ParamSchema nOctaveLayers;
    nOctaveLayers.name = QStringLiteral("nOctaveLayers");
    nOctaveLayers.type = ParamSchema::Type::Int;
    nOctaveLayers.defaultValue = 4;
    nOctaveLayers.min = 1;
    nOctaveLayers.max = 16;
    nOctaveLayers.group = QStringLiteral("KAZE");

    ParamSchema diffusivity;
    diffusivity.name = QStringLiteral("diffusivity");
    diffusivity.type = ParamSchema::Type::Int;
    diffusivity.defaultValue = 1;
    diffusivity.min = 0;
    diffusivity.max = 3;
    diffusivity.group = QStringLiteral("KAZE");
    diffusivity.enumValues = { QStringLiteral("PM_G1"), QStringLiteral("PM_G2"),
                               QStringLiteral("WEICKERT"), QStringLiteral("CHARBONNIER") };
    diffusivity.tooltip = QStringLiteral("扩散类型：0=PM_G1 1=PM_G2 2=WEICKERT 3=CHARBONNIER");

    ParamSchema maxFeatures;
    maxFeatures.name = QStringLiteral("maxFeatures");
    maxFeatures.type = ParamSchema::Type::Int;
    maxFeatures.defaultValue = 0;
    maxFeatures.min = 0;
    maxFeatures.group = QStringLiteral("KAZE");
    maxFeatures.tooltip = QStringLiteral("最大特征点数量，0 表示不限制");

    return { extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity, maxFeatures };
}

bool KazeDescriptorPlugin::validateParams(const QJsonObject &params,
                                          QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("extended") ||
            key == QStringLiteral("upright")) {
            if (!params.value(key).isBool()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 bool").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("nOctaves") ||
                   key == QStringLiteral("nOctaveLayers") ||
                   key == QStringLiteral("diffusivity") ||
                   key == QStringLiteral("maxFeatures")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("threshold")) {
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

std::unique_ptr<IDescriptorWorker> KazeDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[KAZE] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<KazeDescriptorWorker>(params);
}