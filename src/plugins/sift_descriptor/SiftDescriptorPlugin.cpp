#include "SiftDescriptorPlugin.h"

#include <QDebug>

#include "SiftDescriptorWorker.h"

QList<ParamSchema> SiftDescriptorPlugin::paramSchema() const
{
    ParamSchema nOctaveLayers;
    nOctaveLayers.name = QStringLiteral("nOctaveLayers");
    nOctaveLayers.type = ParamSchema::Type::Int;
    nOctaveLayers.defaultValue = 3;
    nOctaveLayers.min = 1;
    nOctaveLayers.max = 10;
    nOctaveLayers.group = QStringLiteral("SIFT");

    ParamSchema contrastThreshold;
    contrastThreshold.name = QStringLiteral("contrastThreshold");
    contrastThreshold.type = ParamSchema::Type::Double;
    contrastThreshold.defaultValue = 0.04;
    contrastThreshold.min = 0.0;
    contrastThreshold.max = 1.0;
    contrastThreshold.group = QStringLiteral("SIFT");
    contrastThreshold.tooltip = QStringLiteral("对比度阈值，过滤低对比度关键点");

    ParamSchema edgeThreshold;
    edgeThreshold.name = QStringLiteral("edgeThreshold");
    edgeThreshold.type = ParamSchema::Type::Double;
    edgeThreshold.defaultValue = 10.0;
    edgeThreshold.min = 1.0;
    edgeThreshold.max = 100.0;
    edgeThreshold.group = QStringLiteral("SIFT");
    edgeThreshold.tooltip = QStringLiteral("边缘阈值，过滤位于边缘的关键点");

    ParamSchema sigma;
    sigma.name = QStringLiteral("sigma");
    sigma.type = ParamSchema::Type::Double;
    sigma.defaultValue = 1.6;
    sigma.min = 0.1;
    sigma.max = 10.0;
    sigma.group = QStringLiteral("SIFT");
    sigma.tooltip = QStringLiteral("高斯金字塔初始尺度");

    ParamSchema maxFeatures;
    maxFeatures.name = QStringLiteral("maxFeatures");
    maxFeatures.type = ParamSchema::Type::Int;
    maxFeatures.defaultValue = 8192;
    maxFeatures.min = 0;
    maxFeatures.group = QStringLiteral("SIFT");
    maxFeatures.tooltip = QStringLiteral("最大特征点数量，0 表示不限制");

    return { nOctaveLayers, contrastThreshold, edgeThreshold, sigma, maxFeatures };
}

bool SiftDescriptorPlugin::validateParams(const QJsonObject &params,
                                          QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("nOctaveLayers") ||
            key == QStringLiteral("maxFeatures")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("contrastThreshold") ||
                   key == QStringLiteral("edgeThreshold") ||
                   key == QStringLiteral("sigma")) {
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

std::unique_ptr<IDescriptorWorker> SiftDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[SIFT] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<SiftDescriptorWorker>(params);
}