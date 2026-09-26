#include "SuperPointDescriptorPlugin.h"

#include <QDebug>

#include "SuperPointDescriptorWorker.h"

QList<ParamSchema> SuperPointDescriptorPlugin::paramSchema() const
{
    ParamSchema scoreThreshold;
    scoreThreshold.name = QStringLiteral("scoreThreshold");
    scoreThreshold.type = ParamSchema::Type::Double;
    scoreThreshold.defaultValue = 0.005;
    scoreThreshold.min = 0.0;
    scoreThreshold.max = 1.0;
    scoreThreshold.group = QStringLiteral("SuperPoint");
    scoreThreshold.tooltip = QStringLiteral("检测阈值，越大提取的点越少");

    ParamSchema maxFeatures;
    maxFeatures.name = QStringLiteral("maxFeatures");
    maxFeatures.type = ParamSchema::Type::Int;
    maxFeatures.defaultValue = 8192;
    maxFeatures.min = 0;
    maxFeatures.group = QStringLiteral("SuperPoint");
    maxFeatures.tooltip = QStringLiteral("最大特征点数量，0 表示不限制");

    ParamSchema nmsKernelSize;
    nmsKernelSize.name = QStringLiteral("nmsKernelSize");
    nmsKernelSize.type = ParamSchema::Type::Int;
    nmsKernelSize.defaultValue = 9;
    nmsKernelSize.min = 1;
    nmsKernelSize.max = 15;
    nmsKernelSize.group = QStringLiteral("SuperPoint");
    nmsKernelSize.tooltip = QStringLiteral("NMS 窗口尺寸(2*nms_radius+1)，SuperPoint 默认 9");

    ParamSchema nmsMode;
    nmsMode.name = QStringLiteral("nmsMode");
    nmsMode.type = ParamSchema::Type::Int;
    nmsMode.defaultValue = 2;
    nmsMode.min = 0;
    nmsMode.max = 2;
    nmsMode.group = QStringLiteral("SuperPoint");
    nmsMode.enumValues = { QStringLiteral("xfeatc 掩码"), QStringLiteral("dilate"),
                           QStringLiteral("官方 batched") };
    nmsMode.tooltip = QStringLiteral("NMS 策略：2=官方 batched(与 pytorch 参考逐点一致) 1=dilate 0=xfeatc 掩码");

    ParamSchema padMultiple;
    padMultiple.name = QStringLiteral("padMultiple");
    padMultiple.type = ParamSchema::Type::Int;
    padMultiple.defaultValue = 32;
    padMultiple.min = 8;
    padMultiple.max = 128;
    padMultiple.group = QStringLiteral("SuperPoint");
    padMultiple.tooltip = QStringLiteral("图像 padding 到该值的倍数（须为 8 的倍数）");

    ParamSchema intraThreads;
    intraThreads.name = QStringLiteral("intraThreads");
    intraThreads.type = ParamSchema::Type::Int;
    intraThreads.defaultValue = 4;
    intraThreads.min = 1;
    intraThreads.max = 8;
    intraThreads.group = QStringLiteral("SuperPoint");
    intraThreads.tooltip = QStringLiteral("ONNX Runtime intra-op 线程数");

    return { scoreThreshold, maxFeatures, nmsKernelSize, nmsMode, padMultiple, intraThreads };
}

bool SuperPointDescriptorPlugin::validateParams(const QJsonObject &params,
                                                QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("maxFeatures") ||
            key == QStringLiteral("nmsKernelSize") ||
            key == QStringLiteral("nmsMode") ||
            key == QStringLiteral("padMultiple") ||
            key == QStringLiteral("intraThreads")) {
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

std::unique_ptr<IDescriptorWorker> SuperPointDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[SuperPoint] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<SuperPointDescriptorWorker>(params);
}
