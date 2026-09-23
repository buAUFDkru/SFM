#include "BruteForceMatcherPlugin.h"

#include <QDebug>

#include "BruteForceMatcherWorker.h"

QList<ParamSchema> BruteForceMatcherPlugin::paramSchema() const
{
    ParamSchema normType;
    normType.name = QStringLiteral("normType");
    normType.type = ParamSchema::Type::Int;
    normType.defaultValue = -1;
    normType.min = -1;
    normType.max = 5;
    normType.group = QStringLiteral("BruteForce");
    normType.tooltip = QStringLiteral("距离度量：-1=按描述符类型自动选择(二进制→Hamming，浮点→L2)；可显式指定 0=NORM_INF 1=NORM_L1 2=NORM_L2 4=NORM_L2SQR 5=NORM_HAMMING 6=NORM_HAMMING2");

    ParamSchema crossCheck;
    crossCheck.name = QStringLiteral("crossCheck");
    crossCheck.type = ParamSchema::Type::Bool;
    crossCheck.defaultValue = false;
    crossCheck.group = QStringLiteral("BruteForce");
    crossCheck.tooltip = QStringLiteral("是否启用交叉验证，仅返回双向一致的匹配");

    return { normType, crossCheck };
}

bool BruteForceMatcherPlugin::validateParams(const QJsonObject &params,
                                             QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("normType")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("crossCheck")) {
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

std::unique_ptr<IMatcherWorker> BruteForceMatcherPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[BruteForce] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<BruteForceMatcherWorker>(params);
}