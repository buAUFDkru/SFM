#include "SampleMatcherPlugin.h"

#include <QDebug>

#include "SampleMatcherWorker.h"

QList<ParamSchema> SampleMatcherPlugin::paramSchema() const
{
    ParamSchema fixedDistance;
    fixedDistance.name = QStringLiteral("fixedDistance");
    fixedDistance.type = ParamSchema::Type::Double;
    fixedDistance.defaultValue = 0.5;
    fixedDistance.min = 0.01;
    fixedDistance.max = 10.0;
    fixedDistance.group = QStringLiteral("示例");
    fixedDistance.tooltip = QStringLiteral("伪匹配的固定距离值");
    return { fixedDistance };
}

bool SampleMatcherPlugin::validateParams(const QJsonObject &params,
                                         QString *errorMsg) const
{
    if (!params.isEmpty() && !params.contains(QStringLiteral("fixedDistance"))) {
        if (errorMsg)
            *errorMsg = QStringLiteral("未知参数（仅支持 fixedDistance）");
        return false;
    }
    if (params.contains(QStringLiteral("fixedDistance"))) {
        const QJsonValue v = params.value(QStringLiteral("fixedDistance"));
        if (!v.isDouble()) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 fixedDistance 类型应为 double");
            return false;
        }
    }
    return true;
}

std::unique_ptr<IMatcherWorker> SampleMatcherPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[SampleMatcher] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<SampleMatcherWorker>(params);
}
