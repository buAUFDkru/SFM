#include "SampleDescriptorPlugin.h"

#include <QDebug>

#include "SampleDescriptorWorker.h"

QList<ParamSchema> SampleDescriptorPlugin::paramSchema() const
{
    ParamSchema scale;
    scale.name = QStringLiteral("scale");
    scale.type = ParamSchema::Type::Double;
    scale.defaultValue = 1.0;
    scale.min = 0.1;
    scale.max = 10.0;
    scale.group = QStringLiteral("示例");
    return { scale };
}

bool SampleDescriptorPlugin::validateParams(const QJsonObject &params,
                                            QString *errorMsg) const
{
    if (!params.isEmpty() && !params.contains(QStringLiteral("scale"))) {
        if (errorMsg)
            *errorMsg = QStringLiteral("缺少参数 scale");
        return false;
    }
    if (params.contains(QStringLiteral("scale"))) {
        const QJsonValue v = params.value(QStringLiteral("scale"));
        if (!v.isDouble()) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 scale 类型应为 double");
            return false;
        }
    }
    return true;
}

std::unique_ptr<IDescriptorWorker> SampleDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[SampleDescriptor] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<SampleDescriptorWorker>(params);
}