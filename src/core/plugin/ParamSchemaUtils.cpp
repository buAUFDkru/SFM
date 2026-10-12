#include "ParamSchemaUtils.h"

#include <QHash>
#include <QMetaType>
#include <cmath>

namespace sfm {
namespace {

bool checkOne(const ParamSchema &s, const QVariant &v, QString *errorMsg)
{
    switch (s.type) {
    case ParamSchema::Type::Int: {
        bool ok = false;
        const int iv = v.toInt(&ok);
        if (!ok && v.typeId() != QMetaType::Bool) {
            // toInt 对纯数字字符串/浮点也可，这里再兜底一次
            ok = v.canConvert<int>();
        }
        if (!ok) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(s.name);
            return false;
        }
        if (s.min.isValid() && iv < s.min.toInt()) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 低于下限 %2").arg(s.name).arg(s.min.toInt());
            return false;
        }
        if (s.max.isValid() && iv > s.max.toInt()) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 超过上限 %2").arg(s.name).arg(s.max.toInt());
            return false;
        }
        return true;
    }
    case ParamSchema::Type::Double: {
        bool ok = false;
        const double dv = v.toDouble(&ok);
        if (!ok) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 类型应为 double").arg(s.name);
            return false;
        }
        if (!std::isfinite(dv)) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 不是有限数值").arg(s.name);
            return false;
        }
        if (s.min.isValid() && dv < s.min.toDouble()) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 低于下限 %2").arg(s.name).arg(s.min.toDouble());
            return false;
        }
        if (s.max.isValid() && dv > s.max.toDouble()) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 超过上限 %2").arg(s.name).arg(s.max.toDouble());
            return false;
        }
        return true;
    }
    case ParamSchema::Type::Bool: {
        if (v.typeId() != QMetaType::Bool && !v.canConvert<bool>()) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 类型应为 bool").arg(s.name);
            return false;
        }
        return true;
    }
    case ParamSchema::Type::Enum: {
        const QString sv = v.toString();
        if (!s.enumValues.contains(sv)) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 取值不在枚举列表中").arg(s.name);
            return false;
        }
        return true;
    }
    case ParamSchema::Type::String:
    case ParamSchema::Type::Path:
        if (!v.canConvert<QString>()) {
            if (errorMsg)
                *errorMsg = QStringLiteral("参数 %1 类型应为 string").arg(s.name);
            return false;
        }
        return true;
    }
    return true;
}

} // namespace

bool validateParamsAgainstSchema(const QList<ParamSchema> &schemas,
                                 const QMap<QString, QVariant> &params,
                                 QString *errorMsg)
{
    QHash<QString, ParamSchema> byName;
    byName.reserve(schemas.size());
    for (const ParamSchema &s : schemas)
        byName.insert(s.name, s);

    for (auto it = params.constBegin(); it != params.constEnd(); ++it) {
        const auto found = byName.constFind(it.key());
        if (found == byName.constEnd())
            continue; // 未声明字段放行（兼容内部 Options 扩展键）
        if (!checkOne(*found, it.value(), errorMsg))
            return false;
    }
    return true;
}

bool validateParamsAgainstSchema(const QList<ParamSchema> &schemas,
                                 const QJsonObject &params,
                                 QString *errorMsg)
{
    return validateParamsAgainstSchema(schemas, params.toVariantMap(), errorMsg);
}

QJsonObject fillParamsFromSchema(const QList<ParamSchema> &schemas,
                                 const QJsonObject &params)
{
    QJsonObject out = params;
    for (const ParamSchema &s : schemas) {
        if (!out.contains(s.name) && s.defaultValue.isValid())
            out.insert(s.name, QJsonValue::fromVariant(s.defaultValue));
    }
    return out;
}

QJsonObject sanitizeParamsAgainstSchema(const QList<ParamSchema> &schemas,
                                        const QJsonObject &params,
                                        QString *warn)
{
    QJsonObject out = params;
    QStringList notes;
    for (const ParamSchema &s : schemas) {
        if (!out.contains(s.name))
            continue;
        const QVariant v = out.value(s.name).toVariant();
        QString err;
        if (checkOne(s, v, &err))
            continue;
        if (s.defaultValue.isValid())
            out.insert(s.name, QJsonValue::fromVariant(s.defaultValue));
        else
            out.remove(s.name);
        notes.append(QStringLiteral("%1 → 默认 (%2)").arg(s.name, err));
    }
    if (warn && !notes.isEmpty()) {
        if (!warn->isEmpty())
            *warn += QLatin1Char(';');
        *warn += notes.join(QLatin1Char(','));
    }
    return out;
}

QList<ParamSchema> mergeParamSchemas(const QList<ParamSchema> &a,
                                     const QList<ParamSchema> &b)
{
    QHash<QString, ParamSchema> byName;
    byName.reserve(a.size() + b.size());
    for (const ParamSchema &s : a)
        byName.insert(s.name, s);
    for (const ParamSchema &s : b)
        byName.insert(s.name, s);
    QList<ParamSchema> out;
    out.reserve(byName.size());
    for (auto it = byName.constBegin(); it != byName.constEnd(); ++it)
        out.append(it.value());
    return out;
}

} // namespace sfm
