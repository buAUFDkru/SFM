#ifndef PARAMSCHEMAUTILS_H
#define PARAMSCHEMAUTILS_H

#include "ParamSchema.h"

#include <QJsonObject>
#include <QList>
#include <QMap>
#include <QString>
#include <QVariant>

namespace sfm {

inline ParamSchema makeIntParam(const QString &name, int def, int mn, int mx,
                                const QString &group, const QString &tip)
{
    ParamSchema s;
    s.name = name;
    s.type = ParamSchema::Type::Int;
    s.defaultValue = def;
    s.min = mn;
    s.max = mx;
    s.group = group;
    s.tooltip = tip;
    return s;
}

inline ParamSchema makeDoubleParam(const QString &name, double def, double mn, double mx,
                                   const QString &group, const QString &tip)
{
    ParamSchema s;
    s.name = name;
    s.type = ParamSchema::Type::Double;
    s.defaultValue = def;
    s.min = mn;
    s.max = mx;
    s.group = group;
    s.tooltip = tip;
    return s;
}

inline ParamSchema makeBoolParam(const QString &name, bool def,
                                 const QString &group, const QString &tip)
{
    ParamSchema s;
    s.name = name;
    s.type = ParamSchema::Type::Bool;
    s.defaultValue = def;
    s.group = group;
    s.tooltip = tip;
    return s;
}

/**
 * @brief 按 schema 校验参数（类型 / 枚举 / 数值范围）。
 *
 * 未出现在 schema 中的键予以放行（兼容 Options 中未暴露到 UI 的字段）。
 * 出现在 schema 中的键则强制检查类型与 min/max。
 */
bool validateParamsAgainstSchema(const QList<ParamSchema> &schemas,
                                 const QMap<QString, QVariant> &params,
                                 QString *errorMsg = nullptr);

bool validateParamsAgainstSchema(const QList<ParamSchema> &schemas,
                                 const QJsonObject &params,
                                 QString *errorMsg = nullptr);

/// 缺键用 schema.defaultValue 补齐；已有键原样保留；未知键保留
QJsonObject fillParamsFromSchema(const QList<ParamSchema> &schemas,
                                 const QJsonObject &params);

/**
 * @brief 对 schema 中已声明的键做软纠正：类型/范围非法则写回 defaultValue。
 * 未知键放行。返回纠正后的对象；若有纠正且 warn 非空则追加说明。
 */
QJsonObject sanitizeParamsAgainstSchema(const QList<ParamSchema> &schemas,
                                        const QJsonObject &params,
                                        QString *warn = nullptr);

/// 合并多组 schema（同名后者覆盖前者）
QList<ParamSchema> mergeParamSchemas(const QList<ParamSchema> &a,
                                     const QList<ParamSchema> &b);

} // namespace sfm

#endif // PARAMSCHEMAUTILS_H
