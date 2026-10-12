#ifndef PARAMSCHEMA_H
#define PARAMSCHEMA_H

#include <QString>
#include <QStringList>
#include <QVariant>

/**
 * @brief 算法参数协议。
 *
 * 每个插件通过 paramSchema() 声明其可配置参数，作为 UI 自动生成、
 * 配置序列化与校验的单一事实来源。
 */
struct ParamSchema
{
    enum class Type : int { Int, Double, Bool, String, Enum, Path };

    QString name;                 ///< 参数键名
    Type    type = Type::Int;
    QVariant defaultValue;
    QVariant min;                 ///< 数值类型下限（可选）
    QVariant max;                 ///< 数值类型上限（可选）
    QStringList enumValues;       ///< Enum 类型候选项
    QString group;                ///< UI 分组
    QString tooltip;              ///< 提示文案
};

#endif // PARAMSCHEMA_H