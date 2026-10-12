#ifndef ISFMPLUGIN_H
#define ISFMPLUGIN_H

#include <QJsonObject>
#include <QList>
#include <QObject>
#include <QString>

#include "ParamSchema.h"

/**
 * @brief 插件基接口（无状态工厂）。
 *
 * 描述符 / 匹配器的公共元信息与参数协议。实现类必须继承 QObject，
 * 以便通过 QPluginLoader 加载并能被 qobject_cast 识别。
 */
class ISfmPlugin : public QObject
{
    Q_OBJECT
public:
    explicit ISfmPlugin(QObject *parent = nullptr) : QObject(parent) {}
    ~ISfmPlugin() override = default;

    virtual QString name() const = 0;          ///< "SIFT"
    virtual QString category() const = 0;      ///< "descriptor" | "matcher" | "dense"
    virtual QString version() const = 0;       ///< "1.0.0"
    virtual QString description() const = 0;

    /// 参数协议（UI / 配置 / 校验共用的单一来源）
    virtual QList<ParamSchema> paramSchema() const = 0;

    /// 校验一组参数是否合法；失败时返回 false 并写入 errorMsg
    virtual bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const = 0;
};

Q_DECLARE_INTERFACE(ISfmPlugin, "com.sfm3.plugin/1.0")

#endif // ISFMPLUGIN_H