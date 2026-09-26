#ifndef PLUGINREGISTRY_H
#define PLUGINREGISTRY_H

#include <QMap>
#include <QObject>
#include <QPluginLoader>
#include <QSet>

#include "PluginInfo.h"

class IDescriptorPlugin;
class IMatcherPlugin;
class ISfmPlugin;

/**
 * @brief 插件注册表（单例）。
 *
 * 扫描指定目录下的 *.dll，通过 QPluginLoader 加载并按接口类型注册；
 * 同时支持注册编译进 core 的内置算法（registerBuiltIn，无需 DLL）。
 * 持有插件/loader 的完整生命周期；加载失败或非法插件跳过，不阻塞其他插件。
 */
class PluginRegistry : public QObject
{
    Q_OBJECT
public:
    static PluginRegistry &instance();

    /// 扫描 dir 下所有 DLL 并注册插件，返回成功加载的插件数
    int scanPlugins(const QString &dir);

    /// 注册编译进 core 的内置算法（registry 接管所有权，parent 挂到自身）。
    /// id 冲突或名称/类别为空时拒绝注册并返回 false。
    bool registerBuiltIn(ISfmPlugin *plugin);

    QList<PluginInfo> plugins() const;
    QList<PluginInfo> byCategory(const QString &category) const;

    /// 按插件 id（= name()）获取插件实例；不存在返回 nullptr
    ISfmPlugin *plugin(const QString &id) const;
    /// 若插件实现了 IDescriptorPlugin 则返回，否则返回 nullptr
    IDescriptorPlugin *descriptorPlugin(const QString &id) const;
    IMatcherPlugin *matcherPlugin(const QString &id) const;

    /// id 是否已被占用（用于注册期唯一性校验）
    bool isIdTaken(const QString &id) const;

private:
    PluginRegistry();
    ~PluginRegistry() override;
    Q_DISABLE_COPY(PluginRegistry)

    QList<QPluginLoader *> m_loaders;      ///< 持有 loader 生命周期
    QMap<QString, ISfmPlugin *> m_plugins; ///< id -> 插件实例
    QMap<QString, PluginInfo> m_pluginInfo;
    QSet<QString> m_ids;
};

#endif // PLUGINREGISTRY_H