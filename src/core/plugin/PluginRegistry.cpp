#include "PluginRegistry.h"

#include <QDebug>
#include <QDir>
#include <QSet>

#include "IDescriptorPlugin.h"
#include "IMatcherPlugin.h"
#include "ISfmPlugin.h"

PluginRegistry &PluginRegistry::instance()
{
    static PluginRegistry reg;
    return reg;
}

PluginRegistry::PluginRegistry() = default;

PluginRegistry::~PluginRegistry()
{
    // loader 以 this 为父对象，随注册表销毁自动 unload
}

int PluginRegistry::scanPlugins(const QString &dir)
{
    QDir d(dir);
    if (!d.exists()) {
        qWarning() << "[PluginRegistry] 目录不存在:" << dir;
        return 0;
    }

    const auto files = d.entryList({"*.dll"}, QDir::Files, QDir::Name);
    int loaded = 0;

    for (const QString &fileName : files) {
        const QString path = d.absoluteFilePath(fileName);
        auto *loader = new QPluginLoader(path, this);

        if (!loader->load()) {
            qWarning() << "[PluginRegistry] 加载失败:" << path << loader->errorString();
            loader->deleteLater();
            continue;
        }

        QObject *obj = loader->instance();
        if (!obj) {
            qWarning() << "[PluginRegistry] 实例化失败:" << path;
            loader->deleteLater();
            continue;
        }

        // 插件只注册了具体接口(IDescriptorPlugin/IMatcherPlugin)的 IID，
        // 基接口 ISfmPlugin 无法被 qobject_cast 识别，因此先 cast 具体接口再用静态聚合。
        auto *descriptor = qobject_cast<IDescriptorPlugin *>(obj);
        auto *matcher    = qobject_cast<IMatcherPlugin *>(obj);
        auto *plugin = descriptor ? static_cast<ISfmPlugin *>(descriptor)
                                  : static_cast<ISfmPlugin *>(matcher);
        if (!plugin) {
            qWarning() << "[PluginRegistry] 非 ISfmPlugin:" << path;
            loader->deleteLater();
            continue;
        }

        const QString name = plugin->name();
        const QString category = plugin->category();
        if (name.isEmpty() || category.isEmpty()) {
            qWarning() << "[PluginRegistry] 名称/类别为空:" << path;
            loader->deleteLater();
            continue;
        }
        if (m_ids.contains(name)) {
            qWarning() << "[PluginRegistry] 插件 id 重复，跳过:" << name << path;
            loader->deleteLater();
            continue;
        }

        PluginInfo info;
        info.id = name;
        info.name = name;
        info.category = category;
        info.version = plugin->version();
        info.description = plugin->description();
        info.filePath = path;
        if (qobject_cast<IDescriptorPlugin *>(plugin))
            info.interfaceType |= PluginInfo::InterfaceDescriptor;
        if (qobject_cast<IMatcherPlugin *>(plugin))
            info.interfaceType |= PluginInfo::InterfaceMatcher;

        m_loaders.append(loader);
        m_plugins.insert(name, plugin);
        m_pluginInfo.insert(name, info);
        m_ids.insert(name);
        ++loaded;

        qInfo() << "[PluginRegistry] 已加载插件:" << name << "(" << category << ") " << path;
    }

    qInfo() << "[PluginRegistry] 扫描完成:" << loaded << "/" << files.size() << "个插件加载成功";
    return loaded;
}

QList<PluginInfo> PluginRegistry::plugins() const
{
    return m_pluginInfo.values();
}

QList<PluginInfo> PluginRegistry::byCategory(const QString &category) const
{
    QList<PluginInfo> result;
    for (const PluginInfo &info : m_pluginInfo) {
        if (info.category == category)
            result.append(info);
    }
    return result;
}

ISfmPlugin *PluginRegistry::plugin(const QString &id) const
{
    return m_plugins.value(id, nullptr);
}

IDescriptorPlugin *PluginRegistry::descriptorPlugin(const QString &id) const
{
    return qobject_cast<IDescriptorPlugin *>(plugin(id));
}

IMatcherPlugin *PluginRegistry::matcherPlugin(const QString &id) const
{
    return qobject_cast<IMatcherPlugin *>(plugin(id));
}

bool PluginRegistry::isIdTaken(const QString &id) const
{
    return m_ids.contains(id);
}