#ifndef PLUGININFO_H
#define PLUGININFO_H

#include <QString>

/// 运行时插件描述（注册表登记用，与 DLL 生命周期无关）
struct PluginInfo
{
    QString id;              ///< 全局唯一 = name()
    QString name;
    QString category;        ///< descriptor / matcher / dense
    QString version;
    QString description;
    QString filePath;
    bool builtin = false;    ///< 内置算法（编译进 core，非 DLL 插件）

    // 插件实现的接口类型标记（仅用于展示/过滤）
    static constexpr int InterfaceUnknown    = 0;
    static constexpr int InterfaceDescriptor = 1 << 0;
    static constexpr int InterfaceMatcher    = 1 << 1;
    int interfaceType = InterfaceUnknown;
};

#endif // PLUGININFO_H