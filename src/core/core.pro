# core.pro —— 核心静态库（插件接口 + 插件注册表）
TEMPLATE = lib
CONFIG  += staticlib
TARGET   = sfmcore
DESTDIR  = $$OUT_PWD

include(../../common.pri)

INCLUDEPATH += . plugin

HEADERS += \
    plugin/ParamSchema.h \
    plugin/PluginTypes.h \
    plugin/ISfmPlugin.h \
    plugin/IDescriptorPlugin.h \
    plugin/IMatcherPlugin.h \
    plugin/PluginInfo.h \
    plugin/PluginRegistry.h

SOURCES += \
    plugin/PluginRegistry.cpp