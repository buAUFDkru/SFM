# core.pro —— 核心静态库（插件接口 + 插件注册表 + 共享算法）
TEMPLATE = lib
CONFIG  += staticlib
TARGET   = sfmcore
DESTDIR  = $$OUT_PWD

include(../../common.pri)

INCLUDEPATH += . plugin data algorithm

HEADERS += \
    plugin/ParamSchema.h \
    plugin/PluginTypes.h \
    plugin/ISfmPlugin.h \
    plugin/IDescriptorPlugin.h \
    plugin/IMatcherPlugin.h \
    plugin/PluginInfo.h \
    plugin/PluginRegistry.h \
    data/SfmTypes.h \
    algorithm/ThreadPool.h \
    algorithm/SfmLogger.h \
    algorithm/Initialization.h

SOURCES += \
    plugin/PluginRegistry.cpp \
    algorithm/ThreadPool.cpp \
    algorithm/SfmLogger.cpp \
    algorithm/Initialization.cpp