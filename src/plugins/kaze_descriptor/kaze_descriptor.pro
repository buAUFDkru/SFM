# kaze_descriptor.pro —— KAZE 描述符插件
TEMPLATE = lib
CONFIG  += plugin
TARGET   = kaze_descriptor

# 部署到 app 可执行目录下的 plugins/，供主程序扫描加载
DESTDIR = $$OUT_PWD/../../app/plugins

include(../../../common.pri)

INCLUDEPATH += ../../core
LIBS += -L$$OUT_PWD/../../core -lsfmcore

HEADERS += \
    KazeDescriptorPlugin.h \
    KazeDescriptorWorker.h

SOURCES += \
    KazeDescriptorPlugin.cpp \
    KazeDescriptorWorker.cpp

DISTFILES += plugin.json