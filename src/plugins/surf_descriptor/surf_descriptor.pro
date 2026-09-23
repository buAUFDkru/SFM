# surf_descriptor.pro —— SURF 描述符插件
TEMPLATE = lib
CONFIG  += plugin
TARGET   = surf_descriptor

# 部署到 app 可执行目录下的 plugins/，供主程序扫描加载
DESTDIR = $$OUT_PWD/../../app/plugins

include(../../../common.pri)

INCLUDEPATH += ../../core
LIBS += -L$$OUT_PWD/../../core -lsfmcore

HEADERS += \
    SurfDescriptorPlugin.h \
    SurfDescriptorWorker.h

SOURCES += \
    SurfDescriptorPlugin.cpp \
    SurfDescriptorWorker.cpp

DISTFILES += plugin.json