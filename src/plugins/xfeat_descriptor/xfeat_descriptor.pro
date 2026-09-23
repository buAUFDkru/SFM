# xfeat_descriptor.pro —— XFeat 描述符插件（基于 ONNX Runtime 推理）
TEMPLATE = lib
CONFIG  += plugin
TARGET   = xfeat_descriptor

# 部署到 app 可执行目录下的 plugins/，供主程序扫描加载
DESTDIR = $$OUT_PWD/../../app/plugins

include(../../../common.pri)

INCLUDEPATH += ../../core
LIBS += -L$$OUT_PWD/../../core -lsfmcore
# ONNX Runtime 仅本插件需要，单独链接（common.pri 只提供 include 路径）
LIBS += -L$$ONNXRUNTIME_DIR/lib -lonnxruntime

HEADERS += \
    XFeatDescriptorPlugin.h \
    XFeatDescriptorWorker.h

SOURCES += \
    XFeatDescriptorPlugin.cpp \
    XFeatDescriptorWorker.cpp

DISTFILES += plugin.json