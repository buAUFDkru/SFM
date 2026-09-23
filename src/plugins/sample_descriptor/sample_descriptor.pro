# sample_descriptor.pro —— P1 示例描述符插件（确定性伪特征，用于验证加载与并发一致性）
TEMPLATE = lib
CONFIG  += plugin
TARGET   = sample_descriptor

# 部署到 app 可执行目录下的 plugins/，供主程序扫描加载
DESTDIR = $$OUT_PWD/../../app/plugins

include(../../../common.pri)

INCLUDEPATH += ../../core
# 链接 core 静态库：接口类(ISfmPlugin/IDescriptorPlugin)的 moc meta-object 位于其中
LIBS += -L$$OUT_PWD/../../core -lsfmcore

HEADERS += \
    SampleDescriptorPlugin.h \
    SampleDescriptorWorker.h

SOURCES += \
    SampleDescriptorPlugin.cpp \
    SampleDescriptorWorker.cpp

DISTFILES += plugin.json