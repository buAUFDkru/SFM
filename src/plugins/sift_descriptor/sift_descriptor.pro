# sift_descriptor.pro —— SIFT 描述符插件
TEMPLATE = lib
CONFIG  += plugin
TARGET   = sift_descriptor

# 部署到 app 可执行文件同级的 plugins/ 子目录（exe 在 release/ 或 debug/ 下），
# 与主程序默认扫描路径 applicationDirPath()/plugins 保持一致
CONFIG(debug, debug|release) {
    DESTDIR = $$OUT_PWD/../../app/debug/plugins
} else {
    DESTDIR = $$OUT_PWD/../../app/release/plugins
}

include(../../../common.pri)

INCLUDEPATH += ../../core
LIBS += -L$$OUT_PWD/../../core -lsfmcore

HEADERS += \
    SiftDescriptorPlugin.h \
    SiftDescriptorWorker.h

SOURCES += \
    SiftDescriptorPlugin.cpp \
    SiftDescriptorWorker.cpp

DISTFILES += plugin.json