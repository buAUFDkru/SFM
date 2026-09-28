# sample_descriptor.pro —— P1 示例描述符插件（确定性伪特征，用于验证加载与并发一致性）
TEMPLATE = lib
CONFIG  += plugin
TARGET   = sample_descriptor

# 部署到 app 可执行文件同级的 plugins/ 子目录（exe 在 release/ 或 debug/ 下），
# 与主程序默认扫描路径 applicationDirPath()/plugins 保持一致
CONFIG(debug, debug|release) {
    DESTDIR = $$OUT_PWD/../../app/debug/plugins
} else {
    DESTDIR = $$OUT_PWD/../../app/release/plugins
}

# 导入库(.lib)/导出表(.exp)重定向到本插件构建目录：
# 插件由 PluginRegistry 运行时动态加载，无需隐式链接导入库，
# 使部署目录 plugins/ 只保留运行时 DLL
win32: QMAKE_LFLAGS += /IMPLIB:$$OUT_PWD/$${TARGET}.lib

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