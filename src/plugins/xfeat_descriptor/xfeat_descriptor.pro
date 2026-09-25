# xfeat_descriptor.pro —— XFeat 描述符插件（基于 ONNX Runtime 推理）
TEMPLATE = lib
CONFIG  += plugin
TARGET   = xfeat_descriptor

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

# 构建后自动把 onnxruntime 运行时 DLL 部署到插件输出目录（plugins/），
# 保证 PluginRegistry 预加载能找到它，避免 Win32 193 加载失败。
win32 {
    QMAKE_POST_LINK += $$QMAKE_COPY $$shell_path($$ONNXRUNTIME_DIR/lib/onnxruntime.dll) $$shell_path($$DESTDIR) $$escape_expand(\n\t)
    QMAKE_POST_LINK += $$QMAKE_COPY $$shell_path($$ONNXRUNTIME_DIR/lib/onnxruntime_providers_shared.dll) $$shell_path($$DESTDIR)
}