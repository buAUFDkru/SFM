# bruteforce_matcher.pro —— BruteForce 匹配器插件
TEMPLATE = lib
CONFIG  += plugin
TARGET   = bruteforce_matcher

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

HEADERS += \
    BruteForceMatcherPlugin.h \
    BruteForceMatcherWorker.h

SOURCES += \
    BruteForceMatcherPlugin.cpp \
    BruteForceMatcherWorker.cpp

DISTFILES += plugin.json