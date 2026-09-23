# bruteforce_matcher.pro —— BruteForce 匹配器插件
TEMPLATE = lib
CONFIG  += plugin
TARGET   = bruteforce_matcher

# 部署到 app 可执行目录下的 plugins/，供主程序扫描加载
DESTDIR = $$OUT_PWD/../../app/plugins

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