# SFM 3.0 顶层工程（qmake subdirs）
TEMPLATE = subdirs

SUBDIRS = \
    src/core \
    src/plugins \
    src/app \
    src/tests

# 显式声明构建依赖顺序
src/plugins.depends = src/core
src/app.depends     = src/core
src/tests.depends   = src/core src/app

OTHER_FILES += \
    AGENTS.md \
    重构设计文档.md \
    插件开发规范.md \
    common.pri