# common.pri —— 共享的第三方依赖与通用编译配置
# 所有 *_ROOT 均可通过同名环境变量覆盖，避免写死绝对路径。
# 用法：子项目 include(../../common.pri)

# ── 第三方根目录（环境变量优先，回退默认安装路径） ──
isEmpty(OPENCV_DIR): OPENCV_DIR = "D:/package/C++Package/opencv460/build/install"

CONFIG += c++17
DEFINES += NOMINMAX WIN32_LEAN_AND_MEAN

# ── OpenCV ──
INCLUDEPATH += $$OPENCV_DIR/include

win32 {
    CONFIG(release, debug|release) {
        LIBS += -L$$OPENCV_DIR/lib -lopencv_world460
    } else {
        LIBS += -L$$OPENCV_DIR/lib -lopencv_world460d
    }
}