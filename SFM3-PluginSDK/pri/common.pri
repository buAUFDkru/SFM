# common.pri —— 共享的第三方依赖与通用编译配置
# 所有 *_ROOT 均可通过同名环境变量覆盖，避免写死绝对路径。
# 用法：子项目 include(../../common.pri)

# ── 第三方根目录（环境变量优先，回退默认安装路径） ──
isEmpty(OPENCV_DIR):     OPENCV_DIR     = "D:/package/C++Package/opencv460/build/install"
isEmpty(ONNXRUNTIME_DIR):ONNXRUNTIME_DIR= "D:/package/C++Package/onnxruntime-win-x64-1.26.0"
isEmpty(PCL_ROOT):       PCL_ROOT       = "D:/package/pcl/PCL 1.15.1"
isEmpty(CERES_ROOT):     CERES_ROOT     = "D:/package/C++Package/ceres2.2"
isEmpty(OPEN3D_ROOT):    OPEN3D_ROOT    = "D:/package/C++Package/open3d-devel-windows-amd64-0.18.0"

CONFIG += c++17
DEFINES += NOMINMAX WIN32_LEAN_AND_MEAN

# ── 算法链依赖（PCL / Ceres / Open3D）────
# 仅当使用方工程设置 SFM3_ALG_DEPS=1 后生效（core 重建管线段 / app / 相关测试）。
# PCL FLANN 必须在 OpenCV 之前 include，否则 flann 命名空间会被 cvflann 覆盖。
equals(SFM3_ALG_DEPS, 1) {
    DEFINES += EIGEN_MAX_ALIGN_BYTES=32
    DEFINES += GLOG_USE_GLOG_EXPORT GLOG_STATIC_DEFINE
    DEFINES += _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING

    INCLUDEPATH += $$PCL_ROOT/3rdParty/FLANN/include
    INCLUDEPATH += $$PCL_ROOT/include/pcl-1.15
    INCLUDEPATH += $$PCL_ROOT/3rdParty/Boost/include/boost-1_87
    INCLUDEPATH += $$PCL_ROOT/3rdParty/Eigen3/include/eigen3
    INCLUDEPATH += $$CERES_ROOT/CeresLib/include
    INCLUDEPATH += $$CERES_ROOT/glogLib/include
    INCLUDEPATH += $$OPEN3D_ROOT/include
    INCLUDEPATH += $$OPEN3D_ROOT/include/open3d/3rdparty

    win32 {
        CONFIG(debug, debug|release) {
            LIBS += -l$$PCL_ROOT/lib/pcl_commond
            LIBS += -l$$PCL_ROOT/lib/pcl_filtersd
            LIBS += -l$$PCL_ROOT/lib/pcl_kdtreed
            LIBS += -l$$PCL_ROOT/lib/pcl_searchd
            LIBS += -l$$PCL_ROOT/lib/pcl_surfaced
            LIBS += -l$$PCL_ROOT/lib/pcl_featuresd
            LIBS += -l$$PCL_ROOT/lib/pcl_octreed
            LIBS += -l$$PCL_ROOT/lib/pcl_iod
            LIBS += -l$$PCL_ROOT/3rdParty/FLANN/lib/flann_s-gd
            LIBS += -l$$CERES_ROOT/CeresLib/lib/ceres-debug
            LIBS += -l$$CERES_ROOT/glogLib/lib/glogd
        } else {
            LIBS += -l$$PCL_ROOT/lib/pcl_common
            LIBS += -l$$PCL_ROOT/lib/pcl_filters
            LIBS += -l$$PCL_ROOT/lib/pcl_kdtree
            LIBS += -l$$PCL_ROOT/lib/pcl_search
            LIBS += -l$$PCL_ROOT/lib/pcl_surface
            LIBS += -l$$PCL_ROOT/lib/pcl_features
            LIBS += -l$$PCL_ROOT/lib/pcl_octree
            LIBS += -l$$PCL_ROOT/lib/pcl_io
            LIBS += -l$$PCL_ROOT/3rdParty/FLANN/lib/flann_s
            LIBS += -l$$CERES_ROOT/CeresLib/lib/ceres
            LIBS += -l$$CERES_ROOT/glogLib/lib/glog
        }
        LIBS += -L$$OPEN3D_ROOT/lib -lOpen3D
        # 注：Open3D.dll 的部署交由 tools/ 部署脚本或运行时 PATH（$$OPEN3D_ROOT/bin）
        # 完成；不在此处 QMAKE_POST_LINK 拷贝（大文件首次落盘易被安全软件锁死导致退出码1）。
    }
}

# ── OpenCV（world 构建，含 contrib 的 xfeatures2d 等模块符号） ──
INCLUDEPATH += $$OPENCV_DIR/include

win32 {
    CONFIG(release, debug|release) {
        LIBS += -L$$OPENCV_DIR/lib -lopencv_world460
    } else {
        LIBS += -L$$OPENCV_DIR/lib -lopencv_world460d
    }
}

# ── ONNX Runtime（XFeat 推理；仅 include 路径全局共享，-lonnxruntime 由 xfeat 插件 .pro 单独链接，
#    避免所有插件/主程序都带上 onnxruntime.dll 运行时依赖） ──
INCLUDEPATH += $$ONNXRUNTIME_DIR/include