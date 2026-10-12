# SFM3 Plugin SDK

为 SFM Reconstructor (≥ v3.0) 开发算法插件所需的全部宿主产物。
配合 `doc/插件开发规范.md` 使用；规范第 9 章列出必须遵守的 ABI 红线。

## 目录结构

    include/sfm3/plugin/   宿主接口头文件（ISfmPlugin / IDescriptorPlugin / IMatcherPlugin /
                           ParamSchema / PluginTypes / PluginInfo）——二进制契约的声明部分
    lib/sfmcore.lib        接口 moc meta-object 链接库（规范红线 #2：禁止自行重编译接口头）
    pri/common.pri         与宿主一致的编译配置与第三方路径（OpenCV 460 / onnxruntime 1.26.0）
    templates/             官方模板插件：sample_descriptor（描述符）、sample_matcher（匹配器）
    doc/插件开发规范.md     插件开发规范（接口三件套、线程模型、ABI 红线、测试要求）
    doc/插件开发流程.md     插件开发流程（从零到可加载 DLL 的分步操作手册）

## 快速开始

1. 复制 `templates/sample_descriptor` 到你的工作目录，按规范第 3/4 节实现三件套
   （工厂 / Worker / plugin.json）。
2. `.pro` 中把宿主内路径改为 SDK 相对路径（两处差异）：

        # 宿主工程内写法                        →  SDK 离线开发写法
        INCLUDEPATH += ../../core               →  INCLUDEPATH += <SDK>/include/sfm3/plugin
        LIBS += -L$$OUT_PWD/../../core -lsfmcore →  LIBS += -L<SDK>/lib -lsfmcore

   其中 `<SDK>` 建议用相对路径变量引入，例如：

        SDK_ROOT = $$PWD/../SFM3-PluginSDK
        INCLUDEPATH += $$SDK_ROOT/include/sfm3/plugin
        LIBS += -L$$SDK_ROOT/lib -lsfmcore

   `common.pri` 同样从 `$$SDK_ROOT/pri` 引入。
3. 部署：构建出的 DLL 与其运行时依赖放到目标机 `plugins/` 目录，须与宿主版本匹配
   （同 Qt 6.10.1 msvc2022_64、MSVC 2022 /MD release、OpenCV 460 world，规范红线 #4~#7）。
4. 验证：把 DLL 放入宿主 `plugins/` 目录后启动宿主，扫描日志出现
   `已加载插件: <name> (<category>)` 即加载成功；`非 ISfmPlugin` = IID/`Q_INTERFACES` 问题；
   `加载失败` = 运行时依赖问题（规范 §9.4）。

## 版本匹配

本 SDK 对应的宿主构建以 `lib/sfmcore.lib` 为准；宿主升级后必须使用配套版本重新发布
的 SDK 重新编译插件（IID 与跨 DLL 结构体布局是版本化契约，规范 §9.2）。
