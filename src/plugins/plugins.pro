# plugins.pro —— 聚合所有算法插件
# OpenCV 自带算法（SIFT/SURF/KAZE 描述符、FLANN/BruteForce/NearestNeighbor 匹配器）
# 已内置进 core（见 core/builtin/），插件模式只留给扩展算法与用户自定义算法。
# 官方模板插件（sample_descriptor / sample_matcher）位于仓库根 templates/plugins/，
# 仅作为 SFM3-PluginSDK 的参考实现分发，不参与软件构建（见 tools/make_plugin_sdk.ps1）。
TEMPLATE = subdirs

SUBDIRS = \
    xfeat_descriptor \
    superpoint_descriptor
