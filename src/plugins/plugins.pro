# plugins.pro —— 聚合所有算法插件
# OpenCV 自带算法（SIFT/SURF/KAZE 描述符、FLANN/BruteForce 匹配器）已内置进 core
# （见 core/builtin/），插件模式只留给扩展算法与用户自定义算法。
TEMPLATE = subdirs

SUBDIRS = \
    sample_descriptor \
    sample_matcher \
    xfeat_descriptor \
    superpoint_descriptor
