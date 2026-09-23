#ifndef STEREOTYPES_H
#define STEREOTYPES_H

#include <QString>

#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "data/SfmTypes.h"

/**
 * @brief 立体匹配对信息
 *
 * 存储一对图像进行立体匹配所需的全部数据，包括两张图像的
 * 内参矩阵、外参位姿（相机到世界变换 R, t）和原始BGR图像。
 *
 * 其中 imageId1 为参考图像（左图），imageId2 为匹配图像（右图）。
 * 位姿约定：P_cam = R * P_world + t，即 R, t 描述的是世界坐标系到相机坐标系的变换。
 */
struct StereoPairInfo
{
    int imageId1;              ///< 参考图像ID（左图）
    int imageId2;              ///< 匹配图像ID（右图）
    cv::Mat K1, K2;            ///< 内参矩阵 (3×3, CV_64FC1)
    cv::Mat R1, t1;            ///< 图像1的位姿：R为旋转矩阵(3×3)，t为平移向量(3×1)，相机→世界
    cv::Mat R2, t2;            ///< 图像2的位姿：R为旋转矩阵(3×3)，t为平移向量(3×1)，相机→世界
    cv::Mat image1, image2;    ///< 原始BGR图像 (CV_8UC3)
    QString imagePath1;        ///< 图像1文件路径（流式加载时使用）
    QString imagePath2;        ///< 图像2文件路径（流式加载时使用）
};

/**
 * @brief 立体匹配结果
 *
 * 存储一对图像的立体匹配输出，包括校正后的图像、视差图、深度图等。
 * 校正后的图像极线水平对齐，可直接用于SGBM匹配。
 *
 * 深度图公式：depth = fx * baseline / disparity
 * 其中 fx 为校正后内参的水平焦距，baseline 为两相机中心之间的基线长度。
 */
struct StereoPairResult
{
    int imageId1;              ///< 参考图像ID
    int imageId2;              ///< 匹配图像ID
    cv::Mat rectifiedImage1;   ///< 校正后的左图（极线水平对齐，CV_8UC3）
    cv::Mat rectifiedImage2;   ///< 校正后的右图（极线水平对齐，CV_8UC3）
    cv::Mat disparity;         ///< 滤波后的视差图 (CV_32FC1, 单位: 像素)
    cv::Mat depth;             ///< 深度图 (CV_32FC1, 单位: 与基线一致，通常为米)
    cv::Mat confidence;        ///< 置信度图 (CV_32FC1, 0~1, WLS滤波的置信度)
    cv::Mat Q;                 ///< 重投影矩阵 (4×4, CV_64FC1), 用于视差到3D的映射
    cv::Mat P1, P2;            ///< 校正后的投影矩阵 (3×4, CV_64FC1), 左右相机在校正后坐标系下的投影
    double baseline;           ///< 基线长度（两相机中心之间的距离，单位: 米）
    cv::Mat K_rect;            ///< 校正后的公共内参 (3×3, CV_64FC1), 从P1左上3×3提取
    cv::Mat R1_rect;           ///< 左图校正旋转矩阵 (3×3, CV_64FC1), 原始相机→校正后相机
    cv::Mat R2_rect;           ///< 右图校正旋转矩阵 (3×3, CV_64FC1), 原始相机→校正后相机
};

/**
 * @brief 密集立体匹配配置参数
 *
 * 控制立体匹配管线的各个阶段参数，包括选图策略、SGBM匹配、WLS滤波和深度图生成。
 */
struct DenseStereoConfig
{
    // ── 选图策略 ──
    int targetPairsPerView = 3;      ///< 每张视图选择的最优匹配对数（贪心策略上限）
    double minOverlapRatio = 0.05;   ///< 最小重叠率阈值（Jaccard系数），低于此值的图像对被排除
    double minAngleDeg = 3.0;        ///< 最小视角差（度），太小则基线过短、深度精度差
    double maxAngleDeg = 60.0;       ///< 最大视角差（度），太大则遮挡严重、匹配困难

    // ── SGBM 参数 ──
    int minDisparity = 0;            ///< 最小视差（像素），通常为0（场景在相机前方）
    int numDisparities = 160;        ///< 视差搜索范围（像素，必须为16的倍数），越大计算越慢
    int blockSize = 3;               ///< 匹配块大小（奇数, 3~11），3为默认推荐值
    int preFilterCap = 31;           ///< 预处理截断值（1~63），控制Sobel滤波的输出范围
    int uniquenessRatio = 10;        ///< 唯一性比率（0~100），越大越严格，次优匹配需比最优差此百分比
    int speckleWindowSize = 100;     ///< 斑点滤波窗口大小（像素），小于此的连通区域被视为噪点
    int speckleRange = 1;            ///< 斑点滤波范围（视差单位），同一区域内最大视差变化
    int disp12MaxDiff = 1;           ///< 左右一致性检查阈值（像素），-1表示不检查
    bool useHHMode = true;           ///< 使用HH模式（全分辨率动态规划），比SGBM更精确但稍慢
    int P1Scale = 8;                 ///< P1惩罚系数缩放: P1 = P1Scale * channels * blockSize^2
    int P2Scale = 32;                ///< P2惩罚系数缩放: P2 = P2Scale * channels * blockSize^2（P2 > P1）

    // ── WLS 滤波参数 ──
    bool useWLSFilter = true;        ///< 是否使用WLS（加权最小二乘）后滤波，显著提升视差图边缘质量
    double wlsLambda = 8000.0;       ///< WLS平滑度参数，越大越平滑（推荐: 1000~10000）
    double wlsSigmaColor = 1.5;      ///< WLS边缘保持参数，越小越敏感于颜色变化（推荐: 0.5~3.0）

    // ── 深度图参数 ──
    double minDepth = 0.1;           ///< 最小深度（米），低于此值的深度被置零
    double maxDepth = 100.0;         ///< 最大深度（米），超过此值的深度被置零

    // ── 视差自动估计 ──
    bool autoDisparityRange = true;  ///< 是否根据匹配点自动估计视差范围（推荐开启）

    // ── 调试输出 ──
    std::string saveRectifiedDir;    ///< 保存校正后图像的目录（空则不保存）
};

#endif // STEREOTYPES_H