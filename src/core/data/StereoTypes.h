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

#endif // STEREOTYPES_H