#ifndef SFM_FRAME_COORDINATE_H
#define SFM_FRAME_COORDINATE_H

/**
 * @file Coordinate.h
 * @brief 算法坐标系 ↔ 显示坐标系的唯一转换点（ADR-005）
 *
 * OpenCV 算法系：右手系，z-forward，y-down。
 * 显示系：绕 Y 轴旋转 180°（x、z 取反，y 保持），det=+1，不产生镜像。
 *
 * 全仓库的 x/z 翻转必须经本头文件；禁止在 pipeline / UI 中手写取反。
 */

#include <cmath>

#include <opencv2/core.hpp>

namespace sfm {
namespace frame {

/// 三元组（避免 core 依赖 Qt / PCL）
struct Vec3f {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
};

/// OpenCV 点 → 显示系
inline Vec3f toDisplay(double x, double y, double z) noexcept
{
    return {static_cast<float>(-x), static_cast<float>(y), static_cast<float>(-z)};
}

/// 3×1 CV_64F 列向量 → 显示系（相机光心 / 世界点）
inline Vec3f toDisplay(const cv::Mat &v) noexcept
{
    return toDisplay(v.at<double>(0), v.at<double>(1), v.at<double>(2));
}

/// 就地：x、z 取反（适用于 pcl::PointXYZRGB 等带 .x/.y/.z 的点）
template <typename PointT>
inline void toDisplayInPlace(PointT &p) noexcept
{
    p.x = -p.x;
    p.z = -p.z;
}

/// 就地转换整朵点云（需提供 begin/end 可遍历的 points 容器）
template <typename CloudT>
inline void toDisplayCloudInPlace(CloudT &cloud)
{
    for (auto &p : cloud.points) {
        toDisplayInPlace(p);
    }
}

/// 显示系 → 算法系（与 toDisplay 互逆，用于 v1 PLY 迁移等）
inline Vec3f toAlgorithm(double x, double y, double z) noexcept
{
    return toDisplay(x, y, z); // 同一变换两次即还原；单次即互逆
}

template <typename PointT>
inline void toAlgorithmInPlace(PointT &p) noexcept
{
    toDisplayInPlace(p);
}

template <typename CloudT>
inline void toAlgorithmCloudInPlace(CloudT &cloud)
{
    for (auto &p : cloud.points) {
        toAlgorithmInPlace(p);
    }
}

} // namespace frame
} // namespace sfm

#endif // SFM_FRAME_COORDINATE_H
