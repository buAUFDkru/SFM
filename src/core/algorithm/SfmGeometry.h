#ifndef SFMGEOMETRY_H
#define SFMGEOMETRY_H

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "data/SfmTypes.h"

/**
 * @brief 稀疏重建共用的几何工具函数
 *
 * 相机模型约定（与 Triangulate / ObservationManager / IncrementalTriangulator 一致）：
 *   P_cam = R * P_world + t
 *   C_world = -Rᵀ · t          （相机光心在世界坐标中的位置）
 */
namespace SfmGeom {

/// 向量中位数（为空返回 0）
inline double medianOf(const std::vector<double> &v)
{
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    const size_t mid = s.size() / 2;
    if (s.size() % 2 == 0) return (s[mid - 1] + s[mid]) * 0.5;
    return s[mid];
}

/// 相机光心在世界坐标系中的位置：C = -Rᵀ·t
inline cv::Vec3d cameraCenter(const cv::Mat &R, const cv::Mat &t)
{
    cv::Mat C = -R.t() * t;
    return cv::Vec3d(C.at<double>(0), C.at<double>(1), C.at<double>(2));
}

/// 世界点 → 相机坐标
inline cv::Vec3d worldToCamera(const cv::Vec3d &Pw, const cv::Mat &R, const cv::Mat &t)
{
    cv::Mat PwMat = (cv::Mat_<double>(3, 1) << Pw[0], Pw[1], Pw[2]);
    cv::Mat Pc = R * PwMat + t;
    return cv::Vec3d(Pc.at<double>(0), Pc.at<double>(1), Pc.at<double>(2));
}

/**
 * @brief 重投影误差（像素）
 * @return 误差值；点在相机后方（深度 ≤ 0）时返回 -1
 */
inline double reprojectionError(const cv::Vec3d &Pw,
                                const cv::Mat &R, const cv::Mat &t,
                                const cv::Mat &K,
                                const cv::Point2d &obs)
{
    if (K.empty()) return -1.0;

    const cv::Vec3d Pc = worldToCamera(Pw, R, t);
    if (Pc[2] <= 0.0) return -1.0;

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    const double u = fx * Pc[0] / Pc[2] + cx;
    const double v = fy * Pc[1] / Pc[2] + cy;

    return std::hypot(u - obs.x, v - obs.y);
}

/// 归一化相机坐标（去内参）：(u,v) → ((u-cx)/fx, (v-cy)/fy)
inline cv::Point2d toNormalized(const cv::Point2d &p, const cv::Mat &K)
{
    return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                       (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

/**
 * @brief 三角化角度：两条视线（光心 → 三维点）之间的夹角（度）
 *
 * 角度过小说明基线相对深度太短，深度估计不可靠（COLMAP min_angle = 1.5°）。
 */
inline double triangulationAngleDeg(const cv::Vec3d &Pw,
                                    const cv::Vec3d &C1, const cv::Vec3d &C2)
{
    cv::Vec3d r1 = Pw - C1;
    cv::Vec3d r2 = Pw - C2;
    const double n1 = cv::norm(r1);
    const double n2 = cv::norm(r2);
    if (n1 < 1e-12 || n2 < 1e-12) return 0.0;

    double cosA = (r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]) / (n1 * n2);
    cosA = std::max(-1.0, std::min(1.0, cosA));
    return std::acos(cosA) * 180.0 / CV_PI;
}

/**
 * @brief 角度重投影误差（度）—— 与像素阈值无关的残差度量
 *
 * 比较「观测方向」与「三维点反投影方向」的夹角。
 * COLMAP 在三角化新建点时使用角度残差（create_max_angle_error = 2°），
 * 因为它不受焦距误差影响，比像素残差更适合尚未标定好的早期重建。
 *
 * @return 夹角（度）；点在相机后方时返回 DBL_MAX
 */
inline double angularErrorDeg(const cv::Point2d &obs, const cv::Vec3d &Pw,
                              const cv::Mat &R, const cv::Mat &t,
                              const cv::Mat &K)
{
    if (K.empty()) return 1e30;

    const cv::Vec3d Pc = worldToCamera(Pw, R, t);
    if (Pc[2] <= 0.0) return 1e30;

    const cv::Point2d nObs = toNormalized(obs, K);
    const double nObsLen = std::hypot(nObs.x, nObs.y, 1.0);
    const double nPtLen = cv::norm(Pc);
    if (nObsLen < 1e-12 || nPtLen < 1e-12) return 1e30;

    const double dot = (nObs.x * Pc[0] + nObs.y * Pc[1] + 1.0 * Pc[2]) / (nObsLen * nPtLen);
    return std::acos(std::max(-1.0, std::min(1.0, dot))) * 180.0 / CV_PI;
}

} // namespace SfmGeom

#endif // SFMGEOMETRY_H
