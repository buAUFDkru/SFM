#ifndef SFMGEOMETRY_H
#define SFMGEOMETRY_H

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
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

// ─────────────────────────────────────────────────────────────────────────────
//  阶段五 5.2：径向/切向畸变模型（OpenCV 系数顺序 k1,k2,p1,p2,k3）
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief 畸变系数是否有效（空矩阵或全零都视为无畸变）
 *
 * 全零时直接走无畸变快路径：既省掉每次迭代开销，也保证「未标定畸变」与
 * 阶段五之前的行为逐位一致（回归安全）。
 */
inline bool hasDistortion(const cv::Mat &dist)
{
    if (dist.empty()) return false;
    const int n = dist.rows * dist.cols;
    for (int i = 0; i < std::min(n, 5); ++i) {
        if (std::abs(dist.at<double>(i)) > 1e-12) return true;
    }
    return false;
}

/// 读取第 i 个畸变系数（越界返回 0，支持只给 k1,k2 的短向量）
inline double distCoeff(const cv::Mat &dist, int i)
{
    if (dist.empty()) return 0.0;
    const int n = dist.rows * dist.cols;
    return (i < n) ? dist.at<double>(i) : 0.0;
}

/**
 * @brief 归一化坐标 → 畸变后的归一化坐标
 *
 * 采用 OpenCV 的 Brown–Conrady（径向 + 切向）模型：
 *   r² = x² + y²
 *   radial = 1 + k1·r² + k2·r⁴ + k3·r⁶
 *   x_d = x·radial + 2·p1·x·y + p2·(r² + 2x²)
 *   y_d = y·radial + p1·(r² + 2y²) + 2·p2·x·y
 */
inline cv::Point2d distortNormalized(const cv::Point2d &p, const cv::Mat &dist)
{
    if (!hasDistortion(dist)) return p;

    const double k1 = distCoeff(dist, 0);
    const double k2 = distCoeff(dist, 1);
    const double p1 = distCoeff(dist, 2);
    const double p2 = distCoeff(dist, 3);
    const double k3 = distCoeff(dist, 4);

    const double x = p.x;
    const double y = p.y;
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

    return cv::Point2d(x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                       y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);
}

/**
 * @brief 畸变后的归一化坐标 → 归一化坐标（去畸变）
 *
 * 畸变是逐点非线性映射，没有闭式逆。用 OpenCV 同款不动点迭代：
 *   x ← (x_d - 切向项) / radial(x)
 * 畸变梯度 < 1 时收敛；15 次迭代对 k1 在 ±0.5 内的镜头足够（残差 < 1e-9）。
 */
inline cv::Point2d undistortNormalized(const cv::Point2d &pd, const cv::Mat &dist)
{
    if (!hasDistortion(dist)) return pd;

    const double k1 = distCoeff(dist, 0);
    const double k2 = distCoeff(dist, 1);
    const double p1 = distCoeff(dist, 2);
    const double p2 = distCoeff(dist, 3);
    const double k3 = distCoeff(dist, 4);

    double x = pd.x;
    double y = pd.y;
    for (int i = 0; i < 15; ++i) {
        const double r2 = x * x + y * y;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        if (std::abs(radial) < 1e-12) break;
        const double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        const double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        x = (pd.x - dx) / radial;
        y = (pd.y - dy) / radial;
    }
    return cv::Point2d(x, y);
}

/**
 * @brief 重投影误差（像素）
 * @param dist 可选畸变系数（k1,k2,p1,p2,k3）；空/全零时按针孔模型
 * @return 误差值；点在相机后方（深度 ≤ 0）时返回 -1
 */
inline double reprojectionError(const cv::Vec3d &Pw,
                                const cv::Mat &R, const cv::Mat &t,
                                const cv::Mat &K,
                                const cv::Point2d &obs,
                                const cv::Mat &dist = cv::Mat())
{
    if (K.empty()) return -1.0;

    const cv::Vec3d Pc = worldToCamera(Pw, R, t);
    if (Pc[2] <= 0.0) return -1.0;

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    const cv::Point2d xd = distortNormalized(
        cv::Point2d(Pc[0] / Pc[2], Pc[1] / Pc[2]), dist);

    const double u = fx * xd.x + cx;
    const double v = fy * xd.y + cy;

    return std::hypot(u - obs.x, v - obs.y);
}

/// 像素坐标 → 无畸变归一化坐标：(u,v) → undistort(((u-cx)/fx, (v-cy)/fy))
inline cv::Point2d toNormalized(const cv::Point2d &p, const cv::Mat &K,
                                const cv::Mat &dist = cv::Mat())
{
    const cv::Point2d q((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
    return undistortNormalized(q, dist);
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
 * @param dist 可选畸变系数；给了就先把观测去畸变再比方向
 * @return 夹角（度）；点在相机后方时返回 DBL_MAX
 */
inline double angularErrorDeg(const cv::Point2d &obs, const cv::Vec3d &Pw,
                              const cv::Mat &R, const cv::Mat &t,
                              const cv::Mat &K,
                              const cv::Mat &dist = cv::Mat())
{
    if (K.empty()) return 1e30;

    const cv::Vec3d Pc = worldToCamera(Pw, R, t);
    if (Pc[2] <= 0.0) return 1e30;

    const cv::Point2d nObs = toNormalized(obs, K, dist);
    const double nObsLen = std::hypot(nObs.x, nObs.y, 1.0);
    const double nPtLen = cv::norm(Pc);
    if (nObsLen < 1e-12 || nPtLen < 1e-12) return 1e30;

    const double dot = (nObs.x * Pc[0] + nObs.y * Pc[1] + 1.0 * Pc[2]) / (nObsLen * nPtLen);
    return std::acos(std::max(-1.0, std::min(1.0, dot))) * 180.0 / CV_PI;
}

// ─────────────────────────────────────────────────────────────────────────────
//  阶段四：图连通分量 / 多分量重建 / 场景归一化
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief 无向图连通分量（并查集）
 *
 * 用于阶段四的多分量重建：匹配图不连通时，各分量之间没有任何几何联系，
 * 无法在同一个坐标系下重建，只能挑最大分量做主重建。
 */
struct GraphComponents
{
    std::vector<int> compId;   ///< 每个节点的分量编号
    int numComponents = 0;     ///< 分量总数（孤立节点自成一个分量）
    std::vector<int> sizes;    ///< 各分量的节点数（下标 = 分量编号）
    int largest = -1;          ///< 最大分量的编号（并列时取编号最小者）
};

inline GraphComponents connectedComponents(int numNodes,
                                           const std::vector<std::pair<int, int>> &edges)
{
    GraphComponents out;
    if (numNodes <= 0) return out;

    // ── 并查集（迭代 find + 路径压缩）─────────────────────────────────────
    std::vector<int> parent(static_cast<size_t>(numNodes));
    for (int i = 0; i < numNodes; ++i) parent[static_cast<size_t>(i)] = i;

    auto find = [&parent](int x) {
        int root = x;
        while (parent[static_cast<size_t>(root)] != root) {
            root = parent[static_cast<size_t>(root)];
        }
        while (parent[static_cast<size_t>(x)] != root) {
            const int next = parent[static_cast<size_t>(x)];
            parent[static_cast<size_t>(x)] = root;
            x = next;
        }
        return root;
    };

    for (const auto &e : edges) {
        if (e.first < 0 || e.first >= numNodes ||
            e.second < 0 || e.second >= numNodes) continue;
        const int r1 = find(e.first);
        const int r2 = find(e.second);
        if (r1 != r2) parent[static_cast<size_t>(r2)] = r1;
    }

    // ── 收集分量 ──────────────────────────────────────────────────────────
    out.compId.assign(static_cast<size_t>(numNodes), -1);
    std::unordered_map<int, int> rootToId;
    rootToId.reserve(static_cast<size_t>(numNodes));
    for (int i = 0; i < numNodes; ++i) {
        const int root = find(i);
        auto it = rootToId.find(root);
        if (it == rootToId.end()) {
            const int newId = static_cast<int>(out.sizes.size());
            rootToId.emplace(root, newId);
            out.sizes.push_back(1);
            it = rootToId.find(root);
            out.compId[static_cast<size_t>(i)] = newId;
        } else {
            out.compId[static_cast<size_t>(i)] = it->second;
            ++out.sizes[static_cast<size_t>(it->second)];
        }
    }

    out.numComponents = static_cast<int>(out.sizes.size());
    out.largest = 0;
    for (int c = 1; c < out.numComponents; ++c) {
        if (out.sizes[static_cast<size_t>(c)] >
            out.sizes[static_cast<size_t>(out.largest)]) {
            out.largest = c;
        }
    }
    return out;
}

/// keepLargestComponent 的结果统计
struct LargestComponentResult
{
    int numComponents = 1;   ///< 过滤前的分量总数
    int mainSize = 0;        ///< 最大分量的图像数
    int isolatedImages = 0;  ///< 没有任何匹配边的图像数
    size_t removedPairs = 0; ///< 被剔除的匹配对数（不属于最大分量的边）
};

/**
 * @brief 阶段四 4.5：图不连通时只保留最大连通分量内部的匹配边（就地修改）
 *
 * 分量之间没有共视关系，位姿与尺度都无法关联；把小分量的边留在匹配表里
 * 只会稀释初始化候选评分、拉低注册率分母。孤立图像（无任何匹配边）单独计数。
 */
inline LargestComponentResult keepLargestComponent(std::vector<MatchPair> &pairs,
                                                   int numImages)
{
    std::vector<std::pair<int, int>> edges;
    edges.reserve(pairs.size());
    std::vector<char> hasEdge(static_cast<size_t>(std::max(numImages, 0)), 0);
    for (const auto &mp : pairs) {
        edges.emplace_back(mp.imageId1, mp.imageId2);
        if (mp.imageId1 >= 0 && mp.imageId1 < numImages) hasEdge[static_cast<size_t>(mp.imageId1)] = 1;
        if (mp.imageId2 >= 0 && mp.imageId2 < numImages) hasEdge[static_cast<size_t>(mp.imageId2)] = 1;
    }

    LargestComponentResult res;
    for (int i = 0; i < numImages; ++i) {
        if (!hasEdge[static_cast<size_t>(i)]) ++res.isolatedImages;
    }

    const GraphComponents comps = connectedComponents(numImages, edges);
    res.numComponents = comps.numComponents;
    res.mainSize = comps.largest >= 0 ? comps.sizes[static_cast<size_t>(comps.largest)]
                                      : numImages;
    if (comps.numComponents <= 1) return res;

    const int mainId = comps.largest;
    std::vector<MatchPair> kept;
    kept.reserve(pairs.size());
    for (MatchPair &mp : pairs) {
        if (comps.compId[static_cast<size_t>(mp.imageId1)] == mainId &&
            comps.compId[static_cast<size_t>(mp.imageId2)] == mainId) {
            kept.push_back(std::move(mp));
        } else {
            ++res.removedPairs;
        }
    }
    pairs = std::move(kept);
    return res;
}

/**
 * @brief 阶段四 4.6：场景归一化（COLMAP NormalizeReconstruction 的简化版）
 *
 * 相似变换，不改变重投影误差与几何形状：
 * 1. 平移：已注册相机光心的质心移到世界原点
 * 2. 缩放：光心到质心的平均距离归一到 targetExtent
 *
 * 旋转不变（R 不动），平移按 t' = s·t + s·R·c 更新（C = -Rᵀ·t ⇒ t = -R·C）。
 * 未注册相机（R/t 为空）原样跳过。自标定的尺度本是任意的，
 * 归一化只是把任意尺度固定下来，提升下游（密集深度范围估计、可视化）数值稳定。
 */
inline void normalizeScene(std::vector<Point3D> &points,
                           std::vector<CameraPose> &poses,
                           double targetExtent = 10.0)
{
    std::vector<cv::Vec3d> centers;
    centers.reserve(poses.size());
    for (const auto &pose : poses) {
        if (pose.registered && !pose.R.empty() && !pose.t.empty()) {
            centers.push_back(cameraCenter(pose.R, pose.t));
        }
    }
    if (centers.size() < 2 || targetExtent <= 0.0) return;

    cv::Vec3d centroid(0.0, 0.0, 0.0);
    for (const auto &c : centers) centroid += c;
    centroid /= static_cast<double>(centers.size());

    double meanDist = 0.0;
    for (const auto &c : centers) meanDist += cv::norm(c - centroid);
    meanDist /= static_cast<double>(centers.size());
    if (meanDist < 1e-12) return;

    const double scale = targetExtent / meanDist;

    for (auto &p : points) {
        p.position = cv::Point3d(scale * (p.position.x - centroid[0]),
                                 scale * (p.position.y - centroid[1]),
                                 scale * (p.position.z - centroid[2]));
    }

    cv::Mat cMat = (cv::Mat_<double>(3, 1) << centroid[0], centroid[1], centroid[2]);
    for (auto &pose : poses) {
        if (!pose.registered || pose.R.empty() || pose.t.empty()) continue;
        pose.t = scale * pose.t + scale * (pose.R * cMat);
    }
}

} // namespace SfmGeom

#endif // SFMGEOMETRY_H
