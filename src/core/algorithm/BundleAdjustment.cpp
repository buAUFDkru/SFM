/**
 * @file BundleAdjustment.cpp
 * @brief 光束法平差实现 —— 局部BA与全局BA（含自标定精化）
 *
 * 局部BA (bundleAdjustment)：
 *   每加入一张新图像后执行，仅优化该图像的R、t和新三角化的三维点，
 *   其余相机和点固定，实现增量式优化，计算量小、速度快。
 *
 * 全局BA (globalBundleAdjustment)：
 *   重建流程末尾执行，联合优化所有外参(R,t)、所有三维点、内参(fx,fy,cx,cy)。
 *   第一张图的位姿固定为世界参考系，使用Huber鲁棒核抑制外点。
 */

#include "BundleAdjustment.h"

#include <QDebug>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

BundleAdjustment *BundleAdjustment::s_instance = nullptr;

BundleAdjustment *BundleAdjustment::instance()
{
    if (!s_instance) {
        s_instance = new BundleAdjustment();
    }
    return s_instance;
}

BundleAdjustment::BundleAdjustment(QObject *parent)
    : QObject(parent)
{
}

// ─── 局部BA用重投影误差（内参固定为常量） ─────────────────────────────────

struct LocalReprojectionError
{
    LocalReprojectionError(double obsX, double obsY,
                           double fx, double fy, double cx, double cy)
        : m_obsX(obsX), m_obsY(obsY),
          m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T predicted_x = T(m_fx) * xp + T(m_cx);
        T predicted_y = T(m_fy) * yp + T(m_cy);

        residuals[0] = predicted_x - T(m_obsX);
        residuals[1] = predicted_y - T(m_obsY);
        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2d &obs,
                                       double fx, double fy,
                                       double cx, double cy)
    {
        return new ceres::AutoDiffCostFunction<LocalReprojectionError, 2, 6, 3>(
            new LocalReprojectionError(obs.x, obs.y, fx, fy, cx, cy));
    }

    double m_obsX, m_obsY;
    double m_fx, m_fy, m_cx, m_cy;
};

// ─── 全局BA用重投影误差（内参作为可优化参数，fx=fy=f） ─────────────────────

/**
 * 自标定重投影误差代价函数
 *
 * 内参 K = [f, cx, cy]，共3个参数
 * 约束 fx = fy = f（像素为正方形），减少自由度避免优化退化
 *
 * 投影模型：
 *   predicted_x = f * xp + cx
 *   predicted_y = f * yp + cy
 */
struct SelfCalibReprojectionError
{
    explicit SelfCalibReprojectionError(double obsX, double obsY)
        : m_obsX(obsX), m_obsY(obsY) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    const T *const K,
                    T *residuals) const
    {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // K = [f, cx, cy]，fx = fy = f
        T predicted_x = K[0] * xp + K[1];
        T predicted_y = K[0] * yp + K[2];

        residuals[0] = predicted_x - T(m_obsX);
        residuals[1] = predicted_y - T(m_obsY);
        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2d &obs)
    {
        // 残差维度2，参数块维度：camera(6), point(3), K(3)
        return new ceres::AutoDiffCostFunction<SelfCalibReprojectionError, 2, 6, 3, 3>(
            new SelfCalibReprojectionError(obs.x, obs.y));
    }

    double m_obsX, m_obsY;
};

// ══════════════════════════════════════════════════════════════════════════════
//  bundleAdjustment —— 局部BA
// ══════════════════════════════════════════════════════════════════════════════

bool BundleAdjustment::bundleAdjustment(
    std::vector<std::array<double, 6>> &cameras,
    std::vector<std::array<double, 3>> &points3d,
    const std::vector<std::tuple<int, int, cv::Point2d>> &observations,
    double fx, double fy, double cx, double cy,
    int newCameraIdx, int newPointStartIdx)
{
    if (cameras.size() < 2 || points3d.empty() || observations.empty())
        return false;

    ceres::Problem problem;

    for (size_t i = 0; i < cameras.size(); ++i) {
        problem.AddParameterBlock(cameras[i].data(), 6);
        if (static_cast<int>(i) != newCameraIdx) {
            problem.SetParameterBlockConstant(cameras[i].data());
        }
    }

    for (size_t i = 0; i < points3d.size(); ++i) {
        problem.AddParameterBlock(points3d[i].data(), 3);
        if (static_cast<int>(i) < newPointStartIdx) {
            problem.SetParameterBlockConstant(points3d[i].data());
        }
    }

    ceres::LossFunction *loss = new ceres::HuberLoss(4.0);

    for (const auto &obs : observations) {
        int camIdx = std::get<0>(obs);
        int ptIdx  = std::get<1>(obs);
        cv::Point2d pt = std::get<2>(obs);

        if (camIdx < 0 || camIdx >= static_cast<int>(cameras.size())) continue;
        if (ptIdx < 0 || ptIdx >= static_cast<int>(points3d.size())) continue;

        ceres::CostFunction *cost =
            LocalReprojectionError::Create(pt, fx, fy, cx, cy);

        problem.AddResidualBlock(
            cost, loss,
            cameras[camIdx].data(),
            points3d[ptIdx].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    qDebug().noquote()
        << QString("[BA局部] 相机#%1, 收敛=%2, %3次迭代, 初始代价=%4, 最终代价=%5")
               .arg(newCameraIdx)
               .arg(summary.termination_type == ceres::CONVERGENCE)
               .arg(summary.iterations.size())
               .arg(summary.initial_cost, 0, 'e', 3)
               .arg(summary.final_cost, 0, 'e', 3);
    return summary.termination_type == ceres::CONVERGENCE;
}

// ══════════════════════════════════════════════════════════════════════════════
//  globalBundleAdjustment —— 全局BA + 自标定精化
// ══════════════════════════════════════════════════════════════════════════════

bool BundleAdjustment::globalBundleAdjustment(
    std::vector<std::array<double, 6>> &cameras,
    std::vector<std::array<double, 3>> &points3d,
    const std::vector<std::tuple<int, int, cv::Point2d>> &observations,
    double &fx, double &fy, double &cx, double &cy,
    bool fixIntrinsics)
{
    if (cameras.size() < 2 || points3d.empty() || observations.empty())
        return false;

    ceres::Problem problem;

    // 内参 K = [f, cx, cy]，约束 fx = fy = f
    // 初始焦距取 fx 和 fy 的均值
    double f_init = (fx + fy) / 2.0;
    double K[3] = {f_init, cx, cy};

    for (size_t i = 0; i < cameras.size(); ++i) {
        problem.AddParameterBlock(cameras[i].data(), 6);
        if (i == 0) {
            problem.SetParameterBlockConstant(cameras[i].data());
        }
    }

    for (auto &pt : points3d) {
        problem.AddParameterBlock(pt.data(), 3);
    }

    ceres::LossFunction *loss = new ceres::HuberLoss(4.0);

    if (fixIntrinsics) {
        // 内参固定：使用 LocalReprojectionError（内参作为常量）
        qDebug().noquote() << "[BA全局] 内参已固定，仅优化外参和三维点";
        for (const auto &obs : observations) {
            int camIdx = std::get<0>(obs);
            int ptIdx  = std::get<1>(obs);
            cv::Point2d pt = std::get<2>(obs);

            if (camIdx < 0 || camIdx >= static_cast<int>(cameras.size())) continue;
            if (ptIdx < 0 || ptIdx >= static_cast<int>(points3d.size())) continue;

            ceres::CostFunction *cost =
                LocalReprojectionError::Create(pt, fx, fy, cx, cy);

            problem.AddResidualBlock(
                cost, loss,
                cameras[camIdx].data(),
                points3d[ptIdx].data());
        }
    } else {
        // 内参可变：使用 SelfCalibReprojectionError，添加 K 参数块
        problem.AddParameterBlock(K, 3);
        for (const auto &obs : observations) {
            int camIdx = std::get<0>(obs);
            int ptIdx  = std::get<1>(obs);
            cv::Point2d pt = std::get<2>(obs);

            if (camIdx < 0 || camIdx >= static_cast<int>(cameras.size())) continue;
            if (ptIdx < 0 || ptIdx >= static_cast<int>(points3d.size())) continue;

            ceres::CostFunction *cost =
                SelfCalibReprojectionError::Create(pt);

            problem.AddResidualBlock(
                cost, loss,
                cameras[camIdx].data(),
                points3d[ptIdx].data(),
                K);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 200;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!fixIntrinsics) {
        // 回写优化后的内参：fx = fy = f
        fx = K[0];
        fy = K[0];
        cx = K[1];
        cy = K[2];
        qDebug().noquote()
            << QString("[BA全局+自标定] 精化内参: f=%1 (fx=fy), cx=%2, cy=%3")
                   .arg(fx, 0, 'f', 2).arg(cx, 0, 'f', 2).arg(cy, 0, 'f', 2);
    } else {
        qDebug().noquote() << "[BA全局] 内参保持固定，未进行自标定精化";
    }

    qDebug().noquote()
        << QString("[BA全局] 收敛=%1, %2次迭代, 初始代价=%3, 最终代价=%4")
               .arg(summary.termination_type == ceres::CONVERGENCE)
               .arg(summary.iterations.size())
               .arg(summary.initial_cost, 0, 'e', 3)
               .arg(summary.final_cost, 0, 'e', 3);

    return summary.termination_type == ceres::CONVERGENCE;
}