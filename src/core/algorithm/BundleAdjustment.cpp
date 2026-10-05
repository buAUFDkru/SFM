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
#include <chrono>

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

// ─── 全局BA用重投影误差（内参作为可优化参数） ───────────────────────────────

/**
 * 自标定重投影误差代价函数（无畸变）
 *
 * 内参 K = [fx, fy, cx, cy]，共4个参数（阶段五 5.3 起 fx/fy 分开存储）。
 *
 * singleFocal（refineFocalPerAxis=false）时 y 方向也用 K[0]：
 * 这才是与旧版 K=[f,cx,cy] 等价的「单一焦距」。**不能**用 SubsetManifold
 * 把 K[1] 钉在初值来模拟 —— 那样 y 方向的焦距误差永远无法被内参吸收，
 * 只能污染位姿和点云。实测 test14（estimateK 初值 944 偏真值 ~2.5%）：
 * 钉 K[1] 时重投影中位 0.21→0.58px，y 用 K[0] 后恢复 0.21px。
 *
 * 投影模型：
 *   predicted_x = fx * xp + cx
 *   predicted_y = (singleFocal ? fx : fy) * yp + cy
 */
struct SelfCalibReprojectionError
{
    explicit SelfCalibReprojectionError(double obsX, double obsY,
                                        bool singleFocal = true)
        : m_obsX(obsX), m_obsY(obsY), m_singleFocal(singleFocal) {}

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

        // K = [fx, fy, cx, cy]
        const T fy = m_singleFocal ? K[0] : K[1];
        T predicted_x = K[0] * xp + K[2];
        T predicted_y = fy * yp + K[3];

        residuals[0] = predicted_x - T(m_obsX);
        residuals[1] = predicted_y - T(m_obsY);
        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2d &obs,
                                       bool singleFocal = true)
    {
        // 残差维度2，参数块维度：camera(6), point(3), K(4)
        return new ceres::AutoDiffCostFunction<SelfCalibReprojectionError, 2, 6, 3, 4>(
            new SelfCalibReprojectionError(obs.x, obs.y, singleFocal));
    }

    double m_obsX, m_obsY;
    bool m_singleFocal;
};

// ─── 全局BA用重投影误差（含径向/切向畸变，阶段五 5.2） ─────────────────────

/**
 * 带畸变的自标定重投影误差
 *
 * 参数块：camera(6), point(3), K(4)=[fx,fy,cx,cy], D(4)=[k1,k2,p1,p2]
 *
 * 畸变作用于**归一化坐标**，与 OpenCV 的 Brown–Conrady 模型一致
 * （见 SfmGeometry.h 的 distortNormalized）：
 *   r² = x² + y²,  radial = 1 + k1·r² + k2·r⁴
 *   x_d = x·radial + 2·p1·x·y + p2·(r² + 2x²)
 *   y_d = y·radial + p1·(r² + 2y²) + 2·p2·x·y
 *
 * 之所以要显式建模畸变：无人机广角（24mm 等效）在画面边缘的畸变位移可达
 * 十几像素，远大于当前 0.2~0.9px 的重投影误差水平 —— 也就是说残差里剩下的
 * 大头很可能就是畸变，把它吸收进位姿会系统性地扭曲相机轨迹。
 */
struct SelfCalibDistReprojectionError
{
    explicit SelfCalibDistReprojectionError(double obsX, double obsY,
                                            bool singleFocal = true)
        : m_obsX(obsX), m_obsY(obsY), m_singleFocal(singleFocal) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    const T *const K,
                    const T *const D,
                    T *residuals) const
    {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];

        const T r2 = xp * xp + yp * yp;
        const T r4 = r2 * r2;
        const T radial = T(1.0) + D[0] * r2 + D[1] * r4;

        const T xd = xp * radial + T(2.0) * D[2] * xp * yp +
                     D[3] * (r2 + T(2.0) * xp * xp);
        const T yd = yp * radial + D[2] * (r2 + T(2.0) * yp * yp) +
                     T(2.0) * D[3] * xp * yp;

        const T fy = m_singleFocal ? K[0] : K[1];
        residuals[0] = K[0] * xd + K[2] - T(m_obsX);
        residuals[1] = fy * yd + K[3] - T(m_obsY);
        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2d &obs,
                                       bool singleFocal = true)
    {
        // 残差维度2，参数块维度：camera(6), point(3), K(4), D(4)
        return new ceres::AutoDiffCostFunction<SelfCalibDistReprojectionError,
                                               2, 6, 3, 4, 4>(
            new SelfCalibDistReprojectionError(obs.x, obs.y, singleFocal));
    }

    double m_obsX, m_obsY;
    bool m_singleFocal;
};

// ══════════════════════════════════════════════════════════════════════════════
//  localBundleAdjustment —— 局部BA（可变相机集合 + 可变点集合）
// ══════════════════════════════════════════════════════════════════════════════

bool BundleAdjustment::localBundleAdjustment(
    std::vector<std::array<double, 6>> &cameras,
    std::vector<std::array<double, 3>> &points3d,
    const std::vector<std::tuple<int, int, cv::Point2d>> &observations,
    double fx, double fy, double cx, double cy,
    const std::vector<char> &cameraVariable,
    const std::vector<char> &pointVariable,
    int refCameraIdx,
    int maxIterations)
{
    if (cameras.size() < 2 || points3d.empty() || observations.empty())
        return false;

    ceres::Problem problem;

    const bool hasCamMask = (cameraVariable.size() == cameras.size());
    const bool hasPtMask  = (pointVariable.size() == points3d.size());

    size_t numVarCameras = 0;
    for (size_t i = 0; i < cameras.size(); ++i) {
        problem.AddParameterBlock(cameras[i].data(), 6);

        const bool isRef = (refCameraIdx >= 0 && static_cast<int>(i) == refCameraIdx);
        const bool variable = hasCamMask ? (cameraVariable[i] != 0) : true;

        if (!variable || isRef) {
            problem.SetParameterBlockConstant(cameras[i].data());
        } else {
            ++numVarCameras;
        }
    }

    size_t numVarPoints = 0;
    for (size_t i = 0; i < points3d.size(); ++i) {
        problem.AddParameterBlock(points3d[i].data(), 3);

        const bool variable = hasPtMask ? (pointVariable[i] != 0) : true;
        if (!variable) {
            problem.SetParameterBlockConstant(points3d[i].data());
        } else {
            ++numVarPoints;
        }
    }

    if (numVarCameras == 0 && numVarPoints == 0) return false;

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
    options.max_num_iterations = maxIterations;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    const std::chrono::steady_clock::time_point baT0 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    const double baMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - baT0).count();

    qDebug().noquote()
        << QString("[BA局部] 可变相机=%1, 可变点=%2, 残差=%3, 收敛=%4, %5次迭代, "
                   "初始代价=%6, 最终代价=%7, %8 ms")
               .arg(static_cast<qlonglong>(numVarCameras))
               .arg(static_cast<qlonglong>(numVarPoints))
               .arg(problem.NumResidualBlocks() * 2)
               .arg(summary.termination_type == ceres::CONVERGENCE)
               .arg(summary.iterations.size())
                .arg(summary.initial_cost, 0, 'e', 3)
                .arg(summary.final_cost, 0, 'e', 3)
               .arg(static_cast<qlonglong>(baMs));
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
    bool fixIntrinsics,
    int refCameraIdx,
    const std::vector<char> *cameraVariable,
    int maxIterations,
    BAIntrinsicsOptions *intr)
{
    if (cameras.size() < 2 || points3d.empty() || observations.empty())
        return false;

    ceres::Problem problem;

    // 阶段五 5.3：内参 K = [fx, fy, cx, cy]（4 个槽位）。
    // 关闭 refineFocalPerAxis 时 cost 函数里 y 方向直接用 K[0]（真正的
    // 单一焦距 fx=fy，与旧版 K=[f,cx,cy] 数值等价）；K[1] 不出现在残差中
    // （梯度恒 0），仍用流形钉住以明确语义。
    const bool refinePerAxis  = (intr != nullptr && intr->refineFocalPerAxis);
    const bool refineDist     = (intr != nullptr && intr->refineDistortion);
    /// 相机数不足时连焦距一起钉死（调用方通过 canRefineIntrinsics 控制）
    const bool canRefineFocal = (intr == nullptr || intr->refineFocalLength);
    const double loScale = (intr != nullptr) ? intr->focalLowerScale : 0.5;
    const double hiScale = (intr != nullptr) ? intr->focalUpperScale : 4.0;

    double K[4] = {fx, fy, cx, cy};
    double D[4] = {0.0, 0.0, 0.0, 0.0};
    if (intr != nullptr) {
        D[0] = intr->k1; D[1] = intr->k2; D[2] = intr->p1; D[3] = intr->p2;
    }

    const bool hasMask = (cameraVariable != nullptr &&
                          cameraVariable->size() == cameras.size());

    for (size_t i = 0; i < cameras.size(); ++i) {
        problem.AddParameterBlock(cameras[i].data(), 6);

        // 参考相机固定（世界坐标系基准）
        const bool isRef = (static_cast<int>(i) == refCameraIdx);
        // 未注册相机固定：它们没有任何观测，参与优化只会引入自由度与噪声
        const bool excluded = hasMask && ((*cameraVariable)[i] == 0);

        if (isRef || excluded) {
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
        // 内参可变：使用 SelfCalib*ReprojectionError，添加 K（+ 可选 D）参数块
        problem.AddParameterBlock(K, 4);
        // 弱角度图上无约束自标定是病态问题：主点会滑向非物理值（实测 cy 漂出
        // 图像外），焦距与主点/场景深度强耦合，诱发 CHOLMOD 非正定。
        // 处理：主点固定为初值（其误差可被相机旋转吸收，影响小），
        // 焦距限制在初值 [loScale, hiScale] 倍内——estimateK 的 60° FOV 假设
        // 可能偏离真值 2~3 倍，上限太紧会卡死在错误焦距上（实测点数骤降）。
        std::vector<int> fixedParams = {2, 3};           // cx, cy
        if (!refinePerAxis) fixedParams.push_back(1);    // fy → 单一焦距 fx=fy
        problem.SetManifold(K, new ceres::SubsetManifold(4, fixedParams));
        if (!canRefineFocal) {
            // 相机数太少：焦距与场景尺度同义，放开只会让 f 乱跳（实测 3228~4176）
            problem.SetParameterBlockConstant(K);
        }
        problem.SetParameterLowerBound(K, 0, loScale * std::abs(K[0]));
        problem.SetParameterUpperBound(K, 0, hiScale * std::abs(K[0]));
        if (refinePerAxis) {
            problem.SetParameterLowerBound(K, 1, loScale * std::abs(K[1]));
            problem.SetParameterUpperBound(K, 1, hiScale * std::abs(K[1]));
        }

        double *dBlock = nullptr;
        if (refineDist) {
            problem.AddParameterBlock(D, 4);
            // 畸变同样需要限幅：无约束时 k1/k2 会与主点、焦距互相拉扯，
            // 收敛到「畸变很大 + 焦距很小」这类等价但不物理的组合。
            const double kMax = intr->maxRadialDistortion;
            const double pMax = intr->maxTangentialDistortion;
            problem.SetParameterLowerBound(D, 0, -kMax);
            problem.SetParameterUpperBound(D, 0,  kMax);
            problem.SetParameterLowerBound(D, 1, -kMax);
            problem.SetParameterUpperBound(D, 1,  kMax);
            problem.SetParameterLowerBound(D, 2, -pMax);
            problem.SetParameterUpperBound(D, 2,  pMax);
            problem.SetParameterLowerBound(D, 3, -pMax);
            problem.SetParameterUpperBound(D, 3,  pMax);
            dBlock = D;
        }

        for (const auto &obs : observations) {
            int camIdx = std::get<0>(obs);
            int ptIdx  = std::get<1>(obs);
            cv::Point2d pt = std::get<2>(obs);

            if (camIdx < 0 || camIdx >= static_cast<int>(cameras.size())) continue;
            if (ptIdx < 0 || ptIdx >= static_cast<int>(points3d.size())) continue;

            if (refineDist) {
                problem.AddResidualBlock(
                    SelfCalibDistReprojectionError::Create(pt, !refinePerAxis),
                    loss,
                    cameras[camIdx].data(), points3d[ptIdx].data(), K, dBlock);
            } else {
                problem.AddResidualBlock(
                    SelfCalibReprojectionError::Create(pt, !refinePerAxis), loss,
                    cameras[camIdx].data(), points3d[ptIdx].data(), K);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = maxIterations;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    const std::chrono::steady_clock::time_point gbaT0 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    const double gbaMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gbaT0).count();

    if (!fixIntrinsics) {
        // 回写优化后的内参
        if (refinePerAxis) {
            fx = K[0];
            fy = K[1];
        } else {
            fx = K[0];
            fy = K[0];
        }
        cx = K[2];
        cy = K[3];

        if (refineDist) {
            qDebug().noquote()
                << QString("[BA全局+自标定] 精化内参: fx=%1 fy=%2 cx=%3 cy=%4 | "
                           "畸变 k1=%5 k2=%6 p1=%7 p2=%8")
                       .arg(fx, 0, 'f', 2).arg(fy, 0, 'f', 2)
                       .arg(cx, 0, 'f', 2).arg(cy, 0, 'f', 2)
                       .arg(D[0], 0, 'e', 3).arg(D[1], 0, 'e', 3)
                       .arg(D[2], 0, 'e', 3).arg(D[3], 0, 'e', 3);
        } else {
            qDebug().noquote()
                << QString("[BA全局+自标定] 精化内参: f=%1 (fx=fy), cx=%2, cy=%3")
                       .arg(fx, 0, 'f', 2).arg(cx, 0, 'f', 2).arg(cy, 0, 'f', 2);
        }
    } else {
        qDebug().noquote() << "[BA全局] 内参保持固定，未进行自标定精化";
    }

    // 畸变结果原地回写（intr 是输入输出参数，调用方直接读 k1..p2 即可）
    if (refineDist && intr != nullptr) {
        intr->k1 = D[0];
        intr->k2 = D[1];
        intr->p1 = D[2];
        intr->p2 = D[3];
    }

    qDebug().noquote()
        << QString("[BA全局] 收敛=%1, %2次迭代, 初始代价=%3, 最终代价=%4, %5 ms")
               .arg(summary.termination_type == ceres::CONVERGENCE)
               .arg(summary.iterations.size())
               .arg(summary.initial_cost, 0, 'e', 3)
               .arg(summary.final_cost, 0, 'e', 3)
               .arg(static_cast<qlonglong>(gbaMs));

    return summary.termination_type == ceres::CONVERGENCE;
}