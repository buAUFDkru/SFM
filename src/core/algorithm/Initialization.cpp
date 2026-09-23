/**
 * @file Initialization.cpp
 * @brief 增量式SFM初始化阶段实现
 *
 * 本文件实现了增量式SFM重建的初始化阶段，包括：
 * 1. 寻找最优初始化图像对（COLMAP式基线评分）
 * 2. 2D-2D相对定向（F/H估计 + GRIC模型选择）
 * 3. 相对位姿R、t恢复（F分解或H分解）
 *
 */

#include "Initialization.h"

#include <QDebug>
#include <QtMath>
#include <algorithm>

#include <opencv2/calib3d.hpp>

// ═════════════════════════════════════════════════════════════════════════════
//  单例实现
// ═════════════════════════════════════════════════════════════════════════════

Initialization *Initialization::s_instance = nullptr;

Initialization *Initialization::instance()
{
    if (!s_instance) {
        s_instance = new Initialization();
    }
    return s_instance;
}

Initialization::Initialization(QObject *parent)
    : QObject(parent)
{
}

// ═════════════════════════════════════════════════════════════════════════════
//  内部辅助函数
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief 计算基础矩阵F的Sampson距离平方
 *
 * Sampson距离是基础矩阵的一阶近似几何误差，计算公式：
 *   d_sampson² = (x₂ᵀ F x₁)² / (‖Fx₁‖²₍₁₂₎ + ‖Fᵀx₂‖²₍₁₂₎)
 *
 * 其中 ‖v‖²₍₁₂₎ 表示向量v前两个分量的平方和（忽略第三分量）。
 * Sampson距离比对称转移误差更高效，是F矩阵的标准误差度量。
 *
 * @param F 基础矩阵（3×3）
 * @param pt1 图像1中的2D点（像素坐标）
 * @param pt2 图像2中的2D点（像素坐标）
 * @return Sampson距离的平方（像素²）
 */
static double sampsonErrorSq(const cv::Mat &F,
                             const cv::Point2d &pt1,
                             const cv::Point2d &pt2)
{
    // 构造齐次坐标 [x, y, 1]ᵀ
    double x1_arr[] = {pt1.x, pt1.y, 1.0};
    double x2_arr[] = {pt2.x, pt2.y, 1.0};
    cv::Mat x1(3, 1, CV_64F, x1_arr);
    cv::Mat x2(3, 1, CV_64F, x2_arr);

    // Fx1 = F * x1, Ftx2 = Fᵀ * x2
    cv::Mat Fx1 = F * x1;
    cv::Mat Ftx2 = F.t() * x2;

    // 分子：x₂ᵀ F x₁ （极线约束的残差）
    double numerator = x2.dot(Fx1);

    // 分母：Fx1和Ftx2的前两个分量平方和
    double denom = Fx1.at<double>(0, 0) * Fx1.at<double>(0, 0)
                 + Fx1.at<double>(1, 0) * Fx1.at<double>(1, 0)
                 + Ftx2.at<double>(0, 0) * Ftx2.at<double>(0, 0)
                 + Ftx2.at<double>(1, 0) * Ftx2.at<double>(1, 0);

    if (denom < 1e-12)
        return 0.0;

    return (numerator * numerator) / denom;
}

/**
 * @brief 计算单应矩阵H的对称转移误差平方
 *
 * 对称转移误差同时考虑正向和反向投影：
 *   d_sym² = d(Hx₁, x₂)² + d(H⁻¹x₂, x₁)²
 *
 * 正向：将x₁通过H投影到图像2，与x₂的欧氏距离
 * 反向：将x₂通过H⁻¹投影到图像1，与x₁的欧氏距离
 *
 * 这比单向转移误差更鲁棒，是H矩阵的标准误差度量。
 *
 * @param H 单应矩阵（3×3）
 * @param pt1 图像1中的2D点（像素坐标）
 * @param pt2 图像2中的2D点（像素坐标）
 * @return 对称转移误差的平方（像素²）
 */
static double symmetricTransferErrorSq(const cv::Mat &H,
                                       const cv::Point2d &pt1,
                                       const cv::Point2d &pt2)
{
    // ── 正向投影误差：H * x₁ → x₂ ──
    double x1_arr[] = {pt1.x, pt1.y, 1.0};
    cv::Mat x1(3, 1, CV_64F, x1_arr);
    cv::Mat Hx1 = H * x1;
    // 齐次坐标归一化
    double u = Hx1.at<double>(0, 0) / Hx1.at<double>(2, 0);
    double v = Hx1.at<double>(1, 0) / Hx1.at<double>(2, 0);
    double forward_err = (u - pt2.x) * (u - pt2.x) + (v - pt2.y) * (v - pt2.y);

    // ── 反向投影误差：H⁻¹ * x₂ → x₁ ──
    cv::Mat Hinv;
    cv::invert(H, Hinv, cv::DECOMP_LU);
    double x2_arr[] = {pt2.x, pt2.y, 1.0};
    cv::Mat x2(3, 1, CV_64F, x2_arr);
    cv::Mat Hinv_x2 = Hinv * x2;
    u = Hinv_x2.at<double>(0, 0) / Hinv_x2.at<double>(2, 0);
    v = Hinv_x2.at<double>(1, 0) / Hinv_x2.at<double>(2, 0);
    double backward_err = (u - pt1.x) * (u - pt1.x) + (v - pt1.y) * (v - pt1.y);

    return forward_err + backward_err;
}

/**
 * @brief GRIC鲁棒截断函数 ρ(e)
 *
 * 对残差进行截断，防止大残差（外点）对GRIC得分产生过大影响：
 *   ρ(e²/σ²) = min(e²/σ², λ₃(r-d))
 *
 * 当残差小于阈值时，直接使用归一化残差；
 * 当残差超过阈值时，截断为固定值，降低外点的影响。
 *
 * @param e_sq 误差的平方
 * @param sigma_sq 噪声方差的平方（σ²）
 * @param threshold 截断阈值 λ₃(r-d)
 * @return 截断后的鲁棒残差值
 */
static double rho_robust(double e_sq, double sigma_sq, double threshold)
{
    return std::min(e_sq / sigma_sq, threshold);
}

/**
 * @brief 计算GRIC（Geometric Robust Information Criterion）得分
 *
 * GRIC用于在多个几何模型间选择最优模型，公式：
 *   GRIC = Σᵢ ρ(eᵢ²/σ²) + λ₁·d·n + λ₂·k
 *
 * 各项含义：
 * - Σ ρ(eᵢ) ：所有数据点的鲁棒残差之和（衡量拟合质量）
 * - λ₁·d·n  ：数据维度的惩罚项（d越大，模型对数据维度的假设越强）
 * - λ₂·k    ：模型复杂度惩罚项（k越大，参数越多，越容易过拟合）
 *
 * GRIC得分越低，模型越优。F模型(d=3,k=7)比H模型(d=2,k=8)复杂度惩罚更低，
 * 因此在拟合质量相近时，F模型会被优先选择。
 *
 * @param errors_sq 每个数据点的误差平方列表
 * @param sigma 噪声标准差（像素）
 * @param d 模型自由度（F:3, H:2）
 * @param k 模型参数个数（F:7, H:8）
 * @param r 数据维度（2D点对应r=4）
 * @param lambda3 截断惩罚系数（默认2.0）
 * @return GRIC得分（越小越优）
 */
static double computeGRIC(const std::vector<double> &errors_sq,
                          double sigma,
                          int d,
                          int k,
                          int r = 4,
                          double lambda3 = 2.0)
{
    const int n = static_cast<int>(errors_sq.size());
    if (n == 0) return std::numeric_limits<double>::max();

    const double sigma_sq = sigma * sigma;
    const double threshold = lambda3 * (r - d);

    // 累加鲁棒残差
    double sum_rho = 0.0;
    for (double e_sq : errors_sq) {
        sum_rho += rho_robust(e_sq, sigma_sq, threshold);
    }

    // GRIC惩罚系数
    const double lambda1 = std::log(4.0);   // ≈ 1.386
    const double lambda2 = std::log(4.0 * n); // 随数据量增长

    double gric = sum_rho + lambda1 * d * n + lambda2 * k;
    return gric;
}

/**
 * @brief 计算中位数
 *
 * 对输入向量排序后取中间值。偶数个元素时取两个中间值的平均。
 * 中位数比均值更鲁棒，不受极端值影响。
 *
 * @param v 输入数值向量
 * @return 中位数值
 */
static double median(const std::vector<double> &v)
{
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    size_t mid = s.size() / 2;
    if (s.size() % 2 == 0) {
        return (s[mid - 1] + s[mid]) * 0.5;
    }
    return s[mid];
}

// ═════════════════════════════════════════════════════════════════════════════
//  公开方法
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief 2D-2D相对定向：估计F和H，用GRIC准则选择最优模型
 *
 * 步骤：
 * 1. 用RANSAC分别估计基础矩阵F和单应矩阵H
 * 2. 对每个匹配点计算F的Sampson误差和H的对称转移误差
 * 3. 分别计算两种模型的GRIC得分
 * 4. 选择GRIC得分更低的模型
 *
 * 特殊处理：当匹配点数 < 8 时，F矩阵无法估计（最少需要8点），
 * 此时直接使用H模型。
 */
int Initialization::selectModel(const std::vector<cv::Point2d> &pts1,
                                 const std::vector<cv::Point2d> &pts2,
                                 cv::Mat &F, cv::Mat &H,
                                 std::vector<uchar> &outInlierMask,
                                 double sigma, double ransacThreshold)
{
    const int n = static_cast<int>(pts1.size());

    // 匹配点不足8个，无法估计F矩阵，退化为H模型
    if (n < 8) {
        H = cv::findHomography(pts1, pts2, cv::RANSAC, ransacThreshold, outInlierMask);
        return 1;
    }

    cv::Mat mask_F, mask_H;

    // RANSAC估计基础矩阵F（7点或8点算法，FM_RANSAC使用8点法）
    F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                                ransacThreshold, 0.99, mask_F);

    // RANSAC估计单应矩阵H
    H = cv::findHomography(pts1, pts2, cv::RANSAC, ransacThreshold, mask_H);

    // 计算每个点在两种模型下的误差
    std::vector<double> errors_F_sq(n), errors_H_sq(n);

    for (int i = 0; i < n; ++i) {
        errors_F_sq[i] = sampsonErrorSq(F, pts1[i], pts2[i]);
        errors_H_sq[i] = symmetricTransferErrorSq(H, pts1[i], pts2[i]);
    }

    // 计算GRIC得分：F模型 d=3(3D场景自由度), k=7(7参数);
    //             H模型 d=2(平面自由度), k=8(8参数)
    double gric_F = computeGRIC(errors_F_sq, sigma, /*d=*/3, /*k=*/7);
    double gric_H = computeGRIC(errors_H_sq, sigma, /*d=*/2, /*k=*/8);

    // 选择GRIC更低的模型
    if (gric_H < gric_F) {
        outInlierMask = mask_H;
        return 1;  // H模型
    } else {
        outInlierMask = mask_F;
        return 0;  // F模型
    }
}

/**
 * @brief 从基础矩阵F恢复相对位姿R、t
 *
 * 步骤：
 * 1. 由F计算本质矩阵E = KᵀFK
 * 2. cv::recoverPose内部对E进行SVD分解，得到4组候选解
 * 3. 通过cheirality约束（三维点在两个相机前方）选择唯一解
 * 4. 返回内点数，≥8视为成功
 */
bool Initialization::recoverPoseFromF(const std::vector<cv::Point2d> &pts1,
                                       const std::vector<cv::Point2d> &pts2,
                                       const cv::Mat &K, const cv::Mat &F,
                                       cv::Mat &R, cv::Mat &t,
                                       std::vector<uchar> &inlierMask)
{
    // F矩阵分解至少需要8个点
    if (pts1.size() < 8) {
        return false;
    }

    // 由基础矩阵F计算本质矩阵E = KᵀFK
    cv::Mat E = K.t() * F * K;

    if (E.empty()) {
        return false;
    }

    // recoverPose：分解E得到R、t，通过正深度验证选最优解
    // 内点阈值500（较大值，因为输入已经是RANSAC内点）
    int inlierCount = cv::recoverPose(E, pts1, pts2, K, R, t, 500, inlierMask);

    return inlierCount >= 8;
}

/**
 * @brief 从单应矩阵H恢复相对位姿R、t
 *
 * 步骤：
 * 1. cv::decomposeHomographyMat分解H，最多得到4组候选解(R,t,n)
 * 2. 对每组解，取前20个匹配点进行三角化
 * 3. 统计在两个相机前方（正深度）的点数
 * 4. 选择正深度点数最多的解作为最终结果
 *
 * H分解的物理含义：
 * - H描述的是平面到平面的映射：x₂ = H·x₁
 * - 分解后得到相机间的旋转R、平移t和平面法向量n
 * - 由于尺度歧义，t只有方向没有尺度
 */
bool Initialization::recoverPoseFromH(const std::vector<cv::Point2d> &pts1,
                                       const std::vector<cv::Point2d> &pts2,
                                       const cv::Mat &K, const cv::Mat &H,
                                       cv::Mat &R, cv::Mat &t)
{
    if (pts1.size() < 4) {
        return false;
    }

    // 分解H矩阵，得到多组候选解
    std::vector<cv::Mat> rotations, translations, normals;
    int nsolutions = cv::decomposeHomographyMat(H, K, rotations, translations, normals);

    if (nsolutions == 0) {
        return false;
    }

    Q_UNUSED(normals);

    // 对每组解，通过正深度验证选择最优解
    int bestIdx = -1;
    int maxPosDepth = -1;

    for (int i = 0; i < nsolutions; ++i) {
        int posDepthCount = 0;

        // 只取前20个点进行验证（加速）
        for (size_t j = 0; j < pts1.size() && j < 20; ++j) {
            // 像素坐标转归一化相机坐标
            cv::Point2d n1 = pixel2cam(pts1[j], K);
            cv::Point2d n2 = pixel2cam(pts2[j], K);

            // 相机1的投影矩阵 [I|0]（世界坐标系与相机1重合）
            cv::Mat T1 = (cv::Mat_<double>(3, 4) <<
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0);

            // 相机2的投影矩阵 [R|t]
            cv::Mat R_i = rotations[i];
            cv::Mat t_i = translations[i];
            cv::Mat T2 = (cv::Mat_<double>(3, 4) <<
                R_i.at<double>(0, 0), R_i.at<double>(0, 1), R_i.at<double>(0, 2), t_i.at<double>(0, 0),
                R_i.at<double>(1, 0), R_i.at<double>(1, 1), R_i.at<double>(1, 2), t_i.at<double>(1, 0),
                R_i.at<double>(2, 0), R_i.at<double>(2, 1), R_i.at<double>(2, 2), t_i.at<double>(2, 0));

            // 线性三角化
            cv::Mat ptsMat1 = (cv::Mat_<double>(2, 1) << n1.x, n1.y);
            cv::Mat ptsMat2 = (cv::Mat_<double>(2, 1) << n2.x, n2.y);
            cv::Mat pts4D;
            cv::triangulatePoints(T1, T2, ptsMat1, ptsMat2, pts4D);

            // 齐次坐标归一化
            double w = pts4D.at<double>(3, 0);
            if (std::abs(w) < 1e-10) continue;

            // 检查相机1前方（z > 0）
            double z1 = pts4D.at<double>(2, 0) / w;
            if (z1 > 0) {
                // 检查相机2前方：P_cam2 = R * P_world + t
                cv::Mat ptCam2 = R_i * (cv::Mat_<double>(3, 1) <<
                    pts4D.at<double>(0, 0) / w,
                    pts4D.at<double>(1, 0) / w,
                    z1) + t_i;
                if (ptCam2.at<double>(2, 0) > 0) {
                    ++posDepthCount;
                }
            }
        }

        // 选择正深度点数最多的解
        if (posDepthCount > maxPosDepth) {
            maxPosDepth = posDepthCount;
            bestIdx = i;
        }
    }

    if (bestIdx < 0) {
        return false;
    }

    R = rotations[bestIdx].clone();
    t = translations[bestIdx].clone();

    return true;
}

/**
 * @brief 寻找最优初始化图像对
 *
 * 采用COLMAP式评分策略，对每个候选对计算综合得分：
 *
 * 1. 估计相对位姿（F或H分解）
 * 2. 三角化内点，计算中位三角化角度
 * 3. 评分 = N_inlier × f(angle)
 *    其中 f(angle) 在 angle≈5-6° 时达到峰值（最优基线/深度比约0.1）
 *
 * 三角化角度权重函数：
 *   angle < 1°  → 权重=0（退化：纯旋转/基线过短）
 *   1° ≤ angle < 5° → 权重线性增长
 *   5° ≤ angle ≤ 10° → 权重=1（最优区间）
 *   10° < angle ≤ 20° → 权重线性衰减
 *   angle > 20° → 权重=0（基线过宽，匹配不可靠）
 */
InitResult Initialization::findInitialPair(const std::vector<FeatureResult> &features,
                                            const std::vector<MatchPair> &matchPairs,
                                            const cv::Mat &K)
{
    InitResult best;
    best.score = -1.0;
    best.modelChoice = -1;

    for (const auto &mp : matchPairs) {
        // ── 过滤匹配点数不足的候选对 ──
        if (mp.matchCount < 30) continue;

        int imgId1 = mp.imageId1;
        int imgId2 = mp.imageId2;

        // ── 从 features 获取关键点坐标 ──
        const auto &kpts1 = features[imgId1].keypoints;
        const auto &kpts2 = features[imgId2].keypoints;

        std::vector<cv::Point2d> pts1, pts2;
        pts1.reserve(mp.matches.size());
        pts2.reserve(mp.matches.size());
        for (const auto &m : mp.matches) {
            pts1.push_back(kpts1[m.queryIdx].pt);
            pts2.push_back(kpts2[m.trainIdx].pt);
        }

        // ── 2D-2D 相对定向 + 位姿恢复 ──
        // 根据 MatcherFactory 中 GRIC 选择的模型恢复位姿
        cv::Mat R, t;
        int modelChoice = -1;
        bool poseOk = false;

        if (mp.modelChoice == 0) {
            // F模型：通过本质矩阵E分解
            std::vector<uchar> inlierMask;
            poseOk = recoverPoseFromF(pts1, pts2, K, mp.F, R, t, inlierMask);
            modelChoice = 0;
        } else {
            // H模型：通过单应矩阵分解
            poseOk = recoverPoseFromH(pts1, pts2, K, mp.H, R, t);
            modelChoice = 1;
        }

        if (!poseOk) continue;

        // ── 计算基线长度 ──
        double baseline = cv::norm(t);

        // ── 三角化所有内点，计算三角化角度 ──
        std::vector<double> angles;
        angles.reserve(pts1.size());

        // 相机1投影矩阵 [I|0]
        cv::Mat T1 = (cv::Mat_<double>(3, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);

        // 相机2投影矩阵 [R|t]
        cv::Mat T2 = (cv::Mat_<double>(3, 4) <<
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

        for (size_t k = 0; k < pts1.size(); ++k) {
            // 像素坐标转归一化相机坐标
            cv::Point2d n1 = pixel2cam(pts1[k], K);
            cv::Point2d n2 = pixel2cam(pts2[k], K);

            // 线性三角化
            cv::Mat ptsMat1 = (cv::Mat_<double>(2, 1) << n1.x, n1.y);
            cv::Mat ptsMat2 = (cv::Mat_<double>(2, 1) << n2.x, n2.y);
            cv::Mat pts4D;
            cv::triangulatePoints(T1, T2, ptsMat1, ptsMat2, pts4D);

            // 齐次坐标归一化
            double w = pts4D.at<double>(3, 0);
            if (std::abs(w) < 1e-10) continue;

            double x = pts4D.at<double>(0, 0) / w;
            double y = pts4D.at<double>(1, 0) / w;
            double z = pts4D.at<double>(2, 0) / w;

            // 三维点必须在相机1前方
            if (z <= 0) continue;

            // 计算该点在相机2坐标系下的坐标
            cv::Mat ptCam2 = R * (cv::Mat_<double>(3, 1) << x, y, z) + t;

            // 三维点也必须在相机2前方
            if (ptCam2.at<double>(2, 0) <= 0) continue;

            // ── 计算三角化角度 ──
            // 相机1射线方向（从光心到三维点）：归一化坐标(n1.x, n1.y, 1)
            double ray1_len = std::sqrt(n1.x * n1.x + n1.y * n1.y + 1.0);
            // 相机2射线方向：归一化后的ptCam2
            double ray2_x = ptCam2.at<double>(0, 0) / ptCam2.at<double>(2, 0);
            double ray2_y = ptCam2.at<double>(1, 0) / ptCam2.at<double>(2, 0);
            double ray2_len = std::sqrt(ray2_x * ray2_x + ray2_y * ray2_y + 1.0);

            // 两射线的夹角余弦
            double cos_angle = (n1.x * ray2_x + n1.y * ray2_y + 1.0) / (ray1_len * ray2_len);
            cos_angle = std::max(-1.0, std::min(1.0, cos_angle));

            double angle_rad = std::acos(cos_angle);
            double angle_deg = angle_rad * 180.0 / std::acos(-1.0);
            angles.push_back(angle_deg);
        }

        if (angles.empty()) continue;

        // 取中位角度（比均值更鲁棒）
        double medAngle = median(angles);

        // ── 角度权重函数 ──
        // 峰值在5°-10°，0°和>20°时权重为0
        double angleWeight = 0.0;
        if (medAngle >= 1.0 && medAngle < 5.0) {
            // 线性增长：1°→0, 5°→1
            angleWeight = (medAngle - 1.0) / 4.0;
        } else if (medAngle >= 5.0 && medAngle <= 10.0) {
            // 最优区间：权重=1
            angleWeight = 1.0;
        } else if (medAngle > 10.0 && medAngle <= 20.0) {
            // 线性衰减：10°→1, 20°→0
            angleWeight = 1.0 - (medAngle - 10.0) / 10.0;
        }
        // <1° 或 >20°：angleWeight = 0.0

        // ── 综合评分 ──
        int validInliers = static_cast<int>(angles.size());
        double pairScore = validInliers * angleWeight;

        // F模型加分（非平面场景更适合初始化，避免退化）
        if (modelChoice == 0) {
            pairScore *= 1.2;
        }

        qDebug().noquote()
            << QString("[初始化] 候选对 #%1↔#%2: 内点=%3, 中位角=%4°, 角度权重=%5, 模型=%6, 基线=%7, 评分=%8")
                   .arg(imgId1).arg(imgId2).arg(validInliers)
                   .arg(medAngle, 0, 'f', 2)
                   .arg(angleWeight, 0, 'f', 3)
                   .arg(modelChoice == 0 ? "F" : "H")
                   .arg(baseline, 0, 'f', 4)
                   .arg(pairScore, 0, 'f', 2);

        // ── 更新最优结果 ──
        if (pairScore > best.score) {
            best.imageId1 = imgId1;
            best.imageId2 = imgId2;
            best.R = R.clone();
            best.t = t.clone();
            best.F = mp.F.clone();
            best.H = mp.H.clone();
            best.modelChoice = modelChoice;
            best.score = pairScore;
            best.baseline = baseline;
            best.medianAngle = medAngle;

            // 保存内点坐标和匹配索引
            best.points1.clear();
            best.points2.clear();
            best.inlierMatches.clear();
            for (const auto &m : mp.matches) {
                best.points1.push_back(kpts1[m.queryIdx].pt);
                best.points2.push_back(kpts2[m.trainIdx].pt);
                best.inlierMatches.push_back(m);
            }
        }
    }

    // ── 输出最终结果 ──
    if (best.score < 0) {
        qDebug().noquote() << "[初始化] 未找到合适的初始化对";
    } else {
        qDebug().noquote()
            << QString("[初始化] 最优初始化对: #%1↔#%2, 评分=%3, 基线=%4, 角度=%5°, 模型=%6")
                   .arg(best.imageId1).arg(best.imageId2)
                   .arg(best.score, 0, 'f', 2)
                   .arg(best.baseline, 0, 'f', 4)
                   .arg(best.medianAngle, 0, 'f', 2)
                   .arg(best.modelChoice == 0 ? "F" : "H");
    }

    return best;
}