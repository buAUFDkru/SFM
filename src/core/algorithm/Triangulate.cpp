/**
 * @file Triangulate.cpp
 * @brief 三角测量类实现 —— 增量式SFM稀疏重建核心算法
 *
 * 本文件实现了增量式SFM稀疏重建流程：
 * 1. 选取种子对，根据GRIC模型选择结果估计相对位姿
 * 2. 对种子对进行三角测量，生成初始三维点云
 * 3. 逐对三角测量 → PnP后方交会注册新相机 → 局部BA优化
 * 4. 全部完成后执行全局BA+自标定精化
 *
 * 光束法平差优化已剥离至 BundleAdjustment 类。
 *
 * 【阶段一健壮性改造】
 * 早期实现中，增量主循环在「三视图共有点不足 / 找不到 secondReg / 三角化无有效点」
 * 等情形下直接 break，导致视角变化大的数据集一旦某张图不满足条件就整体重建失败。
 * 改造要点：
 * - 所有单张图像的失败降级为「记一次失败 → 换下一个候选」，不再终止整个重建
 * - 维护 kpTo3D（关键点索引 → 三维点索引），注册新图像时跨**所有**已注册图像
 *   收集 2D-3D 对应（对应 COLMAP RegisterNextImage），历史点可被复用
 * - 下一张待注册图像按「可见三维点数」排序（对应 COLMAP FindNextImages）
 * - 三角化加入 cheirality 正确判定 + 三角化角 + 重投影误差过滤
 * - PnP 加入内点数 / 内点率 / 中位重投影误差校验，并回传内点掩码
 */

#include "Triangulate.h"
#include "BundleAdjustment.h"

#include <QtMath>
#include <QVector>
#include <QDebug>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <opencv2/calib3d.hpp>
#include <unordered_map>

Triangulate *Triangulate::s_instance = nullptr;

Triangulate *Triangulate::instance()
{
    if (!s_instance) {
        s_instance = new Triangulate();
    }
    return s_instance;
}

Triangulate::Triangulate(QObject *parent)
    : QObject(parent)
{
}

// ═════════════════════════════════════════════════════════════════════════════
//  内部辅助函数
// ═════════════════════════════════════════════════════════════════════════════

namespace {

/// 计算向量中位数（为空返回 0）
double medianOf(const std::vector<double> &v)
{
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    const size_t mid = s.size() / 2;
    if (s.size() % 2 == 0) return (s[mid - 1] + s[mid]) * 0.5;
    return s[mid];
}

/// 相机光心在世界坐标系中的位置：C = -Rᵀ·t
cv::Vec3d cameraCenter(const cv::Mat &R, const cv::Mat &t)
{
    cv::Mat C = -R.t() * t;
    return cv::Vec3d(C.at<double>(0), C.at<double>(1), C.at<double>(2));
}

/// 将世界点投影到像素平面，返回重投影误差（像素）；深度 ≤ 0 时返回 -1
double reprojectionError(const cv::Vec3d &Pw,
                         const cv::Mat &R, const cv::Mat &t,
                         const cv::Mat &K,
                         const cv::Point2d &obs)
{
    cv::Mat PwMat = (cv::Mat_<double>(3, 1) << Pw[0], Pw[1], Pw[2]);
    cv::Mat Pc = R * PwMat + t;
    const double z = Pc.at<double>(2, 0);
    if (z <= 0.0) return -1.0;

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    const double u = fx * Pc.at<double>(0, 0) / z + cx;
    const double v = fy * Pc.at<double>(1, 0) / z + cy;

    return std::hypot(u - obs.x, v - obs.y);
}

/**
 * @brief 计算三角化角度（两条视线之间的夹角，度）
 *
 * 视线定义为「从相机光心指向三维点」的方向。角度过小说明基线相对深度太短，
 * 深度估计不可靠（COLMAP 判据：min_angle = 1.5°）。
 */
double triangulationAngleDeg(const cv::Vec3d &Pw,
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

} // namespace

cv::Mat Triangulate::estimateK(const std::vector<FeatureResult> &features)
{
    if (features.empty() || features[0].imageWidth < 1 || features[0].imageHeight < 1) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    return ::estimateK(features[0].imageWidth, features[0].imageHeight);
}

double Triangulate::computeImageFeatureDistributionScore(
    int imageId, const std::vector<FeatureResult> &features) const
{
    if (imageId < 0 || imageId >= static_cast<int>(features.size())) {
        return 0.0;
    }

    const FeatureResult &feature = features[imageId];
    const int width = feature.imageWidth;
    const int height = feature.imageHeight;
    if (width <= 0 || height <= 0 || feature.keypoints.empty()) {
        return 0.0;
    }

    // 三个网格层级：2x2、4x4、8x8，单格权重分别为 1、2、3
    const int gridCounts[3] = {2, 4, 8};
    const double cellWeights[3] = {1.0, 2.0, 3.0};

    double totalScore = 0.0;

    for (int lvl = 0; lvl < 3; ++lvl) {
        const int gridCount = gridCounts[lvl];
        const double cellWeight = cellWeights[lvl];

        std::vector<bool> occupied(gridCount * gridCount, false);

        const double cellW = static_cast<double>(width) / gridCount;
        const double cellH = static_cast<double>(height) / gridCount;

        for (const auto &kp : feature.keypoints) {
            int gx = static_cast<int>(kp.pt.x / cellW);
            int gy = static_cast<int>(kp.pt.y / cellH);
            gx = std::max(0, std::min(gridCount - 1, gx));
            gy = std::max(0, std::min(gridCount - 1, gy));
            occupied[gy * gridCount + gx] = true;
        }

        int occupiedCount = 0;
        for (bool occ : occupied) {
            if (occ) ++occupiedCount;
        }

        totalScore += occupiedCount * cellWeight;
    }

    return totalScore;
}

void Triangulate::triangulatePair(const MatchPair &pair,
                                   const cv::Mat &K1, const cv::Mat &K2,
                                   const cv::Mat &R1, const cv::Mat &t1,
                                   const cv::Mat &R2, const cv::Mat &t2,
                                   std::vector<cv::Point3d> &outWorldPoints,
                                   std::vector<bool> *outKeepMask)
{
    outWorldPoints.clear();

    cv::Mat T1 = (cv::Mat_<double>(3, 4) <<
        R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0, 0),
        R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1, 0),
        R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2, 0));

    cv::Mat T2 = (cv::Mat_<double>(3, 4) <<
        R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0, 0),
        R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1, 0),
        R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2, 0));

    std::vector<cv::Point2d> pts1 = pair.points1;
    std::vector<cv::Point2d> pts2 = pair.points2;

    if (pts1.empty()) {
        qDebug().noquote()
            << QString("[三角测量] pair #%1↔#%2: 无有效点").arg(pair.imageId1).arg(pair.imageId2);
        return;
    }

    cv::Mat ptsMat1(2, static_cast<int>(pts1.size()), CV_64F);
    cv::Mat ptsMat2(2, static_cast<int>(pts2.size()), CV_64F);
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Point2d n1 = pixel2cam(cv::Point2d(pts1[i].x, pts1[i].y), K1);
        cv::Point2d n2 = pixel2cam(cv::Point2d(pts2[i].x, pts2[i].y), K2);
        ptsMat1.at<double>(0, static_cast<int>(i)) = n1.x;
        ptsMat1.at<double>(1, static_cast<int>(i)) = n1.y;
        ptsMat2.at<double>(0, static_cast<int>(i)) = n2.x;
        ptsMat2.at<double>(1, static_cast<int>(i)) = n2.y;
    }

    cv::Mat pts4D;
    cv::triangulatePoints(T1, T2, ptsMat1, ptsMat2, pts4D);

    outWorldPoints.reserve(pts1.size());
    if (outKeepMask) {
        outKeepMask->resize(pts1.size(), false);
    }

    // 相机光心（世界坐标），用于三角化角度计算
    const cv::Vec3d C1 = cameraCenter(R1, t1);
    const cv::Vec3d C2 = cameraCenter(R2, t2);

    int rejectedBehind = 0;
    int rejectedAngle  = 0;
    int rejectedReproj = 0;

    for (int i = 0; i < pts4D.cols; ++i) {
        double w = pts4D.at<double>(3, i);

        if (std::abs(w) < 1e-10) continue;

        double x = pts4D.at<double>(0, i) / w;
        double y = pts4D.at<double>(1, i) / w;
        double z = pts4D.at<double>(2, i) / w;

        if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
        if (std::isinf(x) || std::isinf(y) || std::isinf(z)) continue;

        const cv::Vec3d Pw(x, y, z);

        // ── cheirality 检查 ──────────────────────────────────────────────
        // triangulatePoints 的输出是【世界坐标】（T1/T2 已含 R,t），
        // 原先判定的 z>0 是世界 z 分量，与"点在相机前方"无关（实现缺陷）。
        // 正确做法：分别检查点在两个相机坐标系下的深度均 > 0。
        cv::Mat PwMat = (cv::Mat_<double>(3, 1) << x, y, z);
        cv::Mat Pc1 = R1 * PwMat + t1;
        cv::Mat Pc2 = R2 * PwMat + t2;
        if (Pc1.at<double>(2, 0) <= 0.0 || Pc2.at<double>(2, 0) <= 0.0) {
            ++rejectedBehind;
            continue;
        }

        // ── 三角化角度过滤（COLMAP min_angle = 1.5°）────────────────────
        const double angleDeg = triangulationAngleDeg(Pw, C1, C2);
        if (angleDeg < m_options.minTriAngleDeg) {
            ++rejectedAngle;
            continue;
        }

        // ── 重投影误差过滤 ──────────────────────────────────────────────
        const double e1 = reprojectionError(Pw, R1, t1, K1, pts1[i]);
        const double e2 = reprojectionError(Pw, R2, t2, K2, pts2[i]);
        if (e1 < 0.0 || e2 < 0.0) {
            ++rejectedBehind;
            continue;
        }
        if (std::max(e1, e2) > m_options.maxReprojErrorPx) {
            ++rejectedReproj;
            continue;
        }

        if (outKeepMask) (*outKeepMask)[i] = true;
        outWorldPoints.emplace_back(x, y, z);
    }

    qDebug().noquote()
        << QString("[三角测量] pair #%1↔#%2: 输入=%3 → 输出=%4 "
                   "(剔除: 后方=%5, 角度=%6, 重投影=%7)")
               .arg(pair.imageId1 + 1).arg(pair.imageId2 + 1)
               .arg(pts1.size()).arg(outWorldPoints.size())
               .arg(rejectedBehind).arg(rejectedAngle).arg(rejectedReproj);
}

bool Triangulate::resection(const cv::Mat &K,
                            const std::vector<cv::Point2d> &imgPts,
                            const std::vector<cv::Point3d> &pts3d,
                            cv::Mat &rvec, cv::Mat &tvec,
                            std::vector<uchar> *outInlierMask,
                            int minNumInliers,
                            double minInlierRatio,
                            double maxReprojErrorPx)
{
    const size_t n = imgPts.size();
    if (n != pts3d.size() || n < 4) {
        return false;
    }
    if (K.empty()) {
        return false;
    }

    const cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    const float ransacErr = static_cast<float>(std::max(1.0, maxReprojErrorPx));

    std::vector<int> inliers;
    cv::Mat rvecRansac, tvecRansac;

    // ── 1. RANSAC 粗估计（ITERATIVE；失败回退 EPnP，对退化场景更鲁棒）──
    // 注意：不启用 useExtrinsicGuess，避免传入尺寸/语义不符的初值导致未定义行为。
    bool ok = cv::solvePnPRansac(pts3d, imgPts, K, distCoeffs,
                                 rvecRansac, tvecRansac,
                                 /*useExtrinsicGuess=*/false,
                                 /*iterationsCount=*/1000,
                                 ransacErr,
                                 /*confidence=*/0.99,
                                 inliers,
                                 cv::SOLVEPNP_ITERATIVE);
    if (!ok) {
        inliers.clear();
        ok = cv::solvePnPRansac(pts3d, imgPts, K, distCoeffs,
                                rvecRansac, tvecRansac,
                                false, 1000, ransacErr, 0.99,
                                inliers,
                                cv::SOLVEPNP_EPNP);
    }
    if (!ok || inliers.empty()) {
        if (outInlierMask) outInlierMask->assign(n, 0);
        return false;
    }

    // ── 2. 用内点集重新精化位姿 ──
    std::vector<cv::Point3d> inPts3d;
    std::vector<cv::Point2d> inPts2d;
    inPts3d.reserve(inliers.size());
    inPts2d.reserve(inliers.size());
    for (int idx : inliers) {
        inPts3d.push_back(pts3d[static_cast<size_t>(idx)]);
        inPts2d.push_back(imgPts[static_cast<size_t>(idx)]);
    }

    cv::Mat rvecRefined = rvecRansac.clone();
    cv::Mat tvecRefined = tvecRansac.clone();
    if (inPts3d.size() >= 4) {
        cv::solvePnP(inPts3d, inPts2d, K, distCoeffs,
                     rvecRefined, tvecRefined,
                     /*useExtrinsicGuess=*/true,
                     cv::SOLVEPNP_ITERATIVE);
    }

    // ── 3. 用精化后的位姿重算重投影误差，重新判定内点 ──
    cv::Mat Rref;
    cv::Rodrigues(rvecRefined, Rref);

    std::vector<double> errors;
    errors.reserve(n);
    std::vector<uchar> mask(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const cv::Vec3d Pw(pts3d[i].x, pts3d[i].y, pts3d[i].z);
        const double e = reprojectionError(Pw, Rref, tvecRefined, K, imgPts[i]);
        if (e >= 0.0 && e <= maxReprojErrorPx) {
            mask[i] = 1;
            errors.push_back(e);
        }
    }

    const int numInliers = static_cast<int>(errors.size());
    const double inlierRatio = static_cast<double>(numInliers) / static_cast<double>(n);
    const double medianErr = medianOf(errors);

    rvec = rvecRefined.clone();
    tvec = tvecRefined.clone();
    if (outInlierMask) *outInlierMask = std::move(mask);

    // ── 4. 质量校验（COLMAP abs_pose_* 三阈值）──
    if (numInliers < minNumInliers) {
        qDebug().noquote()
            << QString("[PnP] 拒绝: 内点数不足 %1/%2 (需≥%3)")
                   .arg(numInliers).arg(n).arg(minNumInliers);
        return false;
    }
    if (inlierRatio < minInlierRatio) {
        qDebug().noquote()
            << QString("[PnP] 拒绝: 内点率过低 %1 (%2/%3, 需≥%4)")
                   .arg(inlierRatio, 0, 'f', 3)
                   .arg(numInliers).arg(n)
                   .arg(minInlierRatio, 0, 'f', 3);
        return false;
    }
    if (medianErr > maxReprojErrorPx) {
        qDebug().noquote()
            << QString("[PnP] 拒绝: 中位重投影误差过大 %1 px (需≤%2)")
                   .arg(medianErr, 0, 'f', 3)
                   .arg(maxReprojErrorPx, 0, 'f', 3);
        return false;
    }

    qDebug().noquote()
        << QString("[PnP] 通过: 内点 %1/%2 (率 %3), 中位误差 %4 px")
               .arg(numInliers).arg(n)
               .arg(inlierRatio, 0, 'f', 3)
               .arg(medianErr, 0, 'f', 3);
    return true;
}

int Triangulate::numVisiblePoints3D(
    int imageId,
    const std::vector<std::unordered_map<int, int>> &kpTo3D,
    const std::set<int> &registeredCameras,
    const std::function<const MatchPair *(int, int)> &findMatchPair) const
{
    std::set<int> seenPointIds;

    for (int regId : registeredCameras) {
        if (regId == imageId) continue;
        if (regId < 0 || regId >= static_cast<int>(kpTo3D.size())) continue;

        const MatchPair *mp = findMatchPair(regId, imageId);
        if (!mp) continue;

        // 提取 (imageId 的关键点索引, regId 的关键点索引)
        const bool imageIdIsFirst = (mp->imageId1 == imageId);
        for (const auto &m : mp->matches) {
            const int kptReg = imageIdIsFirst ? m.trainIdx : m.queryIdx;
            const auto it = kpTo3D[regId].find(kptReg);
            if (it != kpTo3D[regId].end()) {
                seenPointIds.insert(it->second);
            }
        }
    }

    return static_cast<int>(seenPointIds.size());
}

/**
 * @brief 执行增量式三角测量重建（主入口）
 *
 * 完整流程（阶段一改造后）：
 * 1. 使用 initResult 作为种子对，注册两张初始相机（世界坐标系 + 相对位姿）
 * 2. 候选排序：收集跨越已注册/未注册边界的边，按「可见三维点数」降序排列
 *    （对应 COLMAP FindNextImages / RankNextImageMaxVisiblePointsNum）；
 *    已失败过的边排到第二梯队，实现降级选择而非终止
 * 3. 尝试用 bestReg ↔ secondReg 三角化三视图共有点，产出本轮候选新点
 *    （共有点不足仅跳过新点三角化，**不再终止重建**）
 * 4. 跨**所有**已注册图像收集 nextImage 的 2D-3D 对应（含历史点与本轮新点）
 * 5. PnP 后方交会 + 质量校验 → 注册新相机
 * 6. 提交新点与观测，更新 kpTo3D，执行局部 BA
 * 7. 重复步骤 2-6，直到无可用候选（任一图像失败只记一次失败并换下一个候选）
 * 8. 全局 BA + 自标定精化（固定种子相机而非 index 0）
 */
bool Triangulate::triangulate(const std::vector<MatchPair> &matchPairs,
                               const std::vector<FeatureResult> &features,
                               const InitResult &initResult,
                               const ViewGraph &viewGraph,
                               std::vector<Point3D> &outPoints3D,
                               std::vector<CameraPose> &outCameraPoses,
                               bool fixIntrinsics,
                               const cv::Mat &externalK)
{
    outPoints3D.clear();
    outCameraPoses.clear();
    m_lastEmittedPointCount = 0;

    const int numImages = static_cast<int>(features.size());
    if (matchPairs.empty() || numImages < 3) {
        return false;
    }

    // 优先使用调用方提供的内参，保证与初始化阶段（findInitialPair）使用的 K 一致
    cv::Mat K = externalK.empty() ? estimateK(features) : externalK.clone();

    // ── 初始化所有相机的内参和默认外参 ──
    outCameraPoses.resize(numImages);
    for (int c = 0; c < numImages; ++c) {
        outCameraPoses[c].imageId = c;
        outCameraPoses[c].imageWidth = features[c].imageWidth;
        outCameraPoses[c].imageHeight = features[c].imageHeight;
        outCameraPoses[c].K = K.clone();
        outCameraPoses[c].distortion = cv::Mat::zeros(5, 1, CV_64F);
    }

    // ── 种子对：第一张图像设为世界坐标系原点，第二张使用 initResult 的 R、t ──
    if (initResult.imageId1 < 0 || initResult.imageId1 >= numImages ||
        initResult.imageId2 < 0 || initResult.imageId2 >= numImages ||
        initResult.R.empty() || initResult.t.empty()) {
        qDebug().noquote() << "[三角测量] 初始化结果无效";
        return false;
    }

    const int seedId1 = initResult.imageId1;
    const int seedId2 = initResult.imageId2;

    outCameraPoses[seedId1].R = cv::Mat::eye(3, 3, CV_64F);
    outCameraPoses[seedId1].t = cv::Mat::zeros(3, 1, CV_64F);
    outCameraPoses[seedId1].registered = true;
    outCameraPoses[seedId2].R = initResult.R.clone();
    outCameraPoses[seedId2].t = initResult.t.clone();
    outCameraPoses[seedId2].registered = true;

    std::set<int> registeredCameras;
    registeredCameras.insert(seedId1);
    registeredCameras.insert(seedId2);

    // ── BA 观测数据：(imageId, point3DIdx, pixelCoord) ──
    std::vector<std::tuple<int, int, cv::Point2d>> observations;

    // ── 关键点索引 → 三维点索引（按图像下标），阶段二 Track 管理的雏形 ──
    std::vector<std::unordered_map<int, int>> kpTo3D(
        static_cast<size_t>(numImages));

    // ── 匹配对查找表：O(1)  unordered (min,max) → MatchPair* ──
    std::unordered_map<long long, const MatchPair *> pairLookup;
    pairLookup.reserve(matchPairs.size() * 2);
    for (const auto &mp : matchPairs) {
        const int a = std::min(mp.imageId1, mp.imageId2);
        const int b = std::max(mp.imageId1, mp.imageId2);
        pairLookup[(static_cast<long long>(a) << 32) | static_cast<unsigned int>(b)] = &mp;
    }

    auto findMatchPair = [&pairLookup](int idA, int idB) -> const MatchPair * {
        const int a = std::min(idA, idB);
        const int b = std::max(idA, idB);
        const auto it = pairLookup.find((static_cast<long long>(a) << 32) |
                                        static_cast<unsigned int>(b));
        return it == pairLookup.end() ? nullptr : it->second;
    };

    // ── 辅助：从匹配对中提取指定图像的关键点索引 ──
    //  返回 (该图像的kpt索引, 另一张图像的kpt索引)
    auto getKptIndices = [](const MatchPair &mp, int imageId) {
        std::vector<std::pair<int, int>> result;
        if (mp.imageId1 == imageId) {
            for (const auto &m : mp.matches)
                result.emplace_back(m.queryIdx, m.trainIdx);
        } else {
            for (const auto &m : mp.matches)
                result.emplace_back(m.trainIdx, m.queryIdx);
        }
        return result;
    };

    // ── 失败计数：(待注册图像, 已注册图像) → 已尝试次数 ──
    std::map<std::pair<int, int>, int> failedCount;

    struct EdgeCand {
        int unreg = -1;
        int reg = -1;
        int matchCount = 0;
        int visible = 0;
    };

    // ── 收集跨边界候选边并按优先级排序 ──
    //  第一梯队：未失败过的边，按可见三维点数降序（点数相同则按匹配数降序）
    //  第二梯队：失败过但未达上限的边 —— 实现降级重试而非终止重建
    auto collectCandidates = [&]() -> std::vector<EdgeCand> {
        std::vector<EdgeCand> raw;
        for (const auto &edge : viewGraph.edges) {
            const bool aReg = registeredCameras.count(edge.nodeA) != 0;
            const bool bReg = registeredCameras.count(edge.nodeB) != 0;
            if (aReg == bReg) continue;

            const int regNode = aReg ? edge.nodeA : edge.nodeB;
            const int unregNode = aReg ? edge.nodeB : edge.nodeA;

            const auto it = failedCount.find({unregNode, regNode});
            if (it != failedCount.end() && it->second >= m_options.maxRegTrials) continue;

            raw.push_back({unregNode, regNode, edge.matchCount, 0});
        }

        // 同一待注册图像只保留匹配数最大的边
        std::sort(raw.begin(), raw.end(),
                  [](const EdgeCand &a, const EdgeCand &b) {
                      return a.matchCount > b.matchCount;
                  });
        std::vector<EdgeCand> uniq;
        std::set<int> seenUnreg;
        for (const auto &c : raw) {
            if (seenUnreg.insert(c.unreg).second) uniq.push_back(c);
        }

        for (auto &c : uniq) {
            c.visible = numVisiblePoints3D(c.unreg, kpTo3D, registeredCameras, findMatchPair);
        }

        std::vector<EdgeCand> fresh, retried;
        for (const auto &c : uniq) {
            const auto it = failedCount.find({c.unreg, c.reg});
            const bool alreadyFailed = (it != failedCount.end() && it->second > 0);
            (alreadyFailed ? retried : fresh).push_back(c);
        }

        const auto byVisible = [](const EdgeCand &a, const EdgeCand &b) {
            if (a.visible != b.visible) return a.visible > b.visible;
            return a.matchCount > b.matchCount;
        };
        std::sort(fresh.begin(), fresh.end(), byVisible);
        std::sort(retried.begin(), retried.end(), byVisible);

        fresh.insert(fresh.end(), retried.begin(), retried.end());
        return fresh;
    };

    // ═══════════════════════════════════════════════════════════════════════
    //  增量注册主循环
    // ═══════════════════════════════════════════════════════════════════════
    while (true) {
        const std::vector<EdgeCand> candidates = collectCandidates();
        if (candidates.empty()) {
            qDebug().noquote() << "[三角测量] 无更多可注册图像";
            break;
        }

        const EdgeCand &cand = candidates.front();
        const int nextImage = cand.unreg;
        const int bestReg   = cand.reg;

        qDebug().noquote()
            << QString("[三角测量] 选择图像 #%1 (bestReg=#%2, 可见三维点=%3, 匹配数=%4)")
                   .arg(nextImage).arg(bestReg).arg(cand.visible).arg(cand.matchCount);

        // ── 尝试三角化新点：bestReg ↔ secondReg 的三视图共有点 ──
        //  这是"新增三维点"的途径；共有点不足时仅跳过，仍可用历史点注册本图。
        std::vector<cv::Point3d> newPt3d;
        std::vector<int> newPtKptBest;    // 新点在 bestReg 上的关键点索引
        std::vector<int> newPtKptSecond;  // 新点在 secondReg 上的关键点索引
        std::vector<int> newPtKptNext;    // 新点在 nextImage 上的关键点索引
        int secondReg = -1;

        {
            int secondRegMatchCount = 0;
            for (int regId : registeredCameras) {
                if (regId == bestReg) continue;
                const MatchPair *mp = findMatchPair(regId, bestReg);
                if (mp && mp->matchCount > secondRegMatchCount) {
                    secondRegMatchCount = mp->matchCount;
                    secondReg = regId;
                }
            }
        }

        const MatchPair *mpReg = (secondReg >= 0) ? findMatchPair(bestReg, secondReg) : nullptr;
        const MatchPair *mpNew = findMatchPair(bestReg, nextImage);

        if (mpReg && mpNew) {
            const auto indicesReg = getKptIndices(*mpReg, bestReg);
            const auto indicesNew = getKptIndices(*mpNew, bestReg);

            std::unordered_map<int, int> regLookup;
            regLookup.reserve(indicesReg.size());
            for (const auto &kv : indicesReg) regLookup[kv.first] = kv.second;

            for (const auto &kv : indicesNew) {
                const auto it = regLookup.find(kv.first);
                if (it == regLookup.end()) continue;
                newPtKptBest.push_back(kv.first);    // bestReg
                newPtKptSecond.push_back(it->second); // secondReg
                newPtKptNext.push_back(kv.second);    // nextImage
            }
        }

        const int commonCount = static_cast<int>(newPtKptBest.size());
        if (commonCount < m_options.minCommonPoints) {
            qDebug().noquote()
                << QString("[三角测量] 图像 #%1 三视图共有点=%2 (<%3)，跳过新点三角化，"
                           "改用历史三维点注册")
                       .arg(nextImage).arg(commonCount).arg(m_options.minCommonPoints);
            newPtKptBest.clear();
            newPtKptSecond.clear();
            newPtKptNext.clear();
        } else {
            const cv::Mat R1 = outCameraPoses[bestReg].R.clone();
            const cv::Mat t1 = outCameraPoses[bestReg].t.clone();
            const cv::Mat R2 = outCameraPoses[secondReg].R.clone();
            const cv::Mat t2 = outCameraPoses[secondReg].t.clone();

            MatchPair tempPair;
            tempPair.imageId1 = bestReg;
            tempPair.imageId2 = secondReg;
            for (int idx : newPtKptBest)
                tempPair.points1.push_back(features[bestReg].keypoints[idx].pt);
            for (int idx : newPtKptSecond)
                tempPair.points2.push_back(features[secondReg].keypoints[idx].pt);
            for (int k = 0; k < commonCount; ++k)
                tempPair.matches.push_back(cv::DMatch(k, k, 0));

            std::vector<bool> keepMask;
            triangulatePair(tempPair, K, K, R1, t1, R2, t2, newPt3d, &keepMask);

            const auto filterByMask = [&keepMask](std::vector<int> &vec) {
                std::vector<int> filtered;
                for (size_t k = 0; k < vec.size() && k < keepMask.size(); ++k) {
                    if (keepMask[k]) filtered.push_back(vec[k]);
                }
                vec = std::move(filtered);
            };
            filterByMask(newPtKptBest);
            filterByMask(newPtKptSecond);
            filterByMask(newPtKptNext);
        }

        // ── 暂存本轮新点（PnP 失败时回滚）──
        const int basePointIdx = static_cast<int>(outPoints3D.size());
        const int baseObservationIdx = static_cast<int>(observations.size());
        const cv::Mat &bestImg = features[bestReg].imageData;
        for (size_t i = 0; i < newPt3d.size(); ++i) {
            Point3D pt;
            pt.id = static_cast<int>(outPoints3D.size());
            pt.position = newPt3d[i];
            pt.color = cv::Vec3b(128, 128, 128);
            if (i < newPtKptBest.size() && !bestImg.empty()) {
                const cv::KeyPoint &kp = features[bestReg].keypoints[newPtKptBest[i]];
                const int px = static_cast<int>(kp.pt.x + 0.5);
                const int py = static_cast<int>(kp.pt.y + 0.5);
                if (py >= 0 && py < bestImg.rows && px >= 0 && px < bestImg.cols) {
                    pt.color = bestImg.at<cv::Vec3b>(py, px);
                }
            }
            outPoints3D.push_back(pt);
        }

        // ── 把新点登记进 kpTo3D（三张图上的关键点都登记，供后续图像复用）──
        std::vector<std::pair<int, int>> addedKpTo3D;
        const auto stageKpTo3D = [&](int img, int kpt, int ptIdx) {
            kpTo3D[img][kpt] = ptIdx;
            addedKpTo3D.emplace_back(img, kpt);
        };
        for (size_t i = 0; i < newPt3d.size(); ++i) {
            const int ptIdx = basePointIdx + static_cast<int>(i);
            if (i < newPtKptBest.size())   stageKpTo3D(bestReg,   newPtKptBest[i],   ptIdx);
            if (i < newPtKptSecond.size()) stageKpTo3D(secondReg, newPtKptSecond[i], ptIdx);
            if (i < newPtKptNext.size())   stageKpTo3D(nextImage, newPtKptNext[i],   ptIdx);
        }

        const auto rollback = [&]() {
            outPoints3D.resize(static_cast<size_t>(basePointIdx));
            observations.resize(static_cast<size_t>(baseObservationIdx));
            for (const auto &kv : addedKpTo3D) kpTo3D[kv.first].erase(kv.second);
            addedKpTo3D.clear();
        };

        // ── 跨所有已注册图像收集 nextImage 的 2D-3D 对应 ──
        //  对应 COLMAP RegisterNextImage：不再局限于本轮新算的点，
        //  历史已三角化点同样可用于注册，显著提升大视角数据集的注册率。
        std::vector<cv::Point2d> corr2d;
        std::vector<int> corrPtIdx;
        {
            std::set<int> usedPt;
            for (int regId : registeredCameras) {
                if (regId == nextImage) continue;
                const MatchPair *mp = findMatchPair(regId, nextImage);
                if (!mp) continue;
                for (const auto &kv : getKptIndices(*mp, nextImage)) {
                    const int kptNext = kv.first;
                    const int kptReg  = kv.second;
                    const auto it = kpTo3D[regId].find(kptReg);
                    if (it == kpTo3D[regId].end()) continue;
                    if (!usedPt.insert(it->second).second) continue;
                    corr2d.push_back(features[nextImage].keypoints[kptNext].pt);
                    corrPtIdx.push_back(it->second);
                }
            }
        }

        if (static_cast<int>(corr2d.size()) < m_options.absPoseMinNumInliers) {
            qDebug().noquote()
                << QString("[三角测量] 图像 #%1 可用 2D-3D 对应=%2 (<%3)，标记该边失败并换候选")
                       .arg(nextImage).arg(corr2d.size())
                       .arg(m_options.absPoseMinNumInliers);
            rollback();
            failedCount[{nextImage, bestReg}] += 1;
            continue;
        }

        std::vector<cv::Point3d> corr3d;
        corr3d.reserve(corrPtIdx.size());
        for (int pi : corrPtIdx) corr3d.push_back(outPoints3D[pi].position);

        // ── PnP 后方交会注册新相机 ──
        cv::Mat rvec, tvec;
        std::vector<uchar> inlierMask;
        const bool poseOk = resection(outCameraPoses[nextImage].K, corr2d, corr3d,
                                      rvec, tvec, &inlierMask,
                                      m_options.absPoseMinNumInliers,
                                      m_options.absPoseMinInlierRatio,
                                      m_options.absPoseMaxErrorPx);

        if (!poseOk) {
            qDebug().noquote()
                << QString("[三角测量] PnP 注册相机 #%1 失败 (bestReg=#%2)，换次优候选")
                       .arg(nextImage).arg(bestReg);
            rollback();
            failedCount[{nextImage, bestReg}] += 1;
            continue;
        }

        cv::Mat Rnew;
        cv::Rodrigues(rvec, Rnew);
        outCameraPoses[nextImage].R = Rnew.clone();
        outCameraPoses[nextImage].t = tvec.clone();
        outCameraPoses[nextImage].registered = true;
        registeredCameras.insert(nextImage);
        addedKpTo3D.clear();  // 提交：新点的 kpTo3D 记录保留

        // 注册成功：清除该图像作为"待注册端"的失败记录
        for (auto it = failedCount.begin(); it != failedCount.end(); ) {
            it = (it->first.first == nextImage) ? failedCount.erase(it) : std::next(it);
        }

        // ── 记录观测 ──
        //  1) 新点在 bestReg / secondReg 上的观测
        for (size_t i = 0; i < newPt3d.size(); ++i) {
            const int ptIdx = basePointIdx + static_cast<int>(i);
            if (i < newPtKptBest.size()) {
                const auto &kp = features[bestReg].keypoints[newPtKptBest[i]];
                observations.emplace_back(bestReg, ptIdx, cv::Point2d(kp.pt.x, kp.pt.y));
            }
            if (secondReg >= 0 && i < newPtKptSecond.size()) {
                const auto &kp = features[secondReg].keypoints[newPtKptSecond[i]];
                observations.emplace_back(secondReg, ptIdx, cv::Point2d(kp.pt.x, kp.pt.y));
            }
        }
        //  2) nextImage 对所有内点（含历史点）的观测
        for (size_t k = 0; k < corr2d.size(); ++k) {
            if (k >= inlierMask.size() || !inlierMask[k]) continue;
            observations.emplace_back(nextImage, corrPtIdx[k], corr2d[k]);
        }

        qDebug().noquote()
            << QString("[三角测量] PnP 注册相机 #%1 成功（2D-3D 对应=%2, 本轮新点=%3）")
                   .arg(nextImage).arg(corr2d.size()).arg(newPt3d.size());

        // ── 局部 BA：优化新相机 + 本轮新点（其余固定）──
        {
            std::vector<std::array<double, 6>> cameras(static_cast<size_t>(numImages));
            for (int c = 0; c < numImages; ++c) {
                cv::Mat rvecC;
                const cv::Mat Rc = outCameraPoses[c].R.empty()
                                       ? cv::Mat::eye(3, 3, CV_64F) : outCameraPoses[c].R;
                const cv::Mat tc = outCameraPoses[c].t.empty()
                                       ? cv::Mat::zeros(3, 1, CV_64F) : outCameraPoses[c].t;
                cv::Rodrigues(Rc, rvecC);
                cameras[c] = {rvecC.at<double>(0), rvecC.at<double>(1), rvecC.at<double>(2),
                              tc.at<double>(0), tc.at<double>(1), tc.at<double>(2)};
            }

            std::vector<std::array<double, 3>> ptsGlobal;
            ptsGlobal.reserve(outPoints3D.size());
            for (const auto &p : outPoints3D) {
                ptsGlobal.push_back({p.position.x, p.position.y, p.position.z});
            }

            BundleAdjustment::instance()->bundleAdjustment(
                cameras, ptsGlobal, observations,
                K.at<double>(0, 0), K.at<double>(1, 1),
                K.at<double>(0, 2), K.at<double>(1, 2),
                nextImage, basePointIdx);

            cv::Mat rvecUpdated = (cv::Mat_<double>(3, 1) <<
                cameras[nextImage][0], cameras[nextImage][1], cameras[nextImage][2]);
            cv::Mat Rj;
            cv::Rodrigues(rvecUpdated, Rj);
            outCameraPoses[nextImage].R = Rj.clone();
            outCameraPoses[nextImage].t = (cv::Mat_<double>(3, 1) <<
                cameras[nextImage][3], cameras[nextImage][4], cameras[nextImage][5]);

            for (size_t k = 0; k < ptsGlobal.size(); ++k) {
                outPoints3D[k].position = cv::Point3d(
                    ptsGlobal[k][0], ptsGlobal[k][1], ptsGlobal[k][2]);
            }
        }

        // ── 实时推送增量更新（只发新增点）──
        {
            QVector<float> pts, cls;
            const int startIdx = m_lastEmittedPointCount;
            pts.reserve(static_cast<int>(outPoints3D.size() - startIdx) * 3);
            cls.reserve(static_cast<int>(outPoints3D.size() - startIdx) * 3);
            for (size_t i = static_cast<size_t>(startIdx); i < outPoints3D.size(); ++i) {
                const auto &p = outPoints3D[i];
                pts << static_cast<float>(p.position.x)
                    << static_cast<float>(p.position.y)
                    << static_cast<float>(p.position.z);
                cls << p.color[2] / 255.0f
                    << p.color[1] / 255.0f
                    << p.color[0] / 255.0f;
            }
            m_lastEmittedPointCount = static_cast<int>(outPoints3D.size());

            QVector<float> camData;
            camData.reserve(registeredCameras.size() * 6);
            for (int camId : registeredCameras) {
                const cv::Mat C = -outCameraPoses[camId].R.t() * outCameraPoses[camId].t;
                const cv::Mat fwd = outCameraPoses[camId].R.row(2).t();
                camData << static_cast<float>(C.at<double>(0))
                        << static_cast<float>(C.at<double>(1))
                        << static_cast<float>(C.at<double>(2))
                        << static_cast<float>(fwd.at<double>(0))
                        << static_cast<float>(fwd.at<double>(1))
                        << static_cast<float>(fwd.at<double>(2));
            }
            emit incrementalUpdate(pts, cls, camData, startIdx == 0);
        }
    }

    if (outPoints3D.empty() || registeredCameras.size() < 3) {
        qDebug().noquote()
            << QString("[三角测量] 重建失败: 三维点=%1, 已注册相机=%2")
                   .arg(outPoints3D.size()).arg(registeredCameras.size());
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  全局 BA + 自标定精化
    //  固定种子图像（世界坐标系参考）而非 index 0；未注册相机不参与优化
    // ═══════════════════════════════════════════════════════════════════════
    {
        std::vector<std::array<double, 6>> cameras(static_cast<size_t>(numImages));
        std::vector<char> cameraVariable(static_cast<size_t>(numImages), 0);
        for (int c = 0; c < numImages; ++c) {
            cv::Mat rvecC;
            const cv::Mat Rc = outCameraPoses[c].R.empty()
                                   ? cv::Mat::eye(3, 3, CV_64F) : outCameraPoses[c].R;
            const cv::Mat tc = outCameraPoses[c].t.empty()
                                   ? cv::Mat::zeros(3, 1, CV_64F) : outCameraPoses[c].t;
            cv::Rodrigues(Rc, rvecC);
            cameras[c] = {rvecC.at<double>(0), rvecC.at<double>(1), rvecC.at<double>(2),
                          tc.at<double>(0), tc.at<double>(1), tc.at<double>(2)};
            cameraVariable[c] = outCameraPoses[c].registered ? 1 : 0;
        }

        std::vector<std::array<double, 3>> ptsGlobal;
        ptsGlobal.reserve(outPoints3D.size());
        for (const auto &p : outPoints3D) {
            ptsGlobal.push_back({p.position.x, p.position.y, p.position.z});
        }

        qDebug().noquote()
            << QString("[三角测量] 全局BA+自标定: %1 相机, %2 点")
                   .arg(registeredCameras.size())
                   .arg(outPoints3D.size());

        double fx = K.at<double>(0, 0);
        double fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);

        BundleAdjustment::instance()->globalBundleAdjustment(
            cameras, ptsGlobal, observations,
            fx, fy, cx, cy, fixIntrinsics,
            seedId1, &cameraVariable);

        for (int c = 0; c < numImages; ++c) {
            if (!cameraVariable[c]) continue;
            const cv::Mat rvecC = (cv::Mat_<double>(3, 1) <<
                                       cameras[c][0], cameras[c][1], cameras[c][2]);
            cv::Mat Rc;
            cv::Rodrigues(rvecC, Rc);
            outCameraPoses[c].R = Rc.clone();
            outCameraPoses[c].t = (cv::Mat_<double>(3, 1) <<
                                       cameras[c][3], cameras[c][4], cameras[c][5]);
        }

        for (size_t k = 0; k < ptsGlobal.size(); ++k) {
            outPoints3D[k].position = cv::Point3d(
                ptsGlobal[k][0], ptsGlobal[k][1], ptsGlobal[k][2]);
        }

        K.at<double>(0, 0) = fx;
        K.at<double>(1, 1) = fy;
        K.at<double>(0, 2) = cx;
        K.at<double>(1, 2) = cy;
        for (int c = 0; c < numImages; ++c) {
            outCameraPoses[c].K = K.clone();
        }

        // ── 全局BA完成后推送全量更新 ──
        QVector<float> pts, cls;
        pts.reserve(static_cast<int>(outPoints3D.size()) * 3);
        cls.reserve(static_cast<int>(outPoints3D.size()) * 3);
        for (const auto &p : outPoints3D) {
            pts << static_cast<float>(p.position.x)
                << static_cast<float>(p.position.y)
                << static_cast<float>(p.position.z);
            cls << p.color[2] / 255.0f
                << p.color[1] / 255.0f
                << p.color[0] / 255.0f;
        }
        QVector<float> camData;
        camData.reserve(registeredCameras.size() * 6);
        for (int camId : registeredCameras) {
            const cv::Mat C = -outCameraPoses[camId].R.t() * outCameraPoses[camId].t;
            const cv::Mat fwd = outCameraPoses[camId].R.row(2).t();
            camData << static_cast<float>(C.at<double>(0))
                    << static_cast<float>(C.at<double>(1))
                    << static_cast<float>(C.at<double>(2))
                    << static_cast<float>(fwd.at<double>(0))
                    << static_cast<float>(fwd.at<double>(1))
                    << static_cast<float>(fwd.at<double>(2));
        }
        m_lastEmittedPointCount = static_cast<int>(outPoints3D.size());
        emit incrementalUpdate(pts, cls, camData, true);
    }

    qDebug().noquote()
        << QString("[三角测量] 重建完成: 已注册相机=%1/%2, 三维点总数=%3")
               .arg(registeredCameras.size()).arg(numImages).arg(outPoints3D.size());

    return true;
}

