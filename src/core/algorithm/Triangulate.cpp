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
#include "SfmGeometry.h"

#include <QtMath>
#include <QVector>
#include <QDebug>
#include <algorithm>
#include <chrono>
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
    const cv::Vec3d C1 = SfmGeom::cameraCenter(R1, t1);
    const cv::Vec3d C2 = SfmGeom::cameraCenter(R2, t2);

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
        const double angleDeg = SfmGeom::triangulationAngleDeg(Pw, C1, C2);
        if (angleDeg < m_options.minTriAngleDeg) {
            ++rejectedAngle;
            continue;
        }

        // ── 重投影误差过滤 ──────────────────────────────────────────────
        const double e1 = SfmGeom::reprojectionError(Pw, R1, t1, K1, pts1[i]);
        const double e2 = SfmGeom::reprojectionError(Pw, R2, t2, K2, pts2[i]);
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
                            double maxReprojErrorPx,
                            const cv::Mat &dist)
{
    const size_t n = imgPts.size();
    if (n != pts3d.size() || n < 4) {
        return false;
    }
    if (K.empty()) {
        return false;
    }

    // 阶段五 5.2：畸变一旦在全局BA中被标定出来，PnP 就必须按带畸变模型求解，
    // 否则新相机的位姿会被系统性拉偏（OpenCV 直接支持畸变系数）。
    const cv::Mat distCoeffs = dist.empty() ? cv::Mat::zeros(5, 1, CV_64F)
                                            : dist.clone();
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
        const double e = SfmGeom::reprojectionError(Pw, Rref, tvecRefined, K, imgPts[i]);
        if (e >= 0.0 && e <= maxReprojErrorPx) {
            mask[i] = 1;
            errors.push_back(e);
        }
    }

    const int numInliers = static_cast<int>(errors.size());
    const double inlierRatio = static_cast<double>(numInliers) / static_cast<double>(n);
    const double medianErr = SfmGeom::medianOf(errors);

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

// ═════════════════════════════════════════════════════════════════════════════
//  阶段二：COLMAP 式增量重建
// ═════════════════════════════════════════════════════════════════════════════

std::vector<std::pair<int, int>> Triangulate::findNextImages() const
{
    std::vector<std::pair<int, int>> ranked;  // (可见三维点数, 图像ID)

    const int numImages = static_cast<int>(m_features->size());
    for (int i = 0; i < numImages; ++i) {
        if ((*m_poses)[static_cast<size_t>(i)].registered) continue;
        if (m_numRegTrials[static_cast<size_t>(i)] >= m_options.maxRegTrials) continue;

        const int visible = static_cast<int>(m_obs->numVisiblePoints3D(i));
        ranked.emplace_back(visible, i);
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });

    return ranked;
}

bool Triangulate::registerNextImage(int imageId)
{
    const std::chrono::steady_clock::time_point regT0 = std::chrono::steady_clock::now();
    std::vector<int> kptIdx, pointId;
    m_obs->collect2D3DCorrespondences(imageId, kptIdx, pointId);

    if (static_cast<int>(kptIdx.size()) < m_options.absPoseMinNumInliers) {
        qDebug().noquote()
            << QString("[注册] 图像 #%1 可用 2D-3D 对应=%2 (<%3)，跳过")
                   .arg(imageId).arg(kptIdx.size())
                   .arg(m_options.absPoseMinNumInliers);
        return false;
    }

    std::vector<cv::Point2d> pts2d;
    std::vector<cv::Point3d> pts3d;
    pts2d.reserve(kptIdx.size());
    pts3d.reserve(kptIdx.size());
    for (size_t i = 0; i < kptIdx.size(); ++i) {
        if (!m_obs->exists(pointId[i])) continue;
        pts2d.push_back(m_obs->keypointPos(imageId, kptIdx[i]));
        pts3d.push_back(m_obs->point(pointId[i]).position);
    }

    cv::Mat rvec, tvec;
    std::vector<uchar> inlierMask;
    const bool ok = resection((*m_poses)[static_cast<size_t>(imageId)].K,
                              pts2d, pts3d, rvec, tvec, &inlierMask,
                              m_options.absPoseMinNumInliers,
                              m_options.absPoseMinInlierRatio,
                              m_options.absPoseMaxErrorPx,
                              m_dist);
    if (!ok) return false;

    cv::Mat Rnew;
    cv::Rodrigues(rvec, Rnew);
    CameraPose &pose = (*m_poses)[static_cast<size_t>(imageId)];
    pose.R = Rnew.clone();
    pose.t = tvec.clone();
    pose.registered = true;
    m_registeredCameras.insert(imageId);
    m_numRegTrials[static_cast<size_t>(imageId)] = 0;

    const double regMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - regT0).count();

    qDebug().noquote()
        << QString("[注册] 图像 #%1 注册成功（2D-3D=%2, 内点=%3），已注册 %4 台，%5 ms")
               .arg(imageId).arg(pts2d.size())
               .arg(std::count(inlierMask.begin(), inlierMask.end(), 1))
               .arg(m_registeredCameras.size())
               .arg(static_cast<qlonglong>(regMs));

    return true;
}

void Triangulate::triangulateAndRefineImage(int imageId)
{
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    const size_t numTris = m_triangulator->triangulateImage(m_triOptions, imageId);
    const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    m_timing.triangulate += std::chrono::duration<double, std::milli>(t1 - t0).count();

    qDebug().noquote()
        << QString("[三角化] 图像 #%1 新增/续接观测=%2，当前三维点=%3")
               .arg(imageId).arg(static_cast<qlonglong>(numTris))
               .arg(static_cast<qlonglong>(m_obs->numAlivePoints()));

    iterativeLocalRefinement(imageId);
    const std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    emitCurrentState(true);
    const std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    m_timing.emitMs += std::chrono::duration<double, std::milli>(t3 - t2).count();

    qDebug().noquote()
        << QString("[图像耗时] #%1 选图=%2 三角化=%3 局部精化=%4 (邻域%5 BA%6 合并%7 补全%8 补图%9 过滤%10) 推送=%11 ms")
               .arg(imageId)
               .arg(m_timing.findNext, 0, 'f', 0)
               .arg(m_timing.triangulate, 0, 'f', 0)
               .arg(std::chrono::duration<double, std::milli>(t2 - t1).count(), 0, 'f', 0)
               .arg(m_timing.neighbor, 0, 'f', 0)
               .arg(m_timing.localBa, 0, 'f', 0)
               .arg(m_timing.merge, 0, 'f', 0)
               .arg(m_timing.complete, 0, 'f', 0)
               .arg(m_timing.completeImg, 0, 'f', 0)
               .arg(m_timing.filter, 0, 'f', 0)
               .arg(m_timing.emitMs, 0, 'f', 0);
    m_timing.reset();
}

void Triangulate::iterativeLocalRefinement(int imageId)
{
    for (int iter = 0; iter < m_options.baLocalMaxRefinements; ++iter) {
        std::vector<int> variablePoints(m_triangulator->modifiedPoints().begin(),
                                        m_triangulator->modifiedPoints().end());
        if (variablePoints.empty()) break;

        // 第一轮位姿/点被明显移动，做全量过滤；后续轮 BA 仅微调，
        // 跳过全量过滤（只保留可变点的增量过滤），省 ~40% 过滤耗时
        adjustLocalBundle(imageId, variablePoints, iter == 0);
        if (m_lastLocalResiduals == 0) break;

        // 变化率 = (合并 + 补全 + 过滤) / 参与优化的残差数
        const double changed =
            static_cast<double>(m_lastLocalChanged) /
            static_cast<double>(m_lastLocalResiduals);
        if (changed < m_options.baLocalMaxRefinementChange) break;
    }
    m_triangulator->clearModified();
}

void Triangulate::findLocalBundle(int imageId, std::vector<int> &outImageIds) const
{
    outImageIds.clear();

    // 统计与 imageId 共享三维点最多的图像
    std::vector<int> observed;
    m_obs->collectObservedPoints(imageId, observed);

    std::map<int, size_t> shared;
    for (const int pid : observed) {
        if (!m_obs->exists(pid)) continue;
        for (const auto &obs : m_obs->point(pid).observations) {
            if (obs.first == imageId) continue;
            if (!(*m_poses)[static_cast<size_t>(obs.first)].registered) continue;
            shared[obs.first] += 1;
        }
    }

    std::vector<std::pair<int, size_t>> ranked(shared.begin(), shared.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<int, size_t> &a, const std::pair<int, size_t> &b) {
                  return a.second > b.second;
              });

    // 局部BA由「新图 + 最相关的 N-1 张图」组成（COLMAP ba_local_num_images）
    const size_t limit = static_cast<size_t>(std::max(1, m_options.baLocalNumImages - 1));
    for (size_t i = 0; i < ranked.size() && outImageIds.size() < limit; ++i) {
        outImageIds.push_back(ranked[i].first);
    }
    outImageIds.push_back(imageId);
}

void Triangulate::collectObservations(const std::set<int> &imageSet,
                                      const std::set<int> &pointSet,
                                      std::vector<std::tuple<int, int, cv::Point2d>> &out) const
{
    out.clear();
    const auto &points = m_obs->allPoints();
    const auto &alive = m_obs->aliveMask();

    for (size_t pid = 0; pid < points.size(); ++pid) {
        if (!alive[pid]) continue;

        const bool pointWanted = pointSet.empty() || pointSet.count(static_cast<int>(pid)) != 0;
        for (const auto &obs : points[pid].observations) {
            const bool imageWanted = imageSet.empty() || imageSet.count(obs.first) != 0;
            if (!imageWanted && !pointWanted) continue;

            out.emplace_back(obs.first, static_cast<int>(pid),
                             m_obs->keypointPos(obs.first, obs.second));
        }
    }
}

void Triangulate::buildCameraParams(std::vector<std::array<double, 6>> &out) const
{
    const size_t n = m_poses->size();
    out.assign(n, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    for (size_t i = 0; i < n; ++i) {
        const CameraPose &p = (*m_poses)[i];
        const cv::Mat R = p.R.empty() ? cv::Mat::eye(3, 3, CV_64F) : p.R;
        const cv::Mat t = p.t.empty() ? cv::Mat::zeros(3, 1, CV_64F) : p.t;

        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        out[i] = {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                  t.at<double>(0), t.at<double>(1), t.at<double>(2)};
    }
}

void Triangulate::buildPointParams(std::vector<std::array<double, 3>> &out) const
{
    const auto &points = m_obs->allPoints();
    out.assign(points.size(), {0.0, 0.0, 0.0});
    for (size_t i = 0; i < points.size(); ++i) {
        out[i] = {points[i].position.x, points[i].position.y, points[i].position.z};
    }
}

void Triangulate::applyCameraParams(const std::vector<std::array<double, 6>> &cameras,
                                    const std::vector<char> &cameraVariable)
{
    for (size_t i = 0; i < cameras.size() && i < m_poses->size(); ++i) {
        if (i < cameraVariable.size() && !cameraVariable[i]) continue;

        const cv::Mat rvec = (cv::Mat_<double>(3, 1)
                                  << cameras[i][0], cameras[i][1], cameras[i][2]);
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        (*m_poses)[i].R = R.clone();
        (*m_poses)[i].t = (cv::Mat_<double>(3, 1)
                               << cameras[i][3], cameras[i][4], cameras[i][5]);
    }
}

void Triangulate::applyPointParams(const std::vector<std::array<double, 3>> &points)
{
    for (size_t i = 0; i < points.size() && i < m_obs->allPoints().size(); ++i) {
        if (!m_obs->exists(static_cast<int>(i))) continue;
        m_obs->point(static_cast<int>(i)).position =
            cv::Point3d(points[i][0], points[i][1], points[i][2]);
    }
}

void Triangulate::adjustLocalBundle(int imageId, const std::vector<int> &variablePointIds,
                                    bool fullFilter)
{
    m_lastLocalChanged = 0;
    m_lastLocalResiduals = 0;

    std::vector<int> localImages;
    const std::chrono::steady_clock::time_point tN0 = std::chrono::steady_clock::now();
    findLocalBundle(imageId, localImages);
    m_timing.neighbor +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tN0).count();

    const std::set<int> imageSet(localImages.begin(), localImages.end());

    // COLMAP：只把「新点 / 短轨迹点」设为可变；长轨迹点已足够稳定
    std::set<int> pointSet;
    for (const int pid : variablePointIds) {
        if (!m_obs->exists(pid)) continue;
        if (m_obs->point(pid).observations.size() <=
            static_cast<size_t>(m_options.baLocalMaxTrackLength)) {
            pointSet.insert(pid);
        }
    }
    if (pointSet.empty()) return;

    std::vector<std::tuple<int, int, cv::Point2d>> observations;
    collectObservations(imageSet, pointSet, observations);
    if (observations.empty()) return;

    std::vector<std::array<double, 6>> cameras;
    std::vector<std::array<double, 3>> pointsArr;
    buildCameraParams(cameras);
    buildPointParams(pointsArr);

    std::vector<char> cameraVariable(cameras.size(), 0);
    for (const int cid : localImages) {
        if (cid >= 0 && cid < static_cast<int>(cameraVariable.size())) {
            cameraVariable[static_cast<size_t>(cid)] = 1;
        }
    }
    // 参考相机固定（世界坐标系基准），否则整个局部系统存在规范自由度
    cameraVariable[static_cast<size_t>(m_refCameraId)] = 0;

    std::vector<char> pointVariable(pointsArr.size(), 0);
    for (const int pid : pointSet) {
        pointVariable[static_cast<size_t>(pid)] = 1;
    }

    const std::chrono::steady_clock::time_point tBa = std::chrono::steady_clock::now();
    BundleAdjustment::instance()->localBundleAdjustment(
        cameras, pointsArr, observations,
        m_K.at<double>(0, 0), m_K.at<double>(1, 1),
        m_K.at<double>(0, 2), m_K.at<double>(1, 2),
        cameraVariable, pointVariable, m_refCameraId,
        m_options.baLocalMaxIterations);
    m_timing.localBa +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tBa).count();

    applyCameraParams(cameras, cameraVariable);
    applyPointParams(pointsArr);

    // ── BA 之后：合并轨迹 → 补全轨迹 → 过滤外点 ──────────────────────────
    size_t changed = observations.size();  // 参与优化的残差数作为变化率分母
    m_lastLocalResiduals = observations.size();

    if (m_options.enableTrackCompletion) {
        const auto tA = std::chrono::steady_clock::now();
        const size_t merged = m_triangulator->mergeTracks(m_triOptions, variablePointIds);
        const auto tB = std::chrono::steady_clock::now();
        const size_t completed = m_triangulator->completeTracks(m_triOptions, variablePointIds);
        const auto tC = std::chrono::steady_clock::now();
        const size_t completedImg = m_triangulator->completeImage(m_triOptions, imageId);
        const auto tD = std::chrono::steady_clock::now();

        m_timing.merge += std::chrono::duration<double, std::milli>(tB - tA).count();
        m_timing.complete += std::chrono::duration<double, std::milli>(tC - tB).count();
        m_timing.completeImg += std::chrono::duration<double, std::milli>(tD - tC).count();

        qDebug().noquote()
            << QString("[局部精化] 图像#%1: 合并=%2, 补全=%3, 补图=%4")
                   .arg(imageId)
                   .arg(static_cast<qlonglong>(merged))
                   .arg(static_cast<qlonglong>(completed))
                   .arg(static_cast<qlonglong>(completedImg));
    }

    const auto tF0 = std::chrono::steady_clock::now();
    const size_t filtered = m_obs->filterPoints3D(
        m_options.filterMaxReprojErrorPx, m_options.filterMinTriAngleDeg,
        variablePointIds, *m_poses, m_K, m_dist);
    // 全量过滤（局部图上所有观测重算重投影）只在位姿被明显移动的轮次做：
    // 迭代精化的后续轮次 BA 仅 3 次迭代、位姿微调，重复全量过滤是纯浪费
    // （data 实测过滤占总局部精化 ~40%）。收尾的 filterAllPoints3D 兜底。
    size_t filteredImg = 0;
    if (fullFilter) {
        filteredImg = m_obs->filterPoints3DInImages(
            m_options.filterMaxReprojErrorPx, m_options.filterMinTriAngleDeg,
            localImages, *m_poses, m_K, m_dist);
    }
    m_timing.filter +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tF0).count();

    m_lastLocalChanged = filtered + filteredImg;
    Q_UNUSED(changed);
}

void Triangulate::runGlobalBundleAdjustment(bool fixIntrinsics)
{
    if (m_registeredCameras.size() < 2 || m_obs->numAlivePoints() == 0) return;

    std::vector<std::tuple<int, int, cv::Point2d>> observations;
    collectObservations(std::set<int>(), std::set<int>(), observations);
    if (observations.empty()) return;

    std::vector<std::array<double, 6>> cameras;
    std::vector<std::array<double, 3>> pointsArr;
    buildCameraParams(cameras);
    buildPointParams(pointsArr);

    std::vector<char> cameraVariable(cameras.size(), 0);
    for (const int cid : m_registeredCameras) {
        cameraVariable[static_cast<size_t>(cid)] = 1;
    }

    double fx = m_K.at<double>(0, 0);
    double fy = m_K.at<double>(1, 1);
    double cx = m_K.at<double>(0, 2);
    double cy = m_K.at<double>(1, 2);

    qDebug().noquote()
        << QString("[BA全局] 相机=%1, 点=%2, 残差=%3")
               .arg(m_registeredCameras.size())
               .arg(static_cast<qlonglong>(m_obs->numAlivePoints()))
               .arg(observations.size());

    // ── 阶段五 5.2/5.3：内参精化选项 ──────────────────────────────────────
    // 种子对阶段（2 台相机）焦距与主点/场景尺度不可辨识，内参一律钉死；
    // 注册相机数达到 intrinsicsMinCameras 后才放开焦距与畸变。
    const int minCam = m_options.intrinsicsTrusted ? m_options.intrinsicsMinCameras : 0;
    const bool canRefineIntrinsics =
        static_cast<int>(m_registeredCameras.size()) >= minCam;

    BAIntrinsicsOptions intrOpts;
    intrOpts.refineFocalLength = canRefineIntrinsics;
    intrOpts.refineFocalPerAxis = m_options.refineFocalPerAxis && canRefineIntrinsics;
    intrOpts.refineDistortion = m_options.refineDistortion && canRefineIntrinsics;
    if (!m_dist.empty()) {
        intrOpts.k1 = m_dist.at<double>(0);
        intrOpts.k2 = m_dist.at<double>(1);
        intrOpts.p1 = m_dist.at<double>(2);
        intrOpts.p2 = m_dist.at<double>(3);
    }

    BundleAdjustment::instance()->globalBundleAdjustment(
        cameras, pointsArr, observations, fx, fy, cx, cy,
        fixIntrinsics, m_refCameraId, &cameraVariable,
        m_options.baGlobalMaxIterations, &intrOpts);

    applyCameraParams(cameras, cameraVariable);
    applyPointParams(pointsArr);

    m_K.at<double>(0, 0) = fx;
    m_K.at<double>(1, 1) = fy;
    m_K.at<double>(0, 2) = cx;
    m_K.at<double>(1, 2) = cy;

    // 畸变回写：参与后续过滤 / 三角化 / PnP / 统计的投影模型
    if (m_options.refineDistortion && canRefineIntrinsics) {
        m_dist.at<double>(0) = intrOpts.k1;
        m_dist.at<double>(1) = intrOpts.k2;
        m_dist.at<double>(2) = intrOpts.p1;
        m_dist.at<double>(3) = intrOpts.p2;
        m_dist.at<double>(4) = 0.0;   // k3 固定为 0（与 k1/k2 强相关）
    }

    for (auto &pose : *m_poses) {
        pose.K = m_K.clone();
        pose.distortion = m_dist.clone();
    }
    m_triangulator->setK(m_K);
    m_triangulator->setDistortion(m_dist);
}

size_t Triangulate::completeAndMergeTracks()
{
    size_t changed = 0;
    changed += m_triangulator->completeAllTracks(m_triOptions);
    changed += m_triangulator->mergeAllTracks(m_triOptions);
    return changed;
}

void Triangulate::emitCurrentState(bool reset)
{
    std::vector<Point3D> points;
    m_obs->compact(points);

    QVector<float> pts, cls;
    pts.reserve(static_cast<int>(points.size()) * 3);
    cls.reserve(static_cast<int>(points.size()) * 3);
    for (const auto &p : points) {
        pts << static_cast<float>(p.position.x)
            << static_cast<float>(p.position.y)
            << static_cast<float>(p.position.z);
        cls << p.color[2] / 255.0f
            << p.color[1] / 255.0f
            << p.color[0] / 255.0f;
    }

    QVector<float> camData;
    camData.reserve(static_cast<int>(m_registeredCameras.size()) * 6);
    for (const int camId : m_registeredCameras) {
        const CameraPose &pose = (*m_poses)[static_cast<size_t>(camId)];
        if (pose.R.empty() || pose.t.empty()) continue;
        const cv::Mat C = -pose.R.t() * pose.t;
        const cv::Mat fwd = pose.R.row(2).t();
        camData << static_cast<float>(C.at<double>(0))
                << static_cast<float>(C.at<double>(1))
                << static_cast<float>(C.at<double>(2))
                << static_cast<float>(fwd.at<double>(0))
                << static_cast<float>(fwd.at<double>(1))
                << static_cast<float>(fwd.at<double>(2));
    }

    m_lastEmittedPointCount = static_cast<int>(points.size());
    emit incrementalUpdate(pts, cls, camData, reset);
}

/**
 * @brief 执行增量式三角测量重建（主入口）—— 阶段二：COLMAP 式增量重建
 *
 * 流程：
 *  1. 构建特征轨迹对应图（CorrespondenceGraph）+ 观测管理器（ObservationManager）
 *  2. 注册种子对（image1 = 世界原点，image2 = initResult 的相对位姿）
 *  3. 对种子对三角化（TriangulateImage），随后全局BA + 过滤
 *  4. 主循环：FindNextImages → RegisterNextImage → TriangulateImage →
 *     IterativeLocalRefinement，直到没有图像能被注册
 *  5. 收尾：CompleteAndMergeTracks + Retriangulate + IterativeGlobalRefinement
 *
 * 与阶段一的本质差异：注册一张新图**不再**要求任何三视图共有点，
 * 只要它能看到 ≥ absPoseMinNumInliers 个已三角化点即可（COLMAP RegisterNextImage）。
 * 所有三维点统一由 ObservationManager 持有，可被后续图像复用、补全与合并。
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
    Q_UNUSED(viewGraph);

    outPoints3D.clear();
    outCameraPoses.clear();
    m_lastEmittedPointCount = 0;
    m_registeredCameras.clear();

    const int numImages = static_cast<int>(features.size());
    if (matchPairs.empty() || numImages < 3) {
        qDebug().noquote() << "[三角测量] 输入不足（匹配对为空或图像少于 3 张）";
        return false;
    }

    // ── 内参：优先使用调用方提供的，保证与初始化阶段一致 ──────────────────
    m_K = externalK.empty() ? estimateK(features) : externalK.clone();

    m_matchPairs = &matchPairs;
    m_features = &features;
    m_poses = &outCameraPoses;

    // ── 阶段五 5.2：畸变状态初始化（全零 = 针孔模型）─────────────────────
    m_dist = cv::Mat::zeros(5, 1, CV_64F);

    outCameraPoses.assign(static_cast<size_t>(numImages), CameraPose());
    for (int c = 0; c < numImages; ++c) {
        CameraPose &pose = outCameraPoses[static_cast<size_t>(c)];
        pose.imageId = c;
        pose.imageWidth = features[static_cast<size_t>(c)].imageWidth;
        pose.imageHeight = features[static_cast<size_t>(c)].imageHeight;
        pose.K = m_K.clone();
        pose.distortion = m_dist.clone();
        pose.registered = false;
    }
    m_numRegTrials.assign(static_cast<size_t>(numImages), 0);

    // ── 特征轨迹对应图 + 观测管理器 + 增量三角化器 ────────────────────────
    std::vector<int> numKp(static_cast<size_t>(numImages), 0);
    for (int i = 0; i < numImages; ++i) {
        numKp[static_cast<size_t>(i)] =
            static_cast<int>(features[static_cast<size_t>(i)].keypoints.size());
    }

    m_graph = std::make_unique<CorrespondenceGraph>();
    m_graph->build(matchPairs, numKp);
    m_obs = std::make_unique<ObservationManager>(features, *m_graph);
    m_triangulator = std::make_unique<IncrementalTriangulator>(
        *m_graph, *m_obs, features, outCameraPoses, m_K);

    qDebug().noquote()
        << QString("[三角测量] 特征轨迹=%1, 图像=%2").arg(m_graph->numTracks()).arg(numImages);

    // ── 注册种子对 ────────────────────────────────────────────────────────
    if (initResult.imageId1 < 0 || initResult.imageId1 >= numImages ||
        initResult.imageId2 < 0 || initResult.imageId2 >= numImages ||
        initResult.R.empty() || initResult.t.empty()) {
        qDebug().noquote() << "[三角测量] 初始化结果无效";
        return false;
    }

    const int seedId1 = initResult.imageId1;
    const int seedId2 = initResult.imageId2;
    m_refCameraId = seedId1;

    {
        CameraPose &p1 = outCameraPoses[static_cast<size_t>(seedId1)];
        p1.R = cv::Mat::eye(3, 3, CV_64F);
        p1.t = cv::Mat::zeros(3, 1, CV_64F);
        p1.registered = true;

        CameraPose &p2 = outCameraPoses[static_cast<size_t>(seedId2)];
        p2.R = initResult.R.clone();
        p2.t = initResult.t.clone();
        p2.registered = true;
    }
    m_registeredCameras.insert(seedId1);
    m_registeredCameras.insert(seedId2);

    // ── 种子对三角化：三角化角门槛渐进放宽 ─────────────────────────────────
    //
    // COLMAP 用 init_min_tri_angle = 16°，但它有一个隐含前提：种子对必须足够
    // 宽基线（COLMAP 的初始对选择会显式偏好这一点）。视角变化平缓的无人机序列
    // 相邻视角往往只有 10~15°，此时 16° 门槛会把几乎所有点拒掉
    // （实测：9602 个候选中 8118 个因角度不足被拒，最终只建出 0 个点）。
    //
    // 因此改为阶梯放宽：先用严格阈值保证精度，点数不够时逐步降低角度门槛；
    // 最后一步同时放宽「忽略两视图轨迹」限制。每轮只补充上一轮被拒的观测，
    // 不会重复建点。
    {
        TriangulatorOptions initOpts = m_triOptions;
        size_t numTris = 0;

        const double angleLadder[] = {m_options.initMinTriAngleDeg, 8.0, 4.0, 2.0,
                                      m_options.minTriAngleDeg};
        const size_t minSeedPts = static_cast<size_t>(
            std::max(m_options.minSeedPoints3D, m_options.absPoseMinNumInliers * 3));

        for (int step = 0; step < 5; ++step) {
            if (m_obs->numAlivePoints() >= minSeedPts) break;

            initOpts.minAngleDeg = angleLadder[step];
            // 最后一步放宽两视图轨迹限制（COLMAP 默认忽略，但种子对阶段
            // 已注册图像只有 2 张，仅靠多视图轨迹往往凑不出足够多的点）
            if (step == 4) initOpts.ignoreTwoViewTracks = false;

            const size_t before = m_obs->numAlivePoints();
            const size_t n1 = m_triangulator->triangulateImage(initOpts, seedId1);
            const size_t n2 = m_triangulator->triangulateImage(initOpts, seedId2);
            numTris += n1 + n2;

            const auto &st = m_triangulator->stats();
            qDebug().noquote()
                << QString("[三角测量] 种子对 #%1↔#%2 第%3轮: 角门槛=%4° 两视图限制=%5 → "
                           "新增观测=%6, 三维点 %7→%8 | 拒: 两视图=%9 角度过小=%10 残差=%11")
                       .arg(seedId1).arg(seedId2).arg(step + 1)
                       .arg(angleLadder[step], 0, 'f', 1)
                       .arg(initOpts.ignoreTwoViewTracks ? "开" : "关")
                       .arg(static_cast<qlonglong>(n1 + n2))
                       .arg(static_cast<qlonglong>(before))
                       .arg(static_cast<qlonglong>(m_obs->numAlivePoints()))
                       .arg(static_cast<qlonglong>(st.twoView))
                       .arg(static_cast<qlonglong>(st.angleSmall))
                       .arg(static_cast<qlonglong>(st.angleError));
            m_triangulator->clearStats();
        }
        m_triangulator->clearModified();
    }

    if (m_obs->numAlivePoints() == 0) {
        qDebug().noquote() << "[三角测量] 种子对未能三角化出任何三维点，重建失败";
        return false;
    }

    // ── 初始全局BA + 过滤（首轮放宽阈值：内参尚未标定好）──────────────────
    runGlobalBundleAdjustment(fixIntrinsics);
    m_obs->filterAllPoints3D(m_options.initialFilterMaxReprojErrorPx,
                             m_options.filterMinTriAngleDeg,
                             outCameraPoses, m_K, m_dist);
    if (m_obs->numAlivePoints() == 0) {
        qDebug().noquote() << "[三角测量] 初始全局BA后无有效三维点，重建失败";
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  增量注册主循环
    // ═══════════════════════════════════════════════════════════════════════
    while (true) {
        const auto tF = std::chrono::steady_clock::now();
        const std::vector<std::pair<int, int>> ranked = findNextImages();
        m_timing.findNext +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tF).count();
        if (ranked.empty()) {
            qDebug().noquote() << "[三角测量] 无更多可注册图像";
            break;
        }

        bool registeredAny = false;
        for (const auto &cand : ranked) {
            const int visible = cand.first;
            const int imageId = cand.second;

            // 排序为降序，后面的候选可见点数只会更少
            if (visible < m_options.absPoseMinNumInliers) break;

            if (registerNextImage(imageId)) {
                triangulateAndRefineImage(imageId);
                registeredAny = true;
                break;
            }
            ++m_numRegTrials[static_cast<size_t>(imageId)];
        }

        if (!registeredAny) {
            qDebug().noquote() << "[三角测量] 本轮所有候选均注册失败，结束增量注册";
            break;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  收尾：补全/合并 → 重三角化 → 迭代全局BA
    // ═══════════════════════════════════════════════════════════════════════
    if (m_options.enableTrackCompletion) {
        const auto tS = std::chrono::steady_clock::now();
        const size_t changed = completeAndMergeTracks();
        qDebug().noquote()
            << QString("[收尾] 补全+合并观测=%1 (%2 ms)")
                   .arg(static_cast<qlonglong>(changed))
                   .arg(std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - tS).count(), 0, 'f', 0);
    }

    if (m_options.enableRetriangulation) {
        const auto tS = std::chrono::steady_clock::now();
        const size_t numRe = m_triangulator->retriangulate(m_triOptions, matchPairs);
        qDebug().noquote()
            << QString("[收尾] 重三角化观测=%1 (%2 ms)")
                   .arg(static_cast<qlonglong>(numRe))
                   .arg(std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - tS).count(), 0, 'f', 0);
    }

    for (int iter = 0; iter < m_options.globalMaxRefinements; ++iter) {
        const size_t numObsBefore = m_obs->totalObservations();
        if (numObsBefore == 0) break;

        runGlobalBundleAdjustment(fixIntrinsics);

        size_t changed = m_options.enableTrackCompletion ? completeAndMergeTracks() : 0;
        changed += m_obs->filterAllPoints3D(m_options.filterMaxReprojErrorPx,
                                            m_options.filterMinTriAngleDeg,
                                            outCameraPoses, m_K, m_dist);

        const double ratio = static_cast<double>(changed) /
                             static_cast<double>(std::max<size_t>(1, numObsBefore));
        qDebug().noquote()
            << QString("[收尾] 全局精化第%1轮: 变化=%2/%3 (%4)")
                   .arg(iter + 1)
                   .arg(static_cast<qlonglong>(changed))
                   .arg(static_cast<qlonglong>(numObsBefore))
                   .arg(ratio, 0, 'f', 4);

        if (ratio < m_options.globalMaxRefinementChange) break;
    }
    m_triangulator->clearModified();

    // ═══════════════════════════════════════════════════════════════════════
    //  阶段四 4.4：末尾滤波兜底（4px / 1.5° / 短轨迹）
    //  迭代全局精化每轮已含 filterAllPoints3D，这里再做一次兜底：
    //  1) 保证滤波是最后一步（补全/合并不会在滤波之后残留新观测）
    //  2) 显式执行可配置的短轨迹剔除（filterMinTrackLength）
    // ═══════════════════════════════════════════════════════════════════════
    {
        const size_t filteredFinal = m_obs->filterAllPoints3D(
            m_options.filterMaxReprojErrorPx, m_options.filterMinTriAngleDeg,
            outCameraPoses, m_K, m_dist);
        const size_t filteredShort =
            m_obs->filterShortTracks(m_options.filterMinTrackLength);
        if (filteredFinal + filteredShort > 0) {
            qDebug().noquote()
                << QString("[末尾滤波] 外点=%1, 短轨迹(<%2观测)=%3")
                       .arg(static_cast<qlonglong>(filteredFinal))
                       .arg(static_cast<qlonglong>(m_options.filterMinTrackLength))
                       .arg(static_cast<qlonglong>(filteredShort));
        }
    }

    // ── 阶段四 4.6：场景归一化（相似变换，不改重投影误差）─────────────────
    if (m_options.enableSceneNormalization) {
        std::vector<Point3D> normalized;
        std::vector<int> idMap;
        m_obs->compact(normalized, &idMap);
        SfmGeom::normalizeScene(normalized, outCameraPoses,
                                m_options.sceneNormalizationExtent);
        for (size_t oldId = 0; oldId < idMap.size(); ++oldId) {
            if (idMap[oldId] >= 0) {
                m_obs->point(static_cast<int>(oldId)).position =
                    normalized[static_cast<size_t>(idMap[oldId])].position;
            }
        }
        qDebug().noquote()
            << QString("[场景归一化] 相机平均光心距 → %1, 质心 → 原点")
                   .arg(m_options.sceneNormalizationExtent, 0, 'f', 1);
    }

    // ── 重建质量统计（对齐优化方案「验证方法」指标清单）───────────────────
    {
        const auto st = m_obs->computeStats(outCameraPoses, m_K, m_dist);
        qDebug().noquote()
            << QString("[重建统计] 相机=%1/%2, 点=%3, 观测=%4 | "
                       "轨迹长: 均值=%5 中位=%6 | "
                       "重投影: 均值=%7px 中位=%8px 后方=%9 | "
                       "三角化角: 均值=%10° 中位=%11°")
                   .arg(m_registeredCameras.size()).arg(numImages)
                   .arg(static_cast<qlonglong>(st.numPoints))
                   .arg(static_cast<qlonglong>(st.numObservations))
                   .arg(st.meanTrackLength, 0, 'f', 2)
                   .arg(st.medianTrackLength, 0, 'f', 1)
                   .arg(st.meanReprojErrorPx, 0, 'f', 2)
                   .arg(st.medianReprojErrorPx, 0, 'f', 2)
                   .arg(static_cast<qlonglong>(st.numBehindCamera))
                   .arg(st.meanTriAngleDeg, 0, 'f', 1)
                   .arg(st.medianTriAngleDeg, 0, 'f', 1);
    }

    // ── 输出 ──────────────────────────────────────────────────────────────
    m_obs->compact(outPoints3D);
    emitCurrentState(true);

    if (outPoints3D.empty() || m_registeredCameras.size() < 3) {
        qDebug().noquote()
            << QString("[三角测量] 重建失败: 三维点=%1, 已注册相机=%2")
                   .arg(outPoints3D.size()).arg(m_registeredCameras.size());
        return false;
    }

    qDebug().noquote()
        << QString("[三角测量] 重建完成: 已注册相机=%1/%2, 三维点=%3")
               .arg(m_registeredCameras.size()).arg(numImages).arg(outPoints3D.size());
    return true;
}
