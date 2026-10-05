/**
 * @file IncrementalTriangulator.cpp
 * @brief 增量三角化器实现 —— Create / Continue / Complete / Merge / Retriangulate
 *
 * 对齐 COLMAP sfm/incremental_triangulator.cc。
 *
 * 三角化估计部分与 COLMAP 的差异：
 * - COLMAP 用 RANSAC 在两视图样本上估计 + cheirality 检查，残差可选角度/像素
 * - 这里用「全观测 DLT → 内点筛选 → 内点重解」的迭代方案。
 *   因为输入匹配已经过 Lowe 比率检验（多数数据集还经过几何验证），
 *   外点比例较低，两次迭代重加权足以剔除，且避免了实现 RANSAC 采样器。
 */

#include "IncrementalTriangulator.h"

#include <algorithm>
#include <map>

#include <opencv2/calib3d.hpp>

#include "SfmGeometry.h"

// ═════════════════════════════════════════════════════════════════════════════
//  内部辅助
// ═════════════════════════════════════════════════════════════════════════════

namespace {

/// 由 R、t 组装世界→相机的投影矩阵 P = [R | t]（3×4）
cv::Mat projectionMatrix(const cv::Mat &R, const cv::Mat &t)
{
    return (cv::Mat_<double>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
}

/**
 * @brief 多视图 DLT 三角化（观测为归一化相机坐标）
 *
 * 每个观测贡献两条方程：
 *   u·P₃ - P₁ = 0
 *   v·P₃ - P₂ = 0
 * 对 A(2n×4) 做 SVD，取最小奇异值对应的右奇异向量。
 */
bool dltTriangulate(const std::vector<cv::Mat> &P,
                    const std::vector<cv::Point2d> &xy,
                    cv::Point3d &out)
{
    const int n = static_cast<int>(P.size());
    if (n < 2) return false;

    cv::Mat A(2 * n, 4, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double u = xy[static_cast<size_t>(i)].x;
        const double v = xy[static_cast<size_t>(i)].y;
        for (int c = 0; c < 4; ++c) {
            A.at<double>(2 * i, c) =
                u * P[static_cast<size_t>(i)].at<double>(2, c) - P[static_cast<size_t>(i)].at<double>(0, c);
            A.at<double>(2 * i + 1, c) =
                v * P[static_cast<size_t>(i)].at<double>(2, c) - P[static_cast<size_t>(i)].at<double>(1, c);
        }
    }

    cv::Mat w, uMat, vt;
    cv::SVD::compute(A, w, uMat, vt, cv::SVD::FULL_UV);

    cv::Mat X4 = vt.row(3).t();  // 4×1
    const double w4 = X4.at<double>(3, 0);
    if (std::abs(w4) < 1e-12) return false;

    const double x = X4.at<double>(0, 0) / w4;
    const double y = X4.at<double>(1, 0) / w4;
    const double z = X4.at<double>(2, 0) / w4;

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    if (std::abs(x) > 1e12 || std::abs(y) > 1e12 || std::abs(z) > 1e12) return false;

    out = cv::Point3d(x, y, z);
    return true;
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
//  构造
// ═════════════════════════════════════════════════════════════════════════════

IncrementalTriangulator::IncrementalTriangulator(const CorrespondenceGraph &graph,
                                                 ObservationManager &obs,
                                                 const std::vector<FeatureResult> &features,
                                                 std::vector<CameraPose> &poses,
                                                 const cv::Mat &K)
    : m_graph(&graph),
      m_obs(&obs),
      m_features(&features),
      m_poses(&poses),
      m_K(&K)
{
}

cv::Vec3b IncrementalTriangulator::sampleColor(int imageId, int kptIdx) const
{
    if (imageId < 0 || imageId >= static_cast<int>(m_features->size())) {
        return cv::Vec3b(128, 128, 128);
    }
    const FeatureResult &f = (*m_features)[static_cast<size_t>(imageId)];
    if (f.imageData.empty() || kptIdx < 0 ||
        kptIdx >= static_cast<int>(f.keypoints.size())) {
        return cv::Vec3b(128, 128, 128);
    }

    const cv::KeyPoint &kp = f.keypoints[static_cast<size_t>(kptIdx)];
    const int px = static_cast<int>(kp.pt.x + 0.5);
    const int py = static_cast<int>(kp.pt.y + 0.5);
    if (px < 0 || px >= f.imageData.cols || py < 0 || py >= f.imageData.rows) {
        return cv::Vec3b(128, 128, 128);
    }
    return f.imageData.at<cv::Vec3b>(py, px);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Find
// ═════════════════════════════════════════════════════════════════════════════

size_t IncrementalTriangulator::find(int imageId, int kptIdx, int transitivity,
                                     std::vector<CorrData> *out) const
{
    out->clear();

    std::vector<CorrespondenceGraph::Observation> corrs;
    m_graph->extractTransitiveCorrespondences(imageId, kptIdx, transitivity, &corrs);

    out->reserve(corrs.size());
    size_t numTriangulated = 0;

    for (const auto &c : corrs) {
        if (c.first < 0 || c.first >= static_cast<int>(m_poses->size())) continue;
        if (!(*m_poses)[static_cast<size_t>(c.first)].registered) continue;

        CorrData cd;
        cd.imageId = c.first;
        cd.kptIdx = c.second;
        cd.pointId = m_obs->pointIdAt(c.first, c.second);
        cd.xy = m_obs->keypointPos(c.first, c.second);
        out->push_back(cd);

        if (cd.pointId >= 0) ++numTriangulated;
    }

    return numTriangulated;
}

// ═════════════════════════════════════════════════════════════════════════════
//  三角化估计
// ═════════════════════════════════════════════════════════════════════════════

bool IncrementalTriangulator::estimateTriangulation(const std::vector<CorrData> &corrs,
                                                    double minAngleDeg,
                                                    double maxError,
                                                    bool errorIsPixel,
                                                    cv::Point3d &outXyz,
                                                    std::vector<char> &outInlierMask) const
{
    const size_t n = corrs.size();
    outInlierMask.assign(n, 0);
    if (n < 2 || m_K->empty()) return false;

    // ── 准备投影矩阵 / 归一化观测 / 光心 ──────────────────────────────────
    std::vector<cv::Mat> P(n);
    std::vector<cv::Point2d> xy(n);
    std::vector<cv::Vec3d> centers(n);

    for (size_t i = 0; i < n; ++i) {
        const CameraPose &pose = (*m_poses)[static_cast<size_t>(corrs[i].imageId)];
        if (!pose.registered || pose.R.empty() || pose.t.empty()) return false;

        P[i] = projectionMatrix(pose.R, pose.t);
        // 阶段五 5.2：有畸变时观测必须先去畸变，否则射线方向是弯的
        xy[i] = SfmGeom::toNormalized(corrs[i].xy, *m_K, m_dist);
        centers[i] = SfmGeom::cameraCenter(pose.R, pose.t);
    }

    cv::Point3d X;
    if (!dltTriangulate(P, xy, X)) return false;

    // ── 迭代重加权：解 → 选内点 → 重解 ────────────────────────────────────
    std::vector<char> mask(n, 0);
    size_t numInliers = 0;

    auto selectInliers = [&]() -> size_t {
        mask.assign(n, 0);
        size_t count = 0;
        const cv::Vec3d Pw(X.x, X.y, X.z);
        for (size_t i = 0; i < n; ++i) {
            const CameraPose &pose = (*m_poses)[static_cast<size_t>(corrs[i].imageId)];
            const double err = errorIsPixel
                ? SfmGeom::reprojectionError(Pw, pose.R, pose.t, *m_K, corrs[i].xy, m_dist)
                : SfmGeom::angularErrorDeg(corrs[i].xy, Pw, pose.R, pose.t, *m_K, m_dist);

            if (err >= 0.0 && err <= maxError) {
                mask[i] = 1;
                ++count;
            }
        }
        return count;
    };

    for (int iter = 0; iter < 3; ++iter) {
        numInliers = selectInliers();
        if (numInliers < 2) return false;

        std::vector<cv::Mat> PIn;
        std::vector<cv::Point2d> xyIn;
        PIn.reserve(numInliers);
        xyIn.reserve(numInliers);
        for (size_t i = 0; i < n; ++i) {
            if (mask[i]) { PIn.push_back(P[i]); xyIn.push_back(xy[i]); }
        }

        cv::Point3d XNew;
        if (!dltTriangulate(PIn, xyIn, XNew)) break;
        X = XNew;
    }

    // 用最终的 X 再判一次内点，保证输出的 mask 与 outXyz 一致
    numInliers = selectInliers();
    if (numInliers < 2) return false;

    // ── 三角化角下限：内点两两之间的最大夹角 ──────────────────────────────
    if (minAngleDeg > 0.0) {
        const cv::Vec3d Pw(X.x, X.y, X.z);
        double maxAngle = 0.0;
        size_t count = 0;
        std::vector<size_t> inlierIdx;
        for (size_t i = 0; i < n; ++i) {
            if (mask[i]) inlierIdx.push_back(i);
        }
        const size_t lim = std::min<size_t>(inlierIdx.size(), 12);
        for (size_t a = 0; a < lim; ++a) {
            for (size_t b = a + 1; b < lim; ++b) {
                const double ang = SfmGeom::triangulationAngleDeg(
                    Pw, centers[inlierIdx[a]], centers[inlierIdx[b]]);
                maxAngle = std::max(maxAngle, ang);
                ++count;
            }
        }
        if (count > 0 && maxAngle < minAngleDeg) return false;
    }

    outXyz = X;
    outInlierMask = mask;
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Create / Continue
// ═════════════════════════════════════════════════════════════════════════════

size_t IncrementalTriangulator::create(const TriangulatorOptions &options,
                                       const std::vector<CorrData> &corrs)
{
    // 只保留尚未三角化的观测
    std::vector<CorrData> candidates;
    candidates.reserve(corrs.size());
    for (const auto &c : corrs) {
        if (c.pointId < 0) candidates.push_back(c);
    }

    ++m_stats.attempts;

    if (candidates.size() < static_cast<size_t>(options.minTrackLength)) {
        ++m_stats.tooShort;
        return 0;
    }

    // 两视图轨迹缺乏冗余约束，误匹配无法被残差剔除（COLMAP ignore_two_view_tracks）
    if (options.ignoreTwoViewTracks && candidates.size() == 2 &&
        m_graph->isTwoViewObservation(candidates[0].imageId, candidates[0].kptIdx)) {
        ++m_stats.twoView;
        return 0;
    }

    cv::Point3d xyz;
    std::vector<char> inlierMask;
    if (!estimateTriangulation(candidates,
                               options.minAngleDeg,
                               options.createMaxAngleErrorDeg,
                               /*errorIsPixel=*/false,
                               xyz, inlierMask)) {
        // 区分「三角化角不足」与「角度残差过大」：先用 0° 角阈值重试一次
        std::vector<char> mask2;
        if (estimateTriangulation(candidates, 0.0, options.createMaxAngleErrorDeg,
                                  /*errorIsPixel=*/false, xyz, mask2)) {
            ++m_stats.angleSmall;
        } else {
            ++m_stats.angleError;
        }
        return 0;
    }

    std::vector<CorrespondenceGraph::Observation> track;
    std::vector<CorrData> remaining;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i < inlierMask.size() && inlierMask[i]) {
            track.emplace_back(candidates[i].imageId, candidates[i].kptIdx);
        } else {
            remaining.push_back(candidates[i]);
        }
    }

    if (track.size() < static_cast<size_t>(options.minTrackLength)) return 0;

    const int pointId = m_obs->addPoint3D(xyz, track, sampleColor(track[0].first, track[0].second));
    m_modified.insert(pointId);
    ++m_stats.accepted;

    // 递归：剩余观测足够多时再尝试建一个点（COLMAP kMinRecursiveTrackLength = 3）
    const size_t numTris = track.size();
    if (candidates.size() - numTris >= 3 && remaining.size() >= 2) {
        return numTris + create(options, remaining);
    }
    return numTris;
}

size_t IncrementalTriangulator::continueTrack(const TriangulatorOptions &options,
                                              const CorrData &ref,
                                              const std::vector<CorrData> &corrs)
{
    // 参考观测已三角化则无需续接
    if (ref.pointId >= 0) return 0;

    double bestError = 1e30;
    int bestIdx = -1;

    for (size_t i = 0; i < corrs.size(); ++i) {
        const CorrData &c = corrs[i];
        if (c.pointId < 0 || !m_obs->exists(c.pointId)) continue;

        const Point3D &p = m_obs->point(c.pointId);
        const CameraPose &pose = (*m_poses)[static_cast<size_t>(ref.imageId)];
        if (!pose.registered || pose.R.empty() || pose.t.empty()) continue;

        const double err = SfmGeom::angularErrorDeg(
            ref.xy, cv::Vec3d(p.position.x, p.position.y, p.position.z),
            pose.R, pose.t, *m_K, m_dist);

        if (err < bestError) {
            bestError = err;
            bestIdx = static_cast<int>(i);
        }
    }

    if (bestIdx < 0 || bestError > options.continueMaxAngleErrorDeg) return 0;

    const int pointId = corrs[static_cast<size_t>(bestIdx)].pointId;
    m_obs->addObservation(pointId, ref.imageId, ref.kptIdx);
    m_modified.insert(pointId);
    return 1;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Complete / Merge
// ═════════════════════════════════════════════════════════════════════════════

size_t IncrementalTriangulator::complete(const TriangulatorOptions &options, int pointId)
{
    if (!m_obs->exists(pointId)) return 0;

    const Point3D p = m_obs->point(pointId);  // 拷贝：下方会修改点，避免引用失效
    const cv::Vec3d Pw(p.position.x, p.position.y, p.position.z);

    size_t numCompleted = 0;

    // BFS：每层把上一层新加入的观测作为种子，向外扩展一跳
    std::vector<CorrespondenceGraph::Observation> curr(p.observations.begin(), p.observations.end());
    std::vector<CorrespondenceGraph::Observation> next;
    std::set<std::pair<int, int>> visited(p.observations.begin(), p.observations.end());
    std::vector<CorrespondenceGraph::Observation> corrs;

    for (int level = 0; level < options.completeMaxTransitivity && !curr.empty(); ++level) {
        next.clear();

        for (const auto &seed : curr) {
            m_graph->findCorrespondences(seed.first, seed.second, &corrs);
            for (const auto &c : corrs) {
                if (!visited.insert(c).second) continue;
                if (c.first >= static_cast<int>(m_poses->size())) continue;

                const CameraPose &pose = (*m_poses)[static_cast<size_t>(c.first)];
                if (!pose.registered || pose.R.empty() || pose.t.empty()) continue;
                if (m_obs->pointIdAt(c.first, c.second) >= 0) continue;

                const double err = SfmGeom::reprojectionError(
                    Pw, pose.R, pose.t, *m_K, m_obs->keypointPos(c.first, c.second),
                    m_dist);
                if (err < 0.0 || err > options.completeMaxReprojErrorPx) continue;

                if (!m_obs->exists(pointId)) return numCompleted;
                m_obs->addObservation(pointId, c.first, c.second);
                m_modified.insert(pointId);
                next.push_back(c);
                ++numCompleted;
            }
        }

        curr = next;
    }

    return numCompleted;
}

size_t IncrementalTriangulator::merge(const TriangulatorOptions &options, int pointId)
{
    if (!m_obs->exists(pointId)) return 0;

    const double maxSqErr = options.mergeMaxReprojErrorPx * options.mergeMaxReprojErrorPx;

    // 拷贝轨迹：合并过程中点会被修改
    const std::vector<std::pair<int, int>> track = m_obs->point(pointId).observations;

    for (const auto &el : track) {
        std::vector<CorrespondenceGraph::Observation> corrs;
        m_graph->findCorrespondences(el.first, el.second, &corrs);

        for (const auto &c : corrs) {
            if (c.first >= static_cast<int>(m_poses->size())) continue;
            if (!(*m_poses)[static_cast<size_t>(c.first)].registered) continue;

            const int otherId = m_obs->pointIdAt(c.first, c.second);
            if (otherId < 0 || otherId == pointId) continue;
            if (!m_obs->exists(pointId) || !m_obs->exists(otherId)) continue;

            // 同一对点只尝试一次（与访问方向无关）
            const auto key = pointId < otherId
                ? std::make_pair(pointId, otherId)
                : std::make_pair(otherId, pointId);
            if (!m_mergeTrials.insert(key).second) continue;

            const Point3D &pA = m_obs->point(pointId);
            const Point3D &pB = m_obs->point(otherId);

            // 按轨迹长度加权平均
            const double nA = static_cast<double>(pA.observations.size());
            const double nB = static_cast<double>(pB.observations.size());
            const cv::Vec3d merged(
                (nA * pA.position.x + nB * pB.position.x) / (nA + nB),
                (nA * pA.position.y + nB * pB.position.y) / (nA + nB),
                (nA * pA.position.z + nB * pB.position.z) / (nA + nB));

            // 合并后所有观测都必须是内点
            bool ok = true;
            for (const Point3D *p : {&pA, &pB}) {
                for (const auto &obs : p->observations) {
                    const CameraPose &pose = (*m_poses)[static_cast<size_t>(obs.first)];
                    if (!pose.registered || pose.R.empty() || pose.t.empty()) { ok = false; break; }
                    const double err = SfmGeom::reprojectionError(
                        merged, pose.R, pose.t, *m_K,
                        m_obs->keypointPos(obs.first, obs.second), m_dist);
                    if (err < 0.0 || err * err > maxSqErr) { ok = false; break; }
                }
                if (!ok) break;
            }
            if (!ok) continue;

            const size_t numMerged = pA.observations.size() + pB.observations.size();
            const int mergedId = m_obs->mergePoints3D(pointId, otherId);
            m_modified.erase(pointId);
            m_modified.erase(otherId);
            m_modified.insert(mergedId);

            // 递归合并新点（两个原点已被删除）
            const size_t numRecursive = merge(options, mergedId);
            return numRecursive > 0 ? numRecursive : numMerged;
        }
    }

    return 0;
}

size_t IncrementalTriangulator::completeTracks(const TriangulatorOptions &options,
                                               const std::vector<int> &pointIds)
{
    size_t total = 0;
    m_mergeTrials.clear();
    for (const int pid : pointIds) {
        total += complete(options, pid);
    }
    return total;
}

size_t IncrementalTriangulator::completeAllTracks(const TriangulatorOptions &options)
{
    std::vector<int> ids;
    ids.reserve(m_obs->allPoints().size());
    for (size_t i = 0; i < m_obs->allPoints().size(); ++i) {
        if (m_obs->aliveMask()[i]) ids.push_back(static_cast<int>(i));
    }
    return completeTracks(options, ids);
}

size_t IncrementalTriangulator::mergeTracks(const TriangulatorOptions &options,
                                            const std::vector<int> &pointIds)
{
    size_t total = 0;
    m_mergeTrials.clear();
    for (const int pid : pointIds) {
        total += merge(options, pid);
    }
    return total;
}

size_t IncrementalTriangulator::mergeAllTracks(const TriangulatorOptions &options)
{
    std::vector<int> ids;
    ids.reserve(m_obs->allPoints().size());
    for (size_t i = 0; i < m_obs->allPoints().size(); ++i) {
        if (m_obs->aliveMask()[i]) ids.push_back(static_cast<int>(i));
    }
    return mergeTracks(options, ids);
}

// ═════════════════════════════════════════════════════════════════════════════
//  TriangulateImage / CompleteImage
// ═════════════════════════════════════════════════════════════════════════════

size_t IncrementalTriangulator::triangulateImage(const TriangulatorOptions &options,
                                                 int imageId)
{
    if (imageId < 0 || imageId >= static_cast<int>(m_poses->size())) return 0;
    if (!(*m_poses)[static_cast<size_t>(imageId)].registered) return 0;

    const int numKp = static_cast<int>(
        (*m_features)[static_cast<size_t>(imageId)].keypoints.size());

    std::vector<CorrData> corrs;
    size_t numTris = 0;

    for (int kpt = 0; kpt < numKp; ++kpt) {
        const int ownPoint = m_obs->pointIdAt(imageId, kpt);
        if (ownPoint >= 0) continue;  // 已有三维点，交给 Complete 补全

        const size_t numTriangulated = find(imageId, kpt, options.maxTransitivity, &corrs);
        if (corrs.empty()) continue;

        CorrData ref;
        ref.imageId = imageId;
        ref.kptIdx = kpt;
        ref.pointId = ownPoint;
        ref.xy = m_obs->keypointPos(imageId, kpt);

        if (numTriangulated == 0) {
            // 轨迹上没有任何已三角化观测 → 尝试新建点
            corrs.push_back(ref);
            numTris += create(options, corrs);
        } else {
            // 先把参考观测续接到最优已有三维点，再用剩余观测尝试新建
            numTris += continueTrack(options, ref, corrs);
            ref.pointId = m_obs->pointIdAt(imageId, kpt);
            if (ref.pointId < 0) {
                corrs.push_back(ref);
                numTris += create(options, corrs);
            }
        }
    }

    return numTris;
}

size_t IncrementalTriangulator::completeImage(const TriangulatorOptions &options,
                                              int imageId)
{
    if (imageId < 0 || imageId >= static_cast<int>(m_poses->size())) return 0;
    if (!(*m_poses)[static_cast<size_t>(imageId)].registered) return 0;

    const int numKp = static_cast<int>(
        (*m_features)[static_cast<size_t>(imageId)].keypoints.size());

    std::vector<CorrData> corrs;
    size_t numTris = 0;

    for (int kpt = 0; kpt < numKp; ++kpt) {
        const int ownPoint = m_obs->pointIdAt(imageId, kpt);
        if (ownPoint >= 0) {
            numTris += complete(options, ownPoint);  // 补全已有轨迹
            continue;
        }

        if (options.ignoreTwoViewTracks &&
            m_graph->isTwoViewObservation(imageId, kpt)) {
            continue;
        }

        const size_t numTriangulated = find(imageId, kpt, options.maxTransitivity, &corrs);
        if (numTriangulated > 0 || corrs.empty()) continue;

        CorrData ref;
        ref.imageId = imageId;
        ref.kptIdx = kpt;
        ref.pointId = -1;
        ref.xy = m_obs->keypointPos(imageId, kpt);
        corrs.push_back(ref);

        // 补全用像素残差（COLMAP complete_max_reproj_error）
        cv::Point3d xyz;
        std::vector<char> inlierMask;
        if (!estimateTriangulation(corrs,
                                   options.minAngleDeg,
                                   options.completeMaxReprojErrorPx,
                                   /*errorIsPixel=*/true,
                                   xyz, inlierMask)) {
            continue;
        }

        std::vector<CorrespondenceGraph::Observation> track;
        for (size_t i = 0; i < corrs.size(); ++i) {
            if (i < inlierMask.size() && inlierMask[i]) {
                track.emplace_back(corrs[i].imageId, corrs[i].kptIdx);
            }
        }
        if (track.size() < static_cast<size_t>(options.minTrackLength)) continue;

        const int pid = m_obs->addPoint3D(xyz, track, sampleColor(track[0].first, track[0].second));
        m_modified.insert(pid);
        numTris += track.size();
    }

    return numTris;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Retriangulate
// ═════════════════════════════════════════════════════════════════════════════

size_t IncrementalTriangulator::retriangulate(const TriangulatorOptions &options,
                                              const std::vector<MatchPair> &matchPairs)
{
    size_t numTris = 0;

    TriangulatorOptions reOptions = options;
    reOptions.continueMaxAngleErrorDeg = options.reMaxAngleErrorDeg;

    for (const auto &mp : matchPairs) {
        if (mp.imageId1 >= static_cast<int>(m_poses->size()) ||
            mp.imageId2 >= static_cast<int>(m_poses->size())) {
            continue;
        }
        if (!(*m_poses)[static_cast<size_t>(mp.imageId1)].registered ||
            !(*m_poses)[static_cast<size_t>(mp.imageId2)].registered) {
            continue;
        }

        const size_t numTotal = mp.matches.size();
        if (numTotal == 0) continue;

        const double triRatio =
            static_cast<double>(m_obs->numTriangulatedMatches(mp)) /
            static_cast<double>(numTotal);
        if (triRatio >= options.reMinRatio) continue;

        auto key = std::make_pair(std::min(mp.imageId1, mp.imageId2),
                                  std::max(mp.imageId1, mp.imageId2));
        int &trials = m_reTrials[key];
        if (trials >= options.reMaxTrials) continue;
        ++trials;

        for (const auto &m : mp.matches) {
            const int pid1 = m_obs->pointIdAt(mp.imageId1, m.queryIdx);
            const int pid2 = m_obs->pointIdAt(mp.imageId2, m.trainIdx);

            CorrData c1;
            c1.imageId = mp.imageId1;
            c1.kptIdx = m.queryIdx;
            c1.pointId = pid1;
            c1.xy = m_obs->keypointPos(mp.imageId1, m.queryIdx);

            CorrData c2;
            c2.imageId = mp.imageId2;
            c2.kptIdx = m.trainIdx;
            c2.pointId = pid2;
            c2.xy = m_obs->keypointPos(mp.imageId2, m.trainIdx);

            if (pid1 >= 0 && pid2 >= 0) {
                // 两端都已三角化：不做合并（COLMAP 注释：重三角化的对应很可能是错的，
                // 合并会同时毁掉两个已有点）
                continue;
            } else if (pid1 >= 0) {
                numTris += continueTrack(reOptions, c2, {c1});
            } else if (pid2 >= 0) {
                numTris += continueTrack(reOptions, c1, {c2});
            } else {
                // 新建点用原始（更严格）阈值，避免引入漂移
                numTris += create(options, {c1, c2});
            }
        }
    }

    return numTris;
}
