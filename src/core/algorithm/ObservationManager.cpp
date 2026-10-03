/**
 * @file ObservationManager.cpp
 * @brief 三维点与特征轨迹管理 —— 稀疏重建状态的唯一持有者
 *
 * 对齐 COLMAP sfm/observation_manager.h 的核心语义：
 * - AddPoint3D / AddObservation / MergePoints3D / DeletePoint3D
 * - NumVisiblePoints3D（用于 FindNextImages 排序与注册可行性判断）
 * - FilterPoints3D / FilterPoints3DInImages（重投影误差 + 三角化角）
 */

#include "ObservationManager.h"

#include "SfmGeometry.h"

ObservationManager::ObservationManager(const std::vector<FeatureResult> &features,
                                       const CorrespondenceGraph &graph)
    : m_features(&features),
      m_graph(&graph),
      m_kpTo3D(features.size())
{
}

int ObservationManager::numKeypoints(int imageId) const
{
    if (imageId < 0 || imageId >= static_cast<int>(m_features->size())) return 0;
    return static_cast<int>((*m_features)[static_cast<size_t>(imageId)].keypoints.size());
}

cv::Point2d ObservationManager::keypointPos(int imageId, int kptIdx) const
{
    if (imageId < 0 || imageId >= static_cast<int>(m_features->size())) return cv::Point2d();
    const auto &kps = (*m_features)[static_cast<size_t>(imageId)].keypoints;
    if (kptIdx < 0 || kptIdx >= static_cast<int>(kps.size())) return cv::Point2d();
    return kps[static_cast<size_t>(kptIdx)].pt;
}

// ═════════════════════════════════════════════════════════════════════════════
//  三维点增删改
// ═════════════════════════════════════════════════════════════════════════════

int ObservationManager::addPoint3D(
    const cv::Point3d &xyz,
    const std::vector<CorrespondenceGraph::Observation> &track,
    const cv::Vec3b &color)
{
    Point3D p;
    p.id = static_cast<int>(m_points.size());
    p.position = xyz;
    p.color = color;

    // 轨迹去重：同一张图像只保留一个观测
    std::vector<char> seenImage(m_features->size(), 0);
    p.observations.reserve(track.size());
    for (const auto &obs : track) {
        if (obs.first < 0 || obs.first >= static_cast<int>(m_features->size())) continue;
        if (seenImage[static_cast<size_t>(obs.first)]) continue;
        seenImage[static_cast<size_t>(obs.first)] = 1;
        p.observations.emplace_back(obs.first, obs.second);
    }

    m_points.push_back(p);
    m_alive.push_back(1);
    ++m_numAlive;

    const int pointId = p.id;
    for (const auto &obs : p.observations) {
        m_kpTo3D[static_cast<size_t>(obs.first)][obs.second] = pointId;
    }
    return pointId;
}

void ObservationManager::addObservation(int pointId, int imageId, int kptIdx)
{
    if (!exists(pointId)) return;
    if (imageId < 0 || imageId >= static_cast<int>(m_features->size())) return;

    Point3D &p = m_points[static_cast<size_t>(pointId)];
    for (const auto &obs : p.observations) {
        if (obs.first == imageId && obs.second == kptIdx) return;  // 已存在
    }
    p.observations.emplace_back(imageId, kptIdx);
    m_kpTo3D[static_cast<size_t>(imageId)][kptIdx] = pointId;
}

void ObservationManager::deleteObservation(int imageId, int kptIdx)
{
    const int pointId = pointIdAt(imageId, kptIdx);
    if (pointId < 0) return;

    m_kpTo3D[static_cast<size_t>(imageId)].erase(kptIdx);

    Point3D &p = m_points[static_cast<size_t>(pointId)];
    for (auto it = p.observations.begin(); it != p.observations.end(); ++it) {
        if (it->first == imageId && it->second == kptIdx) {
            p.observations.erase(it);
            break;
        }
    }

    if (p.observations.size() < 2) {
        deletePoint3D(pointId);
    }
}

void ObservationManager::deletePoint3D(int pointId)
{
    if (!exists(pointId)) return;

    Point3D &p = m_points[static_cast<size_t>(pointId)];
    for (const auto &obs : p.observations) {
        auto &imgMap = m_kpTo3D[static_cast<size_t>(obs.first)];
        const auto it = imgMap.find(obs.second);
        if (it != imgMap.end() && it->second == pointId) {
            imgMap.erase(it);
        }
    }
    p.observations.clear();
    m_alive[static_cast<size_t>(pointId)] = 0;
    --m_numAlive;
}

int ObservationManager::mergePoints3D(int id1, int id2)
{
    if (!exists(id1) || !exists(id2) || id1 == id2) return id1;

    Point3D &p1 = m_points[static_cast<size_t>(id1)];
    Point3D &p2 = m_points[static_cast<size_t>(id2)];

    // 按轨迹长度加权平均（COLMAP MergePoints3D）
    const double n1 = static_cast<double>(p1.observations.size());
    const double n2 = static_cast<double>(p2.observations.size());
    const double w = n1 + n2;
    if (w > 0.0) {
        p1.position = cv::Point3d(
            (n1 * p1.position.x + n2 * p2.position.x) / w,
            (n1 * p1.position.y + n2 * p2.position.y) / w,
            (n1 * p1.position.z + n2 * p2.position.z) / w);
    }

    // 把 p2 的轨迹并入 p1（先重定向 kpTo3D，避免删除时误删刚写入的映射）
    for (const auto &obs : p2.observations) {
        p1.observations.emplace_back(obs.first, obs.second);
        m_kpTo3D[static_cast<size_t>(obs.first)][obs.second] = id1;
    }
    p2.observations.clear();
    m_alive[static_cast<size_t>(id2)] = 0;
    --m_numAlive;

    return id1;
}

int ObservationManager::pointIdAt(int imageId, int kptIdx) const
{
    if (imageId < 0 || imageId >= static_cast<int>(m_kpTo3D.size())) return -1;
    const auto &imgMap = m_kpTo3D[static_cast<size_t>(imageId)];
    const auto it = imgMap.find(kptIdx);
    return it == imgMap.end() ? -1 : it->second;
}

// ═════════════════════════════════════════════════════════════════════════════
//  可见性统计
// ═════════════════════════════════════════════════════════════════════════════

size_t ObservationManager::numObservations(int imageId) const
{
    if (imageId < 0 || imageId >= static_cast<int>(m_kpTo3D.size())) return 0;
    return m_kpTo3D[static_cast<size_t>(imageId)].size();
}

size_t ObservationManager::numVisiblePoints3D(int imageId) const
{
    const int numKp = numKeypoints(imageId);
    if (numKp == 0) return 0;

    std::vector<CorrespondenceGraph::Observation> corrs;
    size_t count = 0;

    for (int kpt = 0; kpt < numKp; ++kpt) {
        if (pointIdAt(imageId, kpt) >= 0) {
            ++count;   // 自身已三角化（重注册场景）
            continue;
        }
        if (!m_graph->hasCorrespondences(imageId, kpt)) continue;

        m_graph->findCorrespondences(imageId, kpt, &corrs);
        for (const auto &c : corrs) {
            if (pointIdAt(c.first, c.second) >= 0) {
                ++count;
                break;
            }
        }
    }
    return count;
}

size_t ObservationManager::collect2D3DCorrespondences(int imageId,
                                                      std::vector<int> &outKptIdx,
                                                      std::vector<int> &outPointId) const
{
    outKptIdx.clear();
    outPointId.clear();

    const int numKp = numKeypoints(imageId);
    if (numKp == 0) return 0;

    std::vector<CorrespondenceGraph::Observation> corrs;

    for (int kpt = 0; kpt < numKp; ++kpt) {
        const int own = pointIdAt(imageId, kpt);
        if (own >= 0) {
            outKptIdx.push_back(kpt);
            outPointId.push_back(own);
            continue;
        }
        if (!m_graph->hasCorrespondences(imageId, kpt)) continue;

        m_graph->findCorrespondences(imageId, kpt, &corrs);
        for (const auto &c : corrs) {
            const int pid = pointIdAt(c.first, c.second);
            if (pid >= 0) {
                outKptIdx.push_back(kpt);
                outPointId.push_back(pid);
                break;
            }
        }
    }
    return outKptIdx.size();
}

void ObservationManager::collectObservedPoints(int imageId,
                                               std::vector<int> &outPointIds) const
{
    outPointIds.clear();
    if (imageId < 0 || imageId >= static_cast<int>(m_kpTo3D.size())) return;

    std::vector<char> taken(m_points.size(), 0);
    for (const auto &kv : m_kpTo3D[static_cast<size_t>(imageId)]) {
        const int pid = kv.second;
        if (pid < 0 || pid >= static_cast<int>(taken.size())) continue;
        if (taken[static_cast<size_t>(pid)]) continue;
        if (!m_alive[static_cast<size_t>(pid)]) continue;
        taken[static_cast<size_t>(pid)] = 1;
        outPointIds.push_back(pid);
    }
}

size_t ObservationManager::totalObservations() const
{
    size_t total = 0;
    for (size_t i = 0; i < m_points.size(); ++i) {
        if (m_alive[i]) total += m_points[i].observations.size();
    }
    return total;
}

size_t ObservationManager::numTriangulatedMatches(const MatchPair &mp) const
{
    size_t count = 0;
    for (const auto &m : mp.matches) {
        if (pointIdAt(mp.imageId1, m.queryIdx) >= 0 ||
            pointIdAt(mp.imageId2, m.trainIdx) >= 0) {
            ++count;
        }
    }
    return count;
}

// ═════════════════════════════════════════════════════════════════════════════
//  外点过滤
// ═════════════════════════════════════════════════════════════════════════════

size_t ObservationManager::filterPoints3D(double maxReprojErrorPx, double minTriAngleDeg,
                                          const std::vector<int> &pointIds,
                                          const std::vector<CameraPose> &poses,
                                          const cv::Mat &K)
{
    size_t numFiltered = 0;

    for (const int pid : pointIds) {
        if (!exists(pid)) continue;

        Point3D &p = m_points[static_cast<size_t>(pid)];
        const size_t trackLen = p.observations.size();

        if (trackLen < 2) {
            numFiltered += trackLen;
            deletePoint3D(pid);
            continue;
        }

        // ── 1) 重投影误差 / cheirality（逐条观测判定）─────────────────────
        std::vector<std::pair<int, int>> toDelete;
        const cv::Vec3d Pw(p.position.x, p.position.y, p.position.z);

        for (const auto &obs : p.observations) {
            if (obs.first >= static_cast<int>(poses.size())) {
                toDelete.emplace_back(obs.first, obs.second);
                continue;
            }
            const CameraPose &pose = poses[static_cast<size_t>(obs.first)];
            if (!pose.registered || pose.R.empty() || pose.t.empty()) {
                toDelete.emplace_back(obs.first, obs.second);
                continue;
            }
            const double err = SfmGeom::reprojectionError(
                Pw, pose.R, pose.t, K, keypointPos(obs.first, obs.second));
            if (err < 0.0 || err > maxReprojErrorPx) {
                toDelete.emplace_back(obs.first, obs.second);
            }
        }

        if (toDelete.size() >= trackLen - 1) {
            // 剩下的观测已不足以支撑一个三维点
            numFiltered += trackLen;
            deletePoint3D(pid);
            continue;
        }
        if (!toDelete.empty()) {
            numFiltered += toDelete.size();
            for (const auto &obs : toDelete) {
                m_kpTo3D[static_cast<size_t>(obs.first)].erase(obs.second);
                for (auto it = p.observations.begin(); it != p.observations.end(); ++it) {
                    if (it->first == obs.first && it->second == obs.second) {
                        p.observations.erase(it);
                        break;
                    }
                }
            }
        }

        // ── 2) 三角化角下限（整点判定）───────────────────────────────────
        // COLMAP FilterPoints3DWithSmallTriangulationAngle：只有当所有观测两两
        // 之间的夹角都小于阈值时才删点（有一对足够大就保留）。
        if (minTriAngleDeg > 0.0 && exists(pid)) {
            std::vector<cv::Vec3d> centers;
            centers.reserve(p.observations.size());
            bool poseValid = true;
            for (const auto &obs : p.observations) {
                if (obs.first >= static_cast<int>(poses.size())) { poseValid = false; break; }
                const CameraPose &pose = poses[static_cast<size_t>(obs.first)];
                if (!pose.registered || pose.R.empty() || pose.t.empty()) {
                    poseValid = false;
                    break;
                }
                centers.push_back(SfmGeom::cameraCenter(pose.R, pose.t));
            }

            bool keep = false;
            if (poseValid) {
                const size_t n = std::min<size_t>(centers.size(), 12);
                for (size_t i = 0; i < n && !keep; ++i) {
                    for (size_t j = i + 1; j < n; ++j) {
                        if (SfmGeom::triangulationAngleDeg(Pw, centers[i], centers[j]) >= minTriAngleDeg) {
                            keep = true;
                            break;
                        }
                    }
                }
            }

            if (!keep) {
                numFiltered += p.observations.size();
                deletePoint3D(pid);
            }
        }
    }

    return numFiltered;
}

size_t ObservationManager::filterAllPoints3D(double maxReprojErrorPx, double minTriAngleDeg,
                                             const std::vector<CameraPose> &poses,
                                             const cv::Mat &K)
{
    std::vector<int> ids;
    ids.reserve(m_points.size());
    for (size_t i = 0; i < m_points.size(); ++i) {
        if (m_alive[i]) ids.push_back(static_cast<int>(i));
    }
    return filterPoints3D(maxReprojErrorPx, minTriAngleDeg, ids, poses, K);
}

size_t ObservationManager::filterPoints3DInImages(double maxReprojErrorPx, double minTriAngleDeg,
                                                  const std::vector<int> &imageIds,
                                                  const std::vector<CameraPose> &poses,
                                                  const cv::Mat &K)
{
    std::vector<int> ids;
    std::vector<char> taken(m_points.size(), 0);
    for (const int imageId : imageIds) {
        if (imageId < 0 || imageId >= static_cast<int>(m_kpTo3D.size())) continue;
        for (const auto &kv : m_kpTo3D[static_cast<size_t>(imageId)]) {
            const int pid = kv.second;
            if (pid >= 0 && pid < static_cast<int>(taken.size()) &&
                !taken[static_cast<size_t>(pid)]) {
                taken[static_cast<size_t>(pid)] = 1;
                ids.push_back(pid);
            }
        }
    }
    return filterPoints3D(maxReprojErrorPx, minTriAngleDeg, ids, poses, K);
}

void ObservationManager::compact(std::vector<Point3D> &outPoints,
                                 std::vector<int> *outIdMap) const
{
    outPoints.clear();
    outPoints.reserve(m_numAlive);

    if (outIdMap) {
        outIdMap->assign(m_points.size(), -1);
    }

    int newId = 0;
    for (size_t i = 0; i < m_points.size(); ++i) {
        if (!m_alive[i]) continue;
        Point3D p = m_points[i];
        p.id = newId;
        if (outIdMap) (*outIdMap)[i] = newId;
        outPoints.push_back(p);
        ++newId;
    }
}
