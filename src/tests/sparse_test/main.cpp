/**
 * @file main.cpp
 * @brief 稀疏重建「阶段一健壮性改造」回归测试
 *
 * 构造一个视角变化很大的合成场景（4 台相机环绕目标、相邻视角约 90°），
 * 验证以下改造点：
 *
 * 1. TriangulateOptions 默认值与 COLMAP 阈值一致
 * 2. findInitialPairCandidates 返回按评分降序的多个候选（支持多轮重试）
 * 3. 增量重建不再因单张图像的局部失败而整体终止 —— 4 台相机全部成功注册
 * 4. 外部提供的内参 K 在三角测量阶段被正确沿用（此前会被无条件重新估计覆盖）
 * 5. MatcherService::shouldAddEdge 在放宽后的阈值下判定正确
 *
 * 测试使用无噪声的合成数据，因此期望"全部相机注册成功"；
 * 真实数据上的宽容度由 TriangulateOptions 控制。
 */

#include <QCoreApplication>
#include <QDebug>

#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>

#include "algorithm/Initialization.h"
#include "algorithm/MatcherService.h"
#include "algorithm/Triangulate.h"
#include "algorithm/CorrespondenceGraph.h"
#include "algorithm/ObservationManager.h"
#include "algorithm/IncrementalTriangulator.h"
#include "data/SfmTypes.h"

namespace {

int g_failed = 0;
int g_passed = 0;

void check(bool cond, const QString &name)
{
    if (cond) {
        ++g_passed;
        std::printf("  [PASS] %s\n", name.toStdString().c_str());
    } else {
        ++g_failed;
        std::printf("  [FAIL] %s\n", name.toStdString().c_str());
    }
}

cv::Vec3d normalized(const cv::Vec3d &v)
{
    const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return (n < 1e-12) ? cv::Vec3d(0, 0, 0)
                       : cv::Vec3d(v[0] / n, v[1] / n, v[2] / n);
}

/// 构造"相机位于 C、朝向原点"的外参：P_cam = R * P_world + t
void makePose(const cv::Vec3d &C, cv::Mat &R, cv::Mat &t)
{
    const cv::Vec3d fwd = normalized(cv::Vec3d(-C[0], -C[1], -C[2]));
    cv::Vec3d up(0.0, 0.0, 1.0);
    cv::Vec3d right = fwd.cross(up);
    if (cv::norm(right) < 1e-6) {
        up = cv::Vec3d(0.0, 1.0, 0.0);
        right = fwd.cross(up);
    }
    right = normalized(right);
    const cv::Vec3d down = fwd.cross(right);

    R = (cv::Mat_<double>(3, 3) <<
         right[0], right[1], right[2],
         down[0],  down[1],  down[2],
         fwd[0],   fwd[1],   fwd[2]);

    const cv::Mat Cm = (cv::Mat_<double>(3, 1) << C[0], C[1], C[2]);
    t = -R * Cm;
}

cv::Point2d project(const cv::Vec3d &Pw, const cv::Mat &R, const cv::Mat &t,
                    const cv::Mat &K)
{
    const cv::Mat Pm = (cv::Mat_<double>(3, 1) << Pw[0], Pw[1], Pw[2]);
    const cv::Mat Pc = R * Pm + t;
    const double z = Pc.at<double>(2, 0);
    return cv::Point2d(K.at<double>(0, 0) * Pc.at<double>(0, 0) / z + K.at<double>(0, 2),
                       K.at<double>(1, 1) * Pc.at<double>(1, 0) / z + K.at<double>(1, 2));
}

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);

    std::printf("========== 稀疏重建 阶段一 回归测试 ==========\n");

    // ── 场景参数 ──────────────────────────────────────────────────────────
    const int kWidth = 1280, kHeight = 720;
    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        800.0, 0.0,  640.0,
        0.0,   800.0, 360.0,
        0.0,   0.0,  1.0);

    const int numCams = 4;
    const int numPts = 150;
    const double radius = 8.0;

    std::vector<cv::Mat> Rs(numCams), ts(numCams);
    for (int i = 0; i < numCams; ++i) {
        const double a = 2.0 * CV_PI * i / numCams;
        makePose(cv::Vec3d(radius * std::cos(a), radius * std::sin(a), 1.5), Rs[i], ts[i]);
    }

    // 合成三维点（目标附近的立方体内），固定种子保证可复现
    std::vector<cv::Vec3d> worldPts;
    {
        cv::RNG rng(20261003);
        for (int i = 0; i < numPts; ++i) {
            worldPts.emplace_back(rng.uniform(-1.0, 1.0),
                                  rng.uniform(-1.0, 1.0),
                                  rng.uniform(-1.0, 1.0));
        }
    }

    // ── 特征：每台相机都能看到全部点（关键点索引 == 三维点索引）──
    std::vector<FeatureResult> features(static_cast<size_t>(numCams));
    for (int i = 0; i < numCams; ++i) {
        features[i].imageId = i;
        features[i].imageWidth = kWidth;
        features[i].imageHeight = kHeight;
        for (int k = 0; k < numPts; ++k) {
            features[i].keypoints.emplace_back(project(worldPts[k], Rs[i], ts[i], K), 1.0f);
        }
    }

    // ── 匹配对：所有相机两两相连，匹配完整 ──
    std::vector<MatchPair> matchPairs;
    for (int i = 0; i < numCams; ++i) {
        for (int j = i + 1; j < numCams; ++j) {
            std::vector<cv::Point2d> p1, p2;
            for (int k = 0; k < numPts; ++k) {
                p1.push_back(features[i].keypoints[k].pt);
                p2.push_back(features[j].keypoints[k].pt);
            }

            MatchPair mp;
            mp.imageId1 = i;
            mp.imageId2 = j;
            mp.points1 = p1;
            mp.points2 = p2;
            for (int k = 0; k < numPts; ++k) {
                mp.matches.emplace_back(k, k, 0.0f);
            }
            mp.matchCount = numPts;
            mp.modelChoice = 0;   // 非平面场景 → F 模型
            mp.F = cv::findFundamentalMat(p1, p2, cv::FM_8POINT);
            mp.H = cv::findHomography(p1, p2, cv::RANSAC, 3.0);
            matchPairs.push_back(mp);
        }
    }

    // ── 4 相机场景的特征轨迹对应图（阶段二测试用）─────────────────────────
    CorrespondenceGraph graph4;
    {
        std::vector<int> numKp4(static_cast<size_t>(numCams), numPts);
        graph4.build(matchPairs, numKp4);
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[1] TriangulateOptions 默认值（对齐 COLMAP）\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        const TriangulateOptions o;
        check(std::abs(o.minTriAngleDeg - 1.5) < 1e-9,
              QString("minTriAngleDeg == 1.5 (实际 %1)").arg(o.minTriAngleDeg));
        check(o.absPoseMinNumInliers == 30,
              QString("absPoseMinNumInliers == 30 (实际 %1)").arg(o.absPoseMinNumInliers));
        check(std::abs(o.absPoseMinInlierRatio - 0.25) < 1e-9,
              QString("absPoseMinInlierRatio == 0.25 (实际 %1)").arg(o.absPoseMinInlierRatio));
        check(std::abs(o.absPoseMaxErrorPx - 12.0) < 1e-9,
              QString("absPoseMaxErrorPx == 12.0 (实际 %1)").arg(o.absPoseMaxErrorPx));
        check(o.maxRegTrials == 3,
              QString("maxRegTrials == 3 (实际 %1)").arg(o.maxRegTrials));
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[2] findInitialPairCandidates 多候选与排序\n");
    // ═════════════════════════════════════════════════════════════════════
    std::vector<InitResult> cands;
    {
        cands = Initialization::instance()->findInitialPairCandidates(
            features, matchPairs, K, 3);

        check(!cands.empty(), "找到初始化候选");
        check(static_cast<int>(cands.size()) <= 3,
              QString("候选数量受 topK 限制 (实际 %1)").arg(cands.size()));

        bool sorted = true;
        for (size_t i = 1; i < cands.size(); ++i) {
            if (cands[i - 1].score < cands[i].score) sorted = false;
        }
        check(sorted, "候选按评分降序排列");

        const InitResult single =
            Initialization::instance()->findInitialPair(features, matchPairs, K);
        check(single.score >= 0.0, "findInitialPair 返回有效结果");
        if (!cands.empty()) {
            check(single.imageId1 == cands.front().imageId1 &&
                  single.imageId2 == cands.front().imageId2,
                  "findInitialPair 与候选列表首位一致");
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[3] 增量重建：大视角合成场景全部相机注册成功\n");
    // ═════════════════════════════════════════════════════════════════════
    if (!cands.empty()) {
        const ViewGraph viewGraph =
            ViewGraph::buildFromPairs(matchPairs, numCams);

        std::vector<Point3D> pts;
        std::vector<CameraPose> poses;
        const bool ok = Triangulate::instance()->triangulate(
            matchPairs, features, cands.front(), viewGraph, pts, poses,
            /*fixIntrinsics=*/true, K);

        int regCount = 0;
        for (const auto &p : poses) {
            if (p.registered) ++regCount;
        }

        check(ok, "triangulate 返回成功");
        check(regCount == numCams,
              QString("全部 %1 台相机注册成功 (实际 %2)").arg(numCams).arg(regCount));
        check(!pts.empty(), QString("生成三维点 (%1 个)").arg(pts.size()));

        // 内参固定时，三角测量应沿用外部传入的 K
        bool kPreserved = true;
        for (const auto &p : poses) {
            if (!p.registered || p.K.empty()) continue;
            if (std::abs(p.K.at<double>(0, 0) - 800.0) > 1e-6) kPreserved = false;
        }
        check(kPreserved, "外部提供的内参 K 被正确沿用（fx=800 未被重新估计覆盖）");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[4] shouldAddEdge 阈值判定（放宽后的默认值）\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        MatcherService svc(nullptr);

        std::vector<FeatureResult> fs(2);
        fs[0].keypoints.resize(1000);
        fs[1].keypoints.resize(1000);

        MatchPair mp;
        mp.imageId1 = 0;
        mp.imageId2 = 1;

        mp.matchCount = 30;
        check(svc.shouldAddEdge(mp, fs, 30, 0.02), "30 匹配 / 3% 比率 → 建边");

        mp.matchCount = 29;
        check(!svc.shouldAddEdge(mp, fs, 30, 0.02), "29 匹配 → 不建边（低于 minMatchCount）");

        mp.matchCount = 15;
        check(!svc.shouldAddEdge(mp, fs, 30, 0.02), "15 匹配 → 不建边（比率 1.5% 低于 2%）");

        // 旧的严格阈值（50 / 0.05）下，30 个匹配会被拒绝 —— 说明宽基线场景确实被放宽所救
        mp.matchCount = 30;
        check(!svc.shouldAddEdge(mp, fs, 50, 0.05), "旧阈值 50/0.05 下 30 匹配被拒（对照）");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[5] CorrespondenceGraph：并查集合并特征轨迹\n");
    // ═════════════════════════════════════════════════════════════════════
    CorrespondenceGraph graph;
    {
        // 3 张图 × 5 个关键点
        //  kpt0：三视图轨迹（0↔1↔2）
        //  kpt1：两视图轨迹（0↔1）
        std::vector<MatchPair> mps;
        auto addPair = [&](int a, int b, std::vector<std::pair<int, int>> pairs) {
            MatchPair mp;
            mp.imageId1 = a;
            mp.imageId2 = b;
            for (const auto &pr : pairs) {
                mp.matches.emplace_back(pr.first, pr.second, 0.0f);
            }
            mp.matchCount = static_cast<int>(pairs.size());
            mps.push_back(mp);
        };
        addPair(0, 1, {{0, 0}, {1, 1}});
        addPair(1, 2, {{0, 0}});

        graph.build(mps, {5, 5, 5});

        check(graph.numTracks() == 2,
              QString("轨迹数 == 2 (实际 %1)").arg(graph.numTracks()));
        check(graph.hasCorrespondences(0, 0) && !graph.hasCorrespondences(0, 4),
              "孤立关键点无对应，参与匹配的关键点有对应");

        std::vector<CorrespondenceGraph::Observation> corrs;
        graph.findCorrespondences(0, 0, &corrs);
        check(corrs.size() == 2,
              QString("kpt0 的直接对应 == 2 条 (实际 %1)").arg(corrs.size()));

        check(!graph.isTwoViewObservation(0, 0), "三视图轨迹 isTwoViewObservation == false");
        check(graph.isTwoViewObservation(0, 1), "两视图轨迹 isTwoViewObservation == true");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[6] ObservationManager：可见性统计 / 轨迹合并\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        std::vector<FeatureResult> fs(3);
        for (int i = 0; i < 3; ++i) {
            fs[i].imageId = i;
            fs[i].imageWidth = kWidth;
            fs[i].imageHeight = kHeight;
            for (int k = 0; k < 5; ++k) {
                fs[i].keypoints.emplace_back(cv::Point2d(100.0 * k, 50.0 * i), 1.0f);
            }
        }

        ObservationManager obs(fs, graph);

        const int p0 = obs.addPoint3D(cv::Point3d(0, 0, 10), {{0, 0}, {1, 0}}, cv::Vec3b(1, 2, 3));
        check(p0 == 0, "addPoint3D 返回 id 0");
        check(obs.pointIdAt(0, 0) == p0 && obs.pointIdAt(1, 0) == p0,
              "kpTo3D 双向建立");
        check(obs.pointIdAt(2, 0) == -1, "未三角化的关键点返回 -1");

        // 图像 2 的 kpt0 与已三角化观测同属一条轨迹 → 可见
        check(obs.numVisiblePoints3D(2) == 1,
              QString("图像2 可见三维点 == 1 (实际 %1)").arg(obs.numVisiblePoints3D(2)));
        check(obs.numVisiblePoints3D(1) == 1, "图像1 可见三维点 == 1");

        std::vector<int> kptIdx, ptId;
        obs.collect2D3DCorrespondences(2, kptIdx, ptId);
        check(kptIdx.size() == 1 && ptId.size() == 1 && ptId[0] == p0,
              QString("图像2 收集到 1 条 2D-3D 对应 (实际 %1)").arg(kptIdx.size()));

        // 合并两条独立轨迹
        const int p1 = obs.addPoint3D(cv::Point3d(1, 0, 10), {{1, 3}, {2, 3}}, cv::Vec3b(4, 5, 6));
        const size_t aliveBefore = obs.numAlivePoints();
        const int merged = obs.mergePoints3D(p0, p1);
        check(merged == p0, "mergePoints3D 返回保留的点 id");
        check(obs.numAlivePoints() == aliveBefore - 1, "合并后存活点数减 1");
        check(obs.point(merged).observations.size() == 4,
              QString("合并后轨迹长度 == 4 (实际 %1)")
                  .arg(obs.point(merged).observations.size()));
        check(obs.pointIdAt(2, 3) == merged, "被并入的关键点重定向到合并后的点");
        check(!obs.exists(p1), "被合并的点已删除");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[7] IncrementalTriangulator：新建点 + 轨迹补全\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        std::vector<CameraPose> poses(static_cast<size_t>(numCams));
        std::vector<FeatureResult> fs(static_cast<size_t>(numCams));
        for (int i = 0; i < numCams; ++i) {
            poses[i].imageId = i;
            poses[i].imageWidth = kWidth;
            poses[i].imageHeight = kHeight;
            poses[i].K = K.clone();
            poses[i].R = Rs[static_cast<size_t>(i)].clone();
            poses[i].t = ts[static_cast<size_t>(i)].clone();
            poses[i].registered = false;

            fs[i].imageId = i;
            fs[i].imageWidth = kWidth;
            fs[i].imageHeight = kHeight;
            for (int k = 0; k < numPts; ++k) {
                fs[i].keypoints.emplace_back(project(worldPts[static_cast<size_t>(k)],
                                                    Rs[static_cast<size_t>(i)],
                                                    ts[static_cast<size_t>(i)], K), 1.0f);
            }
        }

        ObservationManager obs(fs, graph4);
        IncrementalTriangulator tri(graph4, obs, fs, poses, K);

        // 只注册前两台 → 三角化应产出接近 numPts 个点
        poses[0].registered = true;
        poses[1].registered = true;
        const size_t n0 = tri.triangulateImage(TriangulatorOptions(), 0);
        const size_t n1 = tri.triangulateImage(TriangulatorOptions(), 1);
        check(obs.numAlivePoints() > static_cast<size_t>(numPts) * 0.8,
              QString("两台相机三角化出 %1 个点（期望接近 %2）")
                  .arg(obs.numAlivePoints()).arg(numPts));
        check(n0 + n1 >= static_cast<size_t>(numPts),
              QString("新增/续接观测 %1 条（≥%2）").arg(n0 + n1).arg(numPts));

        // 注册第三台 → CompleteTracks 应把已有轨迹扩展到第三张图
        poses[2].registered = true;
        std::vector<int> ids;
        ids.reserve(obs.allPoints().size());
        for (size_t i = 0; i < obs.allPoints().size(); ++i) {
            if (obs.aliveMask()[i]) ids.push_back(static_cast<int>(i));
        }
        const size_t completed = tri.completeTracks(TriangulatorOptions(), ids);

        size_t maxTrackLen = 0;
        for (const int pid : ids) {
            if (!obs.exists(pid)) continue;
            maxTrackLen = std::max(maxTrackLen, obs.point(pid).observations.size());
        }
        check(completed > 0, QString("补全观测 %1 条").arg(completed));
        check(maxTrackLen >= 3,
              QString("轨迹被扩展到 ≥3 张图像 (实际最长 %1)").arg(maxTrackLen));

        // 重三角化：所有图像对三角化率已接近 1，不应重复造点
        const size_t aliveBefore = obs.numAlivePoints();
        tri.retriangulate(TriangulatorOptions(), matchPairs);
        check(obs.numAlivePoints() == aliveBefore,
              "重三角化不会给已充分重建的图像对重复造点");
    }

    std::printf("\n========== 结果: 通过 %d, 失败 %d ==========\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
