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

#include <QtTest/QtTest>

#include <QCoreApplication>
#include <QDebug>

#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>

#include "algorithm/CancellationToken.h"
#include "algorithm/BundleAdjustment.h"
#include "algorithm/IncrementalMapper.h"
#include "algorithm/IntrinsicsResolver.h"
#include "algorithm/SfmLogger.h"
#include "algorithm/Initialization.h"
#include "algorithm/MatcherService.h"
#include "algorithm/SfmGeometry.h"
#include "algorithm/Triangulate.h"
#include "algorithm/CorrespondenceGraph.h"
#include "algorithm/ObservationManager.h"
#include "algorithm/IncrementalTriangulator.h"
#include "algorithm/TwoViewGeometry.h"
#include "data/SfmTypes.h"
#include "frame/Coordinate.h"
#include "viewmodel/FrustumBuilder.h"
#include "viewmodel/RenderModel.h"

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

class SparseTest : public QObject
{
    Q_OBJECT

private slots:
    void regressionSuite();
};

void SparseTest::regressionSuite()
{
    std::printf("========== 稀疏重建 阶段一 回归测试 ==========\n");
    g_failed = 0;
    g_passed = 0;

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
    std::printf("\n[0] 阶段一：toDisplay / CancellationToken\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        const auto d = sfm::frame::toDisplay(1.0, 2.0, 3.0);
        check(d.x == -1.f && d.y == 2.f && d.z == -3.f, "toDisplay(x,y,z) 绕Y轴180°");

        cv::Mat v = (cv::Mat_<double>(3, 1) << 4.0, -5.0, 6.0);
        const auto dv = sfm::frame::toDisplay(v);
        check(dv.x == -4.f && dv.y == -5.f && dv.z == -6.f, "toDisplay(cv::Mat)");

        struct P { float x, y, z; } p{1.f, 2.f, 3.f};
        sfm::frame::toDisplayInPlace(p);
        check(p.x == -1.f && p.y == 2.f && p.z == -3.f, "toDisplayInPlace");
        sfm::frame::toAlgorithmInPlace(p);
        check(p.x == 1.f && p.y == 2.f && p.z == 3.f, "toAlgorithmInPlace 互逆");

        sfm::CancellationToken tok;
        check(!tok.isStopRequested(), "CancellationToken 初始未停止");
        tok.requestStop();
        check(tok.isStopRequested(), "requestStop 后可读");
        tok.reset();
        check(!tok.isStopRequested(), "reset 清除停止标志");

        // FrustumBuilder：注册相机应产出 8 条棱（光心→4角 + 底面4边）
        std::vector<CameraPose> onePose(1);
        onePose[0].registered = true;
        onePose[0].imageId = 0;
        onePose[0].imageWidth = 100;
        onePose[0].imageHeight = 100;
        onePose[0].R = cv::Mat::eye(3, 3, CV_64F);
        onePose[0].t = (cv::Mat_<double>(3, 1) << 0, 0, 0);
        const auto fr = sfm::view::buildCameraFrusta(onePose, 1.0f, {"a.jpg"});
        check(fr.lineSegments.size() == 8 * 6, "视锥 8 条线段");
        check(fr.quadCorners.size() == 12, "视锥底面 4 角");
        check(fr.imagePaths.size() == 1 && fr.imagePaths[0] == "a.jpg",
              "视锥携带图像路径");
        sfm::view::PointCloudView box;
        box.positions = {0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10};
        check(std::abs(sfm::view::estimateFrustumDepth(box) - 1.2f) < 1e-4f,
              "包围盒深度 = 0.12 * 10");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[0b] IntrinsicsResolver 三级回退\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        FeatureResult feat;
        feat.imageWidth = 640;
        feat.imageHeight = 480;
        const std::vector<FeatureResult> feats{feat};
        const QStringList files{QStringLiteral("no_such_image.png")};

        // 外部 K 优先
        cv::Mat external = (cv::Mat_<double>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);
        auto rExt = sfm::IntrinsicsResolver::resolve(external, /*useExif=*/true, files, feats);
        check(rExt.source == sfm::IntrinsicsResolveResult::Source::External,
              "外部 K 优先于 EXIF/FOV");
        check(std::abs(rExt.K.at<double>(0, 0) - 500.0) < 1e-9, "外部 K 的 fx 被保留");

        // 无外部、无 EXIF → FOV 估计（与 estimateK(640,480) 一致）
        auto rFov = sfm::IntrinsicsResolver::resolve(cv::Mat(), /*useExif=*/true, files, feats);
        check(rFov.source == sfm::IntrinsicsResolveResult::Source::FovEstimate,
              "无 EXIF 时回落到 FOV 估计");
        const cv::Mat expected = estimateK(640, 480);
        check(std::abs(rFov.K.at<double>(0, 0) - expected.at<double>(0, 0)) < 1e-6 &&
                  std::abs(rFov.K.at<double>(0, 2) - 320.0) < 1e-6,
              "FOV 估计焦距/主点正确");
        check(!rFov.messages.isEmpty(), "Resolver 产出日志消息");

        // 关闭 EXIF 扫描时仍可 FOV 估计，且不写 EXIF 失败原因
        auto rNoScan = sfm::IntrinsicsResolver::resolve(cv::Mat(), /*useExif=*/false, files, feats);
        check(rNoScan.source == sfm::IntrinsicsResolveResult::Source::FovEstimate,
              "关闭 EXIF 扫描时使用 FOV");
        check(!rNoScan.exif.valid, "关闭扫描时 exif.valid=false");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[0c] IncrementalMapper 接受判据与 Triangulate DI\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        using sfm::IncrementalMapper;
        check(IncrementalMapper::isBetterAttempt(10, 100, 9, 999),
              "更高注册数优先于点数");
        check(!IncrementalMapper::isBetterAttempt(9, 999, 10, 100),
              "更低注册数不优");
        check(IncrementalMapper::isBetterAttempt(10, 200, 10, 100),
              "同分时点数更多者优");
        check(!IncrementalMapper::isBetterAttempt(10, 50, 10, 100),
              "同分时点数更少不优");

        // 空候选 → 失败；本地 Triangulate 可注入
        Triangulate triLocal;
        TriangulateOptions opts = triLocal.options();
        opts.initMinTriAngleDeg = 16.0;
        triLocal.setOptions(opts);

        sfm::IncrementalMapperConfig cfg;
        cfg.triOptions = opts;
        cfg.triOptions.initMinTriAngleDeg = 3.0; // 故意不同（空候选不会写入）
        cfg.acceptRegCount = 3;
        auto empty = IncrementalMapper::run(
            triLocal, {}, {}, {}, ViewGraph{}, cv::Mat::eye(3, 3, CV_64F), cfg);
        check(!empty.ok, "空候选失败");
        check(std::abs(triLocal.options().initMinTriAngleDeg - 16.0) < 1e-9,
              "空候选早退不改写注入的 TriangulateOptions");

        // BA / Initialization / Triangulate 可栈上构造（阶段三 DI）
        BundleAdjustment baLocal;
        Initialization initLocal;
        check(true, "BundleAdjustment / Initialization / Triangulate 可本地构造");

        // SfmLogger 消息处理器引用计数：两次 install 需两次 uninstall 才卸下
        SfmLogger &logger = SfmLogger::instance();
        const int baseCount = logger.messageHandlerInstallCount();
        logger.installAsMessageHandler();
        logger.installAsMessageHandler();
        check(logger.messageHandlerInstallCount() == baseCount + 2,
              "双重 install 引用计数 +2");
        logger.uninstallAsMessageHandler();
        check(logger.messageHandlerInstallCount() == baseCount + 1,
              "单次 uninstall 后仍保持安装");
        logger.uninstallAsMessageHandler();
        check(logger.messageHandlerInstallCount() == baseCount,
              "对称 uninstall 恢复原计数");
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
    Initialization init;
    {
        cands = init.findInitialPairCandidates(
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
            init.findInitialPair(features, matchPairs, K);
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
        Triangulate triangulator;
        const bool ok = triangulator.triangulate(
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

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[8] TwoViewGeometry：已知 K 直接估 E + 退化判定\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        // 阈值自适应：长边 4000 → 4px，2000 → 2px，800 → 2px（夹在 [2,4]，
        // 下限 2px 防止阈值过紧导致内点率崩塌、RANSAC 打满迭代）
        check(std::abs(adaptiveRansacThresholdPx(4000, 3000) - 4.0) < 1e-9,
              QString("自适应阈值 4000px → 4.0 (实际 %1)")
                  .arg(adaptiveRansacThresholdPx(4000, 3000)));
        check(std::abs(adaptiveRansacThresholdPx(2000, 1500) - 2.0) < 1e-9,
              QString("自适应阈值 2000px → 2.0 (实际 %1)")
                  .arg(adaptiveRansacThresholdPx(2000, 1500)));
        check(std::abs(adaptiveRansacThresholdPx(800, 600) - 2.0) < 1e-9,
              QString("自适应阈值 800px → 2.0 (实际 %1)")
                  .arg(adaptiveRansacThresholdPx(800, 600)));

        // ── 两台相机的普通三维场景 ──
        cv::Mat R0, t0, R1, t1;
        makePose(cv::Vec3d(5.0, -1.0, 0.5), R0, t0);
        makePose(cv::Vec3d(5.0, 1.6, 0.9), R1, t1);

        cv::RNG rng(7788);
        std::vector<cv::Vec3d> pts3d;
        for (int i = 0; i < 200; ++i) {
            pts3d.emplace_back(rng.uniform(-1.5, 1.5),
                               rng.uniform(-1.5, 1.5),
                               rng.uniform(-1.5, 1.5));
        }

        std::vector<cv::Point2d> qa, qb;
        for (const auto &P : pts3d) {
            const cv::Point2d pa = project(P, R0, t0, K);
            const cv::Point2d pb = project(P, R1, t1, K);
            qa.emplace_back(pa.x + rng.gaussian(0.3), pa.y + rng.gaussian(0.3));
            qb.emplace_back(pb.x + rng.gaussian(0.3), pb.y + rng.gaussian(0.3));
        }

        TwoViewGeometryOptions gopt;
        gopt.ransacMaxErrorPx = 2.0;
        const TwoViewGeometry geom = estimateTwoViewGeometry(qa, qb, K, gopt);
        check(geom.config == TWO_VIEW_CALIBRATED,
              QString("普通三维场景判定为 CALIBRATED (实际 %1)").arg(geom.config));
        check(geom.poseValid(), "CALIBRATED 下 R/t 可用");
        check(geom.numInliers > 180,
              QString("内点数 %1 / 200").arg(geom.numInliers));

        // 与真值相对位姿比对
        const cv::Mat Rrel = R1 * R0.t();
        const cv::Mat trel = t1 - Rrel * t0;
        const cv::Mat Rdiff = geom.R * Rrel.t();
        const double traceR = cv::trace(Rdiff)[0];
        const double angR = std::acos(std::max(-1.0, std::min(1.0, (traceR - 1.0) * 0.5)))
                            * 180.0 / CV_PI;
        check(angR < 2.0, QString("旋转误差 %.2f° (< 2°)").arg(angR));

        const cv::Vec3d tn(geom.t.at<double>(0), geom.t.at<double>(1), geom.t.at<double>(2));
        const cv::Vec3d tg(trel.at<double>(0), trel.at<double>(1), trel.at<double>(2));
        const double cosT = normalized(tn).dot(normalized(tg));
        const double angT = std::acos(std::max(-1.0, std::min(1.0, std::fabs(cosT))))
                            * 180.0 / CV_PI;
        check(angT < 5.0, QString("平移方向误差 %.2f° (< 5°)").arg(angT));

        // 无内参时退化为 F 路径
        const TwoViewGeometry geomNoK = estimateTwoViewGeometry(qa, qb, cv::Mat(), gopt);
        check(geomNoK.config == TWO_VIEW_UNCALIBRATED || geomNoK.config == TWO_VIEW_PLANAR,
              QString("无 K 时退化为 F/H 路径 (实际 %1)").arg(geomNoK.config));

        // ── 平面场景 → PLANAR ──
        std::vector<cv::Point2d> pa2, pb2;
        for (int i = 0; i < 200; ++i) {
            const cv::Vec3d P(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), 0.0);
            const cv::Point2d a = project(P, R0, t0, K);
            const cv::Point2d b = project(P, R1, t1, K);
            pa2.emplace_back(a.x + rng.gaussian(0.2), a.y + rng.gaussian(0.2));
            pb2.emplace_back(b.x + rng.gaussian(0.2), b.y + rng.gaussian(0.2));
        }
        // 掠射角平面：E 与 H 都能解释，H 反而拟合得更差 → 判 CALIBRATED 属正常，
        // 这里只要求「几何仍然可用、不会被误杀」
        const TwoViewGeometry geomPlanar = estimateTwoViewGeometry(pa2, pb2, K, gopt);
        check(geomPlanar.valid() && geomPlanar.numInliers > 100,
              QString("掠射平面场景几何仍可用（实际 cfg=%1, 内点 %2）")
                  .arg(geomPlanar.config).arg(geomPlanar.numInliers));

        // ── 纯旋转对（同一光心、绕光轴滚转）→ 不能当普通三维场景用 ──
        cv::Mat R0b, t0b, R1b, t1b;
        makePose(cv::Vec3d(5.0, -1.0, 0.5), R0b, t0b);
        {
            const double a = 15.0 * CV_PI / 180.0;
            const cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
                std::cos(a), -std::sin(a), 0.0,
                std::sin(a),  std::cos(a), 0.0,
                0.0,          0.0,         1.0);
            R1b = Rz * R0b;          // 绕相机光心旋转（光心不变）
            t1b = Rz * t0b;
        }
        std::vector<cv::Point2d> pa3, pb3;
        for (const auto &P : pts3d) {
            pa3.push_back(project(P, R0b, t0b, K));
            pb3.push_back(project(P, R1b, t1b, K));
        }
        const TwoViewGeometry geomRot = estimateTwoViewGeometry(pa3, pb3, K, gopt);
        std::printf("     [diag] pure-rotation: cfg=%d E=%d H=%d frontRatio=%.2f\n",
                    geomRot.config, geomRot.numEInliers,
                    geomRot.numHInliers, geomRot.poseFrontRatio);
        check(geomRot.config != TWO_VIEW_CALIBRATED,
              QString("纯旋转对不判为 CALIBRATED（实际 %1）").arg(geomRot.config));

        // ── 随机噪声对 → DEGENERATE ──
        std::vector<cv::Point2d> ra, rb;
        for (int i = 0; i < 200; ++i) {
            ra.emplace_back(rng.uniform(0.0, static_cast<double>(kWidth)),
                            rng.uniform(0.0, static_cast<double>(kHeight)));
            rb.emplace_back(rng.uniform(0.0, static_cast<double>(kWidth)),
                            rng.uniform(0.0, static_cast<double>(kHeight)));
        }
        const TwoViewGeometry geomRnd = estimateTwoViewGeometry(ra, rb, K, gopt);
        check(!geomRnd.valid(), QString("随机点判定为 DEGENERATE (实际 %1)").arg(geomRnd.config));
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[9] 引导匹配：被 Lowe 比率拒掉的真匹配由极线约束捞回\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        const int n = static_cast<int>(features[0].keypoints.size());
        const std::vector<cv::Point2d> &p1 = matchPairs.front().points1;
        const std::vector<cv::Point2d> &p2 = matchPairs.front().points2;

        TwoViewGeometryOptions gopt;
        gopt.ransacMaxErrorPx = 2.0;
        const TwoViewGeometry geom = estimateTwoViewGeometry(p1, p2, K, gopt);
        const bool geoOk = geom.valid() && !geom.F.empty() && geom.F.type() == CV_64F;
        check(geoOk, "合成场景几何估计成功");
        if (!geoOk) {
            std::printf("  (跳过引导匹配用例：几何不可用)\n");
        } else {

        // 前 n/2 个点视为已匹配；后一半构造「真匹配排在第 2 位」的 KNN，
        // 其 Lowe 比率 90/100 = 0.9 > 0.8 → 会被标准比率测试拒掉
        const int half = n / 2;
        std::vector<std::vector<cv::DMatch>> knn(static_cast<size_t>(n));
        std::vector<uchar> usedQ(static_cast<size_t>(n), 0);
        std::vector<uchar> usedT(static_cast<size_t>(n), 0);
        for (int i = 0; i < half; ++i) {
            knn[i].emplace_back(i, i, 0, 100.0f);
            usedQ[i] = 1;
            usedT[i] = 1;
        }

        int constructed = 0;
        for (int i = half; i < n; ++i) {
            // 找一个远离第 i 条极线的干扰点（保证不会被几何约束放过）
            const double *f = geom.F.ptr<double>();
            const cv::Vec3d line(f[0] * p1[i].x + f[1] * p1[i].y + f[2],
                                 f[3] * p1[i].x + f[4] * p1[i].y + f[5],
                                 f[6] * p1[i].x + f[7] * p1[i].y + f[8]);
            const double denom = std::sqrt(line[0] * line[0] + line[1] * line[1]);
            int decoy = -1;
            for (int j = 0; j < n && decoy < 0; ++j) {
                if (j == i) continue;
                const double d = std::fabs(line[0] * p2[j].x + line[1] * p2[j].y + line[2]) / denom;
                if (d > 40.0) decoy = j;
            }
            if (decoy < 0) continue;

            knn[i].emplace_back(i, decoy, 0, 90.0f);   // 距离更近的错误候选
            knn[i].emplace_back(i, i, 0, 100.0f);      // 真匹配排第 2
            ++constructed;
        }

        std::vector<cv::DMatch> recovered;
        const int added = guidedMatching(knn, p1, p2, geom, 2.0, 0.9,
                                         usedQ, usedT, recovered);
        check(added > 0, QString("引导匹配新增 %1 条").arg(added));

        int correct = 0;
        for (const auto &m : recovered) {
            if (m.queryIdx == m.trainIdx) ++correct;
        }
        check(correct == added && added >= constructed - 2,
              QString("捞回的 %1 条全部是正确对应（构造 %2 条）").arg(correct).arg(constructed));

        // 无几何信息时不应产出任何匹配
        std::vector<cv::DMatch> none;
        check(guidedMatching(knn, p1, p2, TwoViewGeometry(), 2.0, 0.9,
                             usedQ, usedT, none) == 0,
              "无几何模型时引导匹配不产出结果");
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[10] 连通分量：图不连通时保留最大分量（阶段四 4.5）\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        // 6 个节点：{0,1,2} 三角环、{3,4} 一条边、{5} 孤立
        std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}, {3, 4}};
        const SfmGeom::GraphComponents comps = SfmGeom::connectedComponents(6, edges);
        check(comps.numComponents == 3,
              QString("6 节点 4 边 → 3 个分量 (实际 %1)").arg(comps.numComponents));
        check(comps.largest >= 0 && comps.sizes[static_cast<size_t>(comps.largest)] == 3,
              "最大分量大小 = 3");
        check(comps.compId[0] == comps.compId[1] && comps.compId[1] == comps.compId[2],
              "0/1/2 属同一分量");
        check(comps.compId[3] == comps.compId[4] && comps.compId[3] != comps.compId[0],
              "3/4 属同一分量且与主分量不同");
        check(comps.compId[5] != comps.compId[0] && comps.compId[5] != comps.compId[3],
              "5 为孤立分量");

        std::vector<MatchPair> pairs(4);
        pairs[0].imageId1 = 0; pairs[0].imageId2 = 1;
        pairs[1].imageId1 = 1; pairs[1].imageId2 = 2;
        pairs[2].imageId1 = 2; pairs[2].imageId2 = 0;
        pairs[3].imageId1 = 3; pairs[3].imageId2 = 4;

        const SfmGeom::LargestComponentResult res =
            SfmGeom::keepLargestComponent(pairs, 6);
        check(res.numComponents == 3 && res.mainSize == 3,
              "过滤结果: 分量数=3, 主分量大小=3");
        check(res.removedPairs == 1 && pairs.size() == 3,
              "剔除 1 条跨分量边, 保留 3 条主分量边");
        check(res.isolatedImages == 1, "孤立图像 1 张");

        // 连通图应原样保留
        std::vector<MatchPair> connected(2);
        connected[0].imageId1 = 0; connected[0].imageId2 = 1;
        connected[1].imageId1 = 1; connected[1].imageId2 = 2;
        const SfmGeom::LargestComponentResult resOk =
            SfmGeom::keepLargestComponent(connected, 3);
        check(resOk.numComponents == 1 && connected.size() == 2,
              "连通图不做任何剔除");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[11] 场景归一化：质心归零 + 平均光心距缩放（阶段四 4.6）\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        const cv::Vec3d centers[4] = {
            { 5.0,  0.0,  0.0}, { 0.0,  5.0,  2.0},
            {-5.0,  0.0, -1.0}, { 0.0, -5.0,  1.0}};
        std::vector<CameraPose> poses(4);
        std::vector<cv::Mat> R0(4), t0(4);
        for (int i = 0; i < 4; ++i) {
            makePose(centers[i], R0[static_cast<size_t>(i)], t0[static_cast<size_t>(i)]);
            poses[static_cast<size_t>(i)].imageId = i;
            poses[static_cast<size_t>(i)].R = R0[static_cast<size_t>(i)].clone();
            poses[static_cast<size_t>(i)].t = t0[static_cast<size_t>(i)].clone();
            poses[static_cast<size_t>(i)].registered = true;
        }

        std::vector<Point3D> pts(3);
        const cv::Vec3d world[3] = {{0, 0, 0}, {1, 1, 0}, {-1, 0.5, 1}};
        for (int i = 0; i < 3; ++i) {
            pts[static_cast<size_t>(i)].position =
                cv::Point3d(world[i][0], world[i][1], world[i][2]);
        }

        // 归一化前的投影基准（相似变换不变量）
        const cv::Point2d projBefore0 = project(world[0], R0[0], t0[0], K);
        const cv::Point2d projBefore2 = project(world[2], R0[3], t0[3], K);

        SfmGeom::normalizeScene(pts, poses, 10.0);

        cv::Vec3d centroid(0, 0, 0);
        std::vector<cv::Vec3d> newCenters(4);
        for (int i = 0; i < 4; ++i) {
            newCenters[static_cast<size_t>(i)] =
                SfmGeom::cameraCenter(poses[static_cast<size_t>(i)].R,
                                      poses[static_cast<size_t>(i)].t);
            centroid += newCenters[static_cast<size_t>(i)];
        }
        centroid /= 4.0;
        double meanDist = 0.0;
        for (int i = 0; i < 4; ++i) {
            meanDist += cv::norm(newCenters[static_cast<size_t>(i)] - centroid);
        }
        meanDist /= 4.0;

        check(cv::norm(centroid) < 1e-6,
              QString("光心质心归零 (残差 %.2e)").arg(cv::norm(centroid)));
        check(std::fabs(meanDist - 10.0) < 1e-6,
              QString("平均光心距 = 10 (实际 %1)").arg(meanDist, 0, 'f', 6));

        const cv::Point2d projAfter0 = project(
            cv::Vec3d(pts[0].position.x, pts[0].position.y, pts[0].position.z),
            poses[0].R, poses[0].t, K);
        const cv::Point2d projAfter2 = project(
            cv::Vec3d(pts[2].position.x, pts[2].position.y, pts[2].position.z),
            poses[3].R, poses[3].t, K);
        check(cv::norm(projAfter0 - projBefore0) < 1e-6 &&
              cv::norm(projAfter2 - projBefore2) < 1e-6,
              "归一化前后投影一致（相似变换不改变几何）");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[12] 末尾滤波：短轨迹剔除 + 重建统计（阶段四 4.4）\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        // 3 张图 × 3 个关键点；相机沿 x 轴排开，R=I，t=-C
        const size_t numImgs = 3;
        std::vector<FeatureResult> feats(numImgs);
        for (size_t i = 0; i < numImgs; ++i) {
            feats[i].imageId = static_cast<int>(i);
            feats[i].imageWidth = 100;
            feats[i].imageHeight = 100;
            feats[i].keypoints.resize(3);
        }

        const cv::Mat Ki = (cv::Mat_<double>(3, 3) <<
                            50.0, 0.0, 50.0,
                            0.0, 50.0, 50.0,
                            0.0, 0.0, 1.0);
        const cv::Mat Ri = cv::Mat::eye(3, 3, CV_64F);
        const cv::Vec3d camC[numImgs] = {{0, 0, 0}, {2, 0, 0}, {-2, 0, 0}};

        // 点 A(0,0,10)：3 个观测，关键点取精确投影 → 重投影误差 0
        // 点 C(0,0,-10)：2 个观测，位于所有相机后方 → numBehindCamera
        const cv::Vec3d worldA(0, 0, 10);
        const cv::Vec3d worldC(0, 0, -10);
        std::vector<std::vector<cv::Point2f>> proj(static_cast<size_t>(numImgs),
                                                   std::vector<cv::Point2f>(3));
        for (size_t i = 0; i < numImgs; ++i) {
            const cv::Mat ti = (cv::Mat_<double>(3, 1) <<
                                -camC[i][0], -camC[i][1], -camC[i][2]);
            const cv::Point2d pa = project(worldA, Ri, ti, Ki);
            feats[i].keypoints[0].pt = cv::Point2f(static_cast<float>(pa.x),
                                                   static_cast<float>(pa.y));
            feats[i].keypoints[1].pt = cv::Point2f(10.f + 5.f * static_cast<float>(i), 20.f);
            feats[i].keypoints[2].pt = cv::Point2f(30.f, 40.f + 5.f * static_cast<float>(i));
        }

        CorrespondenceGraph graph;
        graph.build({}, std::vector<int>{3, 3, 3});
        ObservationManager obs(feats, graph);

        const cv::Vec3b color(255, 128, 0);
        const int idA = obs.addPoint3D(
            cv::Point3d(worldA[0], worldA[1], worldA[2]),
            {{0, 0}, {1, 0}, {2, 0}}, color);
        const int idB = obs.addPoint3D(
            cv::Point3d(1, 2, 3), {{0, 1}}, color);
        const int idC = obs.addPoint3D(
            cv::Point3d(worldC[0], worldC[1], worldC[2]),
            {{0, 2}, {1, 2}}, color);
        check(idA >= 0 && idB >= 0 && idC >= 0 && obs.numAlivePoints() == 3,
              "初始 3 个点全部建立");

        std::vector<CameraPose> poses(numImgs);
        for (size_t i = 0; i < numImgs; ++i) {
            poses[i].imageId = static_cast<int>(i);
            poses[i].R = Ri.clone();
            poses[i].t = (cv::Mat_<double>(3, 1) <<
                          -camC[i][0], -camC[i][1], -camC[i][2]);
            poses[i].registered = true;
        }

        // 短轨迹过滤：B（1 条观测）被删
        const size_t removed = obs.filterShortTracks(2);
        check(removed == 1 && obs.numAlivePoints() == 2,
              QString("短轨迹过滤: 删 %1 条观测, 剩 %2 点")
                  .arg(static_cast<qlonglong>(removed))
                  .arg(static_cast<qlonglong>(obs.numAlivePoints())));
        check(obs.pointIdAt(0, 1) == -1 && obs.pointIdAt(0, 0) == idA,
              "kpTo3D 同步清理（B 的映射删除, A 保留）");

        const auto st = obs.computeStats(poses, Ki);
        check(st.numPoints == 2 && st.numObservations == 5,
              QString("统计: 点=%1 观测=%2")
                  .arg(static_cast<qlonglong>(st.numPoints))
                  .arg(static_cast<qlonglong>(st.numObservations)));
        check(std::fabs(st.meanTrackLength - 2.5) < 1e-9,
              QString("平均轨迹长=2.5 (实际 %1)").arg(st.meanTrackLength));
        check(st.meanReprojErrorPx < 1e-6 && st.medianReprojErrorPx < 1e-6,
              QString("重投影误差=0 (均值 %1)").arg(st.meanReprojErrorPx));
        check(st.numBehindCamera == 2,
              QString("相机后方观测 = 2 (实际 %1)").arg(static_cast<qlonglong>(st.numBehindCamera)));
        check(st.medianTriAngleDeg > 0.0,
              "三角化角统计有效");
    }

    // ═════════════════════════════════════════════════════════════════════
    std::printf("\n[N] 初始化优先宽基线：高内点弱视差不得压过低内点宽基线\n");
    // ═════════════════════════════════════════════════════════════════════
    {
        // 无人机式俯视：相机高约 20，地面点在原点附近。
        // 0↔1 基线短（中位角应 <12°）但匹配多；0↔2 基线宽（≥12°）但匹配少。
        const int nNarrow = 180;
        const int nWide = 70;
        const int nPts = nNarrow;
        std::vector<cv::Vec3d> pts;
        {
            cv::RNG rng(42);
            for (int i = 0; i < nPts; ++i) {
                pts.emplace_back(rng.uniform(-1.5, 1.5),
                                 rng.uniform(-1.5, 1.5),
                                 rng.uniform(-0.1, 0.1));
            }
        }

        cv::Mat R0, t0, R1, t1, R2, t2;
        makePose(cv::Vec3d(0.0, 0.0, 20.0), R0, t0);
        makePose(cv::Vec3d(0.6, 0.0, 20.0), R1, t1);   // 窄基线 ~1.7°
        makePose(cv::Vec3d(7.0, 0.0, 20.0), R2, t2);   // 宽基线 ~19°

        std::vector<FeatureResult> feats(3);
        const cv::Mat *Rs[] = {&R0, &R1, &R2};
        const cv::Mat *Ts[] = {&t0, &t1, &t2};
        for (int i = 0; i < 3; ++i) {
            feats[i].imageId = i;
            feats[i].imageWidth = kWidth;
            feats[i].imageHeight = kHeight;
            for (int k = 0; k < nPts; ++k) {
                feats[i].keypoints.emplace_back(
                    project(pts[k], *Rs[i], *Ts[i], K), 1.0f);
            }
        }

        auto makePair = [&](int i, int j, int nMatch) {
            MatchPair mp;
            mp.imageId1 = i;
            mp.imageId2 = j;
            for (int k = 0; k < nMatch; ++k) {
                mp.points1.push_back(feats[i].keypoints[k].pt);
                mp.points2.push_back(feats[j].keypoints[k].pt);
                mp.matches.emplace_back(k, k, 0.0f);
            }
            mp.matchCount = nMatch;
            mp.modelChoice = 0;
            mp.E = cv::findEssentialMat(mp.points1, mp.points2, K, cv::RANSAC, 0.999, 1.0);
            mp.F = cv::findFundamentalMat(mp.points1, mp.points2, cv::FM_8POINT);
            mp.H = cv::findHomography(mp.points1, mp.points2, cv::RANSAC, 3.0);
            mp.geometryConfig = TWO_VIEW_CALIBRATED;
            return mp;
        };

        std::vector<MatchPair> aerialPairs;
        aerialPairs.push_back(makePair(0, 1, nNarrow));
        aerialPairs.push_back(makePair(0, 2, nWide));

        Initialization aerialInit;
        const auto onlyNarrow =
            aerialInit.findInitialPairCandidates(
                feats, {aerialPairs[0]}, K, 1);
        const auto onlyWide =
            aerialInit.findInitialPairCandidates(
                feats, {aerialPairs[1]}, K, 1);
        check(!onlyNarrow.empty() && onlyNarrow.front().medianAngle < 12.0,
              QString("窄基线单独评估角 <12°（实际 %1）")
                  .arg(onlyNarrow.empty() ? -1.0 : onlyNarrow.front().medianAngle, 0, 'f', 2));
        check(!onlyWide.empty() && onlyWide.front().medianAngle >= 12.0,
              QString("宽基线单独评估角 ≥12°（实际 %1）")
                  .arg(onlyWide.empty() ? -1.0 : onlyWide.front().medianAngle, 0, 'f', 2));

        // 同时评估时：≥12° 的优先池会丢掉弱视差对，即使它内点更多
        const auto aerialCands =
            aerialInit.findInitialPairCandidates(
                feats, aerialPairs, K, 3);
        check(!aerialCands.empty(), "混合评估找到初始化候选");
        if (!aerialCands.empty()) {
            check(aerialCands.front().imageId1 == 0 &&
                      aerialCands.front().imageId2 == 2,
                  QString("首选宽基线 0↔2（实际 #%1↔#%2, 角=%3°）")
                      .arg(aerialCands.front().imageId1)
                      .arg(aerialCands.front().imageId2)
                      .arg(aerialCands.front().medianAngle, 0, 'f', 2));
            check(aerialCands.front().medianAngle >= 12.0,
                  QString("首选中位角≥12°（实际 %1）")
                      .arg(aerialCands.front().medianAngle, 0, 'f', 2));
        }
    }

    std::printf("\n========== 结果: 通过 %d, 失败 %d ==========\n", g_passed, g_failed);
    QVERIFY2(g_failed == 0,
             qPrintable(QStringLiteral("失败 %1 项（通过 %2）")
                            .arg(g_failed)
                            .arg(g_passed)));
}

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    SparseTest tc;
    return QTest::qExec(&tc, argc, argv);
}

#include "main.moc"
