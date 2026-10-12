// ═════════════════════════════════════════════════════════════════════════════
//  recon_cli —— 稀疏重建命令行验证 / 基准 harness（阶段 0）
//
//  不启动 GUI，直接对某个图像目录跑完整的稀疏链路：
//    特征提取(SIFT) → 特征匹配 → 种子对初始化 → 增量重建
//
//  输出：匹配对数、特征轨迹数、注册相机数、三维点数、每张图像的注册状态。
//  可选：--metrics-out <path.json> 写出自比层五项指标，供 CI 漂移告警消费。
//
//  用法：
//    recon_cli.exe <图像目录> [最大图像数] [auto|sequential|exhaustive]
//                  [--metrics-out path.json] [--dataset-id name]
//                  [--poses-out path.json]
// ═════════════════════════════════════════════════════════════════════════════

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#include "BenchmarkMetrics.h"
#include "algorithm/IncrementalMapper.h"
#include "algorithm/Initialization.h"
#include "algorithm/MatcherService.h"
#include "algorithm/SfmGeometry.h"
#include "algorithm/ThreadPool.h"
#include "algorithm/Triangulate.h"
#include "builtin/registerBuiltIn.h"
#include "data/ExifReader.h"
#include "data/SfmTypes.h"
#include "pipeline/ProjectStore.h"
#include "plugin/IDescriptorPlugin.h"
#include "plugin/PluginRegistry.h"
#include "plugin/PluginTypes.h"

namespace {

void messageHandler(QtMsgType, const QMessageLogContext &, const QString &msg)
{
    std::printf("%s\n", msg.toLocal8Bit().constData());
    std::fflush(stdout);
}

QStringList collectImages(const QString &dir, int maxImages)
{
    QDir d(dir);
    const QStringList filters = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"};
    QStringList files = d.entryList(filters, QDir::Files, QDir::Name);
    for (int i = 0; i < files.size(); ++i) {
        files[i] = d.absoluteFilePath(files[i]);
    }
    if (maxImages > 0 && files.size() > maxImages) {
        files = files.mid(0, maxImages);
    }
    return files;
}

double peakWorkingSetMb()
{
#ifdef Q_OS_WIN
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

double medianReprojectionError(const std::vector<Point3D> &points,
                               const std::vector<CameraPose> &poses,
                               const std::vector<FeatureResult> &features,
                               const cv::Mat &fallbackK)
{
    std::vector<double> errs;
    errs.reserve(points.size() * 2);
    for (const auto &pt : points) {
        const cv::Vec3d Pw(pt.position.x, pt.position.y, pt.position.z);
        for (const auto &obs : pt.observations) {
            const int imageId = obs.first;
            const int kpIdx = obs.second;
            if (imageId < 0 || imageId >= static_cast<int>(poses.size())) continue;
            if (!poses[static_cast<size_t>(imageId)].registered) continue;
            if (imageId >= static_cast<int>(features.size())) continue;
            const auto &kps = features[static_cast<size_t>(imageId)].keypoints;
            if (kpIdx < 0 || kpIdx >= static_cast<int>(kps.size())) continue;

            const CameraPose &pose = poses[static_cast<size_t>(imageId)];
            const cv::Mat &K = pose.K.empty() ? fallbackK : pose.K;
            const cv::Point2d xy(kps[static_cast<size_t>(kpIdx)].pt.x,
                                kps[static_cast<size_t>(kpIdx)].pt.y);
            const double e = SfmGeom::reprojectionError(Pw, pose.R, pose.t, K, xy,
                                                        pose.distortion);
            if (e >= 0.0) errs.push_back(e);
        }
    }
    return sfm::bench::medianOf(std::move(errs));
}

struct CliOptions {
    QString imageDir;
    int maxImages = 0;
    QString matchingMode = QStringLiteral("auto");
    QString metricsOut;
    QString posesOut;
    QString pointsOut;
    QString datasetId;
    // 兼容旧位置参数：可选阈值覆盖
    bool hasInitMinTri = false;
    bool hasMinTri = false;
    bool hasIgnoreTwoView = false;
    double initMinTriAngle = 0.0;
    double minTriAngle = 0.0;
    int ignoreTwoView = 1;
};

void printUsage()
{
    std::printf(
        "用法: recon_cli.exe <图像目录> [最大图像数] [auto|sequential|exhaustive]\n"
        "                  [--metrics-out path.json] [--dataset-id name]\n"
        "                  [--poses-out path.json] [--points-out path.json]\n"
        "                  [initMinTriAngle] [minTriAngle] [ignoreTwoView]\n");
}

bool parseArgs(const QStringList &args, CliOptions *opt, QString *error)
{
    if (args.size() < 2) {
        if (error) *error = QStringLiteral("缺少图像目录");
        return false;
    }
    opt->imageDir = args[1];

    int positional = 0;
    for (int i = 2; i < args.size(); ++i) {
        const QString &a = args[i];
        if (a == QStringLiteral("--metrics-out")) {
            if (i + 1 >= args.size()) {
                if (error) *error = QStringLiteral("--metrics-out 需要路径参数");
                return false;
            }
            opt->metricsOut = args[++i];
            continue;
        }
        if (a.startsWith(QStringLiteral("--metrics-out="))) {
            opt->metricsOut = a.mid(QStringLiteral("--metrics-out=").size());
            continue;
        }
        if (a == QStringLiteral("--dataset-id")) {
            if (i + 1 >= args.size()) {
                if (error) *error = QStringLiteral("--dataset-id 需要名称参数");
                return false;
            }
            opt->datasetId = args[++i];
            continue;
        }
        if (a.startsWith(QStringLiteral("--dataset-id="))) {
            opt->datasetId = a.mid(QStringLiteral("--dataset-id=").size());
            continue;
        }
        if (a == QStringLiteral("--poses-out")) {
            if (i + 1 >= args.size()) {
                if (error) *error = QStringLiteral("--poses-out 需要路径参数");
                return false;
            }
            opt->posesOut = args[++i];
            continue;
        }
        if (a.startsWith(QStringLiteral("--poses-out="))) {
            opt->posesOut = a.mid(QStringLiteral("--poses-out=").size());
            continue;
        }
        if (a == QStringLiteral("--points-out")) {
            if (i + 1 >= args.size()) {
                if (error) *error = QStringLiteral("--points-out 需要路径参数");
                return false;
            }
            opt->pointsOut = args[++i];
            continue;
        }
        if (a.startsWith(QStringLiteral("--points-out="))) {
            opt->pointsOut = a.mid(QStringLiteral("--points-out=").size());
            continue;
        }
        if (a.startsWith(QLatin1Char('-'))) {
            if (error) *error = QStringLiteral("未知选项: %1").arg(a);
            return false;
        }

        // 位置参数：maxImages / matchingMode / initMinTri / minTri / ignoreTwoView
        if (positional == 0) {
            opt->maxImages = a.toInt();
        } else if (positional == 1) {
            opt->matchingMode = a;
        } else if (positional == 2) {
            opt->hasInitMinTri = true;
            opt->initMinTriAngle = a.toDouble();
        } else if (positional == 3) {
            opt->hasMinTri = true;
            opt->minTriAngle = a.toDouble();
        } else if (positional == 4) {
            opt->hasIgnoreTwoView = true;
            opt->ignoreTwoView = a.toInt();
        }
        ++positional;
    }

    if (opt->datasetId.isEmpty()) {
        opt->datasetId = QFileInfo(opt->imageDir).fileName();
        if (opt->datasetId.isEmpty()) opt->datasetId = QStringLiteral("unnamed");
    }
    return true;
}

} // namespace

int main(int argc, char *argv[])
{
    qInstallMessageHandler(messageHandler);
    QCoreApplication app(argc, argv);

    CliOptions opt;
    QString parseError;
    if (!parseArgs(QCoreApplication::arguments(), &opt, &parseError)) {
        std::printf("%s\n", qPrintable(parseError));
        printUsage();
        return 2;
    }

    PluginRegistry *registry = &PluginRegistry::instance();
    registerBuiltInAlgorithms(*registry);

    const QStringList files = collectImages(opt.imageDir, opt.maxImages);
    const int N = files.size();
    if (N < 3) {
        std::printf("图像不足 3 张（实际 %d），目录: %s\n", N, qPrintable(opt.imageDir));
        return 2;
    }
    std::printf("===== 数据集: %s （%d 张）=====\n", qPrintable(opt.imageDir), N);

    sfm::bench::BenchmarkMetrics metrics;
    metrics.datasetId = opt.datasetId;
    metrics.datasetPath = QDir(opt.imageDir).absolutePath();
    metrics.numImages = N;

    QElapsedTimer totalTimer;
    totalTimer.start();
    QElapsedTimer timer;
    timer.start();

    // ── 1) 特征提取 ────────────────────────────────────────────────────────
    IDescriptorPlugin *descPlugin = registry->descriptorPlugin("SIFT");
    if (!descPlugin) {
        std::printf("错误: 未找到 SIFT 描述符插件\n");
        return 2;
    }

    QJsonObject descParams{{"nOctaveLayers", 3},
                           {"contrastThreshold", 0.04},
                           {"maxFeatures", 8192}};

    std::vector<FeatureResult> features(static_cast<size_t>(N));
    size_t totalKp = 0;
    for (int i = 0; i < N; ++i) {
        cv::Mat color = cv::imread(files[i].toStdString(), cv::IMREAD_COLOR);
        if (color.empty()) {
            std::printf("错误: 无法读取 %s\n", qPrintable(files[i]));
            return 2;
        }
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

        ImageEntry entry;
        entry.id = i;
        entry.filePath = files[i];
        entry.imageData = color;
        entry.imageGray = gray;

        auto worker = descPlugin->createWorker(descParams);
        if (!worker) {
            std::printf("错误: createWorker 失败\n");
            return 2;
        }
        FeatureResult fr = worker->extract(entry);
        fr.imageId = i;
        fr.imageWidth = gray.cols;
        fr.imageHeight = gray.rows;
        fr.imageData = color;
        features[static_cast<size_t>(i)] = std::move(fr);
        totalKp += features[static_cast<size_t>(i)].keypoints.size();
    }
    const double extractMs = static_cast<double>(timer.elapsed());
    metrics.stageTimingsMs["feature_extract"] = extractMs;
    std::printf("[特征提取] %d 张, 关键点合计 %zu, 平均 %.0f/张, %.1f s\n",
                N, totalKp, static_cast<double>(totalKp) / N, extractMs / 1000.0);

    // ── 2) 特征匹配 ────────────────────────────────────────────────────────
    cv::Mat K;
    SfmExif::IntrinsicsGuess exifInfo;
    {
        exifInfo = SfmExif::estimateFromExif(files, features);
        if (exifInfo.valid) {
            K = exifInfo.K.clone();
            std::printf("[EXIF焦距] 采用EXIF内参初值: f=%.1fpx (35mm等效 %.1fmm, "
                        "%d/%d 张图含EXIF, 离散度 %.1f~%.1fpx)\n",
                        exifInfo.focalPx, exifInfo.focal35Mm,
                        exifInfo.numWithExif, exifInfo.numImages,
                        exifInfo.focalPxMin, exifInfo.focalPxMax);
        } else {
            std::printf("[EXIF焦距] 不可用（%s），回落按图像尺寸估计 K\n",
                        qPrintable(exifInfo.reason));
            K = estimateK(features[0].imageWidth, features[0].imageHeight);
        }
    }
    std::printf("[内参估计] fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
                K.at<double>(0, 0), K.at<double>(1, 1),
                K.at<double>(0, 2), K.at<double>(1, 2));

    timer.restart();
    QJsonObject matcherParams{{"ratioThreshold", 0.8},
                              {"minMatchCount", 30},
                              {"minMatchRatio", 0.02},
                              {"minNumInliers", 15},
                              {"guidedMatching", true},
                              {"guidedKnn", 5},
                              {"guidedRatio", 0.9},
                              {"guidedBelowInliers", 500},
                              {"adaptiveEdgeThreshold", true},
                              {"matchingMode", opt.matchingMode},
                              {"sequenceWindow", 10},
                              {"exhaustiveMaxImages", 20}};

    Triangulate triangulator;
    sfm::IncrementalMapperConfig mapCfg;
    mapCfg.triOptions = triangulator.options();
    if (opt.hasInitMinTri) mapCfg.triOptions.initMinTriAngleDeg = opt.initMinTriAngle;
    if (opt.hasMinTri) mapCfg.triOptions.minTriAngleDeg = opt.minTriAngle;
    {
        const QByteArray envDist = qgetenv("SFM_REFINE_DISTORTION");
        if (!envDist.isEmpty()) mapCfg.triOptions.refineDistortion = envDist.toInt() != 0;
        const QByteArray envAxis = qgetenv("SFM_FOCAL_PER_AXIS");
        if (!envAxis.isEmpty()) mapCfg.triOptions.refineFocalPerAxis = envAxis.toInt() != 0;
        const QByteArray envMinCam = qgetenv("SFM_INTRINSICS_MIN_CAMERAS");
        if (!envMinCam.isEmpty()) mapCfg.triOptions.intrinsicsMinCameras = envMinCam.toInt();
    }
    mapCfg.triOptions.intrinsicsTrusted = exifInfo.valid;
    mapCfg.triangulatorOptions = triangulator.triangulatorOptions();
    if (opt.hasIgnoreTwoView) {
        mapCfg.triangulatorOptions.ignoreTwoViewTracks = opt.ignoreTwoView != 0;
        mapCfg.applyTriangulatorOptions = true;
    }
    std::printf("[参数] initMinTriAngle=%.1f° minTriAngle=%.1f° ignoreTwoView=%d "
                "refineDistortion=%d fx!=fy=%d minCam=%d trusted=%d\n",
                mapCfg.triOptions.initMinTriAngleDeg, mapCfg.triOptions.minTriAngleDeg,
                opt.hasIgnoreTwoView ? opt.ignoreTwoView : 1,
                mapCfg.triOptions.refineDistortion ? 1 : 0,
                mapCfg.triOptions.refineFocalPerAxis ? 1 : 0,
                mapCfg.triOptions.intrinsicsMinCameras,
                mapCfg.triOptions.intrinsicsTrusted ? 1 : 0);

    const unsigned int cliThreads = std::thread::hardware_concurrency();
    sfm::ThreadPool threadPool(sfm::make_cpu_pool_options(cliThreads));
    std::printf("[线程池] 工作线程数: %u\n", cliThreads);
    MatcherService matcherService(registry);
    std::vector<MatchPair> matchPairs;
    if (!matcherService.matchFeatures(features, "FLANN", matcherParams, matchPairs,
                                      &threadPool,
                                      [](const QString &msg) {
                                          std::printf("  %s\n", qPrintable(msg));
                                      },
                                      K)) {
        std::printf("错误: 特征匹配失败\n");
        return 2;
    }
    const double matchMs = static_cast<double>(timer.elapsed());
    metrics.stageTimingsMs["matching"] = matchMs;
    metrics.matchPairs = static_cast<int>(matchPairs.size());
    std::printf("[特征匹配] 图像对 %zu 条, %.1f s\n", matchPairs.size(), matchMs / 1000.0);
    if (matchPairs.empty()) {
        std::printf("没有任何匹配对，重建无法继续\n");
        return 1;
    }

    int mainCompSize = N;
    {
        const SfmGeom::LargestComponentResult comp =
            SfmGeom::keepLargestComponent(matchPairs, N);
        mainCompSize = comp.mainSize;
        if (comp.numComponents > 1) {
            std::printf("[视图图] 匹配图不连通: %d 个分量, 最大分量 %d/%d 张图, "
                        "剔除跨分量匹配对 %zu 条, 孤立图像 %d 张\n",
                        comp.numComponents, comp.mainSize, N,
                        comp.removedPairs, comp.isolatedImages);
            std::printf("[视图图] 仅在最大连通分量上重建\n");
        }
    }
    metrics.mainComponentSize = mainCompSize;

    // ── 3) 初始化候选 ──────────────────────────────────────────────────────
    timer.restart();
    Initialization initialization;
    const std::vector<InitResult> initCandidates =
        initialization.findInitialPairCandidates(features, matchPairs, K, 3);
    const double initMs = static_cast<double>(timer.elapsed());
    metrics.stageTimingsMs["initialization"] = initMs;
    if (initCandidates.empty()) {
        std::printf("错误: 未找到任何初始化候选对\n");
        return 1;
    }
    std::printf("[初始化] 候选种子对 %zu 组: ", initCandidates.size());
    for (const auto &c : initCandidates) {
        std::printf("#%d↔#%d(%.1f°,%.2f) ", c.imageId1, c.imageId2,
                    c.medianAngle, c.score);
    }
    std::printf("\n");

    // ── 5) 增量重建（IncrementalMapper：种子对重试 + 接受判据）──────────────
    ViewGraph viewGraph = ViewGraph::buildFromPairs(matchPairs, N);
    mapCfg.acceptRegCount = mainCompSize;

    timer.restart();
    const sfm::IncrementalMapperResult mapResult = sfm::IncrementalMapper::run(
        triangulator, initCandidates, matchPairs, features, viewGraph, K, mapCfg,
        nullptr,
        [](const QString &msg) { std::printf("%s\n", qPrintable(msg)); });
    const double incrementalMs = static_cast<double>(timer.elapsed());
    metrics.stageTimingsMs["incremental"] = incrementalMs;

    const int bestReg = mapResult.bestRegCount;
    const size_t bestPts = mapResult.bestPointCount;
    std::vector<CameraPose> bestPoses = mapResult.poses;
    std::vector<Point3D> bestPoints = mapResult.points;

    const double totalMs = static_cast<double>(totalTimer.elapsed());
    metrics.stageTimingsMs["total"] = totalMs;
    metrics.registeredCameras = bestReg;
    metrics.registrationRate =
        N > 0 ? static_cast<double>(bestReg) / static_cast<double>(N) : 0.0;
    metrics.totalPoints = static_cast<int>(bestPts);
    metrics.medianReprojectionError =
        medianReprojectionError(bestPoints, bestPoses, features, K);
    metrics.peakMemoryMb = peakWorkingSetMb();

    // ── 6) 汇总 ────────────────────────────────────────────────────────────
    std::printf("\n========== 结果 ==========\n");
    std::printf("图像总数     : %d\n", N);
    std::printf("注册相机数   : %d (%.0f%%)\n", bestReg,
                100.0 * bestReg / static_cast<double>(N));
    std::printf("三维点数     : %zu\n", bestPts);
    std::printf("中位重投影误差: %.4f px\n", metrics.medianReprojectionError);
    std::printf("峰值内存     : %.1f MB\n", metrics.peakMemoryMb);
    std::printf("总耗时       : %.1f s\n", totalMs / 1000.0);
    std::printf("未注册图像   : ");
    int printed = 0;
    for (size_t i = 0; i < bestPoses.size(); ++i) {
        if (!bestPoses[i].registered) {
            std::printf("#%zu ", i);
            if (++printed >= 20) {
                std::printf("...");
                break;
            }
        }
    }
    if (printed == 0) std::printf("(无)");
    std::printf("\n");

    if (!opt.metricsOut.isEmpty()) {
        QString writeError;
        if (!sfm::bench::writeMetricsJson(metrics, opt.metricsOut, &writeError)) {
            std::printf("错误: %s\n", qPrintable(writeError));
            return 2;
        }
        std::printf("[指标] 已写入 %s\n", qPrintable(opt.metricsOut));
    }

    if (!opt.posesOut.isEmpty()) {
        QString writeError;
        if (!ProjectStore::writeCameraPoses(opt.posesOut, bestPoses, files, &writeError)) {
            std::printf("错误: %s\n", qPrintable(writeError));
            return 2;
        }
        std::printf("[位姿] 已写入 %s\n", qPrintable(opt.posesOut));
    }

    // 手性诊断：光轴是否朝向点云质心；点是否在相机前方（正深度）
    if (!bestPoints.empty() && bestReg >= 3) {
        double pcx = 0, pcy = 0, pcz = 0;
        for (const auto &p : bestPoints) {
            pcx += p.position.x;
            pcy += p.position.y;
            pcz += p.position.z;
        }
        const double invN = 1.0 / static_cast<double>(bestPoints.size());
        pcx *= invN;
        pcy *= invN;
        pcz *= invN;

        int lookAtCloud = 0, regN = 0, posDepth = 0, depthSamples = 0;
        for (const auto &pose : bestPoses) {
            if (!pose.registered || pose.R.empty() || pose.t.empty())
                continue;
            ++regN;
            const cv::Mat C = -pose.R.t() * pose.t;
            const double cx = C.at<double>(0), cy = C.at<double>(1), cz = C.at<double>(2);
            // 光轴 = R^T * (0,0,1) = R 的第三行
            const double vx = pose.R.at<double>(2, 0);
            const double vy = pose.R.at<double>(2, 1);
            const double vz = pose.R.at<double>(2, 2);
            const double tx = pcx - cx, ty = pcy - cy, tz = pcz - cz;
            const double tn = std::sqrt(tx * tx + ty * ty + tz * tz) + 1e-18;
            const double vn = std::sqrt(vx * vx + vy * vy + vz * vz) + 1e-18;
            if ((vx * tx + vy * ty + vz * tz) / (vn * tn) > 0.0)
                ++lookAtCloud;

            // 抽样前 200 个点的深度符号
            const int sample = std::min(200, static_cast<int>(bestPoints.size()));
            for (int i = 0; i < sample; ++i) {
                const auto &P = bestPoints[static_cast<size_t>(i)].position;
                const cv::Mat X = (cv::Mat_<double>(3, 1) << P.x, P.y, P.z);
                const cv::Mat Xc = pose.R * X + pose.t;
                ++depthSamples;
                if (Xc.at<double>(2) > 0.0)
                    ++posDepth;
            }
        }
        std::printf("\n========== 手性诊断 ==========\n");
        std::printf("点云质心     : (%.3f, %.3f, %.3f)\n", pcx, pcy, pcz);
        std::printf("光轴朝向点云 : %d/%d (应接近全部；过低=朝外/手性反)\n",
                    lookAtCloud, regN);
        std::printf("正深度观测   : %d/%d (应接近全部；过低=点在相机背后)\n",
                    posDepth, depthSamples);
    }

    if (!opt.pointsOut.isEmpty() && !bestPoints.empty()) {
        QJsonArray arr;
        const size_t step = std::max<size_t>(1, bestPoints.size() / 5000);
        for (size_t i = 0; i < bestPoints.size(); i += step) {
            const auto &p = bestPoints[i];
            arr.append(QJsonArray{p.position.x, p.position.y, p.position.z});
        }
        QJsonObject root;
        root[QStringLiteral("count")] = static_cast<int>(bestPoints.size());
        root[QStringLiteral("points")] = arr;
        QFile f(opt.pointsOut);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            std::printf("错误: 无法写入点云 %s\n", qPrintable(opt.pointsOut));
            return 2;
        }
        f.write(QJsonDocument(root).toJson(QJsonDocument::Compact));
        f.close();
        std::printf("[点云] 已写入 %s (抽样 %d / %zu)\n",
                    qPrintable(opt.pointsOut), arr.size(), bestPoints.size());
    }

    return bestReg >= 3 ? 0 : 1;
}
