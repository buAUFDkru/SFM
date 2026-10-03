// ═════════════════════════════════════════════════════════════════════════════
//  recon_cli —— 稀疏重建命令行验证工具
//
//  不启动 GUI，直接对某个图像目录跑完整的稀疏链路：
//    特征提取(SIFT) → 特征匹配 → 种子对初始化 → 增量重建
//
//  输出：匹配对数、特征轨迹数、注册相机数、三维点数、每张图像的注册状态。
//  用于在真实数据集上快速判断失败发生在「匹配召回」还是「增量注册」。
//
//  用法：recon_cli.exe <图像目录> [最大图像数] [auto|sequential|exhaustive]
// ═════════════════════════════════════════════════════════════════════════════

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonObject>
#include <QStringList>

#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "algorithm/Initialization.h"
#include "algorithm/MatcherService.h"
#include "algorithm/ThreadPool.h"
#include "algorithm/Triangulate.h"
#include "builtin/registerBuiltIn.h"
#include "data/SfmTypes.h"
#include "plugin/IDescriptorPlugin.h"
#include "plugin/PluginRegistry.h"
#include "plugin/PluginTypes.h"

namespace {

/// Windows 下 qDebug 默认走 OutputDebugString，控制台看不到。
/// 这里接管消息处理器，把 Qt 日志直接打到 stdout，便于离线分析。
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

} // namespace

int main(int argc, char *argv[])
{
    qInstallMessageHandler(messageHandler);
    QCoreApplication app(argc, argv);

    const QStringList args = QCoreApplication::arguments();
    if (args.size() < 2) {
        std::printf("用法: recon_cli.exe <图像目录> [最大图像数] [auto|sequential|exhaustive]\n");
        return 2;
    }

    const QString imageDir = args[1];
    const int maxImages = args.size() > 2 ? args[2].toInt() : 0;
    const QString matchingMode = args.size() > 3 ? args[3] : QString("auto");

    PluginRegistry *registry = &PluginRegistry::instance();
    registerBuiltInAlgorithms(*registry);

    const QStringList files = collectImages(imageDir, maxImages);
    const int N = files.size();
    if (N < 3) {
        std::printf("图像不足 3 张（实际 %d），目录: %s\n", N, qPrintable(imageDir));
        return 2;
    }
    std::printf("===== 数据集: %s （%d 张）=====\n", qPrintable(imageDir), N);

    // ── 1) 特征提取 ────────────────────────────────────────────────────────
    QElapsedTimer timer; timer.start();

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
    std::printf("[特征提取] %d 张, 关键点合计 %zu, 平均 %.0f/张, %.1f s\n",
                N, totalKp, static_cast<double>(totalKp) / N,
                timer.elapsed() / 1000.0);

    // ── 2) 特征匹配 ────────────────────────────────────────────────────────
    timer.restart();
    QJsonObject matcherParams{{"ratioThreshold", 0.8},
                              {"ransacThreshold", 3.0},
                              {"minMatchCount", 30},
                              {"minMatchRatio", 0.02},
                              {"matchingMode", matchingMode},
                              {"sequenceWindow", 10},
                              {"exhaustiveMaxImages", 50}};

    // ── 可选阈值覆盖（调试用）：initMinTriAngle / minTriAngle / ignoreTwoView ──
    {
        TriangulateOptions opt = Triangulate::instance()->options();
        if (args.size() > 4) opt.initMinTriAngleDeg = args[4].toDouble();
        if (args.size() > 5) opt.minTriAngleDeg = args[5].toDouble();
        Triangulate::instance()->setOptions(opt);
        if (args.size() > 6) {
            TriangulatorOptions topt = Triangulate::instance()->triangulatorOptions();
            topt.ignoreTwoViewTracks = args[6].toInt() != 0;
            Triangulate::instance()->setTriangulatorOptions(topt);
        }
        std::printf("[参数] initMinTriAngle=%.1f° minTriAngle=%.1f° ignoreTwoView=%d\n",
                    opt.initMinTriAngleDeg, opt.minTriAngleDeg,
                    args.size() > 6 ? args[6].toInt() : 1);
    }

    sfm::ThreadPool threadPool;
    MatcherService matcherService(registry);
    std::vector<MatchPair> matchPairs;
    if (!matcherService.matchFeatures(features, "FLANN", matcherParams, matchPairs,
                                      &threadPool,
                                      [](const QString &msg) {
                                          std::printf("  %s\n", qPrintable(msg));
                                      })) {
        std::printf("错误: 特征匹配失败\n");
        return 2;
    }
    std::printf("[特征匹配] 图像对 %zu 条, %.1f s\n", matchPairs.size(),
                timer.elapsed() / 1000.0);
    if (matchPairs.empty()) {
        std::printf("没有任何匹配对，重建无法继续\n");
        return 1;
    }

    // ── 3) 内参 ────────────────────────────────────────────────────────────
    const cv::Mat K = Triangulate::instance()->estimateK(features);
    std::printf("[内参估计] fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
                K.at<double>(0, 0), K.at<double>(1, 1),
                K.at<double>(0, 2), K.at<double>(1, 2));

    // ── 4) 初始化候选 ──────────────────────────────────────────────────────
    const std::vector<InitResult> initCandidates =
        Initialization::instance()->findInitialPairCandidates(features, matchPairs, K, 3);
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

    // ── 5) 增量重建（逐个尝试种子对）────────────────────────────────────────
    ViewGraph viewGraph = ViewGraph::buildFromPairs(matchPairs, N);

    int bestReg = 0;
    size_t bestPts = 0;
    std::vector<CameraPose> bestPoses;

    for (size_t t = 0; t < initCandidates.size(); ++t) {
        timer.restart();
        std::vector<Point3D> pts;
        std::vector<CameraPose> poses;

        const bool ok = Triangulate::instance()->triangulate(
            matchPairs, features, initCandidates[t], viewGraph, pts, poses,
            false, K);

        int regCount = 0;
        for (const auto &p : poses) {
            if (p.registered) ++regCount;
        }
        std::printf("[重建] 种子对 #%zu: 成功=%d, 注册相机=%d/%d, 三维点=%zu, %.1f s\n",
                    t + 1, ok ? 1 : 0, regCount, N, pts.size(),
                    timer.elapsed() / 1000.0);

        if (regCount > bestReg || (regCount == bestReg && pts.size() > bestPts)) {
            bestReg = regCount;
            bestPts = pts.size();
            bestPoses = poses;
        }
        if (ok && regCount >= static_cast<int>(0.9 * N)) break;
    }

    // ── 6) 汇总 ────────────────────────────────────────────────────────────
    std::printf("\n========== 结果 ==========\n");
    std::printf("图像总数     : %d\n", N);
    std::printf("注册相机数   : %d (%.0f%%)\n", bestReg,
                100.0 * bestReg / static_cast<double>(N));
    std::printf("三维点数     : %zu\n", bestPts);
    std::printf("未注册图像   : ");
    int printed = 0;
    for (size_t i = 0; i < bestPoses.size(); ++i) {
        if (!bestPoses[i].registered) {
            std::printf("#%zu ", i);
            if (++printed >= 20) { std::printf("..."); break; }
        }
    }
    if (printed == 0) std::printf("(无)");
    std::printf("\n");

    return bestReg >= 3 ? 0 : 1;
}
