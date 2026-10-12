#include "MatcherService.h"

#include "Initialization.h"
#include "ThreadPool.h"
#include "TwoViewGeometry.h"
#include "plugin/IMatcherPlugin.h"
#include "plugin/ParamSchemaUtils.h"
#include "plugin/PluginRegistry.h"

#include <QDebug>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <iterator>
#include <map>
#include <optional>
#include <set>

#include <opencv2/opencv.hpp>

namespace
{

/// 视图图连通性：统计连通分量个数与最大分量规模
void graphComponents(const std::vector<MatchPair> &pairs,
                     const std::vector<FeatureResult> &features,
                     int &largestComponent,
                     int &numComponents)
{
    const int N = static_cast<int>(features.size());
    std::vector<int> indexOf(N, -1);
    for (int k = 0; k < N; ++k) {
        const int id = features[k].imageId;
        if (id >= 0 && id < N) {
            indexOf[id] = k;
        }
    }

    std::vector<std::vector<int>> adj(N);
    for (const MatchPair &mp : pairs) {
        const int a = indexOf[mp.imageId1];
        const int b = indexOf[mp.imageId2];
        if (a < 0 || b < 0 || a == b) {
            continue;
        }
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    std::vector<char> visited(N, 0);
    largestComponent = 0;
    numComponents = 0;
    for (int s = 0; s < N; ++s) {
        if (visited[s]) {
            continue;
        }
        ++numComponents;
        int count = 0;
        std::vector<int> stack{s};
        visited[s] = 1;
        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            ++count;
            for (int v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = 1;
                    stack.push_back(v);
                }
            }
        }
        largestComponent = std::max(largestComponent, count);
    }
}

const char *configName(int cfg)
{
    switch (cfg) {
    case TWO_VIEW_CALIBRATED:   return "E";
    case TWO_VIEW_UNCALIBRATED: return "F";
    case TWO_VIEW_PLANAR:       return "H/planar";
    case TWO_VIEW_PANORAMIC:    return "H/panoramic";
    case TWO_VIEW_DEGENERATE:   return "degenerate";
    default:                    return "undefined";
    }
}

} // namespace

MatcherService::MatcherService(PluginRegistry *registry, QObject *parent)
    : QObject(parent)
    , m_registry(registry)
{
}

QList<ParamSchema> MatcherService::orchestrationParamSchema()
{
    using sfm::makeBoolParam;
    using sfm::makeDoubleParam;
    using sfm::makeIntParam;
    const QString g = QStringLiteral("匹配编排");

    QList<ParamSchema> out;
    out.append(makeDoubleParam(QStringLiteral("ratioThreshold"), 0.8, 0.0, 1.0, g,
                               QStringLiteral("Lowe 比率阈值")));
    out.append(makeDoubleParam(QStringLiteral("ransacThreshold"), 3.0, 0.1, 50.0, g,
                               QStringLiteral("几何验证像素阈值；缺省时按分辨率自适应")));
    out.append(makeIntParam(QStringLiteral("minMatchCount"), 30, 1, 10000, g,
                            QStringLiteral("建边绝对内点下限")));
    out.append(makeDoubleParam(QStringLiteral("minMatchRatio"), 0.02, 0.0, 1.0, g,
                               QStringLiteral("建边相对内点比率")));
    out.append(makeIntParam(QStringLiteral("minNumInliers"), 15, 1, 10000, g,
                            QStringLiteral("两视图几何最少内点数")));
    out.append(makeBoolParam(QStringLiteral("guidedMatching"), true, g,
                             QStringLiteral("极线/单应引导二次匹配")));
    out.append(makeIntParam(QStringLiteral("guidedKnn"), 5, 2, 32, g,
                            QStringLiteral("引导匹配 KNN")));
    out.append(makeDoubleParam(QStringLiteral("guidedRatio"), 0.9, 0.0, 1.0, g,
                               QStringLiteral("引导匹配比率阈值")));
    out.append(makeIntParam(QStringLiteral("guidedBelowInliers"), 500, 0, 100000, g,
                            QStringLiteral("仅对弱对做引导匹配的内点上限")));
    out.append(makeBoolParam(QStringLiteral("adaptiveEdgeThreshold"), true, g,
                             QStringLiteral("图不连通时放宽建边阈值")));
    out.append(makeIntParam(QStringLiteral("maxRansacTrials"), 500, 100, 100000, g,
                            QStringLiteral("RANSAC 迭代上限")));
    out.append(makeDoubleParam(QStringLiteral("ransacConfidence"), 0.999, 0.5, 1.0, g,
                               QStringLiteral("RANSAC 置信度")));

    ParamSchema mode;
    mode.name = QStringLiteral("matchingMode");
    mode.type = ParamSchema::Type::Enum;
    mode.defaultValue = QStringLiteral("auto");
    mode.enumValues = {QStringLiteral("auto"), QStringLiteral("sequential"),
                       QStringLiteral("exhaustive")};
    mode.group = g;
    mode.tooltip = QStringLiteral("候选对生成模式");
    out.append(mode);

    out.append(makeIntParam(QStringLiteral("sequenceWindow"), 10, 1, 200, g,
                            QStringLiteral("序列窗口大小")));
    out.append(makeIntParam(QStringLiteral("exhaustiveMaxImages"), 20, 2, 500, g,
                            QStringLiteral("auto 模式下走穷举的最大图像数")));
    return out;
}

std::unique_ptr<IMatcherWorker> MatcherService::createMatcherWorker(
    const QString &matcherId, const QJsonObject &params) const
{
    IMatcherPlugin *plugin = m_registry ? m_registry->matcherPlugin(matcherId) : nullptr;
    if (!plugin) {
        return nullptr;
    }

    // 剥离编排级参数，其余（normType/crossCheck/trees/checks 等）传给插件 Worker
    QJsonObject workerParams = params;
    const char *orchestrationKeys[] = {
        "ratioThreshold", "ransacThreshold", "minMatchCount", "minMatchRatio",
        "minNumInliers", "guidedMatching", "guidedKnn", "guidedRatio",
        "guidedBelowInliers", "adaptiveEdgeThreshold", "matchingMode",
        "sequenceWindow", "exhaustiveMaxImages", "maxRansacTrials", "ransacConfidence"
    };
    for (const char *key : orchestrationKeys) {
        workerParams.remove(QString::fromLatin1(key));
    }

    return plugin->createWorker(workerParams);
}

// ───  matchAndVerifyPair  单对的匹配 + 几何验证 + 引导匹配  ─────────────────

namespace
{
/// 单对匹配的分阶段耗时探针（仅在 debug 开关打开时输出）
struct PairTimer
{
    double knn = 0.0, geom = 0.0, guided = 0.0, cut = 0.0;
    void dump(int i, int j, int inliers) const
    {
        if (!(std::getenv("SFM_PAIR_TIMING"))) return;
        std::fprintf(stdout,
                     "[配对计时] #%d-#%d knn=%.1f geom=%.1f guided=%.1f cut=%.1f inliers=%d\n",
                     i, j, knn, geom, guided, cut, inliers);
        std::fflush(stdout);
    }
};
} // namespace

bool MatcherService::matchAndVerifyPair(int i, int j, IMatcherWorker *matcher,
                                        const std::vector<FeatureResult> &features,
                                        const cv::Mat &K,
                                        const MatcherOptions &opt,
                                        MatchPair &outPair) const
{
    const auto clk = [] {
        return std::chrono::steady_clock::now();
    };
    PairTimer pt;

    const auto &desc1 = features[i].descriptors;
    const auto &desc2 = features[j].descriptors;
    const auto &kpts1 = features[i].keypoints;
    const auto &kpts2 = features[j].keypoints;

    if (!matcher || desc1.empty() || desc2.empty()) {
        return false;
    }

    // 1. 低层 KNN 匹配。
    //
    // 性能要点：底层 FLANN 每次 knnMatchPair 都会重建一棵索引树，单对 8000 点
    // 的 SIFT 就要 ~700 ms，其中 k=5 与 k=2 的搜索开销相近（都受叶片遍历支配）。
    // 所以固定跑 k=2（k 的大小对搜索成本影响很小，反而浪费）；引导匹配
    // 直接在 k=2 候选上做（见步骤 4）。
    std::vector<std::vector<cv::DMatch>> knnMatches;
    const auto knnT0 = clk();
    if (!matcher->knnMatchPair(desc1, desc2, knnMatches, 2)) {
        return false;
    }
    pt.knn = std::chrono::duration<double, std::milli>(clk() - knnT0).count();

    // 2. Lowe 比率测试
    std::vector<cv::DMatch> goodMatches;
    std::vector<cv::Point2d> pts1, pts2;
    std::vector<uchar> usedQuery(knnMatches.size(), 0);
    std::vector<uchar> usedTrain(kpts2.size(), 0);

    // 抓取 knnMatches 中通过 Lowe 比率测试的最近邻，追加到 out（首次调用会
    // 同时填充 usedQuery / usedTrain，深挖轮次直接复用既有占用表）
    const auto applyKnn = [&](bool rebuildUsed) {
        if (rebuildUsed) {
            std::fill(usedQuery.begin(), usedQuery.end(), 0);
            std::fill(usedTrain.begin(), usedTrain.end(), 0);
        }
        size_t before = goodMatches.size();
        for (size_t qi = 0; qi < knnMatches.size(); ++qi) {
            const auto &km = knnMatches[qi];
            if (km.size() < 2) {
                continue;
            }
            if (km[0].distance >= opt.ratioThreshold * km[1].distance) {
                continue;
            }
            if (km[0].trainIdx < 0 || km[0].trainIdx >= static_cast<int>(kpts2.size())) {
                continue;
            }
            if (rebuildUsed) {
                usedQuery[qi] = 1;
                usedTrain[km[0].trainIdx] = 1;
            } else if (!usedQuery[qi] && !usedTrain[km[0].trainIdx]) {
                usedQuery[qi] = 1;
                usedTrain[km[0].trainIdx] = 1;
            } else {
                continue; // 与首轮重复
            }
            goodMatches.push_back(km[0]);
            pts1.push_back(kpts1[km[0].queryIdx].pt);
            pts2.push_back(kpts2[km[0].trainIdx].pt);
        }
        return goodMatches.size() - before;
    };

    applyKnn(true);
    if (goodMatches.size() < 8) {
        return false;
    }

    // 3. 两视图几何验证：已知 K 时直接估 E，否则估 F；同时估 H 判 PLANAR/PANORAMIC
    TwoViewGeometryOptions gopt;
    gopt.ransacMaxErrorPx = opt.ransacThresholdPx;
    gopt.minNumInliers = opt.minNumInliers;
    gopt.maxNumTrials = opt.maxRansacTrials;
    gopt.confidence = opt.ransacConfidence;

    const auto geomT0 = clk();
    TwoViewGeometry geom = estimateTwoViewGeometry(pts1, pts2, K, gopt);
    const double geomMs = std::chrono::duration<double, std::milli>(clk() - geomT0).count();
    if (!geom.valid()) {
        pt.geom = geomMs;
        pt.dump(i, j, 0);
        return false;
    }

    // 4. 引导匹配：用已估出的模型做空间约束，在 k=2 候选里二次捞取
    //    被 Lowe 比率拒掉的真匹配后重估一次几何。
    //    （实测：真实数据上引导增益为 0，因此不再为弱对补跑深 KNN ——
    //     那会让弱对付出 2 倍 KNN + 2 倍几何的代价却一无所获）
    const auto guidedT0 = clk();
    if (opt.guidedMatching && geom.numInliers < opt.guidedBelowInliers) {
        std::vector<cv::DMatch> extraMatches;
        const int added = guidedMatching(knnMatches, pts1, pts2, geom,
                                         opt.ransacThresholdPx, opt.guidedRatio,
                                         usedQuery, usedTrain, extraMatches);
        if (added > 0) {
            for (const cv::DMatch &m : extraMatches) {
                goodMatches.push_back(m);
                pts1.push_back(kpts1[m.queryIdx].pt);
                pts2.push_back(kpts2[m.trainIdx].pt);
            }
            const TwoViewGeometry geom2 = estimateTwoViewGeometry(pts1, pts2, K, gopt);
            // 重估结果更差（例如引导进来一堆外点）则退回原模型，
            // 此时 inlierMask 只覆盖前 goodMatches.size()-added 条，新增项自然被丢弃
            if (geom2.valid() && geom2.numInliers >= geom.numInliers) {
                geom = geom2;
            }
        }
    }
    pt.geom += std::chrono::duration<double, std::milli>(clk() - guidedT0).count();

    // 5. 按内点掩码裁剪并组装 MatchPair
    const auto cutT0 = clk();
    MatchPair mp;
    mp.imageId1 = features[i].imageId;
    mp.imageId2 = features[j].imageId;

    double distSum = 0.0;
    const size_t n = std::min(goodMatches.size(), geom.inlierMask.size());
    for (size_t k = 0; k < n; ++k) {
        if (!geom.inlierMask[k]) {
            continue;
        }
        mp.matches.push_back(goodMatches[k]);
        mp.points1.push_back(pts1[k]);
        mp.points2.push_back(pts2[k]);
        distSum += goodMatches[k].distance;
    }

    mp.matchCount = static_cast<int>(mp.matches.size());
    pt.cut = std::chrono::duration<double, std::milli>(clk() - cutT0).count();
    pt.dump(i, j, mp.matchCount);
    if (mp.matchCount < 8) {
        return false;
    }
    mp.avgDistance = distSum / mp.matchCount;

    mp.modelChoice = (geom.config == TWO_VIEW_PLANAR || geom.config == TWO_VIEW_PANORAMIC) ? 1 : 0;
    mp.geometryConfig = geom.config;
    if (!geom.F.empty()) mp.F = geom.F.clone();
    if (!geom.H.empty()) mp.H = geom.H.clone();
    if (!geom.E.empty()) mp.E = geom.E.clone();

    outPair = std::move(mp);
    return true;
}

// ───  verifyPairs  批量验证（串行 / 线程池并行）  ───────────────────────────

std::vector<MatchPair> MatcherService::verifyPairs(
    const std::vector<std::pair<int, int>> &pairs,
    const std::vector<FeatureResult> &features,
    const QString &matcherId,
    const QJsonObject &params,
    const cv::Mat &K,
    const MatcherOptions &opt,
    sfm::ThreadPool *threadPool,
    std::function<void(const QString &)> logCallback) const
{
    std::vector<MatchPair> results;

    if (pairs.empty()) {
        return results;
    }

    if (threadPool && pairs.size() > 1) {
        // 并行路径：每个图像对独立为一个任务，逐任务 createWorker() 独占实例
        std::vector<std::future<std::optional<MatchPair>>> futures;
        futures.reserve(pairs.size());

        std::atomic<int> completedCount{0};
        std::atomic<int> matchedCount{0};
        const int logInterval = 5;
        const int totalPairs = static_cast<int>(pairs.size());

        for (const auto &[i, j] : pairs) {
            futures.push_back(threadPool->submit(
                [this, &matcherId, &params, &features, i, j, &K, &opt,
                 &completedCount, &matchedCount, logInterval, totalPairs,
                 &logCallback]() -> std::optional<MatchPair> {
                    const int done = completedCount.fetch_add(1) + 1;
                    auto logProgress = [&]() {
                        if (logCallback && (done % logInterval == 0 || done == totalPairs)) {
                            logCallback(QString("[特征匹配] 进度: %1/%2 对, 累计 %3 内点")
                                            .arg(done).arg(totalPairs)
                                            .arg(matchedCount.load()));
                        }
                    };

                    auto localMatcher = createMatcherWorker(matcherId, params);
                    if (!localMatcher) {
                        logProgress();
                        return std::nullopt;
                    }

                    MatchPair mp;
                    if (!matchAndVerifyPair(i, j, localMatcher.get(), features, K, opt, mp)) {
                        logProgress();
                        return std::nullopt;
                    }
                    matchedCount.fetch_add(mp.matchCount);
                    logProgress();
                    return std::move(mp);
                }));
        }

        for (auto &f : futures) {
            auto result = f.get();
            if (result) {
                results.push_back(std::move(*result));
            }
        }
    } else {
        auto matcher = createMatcherWorker(matcherId, params);
        if (!matcher) {
            return results;
        }
        for (const auto &[i, j] : pairs) {
            MatchPair mp;
            if (matchAndVerifyPair(i, j, matcher.get(), features, K, opt, mp)) {
                results.push_back(std::move(mp));
            }
        }
    }

    return results;
}

// ───  matchFeatures  特征匹配入口（编排）  ──────────────────────────────────

bool MatcherService::matchFeatures(const std::vector<FeatureResult> &features,
                                   const QString &matcherId,
                                   const QJsonObject &params,
                                   std::vector<MatchPair> &outPairs,
                                   sfm::ThreadPool *threadPool,
                                   std::function<void(const QString &)> logCallback,
                                   const cv::Mat &K)
{
    outPairs.clear();

    if (features.size() < 2) {
        return false;
    }

    // 1. 确认匹配器插件可加载（串行/并行任务中会按任务再次创建 Worker）
    auto probe = createMatcherWorker(matcherId, params);
    if (!probe) {
        return false;
    }

    // 2. 编排参数
    MatcherOptions opt;
    opt.ratioThreshold = params.value("ratioThreshold").toDouble(0.8);
    opt.minMatchCount = params.value("minMatchCount").toInt(30);
    opt.minMatchRatio = params.value("minMatchRatio").toDouble(0.02);
    opt.minNumInliers = params.value("minNumInliers").toInt(15);
    opt.guidedMatching = params.value("guidedMatching").toBool(true);
    opt.guidedKnn = params.value("guidedKnn").toInt(5);
    opt.guidedRatio = params.value("guidedRatio").toDouble(0.9);
    opt.guidedBelowInliers = params.value("guidedBelowInliers").toInt(500);
    opt.adaptiveEdgeThreshold = params.value("adaptiveEdgeThreshold").toBool(true);
    opt.maxRansacTrials = params.value("maxRansacTrials").toInt(10000);
    opt.ransacConfidence = params.value("ransacConfidence").toDouble(0.9999);

    // RANSAC 阈值：显式配置优先，否则按图像分辨率自适应（旧实现固定 3px 不随分辨率缩放）
    if (params.contains("ransacThreshold")) {
        opt.ransacThresholdPx = params.value("ransacThreshold").toDouble(3.0);
    } else {
        int maxDim = 0;
        for (const auto &f : features) {
            maxDim = std::max(maxDim, std::max(f.imageWidth, f.imageHeight));
        }
        opt.ransacThresholdPx = adaptiveRansacThresholdPx(maxDim, maxDim);
    }

    const int N = static_cast<int>(features.size());

    // 3. 候选对生成
    //    sequential：每张图像与其后 sequenceWindow 张图匹配（历史行为，窗口现可配）
    //    exhaustive：所有图像两两匹配
    //    auto：小规模数据集（≤ exhaustiveMaxImages，默认 20）自动走穷举，
    //          更大规模走序列窗口 + 传递性匹配。
    //    界限依据：穷举 ≈1.2 s/对（14 图 91 对 27 s，28 图 378 对 107 s），
    //    20 图 190 对 ≈ 55 s 是「全连接收益仍明显、耗时可接受」的平衡点；
    //    超过 20 图后 O(N²) 对数膨胀且边际增益递减，窗口 10 + 传递性匹配
    //    足以恢复非相邻重叠（COLMAP sequential_overlap 同为 10）。
    const QString matchingMode =
        params.value(QStringLiteral("matchingMode")).toString(QStringLiteral("auto")).toLower();
    const int sequenceWindow =
        params.value(QStringLiteral("sequenceWindow")).toInt(10);
    const int exhaustiveMaxImages =
        params.value(QStringLiteral("exhaustiveMaxImages")).toInt(20);
    const bool useExhaustive =
        (matchingMode == QStringLiteral("exhaustive")) ||
        (matchingMode == QStringLiteral("auto") && N <= exhaustiveMaxImages);

    std::vector<std::pair<int, int>> candidatePairs;
    if (useExhaustive) {
        candidatePairs.reserve(static_cast<size_t>(N) * (N - 1) / 2);
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                candidatePairs.push_back({i, j});
            }
        }
    } else {
        candidatePairs.reserve(static_cast<size_t>(N) * std::max(1, sequenceWindow));
        for (int i = 0; i < N; ++i) {
            for (int k = 1; k <= sequenceWindow && i + k < N; ++k) {
                candidatePairs.push_back({i, i + k});
            }
        }
    }

    if (logCallback) {
        logCallback(QString("[匹配] 图像数=%1, 候选对=%2 (%3), 几何阈值=%4 px, 引导匹配=%5")
                        .arg(N).arg(candidatePairs.size())
                        .arg(useExhaustive ? "穷举" : QString("序列窗口%1").arg(sequenceWindow))
                        .arg(opt.ransacThresholdPx, 0, 'f', 2)
                        .arg(opt.guidedMatching ? "开" : "关"));
    }
    qDebug().noquote()
        << QString("[匹配] 图像数=%1, 候选对=%2, 模式=%3, 几何阈值=%4px")
               .arg(N).arg(candidatePairs.size())
               .arg(useExhaustive ? "exhaustive" : "sequential")
               .arg(opt.ransacThresholdPx, 0, 'f', 2);

    // 记录已经尝试过的图像对，避免序列匹配失败的组合在传递性阶段被重复尝试
    std::set<std::pair<int, int>> attempted;
    for (const auto &[i, j] : candidatePairs) {
        attempted.insert({std::min(i, j), std::max(i, j)});
    }

    // 4. 几何验证（保留全部通过几何验证的对，建边阈值稍后按需放宽）
    std::vector<MatchPair> allVerified = verifyPairs(candidatePairs, features, matcherId,
                                                     params, K, opt, threadPool, logCallback);
    if (allVerified.empty()) {
        return false;
    }

    // 5. 建边阈值阶梯：先严后宽。视图图一旦不连通，后续增量重建必然丢图像
    struct EdgeLevel { int minCount; double minRatio; };
    std::vector<EdgeLevel> ladder;
    ladder.push_back({opt.minMatchCount, opt.minMatchRatio});
    if (opt.adaptiveEdgeThreshold) {
        ladder.push_back({std::max(8, opt.minMatchCount * 2 / 3), opt.minMatchRatio * 0.5});
        ladder.push_back({std::max(8, opt.minMatchCount / 3), opt.minMatchRatio * 0.25});
        ladder.push_back({8, opt.minMatchRatio * 0.1});
        ladder.push_back({8, 0.0});
    }

    auto selectEdges = [&](const EdgeLevel &lv) {
        outPairs.clear();
        for (const MatchPair &mp : allVerified) {
            if (shouldAddEdge(mp, features, lv.minCount, lv.minRatio)) {
                outPairs.push_back(mp);
            }
        }
    };

    // 6. 传递性匹配：将已验证匹配视为无向图边，对距离为 2 的节点对尝试匹配
    const int maxTransitiveIterations = 5;
    for (int iter = 0; iter < maxTransitiveIterations; ++iter) {
        selectEdges(ladder.front());
        ViewGraph graph = ViewGraph::buildFromPairs(outPairs, N);

        std::vector<std::pair<int, int>> distance2Pairs;
        for (int u = 0; u < N; ++u) {
            for (int v : graph.neighbors(u)) {
                for (int w : graph.neighbors(v)) {
                    if (w == u) continue;                       // 忽略回到起点的边
                    const int a = std::min(u, w);
                    const int b = std::max(u, w);
                    if (attempted.count({a, b})) continue;      // 已尝试过，跳过
                    if (graph.edgeWeight(u, w) > 0) continue;   // 已存在直接边
                    distance2Pairs.push_back({a, b});
                    attempted.insert({a, b});
                }
            }
        }

        if (distance2Pairs.empty()) {
            break;
        }

        qDebug().noquote()
            << QString("[匹配] 传递性迭代 %1: 发现 %2 对距离-2 候选")
                   .arg(iter + 1).arg(distance2Pairs.size());

        const size_t before = allVerified.size();
        auto more = verifyPairs(distance2Pairs, features, matcherId, params, K, opt,
                                threadPool, logCallback);
        if (more.empty()) {
            break;
        }
        allVerified.insert(allVerified.end(),
                           std::make_move_iterator(more.begin()),
                           std::make_move_iterator(more.end()));
        if (allVerified.size() == before) {
            break;
        }
    }

    // 7. 按阶梯逐级放宽建边，直到视图图连通或不再有增益
    std::vector<MatchPair> bestPairs;
    int bestLargest = -1;
    for (size_t s = 0; s < ladder.size(); ++s) {
        selectEdges(ladder[s]);
        int largest = 0, comps = 0;
        graphComponents(outPairs, features, largest, comps);

        if (logCallback) {
            logCallback(QString("[匹配] 建边档位 %1: 阈值(≥%2, ≥%3) → 边 %4 条, 连通分量 %5, 最大分量 %6/%7")
                            .arg(s + 1).arg(ladder[s].minCount)
                            .arg(ladder[s].minRatio, 0, 'f', 4)
                            .arg(outPairs.size()).arg(comps).arg(largest).arg(N));
        }

        if (comps <= 1) {
            bestPairs = outPairs;
            break;
        }
        if (largest > bestLargest) {
            bestLargest = largest;
            bestPairs = outPairs;
        } else {
            break; // 放宽没有带来连通性增益，再放宽只会引入噪声
        }
    }
    outPairs = bestPairs;

    // 8. 统计
    int sumInliers = 0;
    std::map<int, int> configHist;
    for (const MatchPair &mp : outPairs) {
        sumInliers += mp.matchCount;
        configHist[mp.geometryConfig]++;
    }
    QString cfgStr;
    for (const auto &[cfg, cnt] : configHist) {
        cfgStr += QString("%1×%2  ").arg(configName(cfg)).arg(cnt);
    }
    qDebug().noquote()
        << QString("[匹配] 最终有效匹配对: %1, 内点合计 %2, 平均 %3 | 几何类型: %4")
               .arg(outPairs.size()).arg(sumInliers)
               .arg(outPairs.empty() ? 0.0 : static_cast<double>(sumInliers) / outPairs.size(), 0, 'f', 1)
               .arg(cfgStr);
    if (logCallback) {
        logCallback(QString("[匹配] 最终 %1 对, 平均内点 %2 | 几何类型: %3")
                        .arg(outPairs.size())
                        .arg(outPairs.empty() ? 0.0 : static_cast<double>(sumInliers) / outPairs.size(), 0, 'f', 1)
                        .arg(cfgStr));
    }

    return !outPairs.empty();
}

bool MatcherService::shouldAddEdge(const MatchPair &mp,
                                   const std::vector<FeatureResult> &features,
                                   int minMatchCount, double minMatchRatio) const
{
    int count = mp.matchCount;
    if (count <= 0) {
        return false;
    }

    int i = mp.imageId1;
    int j = mp.imageId2;
    if (i < 0 || i >= static_cast<int>(features.size()) ||
        j < 0 || j >= static_cast<int>(features.size())) {
        return false;
    }

    int minFeatures = std::min(static_cast<int>(features[i].keypoints.size()),
                               static_cast<int>(features[j].keypoints.size()));
    if (minFeatures <= 0) {
        return false;
    }

    double ratio = static_cast<double>(count) / static_cast<double>(minFeatures);
    return ratio >= minMatchRatio && count >= minMatchCount;
}

cv::Mat MatcherService::estimateKFromFeatures(const std::vector<FeatureResult> &features) const
{
    if (features.empty() || features[0].imageWidth < 1 || features[0].imageHeight < 1) {
        return estimateK(640, 480);
    }
    return estimateK(features[0].imageWidth, features[0].imageHeight);
}
