#include "MatcherService.h"

#include "Initialization.h"
#include "ThreadPool.h"
#include "plugin/IMatcherPlugin.h"
#include "plugin/PluginRegistry.h"

#include <QDebug>

#include <algorithm>
#include <atomic>
#include <future>
#include <optional>
#include <set>

#include <opencv2/opencv.hpp>

MatcherService::MatcherService(PluginRegistry *registry, QObject *parent)
    : QObject(parent)
    , m_registry(registry)
{
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
    workerParams.remove("ratioThreshold");
    workerParams.remove("ransacThreshold");
    workerParams.remove("minMatchCount");
    workerParams.remove("minMatchRatio");

    return plugin->createWorker(workerParams);
}

// ───  matchFeatures  特征匹配入口（编排）  ──────────────────────────────────

bool MatcherService::matchFeatures(const std::vector<FeatureResult> &features,
                                   const QString &matcherId,
                                   const QJsonObject &params,
                                   std::vector<MatchPair> &outPairs,
                                   sfm::ThreadPool *threadPool,
                                   std::function<void(const QString &)> logCallback)
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

    // 2. 提取编排通用参数
    double ratioThreshold = params.value("ratioThreshold").toDouble(0.8);
    double ransacThreshold = params.value("ransacThreshold").toDouble(3.0);
    int minMatchCount = params.value("minMatchCount").toInt(50);
    double minMatchRatio = params.value("minMatchRatio").toDouble(0.05);

    const int N = static_cast<int>(features.size());

    // 3. 序列匹配：每张图像与其后最多 5 张图像匹配
    const int sequenceWindow = 5;
    std::vector<std::pair<int, int>> candidatePairs;
    candidatePairs.reserve(N * sequenceWindow);
    for (int i = 0; i < N; ++i) {
        for (int k = 1; k <= sequenceWindow && i + k < N; ++k) {
            candidatePairs.push_back({i, i + k});
        }
    }
    qDebug().noquote()
        << QString("[匹配] 图像数=%1, 序列匹配 %2 对").arg(N).arg(candidatePairs.size());

    // 记录已经尝试过的图像对，避免序列匹配失败的组合在传递性阶段被重复尝试
    std::set<std::pair<int, int>> attempted;
    for (const auto &[i, j] : candidatePairs) {
        attempted.insert({std::min(i, j), std::max(i, j)});
    }

    // 辅助lambda：处理一批候选对，支持串行或并行
    auto processPairs = [&](const std::vector<std::pair<int, int>> &pairs) {
        if (threadPool && pairs.size() > 1) {
            // 并行路径：每个图像对独立为一个任务。
            // 每个任务通过匹配器插件工厂 createWorker() 获取独占实例，
            // 符合"工厂无状态 + Worker 逐任务独占"的插件并发规范。
            std::vector<std::future<std::optional<MatchPair>>> futures;
            futures.reserve(pairs.size());

            // 日志频率限制：使用原子计数器，每完成 5 对输出一次汇总日志
            std::atomic<int> completedCount{0};
            std::atomic<int> matchedCount{0};
            const int logInterval = 5;
            const int totalPairs = static_cast<int>(pairs.size());

            for (const auto &[i, j] : pairs) {
                futures.push_back(threadPool->submit(
                    [this, &matcherId, &params, &features, i, j,
                     ratioThreshold, ransacThreshold, minMatchCount, minMatchRatio,
                     &completedCount, &matchedCount, logInterval, totalPairs,
                     &logCallback]() -> std::optional<MatchPair> {
                         int done = completedCount.fetch_add(1) + 1;
                         auto logProgress = [&]() {
                             if (logCallback && (done % logInterval == 0 || done == totalPairs)) {
                                 logCallback(QString("[特征匹配] 进度: %1/%2 对, 累计 %3 内点")
                                                 .arg(done).arg(totalPairs)
                                                 .arg(matchedCount.load()));
                             }
                         };

                         auto localMatcher = createMatcherWorker(matcherId, params);
                         if (!localMatcher) { logProgress(); return std::nullopt; }

                         MatchPair mp;
                         if (!matchAndVerifyPair(i, j, localMatcher.get(), features,
                                                 ratioThreshold, ransacThreshold, mp)) {
                             logProgress();
                             return std::nullopt;
                         }
                         if (!shouldAddEdge(mp, features, minMatchCount, minMatchRatio)) {
                             logProgress();
                             return std::nullopt;
                         }
                         matchedCount.fetch_add(static_cast<int>(mp.matches.size()));
                         logProgress();
                         return std::move(mp);
                     }));
            }

            for (auto &f : futures) {
                auto result = f.get();
                if (result) {
                    attempted.insert({result->imageId1, result->imageId2});
                    outPairs.push_back(std::move(*result));
                }
            }
        } else {
            // 串行路径：复用同一个低层 Worker 实例
            auto matcher = createMatcherWorker(matcherId, params);
            if (!matcher) {
                return;
            }

            for (const auto &[i, j] : pairs) {
                MatchPair mp;
                if (!matchAndVerifyPair(i, j, matcher.get(), features,
                                        ratioThreshold, ransacThreshold, mp)) {
                    continue;
                }
                if (!shouldAddEdge(mp, features, minMatchCount, minMatchRatio)) {
                    continue;
                }
                attempted.insert({mp.imageId1, mp.imageId2});
                outPairs.push_back(std::move(mp));
            }
        }
    };

    // 4. 对序列候选对进行匹配、验证，并按阈值决定是否加入视图图
    processPairs(candidatePairs);

    // 5. 传递性匹配：将已验证匹配视为无向图边，对距离为 2 的节点对尝试匹配
    //    迭代直到不再新增边，或达到最大迭代次数
    const int maxTransitiveIterations = 5;
    for (int iter = 0; iter < maxTransitiveIterations; ++iter) {
        ViewGraph graph = ViewGraph::buildFromPairs(outPairs, N);

        // 收集所有图距离为 2 且尚未直接相连/未尝试过的无序节点对
        std::vector<std::pair<int, int>> distance2Pairs;
        for (int u = 0; u < N; ++u) {
            std::vector<int> uNeighbors = graph.neighbors(u);
            for (int v : uNeighbors) {
                std::vector<int> vNeighbors = graph.neighbors(v);
                for (int w : vNeighbors) {
                    if (w == u) continue;                       // 忽略回到起点的边
                    int a = std::min(u, w);
                    int b = std::max(u, w);
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

        const size_t before = outPairs.size();
        processPairs(distance2Pairs);
        const bool addedThisIter = outPairs.size() > before;

        if (!addedThisIter) {
            break; // 本迭代没有新增边，后续迭代也不会再新增
        }
    }

    qDebug().noquote()
        << QString("[匹配] 最终有效匹配对: %1").arg(outPairs.size());
    return !outPairs.empty();
}

bool MatcherService::matchAndVerifyPair(int i, int j, IMatcherWorker *matcher,
                                        const std::vector<FeatureResult> &features,
                                        double ratioThreshold,
                                        double ransacThreshold,
                                        MatchPair &outPair) const
{
    const auto &desc1 = features[i].descriptors;
    const auto &desc2 = features[j].descriptors;
    const auto &kpts1 = features[i].keypoints;
    const auto &kpts2 = features[j].keypoints;

    if (!matcher || desc1.empty() || desc2.empty()) {
        return false;
    }

    // 4a. 低层 KNN 匹配（匹配器插件 Worker）
    std::vector<std::vector<cv::DMatch>> knnMatches;
    if (!matcher->knnMatchPair(desc1, desc2, knnMatches, 2)) {
        return false;
    }

    // 4b. Lowe 比率测试：过滤歧义匹配
    std::vector<cv::DMatch> goodMatches;
    for (const auto &km : knnMatches) {
        if (km.size() >= 2) {
            if (km[0].distance < ratioThreshold * km[1].distance) {
                goodMatches.push_back(km[0]);
            }
        }
    }

    if (goodMatches.empty()) {
        return false;
    }

    // 4c. GRIC 验证：同时估计 F 和 H，用 GRIC 选择更适合当前场景的模型
    std::vector<cv::Point2d> pts1, pts2;
    pts1.reserve(goodMatches.size());
    pts2.reserve(goodMatches.size());
    for (const auto &m : goodMatches) {
        pts1.push_back(kpts1[m.queryIdx].pt);
        pts2.push_back(kpts2[m.trainIdx].pt);
    }

    std::vector<uchar> inliers;
    cv::Mat F, H;
    int modelChoice = Initialization::instance()->selectModel(
        pts1, pts2, F, H, inliers, 1.0, ransacThreshold);

    if (inliers.empty()) {
        return false;
    }

    std::vector<cv::DMatch> verifiedMatches;
    for (size_t k = 0; k < inliers.size(); ++k) {
        if (inliers[k]) {
            verifiedMatches.push_back(goodMatches[k]);
        }
    }

    if (verifiedMatches.size() < 8) {
        return false;
    }

    // 4d. 组装 MatchPair
    MatchPair mp;
    mp.imageId1 = features[i].imageId;
    mp.imageId2 = features[j].imageId;
    mp.matches = std::move(verifiedMatches);
    mp.matchCount = static_cast<int>(mp.matches.size());
    mp.modelChoice = modelChoice;
    mp.F = F.clone();
    mp.H = H.clone();

    mp.points1.reserve(mp.matches.size());
    mp.points2.reserve(mp.matches.size());

    double distSum = 0.0;
    for (const auto &m : mp.matches) {
        mp.points1.push_back(kpts1[m.queryIdx].pt);
        mp.points2.push_back(kpts2[m.trainIdx].pt);
        distSum += m.distance;
    }
    mp.avgDistance = mp.matchCount > 0 ? distSum / mp.matchCount : 0.0;

    outPair = std::move(mp);
    return true;
}

bool MatcherService::shouldAddEdge(const MatchPair &mp,
                                   const std::vector<FeatureResult> &features,
                                   int minMatchCount,
                                   double minMatchRatio) const
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