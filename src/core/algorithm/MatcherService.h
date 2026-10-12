#ifndef MATCHERSERVICE_H
#define MATCHERSERVICE_H

#include <QJsonObject>
#include <QObject>
#include <QString>

#include <functional>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "data/SfmTypes.h"
#include "plugin/ParamSchema.h"

namespace sfm { class ThreadPool; }
class IMatcherWorker;
class PluginRegistry;

/// 匹配编排参数（由 QJsonObject 解析而来，见 matchFeatures）
struct MatcherOptions
{
    double ratioThreshold   = 0.8;   ///< Lowe 比率阈值
    double ransacThresholdPx = 3.0;  ///< 几何验证阈值（像素）；未显式配置时按分辨率自适应
    int    minNumInliers    = 15;    ///< 两视图几何判有效的最少内点数
    int    minMatchCount    = 30;    ///< 建边绝对阈值
    double minMatchRatio    = 0.02;  ///< 建边相对阈值（占较少特征数图像的比例）

    bool   guidedMatching    = true; ///< 是否做极线/单应引导的二次匹配
    int    guidedKnn         = 5;    ///< 引导匹配所需的 KNN 候选数（k ≥ 2）
    double guidedRatio       = 0.9;  ///< 幸存者内部的比率阈值
    int    guidedBelowInliers = 500; ///< 仅对内点数低于该值的“弱对”做引导匹配（省时）

    bool   adaptiveEdgeThreshold = true; ///< 图不连通时逐级放宽建边阈值
    /// RANSAC 迭代上限。注意：内点率越低，达到目标置信度所需迭代越多，
    /// 10000 会让「几乎无重叠」的弱对把单次几何估计拖到 2 s 以上；
    /// 这里沿用 COLMAP 的 1000 上限 + 0.999 置信度（fewer trials，够用且快）。
    int    maxRansacTrials = 500;
    double ransacConfidence = 0.999;
};

/**
 * @brief 匹配编排服务（core 管线段）
 *
 * 把 sfm2.0 MatcherFactory::matchFeatures 的编排逻辑移植到 core：
 * 底层两两匹配不再直接创建算法实例，而是经由 PluginRegistry 获取
 * IMatcherPlugin（无状态工厂）并按任务 createWorker() 独占实例，因此
 * 线程池并行路径天然遵守"逐任务独立 Worker"的插件并发规范。
 *
 * 编排流程：
 * 1. 候选对选择：sequential（每图与其后 sequenceWindow 张图匹配，默认 10）
 *    或 exhaustive（两两全匹配）。默认 auto：图像数 ≤ exhaustiveMaxImages
 *    （默认 50）时自动采用穷举，避免无序 / 大视角间隔数据集漏掉真实重叠对
 * 2. 两两低层 KNN 匹配（插件 Worker）+ Lowe 比率过滤
 * 3. 两视图几何验证（TwoViewGeometry：已知 K 直接估 E，否则 F；同时估 H
 *    判定 PLANAR / PANORAMIC）
 * 4. 引导匹配：用已估出的 E/H 做空间约束，在被比率测试拒掉的候选里二次捞取
 *    真匹配，再重估一次几何并按最终内点集裁剪（宽基线召回的主要增量）
 * 5. 建边：先按配置阈值建图；若视图图不连通，则按阶梯逐级放宽阈值重投，
 *    直到最大连通分量不再增大为止（避免"阈值一刀切"砍断弱重叠对）
 * 6. 传递性匹配：对图距离为 2 的节点对迭代匹配，直至不再新增边
 */
class MatcherService : public QObject
{
    Q_OBJECT

public:
    explicit MatcherService(PluginRegistry *registry, QObject *parent = nullptr);

    /// 编排层参数契约（写入 .sfmproj matcher.params 的键，不含插件私有键）
    static QList<ParamSchema> orchestrationParamSchema();

    /**
     * @brief 特征匹配入口（编排层）
     *
     * @param features 特征结果列表（按图像顺序，imageId 与下标一致）
     * @param matcherId 匹配器插件 id（如 "FLANN" / "BruteForce"）
     * @param params 编排参数（QJsonObject）：
     *        ratioThreshold=0.8, ransacThreshold=自适应,
     *        minMatchCount=30, minMatchRatio=0.02,
     *        minNumInliers=15, guidedMatching=true, guidedKnn=5, guidedRatio=0.9,
     *        guidedBelowInliers=500, adaptiveEdgeThreshold=true,
     *        matchingMode=auto|sequential|exhaustive,
     *        sequenceWindow=10, exhaustiveMaxImages=50；
     *        其余键传给底层匹配器 Worker 参数
     * @param outPairs 输出的验证后匹配对
     * @param threadPool 线程池指针（可为 nullptr，走串行路径）
     * @param logCallback 日志回调（可为 nullptr）
     * @param K 相机内参（可选）；非空时走「已知 K 直接估 E」的高精度路径
     * @return 是否至少产生一对有效匹配
     */
    bool matchFeatures(const std::vector<FeatureResult> &features,
                       const QString &matcherId,
                       const QJsonObject &params,
                       std::vector<MatchPair> &outPairs,
                       sfm::ThreadPool *threadPool = nullptr,
                       std::function<void(const QString &)> logCallback = nullptr,
                       const cv::Mat &K = cv::Mat());

    /// 两两匹配 + Lowe 过滤 + 两视图几何验证（+ 可选引导匹配），组装 MatchPair
    bool matchAndVerifyPair(int i, int j, IMatcherWorker *matcher,
                            const std::vector<FeatureResult> &features,
                            const cv::Mat &K,
                            const MatcherOptions &opt,
                            MatchPair &outPair) const;

    /// 验证匹配点数/比率是否达到建边阈值
    bool shouldAddEdge(const MatchPair &mp,
                       const std::vector<FeatureResult> &features,
                       int minMatchCount, double minMatchRatio) const;

    /// 从特征结果估计相机内参 K（无信息时回退 640x480）
    cv::Mat estimateKFromFeatures(const std::vector<FeatureResult> &features) const;

    /// 由匹配器插件工厂创建独立 Worker（剥离编排参数后的参数字典传给插件）
    std::unique_ptr<IMatcherWorker> createMatcherWorker(const QString &matcherId,
                                                        const QJsonObject &params) const;

private:
    /// 批量验证一批候选对（串行或线程池并行），返回几何验证通过的 MatchPair
    std::vector<MatchPair> verifyPairs(const std::vector<std::pair<int, int>> &pairs,
                                       const std::vector<FeatureResult> &features,
                                       const QString &matcherId,
                                       const QJsonObject &params,
                                       const cv::Mat &K,
                                       const MatcherOptions &opt,
                                       sfm::ThreadPool *threadPool,
                                       std::function<void(const QString &)> logCallback) const;

    PluginRegistry *m_registry; ///< 插件注册表（只读访问，线程安全）
};

#endif // MATCHERSERVICE_H
