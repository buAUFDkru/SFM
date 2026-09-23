#ifndef MATCHERSERVICE_H
#define MATCHERSERVICE_H

#include <QJsonObject>
#include <QObject>
#include <QString>

#include <functional>
#include <memory>
#include <vector>

#include "data/SfmTypes.h"

namespace sfm { class ThreadPool; }
class IMatcherWorker;
class PluginRegistry;

/**
 * @brief 匹配编排服务（core 管线段）
 *
 * 把 sfm2.0 MatcherFactory::matchFeatures 的编排逻辑移植到 core：
 * 底层两两匹配不再直接创建算法实例，而是经由 PluginRegistry 获取
 * IMatcherPlugin（无状态工厂）并按任务 createWorker() 独占实例，因此
 * 线程池并行路径天然遵守"逐任务独立 Worker"的插件并发规范。
 *
 * 编排流程（与 sfm2.0 完全一致）：
 * 1. 候选对选择：序列窗口 5（每图与后最多 5 张图匹配）
 * 2. 两两低层 KNN 匹配（插件 Worker）+ Lowe 比率过滤
 * 3. GRIC 几何验证（Initialization::selectModel 同时估计 F/H 并择优）
 * 4. 按 minMatchCount / minMatchRatio 阈值判定是否构成视图图边
 * 5. 传递性匹配：对图距离为 2 的节点对迭代匹配，直至不再新增边
 */
class MatcherService : public QObject
{
    Q_OBJECT

public:
    explicit MatcherService(PluginRegistry *registry, QObject *parent = nullptr);

    /**
     * @brief 特征匹配入口（编排层）
     *
     * @param features 特征结果列表（按图像顺序，imageId 与下标一致）
     * @param matcherId 匹配器插件 id（如 "FLANN" / "BruteForce"）
     * @param params 编排参数（QJsonObject）：
     *        ratioThreshold=0.8, ransacThreshold=3.0,
     *        minMatchCount=50, minMatchRatio=0.05；其余键传给底层匹配器 Worker 参数
     * @param outPairs 输出的验证后匹配对
     * @param threadPool 线程池指针（可为 nullptr，走串行路径）
     * @param logCallback 日志回调（可为 nullptr）
     * @return 是否至少产生一对有效匹配
     */
    bool matchFeatures(const std::vector<FeatureResult> &features,
                       const QString &matcherId,
                       const QJsonObject &params,
                       std::vector<MatchPair> &outPairs,
                       sfm::ThreadPool *threadPool = nullptr,
                       std::function<void(const QString &)> logCallback = nullptr);

    /// 两两匹配 + Lowe 过滤 + GRIC 验证，组装 MatchPair；成功返回 true
    bool matchAndVerifyPair(int i, int j, IMatcherWorker *matcher,
                            const std::vector<FeatureResult> &features,
                            double ratioThreshold, double ransacThreshold,
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
    PluginRegistry *m_registry; ///< 插件注册表（只读访问，线程安全）
};

#endif // MATCHERSERVICE_H