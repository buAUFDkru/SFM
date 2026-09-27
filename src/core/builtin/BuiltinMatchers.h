#ifndef BUILTINMATCHERS_H
#define BUILTINMATCHERS_H

#include <QJsonObject>

#include <opencv2/core.hpp>

#include "plugin/IMatcherPlugin.h"

/**
 * @file BuiltinMatchers.h
 * @brief OpenCV 自带匹配器算法的内置实现（编译进 core，非 DLL 插件）。
 *
 * 与 DLL 插件实现相同的 IMatcherPlugin 接口，经 PluginRegistry::registerBuiltIn
 * 注册后，管线与 UI 无差别使用。Worker 并发模型不变：逐线程 / 任务独占。
 */

// ── FLANN ───────────────────────────────────────────────────────────────────

/**
 * @brief FLANN 匹配 Worker（低层两两匹配接口）。
 *
 * 构造时从 params 读取参数并持有。逐线程 / 任务独占，禁止跨线程共享使用。
 */
class FlannMatcherWorker : public IMatcherWorker
{
public:
    explicit FlannMatcherWorker(const QJsonObject &params);

    bool knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                      std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                      int k) override;
    bool matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                   std::vector<cv::DMatch> &outMatches) override;

private:
    int m_trees = 4;
    int m_checks = 50; // 预留：保持与 sfm2.0 输出一致，未在调用中使用
};

/// FLANN 内置算法（无状态工厂）
class FlannMatcherPlugin : public IMatcherPlugin
{
    Q_OBJECT
    // Qt 6 的 qobject_cast<IMatcherPlugin*> 按 IID 查找，
    // Q_INTERFACES 生成对应 qt_metacast 分支（无需 Q_PLUGIN_METADATA）
    Q_INTERFACES(IMatcherPlugin)
public:
    QString name() const override { return QStringLiteral("FLANN"); }
    QString category() const override { return QStringLiteral("matcher"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于快速近似最近邻搜索(FLANN)的特征匹配");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IMatcherWorker> createWorker(const QJsonObject &params) override;
};

// ── BruteForce ──────────────────────────────────────────────────────────────

/**
 * @brief BruteForce 匹配 Worker（低层两两匹配接口）。
 *
 * 构造时从 params 读取参数并持有。逐线程 / 任务独占，禁止跨线程共享使用。
 */
class BruteForceMatcherWorker : public IMatcherWorker
{
public:
    explicit BruteForceMatcherWorker(const QJsonObject &params);

    bool knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                      std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                      int k) override;
    bool matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                   std::vector<cv::DMatch> &outMatches) override;

private:
    /// 根据 normType 与描述符类型选择距离度量
    int autoNormType(const cv::Mat &desc) const;

    int m_normType = cv::NORM_L2;   ///< 默认 L2（浮点描述子）；-1 表示按描述符类型自动选择
    bool m_crossCheck = false;
};

/// BruteForce 内置算法（无状态工厂）
class BruteForceMatcherPlugin : public IMatcherPlugin
{
    Q_OBJECT
    Q_INTERFACES(IMatcherPlugin)
public:
    QString name() const override { return QStringLiteral("BruteForce"); }
    QString category() const override { return QStringLiteral("matcher"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于暴力搜索的特征匹配，支持 Hamming/L2 等距离度量");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IMatcherWorker> createWorker(const QJsonObject &params) override;
};

// ── NearestNeighbor（归一化描述子余弦最近邻，参考 hloc NearestNeighborMatcher） ──

/**
 * @brief NearestNeighbor 匹配 Worker。
 *
 * 对 L2 归一化描述子计算余弦相似度矩阵（内部对输入行归一化，幂等），
 * 输出 DMatch.distance = sqrt(2*(1-cos))——与 FLANN/BF 的 L2 距离语义一致
 * （归一化向量的欧氏距离），保证管线侧 Lowe 比率测试语义正确。
 * 支持双向一致性校验（mutual check，对齐 BruteForce crossCheck 先例）
 * 与可选距离阈值过滤；比率测试由管线负责（规范 §2.3）。
 * 逐线程 / 任务独占，禁止跨线程共享使用。
 */
class NearestNeighborMatcherWorker : public IMatcherWorker
{
public:
    explicit NearestNeighborMatcherWorker(const QJsonObject &params);

    bool knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                      std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                      int k) override;
    bool matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                   std::vector<cv::DMatch> &outMatches) override;

private:
    bool m_mutualCheck = true;
    double m_distanceThreshold = 0.0;   ///< sqrt(2*(1-cos)) 距离阈值；<=0 表示不过滤
};

/// NearestNeighbor 内置算法（无状态工厂）
class NearestNeighborMatcherPlugin : public IMatcherPlugin
{
    Q_OBJECT
    Q_INTERFACES(IMatcherPlugin)
public:
    QString name() const override { return QStringLiteral("NearestNeighbor"); }
    QString category() const override { return QStringLiteral("matcher"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于归一化描述子余弦相似度的最近邻匹配，支持双向一致性校验(mutual check)");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IMatcherWorker> createWorker(const QJsonObject &params) override;
};

#endif // BUILTINMATCHERS_H
