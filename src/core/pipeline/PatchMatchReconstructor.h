#ifndef PATCHMATCHRECONSTRUCTOR_H
#define PATCHMATCHRECONSTRUCTOR_H

#include "IDenseReconstructor.h"
#include "algorithm/dense/PatchMatchTypes.h"
#include "algorithm/dense/strategies.h"
#include "data/StereoTypes.h"

#include <QString>
#include <QMap>
#include <QVariant>
#include <opencv2/core.hpp>
#include <random>
#include <vector>

namespace sfm { class ThreadPool; }  // 前向声明，避免头文件暴露 ThreadPool 实现

/**
 * @brief 参考 COLMAP PatchMatch 的多视图密集重建器（CPU 实现）
 *
 * 核心思路（参考 colmap::mvs::PatchMatch）：
 *   - 不做极线校正，直接在原始图像坐标系下用 Homography warping 匹配
 *   - 代价函数使用 NCC（对线性光照变化 a·I+b 不变），鲁棒于光照差异
 *   - 每像素搜索深度（暂不搜索法向，假设平面平行于图像平面）
 *   - 随机初始化 → 邻域传播 → 随机细化（指数衰减扰动）
 *   - 多视图代价取最小值（简化版，colmap 用 HMM 动态选择源视图）
 */
class PatchMatchReconstructor : public IDenseReconstructor
{
public:
    /**
     * @brief PatchMatch 参数集合
     *
     * 字段命名与默认值尽量与 COLMAP 的 PatchMatchOptions 保持一致，
     * 方便后续直接替换为 COLMAP CUDA 实现。
     */
    struct Options {
        double depthMin = -1.0;                  ///< 最小深度，<0 时从稀疏点自动估计
        double depthMax = -1.0;                  ///< 最大深度，<0 时从稀疏点自动估计
        double sigmaSpatial = -1.0;              ///< 空间权重sigma，<0 时取 windowRadius
        double sigmaColor = 51.0;                ///< 颜色权重sigma（NCC双边加权，uint8[0,255]图像）
        double nccSigma = 0.6;                   ///< NCC似然函数扩散系数
        double minTriangulationAngle = 0.5;      ///< 选源图像最小光轴夹角(度)，共轴数据集需调低
        double incidentAngleSigma = 0.9;         ///< 入射角似然函数扩散系数
        double geomConsistencyRegularizer = 0.3; ///< 几何一致性项相对权重（保留，暂未使用）
        double geomConsistencyMaxCost = 3.0;     ///< 几何一致性最大代价(像素)（保留，暂未使用）
        double filterMinNcc = 0.1;               ///< 过滤最小NCC阈值
        double filterMinTriangulationAngle = 3.0;///< 过滤最小三角化角度(度)
        double filterGeomConsistencyMaxCost = 1.0;///< 过滤几何一致性最大代价(像素)（保留，暂未使用）
        double cacheSize = 32.0;                 ///< 缓存大小(GB)（保留，CPU实现忽略）
        QString gpuIndex = "-1";                 ///< GPU索引，CPU实现忽略
        int maxImageSize = 1000;                 ///< 最大工作图像尺寸（CPU实现强制限制）
        int windowRadius = 5;                    ///< NCC窗口半径
        int windowStep = 1;                      ///< NCC窗口步长（保留，暂用1）
        int numSamples = 15;                     ///< 每像素随机采样数
        int numIterations = 5;                   ///< 传播迭代轮数
        int filterMinNumConsistent = 2;          ///< 最小一致源视图数
        bool geomConsistency = false;            ///< 是否启用几何一致性（暂未实现）
        bool filter = true;                      ///< 是否启用过滤
        bool allowMissingFiles = false;          ///< 是否允许缺失文件
        bool writeConsistencyGraph = false;      ///< 是否输出一致性图

        void loadFromParams(const QMap<QString, QVariant> &params);
    };

    /**
     * @brief PatchMatch 单视问题定义
     *
     * 参考 COLMAP PatchMatch::Problem：一个参考图像 + 若干源图像。
     */
    struct Problem {
        int refImageIdx = -1;           ///< 参考图像索引
        std::vector<int> srcImageIdxs;  ///< 源图像索引列表
    };

    PatchMatchReconstructor() = default;
    ~PatchMatchReconstructor() override = default;

    QString methodName() const override { return QStringLiteral("PatchMatch"); }
    /// ADR-004：V1 保留作精度对照基准，标记 Deprecated 而非删除
    QString displayName() const override
    {
        return QStringLiteral("PatchMatch 多视图匹配 (Deprecated)");
    }

    void setConfig(const QMap<QString, QVariant> &params) override;
    QList<ParamSchema> paramSchema() const override;
    DenseReconstructionOutput reconstruct(const DenseReconstructionInput &input) override;

private:
    /// PatchMatch 运行时上下文（避免长参数列表）
    struct Context {
        cv::Mat refGray;            ///< 参考灰度图
        cv::Mat K_ref;              ///< 参考内参（已缩放）
        std::vector<sfm::dense::SourceView> sources;
        double depthMin = 0.0;
        double depthMax = 1.0;
        int rows = 0;
        int cols = 0;
        int refImageIdx = -1;       ///< 参考图像索引（诊断用）
        sfm::ThreadPool *pool = nullptr;  ///< 外部传入的线程池（由 reconstruct 创建，整个重建过程复用）
    };

    Options m_options;

    /// ADR-004 具名预设（顺序传播 + NCC + 无先验/JBU）
    sfm::dense::PatchMatchStrategySet m_strategies = sfm::dense::presetSequential();

    /// 为每个参考图像构建源视图集合
    std::vector<Problem> buildProblems(const DenseReconstructionInput &input) const;

    /// 估计每张图像的深度范围 [depthMin, depthMax]
    std::vector<std::pair<double, double>> estimateDepthRanges(
        const DenseReconstructionInput &input) const;

    /// 求解单个 PatchMatch 问题，输出参考视角的深度图结果
    StereoPairResult solveProblem(const DenseReconstructionInput &input,
                                  const Problem &problem,
                                  const std::pair<double, double> &depthRange,
                                  sfm::ThreadPool &pool) const;

    /// 运行 PatchMatch 核心算法：随机初始化 + 传播 + 细化
    /// @param ctx 上下文（参考图、源视图、内参、深度范围）
    /// @param[out] confidence 每像素代价（越小越可信）
    /// @return 深度图 (CV_32FC1)
    cv::Mat runPatchMatch(const Context &ctx, cv::Mat &confidence) const;

    /// 处理单个像素：传播（读邻居）+ 退火式随机细化 + 写回
    /// 红黑网格并行化的基本单元，被 runPatchMatch 在红/黑两阶段中调用。
    /// @param propagateDown true=读左+上邻居, false=读右+下邻居（交替方向保证双向传播）
    /// @param rng 每线程独立的随机数生成器
    void processPixel(int x, int y,
                      cv::Mat &depthMap, cv::Mat &normalMap, cv::Mat &confidence,
                      const Context &ctx,
                      bool propagateDown,
                      std::mt19937 &rng) const;

    /// 计算单像素在给定 (depth, normal) 平面假设下的多视图匹配代价
    /// @param depth 参考像素 (x, y) 的深度
    /// @param nx, ny, nz 法向（相机坐标系，单位向量，nz > 0 指向相机）
    /// @return [0, 2]，越小越一致
    float computePixelCost(int x, int y, float depth,
                           float nx, float ny, float nz,
                           const Context &ctx) const;

};

#endif // PATCHMATCHRECONSTRUCTOR_H
