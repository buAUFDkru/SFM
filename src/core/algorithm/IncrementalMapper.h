#ifndef INCREMENTALMAPPER_H
#define INCREMENTALMAPPER_H

/**
 * @file IncrementalMapper.h
 * @brief 增量稀疏重建编排：种子对多重尝试 + 接受判据（阶段三）
 *
 * 从 ReconstructionPipeline / recon_cli 抽出 COLMAP 式 init_max_reg_trials
 * 循环与「主分量全注册即短路 / ≥3 台兜底」接受逻辑。
 * Triangulate 由调用方注入（每流水线一份），options 按次写入该实例。
 */

#include <functional>
#include <vector>

#include <QString>
#include <QStringList>

#include <opencv2/core.hpp>

#include "CancellationToken.h"
#include "Initialization.h"
#include "Triangulate.h"
#include "data/SfmTypes.h"

namespace sfm {

struct IncrementalMapperConfig
{
    TriangulateOptions triOptions;
    TriangulatorOptions triangulatorOptions;
    bool applyTriangulatorOptions = false; ///< true 时一并覆写 triangulator options
    bool fixIntrinsics = false;
    /// 注册相机数 ≥ 此值则接受并停止尝试（通常 = 主连通分量大小）
    int acceptRegCount = 3;
};

struct IncrementalMapperResult
{
    bool ok = false;
    std::vector<Point3D> points;
    std::vector<CameraPose> poses;
    int bestRegCount = 0;
    size_t bestPointCount = 0;
    int bestTrialIndex = -1; ///< 0-based；无成功尝试时为 -1
    bool cancelled = false;
    QStringList messages;
};

class IncrementalMapper
{
public:
    /// 比较两次尝试：更高注册数优先，同分则点数更多者优
    static bool isBetterAttempt(int regCount, size_t pointCount,
                                int bestReg, size_t bestPts);

    /**
     * @brief 对候选种子对逐个调用 tri.triangulate，保留最优并按 acceptRegCount 短路
     *
     * @param tri    调用方持有的三角化器（options 写入此实例）
     * @param cancel 非空时在每次尝试后检查；置位则带 cancelled=true 返回
     * @param log    可选日志回调（与 messages 并行）
     */
    static IncrementalMapperResult run(
        Triangulate &tri,
        const std::vector<InitResult> &candidates,
        const std::vector<MatchPair> &matchPairs,
        const std::vector<FeatureResult> &features,
        const ViewGraph &viewGraph,
        const cv::Mat &K,
        const IncrementalMapperConfig &config,
        CancellationToken *cancel = nullptr,
        const std::function<void(const QString &)> &log = {});
};

} // namespace sfm

#endif // INCREMENTALMAPPER_H
