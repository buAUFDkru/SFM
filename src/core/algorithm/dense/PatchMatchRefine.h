#ifndef SFM_DENSE_PATCHMATCHREFINE_H
#define SFM_DENSE_PATCHMATCHREFINE_H

/**
 * @file PatchMatchRefine.h
 * @brief JBU 上采样 + 多尺度 coarse-to-fine（ADR-004 / F15f）
 */

#include "algorithm/ThreadPool.h"
#include "algorithm/dense/PatchMatchTypes.h"

#include <functional>
#include <vector>

#include <opencv2/core.hpp>

namespace sfm {
namespace dense {

/// 联合双边上采样（与历史 V2::jointBilateralUpsample 逐位一致）
void jointBilateralUpsample(const cv::Mat &depthLow, const cv::Mat &normalLow,
                            const cv::Mat &grayHigh, cv::Mat &depthHigh, cv::Mat &normalHigh);

/// 单层 PatchMatch 运行时上下文（与 V2::Context 字段对齐，供金字塔拷贝）
struct PatchMatchLevelCtx {
    cv::Mat refGray;
    cv::Mat K_ref;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<SourceViewPod> sources;
    double depthMin = 0.1, depthMax = 100.0;
    int rows = 0, cols = 0;
    int refImageIdx = -1;
    ThreadPool *pool = nullptr;
};

int decideNumScales(int cols, int rows, int maxScales, bool multiscaleEnabled);

/// pyr[0]=最细 … pyr.back()=最粗；输入 finest 会被拷贝为 pyr[0]
std::vector<PatchMatchLevelCtx> buildPyramid(const PatchMatchLevelCtx &finest, int numScales);

int itersForPyramidLevel(int lv, int numScales, int coarseIters, int refineIters);

using SingleScaleRunner = std::function<cv::Mat(
    PatchMatchLevelCtx &level, cv::Mat &confidence, cv::Mat &normalMap,
    const cv::Mat &initDepth, const cv::Mat &initNormal, int numIters)>;

/// 自粗到细；取消时 depthOut 为空并返回 false
bool runCoarseToFine(
    std::vector<PatchMatchLevelCtx> &pyr, int coarseIters, int refineIters,
    const std::function<bool()> &shouldStop,
    const std::function<void(int step, int numScales, int cols, int rows, int iters)> &onLevel,
    const SingleScaleRunner &runLevel, cv::Mat &depthOut, cv::Mat &confidenceOut,
    cv::Mat &normalOut);

} // namespace dense
} // namespace sfm

#endif
