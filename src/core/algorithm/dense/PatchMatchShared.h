#ifndef SFM_DENSE_PATCHMATCHSHARED_H
#define SFM_DENSE_PATCHMATCHSHARED_H

/**
 * @file PatchMatchShared.h
 * @brief V1/V2 完全相同的自由函数（ADR-004 迁移步骤 2）
 */

#include "algorithm/dense/PatchMatchTypes.h"
#include "pipeline/IDenseReconstructor.h"

#include <functional>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace sfm {
namespace dense {

float sampleGrayBilinear(const cv::Mat &gray, float x, float y);

std::vector<Problem> buildProblems(const DenseReconstructionInput &input);

std::vector<std::pair<double, double>>
estimateDepthRanges(const DenseReconstructionInput &input);

/// 接口占位：PatchMatch 已是多视图联合优化，融合在 dense::fuseDepthMaps 中完成
cv::Mat fuseSourceDepthMaps(const cv::Mat &refImage,
                            const std::vector<cv::Mat> &srcDepths,
                            const std::vector<cv::Mat> &srcConfidences);

/// 从参考视角深度图反投影到世界坐标，追加到点云（V1/V2 相同）
void backprojectDepth(const DenseReconstructionInput &input,
                      int refIdx,
                      const cv::Mat &depth,
                      const cv::Mat &refImage,
                      pcl::PointCloud<pcl::PointXYZRGB> &cloud);

/// 红黑棋盘阶段序：与 V1/V2 runPatchMatch 历史行为一致
inline int checkerboardPhaseParity(bool propagateDown, int phase)
{
    const int firstParity = propagateDown ? 0 : 1;
    return (firstParity + phase) % 2;
}

// ── F15j：solveProblem 共享 I/O ──────────────────────────────────────

enum class MaxImageSizePolicy {
    CapAt1000WhenNonPositive,   ///< V1：maxImageSize<=0 → 上限 1000
    UnlimitedWhenNonPositive    ///< V2：maxImageSize<=0 → 不限制
};

using LoadImageFn = std::function<cv::Mat(int imageIdx)>;

double computeWorkScale(int width, int height, int maxImageSize,
                        MaxImageSizePolicy policy);

cv::Mat scaleIntrinsics(const cv::Mat &K, double scale);

struct PreparedRefView {
    bool ok = false;
    cv::Mat refColor; ///< 全分辨率 BGR
    cv::Mat refGray;  ///< 工作分辨率灰度
    cv::Mat K_full;   ///< 原始内参
    cv::Mat K_work;   ///< 缩放后内参
    double scale = 1.0;
    double depthMin = 0.1;
    double depthMax = 100.0;
    int refImageIdx = -1;
};

PreparedRefView prepareReferenceView(const DenseReconstructionInput &input, int refIdx,
                                     const std::pair<double, double> &depthRange,
                                     int maxImageSize, MaxImageSizePolicy policy,
                                     const LoadImageFn &loadImage);

std::vector<SourceView> buildMatSourceViews(const DenseReconstructionInput &input, int refIdx,
                                           const std::vector<int> &srcImageIdxs, double scale,
                                           const LoadImageFn &loadImage);

std::vector<SourceViewPod> buildPodSourceViews(const DenseReconstructionInput &input, int refIdx,
                                              const std::vector<int> &srcImageIdxs, double scale,
                                              const LoadImageFn &loadImage);

int countValidDepthPixels(const cv::Mat &depth);

/// 工作分辨率 → 全分辨率；alsoConfidence 时同步缩放 confidence（V2）
void upscaleToFullResolution(cv::Mat &depth, cv::Mat &confidence, const cv::Size &fullSize,
                             bool alsoConfidence);

} // namespace dense
} // namespace sfm

#endif
