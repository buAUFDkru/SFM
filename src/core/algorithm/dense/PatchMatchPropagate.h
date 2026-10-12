#ifndef SFM_DENSE_PATCHMATCHPROPAGATE_H
#define SFM_DENSE_PATCHMATCHPROPAGATE_H

/**
 * @file PatchMatchPropagate.h
 * @brief 像素传播 + 红黑 strip + runPatchMatch 编排壳（ADR-004 / F15d–e）
 */

#include "algorithm/ThreadPool.h"

#include <cstdint>
#include <functional>
#include <random>

#include <opencv2/core.hpp>

namespace sfm {
namespace pmv2 {
struct PlanarPrior;
}
}

namespace sfm {
namespace dense {

using PixelCostFn =
    std::function<float(int x, int y, float d, float nx, float ny, float nz)>;

/// V2 processPixel 全文：传播 + 退火；prior==nullptr 时走纯光度（与历史逐位一致）
void processPixel(int x, int y, cv::Mat &depthMap, cv::Mat &normalMap, cv::Mat &confidence,
                  int rows, int cols, double fx, double fy, double cx, double cy,
                  double depthMin, double depthMax, bool propagateDown,
                  const sfm::pmv2::PlanarPrior *prior, std::mt19937 &rng,
                  const PixelCostFn &costFn);

/// 红黑一阶段：按行 strip 提交到线程池（RNG 种子 12345+stripIdx，与历史一致）
void runCheckerboardStrips(
    sfm::ThreadPool &pool, int rows, int cols, int parity, bool propagateDown,
    const std::function<bool()> &shouldStop,
    const std::function<void(int x, int y, std::mt19937 &rng)> &visit);

/// 初始化深度/法向（种子 42）；返回复用上采样像素数（无 init 时为 0）
int initDepthNormalMaps(cv::Mat &depthMap, cv::Mat &normalMap, double depthMin, double depthMax,
                        const cv::Mat &initDepth, const cv::Mat &initNormal);

struct PatchMatchIterDiag {
    int iter = 0; ///< 1-based
    int numIters = 0;
    int64_t elapsedMs = 0;
    int validCnt = 0;
    int totalPixels = 0;
    double meanDepth = 0;
    double meanCost = 0;
    float minD = 0;
    float maxD = 0;
};

struct PlanarPriorEvent {
    int supportCount = 0;
    int coverage = 0;
    int totalPixels = 0;
    int64_t elapsedMs = 0;
    bool applied = false; ///< coverage>0 时为 true
};

struct PatchMatchRunConfig {
    int numIters = 5;
    bool wantPrior = false;
    int planePriorStep = 16;
    float planePriorMaxCost = 0.5f;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double depthMin = 0.1, depthMax = 100.0;
};

/// 红黑迭代壳（含可选平面先验 warmup）；取消时释放三张图并返回 false
bool runPatchMatchIterations(sfm::ThreadPool &pool, cv::Mat &depthMap, cv::Mat &normalMap,
                             cv::Mat &confidence, const PatchMatchRunConfig &cfg,
                             const PixelCostFn &costFn, const std::function<bool()> &shouldStop,
                             const std::function<void(const PlanarPriorEvent &)> &onPrior,
                             const std::function<void(const PatchMatchIterDiag &)> &onIterDone);

/// 按置信度阈值滤深度；返回被置零像素数
int filterDepthByCost(cv::Mat &depthMap, const cv::Mat &confidence, float costThreshold);

} // namespace dense
} // namespace sfm

#endif
