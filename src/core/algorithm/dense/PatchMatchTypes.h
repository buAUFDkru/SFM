#ifndef SFM_DENSE_PATCHMATCHTYPES_H
#define SFM_DENSE_PATCHMATCHTYPES_H

/**
 * @file PatchMatchTypes.h
 * @brief PatchMatchCore 共用 DTO（ADR-004 / F15a）
 */

#include <functional>
#include <vector>

#include <opencv2/core.hpp>

namespace sfm { class ThreadPool; }

namespace sfm {
namespace dense {

struct Problem {
    int refImageIdx = -1;
    std::vector<int> srcImageIdxs;
};

struct SourceView {
    cv::Mat gray;
    cv::Mat K_src;
    cv::Mat R_rel;
    cv::Mat T_rel;
    int imageIdx = -1;
};

/// V2 热点路径源视图：相机参数用 POD 数组，避免 Mat::at 间接寻址
struct SourceViewPod {
    cv::Mat gray;
    double K[9] = {};
    double R[9] = {};
    double T[3] = {};
    int imageIdx = -1;
};

struct PatchCostOptions {
    int windowRadius = 5;
    double sigmaSpatial = -1.0;
    double sigmaColor = 51.0;
    int costTopK = 1;
};

/// 热点代价函数使用的轻量上下文（避免把 Qt / 完整 Options 拖进策略）
struct PatchContext {
    const cv::Mat *refGray = nullptr;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    const SourceView *sources = nullptr;
    int numSources = 0;
    double depthMin = 0, depthMax = 1;
    int windowRadius = 5;
    double sigmaSpatial = -1.0;
    double sigmaColor = 51.0;
    int costTopK = 1;
    ThreadPool *pool = nullptr;
};

/// V2 POD 热点上下文（供 NccV2Cost / computeNccPixelCost）
struct PatchContextPod {
    const cv::Mat *refGray = nullptr;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    const SourceViewPod *sources = nullptr;
    int numSources = 0;
    PatchCostOptions opt;
};

struct PlaneHypothesis {
    float nx = 0, ny = 0, nz = 1;
    float d = 0;
    bool valid = false;
};

using PixelVisitor = std::function<void(int x, int y)>;

} // namespace dense
} // namespace sfm

#endif
