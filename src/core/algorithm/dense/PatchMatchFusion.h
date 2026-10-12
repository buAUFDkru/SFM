#ifndef SFM_DENSE_PATCHMATCHFUSION_H
#define SFM_DENSE_PATCHMATCHFUSION_H

/**
 * @file PatchMatchFusion.h
 * @brief 几何一致性过滤 + 深度融合（ADR-004 / F15g）
 */

#include "algorithm/ThreadPool.h"
#include "pipeline/IDenseReconstructor.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

namespace sfm {
namespace dense {

struct ViewGeom {
    double K[9] = {};
    double R[9] = {};
    double Rt[9] = {};
    double t[3] = {};
    double C[3] = {};
    bool valid = false;
};

struct GeomConsistencyOptions {
    double maxReprojPx = 1.0;
    int minConsistent = 2;
};

struct GeomFilterStats {
    long long totalValid = 0;
    long long totalKept = 0;
    bool skippedTooFewViews = false;
};

struct FusionOptions {
    double relErrMax = 0.02;
    int minSupport = 2;
};

struct FusionStats {
    size_t numEmitted = 0;
    size_t numConsumed = 0;
    size_t numLowSupport = 0;
    double seconds = 0;
};

/// 从位姿填充 ViewGeom；失败时 valid=false
ViewGeom makeViewGeom(const CameraPose &pose, const cv::Mat &fallbackK);

GeomFilterStats filterGeometricConsistency(const DenseReconstructionInput &input,
                                           std::vector<StereoPairResult> &results,
                                           ThreadPool &pool,
                                           const GeomConsistencyOptions &opt);

/// images[i] 与 results[i] 对齐；空 Mat 表示该视图无色图
FusionStats fuseDepthMaps(const DenseReconstructionInput &input,
                          const std::vector<StereoPairResult> &results,
                          const std::vector<cv::Mat> &images, ThreadPool &pool,
                          const FusionOptions &opt,
                          pcl::PointCloud<pcl::PointXYZRGB> &cloud);

} // namespace dense
} // namespace sfm

#endif
