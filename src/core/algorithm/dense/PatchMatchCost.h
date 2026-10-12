#ifndef SFM_DENSE_PATCHMATCHCOST_H
#define SFM_DENSE_PATCHMATCHCOST_H

/**
 * @file PatchMatchCost.h
 * @brief NCC 像素代价（ADR-004：F15c V2 POD；F15i V1 Mat）
 */

#include "algorithm/dense/PatchMatchTypes.h"

namespace sfm {
namespace dense {

/// 与历史 V2::computePixelCost 逐位一致的多视图 NCC 代价（POD + TopK）
float computeNccPixelCost(int x, int y, float depth, float nx, float ny, float nz,
                          const PatchContextPod &ctx);

/// 与历史 V1::computePixelCost 逐位一致：Mat 源视图、min-over-sources、内联窗口
float computeNccPixelCostV1(int x, int y, float depth, float nx, float ny, float nz,
                            const PatchContext &ctx);

/// 静态策略：供 PatchMatchCoreT 绑定（无虚表）
class NccV2Cost {
public:
    const char *name() const { return "NCC-V2"; }
    float cost(int x, int y, float depth, float nx, float ny, float nz,
               const PatchContextPod &ctx) const
    {
        return computeNccPixelCost(x, y, depth, nx, ny, nz, ctx);
    }
};

class NccV1Cost {
public:
    const char *name() const { return "NCC-V1"; }
    float cost(int x, int y, float depth, float nx, float ny, float nz,
               const PatchContext &ctx) const
    {
        return computeNccPixelCostV1(x, y, depth, nx, ny, nz, ctx);
    }
};

} // namespace dense
} // namespace sfm

#endif
