#ifndef SFM_DENSE_PATCHMATCHCORE_H
#define SFM_DENSE_PATCHMATCHCORE_H

/**
 * @file PatchMatchCore.h
 * @brief 密集重建共享内核 + 策略集合（ADR-004 / F15a，第一版虚函数绑定）
 */

#include "algorithm/dense/PatchMatchShared.h"
#include "algorithm/dense/strategies.h"

namespace sfm {
namespace dense {

class PatchMatchCore
{
public:
    explicit PatchMatchCore(PatchMatchStrategySet strategies);

    static PatchMatchCore withSequential();   ///< V1 具名预设
    static PatchMatchCore withCheckerboard(); ///< V2 具名预设

    const PatchMatchStrategySet &strategies() const { return m_strategies; }

    /// 经 ICostFunction 虚调用的热点入口（微基准 / 后续接 processPixel）
    float evaluateCost(int x, int y, float depth, float nx, float ny, float nz,
                       const PatchContext &ctx) const;

private:
    PatchMatchStrategySet m_strategies;
};

} // namespace dense
} // namespace sfm

#endif
