#include "algorithm/dense/PatchMatchCore.h"

namespace sfm {
namespace dense {

PatchMatchCore::PatchMatchCore(PatchMatchStrategySet strategies)
    : m_strategies(strategies)
{
}

PatchMatchCore PatchMatchCore::withSequential()
{
    return PatchMatchCore(presetSequential());
}

PatchMatchCore PatchMatchCore::withCheckerboard()
{
    return PatchMatchCore(presetCheckerboard());
}

float PatchMatchCore::evaluateCost(int x, int y, float depth, float nx, float ny, float nz,
                                   const PatchContext &ctx) const
{
    if (!m_strategies.cost)
        return 2.0f;
    return m_strategies.cost->cost(x, y, depth, nx, ny, nz, ctx);
}

} // namespace dense
} // namespace sfm
