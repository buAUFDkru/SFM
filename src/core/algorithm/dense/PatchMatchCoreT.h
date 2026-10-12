#ifndef SFM_DENSE_PATCHMATCHCORET_H
#define SFM_DENSE_PATCHMATCHCORET_H

/**
 * @file PatchMatchCoreT.h
 * @brief F15b/c：编译期策略组合（CRTP / 模板），避开热点虚调用（ADR-004）
 *
 * - withSequential / withCheckerboard：Mat 版 NccCostFunction（微基准 / 简易路径）
 * - withV2Pod：与生产 V2 热点逐位一致的 NccV2Cost（F15c）
 */

#include "algorithm/dense/PatchMatchCost.h"
#include "algorithm/dense/PatchMatchTypes.h"
#include "algorithm/dense/strategies.h"

namespace sfm {
namespace dense {

template <class CostT, class PropT, class PriorT, class RefineT>
struct PatchMatchStrategySetT {
    CostT cost;
    PropT prop;
    PriorT prior;
    RefineT refine;
    const char *presetName = "default";
};

template <class CostT, class PropT, class PriorT, class RefineT>
class PatchMatchCoreT
{
public:
    using StrategySet = PatchMatchStrategySetT<CostT, PropT, PriorT, RefineT>;

    explicit PatchMatchCoreT(StrategySet strategies)
        : m_strategies(std::move(strategies))
    {
    }

    static PatchMatchCoreT withSequential()
    {
        return PatchMatchCoreT(StrategySet{CostT{}, PropT{}, PriorT{}, RefineT{}, "PatchMatch"});
    }

    static PatchMatchCoreT withCheckerboard()
    {
        return PatchMatchCoreT(
            StrategySet{CostT{}, PropT{}, PriorT{}, RefineT{}, "PatchMatchV2"});
    }

    const StrategySet &strategies() const { return m_strategies; }

    /// 直接成员调用（编译期可知类型，可内联 / 去虚）
    template <class Ctx>
    float evaluateCost(int x, int y, float depth, float nx, float ny, float nz,
                       const Ctx &ctx) const
    {
        return m_strategies.cost.cost(x, y, depth, nx, ny, nz, ctx);
    }

private:
    StrategySet m_strategies;
};

using PatchMatchCoreSequentialT =
    PatchMatchCoreT<NccCostFunction, SequentialPropagation, NullPlanePrior, NullRefinement>;
using PatchMatchCoreCheckerboardT =
    PatchMatchCoreT<NccCostFunction, CheckerboardPropagation, FlagPlanePrior,
                    JointBilateralRefinement>;
using PatchMatchCoreV2T =
    PatchMatchCoreT<NccV2Cost, CheckerboardPropagation, FlagPlanePrior,
                    JointBilateralRefinement>;

inline PatchMatchCoreV2T makeV2CoreT()
{
    return PatchMatchCoreV2T(PatchMatchCoreV2T::StrategySet{
        NccV2Cost{}, CheckerboardPropagation{}, FlagPlanePrior{},
        JointBilateralRefinement{}, "PatchMatchV2"});
}

} // namespace dense
} // namespace sfm

#endif
