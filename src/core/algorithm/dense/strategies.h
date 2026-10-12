#ifndef SFM_DENSE_STRATEGIES_H
#define SFM_DENSE_STRATEGIES_H

/**
 * @file strategies.h
 * @brief PatchMatch 四类可插拔策略（ADR-004 / IF-11：第一版虚函数绑定）
 */

#include "algorithm/dense/PatchMatchTypes.h"

#include <opencv2/core.hpp>

namespace sfm {
namespace dense {

class ICostFunction {
public:
    virtual ~ICostFunction() = default;
    virtual const char *name() const = 0;
    virtual float cost(int x, int y, float depth, float nx, float ny, float nz,
                       const PatchContext &ctx) const = 0;
};

class IPropagation {
public:
    virtual ~IPropagation() = default;
    virtual const char *name() const = 0;
    /// downPass=true：左上→右下；false：右下→左上（红黑/棋盘两阶段共用）
    virtual void forEach(int rows, int cols, bool downPass,
                         const PixelVisitor &visit) const = 0;
};

class IPlanePrior {
public:
    virtual ~IPlanePrior() = default;
    virtual const char *name() const = 0;
    virtual bool enabled() const = 0;
    virtual bool query(int x, int y, PlaneHypothesis *out) const = 0;
};

class IRefinement {
public:
    virtual ~IRefinement() = default;
    virtual const char *name() const = 0;
    virtual bool enabled() const = 0;
};

struct PatchMatchStrategySet {
    const ICostFunction *cost = nullptr;
    const IPropagation *prop = nullptr;
    const IPlanePrior *prior = nullptr;
    const IRefinement *refine = nullptr;
    const char *presetName = "default";
};

/// 具名预设（静态策略对象，寿命 = 进程）
PatchMatchStrategySet presetSequential();     ///< V1：顺序传播 + NCC + 无先验/精化
PatchMatchStrategySet presetCheckerboard();   ///< V2：棋盘传播 + NCC + 平面先验 + JBU 标记

// ── 具体策略（可供单测 / 微基准直接拿）──────────────────────────
class NccCostFunction final : public ICostFunction {
public:
    const char *name() const override { return "NCC"; }
    float cost(int x, int y, float depth, float nx, float ny, float nz,
               const PatchContext &ctx) const override;
};

class SequentialPropagation final : public IPropagation {
public:
    const char *name() const override { return "Sequential"; }
    void forEach(int rows, int cols, bool downPass,
                 const PixelVisitor &visit) const override;
};

class CheckerboardPropagation final : public IPropagation {
public:
    const char *name() const override { return "Checkerboard"; }
    void forEach(int rows, int cols, bool downPass,
                 const PixelVisitor &visit) const override;
};

class NullPlanePrior final : public IPlanePrior {
public:
    const char *name() const override { return "None"; }
    bool enabled() const override { return false; }
    bool query(int, int, PlaneHypothesis *out) const override
    {
        if (out) out->valid = false;
        return false;
    }
};

class FlagPlanePrior final : public IPlanePrior {
public:
    const char *name() const override { return "ACMP"; }
    bool enabled() const override { return true; }
    bool query(int, int, PlaneHypothesis *out) const override
    {
        // 实际先验图由 V2/Core 在 run 时注入；此处仅作「启用」标记供预设表达。
        if (out) out->valid = false;
        return false;
    }
};

class NullRefinement final : public IRefinement {
public:
    const char *name() const override { return "None"; }
    bool enabled() const override { return false; }
};

class JointBilateralRefinement final : public IRefinement {
public:
    const char *name() const override { return "JBU"; }
    bool enabled() const override { return true; }
};

} // namespace dense
} // namespace sfm

#endif
