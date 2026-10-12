#include "algorithm/dense/strategies.h"

#include "algorithm/dense/PatchMatchShared.h"
#include "pipeline/PatchMatchReconstructorV2.h" // prepareWindow / aggregateTopK

#include <cmath>
#include <vector>

namespace sfm {
namespace dense {
namespace {

const NccCostFunction &ncc()
{
    static const NccCostFunction s;
    return s;
}
const SequentialPropagation &seq()
{
    static const SequentialPropagation s;
    return s;
}
const CheckerboardPropagation &chk()
{
    static const CheckerboardPropagation s;
    return s;
}
const NullPlanePrior &nullPrior()
{
    static const NullPlanePrior s;
    return s;
}
const FlagPlanePrior &flagPrior()
{
    static const FlagPlanePrior s;
    return s;
}
const NullRefinement &nullRefine()
{
    static const NullRefinement s;
    return s;
}
const JointBilateralRefinement &jbu()
{
    static const JointBilateralRefinement s;
    return s;
}

} // namespace

PatchMatchStrategySet presetSequential()
{
    return PatchMatchStrategySet{&ncc(), &seq(), &nullPrior(), &nullRefine(), "PatchMatch"};
}

PatchMatchStrategySet presetCheckerboard()
{
    return PatchMatchStrategySet{&ncc(), &chk(), &flagPrior(), &jbu(), "PatchMatchV2"};
}

float NccCostFunction::cost(int x, int y, float depth, float nx, float ny, float nz,
                            const PatchContext &ctx) const
{
    if (!ctx.refGray || !ctx.sources || ctx.numSources <= 0)
        return 2.0f;

    const cv::Mat &refGray = *ctx.refGray;
    const int r = ctx.windowRadius;
    const double sigmaSpatial = (ctx.sigmaSpatial > 0) ? ctx.sigmaSpatial : double(r);
    const double invSs2 = 1.0 / (2.0 * sigmaSpatial * sigmaSpatial);
    const double invSc2 = 1.0 / (2.0 * ctx.sigmaColor * ctx.sigmaColor);

    // 平面诱导单应：与 V2 computePixelCost 同构的简化路径（供微基准 / 策略接线）
    // H = K_src · (R - t n^T / d) · K_ref^{-1}
    auto &cache = pmv2::windowCache();
    const int winCount = pmv2::prepareWindow(refGray, x, y, r, invSs2, invSc2, cache);

    std::vector<float> costs;
    costs.reserve(static_cast<size_t>(ctx.numSources));

    const double fx = ctx.fx, fy = ctx.fy, cx = ctx.cx, cy = ctx.cy;
    for (int s = 0; s < ctx.numSources; ++s) {
        const SourceView &src = ctx.sources[s];
        if (src.gray.empty() || src.R_rel.empty() || src.T_rel.empty() || src.K_src.empty())
            continue;

        // 参考相机系平面：n·X = d_plane，像素 (x,y) 深度 depth → 平面参数
        const double X = (x - cx) / fx * depth;
        const double Y = (y - cy) / fy * depth;
        const double Z = depth;
        const double nn = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nn < 1e-9 || Z < 1e-6)
            continue;
        const double ngx = nx / nn, ngy = ny / nn, ngz = nz / nn;
        const double dPlane = ngx * X + ngy * Y + ngz * Z;
        if (std::abs(dPlane) < 1e-9)
            continue;

        cv::Mat nRow = (cv::Mat_<double>(1, 3) << ngx / dPlane, ngy / dPlane, ngz / dPlane);
        cv::Mat H = src.K_src * (src.R_rel - src.T_rel * nRow) *
                    (cv::Mat_<double>(3, 3) << 1.0 / fx, 0, -cx / fx, 0, 1.0 / fy, -cy / fy, 0, 0,
                     1);

        double Hh[9];
        for (int i = 0; i < 9; ++i)
            Hh[i] = H.at<double>(i / 3, i % 3);

        double sumRef = 0, sumSrc = 0, sumRefSq = 0, sumSrcSq = 0, sumRefSrc = 0, sumW = 0;
        int valid = 0;
        for (int k = 0; k < winCount; ++k) {
            const int rx = cache.rx[k];
            const int ry = cache.ry[k];
            double denom = Hh[6] * rx + Hh[7] * ry + Hh[8];
            if (std::abs(denom) < 1e-9)
                continue;
            double inv = 1.0 / denom;
            float sx = float((Hh[0] * rx + Hh[1] * ry + Hh[2]) * inv);
            float sy = float((Hh[3] * rx + Hh[4] * ry + Hh[5]) * inv);
            float srcVal = sampleGrayBilinear(src.gray, sx, sy);
            if (srcVal < 0)
                continue;
            const float refVal = cache.refVal[k];
            const double w = cache.weight[k];
            sumRef += refVal * w;
            sumSrc += srcVal * w;
            sumRefSq += refVal * refVal * w;
            sumSrcSq += srcVal * srcVal * w;
            sumRefSrc += refVal * srcVal * w;
            sumW += w;
            ++valid;
        }
        if (valid < 3 || sumW < 1e-12) {
            costs.push_back(2.0f);
            continue;
        }
        const double invW = 1.0 / sumW;
        const double meanR = sumRef * invW;
        const double meanS = sumSrc * invW;
        const double varR = sumRefSq * invW - meanR * meanR;
        const double varS = sumSrcSq * invW - meanS * meanS;
        const double cov = sumRefSrc * invW - meanR * meanS;
        const double denom = std::sqrt(std::max(varR, 0.0) * std::max(varS, 0.0));
        float ncc = (denom > 1e-12) ? float(cov / denom) : 0.f;
        ncc = std::max(-1.f, std::min(1.f, ncc));
        costs.push_back(1.f - ncc); // 代价：1-NCC ∈ [0,2]
    }

    return pmv2::aggregateTopK(costs, ctx.costTopK, 2.0f);
}

void SequentialPropagation::forEach(int rows, int cols, bool downPass,
                                    const PixelVisitor &visit) const
{
    if (downPass) {
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < cols; ++x)
                visit(x, y);
    } else {
        for (int y = rows - 1; y >= 0; --y)
            for (int x = cols - 1; x >= 0; --x)
                visit(x, y);
    }
}

void CheckerboardPropagation::forEach(int rows, int cols, bool downPass,
                                      const PixelVisitor &visit) const
{
    // 红黑棋盘：downPass 处理 (x+y) 偶数，否则奇数（两轮覆盖全图）
    const int parity = downPass ? 0 : 1;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (((x + y) & 1) == parity)
                visit(x, y);
        }
    }
}

} // namespace dense
} // namespace sfm
