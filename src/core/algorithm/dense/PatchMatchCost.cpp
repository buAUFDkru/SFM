#include "algorithm/dense/PatchMatchCost.h"

#include "algorithm/dense/PatchMatchShared.h"
#include "pipeline/PatchMatchReconstructorV2.h" // pmv2::prepareWindow / aggregateTopK

#include <cmath>
#include <cstdint>
#include <vector>

namespace sfm {
namespace dense {

float computeNccPixelCost(int x, int y, float depth, float nx, float ny, float nz,
                          const PatchContextPod &ctx)
{
    if (!ctx.refGray || !ctx.sources || ctx.numSources <= 0)
        return 2.0f;

    float bestCost = 2.0f;

    const int topK = ctx.opt.costTopK;
    static thread_local std::vector<float> srcCosts;
    if (topK > 1)
        srcCosts.clear();

    const int r = ctx.opt.windowRadius;
    const double sigmaSpatial =
        ctx.opt.sigmaSpatial > 0 ? ctx.opt.sigmaSpatial : double(r);
    const double sigmaColor = ctx.opt.sigmaColor;
    const double invSs2 = 1.0 / (2.0 * sigmaSpatial * sigmaSpatial);
    const double invSc2 = 1.0 / (2.0 * sigmaColor * sigmaColor);

    const double fx = ctx.fx;
    const double fy = ctx.fy;
    const double cx = ctx.cx;
    const double cy = ctx.cy;

    double nKinv = nx * (x - cx) / fx + ny * (y - cy) / fy + nz;
    if (std::abs(nKinv) < 1e-9)
        return 2.0f;
    double planeD = double(depth) * nKinv;
    if (std::abs(planeD) < 1e-9)
        return 2.0f;
    double invPlaneD = 1.0 / planeD;

    sfm::pmv2::WindowCache &cache = sfm::pmv2::windowCache();
    const int winCount =
        sfm::pmv2::prepareWindow(*ctx.refGray, x, y, r, invSs2, invSc2, cache);

    for (int s = 0; s < ctx.numSources; ++s) {
        const SourceViewPod &src = ctx.sources[s];
        if (src.gray.empty())
            continue;

        const double Tx = src.T[0];
        const double Ty = src.T[1];
        const double Tz = src.T[2];

        double Tn_pD00 = Tx * nx * invPlaneD;
        double Tn_pD01 = Tx * ny * invPlaneD;
        double Tn_pD02 = Tx * nz * invPlaneD;
        double Tn_pD10 = Ty * nx * invPlaneD;
        double Tn_pD11 = Ty * ny * invPlaneD;
        double Tn_pD12 = Ty * nz * invPlaneD;
        double Tn_pD20 = Tz * nx * invPlaneD;
        double Tn_pD21 = Tz * ny * invPlaneD;
        double Tn_pD22 = Tz * nz * invPlaneD;

        double M00 = src.R[0] + Tn_pD00;
        double M01 = src.R[1] + Tn_pD01;
        double M02 = src.R[2] + Tn_pD02;
        double M10 = src.R[3] + Tn_pD10;
        double M11 = src.R[4] + Tn_pD11;
        double M12 = src.R[5] + Tn_pD12;
        double M20 = src.R[6] + Tn_pD20;
        double M21 = src.R[7] + Tn_pD21;
        double M22 = src.R[8] + Tn_pD22;

        const double k00 = src.K[0], k01 = src.K[1], k02 = src.K[2];
        const double k10 = src.K[3], k11 = src.K[4], k12 = src.K[5];
        const double k20 = src.K[6], k21 = src.K[7], k22 = src.K[8];

        double KM00 = k00 * M00 + k01 * M10 + k02 * M20;
        double KM01 = k00 * M01 + k01 * M11 + k02 * M21;
        double KM02 = k00 * M02 + k01 * M12 + k02 * M22;
        double KM10 = k10 * M00 + k11 * M10 + k12 * M20;
        double KM11 = k10 * M01 + k11 * M11 + k12 * M21;
        double KM12 = k10 * M02 + k11 * M12 + k12 * M22;
        double KM20 = k20 * M00 + k21 * M10 + k22 * M20;
        double KM21 = k20 * M01 + k21 * M11 + k22 * M21;
        double KM22 = k20 * M02 + k21 * M12 + k22 * M22;

        double H00 = KM00 / fx;
        double H01 = KM01 / fy;
        double H02 = -KM00 * cx / fx - KM01 * cy / fy + KM02;
        double H10 = KM10 / fx;
        double H11 = KM11 / fy;
        double H12 = -KM10 * cx / fx - KM11 * cy / fy + KM12;
        double H20 = KM20 / fx;
        double H21 = KM21 / fy;
        double H22 = -KM20 * cx / fx - KM21 * cy / fy + KM22;

        double sumRef = 0, sumSrc = 0, sumRefSq = 0, sumSrcSq = 0, sumRefSrc = 0;
        double sumW = 0;
        int validCount = 0;

        for (int k = 0; k < winCount; ++k) {
            const int rx = cache.rx[k];
            const int ry = cache.ry[k];

            double denom = H20 * rx + H21 * ry + H22;
            if (std::abs(denom) < 1e-9)
                continue;
            double invDenom = 1.0 / denom;
            double srcX = (H00 * rx + H01 * ry + H02) * invDenom;
            double srcY = (H10 * rx + H11 * ry + H12) * invDenom;

            float srcVal = sampleGrayBilinear(src.gray, float(srcX), float(srcY));
            if (srcVal < 0)
                continue;
            ++validCount;

            const float refVal = cache.refVal[k];
            const double w = cache.weight[k];

            sumRef += refVal * w;
            sumSrc += srcVal * w;
            sumRefSq += refVal * refVal * w;
            sumSrcSq += srcVal * srcVal * w;
            sumRefSrc += refVal * srcVal * w;
            sumW += w;
        }

        if (validCount < (2 * r + 1) * (2 * r + 1) / 4 || sumW < 1e-6)
            continue;

        double meanRef = sumRef / sumW;
        double meanSrc = sumSrc / sumW;
        double varRef = sumRefSq / sumW - meanRef * meanRef;
        double varSrc = sumSrcSq / sumW - meanSrc * meanSrc;
        double covar = sumRefSrc / sumW - meanRef * meanSrc;

        constexpr double kMinVar = 1.0;
        if (varRef < kMinVar || varSrc < kMinVar)
            continue;

        double ncc = covar / std::sqrt(varRef * varSrc);
        ncc = std::max(-1.0, std::min(1.0, ncc));
        double cost = 1.0 - ncc;
        cost = std::max(0.0, std::min(2.0, cost));
        if (topK > 1) {
            srcCosts.push_back(float(cost));
        } else if (cost < bestCost) {
            bestCost = float(cost);
        }
    }

    if (topK > 1)
        return pmv2::aggregateTopK(srcCosts, topK);

    return bestCost;
}

float computeNccPixelCostV1(int x, int y, float depth, float nx, float ny, float nz,
                            const PatchContext &ctx)
{
    // F15i：历史 V1 Mat 版 NCC（min-over-sources），逐位保持
    if (!ctx.refGray || !ctx.sources || ctx.numSources <= 0)
        return 2.0f;

    float bestCost = 2.0f;

    const cv::Mat &refGray = *ctx.refGray;
    const int rows = refGray.rows;
    const int cols = refGray.cols;
    const int r = ctx.windowRadius;
    const double sigmaSpatial = ctx.sigmaSpatial > 0 ? ctx.sigmaSpatial : double(r);
    const double sigmaColor = ctx.sigmaColor;
    const double invSs2 = 1.0 / (2.0 * sigmaSpatial * sigmaSpatial);
    const double invSc2 = 1.0 / (2.0 * sigmaColor * sigmaColor);

    const double fx = ctx.fx;
    const double fy = ctx.fy;
    const double cx = ctx.cx;
    const double cy = ctx.cy;

    double nKinv = nx * (x - cx) / fx + ny * (y - cy) / fy + nz;
    if (std::abs(nKinv) < 1e-9)
        return 2.0f;
    double planeD = double(depth) * nKinv;
    if (std::abs(planeD) < 1e-9)
        return 2.0f;
    double invPlaneD = 1.0 / planeD;

    for (int s = 0; s < ctx.numSources; ++s) {
        const SourceView &src = ctx.sources[s];
        if (src.gray.empty() || src.R_rel.empty() || src.T_rel.empty() || src.K_src.empty())
            continue;

        const double Tx = src.T_rel.at<double>(0);
        const double Ty = src.T_rel.at<double>(1);
        const double Tz = src.T_rel.at<double>(2);

        double Tn_pD00 = Tx * nx * invPlaneD;
        double Tn_pD01 = Tx * ny * invPlaneD;
        double Tn_pD02 = Tx * nz * invPlaneD;
        double Tn_pD10 = Ty * nx * invPlaneD;
        double Tn_pD11 = Ty * ny * invPlaneD;
        double Tn_pD12 = Ty * nz * invPlaneD;
        double Tn_pD20 = Tz * nx * invPlaneD;
        double Tn_pD21 = Tz * ny * invPlaneD;
        double Tn_pD22 = Tz * nz * invPlaneD;

        double M00 = src.R_rel.at<double>(0, 0) + Tn_pD00;
        double M01 = src.R_rel.at<double>(0, 1) + Tn_pD01;
        double M02 = src.R_rel.at<double>(0, 2) + Tn_pD02;
        double M10 = src.R_rel.at<double>(1, 0) + Tn_pD10;
        double M11 = src.R_rel.at<double>(1, 1) + Tn_pD11;
        double M12 = src.R_rel.at<double>(1, 2) + Tn_pD12;
        double M20 = src.R_rel.at<double>(2, 0) + Tn_pD20;
        double M21 = src.R_rel.at<double>(2, 1) + Tn_pD21;
        double M22 = src.R_rel.at<double>(2, 2) + Tn_pD22;

        const cv::Mat &Ks = src.K_src;
        const double k00 = Ks.at<double>(0, 0), k01 = Ks.at<double>(0, 1),
                     k02 = Ks.at<double>(0, 2);
        const double k10 = Ks.at<double>(1, 0), k11 = Ks.at<double>(1, 1),
                     k12 = Ks.at<double>(1, 2);
        const double k20 = Ks.at<double>(2, 0), k21 = Ks.at<double>(2, 1),
                     k22 = Ks.at<double>(2, 2);

        double KM00 = k00 * M00 + k01 * M10 + k02 * M20;
        double KM01 = k00 * M01 + k01 * M11 + k02 * M21;
        double KM02 = k00 * M02 + k01 * M12 + k02 * M22;
        double KM10 = k10 * M00 + k11 * M10 + k12 * M20;
        double KM11 = k10 * M01 + k11 * M11 + k12 * M21;
        double KM12 = k10 * M02 + k11 * M12 + k12 * M22;
        double KM20 = k20 * M00 + k21 * M10 + k22 * M20;
        double KM21 = k20 * M01 + k21 * M11 + k22 * M21;
        double KM22 = k20 * M02 + k21 * M12 + k22 * M22;

        double H00 = KM00 / fx;
        double H01 = KM01 / fy;
        double H02 = -KM00 * cx / fx - KM01 * cy / fy + KM02;
        double H10 = KM10 / fx;
        double H11 = KM11 / fy;
        double H12 = -KM10 * cx / fx - KM11 * cy / fy + KM12;
        double H20 = KM20 / fx;
        double H21 = KM21 / fy;
        double H22 = -KM20 * cx / fx - KM21 * cy / fy + KM22;

        double sumRef = 0, sumSrc = 0, sumRefSq = 0, sumSrcSq = 0, sumRefSrc = 0;
        double sumW = 0;
        int validCount = 0;

        const uint8_t refCenterVal = refGray.at<uint8_t>(y, x);

        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int rx = x + dx;
                int ry = y + dy;
                if (rx < 0 || rx >= cols || ry < 0 || ry >= rows)
                    continue;

                float refVal = refGray.at<uint8_t>(ry, rx);

                double denom = H20 * rx + H21 * ry + H22;
                if (std::abs(denom) < 1e-9)
                    continue;
                double invDenom = 1.0 / denom;
                double srcX = (H00 * rx + H01 * ry + H02) * invDenom;
                double srcY = (H10 * rx + H11 * ry + H12) * invDenom;

                float srcVal = sampleGrayBilinear(src.gray, float(srcX), float(srcY));
                if (srcVal < 0)
                    continue;
                ++validCount;

                double spatialDist2 = double(dx * dx + dy * dy);
                double colorDiff = double(refVal) - double(refCenterVal);
                double colorDist2 = colorDiff * colorDiff;
                double w = std::exp(-spatialDist2 * invSs2 - colorDist2 * invSc2);

                sumRef += refVal * w;
                sumSrc += srcVal * w;
                sumRefSq += refVal * refVal * w;
                sumSrcSq += srcVal * srcVal * w;
                sumRefSrc += refVal * srcVal * w;
                sumW += w;
            }
        }

        if (validCount < (2 * r + 1) * (2 * r + 1) / 4 || sumW < 1e-6)
            continue;

        double meanRef = sumRef / sumW;
        double meanSrc = sumSrc / sumW;
        double varRef = sumRefSq / sumW - meanRef * meanRef;
        double varSrc = sumSrcSq / sumW - meanSrc * meanSrc;
        double covar = sumRefSrc / sumW - meanRef * meanSrc;

        constexpr double kMinVar = 1.0;
        if (varRef < kMinVar || varSrc < kMinVar)
            continue;

        double ncc = covar / std::sqrt(varRef * varSrc);
        ncc = std::max(-1.0, std::min(1.0, ncc));
        double cost = 1.0 - ncc;
        cost = std::max(0.0, std::min(2.0, cost));
        if (cost < bestCost)
            bestCost = float(cost);
    }

    return bestCost;
}

} // namespace dense
} // namespace sfm
