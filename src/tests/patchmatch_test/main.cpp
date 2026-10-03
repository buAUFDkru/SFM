// patchmatch_test —— PatchMatch 代价函数优化的正确性回归测试
//
// 背景：PatchMatchReconstructorV2::computePixelCost 是最内层热点。优化把
// 「参考图窗口像素值 + 双边权重」的计算从源视图循环内提到循环外，
// 消除 N_src × 窗口面积 次冗余 exp 调用。
//
// 该优化必须是**数值等价变换**，不能改变任何一位结果。本测试用
// 优化前的原始实现（golden）与当前实现（走 sfm::pmv2::prepareWindow）
// 在随机数据与边界条件下逐位对比，确保等价性成立。

#include "pipeline/PatchMatchReconstructorV2.h"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

// ═══════════════════════════════════════════════════════════════════
// 优化前的原始实现（golden，逐字对应改动前的 PatchMatchReconstructorV2.cpp）
// ═══════════════════════════════════════════════════════════════════

float sampleGrayBilinearOriginal(const cv::Mat &gray, float x, float y)
{
    if (x < 0 || y < 0 || x > gray.cols - 1 || y > gray.rows - 1) return -1.0f;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, gray.cols - 1);
    int y1 = std::min(y0 + 1, gray.rows - 1);
    float dx = x - x0;
    float dy = y - y0;

    float v00 = gray.at<uint8_t>(y0, x0);
    float v01 = gray.at<uint8_t>(y0, x1);
    float v10 = gray.at<uint8_t>(y1, x0);
    float v11 = gray.at<uint8_t>(y1, x1);

    return (1.0f - dx) * (1.0f - dy) * v00 +
           dx * (1.0f - dy) * v01 +
           (1.0f - dx) * dy * v10 +
           dx * dy * v11;
}

/// NCC 累加器（对应 computePixelCost 中的 6 个累加量 + 有效计数）
struct Accum {
    double sumRef = 0.0;
    double sumSrc = 0.0;
    double sumRefSq = 0.0;
    double sumSrcSq = 0.0;
    double sumRefSrc = 0.0;
    double sumW = 0.0;
    int validCount = 0;
};

/// golden：窗口遍历 + 逐点 exp（优化前的写法）
Accum goldenCost(const cv::Mat &refGray, const cv::Mat &srcGray,
                 int x, int y, int r, double invSs2, double invSc2,
                 const double H[9])
{
    const double H00 = H[0], H01 = H[1], H02 = H[2];
    const double H10 = H[3], H11 = H[4], H12 = H[5];
    const double H20 = H[6], H21 = H[7], H22 = H[8];

    Accum acc;
    const uint8_t refCenterVal = refGray.at<uint8_t>(y, x);

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            int rx = x + dx;
            int ry = y + dy;
            if (rx < 0 || rx >= refGray.cols || ry < 0 || ry >= refGray.rows) continue;

            float refVal = refGray.at<uint8_t>(ry, rx);

            double denom = H20 * rx + H21 * ry + H22;
            if (std::abs(denom) < 1e-9) continue;
            double invDenom = 1.0 / denom;
            double srcX = (H00 * rx + H01 * ry + H02) * invDenom;
            double srcY = (H10 * rx + H11 * ry + H12) * invDenom;

            float srcVal = sampleGrayBilinearOriginal(srcGray, float(srcX), float(srcY));
            if (srcVal < 0) continue;
            ++acc.validCount;

            double spatialDist2 = double(dx * dx + dy * dy);
            double colorDiff = double(refVal) - double(refCenterVal);
            double colorDist2 = colorDiff * colorDiff;
            double w = std::exp(-spatialDist2 * invSs2 - colorDist2 * invSc2);

            acc.sumRef += refVal * w;
            acc.sumSrc += srcVal * w;
            acc.sumRefSq += refVal * refVal * w;
            acc.sumSrcSq += srcVal * srcVal * w;
            acc.sumRefSrc += refVal * srcVal * w;
            acc.sumW += w;
        }
    }

    return acc;
}

/// 当前实现：使用 sfm::pmv2::prepareWindow 预计算窗口，再累加
Accum optimizedCost(const cv::Mat &refGray, const cv::Mat &srcGray,
                    int x, int y, int r, double invSs2, double invSc2,
                    const double H[9])
{
    const double H00 = H[0], H01 = H[1], H02 = H[2];
    const double H10 = H[3], H11 = H[4], H12 = H[5];
    const double H20 = H[6], H21 = H[7], H22 = H[8];

    sfm::pmv2::WindowCache &cache = sfm::pmv2::windowCache();
    const int winCount = sfm::pmv2::prepareWindow(refGray, x, y, r, invSs2, invSc2, cache);

    Accum acc;
    for (int k = 0; k < winCount; ++k) {
        const int rx = cache.rx[k];
        const int ry = cache.ry[k];

        double denom = H20 * rx + H21 * ry + H22;
        if (std::abs(denom) < 1e-9) continue;
        double invDenom = 1.0 / denom;
        double srcX = (H00 * rx + H01 * ry + H02) * invDenom;
        double srcY = (H10 * rx + H11 * ry + H12) * invDenom;

        float srcVal = sampleGrayBilinearOriginal(srcGray, float(srcX), float(srcY));
        if (srcVal < 0) continue;
        ++acc.validCount;

        const float refVal = cache.refVal[k];
        const double w = cache.weight[k];

        acc.sumRef += refVal * w;
        acc.sumSrc += srcVal * w;
        acc.sumRefSq += refVal * refVal * w;
        acc.sumSrcSq += srcVal * srcVal * w;
        acc.sumRefSrc += refVal * srcVal * w;
        acc.sumW += w;
    }

    return acc;
}

bool sameBitwise(const Accum &a, const Accum &b)
{
    return a.validCount == b.validCount
        && a.sumRef == b.sumRef
        && a.sumSrc == b.sumSrc
        && a.sumRefSq == b.sumRefSq
        && a.sumSrcSq == b.sumSrcSq
        && a.sumRefSrc == b.sumRefSrc
        && a.sumW == b.sumW;
}

} // namespace

int main()
{
    int total = 0;
    int failed = 0;

    std::mt19937 rng(20261002u);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::uniform_int_distribution<int> uni8(0, 255);

    // 多组图像尺寸 × 窗口半径
    const int sizes[][2] = {{64, 48}, {37, 23}, {16, 16}, {9, 5}};
    const int radii[] = {1, 2, 5, 7};

    for (const auto &sz : sizes) {
        const int cols = sz[0];
        const int rows = sz[1];

        cv::Mat refGray(rows, cols, CV_8UC1);
        cv::Mat srcGray(rows, cols, CV_8UC1);
        for (int i = 0; i < rows * cols; ++i) {
            refGray.at<uint8_t>(i) = static_cast<uint8_t>(uni8(rng));
            srcGray.at<uint8_t>(i) = static_cast<uint8_t>(uni8(rng));
        }

        for (int r : radii) {
            const double sigmaSpatial = double(r);
            const double sigmaColor = 51.0;
            const double invSs2 = 1.0 / (2.0 * sigmaSpatial * sigmaSpatial);
            const double invSc2 = 1.0 / (2.0 * sigmaColor * sigmaColor);

            // 遍历全部像素，覆盖四个角与四条边的窗口越界情形
            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < cols; ++x) {
                    double H[9];
                    for (int i = 0; i < 9; ++i) H[i] = uni(rng) * 2.0;
                    H[8] = 1.0 + std::abs(uni(rng));  // 避免 denom 恒为 0

                    const Accum g = goldenCost(refGray, srcGray, x, y, r, invSs2, invSc2, H);
                    const Accum o = optimizedCost(refGray, srcGray, x, y, r, invSs2, invSc2, H);

                    ++total;
                    if (!sameBitwise(g, o)) {
                        if (failed < 10) {
                            std::printf("  不一致: size=%dx%d r=%d (x=%d,y=%d) "
                                        "valid %d/%d sumW %.17g/%.17g\n",
                                        cols, rows, r, x, y,
                                        g.validCount, o.validCount, g.sumW, o.sumW);
                        }
                        ++failed;
                    }
                }
            }
        }
    }

    std::printf("[patchmatch_test] 对比 %d 组（覆盖 4 种尺寸 × 4 种窗口半径 × 全部像素）\n", total);

    // ── aggregateTopK 单元测试 ──────────────────────────────────────────
    int topkFailed = 0;
    auto expectNear = [&](float got, float want, const char *what) {
        if (std::fabs(got - want) > 1e-6f) {
            std::printf("  aggregateTopK 失败: %s 期望 %.6f 实际 %.6f\n", what, want, got);
            ++topkFailed;
        }
    };
    {
        // 空 → maxCost
        std::vector<float> v;
        expectNear(sfm::pmv2::aggregateTopK(v, 4), 2.0f, "空输入返回 maxCost");
    }
    {
        // k=1 等价于 min
        std::vector<float> v = {0.9f, 0.2f, 1.5f, 0.4f, 0.7f};
        expectNear(sfm::pmv2::aggregateTopK(v, 1), 0.2f, "k=1 等价 min");
    }
    {
        // k>=n 退化为全部平均
        std::vector<float> v = {0.2f, 0.4f, 0.6f};
        expectNear(sfm::pmv2::aggregateTopK(v, 8), 0.4f, "k>=n 退化为全部平均");
    }
    {
        // 典型 k<n：取最优 k 个平均
        std::vector<float> v = {1.8f, 0.1f, 1.2f, 0.3f, 0.9f, 0.2f, 1.6f, 0.5f};
        // 最小 4 个: 0.1 0.2 0.3 0.5 → 平均 0.275
        expectNear(sfm::pmv2::aggregateTopK(v, 4), 0.275f, "k<n 取最优 k 平均");
    }
    {
        // 含重复值与 maxCost 边界
        std::vector<float> v = {2.0f, 2.0f, 0.0f};
        expectNear(sfm::pmv2::aggregateTopK(v, 2), 1.0f, "重复/边界值");
    }
    if (topkFailed == 0) {
        std::printf("[patchmatch_test] aggregateTopK 单元测试通过（5 组）\n");
    }

    // ── buildPlanarPrior 单元测试 ─────────────────────────────────────
    // 合成一个真实倾斜平面：n·X = ρ，逐像素生成精确深度，代价置 0.05。
    // 先验应当恢复出同一平面：内部像素先验深度≈真值、法向≈真法向。
    int priorFailed = 0;
    {
        const int W = 320, H = 240;
        const double fx = 500.0, fy = 500.0, cx = 160.0, cy = 120.0;
        // 真平面：法向 nz>0（本工程约定，指向相机），ρ=5
        double tn[3] = {0.10, -0.20, 0.97};
        const double tnl = std::sqrt(tn[0]*tn[0] + tn[1]*tn[1] + tn[2]*tn[2]);
        tn[0] /= tnl; tn[1] /= tnl; tn[2] /= tnl;
        const double rho = 5.0;

        cv::Mat depth(H, W, CV_32FC1);
        cv::Mat cost(H, W, CV_32FC1, cv::Scalar(0.05f));
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const double den = tn[0] * (x - cx) / fx + tn[1] * (y - cy) / fy + tn[2];
                depth.at<float>(y, x) = static_cast<float>(rho / den);
            }
        }

        auto prior = sfm::pmv2::buildPlanarPrior(depth, cost, fx, fy, cx, cy,
                                               5, 0.2f, 3.0, 7.0);
        const int cover = cv::countNonZero(prior.mask);
        if (prior.supportCount < 2000) {
            std::printf("  buildPlanarPrior 失败: 支撑点 %d 过少\n", prior.supportCount);
            ++priorFailed;
        }
        if (cover < W * H * 0.80) {
            std::printf("  buildPlanarPrior 失败: 覆盖率 %.1f%% 过低\n",
                        100.0 * cover / (W * H));
            ++priorFailed;
        }
        // 抽查内部若干点：深度与法向都应贴合真平面
        double maxDepthErr = 0.0, minDot = 1.0;
        int checked = 0;
        for (int y = 40; y < H - 40; y += 37) {
            for (int x = 40; x < W - 40; x += 41) {
                if (!prior.mask.at<uint8_t>(y, x)) continue;
                const double den = tn[0] * (x - cx) / fx + tn[1] * (y - cy) / fy + tn[2];
                const double trueD = rho / den;
                maxDepthErr = std::max(maxDepthErr,
                    std::fabs(double(prior.depth.at<float>(y, x)) - trueD));
                const cv::Vec3f n = prior.normal.at<cv::Vec3f>(y, x);
                const double dot = n[0] * tn[0] + n[1] * tn[1] + n[2] * tn[2];
                minDot = std::min(minDot, dot);
                ++checked;
            }
        }
        if (checked < 10 || maxDepthErr > 0.01 || minDot < 0.999) {
            std::printf("  buildPlanarPrior 失败: 抽查 %d 点, 深度误差 %.5f, 法向最小点积 %.5f\n",
                        checked, maxDepthErr, minDot);
            ++priorFailed;
        }

        // 边界情况：全部代价超阈 → 空先验，不得崩溃
        cv::Mat badCost(H, W, CV_32FC1, cv::Scalar(2.0f));
        auto emptyPrior = sfm::pmv2::buildPlanarPrior(depth, badCost, fx, fy, cx, cy,
                                                    5, 0.2f, 3.0, 7.0);
        if (emptyPrior.supportCount != 0 || cv::countNonZero(emptyPrior.mask) != 0) {
            std::printf("  buildPlanarPrior 失败: 高超阈代价应产生空先验\n");
            ++priorFailed;
        }
        if (priorFailed == 0) {
            std::printf("[patchmatch_test] buildPlanarPrior 单元测试通过"
                        "（支撑点 %d, 覆盖 %.1f%%, 深度误差 %.5f, 法向点积 %.5f）\n",
                        prior.supportCount, 100.0 * cover / (W * H), maxDepthErr, minDot);
        }
    }

    // ── 默认配置回归：GUI 默认走 Options{}，默认值即「全量验证」录定配置 ──
    // test14 全量验证（2026-10-03）：multiscale + geomConsistency + costTopK=4
    // + fusion + maxImageSize=1320 + planePrior → 一致率 92.1%、离群点 0.15%
    // （GUI 参考 0.33%）、重投影 2.81px。有人把默认值改回旧行为时此测试会立即失败。
    int defaultsFailed = 0;
    {
        PatchMatchReconstructorV2::Options opt;
        auto expectTrue = [&](bool v, const char *what) {
            if (!v) { std::printf("  默认值失败: %s 应为 true\n", what); ++defaultsFailed; }
        };
        expectTrue(opt.multiscale, "multiscale 默认开启");
        expectTrue(opt.geomConsistency, "geomConsistency 默认开启");
        expectTrue(opt.fusion, "fusion 默认开启");
        expectTrue(opt.planePrior, "planePrior 默认开启（2026-10-03 全量验证后提升）");
        if (opt.costTopK != 4) {
            std::printf("  默认值失败: costTopK 期望 4 实际 %d\n", opt.costTopK);
            ++defaultsFailed;
        }
        if (opt.maxImageSize != 1320) {
            std::printf("  默认值失败: maxImageSize 期望 1320 实际 %d\n", opt.maxImageSize);
            ++defaultsFailed;
        }
        if (defaultsFailed == 0) {
            std::printf("[patchmatch_test] 默认配置断言通过（MS+geom+fusion+topk4+1320+planePrior）\n");
        }
    }

    failed += topkFailed + priorFailed + defaultsFailed;
    if (failed == 0) {
        std::printf("[patchmatch_test] 通过：优化后与 golden 逐位一致\n");
        return 0;
    }
    std::printf("[patchmatch_test] 失败：%d 组不一致\n", failed);
    return 1;
}
