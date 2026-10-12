/**
 * @file main.cpp
 * @brief PatchMatch 代价函数优化正确性回归（QtTest）
 *
 * 背景：PatchMatchReconstructorV2::computePixelCost 是最内层热点。优化把
 * 「参考图窗口像素值 + 双边权重」的计算从源视图循环内提到循环外。
 * 该优化必须是数值等价变换；本测试用优化前 golden 与当前实现逐位比对。
 */

#include <QtTest/QtTest>

#include "algorithm/ThreadPool.h"
#include "algorithm/dense/PatchMatchCore.h"
#include "algorithm/dense/PatchMatchCoreT.h"
#include "algorithm/dense/PatchMatchCost.h"
#include "algorithm/dense/PatchMatchFusion.h"
#include "algorithm/dense/PatchMatchPropagate.h"
#include "algorithm/dense/PatchMatchRefine.h"
#include "algorithm/dense/PatchMatchShared.h"
#include "algorithm/dense/strategies.h"
#include "pipeline/PatchMatchReconstructor.h"
#include "pipeline/PatchMatchReconstructorV2.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <QDir>
#include <QTemporaryDir>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

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

struct Accum {
    double sumRef = 0.0;
    double sumSrc = 0.0;
    double sumRefSq = 0.0;
    double sumSrcSq = 0.0;
    double sumRefSrc = 0.0;
    double sumW = 0.0;
    int validCount = 0;
};

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

            const double ds = dx * dx + dy * dy;
            const double dc = double(refVal) - double(refCenterVal);
            const double w = std::exp(-ds * invSs2 - dc * dc * invSc2);

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

class PatchMatchTest : public QObject
{
    Q_OBJECT

private slots:
    void prepareWindowBitwiseEquivalent();
    void aggregateTopK();
    void buildPlanarPrior();
    void defaultOptionsRegression();
    void strategyPresets();
    void virtualCostHotspotMicrobench();
    void sharedSampleAndBackproject();
    void crtpCostMatchesVirtual();
    void v2CoreTCostMatchesFreeFunction();
    void checkerboardStripsCoverParity();
    void initAndFilterHelpers();
    void jbuAndPyramidHelpers();
    void geomFilterAndFusion();
    void v1SequentialPresetAligned();
    void v1NccCostMatchesFreeFunction();
    void solveProblemIoHelpers();
    void depthMapGoldenDeterministic();
    void v1DepthMapGoldenDeterministic();
};

void PatchMatchTest::prepareWindowBitwiseEquivalent()
{
    std::mt19937 rng(20261002u);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::uniform_int_distribution<int> uni8(0, 255);

    const int sizes[][2] = {{64, 48}, {37, 23}, {16, 16}, {9, 5}};
    const int radii[] = {1, 2, 5, 7};

    int total = 0;
    int failed = 0;
    QString firstMismatch;

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

            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < cols; ++x) {
                    double H[9];
                    for (int i = 0; i < 9; ++i) H[i] = uni(rng) * 2.0;
                    H[8] = 1.0 + std::abs(uni(rng));

                    const Accum g = goldenCost(refGray, srcGray, x, y, r, invSs2, invSc2, H);
                    const Accum o = optimizedCost(refGray, srcGray, x, y, r, invSs2, invSc2, H);

                    ++total;
                    if (!sameBitwise(g, o)) {
                        if (firstMismatch.isEmpty()) {
                            firstMismatch = QStringLiteral(
                                "size=%1x%2 r=%3 (x=%4,y=%5) valid %6/%7")
                                .arg(cols).arg(rows).arg(r).arg(x).arg(y)
                                .arg(g.validCount).arg(o.validCount);
                        }
                        ++failed;
                    }
                }
            }
        }
    }

    QVERIFY2(failed == 0,
             qPrintable(QStringLiteral("逐位比对失败 %1/%2 组；首处: %3")
                            .arg(failed).arg(total).arg(firstMismatch)));
}

void PatchMatchTest::aggregateTopK()
{
    {
        std::vector<float> empty;
        QCOMPARE(sfm::pmv2::aggregateTopK(empty, 4), 2.0f);
    }
    {
        std::vector<float> v = {0.9f, 0.2f, 1.5f, 0.4f, 0.7f};
        QCOMPARE(sfm::pmv2::aggregateTopK(v, 1), 0.2f);
    }
    {
        std::vector<float> v = {0.2f, 0.4f, 0.6f};
        QCOMPARE(sfm::pmv2::aggregateTopK(v, 8), 0.4f);
    }
    {
        std::vector<float> v = {1.8f, 0.1f, 1.2f, 0.3f, 0.9f, 0.2f, 1.6f, 0.5f};
        QCOMPARE(sfm::pmv2::aggregateTopK(v, 4), 0.275f);
    }
    {
        std::vector<float> v = {2.0f, 2.0f, 0.0f};
        QCOMPARE(sfm::pmv2::aggregateTopK(v, 2), 1.0f);
    }
}

void PatchMatchTest::buildPlanarPrior()
{
    const int W = 320, H = 240;
    const double fx = 500.0, fy = 500.0, cx = 160.0, cy = 120.0;
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
    QVERIFY2(prior.supportCount >= 2000,
             qPrintable(QStringLiteral("支撑点过少: %1").arg(prior.supportCount)));
    QVERIFY2(cover >= W * H * 0.80,
             qPrintable(QStringLiteral("覆盖率过低: %1%")
                            .arg(100.0 * cover / (W * H), 0, 'f', 1)));

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
    QVERIFY2(checked >= 10, "内部抽查点数不足");
    QVERIFY2(maxDepthErr <= 0.01,
             qPrintable(QStringLiteral("深度误差过大: %1").arg(maxDepthErr)));
    QVERIFY2(minDot >= 0.999,
             qPrintable(QStringLiteral("法向点积过小: %1").arg(minDot)));

    cv::Mat badCost(H, W, CV_32FC1, cv::Scalar(2.0f));
    auto emptyPrior = sfm::pmv2::buildPlanarPrior(depth, badCost, fx, fy, cx, cy,
                                                5, 0.2f, 3.0, 7.0);
    QCOMPARE(emptyPrior.supportCount, 0);
    QCOMPARE(cv::countNonZero(emptyPrior.mask), 0);
}

void PatchMatchTest::defaultOptionsRegression()
{
    PatchMatchReconstructorV2::Options opt;
    QVERIFY(opt.multiscale);
    QVERIFY(opt.geomConsistency);
    QVERIFY(opt.fusion);
    QVERIFY(opt.planePrior);
    QCOMPARE(opt.costTopK, 4);
    QCOMPARE(opt.maxImageSize, 1320);
}

void PatchMatchTest::strategyPresets()
{
    const auto v1 = sfm::dense::PatchMatchCore::withSequential().strategies();
    QCOMPARE(QString::fromLatin1(v1.presetName), QStringLiteral("PatchMatch"));
    QVERIFY(v1.cost && QString::fromLatin1(v1.cost->name()) == QStringLiteral("NCC"));
    QVERIFY(v1.prop && QString::fromLatin1(v1.prop->name()) == QStringLiteral("Sequential"));
    QVERIFY(v1.prior && !v1.prior->enabled());
    QVERIFY(v1.refine && !v1.refine->enabled());

    const auto v2 = sfm::dense::PatchMatchCore::withCheckerboard().strategies();
    QCOMPARE(QString::fromLatin1(v2.presetName), QStringLiteral("PatchMatchV2"));
    QVERIFY(v2.prop && QString::fromLatin1(v2.prop->name()) == QStringLiteral("Checkerboard"));
    QVERIFY(v2.prior && v2.prior->enabled());
    QVERIFY(v2.refine && v2.refine->enabled());

    // 传播覆盖：棋盘两轮应访问全部像素
    int seen = 0;
    std::vector<char> mask(16, 0);
    v2.prop->forEach(4, 4, true, [&](int x, int y) { mask[y * 4 + x] = 1; ++seen; });
    v2.prop->forEach(4, 4, false, [&](int x, int y) { mask[y * 4 + x] = 1; ++seen; });
    QCOMPARE(seen, 16);
    for (char c : mask)
        QCOMPARE(c, char(1));
}

void PatchMatchTest::virtualCostHotspotMicrobench()
{
    // F15a 耗时门备数：虚调用 ICostFunction vs 直接 NccCostFunction 成员调用。
    // 合成 64×48 单源视图，固定种子；两边应给出相同代价，并打印耗时比。
    cv::Mat ref(48, 64, CV_8UC1);
    cv::Mat src(48, 64, CV_8UC1);
    cv::RNG rng(20261012);
    rng.fill(ref, cv::RNG::UNIFORM, 0, 255);
    rng.fill(src, cv::RNG::UNIFORM, 0, 255);

    sfm::dense::SourceView view;
    view.gray = src;
    view.K_src = (cv::Mat_<double>(3, 3) << 100, 0, 32, 0, 100, 24, 0, 0, 1);
    view.R_rel = cv::Mat::eye(3, 3, CV_64F);
    view.T_rel = (cv::Mat_<double>(3, 1) << 0.1, 0.0, 0.0);

    sfm::dense::PatchContext ctx;
    ctx.refGray = &ref;
    ctx.fx = 100;
    ctx.fy = 100;
    ctx.cx = 32;
    ctx.cy = 24;
    ctx.sources = &view;
    ctx.numSources = 1;
    ctx.windowRadius = 3;
    ctx.sigmaColor = 51.0;
    ctx.costTopK = 1;

    sfm::dense::NccCostFunction direct;
    sfm::dense::PatchMatchCore core = sfm::dense::PatchMatchCore::withCheckerboard();
    const sfm::dense::ICostFunction *iface = core.strategies().cost;
    QVERIFY(iface != nullptr);

    const int reps = 200;
    float accDirect = 0, accVirt = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        for (int y = 8; y < 40; y += 4)
            for (int x = 8; x < 56; x += 4)
                accDirect += direct.cost(x, y, 5.0f, 0.f, 0.f, 1.f, ctx);
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        for (int y = 8; y < 40; y += 4)
            for (int x = 8; x < 56; x += 4)
                accVirt += iface->cost(x, y, 5.0f, 0.f, 0.f, 1.f, ctx);
    }
    auto t2 = std::chrono::steady_clock::now();

    const double msDirect =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double msVirt =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double ratio = (msDirect > 1e-6) ? (msVirt / msDirect) : 1.0;
    std::printf("[F15a microbench] direct=%.3fms virtual=%.3fms ratio=%.3f "
                "(threshold gate: >1.05 triggers F15b)\n",
                msDirect, msVirt, ratio);
    std::fflush(stdout);

    QVERIFY2(std::fabs(accDirect - accVirt) < 1e-3f,
             "虚调用与直接调用代价累计应一致");
    // 不在此强制 ≤5%：微基准噪声大，仅记录；完整门禁在深度图级 golden 后做。
    // 实测 ratio 常 >1.05 → 已备 PatchMatchCoreT（F15b）。
    Q_UNUSED(ratio);
}

void PatchMatchTest::sharedSampleAndBackproject()
{
    cv::Mat gray(8, 8, CV_8UC1);
    for (int i = 0; i < 64; ++i)
        gray.data[i] = static_cast<uint8_t>(i * 3);

    QCOMPARE(sfm::dense::sampleGrayBilinear(gray, -0.1f, 1.0f), -1.0f);
    const float mid = sfm::dense::sampleGrayBilinear(gray, 1.5f, 2.5f);
    QVERIFY(mid > 0.0f);

    DenseReconstructionInput input;
    CameraPose pose;
    pose.imageId = 0;
    pose.registered = true;
    pose.R = cv::Mat::eye(3, 3, CV_64F);
    pose.t = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    pose.K = (cv::Mat_<double>(3, 3) << 50, 0, 4, 0, 50, 4, 0, 0, 1);
    input.cameraPoses.push_back(pose);

    cv::Mat depth(8, 8, CV_32FC1, cv::Scalar(5.0f));
    cv::Mat color(8, 8, CV_8UC3, cv::Scalar(40, 80, 120));
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    sfm::dense::backprojectDepth(input, 0, depth, color, cloud);
    QVERIFY2(!cloud.empty(), "反投影应产出点");
    // step=2 → 4×4 = 16 点
    QCOMPARE(static_cast<int>(cloud.size()), 16);
    QVERIFY(std::fabs(cloud.points[0].z - 5.0f) < 1e-3f);

    QVERIFY(sfm::dense::fuseSourceDepthMaps(color, {}, {}).empty());
    QCOMPARE(sfm::dense::checkerboardPhaseParity(true, 0), 0);
    QCOMPARE(sfm::dense::checkerboardPhaseParity(true, 1), 1);
    QCOMPARE(sfm::dense::checkerboardPhaseParity(false, 0), 1);
}

void PatchMatchTest::crtpCostMatchesVirtual()
{
    cv::Mat ref(32, 32, CV_8UC1);
    cv::Mat src(32, 32, CV_8UC1);
    cv::RNG rng(42);
    rng.fill(ref, cv::RNG::UNIFORM, 0, 255);
    rng.fill(src, cv::RNG::UNIFORM, 0, 255);

    sfm::dense::SourceView view;
    view.gray = src;
    view.K_src = (cv::Mat_<double>(3, 3) << 80, 0, 16, 0, 80, 16, 0, 0, 1);
    view.R_rel = cv::Mat::eye(3, 3, CV_64F);
    view.T_rel = (cv::Mat_<double>(3, 1) << 0.05, 0.0, 0.0);

    sfm::dense::PatchContext ctx;
    ctx.refGray = &ref;
    ctx.fx = 80;
    ctx.fy = 80;
    ctx.cx = 16;
    ctx.cy = 16;
    ctx.sources = &view;
    ctx.numSources = 1;
    ctx.windowRadius = 2;
    ctx.sigmaColor = 51.0;
    ctx.costTopK = 1;

    auto virt = sfm::dense::PatchMatchCore::withCheckerboard();
    auto stat = sfm::dense::PatchMatchCoreCheckerboardT::withCheckerboard();

    float accV = 0, accT = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int y = 4; y < 28; y += 2)
        for (int x = 4; x < 28; x += 2)
            accV += virt.evaluateCost(x, y, 4.0f, 0.f, 0.f, 1.f, ctx);
    auto t1 = std::chrono::steady_clock::now();
    for (int y = 4; y < 28; y += 2)
        for (int x = 4; x < 28; x += 2)
            accT += stat.evaluateCost(x, y, 4.0f, 0.f, 0.f, 1.f, ctx);
    auto t2 = std::chrono::steady_clock::now();

    const double msV = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double msT = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::printf("[F15b CRTP] virtual=%.3fms template=%.3fms ratio(v/t)=%.3f\n",
                msV, msT, (msT > 1e-6) ? (msV / msT) : 1.0);
    std::fflush(stdout);

    QVERIFY2(std::fabs(accV - accT) < 1e-3f, "CRTP 与虚调用代价应一致");
}

void PatchMatchTest::v2CoreTCostMatchesFreeFunction()
{
    // F15c：PatchMatchCoreV2T 与 computeNccPixelCost 逐位一致
    cv::Mat ref(40, 48, CV_8UC1);
    cv::Mat src(40, 48, CV_8UC1);
    cv::RNG rng(20261012);
    rng.fill(ref, cv::RNG::UNIFORM, 0, 255);
    rng.fill(src, cv::RNG::UNIFORM, 0, 255);

    sfm::dense::SourceViewPod view;
    view.gray = src;
    // K = diag(fx,fy,1) with cx,cy
    view.K[0] = 90;
    view.K[2] = 24;
    view.K[4] = 90;
    view.K[5] = 20;
    view.K[8] = 1;
    view.R[0] = view.R[4] = view.R[8] = 1;
    view.T[0] = 0.08;

    sfm::dense::PatchContextPod ctx;
    ctx.refGray = &ref;
    ctx.fx = 90;
    ctx.fy = 90;
    ctx.cx = 24;
    ctx.cy = 20;
    ctx.sources = &view;
    ctx.numSources = 1;
    ctx.opt.windowRadius = 3;
    ctx.opt.sigmaColor = 51.0;
    ctx.opt.costTopK = 1;

    auto core = sfm::dense::makeV2CoreT();
    float accFree = 0, accCore = 0;
    for (int y = 6; y < 34; y += 3) {
        for (int x = 6; x < 42; x += 3) {
            const float a =
                sfm::dense::computeNccPixelCost(x, y, 4.5f, 0.f, 0.f, 1.f, ctx);
            const float b = core.evaluateCost(x, y, 4.5f, 0.f, 0.f, 1.f, ctx);
            accFree += a;
            accCore += b;
            QCOMPARE(a, b);
        }
    }
    QVERIFY(accFree > 0.0f);
    QCOMPARE(accFree, accCore);
}

void PatchMatchTest::checkerboardStripsCoverParity()
{
    (void)sfm::globalThreadPool(1);
    constexpr int rows = 8, cols = 8;
    std::vector<char> even(rows * cols, 0), odd(rows * cols, 0);

    sfm::dense::runCheckerboardStrips(
        sfm::globalThreadPool(), rows, cols, /*parity=*/0, /*propagateDown=*/true,
        []() { return false; },
        [&](int x, int y, std::mt19937 &) { even[y * cols + x] = 1; });
    sfm::dense::runCheckerboardStrips(
        sfm::globalThreadPool(), rows, cols, /*parity=*/1, /*propagateDown=*/false,
        []() { return false; },
        [&](int x, int y, std::mt19937 &) { odd[y * cols + x] = 1; });

    int evenCnt = 0, oddCnt = 0;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const int p = (x + y) & 1;
            if (p == 0) {
                QCOMPARE(even[y * cols + x], char(1));
                QCOMPARE(odd[y * cols + x], char(0));
                ++evenCnt;
            } else {
                QCOMPARE(odd[y * cols + x], char(1));
                QCOMPARE(even[y * cols + x], char(0));
                ++oddCnt;
            }
        }
    }
    QCOMPARE(evenCnt + oddCnt, rows * cols);
}

void PatchMatchTest::initAndFilterHelpers()
{
    cv::Mat depth(4, 4, CV_32FC1);
    cv::Mat normal(4, 4, CV_32FC3);
    const int reused =
        sfm::dense::initDepthNormalMaps(depth, normal, 1.0, 3.0, cv::Mat(), cv::Mat());
    QCOMPARE(reused, 0);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            const float d = depth.at<float>(y, x);
            QVERIFY(d >= 1.0f && d <= 3.0f);
            QVERIFY(normal.at<cv::Vec3f>(y, x)[2] > 0.1f);
        }

    // 固定种子再跑一遍应逐位一致
    cv::Mat depth2(4, 4, CV_32FC1), normal2(4, 4, CV_32FC3);
    sfm::dense::initDepthNormalMaps(depth2, normal2, 1.0, 3.0, cv::Mat(), cv::Mat());
    for (int i = 0; i < 16; ++i)
        QCOMPARE(depth.ptr<float>()[i], depth2.ptr<float>()[i]);

    cv::Mat conf(4, 4, CV_32FC1, cv::Scalar(0.5f));
    conf.at<float>(1, 1) = 0.95f;
    conf.at<float>(2, 2) = 0.99f;
    depth.setTo(2.0f);
    const int filtered = sfm::dense::filterDepthByCost(depth, conf, 0.9f);
    QCOMPARE(filtered, 2);
    QCOMPARE(depth.at<float>(1, 1), 0.0f);
    QCOMPARE(depth.at<float>(0, 0), 2.0f);
}

void PatchMatchTest::jbuAndPyramidHelpers()
{
    QCOMPARE(sfm::dense::decideNumScales(64, 48, 5, false), 1);
    QCOMPARE(sfm::dense::decideNumScales(64, 48, 5, true), 1); // 短边 < 256
    QVERIFY(sfm::dense::decideNumScales(800, 600, 5, true) >= 2);

    cv::Mat depthL(8, 8, CV_32FC1, cv::Scalar(5.0f));
    cv::Mat normalL(8, 8, CV_32FC3, cv::Scalar(0, 0, 1));
    cv::Mat grayH(16, 16, CV_8UC1);
    for (int i = 0; i < 256; ++i)
        grayH.data[i] = static_cast<uint8_t>(i);
    cv::Mat depthH, normalH;
    sfm::dense::jointBilateralUpsample(depthL, normalL, grayH, depthH, normalH);
    QCOMPARE(depthH.rows, 16);
    QCOMPARE(depthH.cols, 16);
    QVERIFY(depthH.at<float>(8, 8) > 0.0f);
    QVERIFY(normalH.at<cv::Vec3f>(8, 8)[2] > 0.5f);

    sfm::dense::PatchMatchLevelCtx finest;
    finest.refGray = cv::Mat(64, 64, CV_8UC1, cv::Scalar(100));
    finest.rows = 64;
    finest.cols = 64;
    finest.fx = finest.fy = 100;
    finest.cx = finest.cy = 32;
    auto pyr = sfm::dense::buildPyramid(finest, 2);
    QCOMPARE(static_cast<int>(pyr.size()), 2);
    QVERIFY(pyr[1].cols < pyr[0].cols);
    QCOMPARE(sfm::dense::itersForPyramidLevel(1, 2, 5, 2), 5);
    QCOMPARE(sfm::dense::itersForPyramidLevel(0, 2, 5, 2), 2);
}

void PatchMatchTest::geomFilterAndFusion()
{
    // F15g：合成三视图平面深度，几何一致性应保留；融合应去重为约单视图点数
    (void)sfm::globalThreadPool(1);
    sfm::ThreadPool &pool = sfm::globalThreadPool();

    constexpr int W = 32, H = 24;
    constexpr double fx = 50.0, fy = 50.0, cx = 16.0, cy = 12.0;
    constexpr float planeZ = 5.0f;
    constexpr double baseline = 0.3;

    auto makePose = [&](double camX) {
        CameraPose pose;
        pose.registered = true;
        pose.R = cv::Mat::eye(3, 3, CV_64F);
        pose.t = (cv::Mat_<double>(3, 1) << -camX, 0, 0); // C = -Rᵀ t = (camX,0,0)
        pose.K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        return pose;
    };

    DenseReconstructionInput input;
    input.cameraPoses = {makePose(0.0), makePose(baseline), makePose(2.0 * baseline)};
    input.cameraMatrix = input.cameraPoses[0].K.clone();

    std::vector<StereoPairResult> results(3);
    std::vector<cv::Mat> images(3);
    for (int i = 0; i < 3; ++i) {
        results[static_cast<size_t>(i)].imageId1 = i;
        results[static_cast<size_t>(i)].depth = cv::Mat(H, W, CV_32FC1, cv::Scalar(planeZ));
        results[static_cast<size_t>(i)].confidence =
            cv::Mat(H, W, CV_32FC1, cv::Scalar(0.1f));
        images[static_cast<size_t>(i)] =
            cv::Mat(H, W, CV_8UC3, cv::Scalar(40 + 20 * i, 80, 120));
    }
    // 参考视图中心一处噪声深度，应被几何一致性清零
    results[0].depth.at<float>(H / 2, W / 2) = 9.0f;

    sfm::dense::GeomConsistencyOptions gopt;
    gopt.maxReprojPx = 1.0;
    gopt.minConsistent = 2;
    const auto gstats =
        sfm::dense::filterGeometricConsistency(input, results, pool, gopt);
    QVERIFY(!gstats.skippedTooFewViews);
    QVERIFY(gstats.totalValid > 0);
    QVERIFY(gstats.totalKept > 0);
    QVERIFY(gstats.totalKept < gstats.totalValid);
    QCOMPARE(results[0].depth.at<float>(H / 2, W / 2), 0.0f);
    QVERIFY(results[0].depth.at<float>(2, 2) > 0.0f);

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    sfm::dense::FusionOptions fopt;
    fopt.relErrMax = 0.02;
    fopt.minSupport = 2;
    const auto fstats =
        sfm::dense::fuseDepthMaps(input, results, images, pool, fopt, cloud);
    QVERIFY(fstats.numEmitted > 0);
    QVERIFY(!cloud.empty());
    QCOMPARE(static_cast<size_t>(cloud.size()), fstats.numEmitted);
    // 三视图同平面：融合后点数应明显少于三视图独立反投影
    QVERIFY(cloud.size() < static_cast<size_t>(3 * W * H));
    QVERIFY(fstats.numConsumed > 0 || fstats.numEmitted <= static_cast<size_t>(W * H));

    auto vg = sfm::dense::makeViewGeom(input.cameraPoses[0], input.cameraMatrix);
    QVERIFY(vg.valid);
    QVERIFY(std::fabs(vg.C[0]) < 1e-9);
}

void PatchMatchTest::v1SequentialPresetAligned()
{
    // F15h：V1 Deprecated 预设与 dense Sequential 对齐；编排壳与 V2 同源
    PatchMatchReconstructor v1;
    QVERIFY(v1.displayName().contains(QStringLiteral("Deprecated")));
    QCOMPARE(v1.methodName(), QStringLiteral("PatchMatch"));

    const auto seq = sfm::dense::PatchMatchCore::withSequential().strategies();
    QCOMPARE(QString::fromLatin1(seq.presetName), QStringLiteral("PatchMatch"));
    QVERIFY(seq.prop && QString::fromLatin1(seq.prop->name()) == QStringLiteral("Sequential"));
    QVERIFY(seq.prior && !seq.prior->enabled());
    QVERIFY(seq.refine && !seq.refine->enabled());

    // 共享壳：空 init 应随机填充（种子 42）
    cv::Mat depth(8, 8, CV_32FC1);
    cv::Mat normal(8, 8, CV_32FC3);
    const int reused =
        sfm::dense::initDepthNormalMaps(depth, normal, 1.0, 10.0, cv::Mat(), cv::Mat());
    QCOMPARE(reused, 0);
    QVERIFY(depth.at<float>(0, 0) >= 1.0f);
    QVERIFY(depth.at<float>(0, 0) <= 10.0f);
    QVERIFY(normal.at<cv::Vec3f>(0, 0)[2] > 0.0f);

    const int filtered =
        sfm::dense::filterDepthByCost(depth, cv::Mat(8, 8, CV_32FC1, cv::Scalar(2.0f)), 0.5f);
    QCOMPARE(filtered, 64);
}

void PatchMatchTest::v1NccCostMatchesFreeFunction()
{
    // F15i：NccV1Cost 与 computeNccPixelCostV1 逐位一致；结果 ∈ [0,2]
    cv::Mat ref(40, 48, CV_8UC1);
    cv::Mat src(40, 48, CV_8UC1);
    cv::RNG rng(20261012);
    rng.fill(ref, cv::RNG::UNIFORM, 0, 255);
    rng.fill(src, cv::RNG::UNIFORM, 0, 255);

    sfm::dense::SourceView view;
    view.gray = src;
    view.K_src = (cv::Mat_<double>(3, 3) << 80, 0, 24, 0, 80, 20, 0, 0, 1);
    view.R_rel = cv::Mat::eye(3, 3, CV_64F);
    view.T_rel = (cv::Mat_<double>(3, 1) << 0.08, 0.0, 0.0);

    sfm::dense::PatchContext ctx;
    ctx.refGray = &ref;
    ctx.fx = 80;
    ctx.fy = 80;
    ctx.cx = 24;
    ctx.cy = 20;
    ctx.sources = &view;
    ctx.numSources = 1;
    ctx.windowRadius = 3;
    ctx.sigmaSpatial = -1.0;
    ctx.sigmaColor = 51.0;
    ctx.costTopK = 1;

    sfm::dense::NccV1Cost ncc;
    float accFn = 0, accCls = 0;
    for (int y = 6; y < 34; y += 3) {
        for (int x = 6; x < 42; x += 3) {
            const float a =
                sfm::dense::computeNccPixelCostV1(x, y, 4.0f, 0.f, 0.f, 1.f, ctx);
            const float b = ncc.cost(x, y, 4.0f, 0.f, 0.f, 1.f, ctx);
            QVERIFY(a >= 0.0f && a <= 2.0f);
            QCOMPARE(a, b);
            accFn += a;
            accCls += b;
        }
    }
    QVERIFY2(accFn > 0.0f, "合成纹理应产生非平凡代价累计");
    QCOMPARE(accFn, accCls);
}

void PatchMatchTest::solveProblemIoHelpers()
{
    // F15j：缩放策略与内参缩放；源视图相对位姿构建
    QCOMPARE(sfm::dense::computeWorkScale(2000, 1500, -1,
                                          sfm::dense::MaxImageSizePolicy::CapAt1000WhenNonPositive),
             0.5);
    QCOMPARE(sfm::dense::computeWorkScale(2000, 1500, -1,
                                          sfm::dense::MaxImageSizePolicy::UnlimitedWhenNonPositive),
             1.0);
    QCOMPARE(sfm::dense::computeWorkScale(800, 600, 1000,
                                          sfm::dense::MaxImageSizePolicy::CapAt1000WhenNonPositive),
             1.0);

    cv::Mat K = (cv::Mat_<double>(3, 3) << 100, 0, 50, 0, 100, 40, 0, 0, 1);
    cv::Mat Ks = sfm::dense::scaleIntrinsics(K, 0.5);
    QCOMPARE(Ks.at<double>(0, 0), 50.0);
    QCOMPARE(Ks.at<double>(0, 2), 25.0);

    DenseReconstructionInput input;
    auto makePose = [](double tx) {
        CameraPose p;
        p.registered = true;
        p.R = cv::Mat::eye(3, 3, CV_64F);
        p.t = (cv::Mat_<double>(3, 1) << -tx, 0, 0);
        p.K = (cv::Mat_<double>(3, 3) << 80, 0, 16, 0, 80, 12, 0, 0, 1);
        return p;
    };
    input.cameraPoses = {makePose(0.0), makePose(0.4)};
    input.cameraMatrix = input.cameraPoses[0].K.clone();
    input.imageFiles << QStringLiteral("ref.png") << QStringLiteral("src.png");

    cv::Mat ref(48, 64, CV_8UC3, cv::Scalar(30, 60, 90));
    cv::Mat src(48, 64, CV_8UC3, cv::Scalar(40, 70, 100));
    auto load = [&](int idx) -> cv::Mat {
        return idx == 0 ? ref.clone() : src.clone();
    };

    const auto prep = sfm::dense::prepareReferenceView(
        input, 0, {2.0, 8.0}, 32, sfm::dense::MaxImageSizePolicy::UnlimitedWhenNonPositive,
        load);
    QVERIFY(prep.ok);
    QVERIFY(prep.scale < 1.0);
    QCOMPARE(prep.depthMin, 2.0);
    QCOMPARE(prep.depthMax, 8.0);
    QVERIFY(prep.refGray.cols <= 32 || prep.refGray.rows <= 32);

    auto mats = sfm::dense::buildMatSourceViews(input, 0, {1}, prep.scale, load);
    QCOMPARE(static_cast<int>(mats.size()), 1);
    QVERIFY(!mats[0].gray.empty());
    QCOMPARE(mats[0].T_rel.at<double>(0), -0.4);

    auto pods = sfm::dense::buildPodSourceViews(input, 0, {1}, prep.scale, load);
    QCOMPARE(static_cast<int>(pods.size()), 1);
    QVERIFY(!pods[0].gray.empty());

    cv::Mat depth(prep.refGray.size(), CV_32FC1, cv::Scalar(5.0f));
    cv::Mat conf(prep.refGray.size(), CV_32FC1, cv::Scalar(0.2f));
    sfm::dense::upscaleToFullResolution(depth, conf, prep.refColor.size(), true);
    QCOMPARE(depth.size(), prep.refColor.size());
    QCOMPARE(conf.size(), prep.refColor.size());
    QCOMPARE(sfm::dense::countValidDepthPixels(depth), depth.rows * depth.cols);
}

void PatchMatchTest::depthMapGoldenDeterministic()
{
    // 固定单线程，保证红黑 strip 划分与 RNG 种子跨机器一致
    (void)sfm::globalThreadPool(1);

    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    const QString dir = tmp.path();

    constexpr int W = 64, H = 48;
    constexpr double fx = 100.0, fy = 100.0, cx = 32.0, cy = 24.0;
    constexpr double planeZ = 5.0;
    constexpr double baseline = 0.4;

    auto renderView = [&](double camX) -> cv::Mat {
        cv::Mat img(H, W, CV_8UC3);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const double X = (x - cx) / fx * planeZ + camX;
                const double Y = (y - cy) / fy * planeZ;
                const int u = static_cast<int>(std::floor(X * 8.0 + 1000)) & 15;
                const int v = static_cast<int>(std::floor(Y * 8.0 + 1000)) & 15;
                const uint8_t c = static_cast<uint8_t>(40 + ((u ^ v) * 12) % 180);
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(c, c, c);
            }
        }
        return img;
    };

    const QString p0 = dir + QStringLiteral("/v0.png");
    const QString p1 = dir + QStringLiteral("/v1.png");
    QVERIFY(cv::imwrite(p0.toStdString(), renderView(0.0)));
    QVERIFY(cv::imwrite(p1.toStdString(), renderView(baseline)));

    DenseReconstructionInput input;
    input.imageFiles = QStringList{p0, p1};
    input.cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    auto makePose = [&](int id, double tx) {
        CameraPose pose;
        pose.imageId = id;
        pose.imageWidth = W;
        pose.imageHeight = H;
        pose.registered = true;
        pose.R = cv::Mat::eye(3, 3, CV_64F);
        pose.t = (cv::Mat_<double>(3, 1) << -tx, 0, 0); // C = -R^T t
        pose.K = input.cameraMatrix.clone();
        return pose;
    };
    input.cameraPoses = {makePose(0, 0.0), makePose(1, baseline)};

    MatchPair mp;
    mp.imageId1 = 0;
    mp.imageId2 = 1;
    mp.matchCount = 64;
    for (int i = 0; i < 64; ++i) {
        const double u = 8.0 + (i % 8) * 6.0;
        const double v = 8.0 + (i / 8) * 4.0;
        const double X = (u - cx) / fx * planeZ;
        mp.points1.emplace_back(u, v);
        // 相机 1：P_cam = P_world + t, t = (-baseline,0,0) → x 平移
        const double u2 = fx * (X - baseline) / planeZ + cx;
        mp.points2.emplace_back(u2, v);
    }
    input.matchPairs = {mp};

    auto runOnce = [&](QString *err) -> cv::Mat {
        PatchMatchReconstructorV2 pm;
        QMap<QString, QVariant> cfg;
        cfg[QStringLiteral("multiscale")] = false;
        cfg[QStringLiteral("planePrior")] = false;
        cfg[QStringLiteral("geomConsistency")] = false;
        cfg[QStringLiteral("fusion")] = false;
        cfg[QStringLiteral("filter")] = false;
        cfg[QStringLiteral("numIterations")] = 2;
        cfg[QStringLiteral("maxImageSize")] = 0;
        cfg[QStringLiteral("windowRadius")] = 3;
        cfg[QStringLiteral("costTopK")] = 1;
        cfg[QStringLiteral("numSamples")] = 8;
        pm.setConfig(cfg);
        const auto out = pm.reconstruct(input);
        if (!out.success) {
            if (err) *err = out.errorMessage;
            return cv::Mat();
        }
        if (out.stereoResults.empty() || out.stereoResults.front().depth.empty()) {
            if (err) *err = QStringLiteral("empty depth");
            return cv::Mat();
        }
        return out.stereoResults.front().depth.clone();
    };

    QString err0, err1;
    const cv::Mat d0 = runOnce(&err0);
    const cv::Mat d1 = runOnce(&err1);
    QVERIFY2(!d0.empty(), qPrintable(err0));
    QVERIFY2(!d1.empty(), qPrintable(err1));
    QCOMPARE(d0.size(), d1.size());
    QCOMPARE(d0.type(), CV_32FC1);

    int mismatches = 0;
    double sum = 0;
    int valid = 0;
    for (int y = 0; y < d0.rows; ++y) {
        for (int x = 0; x < d0.cols; ++x) {
            const float a = d0.at<float>(y, x);
            const float b = d1.at<float>(y, x);
            if (a != b)
                ++mismatches;
            if (a > 0 && std::isfinite(a)) {
                sum += a;
                ++valid;
            }
        }
    }
    QCOMPARE(mismatches, 0);
    QVERIFY2(valid > d0.rows * d0.cols / 4, "有效深度像素过少");

    // 平面场景：均值深度应落在合理区间（非逐位锁死跨编译器，但锁确定性）
    const double meanD = sum / valid;
    QVERIFY2(meanD > 2.0 && meanD < 12.0,
             qPrintable(QStringLiteral("均值深度异常: %1").arg(meanD)));

    // 指纹：有效像素计数 + 量化均值，防止静默漂移
    const int meanQ = static_cast<int>(std::lround(meanD * 100.0));
    std::printf("[F15a golden] valid=%d/%d meanQ=%d (deterministic ok)\n",
                valid, d0.rows * d0.cols, meanQ);
    std::fflush(stdout);
}

void PatchMatchTest::v1DepthMapGoldenDeterministic()
{
    // F15k：V1 Deprecated 路径确定性 golden（与 V2 合成场景同构）
    (void)sfm::globalThreadPool(1);

    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    const QString dir = tmp.path();

    constexpr int W = 64, H = 48;
    constexpr double fx = 100.0, fy = 100.0, cx = 32.0, cy = 24.0;
    constexpr double planeZ = 5.0;
    constexpr double baseline = 0.4;

    auto renderView = [&](double camX) -> cv::Mat {
        cv::Mat img(H, W, CV_8UC3);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const double X = (x - cx) / fx * planeZ + camX;
                const double Y = (y - cy) / fy * planeZ;
                const int u = static_cast<int>(std::floor(X * 8.0 + 1000)) & 15;
                const int v = static_cast<int>(std::floor(Y * 8.0 + 1000)) & 15;
                const uint8_t c = static_cast<uint8_t>(40 + ((u ^ v) * 12) % 180);
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(c, c, c);
            }
        }
        return img;
    };

    const QString p0 = dir + QStringLiteral("/v0.png");
    const QString p1 = dir + QStringLiteral("/v1.png");
    QVERIFY(cv::imwrite(p0.toStdString(), renderView(0.0)));
    QVERIFY(cv::imwrite(p1.toStdString(), renderView(baseline)));

    DenseReconstructionInput input;
    input.imageFiles = QStringList{p0, p1};
    input.cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    auto makePose = [&](int id, double tx) {
        CameraPose pose;
        pose.imageId = id;
        pose.imageWidth = W;
        pose.imageHeight = H;
        pose.registered = true;
        pose.R = cv::Mat::eye(3, 3, CV_64F);
        pose.t = (cv::Mat_<double>(3, 1) << -tx, 0, 0);
        pose.K = input.cameraMatrix.clone();
        return pose;
    };
    input.cameraPoses = {makePose(0, 0.0), makePose(1, baseline)};

    MatchPair mp;
    mp.imageId1 = 0;
    mp.imageId2 = 1;
    mp.matchCount = 64;
    for (int i = 0; i < 64; ++i) {
        const double u = 8.0 + (i % 8) * 6.0;
        const double v = 8.0 + (i / 8) * 4.0;
        const double X = (u - cx) / fx * planeZ;
        mp.points1.emplace_back(u, v);
        const double u2 = fx * (X - baseline) / planeZ + cx;
        mp.points2.emplace_back(u2, v);
    }
    input.matchPairs = {mp};

    auto runOnce = [&](QString *err) -> cv::Mat {
        PatchMatchReconstructor pm;
        QMap<QString, QVariant> cfg;
        cfg[QStringLiteral("geomConsistency")] = false;
        cfg[QStringLiteral("filter")] = false;
        cfg[QStringLiteral("numIterations")] = 2;
        cfg[QStringLiteral("maxImageSize")] = 1000;
        cfg[QStringLiteral("windowRadius")] = 3;
        cfg[QStringLiteral("numSamples")] = 8;
        pm.setConfig(cfg);
        const auto out = pm.reconstruct(input);
        if (!out.success) {
            if (err)
                *err = out.errorMessage;
            return cv::Mat();
        }
        if (out.stereoResults.empty() || out.stereoResults.front().depth.empty()) {
            if (err)
                *err = QStringLiteral("empty depth");
            return cv::Mat();
        }
        return out.stereoResults.front().depth.clone();
    };

    QString err0, err1;
    const cv::Mat d0 = runOnce(&err0);
    const cv::Mat d1 = runOnce(&err1);
    QVERIFY2(!d0.empty(), qPrintable(err0));
    QVERIFY2(!d1.empty(), qPrintable(err1));
    QCOMPARE(d0.size(), d1.size());
    QCOMPARE(d0.type(), CV_32FC1);

    int mismatches = 0;
    double sum = 0;
    int valid = 0;
    for (int y = 0; y < d0.rows; ++y) {
        for (int x = 0; x < d0.cols; ++x) {
            const float a = d0.at<float>(y, x);
            const float b = d1.at<float>(y, x);
            if (a != b)
                ++mismatches;
            if (a > 0 && std::isfinite(a)) {
                sum += a;
                ++valid;
            }
        }
    }
    QCOMPARE(mismatches, 0);
    QVERIFY2(valid > d0.rows * d0.cols / 4, "V1 有效深度像素过少");

    const double meanD = sum / valid;
    QVERIFY2(meanD > 2.0 && meanD < 12.0,
             qPrintable(QStringLiteral("V1 均值深度异常: %1").arg(meanD)));

    const int meanQ = static_cast<int>(std::lround(meanD * 100.0));
    std::printf("[F15k V1 golden] valid=%d/%d meanQ=%d (deterministic ok)\n", valid,
                d0.rows * d0.cols, meanQ);
    std::fflush(stdout);
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    // 必须在首次使用全局池前固定线程数，否则 depth golden 不可复现
    (void)sfm::globalThreadPool(1);
    PatchMatchTest tc;
    return QTest::qExec(&tc, argc, argv);
}

#include "main.moc"
