#include "algorithm/dense/PatchMatchPropagate.h"

#include "algorithm/dense/PatchMatchShared.h"
#include "pipeline/PatchMatchReconstructorV2.h" // pmv2::PlanarPrior / buildPlanarPrior

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <vector>

namespace sfm {
namespace dense {

void processPixel(int x, int y, cv::Mat &depthMap, cv::Mat &normalMap, cv::Mat &confidence,
                  int rows, int cols, double fx, double fy, double cx, double cy,
                  double depthMin, double depthMax, bool propagateDown,
                  const sfm::pmv2::PlanarPrior *prior, std::mt19937 &rng,
                  const PixelCostFn &costFn)
{
    auto genRandom01 = [&rng]() -> float {
        return static_cast<float>(rng()) / static_cast<float>(rng.max());
    };
    auto genRandomSigned = [&genRandom01]() -> float {
        return 2.0f * genRandom01() - 1.0f;
    };

    const float depthRange = float(depthMax - depthMin);
    const float initialDispUpdate = depthRange * 0.5f;
    const float initialNormUpdate = 1.0f;
    const float refineStopThres = 0.1f;

    float currDepth = depthMap.at<float>(y, x);
    cv::Vec3f currNormal = normalMap.at<cv::Vec3f>(y, x);

    float bestDepth = currDepth;
    cv::Vec3f bestNormal = currNormal;
    float bestCost = costFn(x, y, bestDepth, bestNormal[0], bestNormal[1], bestNormal[2]);

    // ── 平面先验（ACMP 受限后验）─────────────────────────────────
    const bool hasPrior = prior && prior->mask.at<uint8_t>(y, x) != 0;
    float priorD = 0.0f;
    cv::Vec3f priorN(0.0f, 0.0f, 1.0f);
    const float priorGamma = 0.5f;
    const float priorBeta = 0.18f;
    const float priorSigmaD = depthRange / 64.0f;
    const float priorSigmaA = 3.14159265f * (5.0f / 180.0f);
    float invTwoSigmaD2 = 0.0f, invTwoSigmaA2 = 0.0f;
    float bestScore = 0.0f;

    if (hasPrior) {
        priorD = prior->depth.at<float>(y, x);
        priorN = prior->normal.at<cv::Vec3f>(y, x);
        invTwoSigmaD2 = 1.0f / (2.0f * priorSigmaD * priorSigmaD);
        invTwoSigmaA2 = 1.0f / (2.0f * priorSigmaA * priorSigmaA);
    }

    auto restrictedScore = [&](float cost, float d, const cv::Vec3f &n) -> float {
        const float dd = d - priorD;
        float dot = priorN[0] * n[0] + priorN[1] * n[1] + priorN[2] * n[2];
        dot = std::max(-1.0f, std::min(1.0f, dot));
        const float da = std::acos(dot);
        const float priorTerm = priorGamma + std::exp(-dd * dd * invTwoSigmaD2) *
                                                 std::exp(-da * da * invTwoSigmaA2);
        return std::exp(-cost * cost / priorBeta) * priorTerm;
    };

    auto accept = [&](float cost, float d, const cv::Vec3f &n) {
        if (!hasPrior) {
            if (cost < bestCost) {
                bestCost = cost;
                bestDepth = d;
                bestNormal = n;
            }
            return;
        }
        const float s = restrictedScore(cost, d, n);
        if (s > bestScore) {
            bestScore = s;
            bestCost = cost;
            bestDepth = d;
            bestNormal = n;
        }
    };

    if (hasPrior) {
        bestScore = restrictedScore(bestCost, bestDepth, bestNormal);
    }

    struct NbSpec {
        int nx, ny;
    };
    NbSpec neighbors[2] = {{x + (propagateDown ? -1 : 1), y},
                           {x, y + (propagateDown ? -1 : 1)}};

    for (const auto &nb : neighbors) {
        if (nb.nx < 0 || nb.nx >= cols || nb.ny < 0 || nb.ny >= rows)
            continue;

        float nbDepth = depthMap.at<float>(nb.ny, nb.nx);
        cv::Vec3f nbNormal = normalMap.at<cv::Vec3f>(nb.ny, nb.nx);
        if (nbDepth <= 0)
            continue;

        double nKinv_nb = nbNormal[0] * (nb.nx - cx) / fx +
                          nbNormal[1] * (nb.ny - cy) / fy + nbNormal[2];
        if (std::abs(nKinv_nb) < 1e-9)
            continue;
        double planeD = double(nbDepth) * nKinv_nb;

        double nKinv_curr =
            nbNormal[0] * (x - cx) / fx + nbNormal[1] * (y - cy) / fy + nbNormal[2];
        if (std::abs(nKinv_curr) < 1e-9)
            continue;
        float newDepth = float(planeD / nKinv_curr);

        if (newDepth < depthMin || newDepth > depthMax)
            continue;

        float cost = costFn(x, y, newDepth, nbNormal[0], nbNormal[1], nbNormal[2]);
        accept(cost, newDepth, nbNormal);
    }

    if (hasPrior) {
        const float cost = costFn(x, y, priorD, priorN[0], priorN[1], priorN[2]);
        accept(cost, priorD, priorN);

        float d = priorD + 3.0f * priorSigmaD * genRandomSigned();
        d = std::max(float(depthMin), std::min(float(depthMax), d));
        cv::Vec3f n = priorN + cv::Vec3f(priorSigmaA * genRandomSigned(),
                                         priorSigmaA * genRandomSigned(),
                                         priorSigmaA * genRandomSigned());
        const float nlen = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (nlen > 1e-6f) {
            n /= nlen;
            if (n[2] <= 0.1f) {
                n[2] = 0.1f;
                n /= std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            }
            const float cost2 = costFn(x, y, d, n[0], n[1], n[2]);
            accept(cost2, d, n);
        }
    }

    {
        float dispUpdate = initialDispUpdate;
        float normUpdate = initialNormUpdate;

        while (dispUpdate > refineStopThres) {
            float deltaD = dispUpdate * genRandomSigned();
            float newDepth = bestDepth + deltaD;
            if (newDepth < depthMin || newDepth > depthMax) {
                dispUpdate *= 0.5f;
                normUpdate *= 0.5f;
                continue;
            }

            cv::Vec3f deltaN(normUpdate * genRandomSigned(), normUpdate * genRandomSigned(),
                             normUpdate * genRandomSigned());
            cv::Vec3f newNormal = bestNormal + deltaN;
            float nlen = std::sqrt(newNormal[0] * newNormal[0] + newNormal[1] * newNormal[1] +
                                   newNormal[2] * newNormal[2]);
            if (nlen < 1e-6f) {
                dispUpdate *= 0.5f;
                normUpdate *= 0.5f;
                continue;
            }
            newNormal /= nlen;

            if (newNormal[2] <= 0.1f) {
                dispUpdate *= 0.5f;
                normUpdate *= 0.5f;
                continue;
            }

            float cost =
                costFn(x, y, newDepth, newNormal[0], newNormal[1], newNormal[2]);
            accept(cost, newDepth, newNormal);

            dispUpdate *= 0.5f;
            normUpdate *= 0.5f;
        }
    }

    depthMap.at<float>(y, x) = bestDepth;
    normalMap.at<cv::Vec3f>(y, x) = bestNormal;
    confidence.at<float>(y, x) = bestCost;
}

void runCheckerboardStrips(
    sfm::ThreadPool &pool, int rows, int cols, int parity, bool propagateDown,
    const std::function<bool()> &shouldStop,
    const std::function<void(int x, int y, std::mt19937 &rng)> &visit)
{
    (void)propagateDown; // 保留签名与调用方语义对齐；条带内仅用 parity 筛选

    const int numThreads = static_cast<int>(pool.thread_count());
    const int stripHeight = std::max(1, rows / (numThreads * 4));

    std::vector<std::future<void>> futures;
    for (int yStart = 0; yStart < rows; yStart += stripHeight) {
        const int yEnd = std::min(yStart + stripHeight, rows);
        const int stripIdx = yStart / stripHeight;

        futures.push_back(pool.submit([yStart, yEnd, cols, parity, stripIdx, &shouldStop,
                                       &visit]() {
            std::mt19937 rng(12345u + static_cast<unsigned>(stripIdx));
            for (int y = yStart; y < yEnd; ++y) {
                if ((y & 15) == 0 && shouldStop && shouldStop())
                    return;
                for (int x = 0; x < cols; ++x) {
                    if (((x + y) & 1) != parity)
                        continue;
                    visit(x, y, rng);
                }
            }
        }));
    }
    for (auto &f : futures)
        f.wait();
}

int initDepthNormalMaps(cv::Mat &depthMap, cv::Mat &normalMap, double depthMin, double depthMax,
                        const cv::Mat &initDepth, const cv::Mat &initNormal)
{
    const int rows = depthMap.rows;
    const int cols = depthMap.cols;
    const bool hasInit = !initDepth.empty() && !initNormal.empty() &&
                         initDepth.size() == depthMap.size() &&
                         initNormal.size() == normalMap.size();

    std::mt19937 rng(42);
    auto genRandom01 = [&]() -> float {
        return static_cast<float>(rng()) / static_cast<float>(rng.max());
    };
    auto genRandomDepth = [&]() -> float {
        return float(depthMin) + float(depthMax - depthMin) * genRandom01();
    };
    auto genRandomNormal = [&]() -> cv::Vec3f {
        float nx, ny, nz;
        do {
            nx = 2.0f * genRandom01() - 1.0f;
            ny = 2.0f * genRandom01() - 1.0f;
            nz = 2.0f * genRandom01() - 1.0f;
        } while (nx * nx + ny * ny + nz * nz < 0.01f || nz <= 0.1f);
        float nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
        return cv::Vec3f(nx / nlen, ny / nlen, nz / nlen);
    };

    int reused = 0;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (hasInit) {
                const float d = initDepth.at<float>(y, x);
                if (d > 0.0f && std::isfinite(d)) {
                    depthMap.at<float>(y, x) = d;
                    normalMap.at<cv::Vec3f>(y, x) = initNormal.at<cv::Vec3f>(y, x);
                    ++reused;
                    continue;
                }
            }
            depthMap.at<float>(y, x) = genRandomDepth();
            normalMap.at<cv::Vec3f>(y, x) = genRandomNormal();
        }
    }
    return hasInit ? reused : 0;
}

bool runPatchMatchIterations(sfm::ThreadPool &pool, cv::Mat &depthMap, cv::Mat &normalMap,
                             cv::Mat &confidence, const PatchMatchRunConfig &cfg,
                             const PixelCostFn &costFn, const std::function<bool()> &shouldStop,
                             const std::function<void(const PlanarPriorEvent &)> &onPrior,
                             const std::function<void(const PatchMatchIterDiag &)> &onIterDone)
{
    const int rows = depthMap.rows;
    const int cols = depthMap.cols;
    const int numIters = cfg.numIters;
    const int warmupIters = cfg.wantPrior ? std::max(1, numIters / 2) : numIters;

    sfm::pmv2::PlanarPrior planarPrior;
    const sfm::pmv2::PlanarPrior *priorPtr = nullptr;

    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < numIters; ++iter) {
        if (shouldStop && shouldStop()) {
            depthMap.release();
            confidence.release();
            normalMap.release();
            return false;
        }

        if (cfg.wantPrior && iter == warmupIters && !priorPtr) {
            auto tp0 = std::chrono::steady_clock::now();
            planarPrior = sfm::pmv2::buildPlanarPrior(
                depthMap, confidence, cfg.fx, cfg.fy, cfg.cx, cfg.cy, cfg.planePriorStep,
                cfg.planePriorMaxCost, cfg.depthMin, cfg.depthMax);
            const int coverage = cv::countNonZero(planarPrior.mask);
            PlanarPriorEvent ev;
            ev.supportCount = planarPrior.supportCount;
            ev.coverage = coverage;
            ev.totalPixels = rows * cols;
            ev.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - tp0)
                               .count();
            ev.applied = coverage > 0;
            if (ev.applied)
                priorPtr = &planarPrior;
            if (onPrior)
                onPrior(ev);
        }

        const bool propagateDown = (iter % 2 == 0);
        for (int phase = 0; phase < 2; ++phase) {
            if (shouldStop && shouldStop())
                break;
            const int parity = checkerboardPhaseParity(propagateDown, phase);
            runCheckerboardStrips(
                pool, rows, cols, parity, propagateDown, shouldStop,
                [&](int x, int y, std::mt19937 &rng) {
                    processPixel(x, y, depthMap, normalMap, confidence, rows, cols, cfg.fx,
                                 cfg.fy, cfg.cx, cfg.cy, cfg.depthMin, cfg.depthMax,
                                 propagateDown, priorPtr, rng, costFn);
                });
        }

        if (shouldStop && shouldStop()) {
            depthMap.release();
            confidence.release();
            normalMap.release();
            return false;
        }

        if (onIterDone) {
            PatchMatchIterDiag diag;
            diag.iter = iter + 1;
            diag.numIters = numIters;
            diag.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
            diag.totalPixels = rows * cols;
            diag.minD = std::numeric_limits<float>::max();
            diag.maxD = std::numeric_limits<float>::lowest();
            double sumDepth = 0, sumCost = 0;
            for (int yy = 0; yy < rows; ++yy) {
                for (int xx = 0; xx < cols; ++xx) {
                    float d = depthMap.at<float>(yy, xx);
                    float c = confidence.at<float>(yy, xx);
                    if (d > 0 && std::isfinite(d)) {
                        ++diag.validCnt;
                        sumDepth += d;
                        diag.minD = std::min(diag.minD, d);
                        diag.maxD = std::max(diag.maxD, d);
                    }
                    sumCost += c;
                }
            }
            diag.meanDepth = diag.validCnt > 0 ? sumDepth / diag.validCnt : 0.0;
            diag.meanCost = sumCost / std::max(1, diag.totalPixels);
            onIterDone(diag);
        }
        t0 = std::chrono::steady_clock::now();
    }
    return true;
}

int filterDepthByCost(cv::Mat &depthMap, const cv::Mat &confidence, float costThreshold)
{
    int filteredCount = 0;
    for (int y = 0; y < depthMap.rows; ++y) {
        for (int x = 0; x < depthMap.cols; ++x) {
            if (confidence.at<float>(y, x) > costThreshold) {
                depthMap.at<float>(y, x) = 0.0f;
                ++filteredCount;
            }
        }
    }
    return filteredCount;
}

} // namespace dense
} // namespace sfm
