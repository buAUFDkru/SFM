#include "algorithm/dense/PatchMatchRefine.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace sfm {
namespace dense {

void jointBilateralUpsample(const cv::Mat &depthLow, const cv::Mat &normalLow,
                            const cv::Mat &grayHigh, cv::Mat &depthHigh, cv::Mat &normalHigh)
{
    const int colsH = grayHigh.cols;
    const int rowsH = grayHigh.rows;
    const int colsL = depthLow.cols;
    const int rowsL = depthLow.rows;

    const double scaleX = static_cast<double>(colsH) / colsL;
    const double scaleY = static_cast<double>(rowsH) / rowsL;

    depthHigh.create(rowsH, colsH, CV_32FC1);
    normalHigh.create(rowsH, colsH, CV_32FC3);

    const double inv2s2 = 1.0 / (2.0 * 1.0 * 1.0);
    const double inv2r2 = 1.0 / (2.0 * 20.0 * 20.0);

    for (int y = 0; y < rowsH; ++y) {
        const uint8_t *gRow = grayHigh.ptr<uint8_t>(y);
        float *dRow = depthHigh.ptr<float>(y);
        cv::Vec3f *nRow = normalHigh.ptr<cv::Vec3f>(y);

        for (int x = 0; x < colsH; ++x) {
            const double plx = (x + 0.5) / scaleX - 0.5;
            const double ply = (y + 0.5) / scaleY - 0.5;
            const int qx0 = static_cast<int>(std::lround(plx));
            const int qy0 = static_cast<int>(std::lround(ply));

            const double Iref = static_cast<double>(gRow[x]);

            double sumW = 0.0, sumD = 0.0;
            double sumNx = 0.0, sumNy = 0.0, sumNz = 0.0;

            for (int dy = -1; dy <= 1; ++dy) {
                const int qy = qy0 + dy;
                if (qy < 0 || qy >= rowsL)
                    continue;
                const float *dLRow = depthLow.ptr<float>(qy);
                const cv::Vec3f *nLRow = normalLow.ptr<cv::Vec3f>(qy);

                for (int dx = -1; dx <= 1; ++dx) {
                    const int qx = qx0 + dx;
                    if (qx < 0 || qx >= colsL)
                        continue;

                    const float dL = dLRow[qx];
                    if (!(dL > 0.0f) || !std::isfinite(dL))
                        continue;

                    const double hx = (qx + 0.5) * scaleX - 0.5;
                    const double hy = (qy + 0.5) * scaleY - 0.5;
                    const int hxI = std::min(std::max(static_cast<int>(hx + 0.5), 0), colsH - 1);
                    const int hyI = std::min(std::max(static_cast<int>(hy + 0.5), 0), rowsH - 1);
                    const double Iq = static_cast<double>(grayHigh.at<uint8_t>(hyI, hxI));

                    const double dsx = plx - qx;
                    const double dsy = ply - qy;
                    const double dc = Iref - Iq;
                    const double w =
                        std::exp(-(dsx * dsx + dsy * dsy) * inv2s2 - dc * dc * inv2r2);

                    sumW += w;
                    sumD += w * dL;
                    const cv::Vec3f &nL = nLRow[qx];
                    sumNx += w * nL[0];
                    sumNy += w * nL[1];
                    sumNz += w * nL[2];
                }
            }

            if (sumW > 1e-9) {
                dRow[x] = static_cast<float>(sumD / sumW);
                cv::Vec3f n(static_cast<float>(sumNx), static_cast<float>(sumNy),
                            static_cast<float>(sumNz));
                const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                nRow[x] = (len > 1e-6f) ? (n / len) : cv::Vec3f(0.0f, 0.0f, 1.0f);
            } else {
                dRow[x] = 0.0f;
                nRow[x] = cv::Vec3f(0.0f, 0.0f, 1.0f);
            }
        }
    }
}

int decideNumScales(int cols, int rows, int maxScales, bool multiscaleEnabled)
{
    int numScales = 1;
    if (!multiscaleEnabled)
        return numScales;
    int w = cols, h = rows;
    while (numScales < maxScales && std::min(w, h) / 2 >= 128) {
        w /= 2;
        h /= 2;
        ++numScales;
    }
    return numScales;
}

std::vector<PatchMatchLevelCtx> buildPyramid(const PatchMatchLevelCtx &finest, int numScales)
{
    std::vector<PatchMatchLevelCtx> pyr(static_cast<size_t>(std::max(1, numScales)));
    pyr[0] = finest;
    for (int lv = 1; lv < numScales; ++lv) {
        const PatchMatchLevelCtx &prev = pyr[static_cast<size_t>(lv - 1)];
        PatchMatchLevelCtx &c = pyr[static_cast<size_t>(lv)];
        c = prev;

        cv::pyrDown(prev.refGray, c.refGray);
        c.rows = c.refGray.rows;
        c.cols = c.refGray.cols;
        const double sx = static_cast<double>(c.cols) / prev.cols;
        const double sy = static_cast<double>(c.rows) / prev.rows;

        c.fx = prev.fx * sx;
        c.fy = prev.fy * sy;
        c.cx = (prev.cx + 0.5) * sx - 0.5;
        c.cy = (prev.cy + 0.5) * sy - 0.5;
        c.K_ref = (cv::Mat_<double>(3, 3) << c.fx, 0, c.cx, 0, c.fy, c.cy, 0, 0, 1);

        for (size_t s = 0; s < c.sources.size(); ++s) {
            cv::Mat g;
            cv::pyrDown(prev.sources[s].gray, g);
            c.sources[s].gray = g;
            c.sources[s].K[0] = prev.sources[s].K[0] * sx;
            c.sources[s].K[2] = (prev.sources[s].K[2] + 0.5) * sx - 0.5;
            c.sources[s].K[4] = prev.sources[s].K[4] * sy;
            c.sources[s].K[5] = (prev.sources[s].K[5] + 0.5) * sy - 0.5;
        }
    }
    return pyr;
}

int itersForPyramidLevel(int lv, int numScales, int coarseIters, int refineIters)
{
    if (lv == numScales - 1)
        return coarseIters;
    if (lv == 0)
        return refineIters;
    return static_cast<int>(std::lround(
        refineIters + double(coarseIters - refineIters) * lv / (numScales - 1)));
}

bool runCoarseToFine(
    std::vector<PatchMatchLevelCtx> &pyr, int coarseIters, int refineIters,
    const std::function<bool()> &shouldStop,
    const std::function<void(int step, int numScales, int cols, int rows, int iters)> &onLevel,
    const SingleScaleRunner &runLevel, cv::Mat &depthOut, cv::Mat &confidenceOut,
    cv::Mat &normalOut)
{
    const int numScales = static_cast<int>(pyr.size());
    if (numScales <= 0 || !runLevel)
        return false;

    cv::Mat depth, confidence, normalMap;

    for (int lv = numScales - 1; lv >= 0; --lv) {
        if (shouldStop && shouldStop()) {
            depth.release();
            depthOut.release();
            return false;
        }
        const int iters = itersForPyramidLevel(lv, numScales, coarseIters, refineIters);
        PatchMatchLevelCtx &level = pyr[static_cast<size_t>(lv)];
        if (onLevel)
            onLevel(numScales - lv, numScales, level.cols, level.rows, iters);

        if (depth.empty()) {
            depth = runLevel(level, confidence, normalMap, cv::Mat(), cv::Mat(), iters);
        } else {
            cv::Mat upDepth, upNormal;
            jointBilateralUpsample(depth, normalMap, level.refGray, upDepth, upNormal);
            depth = runLevel(level, confidence, normalMap, upDepth, upNormal, iters);
        }
        if (depth.empty()) {
            depthOut.release();
            return false;
        }
    }

    depthOut = depth;
    confidenceOut = confidence;
    normalOut = normalMap;
    return true;
}

} // namespace dense
} // namespace sfm
