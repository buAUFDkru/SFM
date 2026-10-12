#include "algorithm/dense/PatchMatchFusion.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <mutex>
#include <utility>

namespace sfm {
namespace dense {
namespace {

void worldToCamera(const ViewGeom &g, const double Xw[3], double Xc[3])
{
    Xc[0] = g.R[0] * Xw[0] + g.R[1] * Xw[1] + g.R[2] * Xw[2] + g.t[0];
    Xc[1] = g.R[3] * Xw[0] + g.R[4] * Xw[1] + g.R[5] * Xw[2] + g.t[1];
    Xc[2] = g.R[6] * Xw[0] + g.R[7] * Xw[1] + g.R[8] * Xw[2] + g.t[2];
}

void cameraToWorld(const ViewGeom &g, const double Xc[3], double Xw[3])
{
    Xw[0] = g.Rt[0] * Xc[0] + g.Rt[1] * Xc[1] + g.Rt[2] * Xc[2] + g.C[0];
    Xw[1] = g.Rt[3] * Xc[0] + g.Rt[4] * Xc[1] + g.Rt[5] * Xc[2] + g.C[1];
    Xw[2] = g.Rt[6] * Xc[0] + g.Rt[7] * Xc[1] + g.Rt[8] * Xc[2] + g.C[2];
}

std::vector<std::vector<int>> buildNeighbors(const std::vector<ViewGeom> &geoms,
                                             const std::vector<StereoPairResult> *resultsOrNull,
                                             int kMaxNeighbors)
{
    const int numRes = static_cast<int>(geoms.size());
    std::vector<std::vector<int>> neighbors(static_cast<size_t>(numRes));
    for (int i = 0; i < numRes; ++i) {
        if (!geoms[static_cast<size_t>(i)].valid)
            continue;
        std::vector<std::pair<double, int>> cand;
        for (int j = 0; j < numRes; ++j) {
            if (i == j || !geoms[static_cast<size_t>(j)].valid)
                continue;
            if (resultsOrNull && resultsOrNull->at(static_cast<size_t>(j)).depth.empty())
                continue;
            const auto &gi = geoms[static_cast<size_t>(i)];
            const auto &gj = geoms[static_cast<size_t>(j)];
            const double dx = gj.C[0] - gi.C[0];
            const double dy = gj.C[1] - gi.C[1];
            const double dz = gj.C[2] - gi.C[2];
            cand.emplace_back(dx * dx + dy * dy + dz * dz, j);
        }
        std::sort(cand.begin(), cand.end());
        for (const auto &[dist, j] : cand) {
            (void)dist;
            neighbors[static_cast<size_t>(i)].push_back(j);
            if (static_cast<int>(neighbors[static_cast<size_t>(i)].size()) >= kMaxNeighbors)
                break;
        }
    }
    return neighbors;
}

} // namespace

ViewGeom makeViewGeom(const CameraPose &pose, const cv::Mat &fallbackK)
{
    ViewGeom g;
    const cv::Mat &Kmat = pose.K.empty() ? fallbackK : pose.K;
    if (Kmat.empty() || pose.R.empty() || pose.t.empty())
        return g;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            g.K[r * 3 + c] = Kmat.at<double>(r, c);
            g.R[r * 3 + c] = pose.R.at<double>(r, c);
            g.Rt[r * 3 + c] = pose.R.at<double>(c, r);
        }
        g.t[r] = pose.t.at<double>(r);
    }
    const cv::Mat C = -pose.R.t() * pose.t;
    for (int r = 0; r < 3; ++r)
        g.C[r] = C.at<double>(r);
    g.valid = true;
    return g;
}

GeomFilterStats filterGeometricConsistency(const DenseReconstructionInput &input,
                                           std::vector<StereoPairResult> &results,
                                           ThreadPool &pool,
                                           const GeomConsistencyOptions &opt)
{
    GeomFilterStats stats;
    const int numRes = static_cast<int>(results.size());
    if (numRes < 2) {
        stats.skippedTooFewViews = true;
        return stats;
    }

    std::vector<ViewGeom> geoms(static_cast<size_t>(numRes));
    for (int i = 0; i < numRes; ++i) {
        const int imgIdx = results[static_cast<size_t>(i)].imageId1;
        if (imgIdx < 0 || imgIdx >= static_cast<int>(input.cameraPoses.size()))
            continue;
        geoms[static_cast<size_t>(i)] =
            makeViewGeom(input.cameraPoses[static_cast<size_t>(imgIdx)], input.cameraMatrix);
    }

    constexpr int kMaxNeighbors = 8;
    auto neighbors = buildNeighbors(geoms, &results, kMaxNeighbors);

    const double maxReprojPx = opt.maxReprojPx;
    const int minConsistent = std::max(1, opt.minConsistent);

    std::vector<cv::Mat> keepMask(static_cast<size_t>(numRes));
    for (int i = 0; i < numRes; ++i) {
        if (results[static_cast<size_t>(i)].depth.empty())
            continue;
        keepMask[static_cast<size_t>(i)] =
            cv::Mat::zeros(results[static_cast<size_t>(i)].depth.size(), CV_8UC1);
    }

    for (int i = 0; i < numRes; ++i) {
        if (!geoms[static_cast<size_t>(i)].valid ||
            results[static_cast<size_t>(i)].depth.empty())
            continue;
        if (static_cast<int>(neighbors[static_cast<size_t>(i)].size()) < minConsistent) {
            keepMask[static_cast<size_t>(i)].setTo(1);
            continue;
        }

        const cv::Mat &Dref = results[static_cast<size_t>(i)].depth;
        const ViewGeom &gi = geoms[static_cast<size_t>(i)];
        const double fx = gi.K[0], fy = gi.K[4], cx = gi.K[2], cy = gi.K[5];

        std::vector<std::future<void>> futures;
        std::vector<std::atomic<long long>> validCnt(static_cast<size_t>(Dref.rows)),
            keptCnt(static_cast<size_t>(Dref.rows));
        for (int r = 0; r < Dref.rows; ++r) {
            validCnt[static_cast<size_t>(r)] = 0;
            keptCnt[static_cast<size_t>(r)] = 0;
        }

        const int stripHeight =
            std::max(1, Dref.rows / static_cast<int>(pool.thread_count() * 4));
        const auto &nbrs = neighbors[static_cast<size_t>(i)];

        for (int yStart = 0; yStart < Dref.rows; yStart += stripHeight) {
            const int yEnd = std::min(yStart + stripHeight, Dref.rows);
            futures.push_back(pool.submit([&, yStart, yEnd]() {
                for (int y = yStart; y < yEnd; ++y) {
                    const float *drow = Dref.ptr<float>(y);
                    uint8_t *mrow = keepMask[static_cast<size_t>(i)].ptr<uint8_t>(y);
                    long long rowValid = 0, rowKept = 0;

                    for (int x = 0; x < Dref.cols; ++x) {
                        const float dRef = drow[x];
                        if (!(dRef > 0.0f) || !std::isfinite(dRef))
                            continue;
                        ++rowValid;

                        double XcRef[3] = {(x - cx) * dRef / fx, (y - cy) * dRef / fy,
                                           double(dRef)};
                        double Xw[3];
                        cameraToWorld(gi, XcRef, Xw);

                        int numConsistent = 0;
                        for (int j : nbrs) {
                            const cv::Mat &Dsrc = results[static_cast<size_t>(j)].depth;
                            const ViewGeom &gj = geoms[static_cast<size_t>(j)];

                            double XcSrc[3];
                            worldToCamera(gj, Xw, XcSrc);
                            if (XcSrc[2] <= 1e-6)
                                continue;
                            const double uj = gj.K[0] * XcSrc[0] / XcSrc[2] + gj.K[2];
                            const double vj = gj.K[4] * XcSrc[1] / XcSrc[2] + gj.K[5];
                            const int ujI = static_cast<int>(uj + 0.5);
                            const int vjI = static_cast<int>(vj + 0.5);
                            if (ujI < 0 || ujI >= Dsrc.cols || vjI < 0 || vjI >= Dsrc.rows)
                                continue;

                            const float dSrc = Dsrc.ptr<float>(vjI)[ujI];
                            if (!(dSrc > 0.0f) || !std::isfinite(dSrc))
                                continue;

                            double XcSrc2[3] = {(ujI - gj.K[2]) * dSrc / gj.K[0],
                                                (vjI - gj.K[5]) * dSrc / gj.K[4], double(dSrc)};
                            double Xw2[3];
                            cameraToWorld(gj, XcSrc2, Xw2);
                            double XcBack[3];
                            worldToCamera(gi, Xw2, XcBack);
                            if (XcBack[2] <= 1e-6)
                                continue;
                            const double ub = gi.K[0] * XcBack[0] / XcBack[2] + gi.K[2];
                            const double vb = gi.K[4] * XcBack[1] / XcBack[2] + gi.K[5];

                            const double errPx = std::hypot(ub - x, vb - y);
                            if (errPx < maxReprojPx)
                                ++numConsistent;
                        }

                        if (numConsistent >= minConsistent) {
                            mrow[x] = 1;
                            ++rowKept;
                        }
                    }

                    validCnt[static_cast<size_t>(y)] = rowValid;
                    keptCnt[static_cast<size_t>(y)] = rowKept;
                }
            }));
        }
        for (auto &f : futures)
            f.wait();

        long long v = 0, k = 0;
        for (int r = 0; r < Dref.rows; ++r) {
            v += validCnt[static_cast<size_t>(r)].load();
            k += keptCnt[static_cast<size_t>(r)].load();
        }
        stats.totalValid += v;
        stats.totalKept += k;
    }

    for (int i = 0; i < numRes; ++i) {
        if (results[static_cast<size_t>(i)].depth.empty() ||
            keepMask[static_cast<size_t>(i)].empty())
            continue;
        float *d = results[static_cast<size_t>(i)].depth.ptr<float>();
        const uint8_t *m = keepMask[static_cast<size_t>(i)].ptr<uint8_t>();
        const size_t n = static_cast<size_t>(results[static_cast<size_t>(i)].depth.rows) *
                         results[static_cast<size_t>(i)].depth.cols;
        for (size_t p = 0; p < n; ++p) {
            if (!m[p])
                d[p] = 0.0f;
        }
    }
    return stats;
}

FusionStats fuseDepthMaps(const DenseReconstructionInput &input,
                          const std::vector<StereoPairResult> &results,
                          const std::vector<cv::Mat> &images, ThreadPool &pool,
                          const FusionOptions &opt, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    FusionStats stats;
    const int numRes = static_cast<int>(results.size());
    if (numRes == 0)
        return stats;

    auto t0 = std::chrono::steady_clock::now();

    std::vector<ViewGeom> geoms(static_cast<size_t>(numRes));
    for (int i = 0; i < numRes; ++i) {
        const int imgIdx = results[static_cast<size_t>(i)].imageId1;
        if (imgIdx < 0 || imgIdx >= static_cast<int>(input.cameraPoses.size()))
            continue;
        if (results[static_cast<size_t>(i)].depth.empty())
            continue;
        geoms[static_cast<size_t>(i)] =
            makeViewGeom(input.cameraPoses[static_cast<size_t>(imgIdx)], input.cameraMatrix);
    }

    constexpr int kMaxNeighbors = 8;
    auto neighbors = buildNeighbors(geoms, nullptr, kMaxNeighbors);

    std::vector<cv::Mat> visited(static_cast<size_t>(numRes));
    for (int i = 0; i < numRes; ++i) {
        if (geoms[static_cast<size_t>(i)].valid)
            visited[static_cast<size_t>(i)] = cv::Mat::zeros(
                results[static_cast<size_t>(i)].depth.rows,
                results[static_cast<size_t>(i)].depth.cols, CV_8UC1);
    }

    const double relErrMax = opt.relErrMax;
    const int minSupport = std::max(1, opt.minSupport);

    for (int i = 0; i < numRes; ++i) {
        if (!geoms[static_cast<size_t>(i)].valid)
            continue;
        const cv::Mat &Di = results[static_cast<size_t>(i)].depth;
        const cv::Mat &Ci = results[static_cast<size_t>(i)].confidence;
        const cv::Mat &Ii =
            (i < static_cast<int>(images.size())) ? images[static_cast<size_t>(i)] : cv::Mat();
        const ViewGeom &gi = geoms[static_cast<size_t>(i)];
        const int rows = Di.rows, cols = Di.cols;
        const auto &nbrs = neighbors[static_cast<size_t>(i)];

        size_t stripEmitted = 0, stripConsumed = 0, stripLowSup = 0;
        std::mutex m;
        std::vector<std::future<void>> futures;
        const int stripHeight = std::max(1, rows / static_cast<int>(pool.thread_count() * 4));

        for (int yStart = 0; yStart < rows; yStart += stripHeight) {
            const int yEnd = std::min(yStart + stripHeight, rows);
            futures.push_back(pool.submit([&, yStart, yEnd]() {
                size_t emitted = 0, consumed = 0, lowSup = 0;
                std::vector<pcl::PointXYZRGB> stripPts;
                stripPts.reserve(static_cast<size_t>((yEnd - yStart) * cols / 4));

                for (int y = yStart; y < yEnd; ++y) {
                    const float *dRow = Di.ptr<float>(y);
                    const float *cRow = Ci.empty() ? nullptr : Ci.ptr<float>(y);
                    const uint8_t *vRow = visited[static_cast<size_t>(i)].ptr<uint8_t>(y);
                    for (int x = 0; x < cols; ++x) {
                        const float d = dRow[x];
                        if (!(d > 0.0f) || !std::isfinite(d))
                            continue;
                        if (vRow[x]) {
                            ++consumed;
                            continue;
                        }

                        const double xc = (x - gi.K[2]) * d / gi.K[0];
                        const double yc = (y - gi.K[5]) * d / gi.K[4];
                        double px = gi.Rt[0] * xc + gi.Rt[1] * yc + gi.Rt[2] * d + gi.C[0];
                        double py = gi.Rt[3] * xc + gi.Rt[4] * yc + gi.Rt[5] * d + gi.C[1];
                        double pz = gi.Rt[6] * xc + gi.Rt[7] * yc + gi.Rt[8] * d + gi.C[2];

                        double w = 1.0 / (1.0 + (cRow ? double(cRow[x]) : 0.0));
                        double sumW = w, sumX = w * px, sumY = w * py, sumZ = w * pz;
                        double sumR = 0, sumG = 0, sumB = 0;
                        if (!Ii.empty() && Ii.type() == CV_8UC3) {
                            const cv::Vec3b c0 = Ii.at<cv::Vec3b>(y, x);
                            sumB = w * c0[0];
                            sumG = w * c0[1];
                            sumR = w * c0[2];
                        }
                        int support = 1;

                        for (int j : nbrs) {
                            const ViewGeom &gj = geoms[static_cast<size_t>(j)];
                            const cv::Mat &Dj = results[static_cast<size_t>(j)].depth;
                            const double Xj =
                                gj.R[0] * px + gj.R[1] * py + gj.R[2] * pz + gj.t[0];
                            const double Yj =
                                gj.R[3] * px + gj.R[4] * py + gj.R[5] * pz + gj.t[1];
                            const double Zj =
                                gj.R[6] * px + gj.R[7] * py + gj.R[8] * pz + gj.t[2];
                            if (Zj <= 0.0)
                                continue;
                            const double u = gj.K[0] * Xj / Zj + gj.K[2];
                            const double v = gj.K[4] * Yj / Zj + gj.K[5];
                            const int uj = static_cast<int>(u + 0.5);
                            const int vj = static_cast<int>(v + 0.5);
                            if (uj < 0 || uj >= Dj.cols || vj < 0 || vj >= Dj.rows)
                                continue;
                            const float dj = Dj.at<float>(vj, uj);
                            if (!(dj > 0.0f) || !std::isfinite(dj))
                                continue;
                            if (std::abs(double(dj) - Zj) / double(dj) > relErrMax)
                                continue;

                            const double xj = (u - gj.K[2]) * dj / gj.K[0];
                            const double yj = (v - gj.K[5]) * dj / gj.K[4];
                            const double qx =
                                gj.Rt[0] * xj + gj.Rt[1] * yj + gj.Rt[2] * dj + gj.C[0];
                            const double qy =
                                gj.Rt[3] * xj + gj.Rt[4] * yj + gj.Rt[5] * dj + gj.C[1];
                            const double qz =
                                gj.Rt[6] * xj + gj.Rt[7] * yj + gj.Rt[8] * dj + gj.C[2];

                            const cv::Mat &Cfj = results[static_cast<size_t>(j)].confidence;
                            const float cfj = Cfj.empty() ? 0.0f : Cfj.at<float>(vj, uj);
                            const double wj = 1.0 / (1.0 + double(cfj));
                            sumW += wj;
                            sumX += wj * qx;
                            sumY += wj * qy;
                            sumZ += wj * qz;
                            const cv::Mat &Ij = (j < static_cast<int>(images.size()))
                                                    ? images[static_cast<size_t>(j)]
                                                    : cv::Mat();
                            if (!Ij.empty() && Ij.type() == CV_8UC3) {
                                const cv::Vec3b cj = Ij.at<cv::Vec3b>(vj, uj);
                                sumB += wj * cj[0];
                                sumG += wj * cj[1];
                                sumR += wj * cj[2];
                            }
                            ++support;
                            visited[static_cast<size_t>(j)].at<uint8_t>(vj, uj) = 1;
                        }

                        if (support < minSupport) {
                            ++lowSup;
                            continue;
                        }

                        pcl::PointXYZRGB pt;
                        pt.x = float(sumX / sumW);
                        pt.y = float(sumY / sumW);
                        pt.z = float(sumZ / sumW);
                        pt.r = uint8_t(std::min(255.0, sumR / sumW + 0.5));
                        pt.g = uint8_t(std::min(255.0, sumG / sumW + 0.5));
                        pt.b = uint8_t(std::min(255.0, sumB / sumW + 0.5));
                        if (pt.r + pt.g + pt.b < 15)
                            continue;
                        stripPts.push_back(pt);
                        ++emitted;
                    }
                }

                std::lock_guard<std::mutex> lk(m);
                cloud.points.insert(cloud.points.end(), stripPts.begin(), stripPts.end());
                stripEmitted += emitted;
                stripConsumed += consumed;
                stripLowSup += lowSup;
            }));
        }
        for (auto &f : futures)
            f.wait();
        stats.numEmitted += stripEmitted;
        stats.numConsumed += stripConsumed;
        stats.numLowSupport += stripLowSup;
    }

    cloud.width = static_cast<uint32_t>(cloud.points.size());
    cloud.height = 1;
    cloud.is_dense = false;
    stats.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return stats;
}

} // namespace dense
} // namespace sfm
