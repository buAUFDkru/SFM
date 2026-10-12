#include "algorithm/dense/PatchMatchShared.h"

#include "data/SfmTypes.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace sfm {
namespace dense {
namespace {

double angleBetweenDirections(const cv::Mat &R1, const cv::Mat &R2)
{
    cv::Mat z1 = (cv::Mat_<double>(3, 1) << R1.at<double>(2, 0), R1.at<double>(2, 1),
                  R1.at<double>(2, 2));
    cv::Mat z2 = (cv::Mat_<double>(3, 1) << R2.at<double>(2, 0), R2.at<double>(2, 1),
                  R2.at<double>(2, 2));
    double dot = z1.dot(z2);
    dot = std::max(-1.0, std::min(1.0, dot));
    return std::acos(dot) * 180.0 / CV_PI;
}

} // namespace

float sampleGrayBilinear(const cv::Mat &gray, float x, float y)
{
    if (x < 0 || y < 0 || x > gray.cols - 1 || y > gray.rows - 1)
        return -1.0f;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, gray.cols - 1);
    int y1 = std::min(y0 + 1, gray.rows - 1);
    float dx = x - x0;
    float dy = y - y0;

    const uint8_t *row0 = gray.ptr<uint8_t>(y0);
    const uint8_t *row1 = gray.ptr<uint8_t>(y1);
    float v00 = row0[x0];
    float v01 = row0[x1];
    float v10 = row1[x0];
    float v11 = row1[x1];

    return (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
           (1.0f - dx) * dy * v10 + dx * dy * v11;
}

std::vector<Problem> buildProblems(const DenseReconstructionInput &input)
{
    const int numImages = static_cast<int>(input.imageFiles.size());
    ViewGraph viewGraph = ViewGraph::buildFromPairs(input.matchPairs, numImages);

    std::vector<Problem> problems;
    problems.reserve(static_cast<size_t>(numImages));

    for (int refIdx = 0; refIdx < numImages; ++refIdx) {
        if (refIdx >= static_cast<int>(input.cameraPoses.size()))
            continue;
        if (!input.cameraPoses[static_cast<size_t>(refIdx)].registered)
            continue;

        Problem problem;
        problem.refImageIdx = refIdx;

        std::vector<std::pair<int, int>> candidates;
        for (int edgeIdx : viewGraph.adj[static_cast<size_t>(refIdx)]) {
            const auto &edge = viewGraph.edges[static_cast<size_t>(edgeIdx)];
            int nb = (edge.nodeA == refIdx) ? edge.nodeB : edge.nodeA;
            if (nb < 0 || nb >= numImages)
                continue;
            if (!input.cameraPoses[static_cast<size_t>(nb)].registered)
                continue;
            candidates.emplace_back(nb, edge.matchCount);
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        const auto &refPose = input.cameraPoses[static_cast<size_t>(refIdx)];
        for (const auto &[srcIdx, matchCount] : candidates) {
            if (problem.srcImageIdxs.size() >= 10)
                break;
            if (matchCount < 5)
                continue;
            const auto &srcPose = input.cameraPoses[static_cast<size_t>(srcIdx)];
            double angle = angleBetweenDirections(refPose.R, srcPose.R);
            if (angle > 60.0)
                continue;
            problem.srcImageIdxs.push_back(srcIdx);
        }

        if (!problem.srcImageIdxs.empty())
            problems.push_back(std::move(problem));
    }
    return problems;
}

std::vector<std::pair<double, double>>
estimateDepthRanges(const DenseReconstructionInput &input)
{
    const int numImages = static_cast<int>(input.imageFiles.size());
    std::vector<std::pair<double, double>> ranges(
        static_cast<size_t>(numImages), {-1.0, -1.0});
    std::vector<std::vector<double>> depthsPerImage(static_cast<size_t>(numImages));

    for (const auto &mp : input.matchPairs) {
        if (mp.imageId1 < 0 || mp.imageId1 >= numImages || mp.imageId2 < 0 ||
            mp.imageId2 >= numImages)
            continue;
        if (mp.points1.size() != mp.points2.size() || mp.points1.empty())
            continue;

        const auto &pose1 = input.cameraPoses[static_cast<size_t>(mp.imageId1)];
        const auto &pose2 = input.cameraPoses[static_cast<size_t>(mp.imageId2)];
        if (!pose1.registered || !pose2.registered)
            continue;

        const cv::Mat &K1 = pose1.K.empty() ? input.cameraMatrix : pose1.K;
        const cv::Mat &K2 = pose2.K.empty() ? input.cameraMatrix : pose2.K;
        if (K1.empty() || K2.empty())
            continue;

        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
        {
            cv::Mat KR1 = K1 * pose1.R;
            KR1.copyTo(P1(cv::Rect(0, 0, 3, 3)));
            cv::Mat Kt1 = K1 * pose1.t;
            Kt1.copyTo(P1(cv::Rect(3, 0, 1, 3)));
        }
        cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
        {
            cv::Mat KR2 = K2 * pose2.R;
            KR2.copyTo(P2(cv::Rect(0, 0, 3, 3)));
            cv::Mat Kt2 = K2 * pose2.t;
            Kt2.copyTo(P2(cv::Rect(3, 0, 1, 3)));
        }

        const size_t n = std::min(mp.points1.size(), size_t(500));
        for (size_t i = 0; i < n; ++i) {
            const cv::Point2d &p1 = mp.points1[i];
            const cv::Point2d &p2 = mp.points2[i];

            cv::Mat A(4, 4, CV_64F);
            cv::Mat(p1.x * P1.row(2) - P1.row(0)).reshape(1, 1).copyTo(A.row(0));
            cv::Mat(p1.y * P1.row(2) - P1.row(1)).reshape(1, 1).copyTo(A.row(1));
            cv::Mat(p2.x * P2.row(2) - P2.row(0)).reshape(1, 1).copyTo(A.row(2));
            cv::Mat(p2.y * P2.row(2) - P2.row(1)).reshape(1, 1).copyTo(A.row(3));

            cv::Mat S, U, Vt;
            cv::SVD::compute(A, S, U, Vt);
            cv::Mat Xh = Vt.row(3).t();
            if (std::abs(Xh.at<double>(3)) < 1e-9)
                continue;
            cv::Mat X = Xh / Xh.at<double>(3);

            cv::Mat Xw = X(cv::Rect(0, 0, 1, 3)).clone();
            cv::Mat Xc1 = pose1.R * Xw + pose1.t;
            double depth1 = Xc1.at<double>(2);
            if (depth1 > 0 && std::isfinite(depth1))
                depthsPerImage[static_cast<size_t>(mp.imageId1)].push_back(depth1);

            cv::Mat Xc2 = pose2.R * Xw + pose2.t;
            double depth2 = Xc2.at<double>(2);
            if (depth2 > 0 && std::isfinite(depth2))
                depthsPerImage[static_cast<size_t>(mp.imageId2)].push_back(depth2);
        }
    }

    for (int i = 0; i < numImages; ++i) {
        auto &depths = depthsPerImage[static_cast<size_t>(i)];
        if (depths.size() < 5)
            continue;
        std::sort(depths.begin(), depths.end());
        size_t p5 = depths.size() * 5 / 100;
        size_t p95 = depths.size() * 95 / 100;
        if (p95 <= p5)
            p95 = p5 + 1;
        double dMin = depths[p5];
        double dMax = depths[p95];
        double margin = (dMax - dMin) * 0.2;
        ranges[static_cast<size_t>(i)] = {std::max(0.01, dMin - margin), dMax + margin};
    }

    const double MAX_DEPTH_RATIO = 30.0;
    for (int i = 0; i < numImages; ++i) {
        auto &r = ranges[static_cast<size_t>(i)];
        if (r.first <= 0)
            continue;
        double ratio = r.second / r.first;
        auto &depths = depthsPerImage[static_cast<size_t>(i)];
        if (ratio > MAX_DEPTH_RATIO && !depths.empty()) {
            double median = depths[depths.size() / 2];
            r.first = std::max(0.01, median / 3.0);
            r.second = median * 3.0;
        }
    }

    double globalMin = std::numeric_limits<double>::max();
    double globalMax = 0.0;
    for (const auto &r : ranges) {
        if (r.first > 0) {
            globalMin = std::min(globalMin, r.first);
            globalMax = std::max(globalMax, r.second);
        }
    }
    if (globalMin > globalMax) {
        globalMin = 0.1;
        globalMax = 100.0;
    }
    for (auto &r : ranges) {
        if (r.first < 0)
            r = {globalMin, globalMax};
    }
    return ranges;
}

cv::Mat fuseSourceDepthMaps(const cv::Mat &refImage,
                            const std::vector<cv::Mat> &srcDepths,
                            const std::vector<cv::Mat> &srcConfidences)
{
    (void)refImage;
    (void)srcDepths;
    (void)srcConfidences;
    return cv::Mat();
}

void backprojectDepth(const DenseReconstructionInput &input,
                      int refIdx,
                      const cv::Mat &depth,
                      const cv::Mat &refImage,
                      pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    if (refIdx < 0 || refIdx >= static_cast<int>(input.cameraPoses.size()))
        return;

    const auto &pose = input.cameraPoses[static_cast<size_t>(refIdx)];
    const cv::Mat K = pose.K.empty() ? input.cameraMatrix.clone() : pose.K.clone();
    if (K.empty())
        return;

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    const cv::Mat R_world = pose.R.t();
    const cv::Mat t_world = -pose.R.t() * pose.t;

    constexpr int step = 2;

    for (int y = 0; y < depth.rows; y += step) {
        for (int x = 0; x < depth.cols; x += step) {
            float d = depth.at<float>(y, x);
            if (d <= 0.0f || !std::isfinite(d))
                continue;

            double Xc = (x - cx) * d / fx;
            double Yc = (y - cy) * d / fy;
            double Zc = d;

            double Xw = R_world.at<double>(0, 0) * Xc + R_world.at<double>(0, 1) * Yc +
                        R_world.at<double>(0, 2) * Zc + t_world.at<double>(0);
            double Yw = R_world.at<double>(1, 0) * Xc + R_world.at<double>(1, 1) * Yc +
                        R_world.at<double>(1, 2) * Zc + t_world.at<double>(1);
            double Zw = R_world.at<double>(2, 0) * Xc + R_world.at<double>(2, 1) * Yc +
                        R_world.at<double>(2, 2) * Zc + t_world.at<double>(2);

            uint8_t r = 128, g = 128, b = 128;
            if (!refImage.empty() && refImage.type() == CV_8UC3) {
                cv::Vec3b c = refImage.at<cv::Vec3b>(y, x);
                b = c[0];
                g = c[1];
                r = c[2];
            }

            if (r + g + b < 15)
                continue;

            pcl::PointXYZRGB pt;
            pt.x = static_cast<float>(Xw);
            pt.y = static_cast<float>(Yw);
            pt.z = static_cast<float>(Zw);
            pt.r = r;
            pt.g = g;
            pt.b = b;
            cloud.push_back(pt);
        }
    }
}

double computeWorkScale(int width, int height, int maxImageSize, MaxImageSizePolicy policy)
{
    const int maxDim = std::max(width, height);
    int target = maxImageSize;
    if (policy == MaxImageSizePolicy::CapAt1000WhenNonPositive) {
        target = maxImageSize > 0 ? maxImageSize : 1000;
    } else if (maxImageSize <= 0) {
        return 1.0;
    }
    if (maxDim > target)
        return static_cast<double>(target) / maxDim;
    return 1.0;
}

cv::Mat scaleIntrinsics(const cv::Mat &K, double scale)
{
    cv::Mat out = K.clone();
    if (out.empty() || scale == 1.0)
        return out;
    out.at<double>(0, 0) *= scale;
    out.at<double>(0, 2) *= scale;
    out.at<double>(1, 1) *= scale;
    out.at<double>(1, 2) *= scale;
    return out;
}

PreparedRefView prepareReferenceView(const DenseReconstructionInput &input, int refIdx,
                                     const std::pair<double, double> &depthRange,
                                     int maxImageSize, MaxImageSizePolicy policy,
                                     const LoadImageFn &loadImage)
{
    PreparedRefView out;
    out.refImageIdx = refIdx;
    if (refIdx < 0 || refIdx >= input.imageFiles.size() ||
        refIdx >= static_cast<int>(input.cameraPoses.size()) || !loadImage) {
        return out;
    }

    out.refColor = loadImage(refIdx);
    if (out.refColor.empty())
        return out;

    const auto &refPose = input.cameraPoses[static_cast<size_t>(refIdx)];
    out.K_full = refPose.K.empty() ? input.cameraMatrix.clone() : refPose.K.clone();
    if (out.K_full.empty())
        return out;

    out.scale = computeWorkScale(out.refColor.cols, out.refColor.rows, maxImageSize, policy);
    cv::Mat refScaled;
    if (out.scale != 1.0)
        cv::resize(out.refColor, refScaled, cv::Size(), out.scale, out.scale, cv::INTER_AREA);
    else
        refScaled = out.refColor;

    cv::cvtColor(refScaled, out.refGray, cv::COLOR_BGR2GRAY);
    out.K_work = scaleIntrinsics(out.K_full, out.scale);
    out.depthMin = depthRange.first > 0 ? depthRange.first : 0.1;
    out.depthMax = depthRange.second > 0 ? depthRange.second : 100.0;
    out.ok = true;
    return out;
}

namespace {

cv::Mat loadScaledGray(const LoadImageFn &loadImage, int imgIdx, double scale)
{
    cv::Mat color = loadImage(imgIdx);
    if (color.empty())
        return {};
    cv::Mat scaled;
    if (scale != 1.0)
        cv::resize(color, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        scaled = color;
    cv::Mat gray;
    cv::cvtColor(scaled, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat resolveScaledK(const CameraPose &pose, const cv::Mat &fallbackK, double scale)
{
    cv::Mat K = pose.K.empty() ? fallbackK.clone() : pose.K.clone();
    return scaleIntrinsics(K, scale);
}

} // namespace

std::vector<SourceView> buildMatSourceViews(const DenseReconstructionInput &input, int refIdx,
                                           const std::vector<int> &srcImageIdxs, double scale,
                                           const LoadImageFn &loadImage)
{
    std::vector<SourceView> sources;
    if (refIdx < 0 || refIdx >= static_cast<int>(input.cameraPoses.size()) || !loadImage)
        return sources;

    const auto &refPose = input.cameraPoses[static_cast<size_t>(refIdx)];
    if (refPose.R.empty() || refPose.t.empty())
        return sources;
    const cv::Mat R_ref_inv = refPose.R.t();

    for (int srcIdx : srcImageIdxs) {
        if (srcIdx < 0 || srcIdx >= input.imageFiles.size() ||
            srcIdx >= static_cast<int>(input.cameraPoses.size()))
            continue;

        SourceView src;
        src.gray = loadScaledGray(loadImage, srcIdx, scale);
        if (src.gray.empty())
            continue;

        const auto &srcPose = input.cameraPoses[static_cast<size_t>(srcIdx)];
        if (srcPose.R.empty() || srcPose.t.empty())
            continue;
        src.K_src = resolveScaledK(srcPose, input.cameraMatrix, scale);
        src.R_rel = srcPose.R * R_ref_inv;
        src.T_rel = srcPose.t - srcPose.R * R_ref_inv * refPose.t;
        src.imageIdx = srcIdx;
        sources.push_back(std::move(src));
    }
    return sources;
}

std::vector<SourceViewPod> buildPodSourceViews(const DenseReconstructionInput &input, int refIdx,
                                              const std::vector<int> &srcImageIdxs, double scale,
                                              const LoadImageFn &loadImage)
{
    std::vector<SourceViewPod> sources;
    if (refIdx < 0 || refIdx >= static_cast<int>(input.cameraPoses.size()) || !loadImage)
        return sources;

    const auto &refPose = input.cameraPoses[static_cast<size_t>(refIdx)];
    if (refPose.R.empty() || refPose.t.empty())
        return sources;
    const cv::Mat R_ref_inv = refPose.R.t();

    for (int srcIdx : srcImageIdxs) {
        if (srcIdx < 0 || srcIdx >= input.imageFiles.size() ||
            srcIdx >= static_cast<int>(input.cameraPoses.size()))
            continue;

        SourceViewPod src;
        src.gray = loadScaledGray(loadImage, srcIdx, scale);
        if (src.gray.empty())
            continue;

        const auto &srcPose = input.cameraPoses[static_cast<size_t>(srcIdx)];
        if (srcPose.R.empty() || srcPose.t.empty())
            continue;
        cv::Mat K_src = resolveScaledK(srcPose, input.cameraMatrix, scale);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                src.K[i * 3 + j] = K_src.at<double>(i, j);
        }
        cv::Mat R_rel = srcPose.R * R_ref_inv;
        cv::Mat T_rel = srcPose.t - srcPose.R * R_ref_inv * refPose.t;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                src.R[i * 3 + j] = R_rel.at<double>(i, j);
            src.T[i] = T_rel.at<double>(i);
        }
        src.imageIdx = srcIdx;
        sources.push_back(std::move(src));
    }
    return sources;
}

int countValidDepthPixels(const cv::Mat &depth)
{
    if (depth.empty())
        return 0;
    int valid = 0;
    for (int y = 0; y < depth.rows; ++y) {
        const float *row = depth.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x) {
            if (row[x] > 0.0f && std::isfinite(row[x]))
                ++valid;
        }
    }
    return valid;
}

void upscaleToFullResolution(cv::Mat &depth, cv::Mat &confidence, const cv::Size &fullSize,
                             bool alsoConfidence)
{
    if (depth.empty() || depth.size() == fullSize)
        return;
    cv::resize(depth, depth, fullSize, 0, 0, cv::INTER_NEAREST);
    if (alsoConfidence && !confidence.empty())
        cv::resize(confidence, confidence, fullSize, 0, 0, cv::INTER_NEAREST);
}

} // namespace dense
} // namespace sfm
