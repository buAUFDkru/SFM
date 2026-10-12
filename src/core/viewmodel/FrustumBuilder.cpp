#include "FrustumBuilder.h"

#include "frame/Coordinate.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sfm {
namespace view {
namespace {

void appendSegment(std::vector<float> &lines,
                   float ax, float ay, float az,
                   float bx, float by, float bz)
{
    lines.push_back(ax);
    lines.push_back(ay);
    lines.push_back(az);
    lines.push_back(bx);
    lines.push_back(by);
    lines.push_back(bz);
}

} // namespace

CameraFrustumView buildCameraFrusta(const std::vector<CameraPose> &poses,
                                    float depth,
                                    const std::vector<std::string> &imageFiles)
{
    CameraFrustumView geom;
    if (depth <= 0.0f)
        return geom;

    for (const auto &pose : poses) {
        if (!pose.registered || pose.R.empty() || pose.t.empty())
            continue;

        const double hw = 0.5 * static_cast<double>(depth);
        const double hh = 0.5 * static_cast<double>(depth) *
                          (pose.imageWidth > 0 && pose.imageHeight > 0
                               ? double(pose.imageHeight) / pose.imageWidth
                               : 0.75);
        const double camCorners[4][3] = {
            {-hw, -hh, depth}, {hw, -hh, depth}, {hw, hh, depth}, {-hw, hh, depth}};

        const cv::Mat Rt = pose.R.t();
        const cv::Mat C = -Rt * pose.t;

        const auto center = sfm::frame::toDisplay(C);
        float cx[4], cy[4], cz[4];
        for (int i = 0; i < 4; ++i) {
            const cv::Mat v = Rt * (cv::Mat_<double>(3, 1) << camCorners[i][0],
                                    camCorners[i][1], camCorners[i][2]) + C;
            const auto d = sfm::frame::toDisplay(v);
            cx[i] = d.x;
            cy[i] = d.y;
            cz[i] = d.z;
            geom.quadCorners.push_back(d.x);
            geom.quadCorners.push_back(d.y);
            geom.quadCorners.push_back(d.z);
        }

        for (int i = 0; i < 4; ++i) {
            appendSegment(geom.lineSegments,
                          center.x, center.y, center.z,
                          cx[i], cy[i], cz[i]);
        }
        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) % 4;
            appendSegment(geom.lineSegments,
                          cx[i], cy[i], cz[i],
                          cx[j], cy[j], cz[j]);
        }

        if (pose.imageId >= 0 &&
            pose.imageId < static_cast<int>(imageFiles.size())) {
            geom.imagePaths.push_back(imageFiles[static_cast<size_t>(pose.imageId)]);
        } else {
            geom.imagePaths.emplace_back();
        }
    }
    return geom;
}

float estimateFrustumDepth(const PointCloudView &cloud,
                           float fraction,
                           float fallback)
{
    if (cloud.empty())
        return fallback;

    float mn[3] = {std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
    float mx[3] = {-std::numeric_limits<float>::max(),
                   -std::numeric_limits<float>::max(),
                   -std::numeric_limits<float>::max()};
    const size_t n = cloud.size();
    for (size_t i = 0; i < n; ++i) {
        const float x = cloud.positions[i * 3 + 0];
        const float y = cloud.positions[i * 3 + 1];
        const float z = cloud.positions[i * 3 + 2];
        mn[0] = std::min(mn[0], x);
        mn[1] = std::min(mn[1], y);
        mn[2] = std::min(mn[2], z);
        mx[0] = std::max(mx[0], x);
        mx[1] = std::max(mx[1], y);
        mx[2] = std::max(mx[2], z);
    }
    const float span = std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});
    const float depth = fraction * span;
    return depth > 0.0f ? depth : fallback;
}

} // namespace view
} // namespace sfm
