#include "IntrinsicsResolver.h"

#include "data/SfmTypes.h"

namespace sfm {
namespace {

cv::Mat estimateKFromFeatures(const std::vector<FeatureResult> &features)
{
    if (features.empty() || features[0].imageWidth < 1 || features[0].imageHeight < 1) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    return ::estimateK(features[0].imageWidth, features[0].imageHeight);
}

void appendKSummary(IntrinsicsResolveResult &r)
{
    if (r.K.empty() || r.K.rows != 3 || r.K.cols != 3)
        return;
    const QString tag = (r.source == IntrinsicsResolveResult::Source::External)
                            ? QStringLiteral("[内参矩阵] 使用外部提供的K")
                            : QStringLiteral("[内参矩阵] 使用自动估计的K");
    r.messages.append(QStringLiteral("%1: fx=%2 fy=%3 cx=%4 cy=%5")
                          .arg(tag)
                          .arg(r.K.at<double>(0, 0), 0, 'f', 2)
                          .arg(r.K.at<double>(1, 1), 0, 'f', 2)
                          .arg(r.K.at<double>(0, 2), 0, 'f', 2)
                          .arg(r.K.at<double>(1, 2), 0, 'f', 2));
}

} // namespace

IntrinsicsResolveResult IntrinsicsResolver::resolve(
    const cv::Mat &externalK,
    bool useExifFocal,
    const QStringList &imageFiles,
    const std::vector<FeatureResult> &features)
{
    IntrinsicsResolveResult r;

    if (useExifFocal) {
        r.exif = SfmExif::estimateFromExif(imageFiles, features);
    }

    if (!externalK.empty()) {
        r.K = externalK.clone();
        r.source = IntrinsicsResolveResult::Source::External;
    } else if (r.exif.valid) {
        r.K = r.exif.K.clone();
        r.source = IntrinsicsResolveResult::Source::Exif;
        r.messages.append(
            QStringLiteral("[EXIF焦距] 采用EXIF内参初值: f=%1px (35mm等效 %2mm, "
                           "%3/%4 张图含EXIF, 离散度 %5~%6px), 主点=(%7,%8)")
                .arg(r.exif.focalPx, 0, 'f', 1)
                .arg(r.exif.focal35Mm, 0, 'f', 1)
                .arg(r.exif.numWithExif).arg(r.exif.numImages)
                .arg(r.exif.focalPxMin, 0, 'f', 1)
                .arg(r.exif.focalPxMax, 0, 'f', 1)
                .arg(r.exif.K.at<double>(0, 2), 0, 'f', 1)
                .arg(r.exif.K.at<double>(1, 2), 0, 'f', 1));
    } else {
        if (useExifFocal && !r.exif.reason.isEmpty()) {
            r.messages.append(QStringLiteral("[EXIF焦距] 不可用（%1），回落到按图像尺寸估计 K")
                                  .arg(r.exif.reason));
        }
        r.K = estimateKFromFeatures(features);
        r.source = IntrinsicsResolveResult::Source::FovEstimate;
    }

    // 相机分组提示（同一次 EXIF 扫描产出，避免重复读盘）
    if (r.exif.numGroups == 1) {
        r.messages.append(QStringLiteral("[相机分组] 全部图像属于同一相机组: %1")
                              .arg(r.exif.dominantGroup));
    } else if (r.exif.numGroups > 1) {
        r.messages.append(
            QStringLiteral("[相机分组] 识别到 %1 个相机分组，主导组: %2"
                           "（内参当前仍按共享一组优化）")
                .arg(r.exif.numGroups).arg(r.exif.dominantGroup));
    }

    appendKSummary(r);
    return r;
}

} // namespace sfm
