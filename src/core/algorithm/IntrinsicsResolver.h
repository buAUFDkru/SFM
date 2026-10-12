#ifndef INTRINSICSRESOLVER_H
#define INTRINSICSRESOLVER_H

/**
 * @file IntrinsicsResolver.h
 * @brief 内参初值三级回退 + EXIF 相机分组（ADR / 阶段三）
 *
 * 优先级：外部提供 K > EXIF 焦距反推 > 按图像尺寸 60° FOV 估计。
 * 从 ReconstructionPipeline 抽出，便于单独单测，不依赖管线状态。
 */

#include <QString>
#include <QStringList>

#include <vector>

#include <opencv2/core.hpp>

#include "data/ExifReader.h"
#include "plugin/PluginTypes.h"

namespace sfm {

struct IntrinsicsResolveResult
{
    enum class Source : int {
        External,     ///< 配置/项目文件提供的 K
        Exif,         ///< EXIF 35mm 等效焦距反推
        FovEstimate   ///< 按首图尺寸 60° FOV 估计
    };

    cv::Mat K;
    Source source = Source::FovEstimate;
    SfmExif::IntrinsicsGuess exif;   ///< 扫描结果（即便最终未采用 EXIF 的 K）
    QStringList messages;            ///< 供管线日志输出的说明行
};

class IntrinsicsResolver
{
public:
    /**
     * @param externalK     非空则优先采用（clone）
     * @param useExifFocal  是否扫描 EXIF（即使最终用外部 K 也会扫描，供分组与 trusted 标志）
     * @param imageFiles    与 features 一一对应的路径
     * @param features      提供实际参与计算的宽高
     */
    static IntrinsicsResolveResult resolve(const cv::Mat &externalK,
                                           bool useExifFocal,
                                           const QStringList &imageFiles,
                                           const std::vector<FeatureResult> &features);
};

} // namespace sfm

#endif // INTRINSICSRESOLVER_H
