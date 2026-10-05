#ifndef EXIFREADER_H
#define EXIFREADER_H

#include <QString>
#include <QStringList>

#include <vector>

#include <opencv2/core.hpp>

#include "plugin/PluginTypes.h"

/**
 * @brief EXIF 元数据读取 + 内参先验估计（优化方案 阶段五 5.1 / 5.3）
 *
 * 无人机航拍 JPG 普遍带 EXIF，其中的「35mm 等效焦距」可以反推出相当准确的
 * 像素焦距初值，从而把自标定从「60° FOV 猜一个值再靠 BA 拉回来」变成
 * 「从一个接近真值的点出发微调」：
 *
 *   1. 弱角度 / 窄基线数据集上，焦距与场景深度、主点强耦合，初值越差越容易
 *      滑向非物理解（阶段三实测出现过 CHOLMOD 非正定）。
 *   2. 收敛更快 —— 实测 106MEDIA 自标定收敛到 f≈3677px，而 EXIF 反推为
 *      3659px，偏差 0.5%，几乎不用优化。
 *
 * 35mm 等效焦距按 **对角线视角等价** 定义（CIPA DCG-001）：
 *   f35 / D35 = f_real / D_sensor，D35 = √(36² + 24²) = 43.2666 mm
 * 因此在本图像（宽 W 高 H、单位像素）上的像素焦距为：
 *   f_px = f35 * √(W² + H²) / 43.2666
 *
 * 若图像被缩放过（EXIF PixelXDimension ≠ 实际宽度），按宽度比例线性缩放。
 *
 * 只解析 JPEG 的 APP1/Exif 段，不依赖 libexif / Qt ImageMetaData。
 */
namespace SfmExif {

/// 35mm 全画幅对角线长度（mm），35mm 等效焦距的定义基准
constexpr double kFullFrameDiagonalMm = 43.26661530055703;

/**
 * @brief 单张图像的 EXIF 摘要
 */
struct ExifInfo
{
    bool valid = false;         ///< 是否成功解析出 EXIF（TIFF IFD0 可读）
    QString make;               ///< 相机厂商（Tag 0x010F）
    QString model;              ///< 相机型号（Tag 0x0110）
    int orientation = 0;        ///< 方向（Tag 0x0112）
    double focalMm = 0.0;       ///< 实际焦距（Tag 0x920A，单位 mm）
    double focal35Mm = 0.0;     ///< 35mm 等效焦距（Tag 0xA405，单位 mm）
    int pixelWidth = 0;         ///< EXIF 记录的图像宽（Tag 0xA002）
    int pixelHeight = 0;        ///< EXIF 记录的图像高（Tag 0xA003）

    /// 是否含有可用的焦距信息（只有实际焦距而没有传感器尺寸时无法换算）
    bool hasFocal() const { return focal35Mm > 0.0; }

    /**
     * @brief 相机分组键（阶段五 5.3）
     *
     * 同一台相机拍的图共享内参；不同相机（厂商+型号+分辨率任一不同）视为不同组。
     */
    QString groupKey(int width, int height) const
    {
        if (!valid) {
            return QStringLiteral("<无EXIF> | %1x%2").arg(width).arg(height);
        }
        return QStringLiteral("%1 %2 | %3x%4")
            .arg(make.trimmed(), model.trimmed())
            .arg(width).arg(height);
    }
};

/**
 * @brief 读取图像文件的 EXIF 信息
 *
 * 只处理 JPEG（SOI=FFD8 + APP1/Exif）。PNG / 无 EXIF 的 JPG 返回 valid=false。
 * 只读文件的前若干个 segment，遇到 SOS（FFDA）即停止，不会整文件加载两次
 * （仍是一次 readAll，但解析量极小）。
 */
ExifInfo read(const QString &filePath);

/**
 * @brief 由 EXIF 汇总出的内参先验
 */
struct IntrinsicsGuess
{
    bool valid = false;          ///< 是否可用（有效图像占比 + 焦距一致性都通过）
    cv::Mat K;                   ///< 由 EXIF 反推的内参矩阵（valid 时有效）
    int numImages = 0;           ///< 参与统计的图像总数
    int numWithExif = 0;         ///< 解析到 35mm 等效焦距的图像数
    int numGroups = 0;           ///< 识别出的相机分组数（5.3）
    double focalPx = 0.0;        ///< 像素焦距（各图中位数）
    double focalPxMin = 0.0;     ///< 最小像素焦距（一致性检查用）
    double focalPxMax = 0.0;     ///< 最大像素焦距
    double focal35Mm = 0.0;      ///< 中位数对应的 35mm 等效焦距
    QString dominantGroup;       ///< 主导分组（图像数最多的组）的键
    QString reason;              ///< 不可用时的原因（用于日志）
    std::vector<int> cameraGroup;///< 每张图的分组 id（索引同输入顺序）
    std::vector<QString> groupKeys;///< 分组 id → 分组键
};

/**
 * @brief 汇总整个数据集的 EXIF 焦距，产出内参先验
 *
 * 判定为「可用」的条件（任一不满足则 valid=false，调用方回退到 estimateK）：
 *   1. 至少 50% 的图像解析到 35mm 等效焦距；
 *   2. 这些焦距的相对极差 (max-min)/median ≤ 10%（同一支镜头）。
 * 取中位数而非均值，避免个别异常 EXIF 把整组带偏。
 *
 * @param imageFiles 图像文件路径（与 features 一一对应）
 * @param features   特征提取结果，提供实际参与计算的图像宽高
 */
IntrinsicsGuess estimateFromExif(const QStringList &imageFiles,
                                 const std::vector<FeatureResult> &features);

} // namespace SfmExif

#endif // EXIFREADER_H
