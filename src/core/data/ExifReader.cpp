/**
 * @file ExifReader.cpp
 * @brief JPEG APP1/Exif 段解析 + 35mm 等效焦距 → 像素焦距换算（阶段五 5.1/5.3）
 */

#include "ExifReader.h"

#include <QByteArray>
#include <QFile>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace SfmExif {
namespace {

// ─── EXIF Tag ─────────────────────────────────────────────────────────────
constexpr uint16_t kTagMake            = 0x010F;
constexpr uint16_t kTagModel           = 0x0110;
constexpr uint16_t kTagOrientation     = 0x0112;
constexpr uint16_t kTagExifIfdPointer  = 0x8769;
constexpr uint16_t kTagFocalLength     = 0x920A;
constexpr uint16_t kTagPixelXDimension = 0xA002;
constexpr uint16_t kTagPixelYDimension = 0xA003;
constexpr uint16_t kTagFocalLength35mm = 0xA405;

/// TIFF 数据类型 → 每元素字节数（0 表示不支持）
inline int typeSize(uint16_t type)
{
    switch (type) {
    case 1:  return 1;  // BYTE
    case 2:  return 1;  // ASCII
    case 3:  return 2;  // SHORT
    case 4:  return 4;  // LONG
    case 5:  return 8;  // RATIONAL (2×LONG)
    case 7:  return 1;  // UNDEFINED
    case 9:  return 4;  // SLONG
    case 10: return 8;  // SRATIONAL (2×SLONG)
    default: return 0;
    }
}

/**
 * @brief TIFF 块只读访问器
 *
 * 所有 TIFF 内部偏移都相对于 TIFF 头（"II"/"MM"）的起始位置。
 */
class TiffView
{
public:
    TiffView(const uchar *data, std::size_t size, std::size_t base, bool bigEndian)
        : m_data(data), m_size(size), m_base(base), m_be(bigEndian) {}

    const uchar *ptr() const { return m_data; }
    std::size_t base() const { return m_base; }
    std::size_t size() const { return m_size; }

    bool inRange(std::size_t off, std::size_t need) const
    {
        return off <= m_size && m_size - off >= need;
    }

    uint16_t u16(std::size_t off) const
    {
        const uchar *p = m_data + off;
        return m_be ? static_cast<uint16_t>((p[0] << 8) | p[1])
                    : static_cast<uint16_t>((p[1] << 8) | p[0]);
    }

    uint32_t u32(std::size_t off) const
    {
        const uchar *p = m_data + off;
        return m_be ? (static_cast<uint32_t>(p[0]) << 24 |
                       static_cast<uint32_t>(p[1]) << 16 |
                       static_cast<uint32_t>(p[2]) << 8  |
                       static_cast<uint32_t>(p[3]))
                    : (static_cast<uint32_t>(p[3]) << 24 |
                       static_cast<uint32_t>(p[2]) << 16 |
                       static_cast<uint32_t>(p[1]) << 8  |
                       static_cast<uint32_t>(p[0]));
    }

    /// 读取 RATIONAL（num/den），分母为 0 时返回 0
    double rational(std::size_t off) const
    {
        if (!inRange(off, 8)) return 0.0;
        const uint32_t den = u32(off + 4);
        if (den == 0) return 0.0;
        return static_cast<double>(u32(off)) / static_cast<double>(den);
    }

private:
    const uchar *m_data;
    std::size_t m_size;
    std::size_t m_base;
    bool m_be;
};

/// 读取 ASCII 字段（EXIF 的 ASCII 常以 '\0' 结尾，需截断）
QString asciiValue(const TiffView &tv, std::size_t pos, uint32_t n)
{
    const char *s = reinterpret_cast<const char *>(tv.ptr() + pos);
    uint32_t len = n;
    for (uint32_t i = 0; i < n; ++i) {
        if (s[i] == '\0') { len = i; break; }
    }
    return QString::fromLatin1(s, static_cast<int>(len)).trimmed();
}

/// 解析一个 IFD；isExifIfd=true 时按 Exif SubIFD 语义读焦距/尺寸类标签
void parseIfd(const TiffView &tv, std::size_t ifdOff, bool isExifIfd, ExifInfo &out)
{
    if (!tv.inRange(ifdOff, 2)) return;
    const uint32_t count = tv.u16(ifdOff);
    const std::size_t entryBase = ifdOff + 2;
    if (!tv.inRange(entryBase, static_cast<std::size_t>(count) * 12)) return;

    for (uint32_t i = 0; i < count; ++i) {
        const std::size_t e = entryBase + static_cast<std::size_t>(i) * 12;
        const uint16_t tag  = tv.u16(e);
        const uint16_t type = tv.u16(e + 2);
        const uint32_t n    = tv.u32(e + 4);
        const int tsz = typeSize(type);
        if (tsz <= 0 || n == 0) continue;

        const std::size_t total = static_cast<std::size_t>(tsz) * n;
        // 值 ≤ 4 字节时内联在「值偏移」字段里，否则该字段是相对 TIFF 头的偏移
        const std::size_t valuePos = (total <= 4) ? (e + 8)
                                                  : (tv.base() + tv.u32(e + 8));
        if (!tv.inRange(valuePos, total)) continue;

        if (isExifIfd) {
            switch (tag) {
            case kTagFocalLength:
                if (type == 5) out.focalMm = tv.rational(valuePos);
                break;
            case kTagFocalLength35mm:
                if (type == 3)      out.focal35Mm = static_cast<double>(tv.u16(valuePos));
                else if (type == 4) out.focal35Mm = static_cast<double>(tv.u32(valuePos));
                break;
            case kTagPixelXDimension:
                if (type == 3)      out.pixelWidth  = tv.u16(valuePos);
                else if (type == 4) out.pixelWidth  = static_cast<int>(tv.u32(valuePos));
                break;
            case kTagPixelYDimension:
                if (type == 3)      out.pixelHeight = tv.u16(valuePos);
                else if (type == 4) out.pixelHeight = static_cast<int>(tv.u32(valuePos));
                break;
            default:
                break;
            }
            continue;
        }

        switch (tag) {
        case kTagMake:
            if (type == 2) out.make = asciiValue(tv, valuePos, n);
            break;
        case kTagModel:
            if (type == 2) out.model = asciiValue(tv, valuePos, n);
            break;
        case kTagOrientation:
            if (type == 3) out.orientation = tv.u16(valuePos);
            break;
        case kTagExifIfdPointer:
            if (type == 4) parseIfd(tv, tv.base() + tv.u32(valuePos), true, out);
            break;
        default:
            break;
        }
    }
}

/// 解析 TIFF 头 + IFD0
void parseTiff(const uchar *data, std::size_t size, std::size_t base, ExifInfo &out)
{
    if (base + 8 > size) return;
    const bool be = (data[base] == 'M' && data[base + 1] == 'M');
    const bool le = (data[base] == 'I' && data[base + 1] == 'I');
    if (!be && !le) return;

    const TiffView tv(data, size, base, be);
    if (tv.u16(base + 2) != 42) return;         // TIFF magic
    parseIfd(tv, base + tv.u32(base + 4), false, out);
    out.valid = true;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════

ExifInfo read(const QString &filePath)
{
    ExifInfo info;

    QFile f(filePath);
    if (!f.open(QIODevice::ReadOnly)) return info;

    // EXIF 一定在文件头部：只读取前 256KB 足够覆盖 APP1 段
    const QByteArray head = f.read(256 * 1024);
    f.close();

    const int n = head.size();
    if (n < 4) return info;
    const uchar *d = reinterpret_cast<const uchar *>(head.constData());

    if (!(d[0] == 0xFF && d[1] == 0xD8)) return info;   // 非 JPEG

    std::size_t pos = 2;
    while (pos + 4 <= static_cast<std::size_t>(n)) {
        if (d[pos] != 0xFF) break;
        const uchar marker = d[pos + 1];

        // 无长度字段的独立标记
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            pos += 2;
            continue;
        }
        if (marker == 0xDA || marker == 0xD9) break;    // SOS / EOI：后面是压缩数据

        const std::size_t len = (static_cast<std::size_t>(d[pos + 2]) << 8) | d[pos + 3];
        if (len < 2 || pos + 2 + len > static_cast<std::size_t>(n)) break;

        if (marker == 0xE1) {                            // APP1
            const std::size_t payload = pos + 4;
            if (payload + 6 <= static_cast<std::size_t>(n) &&
                std::memcmp(d + payload, "Exif\0\0", 6) == 0) {
                parseTiff(d, static_cast<std::size_t>(n), payload + 6, info);
                if (info.valid) return info;
            }
        }
        pos += 2 + len;
    }
    return info;
}

// ═══════════════════════════════════════════════════════════════════════════

IntrinsicsGuess estimateFromExif(const QStringList &imageFiles,
                                 const std::vector<FeatureResult> &features)
{
    IntrinsicsGuess g;

    const int n = std::min(static_cast<int>(imageFiles.size()),
                           static_cast<int>(features.size()));
    g.numImages = n;
    if (n <= 0) {
        g.reason = QStringLiteral("无图像");
        return g;
    }

    g.cameraGroup.assign(static_cast<std::size_t>(n), -1);
    std::map<QString, int> keyToId;
    std::vector<int> groupCounts;

    std::vector<double> focals;         // 像素焦距
    std::vector<double> focal35s;       // 35mm 等效焦距（用于日志）
    std::vector<int> widths, heights;   // 参与焦距统计的图像尺寸

    for (int i = 0; i < n; ++i) {
        const int W = features[static_cast<std::size_t>(i)].imageWidth;
        const int H = features[static_cast<std::size_t>(i)].imageHeight;

        const ExifInfo info = read(imageFiles.at(i));

        // ── 5.3 相机分组 ──────────────────────────────────────────────────
        const QString key = info.groupKey(W, H);
        auto it = keyToId.find(key);
        if (it == keyToId.end()) {
            const int id = static_cast<int>(g.groupKeys.size());
            it = keyToId.emplace(key, id).first;
            g.groupKeys.push_back(key);
            groupCounts.push_back(0);
        }
        g.cameraGroup[static_cast<std::size_t>(i)] = it->second;
        ++groupCounts[static_cast<std::size_t>(it->second)];

        // ── 5.1 焦距换算 ──────────────────────────────────────────────────
        if (!info.valid || !info.hasFocal() || W <= 0 || H <= 0) continue;

        // 图像被缩放过时，焦距要按宽度比例换算到当前像素单位
        double scale = 1.0;
        if (info.pixelWidth > 0) {
            const double r = static_cast<double>(W) / static_cast<double>(info.pixelWidth);
            if (r > 1.02 || r < 0.98) scale = r;
        }

        const double diagPx = std::sqrt(static_cast<double>(W) * W +
                                        static_cast<double>(H) * H) * scale;
        focals.push_back(info.focal35Mm * diagPx / kFullFrameDiagonalMm);
        focal35s.push_back(info.focal35Mm);
        widths.push_back(W);
        heights.push_back(H);
    }

    g.numGroups = static_cast<int>(g.groupKeys.size());
    g.numWithExif = static_cast<int>(focals.size());
    if (g.numGroups > 0) {
        const size_t dom = static_cast<size_t>(
            std::max_element(groupCounts.begin(), groupCounts.end()) - groupCounts.begin());
        g.dominantGroup = g.groupKeys[dom];
    }

    // ── 可用性判定 ────────────────────────────────────────────────────────
    if (focals.size() * 2 < static_cast<size_t>(n)) {
        g.reason = QStringLiteral("仅 %1/%2 张图含可用 EXIF 焦距（需 ≥50%）")
                       .arg(g.numWithExif).arg(n);
        return g;
    }
    if (focals.size() < 2) {
        g.reason = QStringLiteral("有效 EXIF 焦距不足 2 张");
        return g;
    }

    std::vector<double> sorted = focals;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    const double minF = sorted.front();
    const double maxF = sorted.back();
    if (median <= 0.0) {
        g.reason = QStringLiteral("EXIF 焦距非正");
        return g;
    }
    if ((maxF - minF) / median > 0.10) {
        g.reason = QStringLiteral("EXIF 焦距不一致（%1~%2 px，极差 %3%）—— 疑似多镜头/变焦")
                       .arg(minF, 0, 'f', 1).arg(maxF, 0, 'f', 1)
                       .arg((maxF - minF) / median * 100.0, 0, 'f', 1);
        return g;
    }

    // 主点取参与统计图像尺寸的中位数（同组内尺寸一致，取中位即可）
    std::vector<int> sw = widths, sh = heights;
    std::sort(sw.begin(), sw.end());
    std::sort(sh.begin(), sh.end());
    const double cx = sw[sw.size() / 2] * 0.5;
    const double cy = sh[sh.size() / 2] * 0.5;

    g.focalPx    = median;
    g.focalPxMin = minF;
    g.focalPxMax = maxF;

    std::vector<double> s35 = focal35s;
    std::sort(s35.begin(), s35.end());
    g.focal35Mm = s35[s35.size() / 2];

    g.K = (cv::Mat_<double>(3, 3) << median, 0.0,    cx,
                                     0.0,    median, cy,
                                     0.0,    0.0,    1.0);
    g.valid = true;
    return g;
}

} // namespace SfmExif
