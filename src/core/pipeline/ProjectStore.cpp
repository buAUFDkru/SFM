/**
 * @file ProjectStore.cpp
 * @brief 项目文件读写实现（清单 JSON + 相机位姿 JSON + 匹配对二进制）
 */

#include "ProjectStore.h"

#include "algorithm/MatcherService.h"
#include "plugin/IDescriptorPlugin.h"
#include "plugin/IMatcherPlugin.h"
#include "plugin/ParamSchemaUtils.h"
#include "plugin/PluginRegistry.h"

#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>

#include <opencv2/core.hpp>

namespace {

constexpr int kManifestVersion = 2;
constexpr int kPosesVersion = 1;
// v2：MatchPair 新增本质矩阵 E 与两视图几何配置 geometryConfig（阶段三）
constexpr int kPairsVersion = 2;
const char kPairsMagic[] = "SFMPAIRS";

// ── JSON 辅助：cv::Mat ↔ QJsonArray ──

QJsonArray matToJson(const cv::Mat &m)
{
    QJsonArray rows;
    if (m.empty() || m.type() != CV_64F)
        return rows;
    for (int r = 0; r < m.rows; ++r) {
        QJsonArray row;
        for (int c = 0; c < m.cols; ++c)
            row.append(m.at<double>(r, c));
        rows.append(row);
    }
    return rows;
}

cv::Mat jsonToMat(const QJsonValue &v, int rows, int cols)
{
    if (!v.isArray())
        return cv::Mat();
    const QJsonArray arr = v.toArray();
    if (arr.size() != rows)
        return cv::Mat();
    cv::Mat m(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r) {
        const QJsonArray row = arr.at(r).toArray();
        if (row.size() != cols)
            return cv::Mat();
        for (int c = 0; c < cols; ++c)
            m.at<double>(r, c) = row.at(c).toDouble();
    }
    return m;
}

cv::Mat poseMatOrIdentity(const cv::Mat &m, int rows, int cols, bool identity)
{
    if (!m.empty())
        return m;
    return identity ? cv::Mat::eye(rows, cols, CV_64F) : cv::Mat::zeros(rows, cols, CV_64F);
}

} // namespace

namespace ProjectStore {

void migrateManifestCoordinateFrame(ProjectManifest &m)
{
    if (!m.coordinateFrame.isEmpty())
        return;
    m.coordinateFrame = (m.formatVersion >= 2) ? QStringLiteral("opencv")
                                               : QStringLiteral("display");
}

void normalizeManifestAlgorithms(ProjectManifest &m, const PluginRegistry *registry,
                                 QString *warn)
{
    migrateManifestCoordinateFrame(m);
    if (!registry)
        return;

    if (!m.descriptorType.isEmpty()) {
        if (IDescriptorPlugin *desc = registry->descriptorPlugin(m.descriptorType)) {
            const QList<ParamSchema> schemas = desc->paramSchema();
            m.descriptorParams = sfm::sanitizeParamsAgainstSchema(
                schemas, m.descriptorParams, warn);
            m.descriptorParams = sfm::fillParamsFromSchema(schemas, m.descriptorParams);
        }
    }

    if (!m.matcherType.isEmpty()) {
        QList<ParamSchema> schemas = MatcherService::orchestrationParamSchema();
        if (IMatcherPlugin *matcher = registry->matcherPlugin(m.matcherType))
            schemas = sfm::mergeParamSchemas(schemas, matcher->paramSchema());
        m.matcherParams =
            sfm::sanitizeParamsAgainstSchema(schemas, m.matcherParams, warn);
        m.matcherParams = sfm::fillParamsFromSchema(schemas, m.matcherParams);
    }
}

QString manifestPath(const QString &projectDir, const QString &projectName)
{
    return QDir(projectDir).filePath(projectName + QStringLiteral(".sfmproj"));
}

QString findManifestName(const QString &projectDir)
{
    const QFileInfoList entries =
        QDir(projectDir).entryInfoList({QStringLiteral("*.sfmproj")}, QDir::Files, QDir::Name);
    return entries.isEmpty() ? QString() : entries.first().completeBaseName();
}

// ═══════════════════════════════ 清单 ═══════════════════════════════

bool writeManifest(const QString &projectDir, const ProjectManifest &m, QString *error)
{
    QDir dir(projectDir);
    if (!dir.exists() && !dir.mkpath(projectDir)) {
        if (error) *error = QStringLiteral("无法创建项目目录: %1").arg(projectDir);
        return false;
    }

    QJsonObject root;
    root[QStringLiteral("formatVersion")] =
        m.formatVersion > 0 ? m.formatVersion : kManifestVersion;
    root[QStringLiteral("coordinateFrame")] =
        m.coordinateFrame.isEmpty() ? QStringLiteral("opencv") : m.coordinateFrame;
    root[QStringLiteral("name")] = m.name;
    root[QStringLiteral("createdAt")] = m.createdAt.toString(Qt::ISODate);
    root[QStringLiteral("modifiedAt")] = m.modifiedAt.toString(Qt::ISODate);
    root[QStringLiteral("images")] = QJsonArray::fromStringList(m.imageFiles);

    QJsonObject desc;
    desc[QStringLiteral("type")] = m.descriptorType;
    desc[QStringLiteral("params")] = m.descriptorParams;
    root[QStringLiteral("descriptor")] = desc;

    QJsonObject matcher;
    matcher[QStringLiteral("type")] = m.matcherType;
    matcher[QStringLiteral("params")] = m.matcherParams;
    root[QStringLiteral("matcher")] = matcher;

    root[QStringLiteral("fixCameraMatrix")] = m.fixCameraMatrix;
    if (!m.cameraMatrix.empty()) {
        QJsonArray k;
        for (double v : m.cameraMatrix)
            k.append(v);
        root[QStringLiteral("cameraMatrix")] = k;
    }

    QJsonObject results;
    results[QStringLiteral("sparseCloud")] = m.sparseCloudPath;
    results[QStringLiteral("denseCloud")] = m.denseCloudPath;
    results[QStringLiteral("cameraPoses")] = m.cameraPosesPath;
    results[QStringLiteral("matchPairs")] = m.matchPairsPath;
    results[QStringLiteral("totalPoints")] = m.totalPoints;
    results[QStringLiteral("reprojectionError")] = m.reprojectionError;
    results[QStringLiteral("totalElapsedMs")] = m.totalElapsedMs;
    root[QStringLiteral("results")] = results;

    const QString path = manifestPath(projectDir, m.name);
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (error) *error = QStringLiteral("无法写入清单: %1").arg(path);
        return false;
    }
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    file.close();
    return true;
}

bool readManifest(const QString &projectDir, const QString &projectName,
                  ProjectManifest *out, QString *error)
{
    const QString path = manifestPath(projectDir, projectName);
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) *error = QStringLiteral("无法打开清单: %1").arg(path);
        return false;
    }
    const QJsonObject root =
        QJsonDocument::fromJson(file.readAll()).object();
    file.close();

    if (root.isEmpty()) {
        if (error) *error = QStringLiteral("清单内容无效: %1").arg(path);
        return false;
    }

    ProjectManifest m;
    m.formatVersion = root.value(QStringLiteral("formatVersion")).toInt(1);
    m.coordinateFrame = root.value(QStringLiteral("coordinateFrame")).toString();
    m.name = root.value(QStringLiteral("name")).toString(projectName);
    m.createdAt = QDateTime::fromString(root.value(QStringLiteral("createdAt")).toString(), Qt::ISODate);
    m.modifiedAt = QDateTime::fromString(root.value(QStringLiteral("modifiedAt")).toString(), Qt::ISODate);
    m.imageFiles.clear();
    for (const QJsonValue &v : root.value(QStringLiteral("images")).toArray())
        m.imageFiles.append(v.toString());

    const QJsonObject desc = root.value(QStringLiteral("descriptor")).toObject();
    m.descriptorType = desc.value(QStringLiteral("type")).toString();
    m.descriptorParams = desc.value(QStringLiteral("params")).toObject();

    const QJsonObject matcher = root.value(QStringLiteral("matcher")).toObject();
    m.matcherType = matcher.value(QStringLiteral("type")).toString();
    m.matcherParams = matcher.value(QStringLiteral("params")).toObject();

    m.fixCameraMatrix = root.value(QStringLiteral("fixCameraMatrix")).toBool(false);
    const QJsonArray kArr = root.value(QStringLiteral("cameraMatrix")).toArray();
    if (kArr.size() == 9) {
        for (const QJsonValue &v : kArr)
            m.cameraMatrix.push_back(v.toDouble());
    }

    const QJsonObject results = root.value(QStringLiteral("results")).toObject();
    m.sparseCloudPath = results.value(QStringLiteral("sparseCloud"))
                            .toString(QStringLiteral("clouds/sparse.ply"));
    m.denseCloudPath = results.value(QStringLiteral("denseCloud")).toString();
    m.cameraPosesPath = results.value(QStringLiteral("cameraPoses"))
                            .toString(QStringLiteral("cameras/poses.json"));
    m.matchPairsPath = results.value(QStringLiteral("matchPairs"))
                           .toString(QStringLiteral("matches/pairs.bin"));
    m.totalPoints = results.value(QStringLiteral("totalPoints")).toInt();
    m.reprojectionError = results.value(QStringLiteral("reprojectionError")).toDouble();
    m.totalElapsedMs = results.value(QStringLiteral("totalElapsedMs")).toDouble();

    migrateManifestCoordinateFrame(m);

    *out = m;
    return true;
}

// ═══════════════════════════════ 相机位姿 ═══════════════════════════════

bool writeCameraPoses(const QString &filePath, const std::vector<CameraPose> &poses,
                      const QStringList &imageFiles, QString *error)
{
    QJsonObject root;
    root[QStringLiteral("version")] = kPosesVersion;
    root[QStringLiteral("images")] = QJsonArray::fromStringList(imageFiles);

    QJsonArray poseArray;
    for (const CameraPose &p : poses) {
        QJsonObject o;
        o[QStringLiteral("imageId")] = p.imageId;
        o[QStringLiteral("imageWidth")] = p.imageWidth;
        o[QStringLiteral("imageHeight")] = p.imageHeight;
        o[QStringLiteral("registered")] = p.registered;
        o[QStringLiteral("K")] = matToJson(poseMatOrIdentity(p.K, 3, 3, true));
        o[QStringLiteral("R")] = matToJson(poseMatOrIdentity(p.R, 3, 3, true));
        o[QStringLiteral("t")] = matToJson(poseMatOrIdentity(p.t, 3, 1, false));
        o[QStringLiteral("distortion")] = matToJson(poseMatOrIdentity(p.distortion, 5, 1, false));
        poseArray.append(o);
    }
    root[QStringLiteral("poses")] = poseArray;

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (error) *error = QStringLiteral("无法写入相机参数: %1").arg(filePath);
        return false;
    }
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    file.close();
    return true;
}

bool readCameraPoses(const QString &filePath, std::vector<CameraPose> *out, QString *error)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) *error = QStringLiteral("无法打开相机参数: %1").arg(filePath);
        return false;
    }
    const QJsonObject root = QJsonDocument::fromJson(file.readAll()).object();
    file.close();

    std::vector<CameraPose> poses;
    for (const QJsonValue &v : root.value(QStringLiteral("poses")).toArray()) {
        const QJsonObject o = v.toObject();
        CameraPose p;
        p.imageId = o.value(QStringLiteral("imageId")).toInt(-1);
        p.imageWidth = o.value(QStringLiteral("imageWidth")).toInt();
        p.imageHeight = o.value(QStringLiteral("imageHeight")).toInt();
        p.registered = o.value(QStringLiteral("registered")).toBool(false);
        p.K = jsonToMat(o.value(QStringLiteral("K")), 3, 3);
        p.R = jsonToMat(o.value(QStringLiteral("R")), 3, 3);
        p.t = jsonToMat(o.value(QStringLiteral("t")), 3, 1);
        p.distortion = jsonToMat(o.value(QStringLiteral("distortion")), 5, 1);
        poses.push_back(std::move(p));
    }
    *out = std::move(poses);
    return true;
}

// ═══════════════════════════════ 匹配对 ═══════════════════════════════

namespace {

/// 二进制序列化 cv::Mat：empty 标志 + rows/cols/type + 连续数据
void writeMatBin(QDataStream &s, const cv::Mat &m)
{
    if (m.empty()) {
        s << quint8(0);
        return;
    }
    s << quint8(1);
    cv::Mat cont = m.isContinuous() ? m : m.clone();
    s << qint32(cont.rows) << qint32(cont.cols) << qint32(cont.type());
    const qint64 bytes = qint64(cont.total()) * cont.elemSize();
    s.writeRawData(reinterpret_cast<const char *>(cont.data), int(bytes));
}

cv::Mat readMatBin(QDataStream &s)
{
    quint8 has = 0;
    s >> has;
    if (!has)
        return cv::Mat();
    qint32 rows = 0, cols = 0, type = 0;
    s >> rows >> cols >> type;
    cv::Mat m(rows, cols, type);
    const qint64 bytes = qint64(m.total()) * m.elemSize();
    s.readRawData(reinterpret_cast<char *>(m.data), int(bytes));
    return m;
}

} // namespace

bool writeMatchPairs(const QString &filePath, const std::vector<MatchPair> &pairs,
                     QString *error)
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        if (error) *error = QStringLiteral("无法写入匹配对: %1").arg(filePath);
        return false;
    }
    QDataStream s(&file);
    s.setVersion(QDataStream::Qt_6_0);
    s.writeRawData(kPairsMagic, 8);
    s << quint32(kPairsVersion) << quint32(pairs.size());

    for (const MatchPair &mp : pairs) {
        s << qint32(mp.imageId1) << qint32(mp.imageId2)
          << qint32(mp.matchCount) << qint32(mp.modelChoice)
          << mp.avgDistance;

        s << quint32(mp.matches.size());
        for (const cv::DMatch &dm : mp.matches)
            s << qint32(dm.queryIdx) << qint32(dm.trainIdx)
              << qint32(dm.imgIdx) << float(dm.distance);

        s << quint32(mp.points1.size());
        for (const cv::Point2d &p : mp.points1)
            s << p.x << p.y;
        s << quint32(mp.points2.size());
        for (const cv::Point2d &p : mp.points2)
            s << p.x << p.y;

        writeMatBin(s, mp.F);
        writeMatBin(s, mp.H);
        writeMatBin(s, mp.E);
        s << qint32(mp.geometryConfig);
    }
    file.close();
    return true;
}

bool readMatchPairs(const QString &filePath, std::vector<MatchPair> *out, QString *error)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法打开匹配对: %1").arg(filePath);
        return false;
    }
    QDataStream s(&file);
    s.setVersion(QDataStream::Qt_6_0);

    char magic[8] = {};
    if (s.readRawData(magic, 8) != 8 || memcmp(magic, kPairsMagic, 8) != 0) {
        if (error) *error = QStringLiteral("匹配对文件格式无效: %1").arg(filePath);
        return false;
    }
    quint32 version = 0, count = 0;
    s >> version >> count;

    std::vector<MatchPair> pairs;
    pairs.reserve(count);
    for (quint32 i = 0; i < count && s.status() == QDataStream::Ok; ++i) {
        MatchPair mp;
        qint32 id1 = 0, id2 = 0, matchCount = 0, modelChoice = 0;
        s >> id1 >> id2 >> matchCount >> modelChoice >> mp.avgDistance;
        mp.imageId1 = id1;
        mp.imageId2 = id2;
        mp.matchCount = matchCount;
        mp.modelChoice = modelChoice;

        quint32 n = 0;
        s >> n;
        mp.matches.reserve(n);
        for (quint32 j = 0; j < n; ++j) {
            qint32 q = 0, t = 0, img = 0;
            float d = 0.f;
            s >> q >> t >> img >> d;
            mp.matches.push_back({q, t, img, d});
        }

        s >> n;
        mp.points1.reserve(n);
        for (quint32 j = 0; j < n; ++j) {
            double x = 0, y = 0;
            s >> x >> y;
            mp.points1.push_back({x, y});
        }
        s >> n;
        mp.points2.reserve(n);
        for (quint32 j = 0; j < n; ++j) {
            double x = 0, y = 0;
            s >> x >> y;
            mp.points2.push_back({x, y});
        }

        mp.F = readMatBin(s);
        mp.H = readMatBin(s);
        if (version >= 2) {
            mp.E = readMatBin(s);
            qint32 cfg = 0;
            s >> cfg;
            mp.geometryConfig = cfg;
        }
        pairs.push_back(std::move(mp));
    }
    file.close();

    if (s.status() != QDataStream::Ok) {
        if (error) *error = QStringLiteral("匹配对文件读取不完整: %1").arg(filePath);
        return false;
    }
    *out = std::move(pairs);
    return true;
}

} // namespace ProjectStore
