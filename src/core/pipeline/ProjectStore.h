#ifndef PROJECTSTORE_H
#define PROJECTSTORE_H

#include <QDateTime>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <vector>

#include "data/SfmTypes.h"

class PluginRegistry;

/**
 * @brief 项目清单（.sfmproj）—— 描述一个重建项目的全部元数据
 *
 * 项目采用目录式结构，清单文件位于项目根目录：
 *   MyProject/
 *   ├── MyProject.sfmproj      ← 本清单（JSON）
 *   ├── clouds/sparse.ply      ← 稀疏点云（v2=算法坐标系；v1=显示系）
 *   ├── clouds/dense.ply       ← 密集点云（存在时）
 *   ├── cameras/poses.json     ← 相机内外参（JSON，算法坐标系）
 *   └── matches/pairs.bin      ← 稀疏匹配对（紧凑二进制，密集续算输入）
 *
 * 图像文件引用原路径（不复制进项目）；移动或删除源图像后项目
 * 无法重新执行完整重建，但点云查看与密集续算不受影响。
 *
 * formatVersion：1=PLY 为显示系；2=PLY 为算法系（与 poses 一致，ADR-005）。
 */
struct ProjectManifest
{
    int formatVersion = 2;
    /// "opencv"（算法系）或 "display"（v1 遗留）；空则按 formatVersion 推断
    QString coordinateFrame;
    QString name;                       ///< 项目名称
    QDateTime createdAt;
    QDateTime modifiedAt;

    QStringList imageFiles;             ///< 输入图像绝对路径列表（引用原路径，imageId 即下标）

    // 算法配置（恢复 UI 表单 / 全量重跑使用）
    QString descriptorType;
    QJsonObject descriptorParams;
    QString matcherType;
    QJsonObject matcherParams;

    // 创建项目时检测到的固定内参（可空）
    bool fixCameraMatrix = false;
    std::vector<double> cameraMatrix;   ///< 9 个元素行优先（3×3）；空表示未固定

    // 结果文件（相对项目根目录的路径）
    QString sparseCloudPath = QStringLiteral("clouds/sparse.ply");
    QString denseCloudPath;             ///< 空表示无密集结果
    QString cameraPosesPath = QStringLiteral("cameras/poses.json");
    QString matchPairsPath = QStringLiteral("matches/pairs.bin");

    int totalPoints = 0;
    double reprojectionError = 0.0;
    double totalElapsedMs = 0.0;
};

/**
 * @brief 项目文件读写（清单 JSON + 相机位姿 JSON + 匹配对二进制）
 *
 * 纯 Qt + OpenCV Core 实现，不依赖 PCL；点云的读写由
 * ReconstructionPipeline（PCL 侧）负责。
 */
namespace ProjectStore
{
/// 读时坐标系迁移：coordinateFrame 为空时 v1→display、v≥2→opencv（不改 formatVersion）
void migrateManifestCoordinateFrame(ProjectManifest &m);

/**
 * @brief schema 驱动：补缺键 + 软纠正非法值（ADR-002）
 *
 * registry 为空时仅做坐标系迁移。descriptor/matcher 插件缺失时跳过对应侧，不失败。
 * warn 收集纠正说明（可选）。
 */
void normalizeManifestAlgorithms(ProjectManifest &m, const PluginRegistry *registry,
                                 QString *warn = nullptr);

/// 写清单到 <projectDir>/<name>.sfmproj（自动创建项目子目录结构）
bool writeManifest(const QString &projectDir, const ProjectManifest &m,
                   QString *error = nullptr);

/// 从 <projectDir>/<name>.sfmproj 读清单（name 由调用者先行指定或扫描唯一 *.sfmproj）
bool readManifest(const QString &projectDir, const QString &projectName,
                  ProjectManifest *out, QString *error = nullptr);

/// 在项目目录中扫描 *.sfmproj 清单文件（取字典序第一个），未找到返回空串
QString findManifestName(const QString &projectDir);

/// 清单文件标准路径：<projectDir>/<name>.sfmproj
QString manifestPath(const QString &projectDir, const QString &projectName);

// ── 相机位姿（JSON，人类可读，外部工具可解析）──
bool writeCameraPoses(const QString &filePath, const std::vector<CameraPose> &poses,
                      const QStringList &imageFiles, QString *error = nullptr);
bool readCameraPoses(const QString &filePath, std::vector<CameraPose> *out,
                     QString *error = nullptr);

// ── 匹配对（紧凑二进制：points1/points2 数量大，JSON 又慢又肥）──
bool writeMatchPairs(const QString &filePath, const std::vector<MatchPair> &pairs,
                     QString *error = nullptr);
bool readMatchPairs(const QString &filePath, std::vector<MatchPair> *out,
                    QString *error = nullptr);

} // namespace ProjectStore

#endif // PROJECTSTORE_H
