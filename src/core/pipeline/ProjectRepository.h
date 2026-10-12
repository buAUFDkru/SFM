#ifndef PROJECTREPOSITORY_H
#define PROJECTREPOSITORY_H

/**
 * @file ProjectRepository.h
 * @brief 项目持久化门面（阶段二）：清单 / 位姿 / 匹配 / PLY / cameramatrix
 *
 * PLY 统一存算法坐标系（ADR-005）；formatVersion≥2 + coordinateFrame=opencv。
 * 读 v1（显示系 PLY）时在加载边界做反向翻转，再转显示系供 UI。
 */

#include <functional>
#include <vector>

#include <QString>
#include <QStringList>

#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "pipeline/ProjectStore.h"

struct ReconstructionConfig;
struct ReconstructionOutput;

namespace sfm {

/// 磁盘上点云坐标系（与 ProjectManifest::coordinateFrame 对应）
enum class CloudFileFrame {
    OpenCv,  ///< 算法系（v2 默认）
    Display  ///< 显示系（v1 遗留）
};

class ProjectRepository
{
public:
    using LogFn = std::function<void(const QString &)>;
    using CloudPtr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;

    static bool saveCloudPly(const CloudPtr &cloud, const QString &filePath,
                             const QString &stageLabel, const LogFn &log);

    /**
     * @brief 读 PLY 并产出**显示坐标系**点云（供 lastOutput / UI）
     * @param storedFrame 文件内坐标系；OpenCv 时先 toDisplay
     */
    static bool loadCloudAsDisplay(const QString &filePath, CloudFileFrame storedFrame,
                                   CloudPtr *outDisplay, QString *error = nullptr);

    /// 将显示系点云克隆为算法系后写入（saveProject 用）
    static bool saveDisplayCloudAsAlgorithm(const CloudPtr &displayCloud,
                                            const QString &filePath,
                                            const QString &stageLabel,
                                            const LogFn &log);

    static bool writeCameraMatrixTxt(const QString &filePath,
                                     const std::vector<CameraPose> &poses,
                                     const QStringList &imageFiles,
                                     QString *error = nullptr);

    static CloudFileFrame frameFromManifest(const ProjectManifest &m);

    /// projectPath 优先，否则用首张图像目录；创建父目录；空配置返回空串
    static QString resolveOutputPath(const ReconstructionConfig &config,
                                     const QString &relativeFile);

    /**
     * @brief 保存完整项目（清单 + 侧车 + 算法系 PLY）
     *
     * lastOutput 内点云为显示系；写入前转换为算法系。
     * 清单固定写 formatVersion=2、coordinateFrame=opencv。
     */
    static bool saveProject(const ReconstructionConfig &config,
                            ProjectManifest *manifestInOut,
                            const std::vector<CameraPose> &cameraPoses,
                            const std::vector<MatchPair> &matchPairs,
                            const ReconstructionOutput &lastOutput,
                            const LogFn &log,
                            QString *error = nullptr);

    /**
     * @brief 打开项目并恢复管线状态；点云经迁移后以显示系写入 lastOutput
     */
    static bool loadProject(const QString &path,
                            ReconstructionConfig *config,
                            ProjectManifest *manifestOut,
                            std::vector<CameraPose> *cameraPoses,
                            std::vector<MatchPair> *matchPairs,
                            cv::Mat *K,
                            bool *sparseReady,
                            ReconstructionOutput *lastOutput,
                            const LogFn &log,
                            QString *error = nullptr);
};

} // namespace sfm

#endif // PROJECTREPOSITORY_H
