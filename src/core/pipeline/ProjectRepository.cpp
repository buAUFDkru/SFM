#include "ProjectRepository.h"

#include "builtin/registerBuiltIn.h"
#include "data/ReconstructionOutput.h"
#include "frame/Coordinate.h"
#include "pipeline/ReconstructionPipeline.h"
#include "plugin/PluginRegistry.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>

namespace sfm {

bool ProjectRepository::saveCloudPly(const CloudPtr &cloud, const QString &filePath,
                                     const QString &stageLabel, const LogFn &log)
{
    if (!cloud || cloud->empty()) {
        if (log)
            log(QStringLiteral("[%1] 点云为空，跳过保存").arg(stageLabel));
        return false;
    }
    if (filePath.isEmpty()) {
        if (log)
            log(QStringLiteral("[%1] 未配置输出路径，跳过保存").arg(stageLabel));
        return false;
    }
    try {
        QFileInfo(filePath).dir().mkpath(QStringLiteral("."));
        pcl::io::savePLYFileBinary(filePath.toStdString(), *cloud);
        if (log) {
            log(QStringLiteral("[%1] 点云已保存: %2 (点数: %3)")
                    .arg(stageLabel)
                    .arg(filePath)
                    .arg(cloud->size()));
        }
        return true;
    } catch (const std::exception &e) {
        if (log)
            log(QStringLiteral("[%1] 点云保存异常: %2").arg(stageLabel).arg(e.what()));
    } catch (...) {
        if (log)
            log(QStringLiteral("[%1] 点云保存异常: 未知错误").arg(stageLabel));
    }
    return false;
}

bool ProjectRepository::loadCloudAsDisplay(const QString &filePath,
                                           CloudFileFrame storedFrame,
                                           CloudPtr *outDisplay, QString *error)
{
    if (!outDisplay) {
        if (error)
            *error = QStringLiteral("outDisplay 为空");
        return false;
    }
    if (!QFileInfo::exists(filePath)) {
        if (error)
            *error = QStringLiteral("点云文件不存在: %1").arg(filePath);
        return false;
    }
    CloudPtr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    const pcl::console::VERBOSITY_LEVEL prev = pcl::console::getVerbosityLevel();
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    try {
        pcl::io::loadPLYFile(filePath.toStdString(), *cloud);
    } catch (const std::exception &e) {
        pcl::console::setVerbosityLevel(prev);
        if (error)
            *error = QStringLiteral("点云加载失败: %1").arg(e.what());
        return false;
    } catch (...) {
        pcl::console::setVerbosityLevel(prev);
        if (error)
            *error = QStringLiteral("点云加载失败: 未知错误");
        return false;
    }
    pcl::console::setVerbosityLevel(prev);

    if (storedFrame == CloudFileFrame::OpenCv)
        sfm::frame::toDisplayCloudInPlace(*cloud);
    // Display：文件已是显示系，直接供 UI
    *outDisplay = cloud;
    return true;
}

bool ProjectRepository::saveDisplayCloudAsAlgorithm(const CloudPtr &displayCloud,
                                                    const QString &filePath,
                                                    const QString &stageLabel,
                                                    const LogFn &log)
{
    if (!displayCloud || displayCloud->empty())
        return saveCloudPly(displayCloud, filePath, stageLabel, log);

    CloudPtr alg(new pcl::PointCloud<pcl::PointXYZRGB>);
    *alg = *displayCloud;
    sfm::frame::toAlgorithmCloudInPlace(*alg);
    return saveCloudPly(alg, filePath, stageLabel, log);
}

CloudFileFrame ProjectRepository::frameFromManifest(const ProjectManifest &m)
{
    if (m.coordinateFrame == QStringLiteral("display"))
        return CloudFileFrame::Display;
    if (m.coordinateFrame == QStringLiteral("opencv"))
        return CloudFileFrame::OpenCv;
    // 未标注：v2+ 默认算法系；v1 遗留为显示系
    return (m.formatVersion >= 2) ? CloudFileFrame::OpenCv : CloudFileFrame::Display;
}

QString ProjectRepository::resolveOutputPath(const ReconstructionConfig &config,
                                             const QString &relativeFile)
{
    const QString saveDir =
        !config.projectPath.isEmpty()
            ? config.projectPath
            : (!config.imageFiles.isEmpty()
                   ? QFileInfo(config.imageFiles[0]).absolutePath()
                   : QString());
    if (saveDir.isEmpty())
        return {};
    const QString path = saveDir + QLatin1Char('/') + relativeFile;
    QDir().mkpath(QFileInfo(path).absolutePath());
    return path;
}

bool ProjectRepository::writeCameraMatrixTxt(const QString &filePath,
                                             const std::vector<CameraPose> &poses,
                                             const QStringList &imageFiles,
                                             QString *error)
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (error)
            *error = QStringLiteral("无法写入 cameramatrix.txt: %1").arg(filePath);
        return false;
    }
    QTextStream out(&file);
    out << "# Camera parameters estimated by SFMReconstructor\n";
    out << "# Format per camera: ImageName, then K(3x3), R(3x3), t(3x1)\n\n";
    for (const auto &pose : poses) {
        if (!pose.registered)
            continue;
        const QString imgName =
            (pose.imageId >= 0 && pose.imageId < imageFiles.size())
                ? QFileInfo(imageFiles[pose.imageId]).fileName()
                : QStringLiteral("image_%1").arg(pose.imageId);
        out << imgName << "\n";

        const cv::Mat K = pose.K.empty() ? cv::Mat::eye(3, 3, CV_64F) : pose.K;
        out << "# K (intrinsics)\n";
        for (int r = 0; r < 3; ++r) {
            out << "K " << K.at<double>(r, 0) << " " << K.at<double>(r, 1) << " "
                << K.at<double>(r, 2) << "\n";
        }
        const cv::Mat R = pose.R.empty() ? cv::Mat::eye(3, 3, CV_64F) : pose.R;
        out << "# R (rotation)\n";
        for (int r = 0; r < 3; ++r) {
            out << "R " << R.at<double>(r, 0) << " " << R.at<double>(r, 1) << " "
                << R.at<double>(r, 2) << "\n";
        }
        const cv::Mat t = pose.t.empty() ? cv::Mat::zeros(3, 1, CV_64F) : pose.t;
        out << "# t (translation)\n";
        out << "t " << t.at<double>(0) << " " << t.at<double>(1) << " "
            << t.at<double>(2) << "\n\n";
    }
    return true;
}

bool ProjectRepository::saveProject(const ReconstructionConfig &config,
                                    ProjectManifest *manifestInOut,
                                    const std::vector<CameraPose> &cameraPoses,
                                    const std::vector<MatchPair> &matchPairs,
                                    const ReconstructionOutput &lastOutput,
                                    const LogFn &log,
                                    QString *error)
{
    const auto fail = [error](const QString &msg) {
        if (error)
            *error = msg;
        return false;
    };
    if (!manifestInOut)
        return fail(QStringLiteral("manifestInOut 为空"));
    if (config.projectPath.isEmpty() || config.projectName.isEmpty())
        return fail(QStringLiteral("未打开项目，无法保存"));

    QDir(config.projectPath).mkpath(QStringLiteral("clouds"));
    QDir(config.projectPath).mkpath(QStringLiteral("cameras"));
    QDir(config.projectPath).mkpath(QStringLiteral("matches"));

    ProjectManifest &m = *manifestInOut;
    const QDateTime prevCreated = m.createdAt;
    const QString prevName = m.name;
    m.formatVersion = 2;
    m.coordinateFrame = QStringLiteral("opencv");
    m.name = config.projectName;
    m.createdAt = (prevName == config.projectName && prevCreated.isValid())
                      ? prevCreated
                      : QDateTime::currentDateTime();
    m.modifiedAt = QDateTime::currentDateTime();
    m.imageFiles = config.imageFiles;
    m.descriptorType = config.descriptorType;
    m.descriptorParams = config.descriptorParams;
    m.matcherType = config.matcherType;
    m.matcherParams = config.matcherParams;
    registerBuiltInAlgorithms(PluginRegistry::instance());
    ProjectStore::normalizeManifestAlgorithms(m, &PluginRegistry::instance(), nullptr);
    m.fixCameraMatrix = config.fixCameraMatrix;
    m.cameraMatrix.clear();
    if (config.fixCameraMatrix && !config.cameraMatrix.empty()) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                m.cameraMatrix.push_back(config.cameraMatrix.at<double>(r, c));
    }

    const QString sparsePath =
        QDir(config.projectPath).filePath(QStringLiteral("clouds/sparse.ply"));
    saveDisplayCloudAsAlgorithm(lastOutput.sparseCloud, sparsePath,
                                QStringLiteral("项目保存"), log);
    if (lastOutput.denseCloud && !lastOutput.denseCloud->empty()) {
        const QString densePath =
            QDir(config.projectPath).filePath(QStringLiteral("clouds/dense.ply"));
        saveDisplayCloudAsAlgorithm(lastOutput.denseCloud, densePath,
                                    QStringLiteral("项目保存"), log);
        m.denseCloudPath = QStringLiteral("clouds/dense.ply");
    }

    m.sparseCloudPath = QStringLiteral("clouds/sparse.ply");
    m.cameraPosesPath = QStringLiteral("cameras/poses.json");
    m.matchPairsPath = QStringLiteral("matches/pairs.bin");
    m.totalPoints = lastOutput.totalPoints;
    m.reprojectionError = lastOutput.reprojectionError;
    m.totalElapsedMs = lastOutput.totalElapsedMs;

    QString err;
    if (!cameraPoses.empty()) {
        if (!ProjectStore::writeCameraPoses(
                QDir(config.projectPath).filePath(m.cameraPosesPath), cameraPoses,
                config.imageFiles, &err))
            return fail(err);
    }
    if (!matchPairs.empty()) {
        if (!ProjectStore::writeMatchPairs(
                QDir(config.projectPath).filePath(m.matchPairsPath), matchPairs,
                &err))
            return fail(err);
    }
    if (!ProjectStore::writeManifest(config.projectPath, m, &err))
        return fail(err);

    if (log)
        log(QStringLiteral("[项目] 已保存: %1").arg(config.projectPath));
    return true;
}

bool ProjectRepository::loadProject(const QString &path,
                                    ReconstructionConfig *config,
                                    ProjectManifest *manifestOut,
                                    std::vector<CameraPose> *cameraPoses,
                                    std::vector<MatchPair> *matchPairs,
                                    cv::Mat *K,
                                    bool *sparseReady,
                                    ReconstructionOutput *lastOutput,
                                    const LogFn &log,
                                    QString *error)
{
    const auto fail = [error](const QString &msg) {
        if (error)
            *error = msg;
        return false;
    };
    if (!config || !manifestOut || !cameraPoses || !matchPairs || !K ||
        !sparseReady || !lastOutput)
        return fail(QStringLiteral("loadProject 参数不完整"));

    QString projectDir = path;
    QString projectName;
    const QFileInfo info(path);
    if (info.isFile()) {
        projectDir = info.absolutePath();
        projectName = info.completeBaseName();
    } else {
        projectName = ProjectStore::findManifestName(projectDir);
    }
    if (projectName.isEmpty())
        return fail(QStringLiteral("未找到 .sfmproj 清单: %1").arg(path));

    ProjectManifest m;
    QString err;
    if (!ProjectStore::readManifest(projectDir, projectName, &m, &err))
        return fail(err);

    // 保证内置算法 schema 可用（幂等）
    registerBuiltInAlgorithms(PluginRegistry::instance());
    QString schemaWarn;
    ProjectStore::normalizeManifestAlgorithms(m, &PluginRegistry::instance(),
                                              &schemaWarn);
    if (log && !schemaWarn.isEmpty())
        log(QStringLiteral("[项目] 参数已按 schema 纠正: %1").arg(schemaWarn));

    config->imageFiles = m.imageFiles;
    config->projectPath = projectDir;
    config->projectName = m.name;
    config->descriptorType = m.descriptorType;
    config->descriptorParams = m.descriptorParams;
    config->matcherType = m.matcherType;
    config->matcherParams = m.matcherParams;
    config->fixCameraMatrix = m.fixCameraMatrix;
    config->cameraMatrix = cv::Mat();
    if (m.cameraMatrix.size() == 9) {
        config->cameraMatrix = cv::Mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                config->cameraMatrix.at<double>(r, c) = m.cameraMatrix[r * 3 + c];
    }

    *sparseReady = false;
    cameraPoses->clear();
    matchPairs->clear();
    *K = cv::Mat();
    *lastOutput = ReconstructionOutput();
    *manifestOut = m;

    if (!m.cameraPosesPath.isEmpty()) {
        std::vector<CameraPose> poses;
        if (!ProjectStore::readCameraPoses(
                projectDir + QStringLiteral("/") + m.cameraPosesPath, &poses, &err)) {
            if (log)
                log(QStringLiteral("[项目] 警告: %1").arg(err));
        } else if (!poses.empty()) {
            *cameraPoses = std::move(poses);
            *sparseReady = true;
            for (const auto &pose : *cameraPoses) {
                if (pose.registered && !pose.K.empty()) {
                    *K = pose.K.clone();
                    break;
                }
            }
        }
    }

    if (!m.matchPairsPath.isEmpty()) {
        std::vector<MatchPair> pairs;
        if (!ProjectStore::readMatchPairs(
                projectDir + QStringLiteral("/") + m.matchPairsPath, &pairs, &err)) {
            if (log)
                log(QStringLiteral("[项目] 警告: %1").arg(err));
        } else {
            *matchPairs = std::move(pairs);
        }
    }

    const CloudFileFrame frame = frameFromManifest(m);
    if (!m.sparseCloudPath.isEmpty()) {
        CloudPtr display;
        if (loadCloudAsDisplay(projectDir + QStringLiteral("/") + m.sparseCloudPath,
                               frame, &display, &err)) {
            lastOutput->sparseCloud = display;
        } else if (log && QFileInfo::exists(projectDir + QStringLiteral("/") +
                                            m.sparseCloudPath)) {
            log(QStringLiteral("[项目] 稀疏点云: %1").arg(err));
        }
    }
    if (!m.denseCloudPath.isEmpty()) {
        CloudPtr display;
        if (loadCloudAsDisplay(projectDir + QStringLiteral("/") + m.denseCloudPath,
                               frame, &display, &err)) {
            lastOutput->denseCloud = display;
        } else if (log && QFileInfo::exists(projectDir + QStringLiteral("/") +
                                            m.denseCloudPath)) {
            log(QStringLiteral("[项目] 密集点云: %1").arg(err));
        }
    }

    lastOutput->cameraPoses = *cameraPoses;
    lastOutput->totalPoints = m.totalPoints;
    lastOutput->reprojectionError = m.reprojectionError;
    lastOutput->totalElapsedMs = m.totalElapsedMs;

    if (log) {
        log(QStringLiteral("[项目] 已打开: %1（%2 张图像, 稀疏 %3 点, frame=%4）")
                .arg(projectDir)
                .arg(m.imageFiles.size())
                .arg(lastOutput->sparseCloud
                         ? static_cast<int>(lastOutput->sparseCloud->size())
                         : 0)
                .arg(frame == CloudFileFrame::OpenCv ? QStringLiteral("opencv")
                                                     : QStringLiteral("display")));
    }
    return true;
}

} // namespace sfm
