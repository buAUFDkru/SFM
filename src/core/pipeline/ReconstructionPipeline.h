#ifndef RECONSTRUCTIONPIPELINE_H
#define RECONSTRUCTIONPIPELINE_H

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVariant>
#include <QVector>

#include <condition_variable>
#include <memory>
#include <mutex>

#include "algorithm/CancellationToken.h"
#include "algorithm/ProgressSink.h"
#include "data/SfmTypes.h"
#include "data/StereoTypes.h"
#include "pipeline/ProjectStore.h"
#include "viewmodel/RenderModel.h"

class Triangulate;
struct ReconstructionOutput; ///< core 内部类型（含 PCL），见 ReconstructionOutput.h

/**
 * @brief 重建配置结构体
 *
 * 描述符/匹配器以插件 id 命名，参数通过 QJsonObject 透传（与插件 ParamSchema 对齐）。
 */
struct ReconstructionConfig
{
    QStringList imageFiles;
    QString projectPath;    ///< 项目根目录（输出文件保存路径）
    QString projectName;    ///< 项目名称（清单文件 <name>.sfmproj；空表示未开项目）

    // ── 密集重建阶段配置（方法实现于 P4，接口先占位） ──
    QString denseReconstructionMethod;          ///< 密集重建方法名（空表示在稀疏重建后询问用户）
    QMap<QString, QVariant> denseReconstructionParams; ///< 方法特有参数

    /// 描述符/匹配器调优默认值见 presets/*.json（ADR-002），经
    /// makeDefaultReconstructionConfig() / loadBuiltinReconstructionPreset 注入。
    QString descriptorType = "SIFT"; ///< 描述符插件 id（语言级缺省）
    QJsonObject descriptorParams;    ///< 由预设或 UI 填充

    QString matcherType = "FLANN"; ///< 匹配器插件 id（语言级缺省）
    QJsonObject matcherParams;     ///< 由预设或 UI 填充

    int threadCount = 0; ///< 线程池线程数；0 表示使用物理核心数近似值

    int baMaxIterations = 50; ///< 保留字段（sfm2.0 兼容），BA 内部使用固定迭代
    double voxelSize = 0.01;  ///< 保留字段（点云下采样粒度）
    bool useGpu = false;

    cv::Mat cameraMatrix;         ///< 外部提供的相机内参矩阵（空则自动估计）
    bool fixCameraMatrix = false; ///< 若从 cameramatrix.txt 读取，则全局 BA 不优化内参

    // ── 阶段五：相机模型（布尔语义缺省；业务组合见预设）────────────────
    bool useExifFocal = true;
    bool refineDistortion = false;
    bool refineFocalPerAxis = false;
};

namespace sfm { class ThreadPool; }

/**
 * @brief 重建回调接口
 *
 * ReconstructionPipeline 不再继承 QObject，所有需要与 UI 交互的事件
 * 都通过此回调接口转发。实现者负责线程安全（通常在实现中将回调转发到 UI 线程）。
 */
class IReconstructionCallbacks
{
public:
    virtual ~IReconstructionCallbacks() = default;

    virtual void onProgress(const QString &stage, int percent) = 0;
    virtual void onLog(const QString &message) = 0;
    virtual void onStagePointCloudReady(const QVector<float> &points,
                                        const QVector<float> &colors,
                                        const QString &stageName) = 0;

    /// 请求 UI 选择密集重建方法；实现者需要在合适的时机调用 setDenseReconstructionMethod
    virtual void onDenseReconstructionMethodRequested() = 0;
};

/**
 * @brief 重建流水线类
 *
 * 纯逻辑类，不继承 QObject，不绑定任何线程。
 * 通过 IReconstructionCallbacks 将事件输出给调用者。
 *
 * SFM重建流程（sfm3.0 插件化版）：
 * 1. 特征提取（IDescriptorPlugin 插件，逐任务独立 Worker）
 * 2. 特征匹配（MatcherService 编排 + IMatcherPlugin 低层匹配）
 * 3. 稀疏重建（Initialization + Triangulate + BundleAdjustment）
 * 4. 点云优化（PointCloudBuilder）
 * 5. 密集立体匹配（DenseReconstructionFactory，方法实现于 P4）
 * 6. 密集点云生成
 */
class ReconstructionPipeline
{
public:
    ReconstructionPipeline();
    ~ReconstructionPipeline();

    void setConfig(const ReconstructionConfig &config);
    ReconstructionConfig config() const;

    /// 设置回调接口（非线程安全，应在执行前设置）
    void setCallbacks(IReconstructionCallbacks *callbacks);

    /// 进度/日志出口（ADR-003）；优先于 IReconstructionCallbacks::onProgress/onLog
    void setProgressSink(sfm::ProgressSink *sink);
    sfm::ProgressSink *progressSink() const { return m_progress; }

    /// 执行完整流水线；结果写入 lastOutput() / lastView()
    void execute();

    /// 仅执行稀疏重建（阶段 1-4），结果保存在内部状态中
    bool executeSparseReconstruction();

    /// 仅执行密集重建（阶段 5-6），要求已先执行稀疏重建
    bool executeDenseReconstruction();

    /// 是否可以执行密集重建（已有稀疏重建结果）
    bool denseReconstructionAvailable() const;

    void requestStop();
    bool isStopRequested() const;

    void exportPointCloud(const QString &filePath, const ReconstructionOutput &output);
    void exportPointCloud(const QString &filePath);

    /**
     * @brief 保存项目到 m_config.projectPath（自动保存与手动保存共用）
     *
     * 写入点云（clouds/sparse.ply、clouds/dense.ply）、相机参数
     * （cameras/poses.json）、匹配对（matches/pairs.bin）与清单
     * （<projectName>.sfmproj）。projectName 为空时视为无项目，返回 false。
     */
    bool saveProject(QString *error = nullptr);

    /**
     * @brief 打开项目：读清单并恢复管线状态（图像/算法配置/相机位姿/匹配对）
     *
     * 恢复后 denseReconstructionAvailable() 为 true（清单含位姿时），
     * 可直接 executeDenseReconstruction() 续算密集重建；
     * 稀疏/密集点云从 PLY 加载到 lastOutput() 供 UI 显示。
     * @param path 清单文件路径（*.sfmproj）或项目目录
     */
    bool loadProject(const QString &path, QString *error = nullptr);

    /// 打开项目后的清单（未打开时无效，需先判断 loadProject 返回值）
    const ProjectManifest &manifest() const { return m_manifest; }

    /// core 内部输出（含 PCL）；UI 请用 lastView()
    const ReconstructionOutput &lastOutput() const;

    /// 渲染边界视图（无第三方类型，显示坐标系）
    sfm::view::ReconstructionView lastView() const;

    /// 稀疏/密集点云是否非空（不暴露 PCL）
    bool hasCloud(bool dense) const;

    /// 本流水线持有的三角化器（供 PipelineRunner 连接 incrementalUpdate）
    Triangulate &triangulator();
    const Triangulate &triangulator() const;

    /**
     * @brief 设置用户选择的密集重建方法（由 UI 在收到请求后调用）
     */
    void setDenseReconstructionMethod(const QString &method,
                                      const QMap<QString, QVariant> &params);

private:
    void log(const QString &message);
    void setProgress(const QString &stage, int percent);

    /// 构造项目输出目录下的完整文件路径
    QString resolveProjectFilePath(const QString &fileName) const;

    /// 在稀疏重建完成后等待 UI 选择密集重建方法
    void waitForDenseReconstructionMethod();

    /// 执行密集重建阶段 5-6（委托 DenseReconstructionStage）
    void runDenseStages(ReconstructionOutput &output);

    IReconstructionCallbacks *m_callbacks = nullptr;
    sfm::ProgressSink *m_progress = nullptr;
    /// 本实例是否已对 SfmLogger 做过一次 install（避免 setCallbacks 重复 +1）
    bool m_sfmLoggerInstalled = false;
    ReconstructionConfig m_config;
    std::unique_ptr<ReconstructionOutput> m_lastOutput;
    std::unique_ptr<Triangulate> m_triangulate; ///< 阶段三 DI：每流水线一份
    sfm::CancellationToken m_cancel;

    // 阶段暂停同步：等待 UI 选择密集重建方法
    std::mutex m_denseMethodMutex;
    std::condition_variable m_denseMethodCv;
    bool m_denseMethodReady = false;

    // 稀疏重建中间结果，供后续单独执行密集重建使用
    bool m_sparseReady = false;
    std::vector<CameraPose> m_cameraPoses;
    std::vector<MatchPair> m_matchPairs;
    std::vector<FeatureResult> m_features;
    cv::Mat m_K;
    std::vector<Point3D> m_optPoints;

    ProjectManifest m_manifest;  ///< 打开项目后的清单快照
};

#endif // RECONSTRUCTIONPIPELINE_H