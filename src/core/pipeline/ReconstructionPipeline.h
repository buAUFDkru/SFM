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

#include "data/ReconstructionOutput.h"
#include "data/SfmTypes.h"
#include "data/StereoTypes.h"

/**
 * @brief 重建配置结构体
 *
 * 描述符/匹配器以插件 id 命名，参数通过 QJsonObject 透传（与插件 ParamSchema 对齐）。
 */
struct ReconstructionConfig
{
    QStringList imageFiles;
    QString projectPath;    ///< 项目路径（输出文件保存路径）

    // ── 密集重建阶段配置（方法实现于 P4，接口先占位） ──
    QString denseReconstructionMethod;          ///< 密集重建方法名（空表示在稀疏重建后询问用户）
    QMap<QString, QVariant> denseReconstructionParams; ///< 方法特有参数

    QString descriptorType = "SIFT";            ///< 描述符插件 id
    QJsonObject descriptorParams{
        {"nOctaveLayers", 3},
        {"contrastThreshold", 0.04},
        {"maxFeatures", 8192},
    };

    QString matcherType = "FLANN";              ///< 匹配器插件 id
    QJsonObject matcherParams{
        {"ratioThreshold", 0.8},
        {"ransacThreshold", 3.0},
        {"minMatchCount", 50},
        {"minMatchRatio", 0.05},
    };

    int threadCount = 0;  ///< 线程池线程数；0 表示使用物理核心数近似值

    int baMaxIterations = 50;   ///< 保留字段（sfm2.0 兼容），BA 内部使用固定迭代
    double voxelSize = 0.01;    ///< 保留字段（点云下采样粒度）
    bool useGpu = false;

    cv::Mat cameraMatrix;  ///< 外部提供的相机内参矩阵（空则自动估计）
    bool fixCameraMatrix = false;  ///< 若从cameramatrix.txt读取，则全局BA不优化内参
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

    ReconstructionOutput execute();

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

    void saveProject(const QString &filePath, const ReconstructionOutput &output);
    void saveProject(const QString &filePath);

    const ReconstructionOutput &lastOutput() const { return m_lastOutput; }

    /**
     * @brief 设置用户选择的密集重建方法（由 UI 在收到请求后调用）
     */
    void setDenseReconstructionMethod(const QString &method,
                                      const QMap<QString, QVariant> &params);

private:
    void log(const QString &message);
    void setProgress(const QString &stage, int percent);

    /// 获取共享线程池（稀疏/密集重建共用）；不存在时按 m_config.threadCount 创建
    sfm::ThreadPool *ensureThreadPool();

    /// 发射阶段性点云信号
    void emitStageCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                        const QString &stageName);

    /// 将点云保存到指定路径的 PLY 文件（统一保存入口，PLY 二进制格式）
    void saveStageCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                        const QString &filePath,
                        const QString &stageLabel);

    /// 构造项目输出目录下的完整文件路径
    QString resolveProjectFilePath(const QString &fileName) const;

    /// 在稀疏重建完成后等待 UI 选择密集重建方法
    void waitForDenseReconstructionMethod();

    /// 执行密集重建阶段 5-6（被 execute 和 executeDenseReconstruction 共用）
    void runDenseStages(ReconstructionOutput &output);

    /// 通过描述符插件从磁盘图像文件批量提取特征（支持线程池并行）
    bool extractFeaturesViaPlugins(sfm::ThreadPool *threadPool);

    IReconstructionCallbacks *m_callbacks = nullptr;
    ReconstructionConfig m_config;
    ReconstructionOutput m_lastOutput;
    bool m_stopRequested = false;

    // 共享线程池：execute() 每轮重建重建一次，稀疏与密集阶段复用同一实例
    std::unique_ptr<sfm::ThreadPool> m_threadPool;

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
};

#endif // RECONSTRUCTIONPIPELINE_H