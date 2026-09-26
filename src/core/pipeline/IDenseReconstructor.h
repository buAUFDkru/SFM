#ifndef IDENSERECONSTRUCTOR_H
#define IDENSERECONSTRUCTOR_H

#include <QString>
#include <QMap>
#include <QVariant>
#include <functional>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "data/SfmTypes.h"
#include "data/StereoTypes.h"

/**
 * @brief 密集重建统一输入数据
 *
 * 所有密集重建方法共享的输入数据，由各阶段结果汇总而成。
 */
struct DenseReconstructionInput
{
    std::vector<CameraPose> cameraPoses;   ///< 相机位姿（世界→相机）
    std::vector<MatchPair> matchPairs;     ///< 稀疏匹配对
    std::vector<FeatureResult> features;   ///< 特征提取结果
    cv::Mat cameraMatrix;                  ///< 相机内参（可能为空，由方法自行处理）
    QStringList imageFiles;                ///< 原始图像路径列表
    QString projectPath;                   ///< 项目输出路径
};

/**
 * @brief 密集重建统一输出数据
 *
 * 各方法至少应生成密集点云；stereoResults 等可选字段用于兼容现有流程。
 */
struct DenseReconstructionOutput
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr denseCloud;  ///< 密集点云
    std::vector<StereoPairResult> stereoResults;        ///< 可选：深度图/视差图结果
    bool success = false;                               ///< 是否成功
    QString errorMessage;                               ///< 失败时的错误信息
};

/**
 * @brief 密集重建方法抽象接口
 *
 * 所有密集重建算法（如 SGBM、MVS、NeRF 等）均实现此接口。
 * 通过 DenseReconstructionFactory 按名称创建实例。
 *
 * 日志回调：通过 setLogCallback() 设置，重建过程中的进度日志通过回调实时输出。
 * 这对长时间运行的密集重建（如 SGBM 立体匹配可能耗时数十分钟）尤为重要，
 * 避免整个阶段无任何日志输出导致用户误以为程序卡死。
 *
 * 线程池：使用 sfm::globalThreadPool() 进程级全局共享池，
 * 与稀疏重建等其他阶段共用，避免多套线程池叠加导致线程超订。
 */
class IDenseReconstructor
{
public:
    /// 日志回调类型，接收一条日志文本
    using LogCallback = std::function<void(const QString &)>;

    virtual ~IDenseReconstructor() = default;

    /**
     * @brief 方法唯一标识名（用于配置和工厂查找）
     */
    virtual QString methodName() const = 0;

    /**
     * @brief 方法显示名称（用于 UI 展示）
     */
    virtual QString displayName() const = 0;

    /**
     * @brief 设置方法特有参数
     * @param params 方法参数表（由对话框或配置文件提供）
     */
    virtual void setConfig(const QMap<QString, QVariant> &params) = 0;

    /**
     * @brief 设置日志回调
     * @param callback 日志回调函数，为空时表示不输出日志
     *
     * 实现应在 reconstruct() 过程中通过 callback 输出进度日志，
     * 让 UI 能实时显示重建进度。
     */
    virtual void setLogCallback(LogCallback callback) { m_logCallback = std::move(callback); }

    /**
     * @brief 执行密集重建
     * @param input 输入数据（位姿、匹配、图像等）
     * @return 重建结果
     */
    virtual DenseReconstructionOutput reconstruct(const DenseReconstructionInput &input) = 0;

protected:
    LogCallback m_logCallback;

    /// 便捷方法：若有日志回调则输出日志
    void log(const QString &message) const
    {
        if (m_logCallback) m_logCallback(message);
    }
};

using IDenseReconstructorPtr = std::unique_ptr<IDenseReconstructor>;

#endif // IDENSERECONSTRUCTOR_H
