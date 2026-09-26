#ifndef BUILTINDESCRIPTORS_H
#define BUILTINDESCRIPTORS_H

#include <QJsonObject>

#include "plugin/IDescriptorPlugin.h"

/**
 * @file BuiltinDescriptors.h
 * @brief OpenCV 自带描述符算法的内置实现（编译进 core，非 DLL 插件）。
 *
 * 与 DLL 插件实现相同的 IDescriptorPlugin 接口，经 PluginRegistry::registerBuiltIn
 * 注册后，管线与 UI 无差别使用。Worker 并发模型不变：逐线程 / 任务独占。
 */

// ── SIFT ────────────────────────────────────────────────────────────────────

/**
 * @brief SIFT 描述符 Worker。
 *
 * 构造时从 params 读取参数并持有，extract() 创建 SIFT 检测器完成提取。
 * 逐线程 / 任务独占，禁止跨线程共享使用。
 */
class SiftDescriptorWorker : public IDescriptorWorker
{
public:
    explicit SiftDescriptorWorker(const QJsonObject &params);

    FeatureResult extract(const ImageEntry &image) override;

private:
    int m_nOctaveLayers = 3;
    double m_contrastThreshold = 0.04;
    double m_edgeThreshold = 10.0;
    double m_sigma = 1.6;
    int m_maxFeatures = 8192;
};

/// SIFT 内置算法（无状态工厂）
class SiftDescriptorPlugin : public IDescriptorPlugin
{
    Q_OBJECT
    // Qt 6 的 qobject_cast<IDescriptorPlugin*> 按 IID 查找，
    // Q_INTERFACES 生成对应 qt_metacast 分支（无需 Q_PLUGIN_METADATA）
    Q_INTERFACES(IDescriptorPlugin)
public:
    QString name() const override { return QStringLiteral("SIFT"); }
    QString category() const override { return QStringLiteral("descriptor"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于尺度不变特征变换(SIFT)的关键点检测与描述子提取");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IDescriptorWorker> createWorker(const QJsonObject &params) override;
};

// ── SURF ────────────────────────────────────────────────────────────────────

/**
 * @brief SURF 描述符 Worker。
 *
 * 构造时从 params 读取参数并持有，extract() 创建 SURF 检测器完成提取。
 * 逐线程 / 任务独占，禁止跨线程共享使用。
 */
class SurfDescriptorWorker : public IDescriptorWorker
{
public:
    explicit SurfDescriptorWorker(const QJsonObject &params);

    FeatureResult extract(const ImageEntry &image) override;

private:
    double m_hessianThreshold = 100.0;
    int m_nOctaves = 4;
    int m_nOctaveLayers = 3;
    bool m_extended = false;
    bool m_upright = false;
    int m_maxFeatures = 8192;
};

/// SURF 内置算法（无状态工厂）
class SurfDescriptorPlugin : public IDescriptorPlugin
{
    Q_OBJECT
    Q_INTERFACES(IDescriptorPlugin)
public:
    QString name() const override { return QStringLiteral("SURF"); }
    QString category() const override { return QStringLiteral("descriptor"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于加速稳健特征(SURF)的关键点检测与描述子提取");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IDescriptorWorker> createWorker(const QJsonObject &params) override;
};

// ── KAZE ────────────────────────────────────────────────────────────────────

/**
 * @brief KAZE 描述符 Worker。
 *
 * 构造时从 params 读取参数并持有，extract() 创建 KAZE 检测器完成提取。
 * 逐线程 / 任务独占，禁止跨线程共享使用。
 */
class KazeDescriptorWorker : public IDescriptorWorker
{
public:
    explicit KazeDescriptorWorker(const QJsonObject &params);

    FeatureResult extract(const ImageEntry &image) override;

private:
    bool m_extended = false;
    bool m_upright = false;
    double m_threshold = 0.001;
    int m_nOctaves = 4;
    int m_nOctaveLayers = 4;
    int m_diffusivity = 1;
    int m_maxFeatures = 8192;
};

/// KAZE 内置算法（无状态工厂）
class KazeDescriptorPlugin : public IDescriptorPlugin
{
    Q_OBJECT
    Q_INTERFACES(IDescriptorPlugin)
public:
    QString name() const override { return QStringLiteral("KAZE"); }
    QString category() const override { return QStringLiteral("descriptor"); }
    QString version() const override { return QStringLiteral("1.0.0"); }
    QString description() const override
    {
        return QStringLiteral("基于非线性尺度空间(KAZE)的关键点检测与描述子提取");
    }

    QList<ParamSchema> paramSchema() const override;
    bool validateParams(const QJsonObject &params, QString *errorMsg = nullptr) const override;
    std::unique_ptr<IDescriptorWorker> createWorker(const QJsonObject &params) override;
};

#endif // BUILTINDESCRIPTORS_H
