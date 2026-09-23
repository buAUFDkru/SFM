#ifndef IDESCRIPTORPLUGIN_H
#define IDESCRIPTORPLUGIN_H

#include <QJsonObject>
#include <memory>

#include "ISfmPlugin.h"
#include "PluginTypes.h"

/**
 * @brief 有状态算法 Worker：内部持有算法对象（如 cv::Ptr<cv::SIFT>）。
 *
 * 每个线程 / 任务必须通过工厂 createWorker() 获取独占实例，
 * 严禁多个线程共享同一 Worker 调用 extract()。
 */
class IDescriptorWorker
{
public:
    virtual ~IDescriptorWorker() = default;

    /// 提取单张图像特征；仅允许单线程调用
    virtual FeatureResult extract(const ImageEntry &image) = 0;
};

/**
 * @brief 描述符插件（无状态工厂）。
 *
 * createWorker() 依据 params 校验并创建独立的 Worker。工厂本身不持有
 * 算法对象，可被多线程安全并发调用。
 */
class IDescriptorPlugin : public ISfmPlugin
{
    Q_OBJECT
public:
    explicit IDescriptorPlugin(QObject *parent = nullptr) : ISfmPlugin(parent) {}

    virtual std::unique_ptr<IDescriptorWorker> createWorker(const QJsonObject &params) = 0;
};

Q_DECLARE_INTERFACE(IDescriptorPlugin, "com.sfm3.descriptor/1.0")

#endif // IDESCRIPTORPLUGIN_H