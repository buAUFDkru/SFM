#ifndef SIFTDESCRIPTORWORKER_H
#define SIFTDESCRIPTORWORKER_H

#include <QJsonObject>

#include "plugin/IDescriptorPlugin.h"

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

#endif // SIFTDESCRIPTORWORKER_H