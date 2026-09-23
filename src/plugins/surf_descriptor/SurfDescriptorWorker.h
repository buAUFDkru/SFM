#ifndef SURFDESCRIPTORWORKER_H
#define SURFDESCRIPTORWORKER_H

#include <QJsonObject>

#include "plugin/IDescriptorPlugin.h"

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
    int m_maxFeatures = 0;
};

#endif // SURFDESCRIPTORWORKER_H