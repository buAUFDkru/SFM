#ifndef KAZEDESCRIPTORWORKER_H
#define KAZEDESCRIPTORWORKER_H

#include <QJsonObject>

#include "plugin/IDescriptorPlugin.h"

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
    int m_maxFeatures = 0;
};

#endif // KAZEDESCRIPTORWORKER_H