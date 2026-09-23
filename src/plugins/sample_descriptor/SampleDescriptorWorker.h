#ifndef SAMPLEDESCRIPTORWORKER_H
#define SAMPLEDESCRIPTORWORKER_H

#include <QJsonObject>

#include "plugin/IDescriptorPlugin.h"

/**
 * @brief 示例描述符 Worker。
 *
 * extract() 是输入图像的确定性纯函数（伪特征），刻意不依赖真实算法，
 * 以便并发一致性测试可断言"并行结果 == 串行结果"。
 */
class SampleDescriptorWorker : public IDescriptorWorker
{
public:
    explicit SampleDescriptorWorker(const QJsonObject &params);

    FeatureResult extract(const ImageEntry &image) override;

private:
    double m_scale = 1.0;
};

#endif // SAMPLEDESCRIPTORWORKER_H