#ifndef FEATUREEXTRACTIONSTAGE_H
#define FEATUREEXTRACTIONSTAGE_H

/**
 * @file FeatureExtractionStage.h
 * @brief 阶段 1/6：描述符插件批量特征提取（F13）
 */

#include <vector>

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include "algorithm/CancellationToken.h"
#include "data/SfmTypes.h"
#include "pipeline/StageCallbacks.h"

namespace sfm { class ThreadPool; }

namespace sfm {

struct FeatureExtractionInput {
    QStringList imageFiles;
    QString descriptorType;
    QJsonObject descriptorParams;
};

class FeatureExtractionStage
{
public:
    /// 成功时 outFeatures 与 imageFiles 等长；失败或取消返回 false。
    /// 自行输出阶段 1/6 起止日志与进度（progress 可空）。
    static bool run(const FeatureExtractionInput &in,
                    std::vector<FeatureResult> *outFeatures,
                    ThreadPool *threadPool,
                    CancellationToken *cancel,
                    const StageLogFn &log,
                    const StageProgressFn &progress = {});
};

} // namespace sfm

#endif // FEATUREEXTRACTIONSTAGE_H
