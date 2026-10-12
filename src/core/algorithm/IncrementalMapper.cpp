#include "IncrementalMapper.h"

#include <algorithm>

namespace sfm {

bool IncrementalMapper::isBetterAttempt(int regCount, size_t pointCount,
                                        int bestReg, size_t bestPts)
{
    if (regCount > bestReg)
        return true;
    if (regCount == bestReg && pointCount > bestPts)
        return true;
    return false;
}

IncrementalMapperResult IncrementalMapper::run(
    Triangulate &tri,
    const std::vector<InitResult> &candidates,
    const std::vector<MatchPair> &matchPairs,
    const std::vector<FeatureResult> &features,
    const ViewGraph &viewGraph,
    const cv::Mat &K,
    const IncrementalMapperConfig &config,
    CancellationToken *cancel,
    const std::function<void(const QString &)> &log)
{
    IncrementalMapperResult out;
    const auto emitLog = [&](const QString &msg) {
        out.messages.append(msg);
        if (log)
            log(msg);
    };

    if (candidates.empty()) {
        emitLog(QStringLiteral("错误: 未找到合适的初始化图像对，终止重建"));
        return out;
    }

    tri.setOptions(config.triOptions);
    if (config.applyTriangulatorOptions)
        tri.setTriangulatorOptions(config.triangulatorOptions);

    const int acceptReg = std::max(3, config.acceptRegCount);
    const int numImages = static_cast<int>(features.size());
    bool shortCircuit = false;

    for (size_t t = 0; t < candidates.size(); ++t) {
        const InitResult &initResult = candidates[t];
        emitLog(QStringLiteral("[初始化] 尝试种子对 #%1: 图像 #%2 ↔ #%3, 评分=%4")
                    .arg(t + 1)
                    .arg(initResult.imageId1)
                    .arg(initResult.imageId2)
                    .arg(initResult.score, 0, 'f', 4));

        std::vector<Point3D> ptsTry;
        std::vector<CameraPose> posesTry;
        const bool okTry = tri.triangulate(
            matchPairs, features, initResult, viewGraph, ptsTry, posesTry,
            config.fixIntrinsics, K);

        int regCount = 0;
        for (const auto &p : posesTry) {
            if (p.registered)
                ++regCount;
        }

        emitLog(QStringLiteral("[稀疏重建] 第 %1 次尝试: 注册相机 %2/%3, 三维点 %4")
                    .arg(t + 1)
                    .arg(regCount)
                    .arg(numImages)
                    .arg(ptsTry.size()));

        if (okTry && regCount > 0) {
            if (isBetterAttempt(regCount, ptsTry.size(),
                                out.bestRegCount, out.bestPointCount)) {
                out.points = std::move(ptsTry);
                out.poses = std::move(posesTry);
                out.bestRegCount = regCount;
                out.bestPointCount = out.points.size();
                out.bestTrialIndex = static_cast<int>(t);
            }
            if (regCount >= acceptReg) {
                shortCircuit = true;
                break;
            }
        }

        if (cancel && cancel->isStopRequested()) {
            emitLog(QStringLiteral("[稀疏重建] 收到停止请求"));
            out.cancelled = true;
            return out;
        }
    }

    // 未能短路到主分量全注册时，≥3 台相机仍可接受
    out.ok = shortCircuit || (out.bestRegCount >= 3);
    if (!out.ok) {
        emitLog(QStringLiteral("错误: 三角测量失败（%1 组种子对均未能重建出足够相机，"
                               "最佳注册数=%2）")
                    .arg(candidates.size())
                    .arg(out.bestRegCount));
    }

    return out;
}

} // namespace sfm
