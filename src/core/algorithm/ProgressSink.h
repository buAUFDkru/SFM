#ifndef PROGRESSSINK_H
#define PROGRESSSINK_H

/**
 * @file ProgressSink.h
 * @brief 进度与日志出口（ADR-003），与 CancellationToken 成对传递
 */

#include <QString>

namespace sfm {

class ProgressSink
{
public:
    virtual ~ProgressSink() = default;

    virtual void onStage(const QString &stage, int percent) = 0;
    virtual void onLog(const QString &message) = 0;

    /// 强制刷出已缓冲的日志（阶段边界 / 流水线结束）
    virtual void flush() {}
};

} // namespace sfm

#endif
