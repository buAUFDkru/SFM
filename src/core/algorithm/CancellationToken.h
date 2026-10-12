#ifndef CANCELLATIONTOKEN_H
#define CANCELLATIONTOKEN_H

/**
 * @file CancellationToken.h
 * @brief 跨线程取消令牌（ADR-003）
 *
 * UI 线程 requestStop()，工作线程 isStopRequested() 轮询。
 * 使用 std::atomic 消除原 bool m_stopRequested 的数据竞争 UB。
 */

#include <atomic>

namespace sfm {

class CancellationToken
{
public:
    void requestStop() noexcept
    {
        m_stop.store(true, std::memory_order_relaxed);
    }

    bool isStopRequested() const noexcept
    {
        return m_stop.load(std::memory_order_relaxed);
    }

    /// 开始新一轮流水线前清除停止标志
    void reset() noexcept
    {
        m_stop.store(false, std::memory_order_relaxed);
    }

private:
    std::atomic<bool> m_stop{false};
};

} // namespace sfm

#endif // CANCELLATIONTOKEN_H
