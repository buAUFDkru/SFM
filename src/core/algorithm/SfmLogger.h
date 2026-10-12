#ifndef SFMLOGGER_H
#define SFMLOGGER_H

#include <QString>
#include <QMutex>
#include <functional>
#include <vector>

/**
 * @brief 项目统一日志器
 *
 * 提供线程安全的统一日志输出入口。推荐所有模块通过 SfmLogger::instance().log()
 * 输出日志，同时通过 qInstallMessageHandler() 捕获已有的 qDebug/qWarning 输出。
 *
 * 特点：
 *   - 线程安全：内部使用 QMutex 保护回调列表
 *   - 回调机制：可注册多个日志回调，通常 UI 注册一个回调将日志显示到 LogConsole
 *   - 无回调时回退到原始 Qt 消息处理（通常是控制台/调试器输出）
 *   - installAsMessageHandler / uninstallAsMessageHandler 采用引用计数
 *     （ADR-003）：多个 ReconstructionPipeline 并存时，最后一个 uninstall 才真正卸下
 *
 * 使用示例：
 *   SfmLogger::instance().log("[模块] 消息");
 *   SfmLogger::instance().installAsMessageHandler();
 */
class SfmLogger
{
public:
    using LogCallback = std::function<void(const QString &)>;

    static SfmLogger &instance();

    /// 设置默认日志回调（通常由 ReconstructionPipeline 设置为 onLog 回调）
    /// 旧的默认回调会被替换
    void setDefaultCallback(LogCallback callback);

    /// 添加日志回调（观察者模式，允许多个接收者）
    void addCallback(LogCallback callback);

    /// 清空所有日志回调
    void clearCallbacks();

    /// 输出一条日志，所有回调都会收到
    void log(const QString &message);

    /// 安装为 Qt 全局消息处理器（引用计数 +1；首次调用才真正 qInstall）
    void installAsMessageHandler();

    /// 卸载请求（引用计数 -1；减到 0 时才恢复默认处理器）
    void uninstallAsMessageHandler();

    /// 当前消息处理器安装引用计数（测试 / 多实例析构用）
    int messageHandlerInstallCount() const;

    /// 便捷模板：支持可打印类型
    template<typename T>
    void log(const T &value)
    {
        log(QString::number(value));
    }

private:
    SfmLogger() = default;
    ~SfmLogger() = default;
    SfmLogger(const SfmLogger &) = delete;
    SfmLogger &operator=(const SfmLogger &) = delete;

    static void messageHandler(QtMsgType type, const QMessageLogContext &context,
                               const QString &msg);

    mutable QMutex m_mutex;
    std::vector<LogCallback> m_callbacks;
    QtMessageHandler m_originalHandler = nullptr;
    int m_installCount = 0;
};

/// 全局便捷函数
inline void sfmLog(const QString &message)
{
    SfmLogger::instance().log(message);
}

#endif // SFMLOGGER_H
