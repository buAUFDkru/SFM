#include "SfmLogger.h"

#include <QDebug>
#include <QDateTime>

SfmLogger &SfmLogger::instance()
{
    static SfmLogger logger;
    return logger;
}

void SfmLogger::setDefaultCallback(LogCallback callback)
{
    QMutexLocker lock(&m_mutex);
    m_callbacks.clear();
    if (callback) {
        m_callbacks.push_back(std::move(callback));
    }
}

void SfmLogger::addCallback(LogCallback callback)
{
    if (!callback) return;
    QMutexLocker lock(&m_mutex);
    m_callbacks.push_back(std::move(callback));
}

void SfmLogger::clearCallbacks()
{
    QMutexLocker lock(&m_mutex);
    m_callbacks.clear();
}

void SfmLogger::log(const QString &message)
{
    QString timestamped = QString("[%1] %2")
                              .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
                              .arg(message);

    QMutexLocker lock(&m_mutex);
    // 复制回调列表，避免在回调执行期间被修改导致问题
    std::vector<LogCallback> callbacks = m_callbacks;
    QtMessageHandler original = m_originalHandler;
    lock.unlock();

    if (callbacks.empty()) {
        // 无回调时回退到原始 Qt 消息处理器，避免直接 qDebug 导致递归
        // （因为当前可能已安装为消息处理器）
        if (original) {
            original(QtDebugMsg, QMessageLogContext(), timestamped);
        } else {
            qDebug().noquote() << timestamped;
        }
        return;
    }

    for (const auto &cb : callbacks) {
        if (cb) cb(timestamped);
    }
}

void SfmLogger::installAsMessageHandler()
{
    QMutexLocker lock(&m_mutex);
    if (m_installed) return;
    m_originalHandler = qInstallMessageHandler(SfmLogger::messageHandler);
    m_installed = true;
}

void SfmLogger::uninstallAsMessageHandler()
{
    QMutexLocker lock(&m_mutex);
    if (!m_installed) return;
    qInstallMessageHandler(m_originalHandler);
    m_originalHandler = nullptr;
    m_installed = false;
}

void SfmLogger::messageHandler(QtMsgType type, const QMessageLogContext &context,
                               const QString &msg)
{
    Q_UNUSED(type)
    Q_UNUSED(context)

    SfmLogger &logger = SfmLogger::instance();

    QString timestamped = QString("[%1] %2")
                              .arg(QDateTime::currentDateTime().toString("HH:mm:ss"))
                              .arg(msg);

    QMutexLocker lock(&logger.m_mutex);
    bool installed = logger.m_installed;
    QtMessageHandler original = logger.m_originalHandler;
    std::vector<LogCallback> callbacks = logger.m_callbacks;
    lock.unlock();

    if (callbacks.empty()) {
        // 没有 UI 回调时，转发给原始 Qt 消息处理器（恢复默认输出行为）
        if (installed && original) {
            original(type, context, msg);
        } else {
            // 未安装时的默认行为：直接输出到 stderr/调试器
            qDebug().noquote() << timestamped;
        }
        return;
    }

    for (const auto &cb : callbacks) {
        if (cb) cb(timestamped);
    }
}
