#ifndef LOGCONSOLE_H
#define LOGCONSOLE_H

#include <QPlainTextEdit>

/**
 * @brief 日志输出面板
 *
 * 继承自 QPlainTextEdit，封装重建流水线中使用的日志输出功能。
 * 支持：
 *   - 追加普通信息、警告、错误信息（带颜色区分）
 *   - 清空日志
 *   - 限制最大行数，防止长时间运行后占用过多内存
 *   - 批量刷新（appendBatch）：一次追加整批日志，减少重绘
 */
class LogConsole : public QPlainTextEdit
{
    Q_OBJECT

public:
    explicit LogConsole(QWidget *parent = nullptr);

public slots:
    /// 追加一条普通信息（浅色）
    void appendInfo(const QString &text);
    /// 追加一条警告（黄色）
    void appendWarning(const QString &text);
    /// 追加一条错误（红色）
    void appendError(const QString &text);
    /// 批量追加普通信息（用作 PipelineRunner::logBatchEmitted 接入）
    void appendBatch(const QStringList &messages);

    /// 清空日志
    void clearLog();

private:
    /// 将 HTML 格式化文本追加到面板，并限制最大行数
    void appendHtmlLine(const QString &html);

    static constexpr int MaxLines = 5000;  ///< 日志最大保留行数
};

#endif // LOGCONSOLE_H