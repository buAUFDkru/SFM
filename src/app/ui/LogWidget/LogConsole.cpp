/**
 * @file LogConsole.cpp
 * @brief 日志输出面板实现
 */

#include "LogConsole.h"

#include <QDateTime>
#include <QScrollBar>

LogConsole::LogConsole(QWidget *parent)
    : QPlainTextEdit(parent)
{
    setReadOnly(true);
    setLineWrapMode(QPlainTextEdit::WidgetWidth);
    setMaximumBlockCount(MaxLines);

    setStyleSheet(
        "QPlainTextEdit {"
        "  background-color: #1e1e1e;"
        "  color: #d4d4d4;"
        "  font-family: Consolas, 'Courier New', monospace;"
        "  font-size: 12px;"
        "  border: none;"
        "}"
    );
}

void LogConsole::appendInfo(const QString &text)
{
    appendHtmlLine(QString("<span style=\"color:#d4d4d4;\">%1</span>")
                       .arg(text.toHtmlEscaped()));
}

void LogConsole::appendWarning(const QString &text)
{
    appendHtmlLine(QString("<span style=\"color:#ffcc00;\">%1</span>")
                       .arg(text.toHtmlEscaped()));
}

void LogConsole::appendError(const QString &text)
{
    appendHtmlLine(QString("<span style=\"color:#ff6666;\">%1</span>")
                       .arg(text.toHtmlEscaped()));
}

void LogConsole::appendBatch(const QStringList &messages)
{
    for (const QString &msg : messages)
        appendInfo(msg);
}

void LogConsole::appendHtmlLine(const QString &html)
{
    const QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
    appendHtml(QString("[%1] %2").arg(timestamp, html));

    QScrollBar *bar = verticalScrollBar();
    bar->setValue(bar->maximum());
}

void LogConsole::clearLog()
{
    clear();
}