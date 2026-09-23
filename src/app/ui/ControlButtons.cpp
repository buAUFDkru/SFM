/**
 * @file ControlButtons.cpp
 * @brief 重建控制按钮实现
 */

#include "ControlButtons.h"

#include <QHBoxLayout>
#include <QPushButton>

ControlButtons::ControlButtons(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

void ControlButtons::setupUI()
{
    QHBoxLayout *layout = new QHBoxLayout(this);

    m_startBtn = new QPushButton(tr("▶ 开始重建"), this);
    m_startBtn->setObjectName("startBtn");
    m_startBtn->setStyleSheet(
        "QPushButton {"
        "  background-color: #4CAF50;"
        "  color: white;"
        "  font-weight: bold;"
        "  padding: 8px 16px;"
        "  border-radius: 4px;"
        "}"
        "QPushButton:hover { background-color: #45a049; }"
        "QPushButton:disabled { background-color: #cccccc; }"
    );
    layout->addWidget(m_startBtn);

    m_stopBtn = new QPushButton(tr("■ 停止"), this);
    m_stopBtn->setObjectName("stopBtn");
    m_stopBtn->setStyleSheet(
        "QPushButton {"
        "  background-color: #f44336;"
        "  color: white;"
        "  font-weight: bold;"
        "  padding: 8px 16px;"
        "  border-radius: 4px;"
        "}"
        "QPushButton:hover { background-color: #da190b; }"
        "QPushButton:disabled { background-color: #cccccc; }"
    );
    m_stopBtn->setEnabled(false);
    layout->addWidget(m_stopBtn);

    connect(m_startBtn, &QPushButton::clicked, this, [this]() {
        setRunning(true);
        emit startClicked();
    });
    connect(m_stopBtn, &QPushButton::clicked, this, [this]() {
        setRunning(false);
        emit stopClicked();
    });
}

void ControlButtons::setRunning(bool running)
{
    m_running = running;
    m_startBtn->setEnabled(!running);
    m_stopBtn->setEnabled(running);
}