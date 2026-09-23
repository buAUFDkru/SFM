#ifndef CONTROLBUTTONS_H
#define CONTROLBUTTONS_H

#include <QWidget>

class QPushButton;

/**
 * @brief 重建开始/停止控制按钮组
 *
 * 提供两个状态互斥的按钮：
 *   - 开始重建（绿色）：未运行时可用
 *   - 停止（红色）：运行时可用
 *
 * 通过 setRunning() 统一切换按钮可用状态，并对外发射 startClicked / stopClicked 信号。
 */
class ControlButtons : public QWidget
{
    Q_OBJECT

public:
    explicit ControlButtons(QWidget *parent = nullptr);

    /// 设置当前运行状态：running=true 时禁用开始按钮、启用停止按钮
    void setRunning(bool running);

signals:
    void startClicked();
    void stopClicked();

private:
    void setupUI();

    QPushButton *m_startBtn = nullptr;
    QPushButton *m_stopBtn = nullptr;
    bool m_running = false;
};

#endif // CONTROLBUTTONS_H