#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

/// P1 占位主窗口：仅用于验证 app 可启动。后续在 P3/P4 完善界面。
class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
};

#endif // MAINWINDOW_H