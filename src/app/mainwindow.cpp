#include "mainwindow.h"

#include <QLabel>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    auto *label = new QLabel(QStringLiteral("SFM 3.0 — P1 骨架"), this);
    label->setAlignment(Qt::AlignCenter);
    setCentralWidget(label);
    resize(800, 500);
}