#include <QApplication>
#include <QFile>
#include <QSurfaceFormat>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

#include "ui/MainWindow.h"

int main(int argc, char *argv[])
{
    // 尽早禁用 OpenCV 内部多线程，避免与项目 ThreadPool 叠加导致线程爆炸/崩溃。
    // 必须在任何 OpenCV 调用之前设置，因此放在 main 开头。
    cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);

    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("SFM Reconstructor"));
    app.setApplicationVersion(QStringLiteral("3.0.0"));
    app.setOrganizationName(QStringLiteral("SFMReconstructor"));

    // 全局样式
    QFile styleFile(QStringLiteral(":/style.qss"));
    if (styleFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        app.setStyleSheet(QString::fromUtf8(styleFile.readAll()));
        styleFile.close();
    }

    // OpenGL 3.3 Core + 深度/模板缓冲
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setSamples(4);
    QSurfaceFormat::setDefaultFormat(format);

    // 支持命令行插件目录覆盖：SFMReconstructor.exe [pluginsDir]
    const QString pluginsDir = (argc > 1)
                                   ? QString::fromLocal8Bit(argv[1])
                                   : QApplication::applicationDirPath() + "/plugins";

    MainWindow window(pluginsDir);
    window.resize(1260, 760);
    window.show();

    return app.exec();
}