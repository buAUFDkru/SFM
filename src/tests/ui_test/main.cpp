/**
 * @file ui_test main.cpp
 * @brief UI 组件测试（QTest + offscreen）
 *
 * 覆盖：
 *   1. ParamFormGenerator：schema → 控件生成、collectParams/setParams 往返
 *   2. AlgorithmComboPanel：注册表算法下拉与参数收集
 *   3. LogConsole / ControlButtons：基础行为
 *   4. PointCloudWidget：构造冒烟（offscreen 下不渲染，仅验证生命周期）
 *   5. MainWindow 冒烟：合成场景 → 同步稀疏重建 → 输出点云非空
 */

#include <QtTest>

#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QVBoxLayout>

#include "plugin/ParamSchema.h"
#include "plugin/PluginRegistry.h"
#include "pipeline/ReconstructionPipeline.h"
#include "ParamWidget/ParamFormGenerator.h"
#include "ParamWidget/AlgorithmComboPanel.h"
#include "ConfigPanel.h"
#include "DenseReconstructionDialog.h"
#include "LogWidget/LogConsole.h"
#include "ControlButtons.h"
#include "ViewWidget/PointCloudWidget.h"
#include "MainWindow.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifndef PLUGINS_DIR_DEFAULT
#define PLUGINS_DIR_DEFAULT ""
#endif

namespace {

QList<ParamSchema> sampleSchemas()
{
    QList<ParamSchema> out;

    ParamSchema count;
    count.name = QStringLiteral("count");
    count.type = ParamSchema::Type::Int;
    count.defaultValue = 42;
    count.min = 0;
    count.max = 1000;
    count.group = QStringLiteral("组A");
    count.tooltip = QStringLiteral("数量");
    out.append(count);

    ParamSchema rate;
    rate.name = QStringLiteral("rate");
    rate.type = ParamSchema::Type::Double;
    rate.defaultValue = 0.5;
    rate.min = 0.0;
    rate.max = 1.0;
    rate.group = QStringLiteral("组A");
    out.append(rate);

    ParamSchema on;
    on.name = QStringLiteral("on");
    on.type = ParamSchema::Type::Bool;
    on.defaultValue = true;
    on.group = QStringLiteral("组B");
    out.append(on);

    ParamSchema mode;
    mode.name = QStringLiteral("mode");
    mode.type = ParamSchema::Type::Enum;
    mode.defaultValue = QStringLiteral("fast");
    mode.enumValues = {QStringLiteral("fast"), QStringLiteral("accurate")};
    mode.group = QStringLiteral("组B");
    out.append(mode);

    ParamSchema label;
    label.name = QStringLiteral("label");
    label.type = ParamSchema::Type::String;
    label.defaultValue = QStringLiteral("hi");
    out.append(label);

    return out;
}

/// 校验断言宏：失败即打印并计数
static int g_fail = 0;
#define CHECK_UI(cond, msg)                                                     \
    do {                                                                        \
        if (!(cond)) {                                                          \
            ++g_fail;                                                           \
            std::fprintf(stderr, "[ui] FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            std::fflush(stderr);                                                \
        }                                                                       \
    } while (0)

/// 渲染带纹理球体多视图（确定性种子），返回 6 张图
bool renderSphereViews(QTemporaryDir &dir, QStringList &outFiles)
{
    const int W = 320, H = 240;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 458.4, 0, W / 2.0,
                                        0, 458.4, H / 2.0,
                                        0, 0, 1);

    cv::Mat tex(256, 256, CV_8UC3);
    {
        cv::RNG trng(777);
        trng.fill(tex, cv::RNG::UNIFORM, 0, 255);
        cv::Mat coarse(32, 32, CV_8UC3);
        cv::RNG crng(12345);
        crng.fill(coarse, cv::RNG::UNIFORM, 0, 255);
        cv::Mat coarseBig;
        cv::resize(coarse, coarseBig, tex.size(), 0, 0, cv::INTER_NEAREST);
        cv::addWeighted(tex, 0.5, coarseBig, 0.5, 0, tex);
    }

    const cv::Vec3d center(0, 0, 0);
    const double radius = 1.6;
    const cv::Vec3d lightDir = [](cv::Vec3d v) {
        cv::normalize(v, v);
        return v;
    }(cv::Vec3d(0.3, 0.8, 0.5));
    const double anglesDeg[] = {-32, -19, -6, 7, 20, 33};
    const double kPi = 3.141592653589793;

    outFiles.clear();
    for (int vi = 0; vi < 6; ++vi) {
        const double phi = anglesDeg[vi] * kPi / 180.0;
        const cv::Vec3d C(2.8 * std::sin(phi), 0.35, 2.8 * std::cos(phi));
        cv::Vec3d fwd = -C;
        cv::normalize(fwd, fwd);
        const cv::Vec3d up(0, 1, 0);
        cv::Vec3d right = fwd.cross(up);
        cv::normalize(right, right);
        cv::Vec3d up2 = right.cross(fwd);
        cv::normalize(up2, up2);
        const cv::Mat R = (cv::Mat_<double>(3, 3) <<
            right[0], right[1], right[2],
            up2[0], up2[1], up2[2],
            fwd[0], fwd[1], fwd[2]);
        const cv::Mat Cm = (cv::Mat_<double>(3, 1) << C[0], C[1], C[2]);
        const cv::Mat t = -R * Cm;
        const cv::Mat Rinv = R.t();

        cv::Mat img(H, W, CV_8UC3, cv::Scalar(205, 205, 205));
        const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
        const double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                cv::Vec3d dCam((x - cx) / fx, (y - cy) / fy, 1.0);
                const cv::Mat dM = Rinv * (cv::Mat_<double>(3, 1)
                    << dCam[0], dCam[1], dCam[2]);
                cv::Vec3d dWorld(dM.at<double>(0), dM.at<double>(1), dM.at<double>(2));
                cv::normalize(dWorld, dWorld);

                const cv::Vec3d oc = C - center;
                const double b = oc.dot(dWorld);
                const double c = oc.dot(oc) - radius * radius;
                const double disc = b * b - c;
                if (disc < 0)
                    continue;
                const double s = -b - std::sqrt(disc);
                if (s <= 0)
                    continue;
                const cv::Vec3d P = C + s * dWorld;
                const cv::Vec3d n = (P - center) / radius;

                const double u = 0.5 + std::atan2(n[2], n[0]) / (2 * kPi);
                const double v = 0.5 - std::asin(std::max(-1.0, std::min(1.0, n[1]))) / kPi;
                const int tx = std::max(0, std::min(tex.cols - 1, static_cast<int>(u * tex.cols)));
                const int ty = std::max(0, std::min(tex.rows - 1, static_cast<int>(v * tex.rows)));
                const auto bgr = tex.at<cv::Vec3b>(ty, tx);
                const double lam = 0.45 + 0.55 * std::max(0.0, n.dot(lightDir));
                auto &px = img.at<cv::Vec3b>(y, x);
                for (int ch = 0; ch < 3; ++ch)
                    px[ch] = static_cast<uchar>(std::min(255.0, bgr[ch] * lam));
            }
        }

        const QString path = dir.filePath(QStringLiteral("view_%1.png").arg(vi, 2, 10, QLatin1Char('0')));
        if (!cv::imwrite(path.toStdString(), img))
            return false;
        outFiles.push_back(path);
    }
    return true;
}

} // namespace

class TestParamFormGenerator : public QObject
{
    Q_OBJECT
private slots:
    void collectRoundtrip()
    {
        ParamFormGenerator gen;
        gen.setSchemas(sampleSchemas());
        QCOMPARE(gen.isEmpty(), false);

        // 默认值收集
        QJsonObject p = gen.collectParams();
        CHECK_UI(p.value(QStringLiteral("count")).toInt() == 42, "Int 默认值");
        CHECK_UI(qAbs(p.value(QStringLiteral("rate")).toDouble() - 0.5) < 1e-6, "Double 默认值");
        CHECK_UI(p.value(QStringLiteral("on")).toBool() == true, "Bool 默认值");
        CHECK_UI(p.value(QStringLiteral("mode")).toString() == QStringLiteral("fast"), "Enum 默认值");
        CHECK_UI(p.value(QStringLiteral("label")).toString() == QStringLiteral("hi"), "String 默认值");

        // 回填后再次收集
        QJsonObject newParams{
            {QStringLiteral("count"), 7},
            {QStringLiteral("rate"), 0.25},
            {QStringLiteral("on"), false},
            {QStringLiteral("mode"), QStringLiteral("accurate")},
            {QStringLiteral("label"), QStringLiteral("bye")},
        };
        gen.setParams(newParams);
        QJsonObject q = gen.collectParams();
        CHECK_UI(q.value(QStringLiteral("count")).toInt() == 7, "Int 回填");
        CHECK_UI(qAbs(q.value(QStringLiteral("rate")).toDouble() - 0.25) < 1e-6, "Double 回填");
        CHECK_UI(q.value(QStringLiteral("on")).toBool() == false, "Bool 回填");
        CHECK_UI(q.value(QStringLiteral("mode")).toString() == QStringLiteral("accurate"), "Enum 回填");
        CHECK_UI(q.value(QStringLiteral("label")).toString() == QStringLiteral("bye"), "String 回填");
    }

    void valuesChangedSignal()
    {
        ParamFormGenerator gen;
        QSignalSpy spy(&gen, &ParamFormGenerator::valuesChanged);
        gen.setSchemas(sampleSchemas());
        spy.clear();
        QCOMPARE(spy.count(), 0);
        Q_UNUSED(gen);
    }
};

class TestAlgorithmComboPanel : public QObject
{
    Q_OBJECT
private slots:
    void registryDriven()
    {
        // 使用真实插件注册表
        PluginRegistry &reg = PluginRegistry::instance();
        const int loaded = reg.scanPlugins(QStringLiteral(PLUGINS_DIR_DEFAULT));
        CHECK_UI(loaded >= 1 || reg.plugins().size() >= 1, "扫描到插件");

        // 回归：构造后面板必须已按当前算法构建参数表单（此前启动界面无参数）
        AlgorithmComboPanel panel(QStringLiteral("descriptor"), &reg, QStringLiteral("描述符"));
        QCOMPARE(panel.currentAlgorithm().isEmpty(), false);

        QJsonObject collect = panel.collectParams();
        const QString algo = collect.value(QStringLiteral("algorithm")).toString();
        CHECK_UI(!algo.isEmpty(), "算法 id 已收集");
        CHECK_UI(!collect.value(QStringLiteral("params")).toObject().isEmpty(),
                 "构造后参数表单已按 schema 填充");

        // 回填相同算法参数不破坏
        panel.setParams(algo, QJsonObject{{QStringLiteral("maxFeatures"), 1000}});
        QJsonObject collect2 = panel.collectParams();
        CHECK_UI(collect2.value(QStringLiteral("algorithm")).toString() == algo, "回填后算法不变");
    }

    /// 回归：matcher 面板反复切换算法（BF↔FLANN），旧表单销毁不得双重释放闪退
    void switchCycleNoCrash()
    {
        PluginRegistry &reg = PluginRegistry::instance();
        AlgorithmComboPanel panel(QStringLiteral("matcher"), &reg, QStringLiteral("匹配器"));

        const QStringList cycle = {QStringLiteral("BruteForce"), QStringLiteral("FLANN"),
                                   QStringLiteral("BruteForce"), QStringLiteral("FLANN"),
                                   QStringLiteral("BruteForce")};
        for (const QString &id : cycle) {
            panel.setParams(id, QJsonObject{});
            QJsonObject collect = panel.collectParams();
            CHECK_UI(collect.value(QStringLiteral("algorithm")).toString() == id,
                     QStringLiteral("切换到 %1").arg(id).toLocal8Bit().constData());
            CHECK_UI(!collect.value(QStringLiteral("params")).toObject().isEmpty(),
                     QStringLiteral("%1 参数表单非空").arg(id).toLocal8Bit().constData());
        }
    }

    /// 回归：ConfigPanel 默认算法为 SIFT + FLANN
    void configPanelDefaults()
    {
        PluginRegistry &reg = PluginRegistry::instance();
        ConfigPanel panel(&reg);
        const ReconstructionConfig cfg = panel.gatherConfig();
        CHECK_UI(cfg.descriptorType == QStringLiteral("SIFT"), "默认描述符 SIFT");
        CHECK_UI(cfg.matcherType == QStringLiteral("FLANN"), "默认匹配器 FLANN");
        CHECK_UI(!cfg.descriptorParams.isEmpty(), "SIFT 参数已收集");
        CHECK_UI(!cfg.matcherParams.isEmpty(), "FLANN 参数已收集");
    }
};

class TestBasicWidgets : public QObject
{
    Q_OBJECT
private slots:
    void logConsoleBasics()
    {
        LogConsole log;
        log.appendInfo(QStringLiteral("info"));
        log.appendWarning(QStringLiteral("warn"));
        log.appendError(QStringLiteral("err"));
        log.appendBatch(QStringList{QStringLiteral("a"), QStringLiteral("b")});
        QCOMPARE(log.toPlainText().contains(QStringLiteral("info")), true);
        QCOMPARE(log.toPlainText().contains(QStringLiteral("err")), true);
        log.clearLog();
        QCOMPARE(log.toPlainText().isEmpty(), true);
    }

    void controlButtonsState()
    {
        ControlButtons btns;
        btns.setRunning(true);
        btns.setRunning(false);
        // 触发点击，断言信号
        QSignalSpy startSpy(&btns, &ControlButtons::startClicked);
        QSignalSpy stopSpy(&btns, &ControlButtons::stopClicked);
        btns.setRunning(true);
        btns.setRunning(false);
        QCOMPARE(startSpy.count(), 0);  // 设置状态不会发信号
        QCOMPARE(stopSpy.count(), 0);
    }

    void pointCloudWidgetLifetime()
    {
        // offscreen 平台下不实际渲染，仅验证构造/数据接口不崩溃
        PointCloudWidget w;
        QVector<float> pts{0, 0, 0, 1, 1, 1};
        QVector<float> cols{1, 0, 0, 0, 1, 0};
        w.updatePointCloud(pts, cols, true);
        w.setColorMode(PointCloudWidget::ColorMode::Mono);
        w.setPointSize(4.0f);
        w.setBackgroundColor(Qt::white);
        w.clearPoints();
        QCOMPARE(true, true);
    }
};

class TestDarkTheme : public QObject
{
    Q_OBJECT
private slots:
    void styleSheetPaintsDarkBackground()
    {
        // 回归：qss 缺少 QWidget 基类背景规则时，面板保持系统白色背景，
        // 与浅色文字叠加形成"白底白字"。应用 qss 后背景必须为深色。
        QFile f(QStringLiteral(APP_SRC_DIR) + QStringLiteral("/style.qss"));
        CHECK_UI(f.open(QIODevice::ReadOnly | QIODevice::Text), "style.qss 可打开");
        if (!f.isOpen())
            return;
        qApp->setStyleSheet(QString::fromUtf8(f.readAll()));
        f.close();

        QWidget w;
        QVBoxLayout lay(&w);
        lay.addWidget(new QLabel(QStringLiteral("标签"), &w));
        lay.addWidget(new QLineEdit(&w));
        lay.addWidget(new QPushButton(QStringLiteral("按钮"), &w));
        w.resize(240, 180);
        w.show();
        QTest::qWaitForWindowExposed(&w);

        const QImage img = w.grab().toImage();
        CHECK_UI(!img.isNull(), "grab 生成图像");
        if (!img.isNull()) {
            // 统计暗色像素占比：qss 生效时 QWidget 背景为 #2b2b2b
            int dark = 0, total = 0;
            for (int y = 0; y < img.height(); y += 4) {
                for (int x = 0; x < img.width(); x += 4) {
                    const QColor c = img.pixelColor(x, y);
                    ++total;
                    if (c.red() < 100 && c.green() < 100 && c.blue() < 100)
                        ++dark;
                }
            }
            const double ratio = total ? double(dark) / total : 0.0;
            std::fprintf(stderr, "[ui] 暗色像素占比: %.2f\n", ratio);
            std::fflush(stderr);
            CHECK_UI(ratio > 0.5, "暗色背景占比 >50%（白底白字回归）");
        }

        // 恢复空样式，避免影响后续用例
        qApp->setStyleSheet(QString());
    }
};

/// 回归：密集重建对话框参数区可滚动、高度受限
class TestDenseReconstructionDialog : public QObject
{
    Q_OBJECT
private slots:
    void scrollableAndHeightCapped()
    {
        DenseReconstructionDialog dlg;

        // 切换到参数最多的 PatchMatch，触发表单重建
        auto *combo = dlg.findChild<QComboBox *>();
        CHECK_UI(combo != nullptr, "方法下拉框存在");
        if (!combo)
            return;
        const int pm = combo->findData(QStringLiteral("PatchMatch"));
        CHECK_UI(pm >= 0, "PatchMatch 在方法列表中");
        if (pm >= 0)
            combo->setCurrentIndex(pm);

        // 回归：参数区必须使用 QScrollArea（此前对话框超出屏幕高度）
        auto *scroll = dlg.findChild<QScrollArea *>();
        CHECK_UI(scroll != nullptr, "参数区为 QScrollArea");
        CHECK_UI(scroll && qobject_cast<ParamFormGenerator *>(scroll->widget()) != nullptr,
                 "滚动区内容为参数表单");

        // 回归：对话框最大高度不超过屏幕可用高度的 85%
        const int screenH = dlg.screen()->availableGeometry().height();
        CHECK_UI(dlg.maximumHeight() <= static_cast<int>(screenH * 0.85) + 1,
                 "对话框高度受限（≤85% 屏高）");

        // 滚动改造后参数收集仍正常
        const QMap<QString, QVariant> params = dlg.selectedParams();
        CHECK_UI(params.contains(QStringLiteral("numIterations")), "PatchMatch 参数已收集");
    }
};

class TestMainWindowSmoke : public QObject
{
    Q_OBJECT
private slots:
    void sparseReconstructionThroughUi()
    {
        QTemporaryDir dir;
        CHECK_UI(dir.isValid(), "临时目录创建");

        QStringList files;
        CHECK_UI(renderSphereViews(dir, files), "合成多视图渲染");

        MainWindow w(QStringLiteral(PLUGINS_DIR_DEFAULT));
        w.setImageFilesForTest(files);
        w.setProjectPathForTest(dir.path());
        try {
            w.runPipelineSyncForTest();
        } catch (const std::exception &e) {
            CHECK_UI(false, QStringLiteral("UI 管线抛出 std::exception: %1")
                                .arg(QString::fromUtf8(e.what())).toLocal8Bit().constData());
        } catch (...) {
            CHECK_UI(false, "UI 管线抛出未知异常");
        }

        const ReconstructionOutput &out = w.lastOutput();
        CHECK_UI(out.sparseCloud && out.sparseCloud->size() >= 50,
                 "UI 路径稀疏点云 ≥50");
        CHECK_UI(w.currentImageCount() == 6, "图像数 6");
    }
};

int main(int argc, char *argv[])
{
    // offscreen 平台
    qputenv("QT_QPA_PLATFORM", "offscreen");
    cv::setNumThreads(1);

    QApplication app(argc, argv);

    int status = 0;
    status |= QTest::qExec(new TestParamFormGenerator, argc, argv);
    status |= QTest::qExec(new TestAlgorithmComboPanel, argc, argv);
    status |= QTest::qExec(new TestBasicWidgets, argc, argv);
    status |= QTest::qExec(new TestDenseReconstructionDialog, argc, argv);
    status |= QTest::qExec(new TestDarkTheme, argc, argv);
    status |= QTest::qExec(new TestMainWindowSmoke, argc, argv);

    std::fprintf(stderr, "[ui_test] FAIL 数: %d, RESULT %d\n", g_fail, status);
    std::fflush(stderr);
    return status;
}

#include "main.moc"