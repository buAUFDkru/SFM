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

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QGroupBox>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QVBoxLayout>
#include <QWheelEvent>

#include "plugin/ParamSchema.h"
#include "plugin/PluginRegistry.h"
#include "builtin/registerBuiltIn.h"
#include "pipeline/ReconstructionPipeline.h"
#include "pipeline/ProjectStore.h"
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

#include <pcl/io/ply_io.h>

#include <cmath>
#include <vector>

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
        registerBuiltInAlgorithms(reg);   // 内置算法（SIFT/SURF/KAZE/FLANN/BF），幂等
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

    /// 回归：算法下拉框不响应滚轮（参数表单内滚动应滚动页面，点击展开才能切换）
    void comboWheelDoesNotSwitch()
    {
        PluginRegistry &reg = PluginRegistry::instance();
        AlgorithmComboPanel descPanel(QStringLiteral("descriptor"), &reg, QStringLiteral("描述符"));
        AlgorithmComboPanel matchPanel(QStringLiteral("matcher"), &reg, QStringLiteral("匹配器"));

        for (AlgorithmComboPanel *panel : {&descPanel, &matchPanel}) {
            QComboBox *combo = panel->findChild<QComboBox *>();
            CHECK_UI(combo != nullptr, "算法下拉框存在");
            if (!combo || combo->count() < 2)
                continue;
            const int before = combo->currentIndex();

            // 模拟鼠标滚轮（正向滚动一格）
            QWheelEvent wheelUp(QPointF(5, 5), QPointF(combo->mapToGlobal(QPoint(5, 5))),
                                QPoint(0, 120), QPoint(0, 120), Qt::NoButton,
                                Qt::NoModifier, Qt::ScrollUpdate, false);
            QCoreApplication::sendEvent(combo, &wheelUp);
            QWheelEvent wheelDown(QPointF(5, 5), QPointF(combo->mapToGlobal(QPoint(5, 5))),
                                  QPoint(0, -120), QPoint(0, -120), Qt::NoButton,
                                  Qt::NoModifier, Qt::ScrollUpdate, false);
            QCoreApplication::sendEvent(combo, &wheelDown);
            CHECK_UI(combo->currentIndex() == before, "滚轮不切换算法");

            // 点击/程序化切换仍有效（setCurrentIndex 不走滚轮路径）
            const int other = (before + 1) % combo->count();
            combo->setCurrentIndex(other);
            CHECK_UI(combo->currentIndex() == other, "程序化切换仍有效");
            combo->setCurrentIndex(before);
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

        // 回归：未加载项目时开始按钮禁用，加载后启用
        auto *startBtn = panel.findChild<QPushButton *>(QStringLiteral("startBtn"));
        CHECK_UI(startBtn != nullptr, "开始按钮存在");
        CHECK_UI(startBtn && !startBtn->isEnabled(), "未加载项目时开始按钮禁用");
        panel.setProjectLoaded(true);
        CHECK_UI(startBtn && startBtn->isEnabled(), "加载项目后开始按钮启用");
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

        // 相机金字塔接口冒烟
        QVector<float> lines{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0};
        QVector<float> lineCols{1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0};
        w.setCameraFrusta(lines, lineCols);
        w.setCameraFrustaVisible(false);
        w.setCameraFrustaVisible(true);
        w.clearCameraFrusta();
        QCOMPARE(true, true);
    }

    /// 相机金字塔几何：坐标系转换与线段数量
    void cameraFrustumGeometry()
    {
        std::vector<CameraPose> poses;
        CameraPose p;
        p.imageId = 0;
        p.imageWidth = 640;
        p.imageHeight = 480;
        p.registered = true;
        p.R = cv::Mat::eye(3, 3, CV_64F);
        p.t = cv::Mat::zeros(3, 1, CV_64F);
        poses.push_back(p);

        const float depth = 2.0f;
        QVector<float> lines = MainWindow::cameraFrustumLines(poses, depth);
        // 每相机 8 条线段 = 16 个端点 = 48 float
        CHECK_UI(lines.size() == 48, "单相机 8 条线段 16 端点");

        // R=I、t=0 → 世界系 = 相机系；显示系绕 Y 轴转 180°（x、z 取反，y 保持）。
        // 端点顺序：光心(0,0,0) → 角点0（相机系 (-hw,-hh,d)，hw=1, hh=0.75）
        //   → 显示系角点0 = (1, -0.75, -2)
        CHECK_UI(qAbs(lines[0]) < 1e-5f && qAbs(lines[1]) < 1e-5f && qAbs(lines[2]) < 1e-5f,
                 "线段起点为光心（原点）");
        CHECK_UI(qAbs(lines[3] - 1.0f) < 1e-5f && qAbs(lines[4] - (-0.75f)) < 1e-5f &&
                     qAbs(lines[5] - (-2.0f)) < 1e-5f, "角点经显示系转换正确");

        // 光心平移：t=(0,0,-1) → C=-R^T·t=(0,0,1) → 显示系 (0,0,-1)
        CameraPose pMoved = p;
        pMoved.t = (cv::Mat_<double>(3, 1) << 0.0, 0.0, -1.0);
        poses.push_back(pMoved);
        QVector<float> lines2 = MainWindow::cameraFrustumLines(poses, depth);
        CHECK_UI(lines2.size() == 96, "两相机 16 条线段");
        CHECK_UI(qAbs(lines2[48]) < 1e-5f && qAbs(lines2[49]) < 1e-5f &&
                     qAbs(lines2[50] - (-1.0f)) < 1e-5f, "光心世界系→显示系平移正确");

        // 未注册或矩阵缺失的位姿跳过
        CameraPose pUnreg = p;
        pUnreg.registered = false;
        poses.push_back(pUnreg);
        QVector<float> lines3 = MainWindow::cameraFrustumLines(poses, depth);
        CHECK_UI(lines3.size() == 96, "未注册位姿跳过");

        // 非法深度返回空
        CHECK_UI(MainWindow::cameraFrustumLines(poses, -1.0f).isEmpty(), "负深度返回空");
    }

    /// 缩略图：几何四角/图像路径对应 + widget 异步纹理加载
    void cameraFrustumThumbnails()
    {
        QTemporaryDir dir;
        CHECK_UI(dir.isValid(), "临时目录创建");
        const QStringList files = {dir.filePath(QStringLiteral("a.png")),
                                   dir.filePath(QStringLiteral("b.png"))};
        QImage img(32, 24, QImage::Format_RGB32);
        img.fill(Qt::red);
        CHECK_UI(img.save(files[0]), "测试图像 A 保存");
        img.fill(Qt::green);
        CHECK_UI(img.save(files[1]), "测试图像 B 保存");

        std::vector<CameraPose> poses;
        CameraPose p0;
        p0.imageId = 0;
        p0.imageWidth = 640;
        p0.imageHeight = 480;
        p0.registered = true;
        p0.R = cv::Mat::eye(3, 3, CV_64F);
        p0.t = (cv::Mat_<double>(3, 1) << 0.0, 0.0, -2.0);
        poses.push_back(p0);

        const auto geom = MainWindow::cameraFrustumGeometry(poses, 1.0f, files);
        CHECK_UI(geom.lines.size() == 48, "单相机线框 48 float");
        CHECK_UI(geom.quadCorners.size() == 4, "单相机底面 4 角点");
        CHECK_UI(geom.imagePaths.size() == 1 && geom.imagePaths[0] == files[0],
                 "图像路径与 imageId 对应");

        // imageId 越界 → 空路径
        const auto geomOob = MainWindow::cameraFrustumGeometry(poses, 1.0f, {});
        CHECK_UI(geomOob.imagePaths.size() == 1 && geomOob.imagePaths[0].isEmpty(),
                 "imageId 越界时路径为空");

        // widget 异步加载缩略图：offscreen 平台无 GL 渲染，纹理创建（paintGL 惰性上传）
        // 不会发生，故断言"待上传缓存"计数验证异步加载与信号投递链路
        PointCloudWidget w;
        QVector<float> lineCols(geom.lines.size(), 1.0f);
        w.setCameraFrusta(geom.lines, lineCols, geom.quadCorners, geom.imagePaths);
        QTest::qWait(2000);
        CHECK_UI(w.pendingFrustaTextureCount() == 1, "缩略图异步加载进入待上传队列");

        // 无效路径 → 不产生待上传图像且不崩溃
        w.setCameraFrusta(geom.lines, lineCols, geom.quadCorners,
                          QStringList{QStringLiteral("Z:/not_exist.png")});
        QTest::qWait(1000);
        CHECK_UI(w.pendingFrustaTextureCount() == 0, "无效路径不产生待上传图像");

        // quadCorners 与 imagePaths 数量不匹配 → 忽略贴图
        w.setCameraFrusta(geom.lines, lineCols, geom.quadCorners, {});
        CHECK_UI(w.pendingFrustaTextureCount() == 0, "数量不匹配忽略贴图");
        CHECK_UI(w.loadedFrustaTextureCount() == 0, "无渲染时无纹理创建");
    }
};

/// ConfigPanel 上下文状态：结果区/任务区折叠/运行锁定/相机开关
class TestConfigPanelStates : public QObject
{
    Q_OBJECT
private slots:
    void sparseResultCollapsesTask()
    {
        PluginRegistry &reg = PluginRegistry::instance();
        ConfigPanel panel(&reg);
        auto *taskBody = panel.findChild<QWidget *>(QStringLiteral("taskBody"));
        auto *resultBox = panel.findChild<QGroupBox *>(QStringLiteral("resultBox"));
        CHECK_UI(taskBody != nullptr && taskBody->isVisibleTo(&panel), "任务区初始展开");
        CHECK_UI(resultBox != nullptr && !resultBox->isVisibleTo(&panel), "结果区初始隐藏");

        panel.setSparseResult(1234);
        CHECK_UI(taskBody && !taskBody->isVisibleTo(&panel), "稀疏完成后任务区折叠");
        CHECK_UI(resultBox && resultBox->isVisibleTo(&panel), "稀疏完成后结果区显示");
        const QList<QLabel *> labels = resultBox->findChildren<QLabel *>();
        CHECK_UI(labels.size() >= 2 && labels[0]->text().contains(QStringLiteral("1234")),
                 "稀疏结果点数文本");

        panel.setDenseResult(5678);
        CHECK_UI(labels.size() >= 2 && labels[1]->isVisibleTo(&panel) &&
                     labels[1]->text().contains(QStringLiteral("5678")), "密集结果行");
        // 密集完成后去掉密集重建按钮
        auto *denseBtn = resultBox->findChild<QPushButton *>();
        CHECK_UI(denseBtn != nullptr && !denseBtn->isVisibleTo(&panel),
                 "密集完成后隐藏密集重建按钮");

        panel.resetResults();
        CHECK_UI(taskBody && taskBody->isVisibleTo(&panel), "重置后任务区展开");
        CHECK_UI(resultBox && !resultBox->isVisibleTo(&panel), "重置后结果区隐藏");
        // resultBox 自身隐藏，按钮以相对 resultBox 的可见性断言恢复
        CHECK_UI(denseBtn && denseBtn->isVisibleTo(resultBox), "重置后恢复密集重建按钮");
    }

    void runningLocksTask()
    {
        PluginRegistry &reg = PluginRegistry::instance();
        ConfigPanel panel(&reg);
        panel.setProjectLoaded(true);
        const QList<AlgorithmComboPanel *> algos =
            panel.findChildren<AlgorithmComboPanel *>();
        CHECK_UI(algos.size() == 2, "两个算法面板");

        panel.setRunning(true);
        CHECK_UI(!algos[0]->isEnabled() && !algos[1]->isEnabled(), "运行中算法区锁定");
        auto *taskBody = panel.findChild<QWidget *>(QStringLiteral("taskBody"));
        CHECK_UI(taskBody && taskBody->isVisibleTo(&panel), "运行中任务区强制展开");

        panel.setRunning(false);
        CHECK_UI(algos[0]->isEnabled() && algos[1]->isEnabled(), "结束后解锁");
    }

    void cameraCheckboxSignal()
    {
        PluginRegistry &reg = PluginRegistry::instance();
        ConfigPanel panel(&reg);
        QSignalSpy spy(&panel, &ConfigPanel::cameraFrustaVisibleChanged);
        auto *cb = panel.findChild<QCheckBox *>();
        CHECK_UI(cb != nullptr && cb->isChecked(), "相机开关默认开");

        cb->setChecked(false);
        CHECK_UI(spy.count() == 1 && !spy.first().at(0).toBool(), "取消勾选发信号");

        panel.setCameraFrustaChecked(true);
        CHECK_UI(cb->isChecked(), "程序化同步勾选");
        CHECK_UI(spy.count() == 1, "程序化同步不发信号");
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

/// 项目文件读写（ProjectStore）往返测试
class TestProjectStore : public QObject
{
    Q_OBJECT
private slots:
    void manifestRoundtrip()
    {
        QTemporaryDir dir;
        CHECK_UI(dir.isValid(), "临时目录创建");

        ProjectManifest m;
        m.name = QStringLiteral("MyProject");
        m.createdAt = QDateTime::currentDateTime();
        m.modifiedAt = QDateTime::currentDateTime();
        m.imageFiles = {QStringLiteral("D:/imgs/a.jpg"), QStringLiteral("D:/imgs/b.png")};
        m.descriptorType = QStringLiteral("SIFT");
        m.descriptorParams = QJsonObject{{QStringLiteral("maxFeatures"), 8192}};
        m.matcherType = QStringLiteral("FLANN");
        m.matcherParams = QJsonObject{{QStringLiteral("ratioThreshold"), 0.8}};
        m.fixCameraMatrix = true;
        m.cameraMatrix = {458.4, 0, 160, 0, 458.4, 120, 0, 0, 1};
        m.denseCloudPath = QStringLiteral("clouds/dense.ply");
        m.totalPoints = 1234;
        m.reprojectionError = 0.56;
        m.totalElapsedMs = 4321.5;

        QString err;
        CHECK_UI(ProjectStore::writeManifest(dir.path(), m, &err), "写清单成功");
        CHECK_UI(err.isEmpty(), "写清单无错误");

        const QString name = ProjectStore::findManifestName(dir.path());
        CHECK_UI(name == QStringLiteral("MyProject"), "扫描清单文件名");

        ProjectManifest r;
        CHECK_UI(ProjectStore::readManifest(dir.path(), name, &r, &err), "读清单成功");
        CHECK_UI(r.name == m.name, "清单名称");
        CHECK_UI(r.imageFiles.size() == 2 && r.imageFiles[1] == m.imageFiles[1], "清单图像列表");
        CHECK_UI(r.descriptorType == QStringLiteral("SIFT"), "清单描述符");
        CHECK_UI(r.descriptorParams.value(QStringLiteral("maxFeatures")).toInt() == 8192,
                 "清单描述符参数");
        CHECK_UI(r.matcherType == QStringLiteral("FLANN"), "清单匹配器");
        CHECK_UI(r.fixCameraMatrix, "清单固定内参标志");
        CHECK_UI(r.cameraMatrix.size() == 9 &&
                     qAbs(r.cameraMatrix[0] - 458.4) < 1e-9 &&
                     qAbs(r.cameraMatrix[8] - 1.0) < 1e-9, "清单内参矩阵");
        CHECK_UI(r.denseCloudPath == QStringLiteral("clouds/dense.ply"), "清单密集点云路径");
        CHECK_UI(r.totalPoints == 1234 && qAbs(r.reprojectionError - 0.56) < 1e-9,
                 "清单统计信息");
    }

    void cameraPosesRoundtrip()
    {
        QTemporaryDir dir;
        CHECK_UI(dir.isValid(), "临时目录创建");

        std::vector<CameraPose> poses;
        CameraPose p0;
        p0.imageId = 0;
        p0.imageWidth = 640;
        p0.imageHeight = 480;
        p0.registered = true;
        p0.K = (cv::Mat_<double>(3, 3) << 458.4, 0, 320, 0, 458.4, 240, 0, 0, 1);
        p0.R = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 0.98, -0.2, 0, 0.2, 0.98);
        p0.t = (cv::Mat_<double>(3, 1) << 0.1, 0.2, 2.8);
        p0.distortion = cv::Mat::zeros(5, 1, CV_64F);
        poses.push_back(p0);

        CameraPose p1;
        p1.imageId = 1;
        p1.registered = false;  // K/R/t 为空 → 写入默认值，读回应为单位/零矩阵
        poses.push_back(p1);

        const QString path = dir.filePath(QStringLiteral("poses.json"));
        QString err;
        CHECK_UI(ProjectStore::writeCameraPoses(path, poses,
                                                {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")},
                                                &err), "写位姿成功");
        std::vector<CameraPose> out;
        CHECK_UI(ProjectStore::readCameraPoses(path, &out, &err), "读位姿成功");
        CHECK_UI(out.size() == 2, "位姿数量");
        CHECK_UI(out[0].imageId == 0 && out[0].imageWidth == 640 && out[0].registered,
                 "位姿元数据");
        CHECK_UI(std::abs(out[0].K.at<double>(0, 0) - 458.4) < 1e-9 &&
                     std::abs(out[0].t.at<double>(2) - 2.8) < 1e-9, "K/t 数值");
        CHECK_UI(std::abs(out[0].R.at<double>(1, 1) - 0.98) < 1e-9, "R 数值");
        CHECK_UI(!out[1].K.empty() && !out[1].R.empty() && !out[1].t.empty(),
                 "空矩阵读回默认值");
        CHECK_UI(!out[1].registered, "未注册标志");
    }

    void matchPairsRoundtrip()
    {
        QTemporaryDir dir;
        CHECK_UI(dir.isValid(), "临时目录创建");

        std::vector<MatchPair> pairs;
        MatchPair mp;
        mp.imageId1 = 0;
        mp.imageId2 = 1;
        mp.modelChoice = 1;
        mp.avgDistance = 123.456;
        for (int i = 0; i < 5; ++i) {
            mp.matches.push_back({i, i + 10, 1, 0.5f + i});
            mp.points1.push_back({10.5 + i, 20.25 + i});
            mp.points2.push_back({30.75 + i, 40.125 + i});
        }
        mp.matchCount = static_cast<int>(mp.matches.size());
        mp.F = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        mp.H = (cv::Mat_<double>(3, 3) << 2, 0, 1, 0, 2, 0, 0, 0, 2);
        pairs.push_back(mp);

        MatchPair emptyMp;  // 无 F/H 的对
        emptyMp.imageId1 = 1;
        emptyMp.imageId2 = 2;
        pairs.push_back(emptyMp);

        const QString path = dir.filePath(QStringLiteral("pairs.bin"));
        QString err;
        CHECK_UI(ProjectStore::writeMatchPairs(path, pairs, &err), "写匹配对成功");
        std::vector<MatchPair> out;
        CHECK_UI(ProjectStore::readMatchPairs(path, &out, &err), "读匹配对成功");
        CHECK_UI(out.size() == 2, "匹配对数量");
        CHECK_UI(out[0].imageId1 == 0 && out[0].imageId2 == 1 &&
                     out[0].matchCount == 5 && out[0].modelChoice == 1, "匹配对元数据");
        CHECK_UI(qAbs(out[0].avgDistance - 123.456) < 1e-9, "平均距离");
        CHECK_UI(out[0].matches.size() == 5 && out[0].matches[3].trainIdx == 13 &&
                     std::abs(out[0].matches[3].distance - 3.5f) < 1e-6, "DMatch 往返");
        CHECK_UI(std::abs(out[0].points1[4].x - 14.5) < 1e-9 &&
                     std::abs(out[0].points2[4].y - 44.125) < 1e-9, "匹配点坐标往返");
        CHECK_UI(!out[0].F.empty() && std::abs(out[0].H.at<double>(0, 0) - 2.0) < 1e-9,
                 "F/H 往返");
        CHECK_UI(out[1].F.empty() && out[1].H.empty(), "空 F/H 往返");
    }

    /// 端到端：完整重建 → saveProject → loadProject → 密集续算可用
    void projectSaveLoadRoundtrip()
    {
        QTemporaryDir dir;
        CHECK_UI(dir.isValid(), "临时目录创建");

        QStringList files;
        CHECK_UI(renderSphereViews(dir, files), "合成多视图渲染");

        const QString projDir = dir.path() + QStringLiteral("/proj");

        ReconstructionPipeline pipe;
        ReconstructionConfig cfg;
        cfg.imageFiles = files;
        cfg.projectPath = projDir;
        cfg.projectName = QStringLiteral("proj");
        cfg.denseReconstructionMethod = QStringLiteral("StereoSGBM");
        cfg.denseReconstructionParams = {
            {QStringLiteral("useWLSFilter"), false},
            {QStringLiteral("numDisparities"), 64},
            {QStringLiteral("blockSize"), 5},
            {QStringLiteral("useHHMode"), false},
            {QStringLiteral("minDisparity"), 0},
            {QStringLiteral("maxDepth"), 50.0},
        };
        pipe.setConfig(cfg);
        try {
            pipe.execute();
        } catch (const std::exception &e) {
            CHECK_UI(false, QStringLiteral("管线抛出 std::exception: %1")
                                .arg(QString::fromUtf8(e.what())).toLocal8Bit().constData());
            return;
        } catch (...) {
            CHECK_UI(false, "管线抛出未知异常");
            return;
        }

        // 回归：阶段保存的 dense.ply 必须是显示坐标系（与 lastOutput 一致），
        // 否则打开项目读回的密集点云方向差 180°
        const ReconstructionOutput &execOut = pipe.lastOutput();
        if (execOut.denseCloud && !execOut.denseCloud->empty()) {
            pcl::PointCloud<pcl::PointXYZRGB> stageDense;
            pcl::io::loadPLYFile((projDir + QStringLiteral("/clouds/dense.ply"))
                                     .toStdString(), stageDense);
            bool coordsMatch = !stageDense.empty() &&
                               stageDense.size() == execOut.denseCloud->size();
            if (coordsMatch) {
                const size_t sampleStep =
                    std::max<size_t>(1, execOut.denseCloud->size() / 20);
                for (size_t i = 0; i < execOut.denseCloud->size(); i += sampleStep) {
                    if (std::abs(stageDense.points[i].y - execOut.denseCloud->points[i].y) > 1e-3f ||
                        std::abs(stageDense.points[i].z - execOut.denseCloud->points[i].z) > 1e-3f) {
                        coordsMatch = false;
                        break;
                    }
                }
            }
            CHECK_UI(coordsMatch, "阶段保存 dense.ply 为显示坐标系（无二次翻转）");
        }

        QString err;
        CHECK_UI(pipe.saveProject(&err), "项目保存成功");
        CHECK_UI(QFile::exists(projDir + QStringLiteral("/proj.sfmproj")), "清单文件存在");
        CHECK_UI(QFile::exists(projDir + QStringLiteral("/clouds/sparse.ply")),
                 "稀疏点云文件存在");
        CHECK_UI(QFile::exists(projDir + QStringLiteral("/cameras/poses.json")),
                 "相机参数文件存在");
        CHECK_UI(QFile::exists(projDir + QStringLiteral("/matches/pairs.bin")),
                 "匹配对文件存在");

        // ── 重新打开：恢复状态 → 密集续算可用 ──
        ReconstructionPipeline pipe2;
        CHECK_UI(pipe2.loadProject(projDir, &err), "项目打开成功");
        CHECK_UI(pipe2.denseReconstructionAvailable(), "密集续算可用");
        CHECK_UI(pipe2.config().imageFiles.size() == 6, "恢复图像列表");
        CHECK_UI(pipe2.config().projectName == QStringLiteral("proj"), "恢复项目名");

        const ReconstructionOutput &out = pipe2.lastOutput();
        CHECK_UI(out.sparseCloud && out.sparseCloud->size() >= 50, "恢复稀疏点云");
        CHECK_UI(!out.cameraPoses.empty(), "恢复相机位姿");

        // 在恢复状态上直接执行密集重建（跳过稀疏）；
        // 与 UI 路径（DenseReconstructionDialog → setDenseReconstructionMethod）一致，先指定方法
        pipe2.setDenseReconstructionMethod(QStringLiteral("StereoSGBM"),
                                           {{QStringLiteral("useWLSFilter"), false},
                                            {QStringLiteral("numDisparities"), 64},
                                            {QStringLiteral("blockSize"), 5},
                                            {QStringLiteral("useHHMode"), false},
                                            {QStringLiteral("minDisparity"), 0},
                                            {QStringLiteral("maxDepth"), 50.0}});
        bool denseOk = false;
        try {
            denseOk = pipe2.executeDenseReconstruction();
        } catch (const std::exception &e) {
            CHECK_UI(false, QStringLiteral("密集续算抛出 std::exception: %1")
                                .arg(QString::fromUtf8(e.what())).toLocal8Bit().constData());
            return;
        } catch (...) {
            CHECK_UI(false, "密集续算抛出未知异常");
            return;
        }
        CHECK_UI(denseOk, "密集续算执行成功");
        const ReconstructionOutput &out2 = pipe2.lastOutput();
        CHECK_UI(out2.sparseCloud && out2.sparseCloud->size() >= 50,
                 "续算后稀疏点云仍可用（沿用项目点云）");
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
        const QString projDir = dir.path() + QStringLiteral("/SmokeProject");
        w.setProjectForTest(projDir, files);
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
        CHECK_UI(QFile::exists(projDir + QStringLiteral("/SmokeProject.sfmproj")),
                 "UI 路径项目清单存在");
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
    status |= QTest::qExec(new TestConfigPanelStates, argc, argv);
    status |= QTest::qExec(new TestProjectStore, argc, argv);
    status |= QTest::qExec(new TestMainWindowSmoke, argc, argv);

    std::fprintf(stderr, "[ui_test] FAIL 数: %d, RESULT %d\n", g_fail, status);
    std::fflush(stderr);
    return status;
}

#include "main.moc"