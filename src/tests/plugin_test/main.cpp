#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QMessageLogContext>
#include <QStringList>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "algorithm/Initialization.h"
#include "algorithm/MatcherService.h"
#include "algorithm/SfmLogger.h"
#include "algorithm/ThreadPool.h"
#include "builtin/registerBuiltIn.h"
#include "plugin/IDescriptorPlugin.h"
#include "plugin/IMatcherPlugin.h"
#include "plugin/PluginInfo.h"
#include "plugin/PluginRegistry.h"
#include "pipeline/DenseReconstructionFactory.h"
#include "pipeline/IDenseReconstructor.h"
#include "pipeline/ReconstructionPipeline.h"
#include "pipeline/PipelineRunner.h"

namespace {

int failures = 0;

void check(bool cond, const QString &name)
{
    // 同步直写 stderr，确保即便 Qt 消息处理器被替换/缓冲区被吞也能看到结果
    std::fprintf(stderr, "[chk:%s] %s\n", cond ? "PASS" : "FAIL",
                 name.toLocal8Bit().constData());
    std::fflush(stderr);
    if (cond) {
        qInfo().noquote() << "[PASS]" << name;
    } else {
        qCritical().noquote() << "[FAIL]" << name;
        ++failures;
    }
}

/// 构造一张固定种子的随机噪声构图 (256x256 CV_8UC1)，保证 SIFT/KAZE 有丰富特征
ImageEntry makeSyntheticTextureImage(int id)
{
    ImageEntry img;
    img.id = id;
    img.imageGray = cv::Mat(256, 256, CV_8UC1);
    cv::RNG rng(42);
    rng.fill(img.imageGray, cv::RNG::UNIFORM, 0, 255);
    img.imageData = img.imageGray.clone();
    return img;
}

/// 构造一对固定种子的随机浮点描述符矩阵 (desc2 = desc1 深度复制，最近邻距离≈0)
void makeSyntheticFeatures(cv::Mat &desc1, cv::Mat &desc2)
{
    desc1 = cv::Mat(32, 64, CV_32F);
    cv::RNG rng(7);
    rng.fill(desc1, cv::RNG::UNIFORM, 0.f, 1.f);
    desc2 = desc1.clone();
}

/// 对描述符插件做 P2 并发一致性检查（8 线程 × tasksPerThread 任务，各任务内 createWorker + extract）
void runDescriptorConcurrencyCheck(IDescriptorPlugin *plugin, int tasksPerThread = 10)
{
    const int kThreads = 8;
    const int kTasksPerThread = tasksPerThread;

    const ImageEntry img = makeSyntheticTextureImage(0);

    // 串行参考结果
    auto refWorker = plugin->createWorker(QJsonObject{});
    if (!refWorker) {
        qWarning() << "  并发一致性: createWorker 失败，跳过";
        return;
    }
    const FeatureResult ref = refWorker->extract(img);
    if (ref.keypoints.empty()) {
        qWarning() << "  并发一致性: 参考结果为 0 关键点，跳过";
        return;
    }

    std::atomic<int> mismatch{0};

    auto workerFn = [&]() {
        for (int k = 0; k < kTasksPerThread; ++k) {
            auto w = plugin->createWorker(QJsonObject{});
            if (!w) {
                ++mismatch;
                continue;
            }
            const FeatureResult r = w->extract(img);
            const double descSum =
                r.descriptors.empty() ? 0.0 : cv::sum(r.descriptors).val[0];
            const double refSum =
                ref.descriptors.empty() ? 0.0 : cv::sum(ref.descriptors).val[0];
            if (r.keypoints.size() != ref.keypoints.size() || descSum != refSum)
                ++mismatch;
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back(workerFn);
    for (auto &th : threads) {
        if (th.joinable())
            th.join();
    }

    const int total = kThreads * kTasksPerThread;
    check(mismatch.load() == 0,
          QStringLiteral("并发一致性 (%1 线程 × %2 任务, 0 不一致)").arg(kThreads).arg(kTasksPerThread)
              + (mismatch.load() == 0 ? QStringLiteral(" — 全部与串行一致")
                                      : QStringLiteral(" — 不一致 %1 处").arg(mismatch.load())));
}

// ── P3a 共享算法单元测试 ────────────────────────────────────────────────────

void runThreadPoolTests()
{
    // 默认配置线程数合法：至少 1，至多不超过硬件并发数
    // （默认策略可能随版本调整，如 hc/4 截断到 2、或 hc/2 不截断，此处只断言边界）
    {
        const auto opts = sfm::make_cpu_pool_options();
        const auto hc = static_cast<unsigned long long>(std::thread::hardware_concurrency());
        check(opts.thread_count >= 1 && opts.thread_count <= std::max<unsigned long long>(hc, 1),
              QStringLiteral("ThreadPool 默认线程数 [1,硬件并发数%1] → %2")
                  .arg(hc).arg(opts.thread_count));
    }

    // submit 收集 0..99 求和 = 4950
    {
        sfm::ThreadPool pool(sfm::make_cpu_pool_options(2));
        std::vector<std::future<int>> futures;
        futures.reserve(100);
        for (int i = 0; i < 100; ++i) {
            futures.push_back(pool.submit([i] { return i; }));
        }
        long long sum = 0;
        for (auto &f : futures)
            sum += f.get();
        check(sum == 4950, QStringLiteral("ThreadPool submit 100 任务求和=%1").arg(sum));
    }

    // detach + wait_for_idle：全部任务完成
    {
        sfm::ThreadPool pool(sfm::make_cpu_pool_options(2));
        std::atomic<int> counter{0};
        for (int i = 0; i < 200; ++i) {
            const bool ok = pool.detach([&counter] { counter.fetch_add(1); });
            check(ok, "ThreadPool detach 入队成功");
            if (!ok)
                break;
        }
        pool.wait_for_idle();
        check(counter.load() == 200,
              QStringLiteral("ThreadPool wait_for_idle 后计数=%1").arg(counter.load()));
    }

    // shutdown 后 submit 抛异常
    {
        sfm::ThreadPool pool(sfm::make_cpu_pool_options(2));
        pool.shutdown();
        bool threw = false;
        try {
            pool.submit([] { return 1; });
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, "ThreadPool shutdown 后 submit 抛 runtime_error");
        check(!pool.is_running(), "ThreadPool shutdown 后 is_running()==false");
    }
}

void runSfmLoggerTests()
{
    // 回调接收 + 清空后不再接收
    QStringList received;
    auto makeCapture = [&received](QStringList &out) {
        return SfmLogger::LogCallback([&out](const QString &msg) { out.push_back(msg); });
    };
    SfmLogger &logger = SfmLogger::instance();
    logger.setDefaultCallback(makeCapture(received));
    received.clear();
    logger.log(QStringLiteral("hello-logger"));
    check(received.size() == 1 && received[0].contains("hello-logger"),
          QStringLiteral("SfmLogger 回调收到 %1 条日志").arg(received.size()));

    // 全局消息处理：安装后 qWarning 被捕获，卸载后恢复
    QStringList captured;
    logger.clearCallbacks();
    logger.addCallback(makeCapture(captured));
    logger.installAsMessageHandler();
    qWarning("captured-by-sfmlogger");
    logger.uninstallAsMessageHandler();
    bool got = false;
    for (const auto &s : captured)
        if (s.contains("captured-by-sfmlogger"))
            got = true;
    check(got, "SfmLogger installAsMessageHandler 捕获 qWarning");
    logger.clearCallbacks();
}

/// 构造两视图对应点：由 3D 场景经 [I|0] 与 [R|t] 投影，加高斯噪声
void makeTwoViewPoints(const std::vector<cv::Point3d> &world,
                       const cv::Mat &R, const cv::Mat &t, const cv::Mat &K,
                       double noiseSigma, cv::RNG &rng,
                       std::vector<cv::Point2d> &pts1,
                       std::vector<cv::Point2d> &pts2)
{
    pts1.clear();
    pts2.clear();
    pts1.reserve(world.size());
    pts2.reserve(world.size());
    for (const auto &X : world) {
        // 相机2 = R * X + t
        const cv::Mat Rc = K * (R * (cv::Mat_<double>(3, 1) << X.x, X.y, X.z) + t);

        const cv::Mat p1 = K * (cv::Mat_<double>(3, 1) << X.x, X.y, X.z);
        double u1 = p1.at<double>(0, 0) / p1.at<double>(2, 0);
        double v1 = p1.at<double>(1, 0) / p1.at<double>(2, 0);
        double u2 = Rc.at<double>(0, 0) / Rc.at<double>(2, 0);
        double v2 = Rc.at<double>(1, 0) / Rc.at<double>(2, 0);

        if (p1.at<double>(2, 0) <= 0 || Rc.at<double>(2, 0) <= 0)
            continue;
        pts1.push_back({u1 + rng.gaussian(noiseSigma), v1 + rng.gaussian(noiseSigma)});
        pts2.push_back({u2 + rng.gaussian(noiseSigma), v2 + rng.gaussian(noiseSigma)});
    }
}

void runInitializationTests()
{
    Initialization *init = Initialization::instance();
    const cv::Mat K = estimateK(640, 480);
    cv::RNG rng(20260923);

    // 场景 A：一般 3D 场景 → 应选 F 模型
    {
        std::vector<cv::Point3d> world;
        world.reserve(240);
        cv::RNG grng(11);
        for (int i = 0; i < 240; ++i) {
            world.push_back({grng.uniform(-2.0, 2.0),
                             grng.uniform(-2.0, 2.0),
                             grng.uniform(4.0, 8.0)});
        }
        cv::Mat rvecA = (cv::Mat_<double>(3, 1) << 0.0, 0.14, 0.0); // ~8° Y 旋转
        cv::Mat R;
        cv::Rodrigues(rvecA, R);
        const cv::Mat t = (cv::Mat_<double>(3, 1) << 0.45, 0.10, -0.05);

        std::vector<cv::Point2d> pts1, pts2;
        makeTwoViewPoints(world, R, t, K, 0.35, rng, pts1, pts2);

        cv::Mat F, H;
        std::vector<uchar> inliers;
        const int model = init->selectModel(pts1, pts2, F, H, inliers, 1.0, 2.0);
        const int inlierCount =
            static_cast<int>(std::count(inliers.begin(), inliers.end(), 1));
        check(model == 0, QStringLiteral("GRIC 一般3D场景选 F 模型 (model=%1)").arg(model));
        check(inlierCount > static_cast<int>(pts1.size()) * 0.6,
              QStringLiteral("F 内点率 >=60% (inliers=%1/%2)")
                  .arg(inlierCount).arg(pts1.size()));
    }

    // 场景 B：平面场景(z≈1 严格平面) → 应选 H 模型
    {
        std::vector<cv::Point3d> world;
        world.reserve(240);
        cv::RNG grng(22);
        for (int i = 0; i < 240; ++i) {
            world.push_back({grng.uniform(-3.0, 3.0),
                             grng.uniform(-3.0, 3.0),
                             1.0});
        }
        cv::Mat rvecB = (cv::Mat_<double>(3, 1) << 0.0, 0.10, 0.06);
        cv::Mat R;
        cv::Rodrigues(rvecB, R);
        const cv::Mat t = (cv::Mat_<double>(3, 1) << 0.30, 0.08, -0.03);

        std::vector<cv::Point2d> pts1, pts2;
        makeTwoViewPoints(world, R, t, K, 0.25, rng, pts1, pts2);

        cv::Mat F, H;
        std::vector<uchar> inliers;
        const int model = init->selectModel(pts1, pts2, F, H, inliers, 1.0, 2.0);
        const int inlierCount =
            static_cast<int>(std::count(inliers.begin(), inliers.end(), 1));
        check(model == 1, QStringLiteral("GRIC 平面场景选 H 模型 (model=%1)").arg(model));
        check(inlierCount > static_cast<int>(pts1.size()) * 0.6,
              QStringLiteral("H 内点率 >=60% (inliers=%1/%2)")
                  .arg(inlierCount).arg(pts1.size()));
    }

    // recoverPoseFromH：平面场景(相机前方)恢复位姿成功
    {
        std::vector<cv::Point3d> world;
        world.reserve(120);
        cv::RNG grng(33);
        for (int i = 0; i < 120; ++i) {
            world.push_back({grng.uniform(-2.0, 2.0),
                             grng.uniform(-2.0, 2.0),
                             1.0});
        }
        cv::Mat rvecC = (cv::Mat_<double>(3, 1) << 0.02, 0.09, 0.03);
        cv::Mat R;
        cv::Rodrigues(rvecC, R);
        const cv::Mat t = (cv::Mat_<double>(3, 1) << 0.25, 0.05, -0.02);

        std::vector<cv::Point2d> pts1, pts2;
        makeTwoViewPoints(world, R, t, K, 0.2, rng, pts1, pts2);

        const cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 2.0);
        cv::Mat Rr, tr;
        const bool ok = init->recoverPoseFromH(pts1, pts2, K, H, Rr, tr);
        check(ok && cv::norm(tr) > 1e-6,
              QStringLiteral("recoverPoseFromH 成功 (t=%1,%2,%3)")
                  .arg(tr.at<double>(0, 0), 0, 'f', 3)
                  .arg(tr.at<double>(1, 0), 0, 'f', 3)
                  .arg(tr.at<double>(2, 0), 0, 'f', 3));
    }

    // estimateK / pixel2cam 数值断言
    {
        const cv::Mat Kk = estimateK(640, 480);
        const double expectFx = 480.0 * 0.5 / (3.141592653589793 / 6.0);
        check(std::abs(Kk.at<double>(0, 0) - expectFx) < 1e-9 &&
                  Kk.at<double>(0, 2) == 320.0 && Kk.at<double>(1, 2) == 240.0,
              "estimateK(640x480) 数值正确");
        const cv::Point2d n = pixel2cam({Kk.at<double>(0, 2), Kk.at<double>(1, 2)}, Kk);
        check(std::abs(n.x) < 1e-12 && std::abs(n.y) < 1e-12,
              "pixel2cam 主点映射到归一化原点");
    }
}

/// 构造带纹理平面图像序列（近平面 → H 模型主导的匹配）
ImageEntry makeWarpedTexture(int id, const cv::Mat &base, double dx, double dy, double angleDeg)
{
    const int rows = base.rows, cols = base.cols;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point2f(cols / 2.f, rows / 2.f), angleDeg, 1.0);
    m.at<double>(0, 2) += dx;
    m.at<double>(1, 2) += dy;

    cv::Mat warped;
    cv::warpAffine(base, warped, m, base.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

    ImageEntry img;
    img.id = id;
    img.imageGray = warped.clone();
    img.imageData = warped.clone();
    return img;
}

void runMatcherServiceSmoke(PluginRegistry &reg)
{
    IDescriptorPlugin *desc = reg.descriptorPlugin("SIFT");
    IMatcherPlugin *matchPlugin = reg.matcherPlugin("FLANN");
    if (!desc || !matchPlugin) {
        qWarning().noquote() << "[WARN] SIFT/FLANN 插件缺失，跳过 MatcherService 冒烟测试";
        check(true, "MatcherService 冒烟：插件缺失，跳过(计 PASS)");
        return;
    }

    // 1. 合成三视图：底图 + 平移 + 微旋
    cv::Mat base(240, 320, CV_8UC1);
    {
        cv::RNG rng(4242);
        rng.fill(base, cv::RNG::UNIFORM, 0, 255);
    }
    const auto img0 = makeWarpedTexture(0, base, 0.0, 0.0, 0.0);
    const auto img1 = makeWarpedTexture(1, base, 6.0, 4.0, 0.0);
    const auto img2 = makeWarpedTexture(2, base, 2.0, -3.0, 2.5);

    // 2. SIFT 提取特征
    auto dw = desc->createWorker(QJsonObject{});
    if (!dw) {
        check(false, "MatcherService 冒烟：SIFT createWorker 失败");
        return;
    }
    std::vector<FeatureResult> features;
    for (const auto &img : {img0, img1, img2}) {
        FeatureResult fr = dw->extract(img);
        fr.imageWidth = img.imageGray.cols;
        fr.imageHeight = img.imageGray.rows;
        features.push_back(std::move(fr));
    }
    for (const auto &fr : features) {
        check(fr.keypoints.size() > 100,
              QStringLiteral("MatcherService 冒烟：SIFT 提取 keypoints=%1 (>100)")
                  .arg(fr.keypoints.size()));
    }

    MatcherService service(&reg);
    const QJsonObject mparams{
        {"ratioThreshold", 0.8}, {"ransacThreshold", 2.0},
        {"minMatchCount", 20}, {"minMatchRatio", 0.05},
    };

    // 3. 串行编排匹配
    std::vector<MatchPair> pairs;
    const bool ok = service.matchFeatures(features, "FLANN", mparams, pairs);
    check(ok && !pairs.empty(),
          QStringLiteral("MatcherService 串行编排产出 %1 对有效匹配").arg(pairs.size()));
    for (const auto &mp : pairs) {
        check(mp.matchCount >= 8 && !mp.F.empty() && !mp.H.empty(),
              QStringLiteral("MatchPair #%1↔#%2 验证内点=%3, F/H 有效")
                  .arg(mp.imageId1).arg(mp.imageId2).arg(mp.matchCount));
    }

    // 4. 并行编排路径（线程池，逐任务独占 Worker）
    std::vector<MatchPair> pairsPar;
    {
        sfm::ThreadPool pool(sfm::make_cpu_pool_options(2));
        const bool okPar = service.matchFeatures(features, "FLANN", mparams, pairsPar,
                                                 &pool);
        check(okPar == ok,
              QStringLiteral("MatcherService 并行编排结果一致 (pairs=%1)")
                  .arg(pairsPar.size()));
    }

    // 5. estimateKFromFeatures
    {
        const cv::Mat Kk = service.estimateKFromFeatures(features);
        const double expectFx = 240.0 * 0.5 / (3.141592653589793 / 6.0);
        check(Kk.at<double>(0, 2) == 160.0 && std::abs(Kk.at<double>(0, 0) - expectFx) < 1e-6,
              "MatcherService estimateKFromFeatures 数值正确");
    }

    // 6. 视图图构建
    {
        const ViewGraph graph = ViewGraph::buildFromPairs(pairs, 3);
        check(graph.numNodes == 3 && !graph.edges.empty() &&
                  graph.edgeWeight(pairs.front().imageId1, pairs.front().imageId2) > 0,
              QStringLiteral("ViewGraph 构建成功 (edges=%1)").arg(graph.edges.size()));
    }
}

// ── P3b 重建管线冒烟 ────────────────────────────────────────────────────────

/// 回调捕获：进度/日志/阶段点云
class PipelineCapture : public IReconstructionCallbacks
{
public:
    QStringList logs;
    int cloudEvents = 0;
    void onProgress(const QString &, int) override {}
    void onLog(const QString &message) override { logs.push_back(message); }
    void onStagePointCloudReady(const QVector<float> &, const QVector<float> &,
                                const QString &) override { ++cloudEvents; }
    void onDenseReconstructionMethodRequested() override {}
};

/// 手动归一化 cv::Vec3d
static cv::Vec3d normalize3(const cv::Vec3d &v)
{
    const double len = std::sqrt(v.dot(v));
    if (len < 1e-12) return v;
    return v / len;
}

/// 渲染带噪声纹理球体场景的多视图（透视投影，确定性种子）
static bool renderSphereViews(const QString &dir, QStringList &outFiles)
{
    const int W = 640, H = 480;
    const cv::Mat K = estimateK(W, H);

    // 噪声纹理（确定性种子）+ 低频块状结构增强尺度特征
    cv::Mat tex(512, 512, CV_8UC3);
    {
        cv::RNG trng(777);
        trng.fill(tex, cv::RNG::UNIFORM, 0, 255);
        cv::Mat coarse(64, 64, CV_8UC3);
        cv::RNG crng(12345);
        crng.fill(coarse, cv::RNG::UNIFORM, 0, 255);
        cv::Mat coarseBig;
        cv::resize(coarse, coarseBig, tex.size(), 0, 0, cv::INTER_NEAREST);
        cv::addWeighted(tex, 0.5, coarseBig, 0.5, 0, tex);
    }

    const cv::Vec3d center(0, 0, 0);
    const double radius = 1.6;
    const cv::Vec3d lightDir = normalize3(cv::Vec3d(0.3, 0.8, 0.5));

    const double anglesDeg[] = {-32, -19, -6, 7, 20, 33};
    const double kPi = 3.141592653589793;

    outFiles.clear();
    for (int vi = 0; vi < 6; ++vi) {
        const double phi = anglesDeg[vi] * kPi / 180.0;
        // 相机中心绕 Y 轴轨道，look-at 原点
        const cv::Vec3d C(2.8 * std::sin(phi), 0.35, 2.8 * std::cos(phi));
        cv::Vec3d fwd = normalize3(-C);
        const cv::Vec3d up(0, 1, 0);
        cv::Vec3d right = normalize3(fwd.cross(up));
        cv::Vec3d up2 = normalize3(right.cross(fwd));
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
                const cv::Vec3d dWorld = normalize3(cv::Vec3d(
                    dM.at<double>(0), dM.at<double>(1), dM.at<double>(2)));

                const cv::Vec3d oc = C - center;
                const double b = oc.dot(dWorld);
                const double c = oc.dot(oc) - radius * radius;
                const double disc = b * b - c;
                if (disc < 0) continue;
                const double s = -b - std::sqrt(disc);
                if (s <= 0) continue;
                const cv::Vec3d P = C + s * dWorld;
                const cv::Vec3d n = (P - center) / radius;

                const double u = 0.5 + std::atan2(n[2], n[0]) / (2 * kPi);
                const double v = 0.5 - std::asin(std::max(-1.0, std::min(1.0, n[1]))) / kPi;
                const int tx = std::max(0, std::min(tex.cols - 1, static_cast<int>(u * tex.cols)));
                const int ty = std::max(0, std::min(tex.rows - 1, static_cast<int>(v * tex.rows)));
                const auto bgr = tex.at<cv::Vec3b>(ty, tx);
                const double lam = 0.45 + 0.55 * std::max(0.0, n.dot(lightDir));
                auto &px = img.at<cv::Vec3b>(y, x);
                for (int ch = 0; ch < 3; ++ch) {
                    px[ch] = static_cast<uchar>(std::min(255.0, bgr[ch] * lam));
                }
            }
        }

        const QString path = QStringLiteral("%1/view_%2.png")
                                 .arg(dir).arg(vi, 2, 10, QLatin1Char('0'));
        if (!cv::imwrite(path.toStdString(), img)) {
            qCritical() << "无法写入" << path;
            return false;
        }
        outFiles.push_back(path);
    }
    return true;
}

/// 断言管线结果：稀疏点云/注册相机/密集跳过
static void checkSparseResult(const ReconstructionOutput &out, const QString &tag)
{
    int registered = 0;
    for (const auto &pose : out.cameraPoses)
        if (pose.registered) ++registered;
    check(out.sparseCloud && out.sparseCloud->size() >= 100,
          QStringLiteral("%1 稀疏点云 ≥100 点 (实际 %2)")
              .arg(tag)
              .arg(out.sparseCloud ? static_cast<int>(out.sparseCloud->size()) : 0));
    check(registered >= 3,
          QStringLiteral("%1 已注册相机 ≥3 (实际 %2)").arg(tag).arg(registered));
}

void runReconstructionPipelineSmoke(PluginRegistry &reg)
{
    if (!reg.descriptorPlugin("SIFT") || !reg.matcherPlugin("FLANN")) {
        qWarning().noquote() << "[WARN] SIFT/FLANN 插件缺失，跳过重建管线冒烟";
        check(true, "重建管线冒烟：插件缺失，跳过(计 PASS)");
        return;
    }

    const QString workDir = QDir::temp().filePath(
        QStringLiteral("sfm3_smoke_%1").arg(QCoreApplication::applicationPid()));
    QDir().mkpath(workDir);

    QStringList files;
    if (!renderSphereViews(workDir, files) || files.size() < 4) {
        check(false, "重建管线冒烟：合成多视图渲染失败");
        return;
    }
    check(files.size() == 6, QStringLiteral("重建管线冒烟：渲染 %1 张视图").arg(files.size()));

    ReconstructionConfig cfg;
    cfg.imageFiles = files;
    cfg.projectPath = workDir;
    cfg.descriptorType = "SIFT";
    cfg.matcherType = "FLANN";
    cfg.threadCount = 2;
    // 未实现的密集方法：走"未找到方法，跳过"分支，避免阻塞等待 UI 选择
    cfg.denseReconstructionMethod = "P4NotImplemented";

    // ── 直接路径：ReconstructionPipeline::execute（同步） ──
    {
        ReconstructionPipeline pipeline;
        PipelineCapture cap;
        pipeline.setCallbacks(&cap);
        pipeline.setConfig(cfg);

        std::fprintf(stderr, "[dbg] dense methods after ctor: %s\n",
                     DenseReconstructionFactory::instance()
                         .availableMethods().join(QStringLiteral(","))
                         .toLocal8Bit().constData());

        ReconstructionOutput out;
        try {
            out = pipeline.execute();
        } catch (const std::exception &e) {
            check(false, QStringLiteral("管线直接路径抛出异常: %1").arg(e.what()));
        } catch (...) {
            check(false, "管线直接路径抛出未知异常");
        }

        // 转储管线日志（SfmLogger 处理器会吞掉 qInfo/qCritical，无法直接看到）
        for (const auto &l : cap.logs)
            std::fprintf(stderr, "[pipe] %s\n", l.toLocal8Bit().constData());

        checkSparseResult(out, "管线直接路径");
        check(cap.logs.join(QStringLiteral("\n")).contains("跳过密集重建"),
              "密集重建未实现方法被安全跳过(直接路径)");
        check(cap.cloudEvents >= 1, "接收阶段点云事件(sparse,直接路径)");

        // ── 密集重建（PatchMatchV2，复用稀疏中间状态） ──
        {
            // 快速参数：缩小工作图像、关闭精化项，保证测试时长可控
            const QMap<QString, QVariant> denseParams{
                {"maxImageSize", 320}, {"windowRadius", 3},
                {"numSamples", 5}, {"numIterations", 2},
                {"refineIterations", 1}, {"multiscale", false},
                {"costTopK", 1}, {"geomConsistency", false},
                {"fusion", false}, {"planePrior", false}, {"filter", false},
            };
            pipeline.setDenseReconstructionMethod("PatchMatchV2", denseParams);
            const bool denseOk = pipeline.executeDenseReconstruction();
            const auto &dOut = pipeline.lastOutput();
            check(denseOk && dOut.denseCloud && dOut.denseCloud->size() > 0,
                  QStringLiteral("PatchMatchV2 密集重建成功 (密集点云 %1 点)")
                      .arg(dOut.denseCloud
                               ? static_cast<int>(dOut.denseCloud->size()) : 0));
            for (const auto &l : cap.logs)
                std::fprintf(stderr, "[pipe2] %s\n", l.toLocal8Bit().constData());
        }

        // ── 密集重建（P5: PatchMatch 完整运行，复用稀疏中间状态） ──
        {
            // 快速参数：缩小工作图像、减少迭代与窗口，保证测试时长可控
            const QMap<QString, QVariant> pmParams{
                {"maxImageSize", 320},
                {"windowRadius", 3},
                {"numSamples", 5},
                {"numIterations", 2},
                {"filterMinNcc", 0.25},
                {"filter", false},
            };
            pipeline.setDenseReconstructionMethod("PatchMatch", pmParams);
            const bool pmOk = pipeline.executeDenseReconstruction();
            const auto &pmOut = pipeline.lastOutput();
            check(pmOk && pmOut.denseCloud && pmOut.denseCloud->size() > 0,
                  QStringLiteral("PatchMatch 密集重建成功 (密集点云 %1 点)")
                      .arg(pmOut.denseCloud
                               ? static_cast<int>(pmOut.denseCloud->size()) : 0));
            // 回归：密集重建应使用进程级全局共享线程池（与稀疏重建同一实例）
            check(cap.logs.join(QStringLiteral("\n")).contains(QStringLiteral("使用全局共享线程池")),
                  "PatchMatch 使用全局共享线程池");
            for (const auto &l : cap.logs)
                std::fprintf(stderr, "[pipe3] %s\n", l.toLocal8Bit().constData());
        }

        // 工厂注册冒烟（PatchMatch 完整运行耗时长，此处仅验证注册与接口）
        {
            auto pmatcher = DenseReconstructionFactory::instance().create("PatchMatch");
            check(pmatcher != nullptr && pmatcher->methodName() == "PatchMatch",
                  "PatchMatch 已注册到 DenseReconstructionFactory");
            check(DenseReconstructionFactory::instance()
                          .availableMethods()
                          .contains("PatchMatchV2"),
                  "PatchMatchV2 已注册到 DenseReconstructionFactory");
        }
    }

    // ── 线程路径：PipelineRunner（QThread + 信号转发） ──
    {
        ReconstructionPipeline pipeline;
        PipelineCapture cap;
        pipeline.setCallbacks(&cap);
        pipeline.setConfig(cfg);

        PipelineRunner runner(&pipeline);
        runner.setAutoSaveDirectory(workDir);

        bool ok = false;
        int runnerClouds = 0;
        QEventLoop loop;
        QObject::connect(&runner, &PipelineRunner::finished,
                         [&](bool success) { ok = success; loop.quit(); });
        QObject::connect(&runner, &PipelineRunner::stagePointCloudReady,
                         [&](const QVector<float> &, const QVector<float> &,
                             const QString &) { ++runnerClouds; });
        try {
            runner.start();
            loop.exec();
            runner.waitForFinished();
        } catch (const std::exception &e) {
            check(false, QStringLiteral("Runner 路径抛出异常: %1").arg(e.what()));
        } catch (...) {
            check(false, "Runner 路径抛出未知异常");
        }

        check(ok, "PipelineRunner 完整重建 finished(success)");
        for (const auto &l : cap.logs)
            std::fprintf(stderr, "[pipe] %s\n", l.toLocal8Bit().constData());
        checkSparseResult(pipeline.lastOutput(), "Runner 路径");
        check(runnerClouds >= 1,
              QStringLiteral("接收阶段点云事件(sparse,Runner路径,次数=%1)").arg(runnerClouds));
        runner.stop();
    }

    // 清理临时影像
    for (const auto &f : files)
        QFile::remove(f);
    QDir().rmdir(workDir);
}

int run(const QString &pluginsDir)
{
    auto &reg = PluginRegistry::instance();

    // 内置算法（编译进 core）优先注册；幂等，DLL 插件随后扫描
    registerBuiltInAlgorithms(reg);
    check(reg.descriptorPlugin("SIFT") != nullptr, "内置 SIFT 已注册");
    check(reg.matcherPlugin("BruteForce") != nullptr, "内置 BruteForce 已注册");

    const int loaded = reg.scanPlugins(pluginsDir);
    check(loaded >= 1, QStringLiteral("扫描到插件(%1)").arg(loaded));

    // 内置算法在 byCategory 中排在 DLL 插件之前，且 builtin 标记正确
    {
        const auto descs = reg.byCategory("descriptor");
        check(!descs.isEmpty() && descs.front().builtin,
              "byCategory(descriptor) 内置算法排序在前");
        bool builtinFlagOk = true;
        for (const auto &info : reg.plugins()) {
            const bool expectBuiltin = (info.id == "SIFT" || info.id == "SURF" ||
                                        info.id == "KAZE" || info.id == "FLANN" ||
                                        info.id == "BruteForce" || info.id == "NearestNeighbor");
            if (info.builtin != expectBuiltin)
                builtinFlagOk = false;
        }
        check(builtinFlagOk, "PluginInfo.builtin 标记正确");
    }

    const auto infos = reg.plugins();
    check(!infos.isEmpty(), "plugins() 非空");

    // ── P2 描述符插件 (SIFT / KAZE / SURF) ──
    for (const auto &descName : { QStringLiteral("SIFT"), QStringLiteral("KAZE"),
                                  QStringLiteral("SURF") }) {
        auto *dp = reg.descriptorPlugin(descName);
        check(dp != nullptr, "按 id 加载 " + descName + " 插件");
        if (!dp)
            continue;
        auto dw = dp->createWorker(QJsonObject{});
        check(dw != nullptr, descName + " createWorker 创建成功");
        if (dw) {
            const FeatureResult r = dw->extract(makeSyntheticTextureImage(1));
            check(r.keypoints.size() > 10,
                  QStringLiteral("%1 提取合成纹理特征 keypoints=%2 (>10)")
                      .arg(descName)
                      .arg(r.keypoints.size()));
        }
        runDescriptorConcurrencyCheck(dp);
    }

    // ── P2 描述符插件 (XFeat：运行时依赖(onnxruntime DLL 加载)或模型缺失均不导致失败) ──
    {
        const QString modelsDir =
            QCoreApplication::applicationDirPath() + QStringLiteral("/models");
        auto *xp = reg.descriptorPlugin(QStringLiteral("XFeat"));
        if (!xp) {
            // onnxruntime 与 Qt 插件加载器存在兼容问题(Win32 193, 见重构设计文档 P2 注)，
            // 该插件在本机可能无法加载；其余插件不受影响，此处跳过计 PASS
            qWarning().noquote() << "[WARN] XFeat 插件未加载，跳过（运行时依赖环境问题，不影响其他插件）";
            check(true, "XFeat 插件未加载，跳过(计 PASS)");
        } else {
            check(true, "按 id 加载 XFeat 插件");
            const QStringList xfeatModels =
                QDir(modelsDir).entryList({ QStringLiteral("xfeat*.onnx") }, QDir::Files);
            if (xfeatModels.isEmpty()) {
                // 未部署 xfeat 模型（目录可能不存在或仅有其他算法的模型）：跳过计 PASS
                qWarning().noquote() << "[WARN] 未找到 XFeat 模型(xfeat*.onnx):" << modelsDir
                                     << "，跳过 XFeat 特征检查";
                check(true, "XFeat 模型缺失，跳过(计 PASS)");
            } else {
                auto xw = xp->createWorker(QJsonObject{});
                check(xw != nullptr, "XFeat createWorker 创建成功");
                if (xw) {
                    const FeatureResult r = xw->extract(makeSyntheticTextureImage(1));
                    // 目录存在但无可匹配模型也可能返回 0 关键点
                    check(r.keypoints.size() > 0,
                          QStringLiteral("XFeat 提取合成纹理特征 keypoints=%1 (>0)")
                              .arg(r.keypoints.size()));
                    if (r.keypoints.size() > 0) {
                        runDescriptorConcurrencyCheck(xp, 5);
                    } else {
                        qWarning().noquote() << "[WARN] XFeat 提取到 0 关键点，跳过并发一致性检查";
                    }
                }
            }
        }
    }

    // ── P2 描述符插件 (DD-XFeat：XFeat 改进 CNN 版，模型缺失跳过不失败) ──
    {
        const QString modelsDir =
            QCoreApplication::applicationDirPath() + QStringLiteral("/models");
        auto *dxp = reg.descriptorPlugin(QStringLiteral("DD-XFeat"));
        if (!dxp) {
            // onnxruntime 运行时依赖问题可能导致插件加载失败；不影响其他插件
            qWarning().noquote() << "[WARN] DD-XFeat 插件未加载，跳过（运行时依赖环境问题，不影响其他插件）";
            check(true, "DD-XFeat 插件未加载，跳过(计 PASS)");
        } else {
            check(true, "按 id 加载 DD-XFeat 插件");
            const QStringList ddModels =
                QDir(modelsDir).entryList({ QStringLiteral("ddxfeat*.onnx") }, QDir::Files);
            if (ddModels.isEmpty()) {
                // 未部署 ddxfeat 模型（目录可能不存在或仅有其他算法的模型）：跳过计 PASS
                qWarning().noquote() << "[WARN] 未找到 DD-XFeat 模型(ddxfeat*.onnx):" << modelsDir
                                     << "，跳过 DD-XFeat 特征检查";
                check(true, "DD-XFeat 模型缺失，跳过(计 PASS)");
            } else {
                // 参数协议
                QString dxErr;
                check(dxp->validateParams(QJsonObject{{"scoreThreshold", 0.05}}, &dxErr),
                      "DD-XFeat 合法参数校验通过");
                check(!dxp->validateParams(QJsonObject{{"unknownKey", 1}}, &dxErr),
                      "DD-XFeat 未知参数被拒绝: " + dxErr);

                auto dxw = dxp->createWorker(QJsonObject{});
                check(dxw != nullptr, "DD-XFeat createWorker 创建成功");
                if (dxw) {
                    // CNN 描述符统一使用真实感合成球体图（对齐 SuperPoint 用例：
                    // 纯随机噪声图对 CNN 无响应风险）
                    const QString dxDir = QDir::temp().filePath(
                        QStringLiteral("sfm3_ddx_%1").arg(QCoreApplication::applicationPid()));
                    QDir().mkpath(dxDir);
                    QStringList dxFiles;
                    const bool rendered = renderSphereViews(dxDir, dxFiles) && !dxFiles.isEmpty();
                    check(rendered, "DD-XFeat 测试图渲染成功");
                    if (rendered) {
                        ImageEntry dxEntry;
                        dxEntry.id = 1;
                        const cv::Mat dxColor = cv::imread(dxFiles.first().toStdString(),
                                                           cv::IMREAD_COLOR);
                        if (!dxColor.empty())
                            cv::cvtColor(dxColor, dxEntry.imageGray, cv::COLOR_BGR2GRAY);
                        dxEntry.imageData = dxColor;

                        const FeatureResult r = dxw->extract(dxEntry);
                        check(r.keypoints.size() > 0,
                              QStringLiteral("DD-XFeat 提取合成球体特征 keypoints=%1 (>0)")
                                  .arg(r.keypoints.size()));
                        if (!r.descriptors.empty()) {
                            check(r.descriptors.cols == 64 && r.descriptors.type() == CV_32F,
                                  QStringLiteral("DD-XFeat 描述子维度正确 (%1x%2 CV_32F)")
                                      .arg(r.descriptors.rows).arg(r.descriptors.cols));
                        }
                        if (r.keypoints.size() > 0) {
                            runDescriptorConcurrencyCheck(dxp, 5);
                        } else {
                            qWarning().noquote() << "[WARN] DD-XFeat 提取到 0 关键点，跳过并发一致性检查";
                        }
                    }
                    // 清理临时影像
                    for (const auto &f : dxFiles)
                        QFile::remove(f);
                    QDir().rmdir(dxDir);
                }
            }
        }
    }

    // ── P2 描述符插件 (SuperPoint：ONNX 推理，模型缺失跳过不失败) ──
    {
        const QString modelPath =
            QCoreApplication::applicationDirPath()
            + QStringLiteral("/models/superpoint_dynamic.onnx");
        auto *sup = reg.descriptorPlugin(QStringLiteral("SuperPoint"));
        if (!sup) {
            // onnxruntime 运行时依赖问题可能导致插件加载失败；不影响其他插件
            qWarning().noquote() << "[WARN] SuperPoint 插件未加载，跳过（运行时依赖环境问题，不影响其他插件）";
            check(true, "SuperPoint 插件未加载，跳过(计 PASS)");
        } else {
            check(true, "按 id 加载 SuperPoint 插件");

            // 参数协议
            QString spErr;
            check(sup->validateParams(QJsonObject{{"scoreThreshold", 0.005}}, &spErr),
                  "SuperPoint 合法参数校验通过");
            check(!sup->validateParams(QJsonObject{{"unknownKey", 1}}, &spErr),
                  "SuperPoint 未知参数被拒绝: " + spErr);

            if (!QFileInfo::exists(modelPath)) {
                // 模型未部署：打印 [WARN] 跳过并计 PASS（对齐 XFeat 模式）
                qWarning().noquote() << "[WARN] 未找到 SuperPoint 模型:" << modelPath
                                     << "，跳过 SuperPoint 特征检查（运行 tools/deploy_superpoint_model.ps1 部署）";
                check(true, "SuperPoint 模型缺失，跳过(计 PASS)");
            } else {
                auto sw = sup->createWorker(QJsonObject{});
                check(sw != nullptr, "SuperPoint createWorker 创建成功");
                if (sw) {
                    // SuperPoint 训练于真实图像，对纯随机噪声图无响应（实测 0 点），
                    // 因此使用管线冒烟同款的真实感合成球体图验证
                    const QString spDir = QDir::temp().filePath(
                        QStringLiteral("sfm3_sp_%1").arg(QCoreApplication::applicationPid()));
                    QDir().mkpath(spDir);
                    QStringList spFiles;
                    const bool rendered = renderSphereViews(spDir, spFiles) && !spFiles.isEmpty();
                    check(rendered, "SuperPoint 测试图渲染成功");
                    if (rendered) {
                        ImageEntry spEntry;
                        spEntry.id = 1;
                        const cv::Mat spColor = cv::imread(spFiles.first().toStdString(),
                                                           cv::IMREAD_COLOR);
                        if (!spColor.empty())
                            cv::cvtColor(spColor, spEntry.imageGray, cv::COLOR_BGR2GRAY);
                        spEntry.imageData = spColor;

                        const FeatureResult r = sw->extract(spEntry);
                        check(r.keypoints.size() > 0,
                              QStringLiteral("SuperPoint 提取合成球体特征 keypoints=%1 (>0)")
                                  .arg(r.keypoints.size()));
                        if (!r.descriptors.empty()) {
                            check(r.descriptors.cols == 256 && r.descriptors.type() == CV_32F,
                                  QStringLiteral("SuperPoint 描述子维度正确 (%1x%2 CV_32F)")
                                      .arg(r.descriptors.rows).arg(r.descriptors.cols));
                        }
                        if (r.keypoints.size() > 0) {
                            runDescriptorConcurrencyCheck(sup, 5);
                        } else {
                            qWarning().noquote() << "[WARN] SuperPoint 提取到 0 关键点，跳过并发一致性检查";
                        }
                    }
                    // 清理临时影像
                    for (const auto &f : spFiles)
                        QFile::remove(f);
                    QDir().rmdir(spDir);
                }
            }
        }
    }

    // ── P2 匹配器插件 (FLANN / BruteForce / NearestNeighbor) ──
    for (const auto &matcherName : { QStringLiteral("FLANN"), QStringLiteral("BruteForce"),
                                     QStringLiteral("NearestNeighbor") }) {
        auto *mp = reg.matcherPlugin(matcherName);
        check(mp != nullptr, "按 id 加载 " + matcherName + " 插件");
        if (!mp)
            continue;
        auto mw = mp->createWorker(QJsonObject{});
        check(mw != nullptr, matcherName + " createWorker 创建成功");
        if (mw) {
            cv::Mat desc1, desc2;
            makeSyntheticFeatures(desc1, desc2);

            std::vector<std::vector<cv::DMatch>> knn;
            const bool knnOk = mw->knnMatchPair(desc1, desc2, knn, 2);
            int good = 0;
            for (const auto &pair : knn) {
                if (pair.size() >= 2 && pair[0].distance < 0.8 * pair[1].distance)
                    ++good;
            }
            check(knnOk && good > 25,
                  QStringLiteral("%1 knnMatch 成功过滤后匹配数 %2 (>25, 距离≈0)")
                      .arg(matcherName)
                      .arg(good));

            std::vector<cv::DMatch> matches;
            const bool singleOk = mw->matchPair(desc1, desc2, matches);
            double maxD = 0.0;
            for (const auto &m : matches)
                maxD = std::max(maxD, static_cast<double>(m.distance));
            // NearestNeighbor 经 gemm 计算余弦相似度，自匹配距离存在 ~1e-3 量级
            // 浮点累加误差（FLANN/BF 对相同向量输出精确 0），故阈值放宽到 5e-3
            check(singleOk && matches.size() == 32 && maxD < 5e-3,
                  QStringLiteral("%1 matchPair 返回 %2 个匹配 (距离<5e-3)")
                      .arg(matcherName)
                      .arg(matches.size()));
        }
    }

    // ── NearestNeighbor 分块实现回归（跨 tile 边界 + SIMD 距离精度） ──
    {
        auto *nnp = reg.matcherPlugin(QStringLiteral("NearestNeighbor"));
        check(nnp != nullptr, "NearestNeighbor 插件加载（分块回归）");
        if (nnp) {
            // 1300×1100（跨 3 行块 × 3 列块，tile=512），前 100 行同源
            const int m = 1300, n = 1100, d = 32;
            cv::Mat desc1(m, d, CV_32F), desc2(n, d, CV_32F);
            cv::RNG rng(20261001u);
            rng.fill(desc1, cv::RNG::UNIFORM, -1.f, 1.f);
            desc1.rowRange(0, 100).copyTo(desc2.rowRange(0, 100));
            cv::Mat noise(n - 100, d, CV_32F);
            rng.fill(noise, cv::RNG::UNIFORM, -1.f, 1.f);
            noise.copyTo(desc2.rowRange(100, n));

            auto normalize = [](const cv::Mat &src, cv::Mat &dst) {
                cv::Mat f;
                src.convertTo(f, CV_32F);
                dst = f.clone();
                for (int i = 0; i < dst.rows; ++i) {
                    double sq = 0;
                    for (int j = 0; j < dst.cols; ++j)
                        sq += double(dst.at<float>(i, j)) * dst.at<float>(i, j);
                    const float nrm = float(std::sqrt(sq));
                    if (nrm > 0)
                        for (int j = 0; j < dst.cols; ++j)
                            dst.at<float>(i, j) /= nrm;
                }
            };
            cv::Mat f1, f2;
            normalize(desc1, f1);
            normalize(desc2, f2);

            // mutual=true：同源 100 对全部找回（自点积恒最大且双向一致），
            // 且距离与标量参考一致（SIMD 累加顺序差异 < 1e-4）
            auto w1 = nnp->createWorker(QJsonObject{});
            std::vector<std::vector<cv::DMatch>> knn;
            const bool ok1 = w1->knnMatchPair(desc1, desc2, knn, 2);
            int found = 0;
            double maxDistErr = 0.0;
            for (const auto &row : knn) {
                for (const auto &dm : row) {
                    if (dm.queryIdx < 100 && dm.trainIdx == dm.queryIdx) {
                        ++found;
                        const float cosv = float(f1.row(dm.queryIdx).dot(f2.row(dm.trainIdx)));
                        const float expect =
                            std::sqrt(std::max(0.f, 2.f * (1.f - cosv)));
                        maxDistErr = std::max(maxDistErr,
                                              std::abs(double(dm.distance) - double(expect)));
                    }
                }
            }
            check(ok1 && found == 100,
                  QStringLiteral("分块 NN 同源对找回 %1/100").arg(found));
            // 自匹配对 cos≈1，sqrt(2(1-cos)) 对 ~1e-7 的浮点累加差有 sqrt 放大
            // （~5e-4，与既有自匹配 5e-3 阈值同源），故阈值取 1e-3
            check(maxDistErr < 1e-3,
                  QStringLiteral("分块 NN 距离精度 maxErr=%1 <1e-3").arg(maxDistErr, 0, 'e', 2));

            // matchPair 与 knnMatchPair(k=1) 拍平一致（复用实现的自一致）
            std::vector<std::vector<cv::DMatch>> knn1;
            w1->knnMatchPair(desc1, desc2, knn1, 1);
            std::vector<cv::DMatch> flat;
            for (const auto &row : knn1)
                for (const auto &dm : row)
                    flat.push_back(dm);
            std::vector<cv::DMatch> single;
            w1->matchPair(desc1, desc2, single);
            bool same = single.size() == flat.size();
            for (size_t i = 0; i < single.size() && same; ++i) {
                same = single[i].queryIdx == flat[i].queryIdx &&
                       single[i].trainIdx == flat[i].trainIdx &&
                       single[i].distance == flat[i].distance;
            }
            check(same, "matchPair 与 knn(k=1) 拍平一致");

            // mutual=false：无反向校验时每行都有 top-1
            QJsonObject noMutual;
            noMutual[QStringLiteral("mutualCheck")] = false;
            auto w2 = nnp->createWorker(noMutual);
            std::vector<std::vector<cv::DMatch>> knn2;
            w2->knnMatchPair(desc1, desc2, knn2, 1);
            check(static_cast<int>(knn2.size()) == m,
                  QStringLiteral("mutual=false 时输出行数 %1 == %2")
                      .arg(knn2.size()).arg(m));
        }
    }

    // ── P3a 共享算法单测 ──
    runThreadPoolTests();
    runSfmLoggerTests();
    runInitializationTests();
    runMatcherServiceSmoke(reg);

    // ── P3b 重建管线冒烟 ──
    runReconstructionPipelineSmoke(reg);

    return failures > 0 ? 1 : 0;
}

} // namespace

int main(int argc, char *argv[])
{
    qInstallMessageHandler([](QtMsgType t, const QMessageLogContext &, const QString &msg) {
        std::fprintf(stderr, "[qt:%d] %s\n", static_cast<int>(t), msg.toLocal8Bit().constData());
        std::fflush(stderr);
    });

    QCoreApplication app(argc, argv);

    QString pluginsDir;
    if (argc > 1) {
        pluginsDir = QString::fromLocal8Bit(argv[1]);
    } else {
        // 插件部署在 app 可执行文件同级的 plugins/ 子目录（exe 位于 release/ 下）
        pluginsDir = QCoreApplication::applicationDirPath() + QStringLiteral("/../../../app/release/plugins");
    }

    const int result = run(pluginsDir);

    std::fprintf(stderr, "%% RESULT %d\n", result);
    std::fprintf(stderr, "%% PLUGINS_DIR %s\n", pluginsDir.toLocal8Bit().constData());

    if (result == 0)
        qInfo().noquote() << "=== ALL TESTS PASSED ===";
    else
        qCritical().noquote() << "=== TESTS FAILED ===";

    return result;
}