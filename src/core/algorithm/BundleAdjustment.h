#ifndef BUNDLEADJUSTMENT_H
#define BUNDLEADJUSTMENT_H

#include <QObject>
#include <array>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>

/**
 * @brief 全局 BA 的内参精化选项（阶段五）
 *
 * 默认与阶段五之前完全一致：只精化一个共享焦距 fx=fy，主点固定，不碰畸变。
 * 这样即使调用方不传任何东西，行为也不会变（回归安全）。
 */
struct BAIntrinsicsOptions
{
    /**
     * 是否允许优化焦距（阶段五 5.3 的开关之一）
     *
     * 这是**门控位**：Triangulate 在注册相机数不足 intrinsicsMinCameras 时
     * 把它置 false，全局 BA 会把内参整块钉死（焦距、主点都不动）。
     * 与 refineFocalPerAxis 正交：后者管「fx 与 fy 是否分开」，前者管
     * 「焦距到底动不动」。
     */
    bool refineFocalLength = true;

    /// 5.3：fx 与 fy 各自独立优化（像素非正方形 / 主点方向标定不准时更贴合）。
    /// 默认关闭 —— 无人机相机像素是正方形，强加一个自由度只会让焦距与主点
    /// 更纠缠，而密集重建用的正是 fx/fy，分开后会引入非物理的长宽比。
    bool refineFocalPerAxis = false;

    /// 5.2：联合精化径向/切向畸变 k1,k2,p1,p2
    ///
    /// **默认 false** —— 实机三数据集对照（各跑 2 轮，结果稳定可复现）：
    /// 打开后重投影中位确实下降（106MEDIA 0.66→0.49px），但覆盖率腰斩
    /// （6389→2769 点，实测 k1≈+0.011 在画面角落的位移就有 ~10px），
    /// 是「模型更贴合、点云更稀」的取舍；点云是下游产物，取默认值关闭。
    /// 强畸变镜头（k1 绝对值 > 0.1）建议打开。
    bool refineDistortion = false;

    /// 畸变系数（OpenCV 顺序 k1,k2,p1,p2），输入输出。k3 固定为 0
    /// —— 广角镜头 2 个径向系数已足够，k3 与 k1/k2 强相关，一起放开会退化。
    double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;

    /// 焦距上下界 = 初值 × [focalLowerScale, focalUpperScale]
    double focalLowerScale = 0.5;
    double focalUpperScale = 4.0;

    /// k1/k2 的绝对值上界。0.5 对广角无人机镜头已是很宽松的范围
    /// （典型 k1 在 -0.3 ~ -0.05），越界说明优化已滑向非物理解。
    double maxRadialDistortion = 0.5;

    /// p1/p2 的绝对值上界（切向畸变通常 < 0.01）
    double maxTangentialDistortion = 0.1;
};

/**
 * @brief 光束法平差类（单例模式）
 *
 * 负责SFM重建中的光束法平差优化，包含两种优化模式：
 * 1. bundleAdjustment：局部BA，每加入一张新图像后执行，仅优化新相机位姿和三维点
 * 2. globalBundleAdjustment：全局BA + 自标定精化，优化所有内参、外参和三维点
 *
 * 两种方法均使用Ceres Solver中的Huber鲁棒核函数抑制外点影响。
 */
class BundleAdjustment : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 获取单例实例
     * @return BundleAdjustment实例指针
     */
    static BundleAdjustment *instance();

    /**
     * @brief 局部光束法平差 —— 每注册一张新图像后执行（COLMAP AdjustLocalBundle）
     *
     * 优化「局部图像集合」的位姿 + 「可变三维点集合」的坐标，其余保持固定。
     *
     * 阶段一只优化最新注册的 1 个相机和本轮新点，收敛范围过窄：
     * 新相机的误差会全部挤压到新点上，历史点无法被修正，漂移一旦产生
     * 就再也无法挽回。COLMAP 的做法是取与新图共享观测最多的 Top-N 张图
     * （ba_local_num_images = 6）一起优化，这也是局部BA能抑制漂移的关键。
     *
     * @param cameras 所有相机位姿数组 [rx, ry, rz, tx, ty, tz]
     * @param points3d 所有三维点坐标数组 [X, Y, Z]
     * @param observations 参与优化的观测（相机索引, 三维点索引, 像素坐标）
     *        由调用方裁剪到局部集合，避免无关残差拖慢求解
     * @param fx 焦距x（局部BA不优化内参）
     * @param fy 焦距y
     * @param cx 主点x
     * @param cy 主点y
     * @param cameraVariable 长度 = cameras.size()，1=参与优化，0=固定
     * @param pointVariable 长度 = points3d.size()，1=参与优化，0=固定
     * @param refCameraIdx 世界坐标系参考相机（固定，消除规范自由度）；
     *        < 0 表示不额外固定（此时由调用方保证至少有一个相机固定）
     * @param maxIterations Ceres 迭代上限。默认 25 —— 对齐 COLMAP
     *        ba_local_max_num_iterations：局部 BA 每张图都会跑，后续图像的 BA
     *        会继续修正同一批相机/点，没必要在一次里收敛到 1e-6 相对下降；
     *        实测原上限 100 时普遍要跑 43~61 次迭代，占整个重建 46% 的时间。
     * @return 是否收敛
     */
    bool localBundleAdjustment(
        std::vector<std::array<double, 6>> &cameras,
        std::vector<std::array<double, 3>> &points3d,
        const std::vector<std::tuple<int, int, cv::Point2d>> &observations,
        double fx, double fy, double cx, double cy,
        const std::vector<char> &cameraVariable,
        const std::vector<char> &pointVariable,
        int refCameraIdx = -1,
        int maxIterations = 25);

    /**
     * @brief 全局光束法平差 + 自标定精化 —— 重建流程末尾执行
     *
     * 联合优化所有相机外参（R, t）、所有三维点坐标和相机内参（f, cx, cy）。
     * 第一张图像的外参固定为世界坐标系参考，其余全部自由优化。
     * 使用HuberLoss鲁棒核函数（阈值4.0）抑制外点对优化的干扰。
     *
     * 内参约束：fx = fy = f（像素为正方形，单焦距参数），减少自由度避免退化。
     *
     * @param cameras 所有相机位姿数组 [rx, ry, rz, tx, ty, tz]，索引0恒定
     * @param points3d 所有三维点坐标数组 [X, Y, Z]
     * @param observations 所有观测数据
     * @param fx 输入输出：焦距x（参与精化，优化后fx=fy=f）
     * @param fy 输入输出：焦距y（参与精化，优化后fx=fy=f）
     * @param cx 输入输出：主点x（参与精化）
     * @param cy 输入输出：主点y（参与精化）
     * @param fixIntrinsics 是否固定内参
     * @param refCameraIdx 世界坐标系参考相机（该相机外参固定）。
     *        传入种子图像 ID 而非硬编码 0 —— 种子图像未必是第 0 张，
     *        固定一个无观测的相机会让整个系统存在 7 自由度规范自由度而漂移。
     * @param cameraVariable 可选：长度 = cameras.size()，1=参与优化，0=固定
     *        （未注册相机应传 0）。为 nullptr 时表示全部参与优化。
     * @param maxIterations Ceres 迭代上限。默认 50 —— 对齐 COLMAP
     *        ba_global_max_num_iterations（原上限 200，实测收尾三轮只用到
     *        25/16/16 次，200 只是给「不收敛时跑满」的兜底，反而可能拖到分钟级）。
     * @param intr 可选：阶段五的内参精化选项（fx≠fy / 畸变），**输入输出** ——
     *        k1,k2,p1,p2 会被原地更新为优化结果。nullptr 或 fixIntrinsics 为
     *        真时不生效，等价于阶段五之前的行为。
     * @return 是否收敛
     */
    bool globalBundleAdjustment(
        std::vector<std::array<double, 6>> &cameras,
        std::vector<std::array<double, 3>> &points3d,
        const std::vector<std::tuple<int, int, cv::Point2d>> &observations,
        double &fx, double &fy, double &cx, double &cy,
        bool fixIntrinsics = false,
        int refCameraIdx = 0,
        const std::vector<char> *cameraVariable = nullptr,
        int maxIterations = 50,
        BAIntrinsicsOptions *intr = nullptr);

private:
    explicit BundleAdjustment(QObject *parent = nullptr);
    ~BundleAdjustment() override = default;
    BundleAdjustment(const BundleAdjustment &) = delete;
    BundleAdjustment &operator=(const BundleAdjustment &) = delete;

    static BundleAdjustment *s_instance;
};

#endif // BUNDLEADJUSTMENT_H