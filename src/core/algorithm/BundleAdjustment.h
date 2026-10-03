#ifndef BUNDLEADJUSTMENT_H
#define BUNDLEADJUSTMENT_H

#include <QObject>
#include <array>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>

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
     * @return 是否收敛
     */
    bool localBundleAdjustment(
        std::vector<std::array<double, 6>> &cameras,
        std::vector<std::array<double, 3>> &points3d,
        const std::vector<std::tuple<int, int, cv::Point2d>> &observations,
        double fx, double fy, double cx, double cy,
        const std::vector<char> &cameraVariable,
        const std::vector<char> &pointVariable,
        int refCameraIdx = -1);

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
     * @return 是否收敛
     */
    bool globalBundleAdjustment(
        std::vector<std::array<double, 6>> &cameras,
        std::vector<std::array<double, 3>> &points3d,
        const std::vector<std::tuple<int, int, cv::Point2d>> &observations,
        double &fx, double &fy, double &cx, double &cy,
        bool fixIntrinsics = false,
        int refCameraIdx = 0,
        const std::vector<char> *cameraVariable = nullptr);

private:
    explicit BundleAdjustment(QObject *parent = nullptr);
    ~BundleAdjustment() override = default;
    BundleAdjustment(const BundleAdjustment &) = delete;
    BundleAdjustment &operator=(const BundleAdjustment &) = delete;

    static BundleAdjustment *s_instance;
};

#endif // BUNDLEADJUSTMENT_H