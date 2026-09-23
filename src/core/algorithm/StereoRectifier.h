#ifndef STEREORECTIFIER_H
#define STEREORECTIFIER_H

#include <functional>
#include <QString>
#include <opencv2/core.hpp>

#include "data/StereoTypes.h"

/**
 * @brief 立体校正器
 *
 * 负责立体匹配对图像的极线校正（rectification）。
 * 独立于具体匹配算法（SGBM/PatchMatch等），可为不同的密集重建方法复用。
 *
 * 校正目标：使两幅图像的极线水平对齐，将二维匹配搜索简化为一维搜索。
 *
 * 实现：自定义Bouguet算法变体（参考 Python 项目 StereoRectify-main），
 *       将基线旋转到x轴方向并自适应构造统一内参。
 *
 * 坐标系约定：
 *   - 位姿 R, t 描述世界到相机的变换: P_cam = R * P_world + t
 *   - 相机中心在世界坐标系下: C = -R^T * t
 */
class StereoRectifier
{
public:
    StereoRectifier() = default;
    ~StereoRectifier() = default;

    /**
     * @brief 设置日志回调函数
     * @param callback 日志回调，传入空 std::function 则恢复默认 qDebug 输出
     */
    void setLogCallback(std::function<void(const QString &)> callback)
    { m_logCallback = std::move(callback); }

    /**
     * @brief 对一对图像进行极线校正
     *
     * 处理流程：
     *   1. 计算相机1到相机2的相对位姿: R = R2 * R1^T, T = t2 - R2 * R1^T * t1
     *   2. 构造旋转矩阵 R_rect 将 T 旋转到x轴正方向，使基线水平
     *      （右相机位于 +x，同一空间点视差 d = x_left - x_right 为正）
     *   3. R1_rect = R_rect * R, R2_rect = R_rect
     *   4. 自适应构造统一新内参 K_new（左右相同）
     *   5. 通过角点投影计算新图像尺寸和主点
     *   6. 构造 P1, P2, Q 投影/重投影矩阵
     *   7. initUndistortRectifyMap + remap 进行图像重映射
     *
     * @param pair 立体匹配对信息（需包含K1/K2, R1/R2, t1/t2, image1/image2）
     * @param result [out] 输出校正结果（包含校正图像、Q矩阵、投影矩阵、基线等）
     * @return 成功返回true，输入为空或计算失败返回false
     */
    bool rectifyPair(const StereoPairInfo &pair, StereoPairResult &result);

    /**
     * @brief 计算两个相机之间的基线长度
     *
     * 相机中心: C = R^T * (-t)
     * 基线 = ||C1 - C2||
     */
    static double computeBaseline(const cv::Mat &R1, const cv::Mat &t1,
                                  const cv::Mat &R2, const cv::Mat &t2);

    /**
     * @brief 计算两个相机光轴之间的角度（度）
     *
     * 光轴方向在世界坐标系下: d = R^T * [0,0,1]^T = R的第3列
     * 角度 = arccos(d1 · d2)
     */
    static double computeViewAngle(const cv::Mat &R1, const cv::Mat &R2);

private:
    /// 内部日志输出：有回调则用回调，否则回退到 qDebug
    void log(const QString &message) const;

    std::function<void(const QString &)> m_logCallback;  ///< 日志回调
};

#endif // STEREORECTIFIER_H
