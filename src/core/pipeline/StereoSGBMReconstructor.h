#ifndef STEREOSGBMRECONSTRUCTOR_H
#define STEREOSGBMRECONSTRUCTOR_H

#include "IDenseReconstructor.h"
#include "data/StereoTypes.h"
#include "algorithm/StereoRectifier.h"

#include <QMap>
#include <QVariant>
#include <functional>

/**
 * @brief 基于 SGBM 立体匹配的密集重建实现
 *
 * 合并自原 DenseStereo 类的 SGBM 匹配与深度图生成逻辑，
 * 通过组合 StereoRectifier 复用极线校正部分。
 *
 * 同时实现 IDenseReconstructor 接口，使其可被工厂创建并接入流水线。
 *
 * 管线步骤：
 *   1. 选图策略 — 基于稀疏点云重叠度和基线角度，为每张视图选择最优匹配伙伴
 *   2. 极线校正 — 委托给 StereoRectifier
 *   3. SGBM立体匹配 — 半全局块匹配计算视差图，可选WLS后滤波
 *   4. 视差→深度 — 根据焦距和基线将视差转换为深度值
 */
class StereoSGBMReconstructor : public IDenseReconstructor
{
public:
    StereoSGBMReconstructor() = default;
    ~StereoSGBMReconstructor() override = default;

    // ── IDenseReconstructor 接口实现 ──
    QString methodName() const override { return QStringLiteral("StereoSGBM"); }
    QString displayName() const override { return QStringLiteral("SGBM 立体匹配"); }
    void setConfig(const QMap<QString, QVariant> &params) override;
    void setLogCallback(LogCallback callback) override
    {
        m_logCallback = callback;
        m_rectifier.setLogCallback(callback);
    }
    DenseReconstructionOutput reconstruct(const DenseReconstructionInput &input) override;

    // ── 暴露给其他重建器（如 PatchMatch）复用的完整管线 ──

    /**
     * @brief 选择立体匹配图像对
     *
     * 基于稀疏点云重叠度和基线角度，为每张视图选择最优的匹配伙伴。
     *
     * 评分机制：
     *   - 重叠率：Jaccard系数 = 共享特征数 / 并集特征数
     *   - 角度得分：高斯核 exp(-(angle-15)^2/200)，峰值在15度
     *   - 综合评分 = 重叠率 * 0.6 + 角度得分 * 0.4
     *
     * @param cameraPoses 所有相机的位姿
     * @param matchPairs  稀疏匹配结果
     * @param numImages   图像总数
     * @return 选中的立体匹配对列表
     */
    std::vector<StereoPairInfo> selectStereoPairs(
        const std::vector<CameraPose> &cameraPoses,
        const std::vector<MatchPair> &matchPairs,
        int numImages);

    /**
     * @brief 处理单个立体匹配对（完整流程）
     *
     * 依次执行：极线校正 → 匹配点坐标变换 → 立体匹配 → 视差转深度。
     * 图像在内部按需流式加载，处理完一对立即释放。
     *
     * @param pair 立体匹配对信息
     * @param matchedPoints1 左图匹配点（可选，用于自动视差估计）
     * @param matchedPoints2 右图匹配点（可选，用于自动视差估计）
     * @return 匹配结果
     */
    StereoPairResult processPair(
        const StereoPairInfo &pair,
        const std::vector<cv::Point2d> &matchedPoints1 = {},
        const std::vector<cv::Point2d> &matchedPoints2 = {});

    /**
     * @brief 处理所有立体匹配对（完整流程）
     *
     * 对每对图像依次执行：极线校正 → 立体匹配 → 视差转深度。
     * 自动从matchPairs中查找每对图像的匹配点用于视差范围估计。
     *
     * @param pairs 立体匹配对列表
     * @param matchPairs 稀疏匹配结果
     * @return 所有对的匹配结果
     */
    std::vector<StereoPairResult> processAllPairs(
        const std::vector<StereoPairInfo> &pairs,
        const std::vector<MatchPair> &matchPairs);

    /**
     * @brief 设置结构体形式的配置（供内部或测试直接使用）
     */
    void setStereoConfig(const DenseStereoConfig &config) { m_config = config; }

    /**
     * @brief 获取当前配置
     */
    const DenseStereoConfig &stereoConfig() const { return m_config; }

private:
    /**
     * @brief SGBM立体匹配 + WLS滤波
     *
     * 对校正后的图像对进行SGBM匹配，可选WLS后处理滤波。
     *
     * @param result [in/out] 校正后的图像对，输出视差图
     * @param matchedPoints1 左图匹配点（校正后坐标，用于自动视差范围）
     * @param matchedPoints2 右图匹配点（校正后坐标）
     * @return 成功返回true
     */
    bool stereoMatch(StereoPairResult &result,
                     const std::vector<cv::Point2d> &matchedPoints1 = {},
                     const std::vector<cv::Point2d> &matchedPoints2 = {});

    /**
     * @brief 将视差图转换为深度图
     *
     * 深度公式: depth = fx * baseline / disparity
     *
     * @param result [in/out] 包含视差图的结果，将填充depth字段
     * @return 成功返回true
     */
    bool disparityToDepth(StereoPairResult &result);

    /**
     * @brief 根据校正后的匹配点自动估计视差搜索范围
     *
     * @param points1 校正后左图匹配点
     * @param points2 校正后右图匹配点
     * @param minDisp [out] 最小视差
     * @param numDisp [out] 视差搜索范围（16的倍数）
     */
    void estimateDisparityRange(
        const std::vector<cv::Point2d> &points1,
        const std::vector<cv::Point2d> &points2,
        int &minDisp, int &numDisp) const;

    QMap<QString, QVariant> m_params;       ///< 接口传入的原始参数表
    DenseStereoConfig m_config;             ///< 当前配置
    StereoRectifier m_rectifier;            ///< 立体校正器
    int m_rectifyPairIdx = 0;               ///< 校正图像保存序号（调试用）
};

#endif // STEREOSGBMRECONSTRUCTOR_H
