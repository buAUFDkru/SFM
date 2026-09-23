#ifndef PLUGINTYPES_H
#define PLUGINTYPES_H

#include <QImage>
#include <QString>
#include <QVector>

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

/**
 * @brief 跨 DLL 共享的数据类型（app 与插件共用同一套 Qt/OpenCV 运行库，布局一致）。
 *
 * 仅使用 cv::Mat / QString 等运行时统一的类型；std::vector 用于经受控的确定性类型。
 */

/// 单张输入图像
struct ImageEntry
{
    int id = -1;
    QString filePath;
    QImage thumbnail;
    cv::Mat imageData;   ///< BGR
    cv::Mat imageGray;
};

/// 单张图像的特征提取结果
struct FeatureResult
{
    int imageId = -1;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat imageData;   ///< 原始BGR图像数据（用于 PnP/三角化阶段点云着色）
    double elapsedMs = 0.0;
};

/// 图像对匹配结果
struct MatchPairResult
{
    int imageId1 = -1;
    int imageId2 = -1;
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2d> points1, points2;
    int matchCount = 0;
    double avgDistance = 0.0;
};

#endif // PLUGINTYPES_H