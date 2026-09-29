#ifndef POINTCLOUDWIDGET_H
#define POINTCLOUDWIDGET_H

#include <QColor>
#include <QImage>
#include <QMatrix4x4>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLWidget>
#include <QPoint>
#include <QQuaternion>
#include <QStringList>
#include <QVector>
#include <QVector3D>

class QOpenGLTexture;

/**
 * @brief 三维点云渲染控件
 *
 * 基于 Qt QOpenGLWidget + OpenGL 3.3 Core Profile 实现的交互式点云查看器。
 * 相机模型参考 CloudCompare 的 ccGLWindowInterface：
 *   - 轨道相机（orbit camera）：以点云包围盒中心为 pivot，绕其旋转
 *   - 轨迹球旋转（trackball / arcball）
 *   - 相机推拉缩放（dolly）
 *   - 世界空间平移
 *   - 自动适配：载入点云后自动框选包围盒
 *
 * 增强（P4b）：
 *   - 着色模式切换（RGB / 单色）
 */
class PointCloudWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT

public:
    enum class ColorMode : int { Rgb = 0, Mono = 1 };

    explicit PointCloudWidget(QWidget *parent = nullptr);
    ~PointCloudWidget() override;

public slots:
    /// 全量替换当前点云数据（resetView=true 时自动适配视角）
    void updatePointCloud(const QVector<float> &points,
                          const QVector<float> &colors,
                          bool resetView = false);

    /// 向当前点云追加数据（resetView=true 时重新适配视角）
    void appendPointCloud(const QVector<float> &points,
                          const QVector<float> &colors,
                          bool resetView = false);

    /// 清空所有点云数据
    void clearPoints();

    /// 设置点云渲染点大小（像素）
    void setPointSize(float size);

    /// 设置着色模式（RGB / 单色）
    void setColorMode(ColorMode mode);

    /// 设置背景颜色
    void setBackgroundColor(const QColor &color);

    /// 重置视角到默认状态并自动适配点云包围盒
    void resetView();

    /// 俯视图（从 +Y 向 -Y 观察）
    void setTopView();

    /// 正视图（从 +Z 向 -Z 观察）
    void setFrontView();

    /// 侧视图（从 +X 向 -X 观察）
    void setSideView();

    /// 线框模式开关（点云无几何拓扑，此处仅作为渲染状态占位）
    void setWireframeMode(bool enabled);

    /// 设置相机位姿金字塔线框（成对线段端点 x,y,z 与对应 RGB 颜色，替换现有内容）。
    /// quadCorners：每相机 4 个底面角（相机系顺序 (-x,-y)(+x,-y)(+x,+y)(-x,+y)，
    /// 展开 4 项/相机），与 imagePaths 一一对应时异步加载图像缩略图贴到底面
    /// （UV 按从场景侧看正立的方向）。imagePaths 为空串的项不贴图。
    void setCameraFrusta(const QVector<float> &lineVertices,
                         const QVector<float> &lineColors,
                         const QVector<QVector3D> &quadCorners = {},
                         const QStringList &imagePaths = {});

    /// 清空相机位姿金字塔（含缩略图纹理）
    void clearCameraFrusta();

    /// 相机金字塔显示开关
    void setCameraFrustaVisible(bool visible);

    /// 已成功加载缩略图纹理的相机数量（测试自检用）
    int loadedFrustaTextureCount() const;

    /// 已接收、待 paintGL 上传的缩略图数量（测试自检用；offscreen 平台无渲染，
    /// 纹理创建不会发生，用此计数验证异步加载链路）
    int pendingFrustaTextureCount() const;

private slots:
    /// 后台缩略图加载完成（generation 校验防止旧任务回写到已替换的数据）
    void onFrustaThumbnailReady(int generation, int index, const QImage &image);

signals:
    /// 缩略图在工作线程加载完成后发射（经队列连接回主线程上传纹理）
    void frustaThumbnailReady(int generation, int index, const QImage &image);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    GLuint createProgram(const char *vertexSrc, const char *fragmentSrc);
    void updateBuffers();
    void updateLineBuffers();
    void updateQuadBuffers();
    void startThumbnailLoading();
    void updateModelViewMatrix();
    void fitToData();
    void recomputeBBox();
    QVector3D mapToTrackball(const QPoint &pos) const;
    void applyTrackballRotation(const QPoint &lastPos, const QPoint &currPos);

    GLuint m_shaderProgram = 0;
    GLuint m_vbo = 0;
    GLuint m_vao = 0;

    // 相机位姿金字塔线框（GL_LINES，端点成对；与点云共用着色器，颜色取 attribute 1）
    GLuint m_lineVao = 0;
    GLuint m_lineVbo = 0;
    QVector<float> m_lineVertices;
    QVector<float> m_lineColors;
    bool m_frustaVisible = true;

    // 相机缩略图四边形（GL_TRIANGLES，顶点交错 pos3+uv2；独立纹理着色器）
    GLuint m_quadShaderProgram = 0;
    GLuint m_quadVao = 0;
    GLuint m_quadVbo = 0;
    int m_quadVertexCount = 0;                  // 6 顶点/相机
    QVector<QVector3D> m_frustaQuadCorners;     // 每相机 4 个底面角（显示系）
    QStringList m_frustaImagePaths;             // 每相机对应图像路径（空串不贴图）
    QVector<QImage> m_frustaPendingTextures;    // 待上传缩略图（paintGL 中创建纹理，
                                                // 避免在未初始化 GL 上下文时跨线程上传）
    QVector<QOpenGLTexture *> m_frustaTextures; // 与相机序一致，未加载为 nullptr
    int m_frustaGeneration = 0;                 // 异步加载任务代数，替换数据时递增

    QMatrix4x4 m_projection;
    QMatrix4x4 m_modelView;

    QVector3D m_pivot;
    float m_cameraDistance = 5.0f;
    QQuaternion m_rotation;
    float m_fov = 45.0f;

    QVector<float> m_points;
    QVector<float> m_colors;

    QVector3D m_bboxMin;
    QVector3D m_bboxMax;
    bool m_bboxValid = false;

    float m_pointSize = 2.0f;
    QColor m_backgroundColor;
    bool m_wireframe = false;
    ColorMode m_colorMode = ColorMode::Rgb;

    QPoint m_lastMousePos;
    bool m_rotating = false;
    bool m_panning = false;
};

#endif // POINTCLOUDWIDGET_H