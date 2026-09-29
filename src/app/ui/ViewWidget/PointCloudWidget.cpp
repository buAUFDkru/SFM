/**
 * @file PointCloudWidget.cpp
 * @brief 三维点云渲染控件实现（增强版：相机轨迹 + 着色模式）
 *
 * 相机模型为轨道相机（orbit camera），交互参考 CloudCompare ccGLWindowInterface：
 *   - 左键拖拽：轨迹球旋转，绕点云中心 pivot 转动
 *   - 右键拖拽：世界空间平移
 *   - 滚轮：相机推拉（dolly）
 */

#include "PointCloudWidget.h"

#include <QImageReader>
#include <QMouseEvent>
#include <QOpenGLTexture>
#include <QThreadPool>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
/// 缩略图最长边像素（纹理上限，控制显存与加载耗时）
constexpr int kThumbMaxSize = 256;
} // namespace

PointCloudWidget::PointCloudWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_backgroundColor(Qt::black)
{
    setFocusPolicy(Qt::StrongFocus);
    connect(this, &PointCloudWidget::frustaThumbnailReady,
            this, &PointCloudWidget::onFrustaThumbnailReady);
    resetView();
}

PointCloudWidget::~PointCloudWidget()
{
    makeCurrent();
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    if (m_vbo)
        glDeleteBuffers(1, &m_vbo);
    if (m_lineVao)
        glDeleteVertexArrays(1, &m_lineVao);
    if (m_lineVbo)
        glDeleteBuffers(1, &m_lineVbo);
    if (m_quadVao)
        glDeleteVertexArrays(1, &m_quadVao);
    if (m_quadVbo)
        glDeleteBuffers(1, &m_quadVbo);
    for (QOpenGLTexture *tex : m_frustaTextures)
        delete tex;
    if (m_shaderProgram)
        glDeleteProgram(m_shaderProgram);
    if (m_quadShaderProgram)
        glDeleteProgram(m_quadShaderProgram);
    doneCurrent();
}

// ── 视角控制 ──

void PointCloudWidget::resetView()
{
    m_rotation = QQuaternion();
    m_pivot = QVector3D(0.0f, 0.0f, 0.0f);
    m_cameraDistance = 5.0f;
    fitToData();
    updateModelViewMatrix();
}

void PointCloudWidget::setFrontView()
{
    m_rotation = QQuaternion();
    updateModelViewMatrix();
    update();
}

void PointCloudWidget::setTopView()
{
    m_rotation = QQuaternion::fromAxisAndAngle(1.0f, 0.0f, 0.0f, -90.0f);
    updateModelViewMatrix();
    update();
}

void PointCloudWidget::setSideView()
{
    m_rotation = QQuaternion::fromAxisAndAngle(0.0f, 1.0f, 0.0f, 90.0f);
    updateModelViewMatrix();
    update();
}

void PointCloudWidget::setWireframeMode(bool enabled)
{
    m_wireframe = enabled;
    update();
}

void PointCloudWidget::setPointSize(float size)
{
    m_pointSize = size;
    update();
}

void PointCloudWidget::setColorMode(ColorMode mode)
{
    m_colorMode = mode;
    update();
}

void PointCloudWidget::setBackgroundColor(const QColor &color)
{
    m_backgroundColor = color;
    update();
}

// ── OpenGL 初始化与着色器 ──

void PointCloudWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    m_shaderProgram = createProgram(
        R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aColor;
        uniform mat4 mvp;
        uniform float pointSize;
        uniform int monoColor;   // 0: RGB, 1: 单色（默认白色）
        out vec3 vColor;
        void main()
        {
            gl_Position = mvp * vec4(aPos, 1.0);
            gl_PointSize = pointSize;
            vColor = (monoColor == 1) ? vec3(0.9, 0.9, 0.9) : aColor;
        }
        )",
        R"(
        #version 330 core
        in vec3 vColor;
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(vColor, 1.0);
        }
        )");
    if (!m_shaderProgram) {
        qWarning("[PointCloudWidget] 点云着色器初始化失败");
        return;
    }

    // 缩略图四边形着色器（独立 program，attribute 1 为 UV 而非颜色）
    m_quadShaderProgram = createProgram(
        R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec2 aUV;
        uniform mat4 mvp;
        out vec2 vUV;
        void main()
        {
            gl_Position = mvp * vec4(aPos, 1.0);
            vUV = aUV;
        }
        )",
        R"(
        #version 330 core
        in vec2 vUV;
        uniform sampler2D frustaTex;
        out vec4 FragColor;
        void main()
        {
            FragColor = texture(frustaTex, vUV);
        }
        )");
    if (!m_quadShaderProgram)
        qWarning("[PointCloudWidget] 缩略图着色器初始化失败（仅影响缩略图显示）");

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenVertexArrays(1, &m_lineVao);
    glGenBuffers(1, &m_lineVbo);
    glGenVertexArrays(1, &m_quadVao);
    glGenBuffers(1, &m_quadVbo);

    updateBuffers();
    updateLineBuffers();
    updateQuadBuffers();
}

GLuint PointCloudWidget::createProgram(const char *vertexSrc, const char *fragmentSrc)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSrc, nullptr);
    glCompileShader(vertexShader);

    GLint success = 0;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, log);
        qWarning("[PointCloudWidget] 顶点着色器编译失败: %s", log);
        glDeleteShader(vertexShader);
        return 0;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSrc, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, log);
        qWarning("[PointCloudWidget] 片段着色器编译失败: %s", log);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        qWarning("[PointCloudWidget] 着色器程序链接失败: %s", log);
        glDeleteProgram(program);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

void PointCloudWidget::resizeGL(int w, int h)
{
    if (h == 0) h = 1;
    glViewport(0, 0, w, h);

    m_projection.setToIdentity();
    m_projection.perspective(m_fov, float(w) / float(h), 0.1f, 100000.0f);
}

void PointCloudWidget::paintGL()
{
    QColor c = m_backgroundColor;
    glClearColor(c.redF(), c.greenF(), c.blueF(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!m_shaderProgram)
        return;

    glUseProgram(m_shaderProgram);

    QMatrix4x4 mvp = m_projection * m_modelView;
    GLint mvpLoc = glGetUniformLocation(m_shaderProgram, "mvp");
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, mvp.constData());

    GLint psLoc = glGetUniformLocation(m_shaderProgram, "pointSize");
    glUniform1f(psLoc, m_pointSize);

    GLint monoLoc = glGetUniformLocation(m_shaderProgram, "monoColor");
    glUniform1i(monoLoc, (m_colorMode == ColorMode::Mono) ? 1 : 0);

    // 点云
    if (!m_points.isEmpty()) {
        glBindVertexArray(m_vao);
        glDrawArrays(GL_POINTS, 0, m_points.size() / 3);
        glBindVertexArray(0);
    }

    // 相机位姿金字塔线框（颜色取 attribute，不受单色模式影响）
    if (m_frustaVisible && !m_lineVertices.isEmpty()) {
        glUniform1i(monoLoc, 0);
        glLineWidth(1.5f);
        glBindVertexArray(m_lineVao);
        glDrawArrays(GL_LINES, 0, m_lineVertices.size() / 3);
        glBindVertexArray(0);
    }

    // 相机缩略图四边形（逐相机绘制；待上传缩略图在此惰性创建纹理）
    if (m_frustaVisible && m_quadShaderProgram && m_quadVertexCount > 0) {
        glUseProgram(m_quadShaderProgram);
        GLint quadMvpLoc = glGetUniformLocation(m_quadShaderProgram, "mvp");
        glUniformMatrix4fv(quadMvpLoc, 1, GL_FALSE, mvp.constData());
        GLint texLoc = glGetUniformLocation(m_quadShaderProgram, "frustaTex");
        glUniform1i(texLoc, 0);
        glActiveTexture(GL_TEXTURE0);
        const int quads = m_quadVertexCount / 6;
        for (int i = 0; i < quads; ++i) {
            // 惰性纹理上传：paintGL 时上下文必然已初始化
            if (!m_frustaTextures.value(i)
                && i < m_frustaPendingTextures.size()
                && !m_frustaPendingTextures[i].isNull()) {
                m_frustaTextures[i] = new QOpenGLTexture(m_frustaPendingTextures[i]);
                m_frustaTextures[i]->setMinMagFilters(QOpenGLTexture::Linear,
                                                      QOpenGLTexture::Linear);
                m_frustaTextures[i]->setWrapMode(QOpenGLTexture::ClampToEdge);
                m_frustaPendingTextures[i] = QImage();
            }
            if (!m_frustaTextures.value(i))
                continue;
            glBindVertexArray(m_quadVao);
            m_frustaTextures[i]->bind();
            glDrawArrays(GL_TRIANGLES, i * 6, 6);
            m_frustaTextures[i]->release();
        }
        glBindVertexArray(0);
        glUseProgram(m_shaderProgram);   // 恢复点/线着色器
    }
}

// ── VBO 上传 ──

void PointCloudWidget::updateBuffers()
{
    makeCurrent();
    if (!m_vao || !m_vbo) {
        doneCurrent();
        return;
    }

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    size_t pointBytes = m_points.size() * sizeof(float);
    size_t colorBytes = m_colors.size() * sizeof(float);
    size_t totalBytes = pointBytes + colorBytes;

    glBufferData(GL_ARRAY_BUFFER, totalBytes, nullptr, GL_STATIC_DRAW);
    if (pointBytes > 0)
        glBufferSubData(GL_ARRAY_BUFFER, 0, pointBytes, m_points.constData());
    if (colorBytes > 0)
        glBufferSubData(GL_ARRAY_BUFFER, pointBytes, colorBytes, m_colors.constData());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
                          reinterpret_cast<void *>(pointBytes));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    doneCurrent();
}

void PointCloudWidget::updateLineBuffers()
{
    makeCurrent();
    if (!m_lineVao || !m_lineVbo) {
        doneCurrent();
        return;
    }

    glBindVertexArray(m_lineVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_lineVbo);

    size_t vertexBytes = m_lineVertices.size() * sizeof(float);
    size_t colorBytes = m_lineColors.size() * sizeof(float);

    glBufferData(GL_ARRAY_BUFFER, vertexBytes + colorBytes, nullptr, GL_STATIC_DRAW);
    if (vertexBytes > 0)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertexBytes, m_lineVertices.constData());
    if (colorBytes > 0)
        glBufferSubData(GL_ARRAY_BUFFER, vertexBytes, colorBytes, m_lineColors.constData());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
                          reinterpret_cast<void *>(vertexBytes));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    doneCurrent();
}

// ── 相机位姿金字塔 ──

void PointCloudWidget::setCameraFrusta(const QVector<float> &lineVertices,
                                       const QVector<float> &lineColors,
                                       const QVector<QVector3D> &quadCorners,
                                       const QStringList &imagePaths)
{
    // 递增任务代数：作废仍在后台加载的旧缩略图任务
    ++m_frustaGeneration;

    m_lineVertices = lineVertices;
    m_lineColors = lineColors;
    updateLineBuffers();

    // 缩略图四边形：替换纹理数组并重建顶点缓冲
    qDeleteAll(m_frustaTextures);
    m_frustaTextures.clear();
    m_frustaPendingTextures.clear();
    const int cameraCount = quadCorners.size() / 4;
    const bool hasImages = cameraCount > 0 && imagePaths.size() == cameraCount;
    if (hasImages) {
        m_frustaQuadCorners = quadCorners;
        m_frustaImagePaths = imagePaths;
        m_frustaTextures.resize(cameraCount);
        m_frustaTextures.fill(nullptr);
    } else {
        m_frustaQuadCorners.clear();
        m_frustaImagePaths.clear();
    }
    updateQuadBuffers();

    if (hasImages)
        startThumbnailLoading();
    update();
}

void PointCloudWidget::clearCameraFrusta()
{
    ++m_frustaGeneration;
    qDeleteAll(m_frustaTextures);
    m_frustaTextures.clear();
    m_frustaPendingTextures.clear();
    m_frustaQuadCorners.clear();
    m_frustaImagePaths.clear();
    m_quadVertexCount = 0;
    updateQuadBuffers();
    update();
}

void PointCloudWidget::setCameraFrustaVisible(bool visible)
{
    m_frustaVisible = visible;
    update();
}

int PointCloudWidget::loadedFrustaTextureCount() const
{
    int count = 0;
    for (const QOpenGLTexture *tex : m_frustaTextures) {
        if (tex)
            ++count;
    }
    return count;
}

int PointCloudWidget::pendingFrustaTextureCount() const
{
    int count = 0;
    for (const QImage &img : m_frustaPendingTextures) {
        if (!img.isNull())
            ++count;
    }
    return count;
}

void PointCloudWidget::updateQuadBuffers()
{
    makeCurrent();
    if (!m_quadVao || !m_quadVbo) {
        doneCurrent();
        return;
    }

    // 交错顶点 pos(3)+uv(2)，每相机 6 顶点；UV 按从场景侧看正立的方向：
    // 观察者面朝 -z 时其"右"为相机系 -x、"上"为相机系 -y，
    // 故左上=c1(+x,-y)、右上=c0(-x,-y)、右下=c3(-x,+y)、左下=c2(+x,+y)
    QVector<float> interleaved;
    interleaved.reserve(m_frustaImagePaths.size() * 6 * 5);
    const float uv[6][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 1}};
    for (int i = 0; i < m_frustaImagePaths.size(); ++i) {
        const QVector3D c[4] = {
            m_frustaQuadCorners.at(i * 4 + 0), m_frustaQuadCorners.at(i * 4 + 1),
            m_frustaQuadCorners.at(i * 4 + 2), m_frustaQuadCorners.at(i * 4 + 3)};
        // 顶点次序（uv 见上）：c1, c0, c3,  c1, c3, c2
        const QVector3D verts[6] = {c[1], c[0], c[3], c[1], c[3], c[2]};
        for (int v = 0; v < 6; ++v) {
            interleaved << verts[v].x() << verts[v].y() << verts[v].z()
                        << uv[v][0] << uv[v][1];
        }
    }

    m_quadVertexCount = interleaved.size() / 5;

    glBindVertexArray(m_quadVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVbo);
    glBufferData(GL_ARRAY_BUFFER, interleaved.size() * sizeof(float),
                 interleaved.isEmpty() ? nullptr : interleaved.constData(),
                 GL_STATIC_DRAW);
    if (m_quadVertexCount > 0) {
        constexpr int stride = 5 * sizeof(float);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void *>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void *>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    doneCurrent();
}

void PointCloudWidget::startThumbnailLoading()
{
    const int generation = m_frustaGeneration;
    for (int i = 0; i < m_frustaImagePaths.size(); ++i) {
        const QString path = m_frustaImagePaths.at(i);
        const int index = i;
        if (path.isEmpty())
            continue;
        QThreadPool::globalInstance()->start([this, generation, index, path]() {
            QImageReader reader(path);
            reader.setAutoTransform(true);
            const QImage img = reader.read();
            if (img.isNull())
                return;
            const QImage thumb =
                img.scaled(kThumbMaxSize, kThumbMaxSize,
                           Qt::KeepAspectRatio, Qt::SmoothTransformation);
            if (thumb.isNull())
                return;
            emit frustaThumbnailReady(generation, index, thumb);
        });
    }
}

void PointCloudWidget::onFrustaThumbnailReady(int generation, int index, const QImage &image)
{
    // 数据已被替换（新项目/新金字塔），丢弃过期结果
    if (generation != m_frustaGeneration || index >= m_frustaImagePaths.size())
        return;

    // 只缓存 QImage，纹理推迟到 paintGL 中创建（彼时上下文必然可用），
    // 避免在未显示/未初始化 GL 的 widget 上 makeCurrent 导致崩溃
    if (m_frustaPendingTextures.size() < m_frustaImagePaths.size())
        m_frustaPendingTextures.resize(m_frustaImagePaths.size());
    m_frustaPendingTextures[index] = image;
    update();
}

// ── 轨道相机矩阵 ──

void PointCloudWidget::updateModelViewMatrix()
{
    m_modelView.setToIdentity();
    m_modelView.translate(0.0f, 0.0f, -m_cameraDistance);
    m_modelView.rotate(m_rotation);
    m_modelView.translate(-m_pivot);
}

// ── 点云数据接口 ──

void PointCloudWidget::updatePointCloud(const QVector<float> &points,
                                        const QVector<float> &colors,
                                        bool resetView)
{
    m_points = points;
    m_colors = colors;

    if (m_colors.size() < m_points.size()) {
        int needed = m_points.size() - m_colors.size();
        for (int i = 0; i < needed; ++i)
            m_colors.append(1.0f);
    }

    recomputeBBox();
    updateBuffers();

    if (resetView)
        fitToData();
    updateModelViewMatrix();
    update();
}

void PointCloudWidget::appendPointCloud(const QVector<float> &points,
                                        const QVector<float> &colors,
                                        bool resetView)
{
    m_points.append(points);
    m_colors.append(colors);

    if (m_colors.size() < m_points.size()) {
        int needed = m_points.size() - m_colors.size();
        for (int i = 0; i < needed; ++i)
            m_colors.append(1.0f);
    }

    recomputeBBox();
    updateBuffers();

    if (resetView)
        fitToData();
    updateModelViewMatrix();
    update();
}

void PointCloudWidget::clearPoints()
{
    m_points.clear();
    m_colors.clear();
    m_bboxValid = false;
    updateBuffers();
    update();
}

// ── 包围盒与自动适配 ──

void PointCloudWidget::recomputeBBox()
{
    if (m_points.isEmpty()) {
        m_bboxValid = false;
        return;
    }

    m_bboxMin = QVector3D(m_points[0], m_points[1], m_points[2]);
    m_bboxMax = m_bboxMin;

    for (int i = 3; i + 2 < m_points.size(); i += 3) {
        float x = m_points[i], y = m_points[i + 1], z = m_points[i + 2];
        m_bboxMin.setX(std::min(m_bboxMin.x(), x));
        m_bboxMin.setY(std::min(m_bboxMin.y(), y));
        m_bboxMin.setZ(std::min(m_bboxMin.z(), z));
        m_bboxMax.setX(std::max(m_bboxMax.x(), x));
        m_bboxMax.setY(std::max(m_bboxMax.y(), y));
        m_bboxMax.setZ(std::max(m_bboxMax.z(), z));
    }
    m_bboxValid = true;
}

void PointCloudWidget::fitToData()
{
    if (!m_bboxValid) {
        m_pivot = QVector3D(0.0f, 0.0f, 0.0f);
        m_cameraDistance = 5.0f;
        return;
    }

    m_pivot = (m_bboxMin + m_bboxMax) * 0.5f;

    QVector3D extent = m_bboxMax - m_bboxMin;
    float maxDim = std::max({extent.x(), extent.y(), extent.z()});
    if (maxDim < 1e-6f)
        maxDim = 1.0f;

    float halfFovRad = m_fov * 0.5f * kPi / 180.0f;
    m_cameraDistance = (maxDim * 0.5f) / std::tan(halfFovRad) * 1.2f;
    if (m_cameraDistance < 0.1f)
        m_cameraDistance = 0.1f;
}

// ── 轨迹球旋转 ──

QVector3D PointCloudWidget::mapToTrackball(const QPoint &pos) const
{
    float w = float(width());
    float h = float(height());
    if (w < 1.0f || h < 1.0f)
        return QVector3D(0.0f, 0.0f, 1.0f);

    float x = (2.0f * pos.x() - w) / w;
    float y = (h - 2.0f * pos.y()) / h;

    float d2 = x * x + y * y;
    if (d2 > 1.0f) {
        float s = 1.0f / std::sqrt(d2);
        return QVector3D(x * s, y * s, 0.0f);
    }
    return QVector3D(x, y, std::sqrt(1.0f - d2));
}

void PointCloudWidget::applyTrackballRotation(const QPoint &lastPos, const QPoint &currPos)
{
    QVector3D p0 = mapToTrackball(lastPos);
    QVector3D p1 = mapToTrackball(currPos);

    float dot = QVector3D::dotProduct(p0, p1);
    dot = std::clamp(dot, -1.0f, 1.0f);
    float angle = std::acos(dot);

    if (angle < 1e-4f)
        return;

    QVector3D axis = QVector3D::crossProduct(p0, p1);
    if (axis.lengthSquared() < 1e-10f)
        return;
    axis.normalize();

    QVector3D worldAxis = m_rotation.conjugated().rotatedVector(axis);

    float angleDeg = angle * 180.0f / kPi;
    QQuaternion q = QQuaternion::fromAxisAndAngle(worldAxis, angleDeg);
    m_rotation = q * m_rotation;
    m_rotation.normalize();
}

// ── 鼠标交互 ──

void PointCloudWidget::mousePressEvent(QMouseEvent *event)
{
    m_lastMousePos = event->pos();

    if (event->button() == Qt::LeftButton)
        m_rotating = true;
    else if (event->button() == Qt::RightButton)
        m_panning = true;
}

void PointCloudWidget::mouseMoveEvent(QMouseEvent *event)
{
    QPoint delta = event->pos() - m_lastMousePos;
    QPoint last = m_lastMousePos;
    m_lastMousePos = event->pos();

    if (m_rotating) {
        applyTrackballRotation(last, event->pos());
        updateModelViewMatrix();
        update();
    } else if (m_panning) {
        float halfFovRad = m_fov * 0.5f * kPi / 180.0f;
        float pixSize = (2.0f * m_cameraDistance * std::tan(halfFovRad))
                        / float(width());

        QVector3D camShift(delta.x() * pixSize, -delta.y() * pixSize, 0.0f);
        QVector3D worldShift = m_rotation.conjugated().rotatedVector(camShift);
        m_pivot -= worldShift;
        updateModelViewMatrix();
        update();
    }
}

void PointCloudWidget::mouseReleaseEvent(QMouseEvent *event)
{
    Q_UNUSED(event);
    m_rotating = false;
    m_panning = false;
}

void PointCloudWidget::wheelEvent(QWheelEvent *event)
{
    QPoint delta = event->angleDelta();
    if (delta.y() == 0)
        return;

    float steps = delta.y() / 120.0f;
    float factor = std::pow(0.9f, steps);
    m_cameraDistance *= factor;

    m_cameraDistance = std::clamp(m_cameraDistance, 0.01f, 50000.0f);

    updateModelViewMatrix();
    update();
}