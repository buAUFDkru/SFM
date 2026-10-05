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
#include <QOpenGLContext>
#include <QOpenGLTexture>
#include <QSurfaceFormat>
#include <QThreadPool>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr int kInteractivePointBudget = 750000;
constexpr int kWheelIdleMs = 120;
constexpr GLenum kGlMaxVertexAttribStride = 0x82E5;
/// 缩略图最长边像素（纹理上限，控制显存与加载耗时）
constexpr int kThumbMaxSize = 256;
} // namespace

PointCloudWidget::PointCloudWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_backgroundColor(Qt::black)
{
    setFocusPolicy(Qt::StrongFocus);
    m_thumbnailThreadPool.setMaxThreadCount(2);
    m_wheelIdleTimer.setSingleShot(true);
    m_wheelIdleTimer.setInterval(kWheelIdleMs);
    connect(&m_wheelIdleTimer, &QTimer::timeout, this, [this]() {
        m_wheelInteracting = false;
        update();
    });
    connect(this, &PointCloudWidget::frustaThumbnailReady,
            this, &PointCloudWidget::onFrustaThumbnailReady);
    resetView();
}

PointCloudWidget::~PointCloudWidget()
{
    ++m_frustaGeneration;
    m_thumbnailThreadPool.clear();
    m_thumbnailThreadPool.waitForDone();

    if (!m_glInitialized) {
        qDeleteAll(m_frustaTextures);
        return;
    }

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
    qDeleteAll(m_frustaTextures);
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
    m_glInitialized = true;

    const QOpenGLContext *glContext = QOpenGLContext::currentContext();
    if (glContext) {
        const QSurfaceFormat format = glContext->format();
        const bool supportsStrideLimit = format.majorVersion() > 4
                                         || (format.majorVersion() == 4
                                             && format.minorVersion() >= 4)
                                         || glContext->hasExtension(
                                             QByteArrayLiteral("GL_ARB_vertex_attrib_binding"));
        if (supportsStrideLimit) {
            GLint maxStride = 0;
            glGetIntegerv(kGlMaxVertexAttribStride, &maxStride);
            if (maxStride > 0)
                m_maxVertexAttribStride = maxStride;
        }
    }

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
    m_mvpLocation = glGetUniformLocation(m_shaderProgram, "mvp");
    m_pointSizeLocation = glGetUniformLocation(m_shaderProgram, "pointSize");
    m_monoColorLocation = glGetUniformLocation(m_shaderProgram, "monoColor");

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
    if (!m_quadShaderProgram) {
        qWarning("[PointCloudWidget] 缩略图着色器初始化失败（仅影响缩略图显示）");
    } else {
        m_quadMvpLocation = glGetUniformLocation(m_quadShaderProgram, "mvp");
        m_frustaTexLocation = glGetUniformLocation(m_quadShaderProgram, "frustaTex");
    }

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
    updateProjectionMatrix();
}

void PointCloudWidget::paintGL()
{
    uploadDirtyBuffers();

    const QColor c = m_backgroundColor;
    glClearColor(c.redF(), c.greenF(), c.blueF(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!m_shaderProgram)
        return;

    glUseProgram(m_shaderProgram);

    const QMatrix4x4 mvp = m_projection * m_modelView;
    glUniformMatrix4fv(m_mvpLocation, 1, GL_FALSE, mvp.constData());
    glUniform1f(m_pointSizeLocation, m_pointSize);
    glUniform1i(m_monoColorLocation, (m_colorMode == ColorMode::Mono) ? 1 : 0);

    // 点云：交互时通过同一 VBO 的 attribute stride 做均匀抽样。
    if (!m_points.isEmpty()) {
        const int pointStride = currentPointStride();
        configurePointAttributes(pointStride);
        const int pointCount = m_points.size() / 3;
        const int drawCount = (pointCount + pointStride - 1) / pointStride;
        glBindVertexArray(m_vao);
        glDrawArrays(GL_POINTS, 0, drawCount);
        glBindVertexArray(0);
    }

    // 相机位姿金字塔线框（颜色取 attribute，不受单色模式影响）
    if (m_frustaVisible && !m_lineVertices.isEmpty()) {
        glUniform1i(m_monoColorLocation, 0);
        glBindVertexArray(m_lineVao);
        glDrawArrays(GL_LINES, 0, m_lineVertices.size() / 3);
        glBindVertexArray(0);
    }

    // 相机缩略图四边形（逐相机绘制；待上传缩略图在此惰性创建纹理）
    if (m_frustaVisible && m_quadShaderProgram && m_quadVertexCount > 0) {
        glUseProgram(m_quadShaderProgram);
        glUniformMatrix4fv(m_quadMvpLocation, 1, GL_FALSE, mvp.constData());
        glUniform1i(m_frustaTexLocation, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(m_quadVao);
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
    m_pointBufferDirty = true;
    m_pointAttribStride = 0;
}

void PointCloudWidget::updateLineBuffers()
{
    m_lineBufferDirty = true;
}

void PointCloudWidget::uploadDirtyBuffers()
{
    if (m_pointBufferDirty && m_vao && m_vbo) {
        glBindVertexArray(m_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

        const GLsizeiptr pointBytes = GLsizeiptr(m_points.size()) * sizeof(float);
        const GLsizeiptr colorBytes = GLsizeiptr(m_colors.size()) * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, pointBytes + colorBytes, nullptr, GL_DYNAMIC_DRAW);
        if (pointBytes > 0)
            glBufferSubData(GL_ARRAY_BUFFER, 0, pointBytes, m_points.constData());
        if (colorBytes > 0)
            glBufferSubData(GL_ARRAY_BUFFER, pointBytes, colorBytes, m_colors.constData());

        m_pointBufferDirty = false;
        m_pointAttribStride = 0;
    }

    if (m_lineBufferDirty && m_lineVao && m_lineVbo) {
        glBindVertexArray(m_lineVao);
        glBindBuffer(GL_ARRAY_BUFFER, m_lineVbo);

        const GLsizeiptr vertexBytes = GLsizeiptr(m_lineVertices.size()) * sizeof(float);
        const GLsizeiptr colorBytes = GLsizeiptr(m_lineColors.size()) * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, vertexBytes + colorBytes, nullptr, GL_DYNAMIC_DRAW);
        if (vertexBytes > 0)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertexBytes, m_lineVertices.constData());
        if (colorBytes > 0)
            glBufferSubData(GL_ARRAY_BUFFER, vertexBytes, colorBytes, m_lineColors.constData());

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void *>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
                              reinterpret_cast<void *>(vertexBytes));
        glEnableVertexAttribArray(1);
        m_lineBufferDirty = false;
    }

    if (m_quadBufferDirty && m_quadVao && m_quadVbo) {
        glBindVertexArray(m_quadVao);
        glBindBuffer(GL_ARRAY_BUFFER, m_quadVbo);
        const GLsizeiptr quadBytes = GLsizeiptr(m_quadVertices.size()) * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, quadBytes,
                     m_quadVertices.isEmpty() ? nullptr : m_quadVertices.constData(),
                     GL_DYNAMIC_DRAW);
        if (m_quadVertexCount > 0) {
            constexpr GLsizei stride = 5 * sizeof(float);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void *>(0));
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void *>(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
        }
        m_quadBufferDirty = false;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void PointCloudWidget::configurePointAttributes(int pointStride)
{
    const int maxPointStride = std::max(1, m_maxVertexAttribStride / int(3 * sizeof(float)));
    pointStride = std::clamp(pointStride, 1, maxPointStride);
    if (pointStride == m_pointAttribStride)
        return;

    const GLsizei strideBytes = GLsizei(pointStride * 3 * sizeof(float));
    const GLsizeiptr pointBytes = GLsizeiptr(m_points.size()) * sizeof(float);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, strideBytes,
                          reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, strideBytes,
                          reinterpret_cast<void *>(pointBytes));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    m_pointAttribStride = pointStride;
}

int PointCloudWidget::currentPointStride() const
{
    if ((!m_rotating && !m_panning && !m_wheelInteracting) || m_points.isEmpty())
        return 1;

    const int pointCount = m_points.size() / 3;
    const int desiredStride = 1 + (pointCount - 1) / kInteractivePointBudget;
    const int maxPointStride = std::max(1, m_maxVertexAttribStride / int(3 * sizeof(float)));
    return std::min(desiredStride, maxPointStride);
}

// ── 相机位姿金字塔 ──

void PointCloudWidget::setCameraFrusta(const QVector<float> &lineVertices,
                                       const QVector<float> &lineColors,
                                       const QVector<QVector3D> &quadCorners,
                                       const QStringList &imagePaths)
{
    if (lineVertices == m_lineVertices && lineColors == m_lineColors
        && quadCorners == m_frustaQuadCorners && imagePaths == m_frustaImagePaths) {
        return;
    }

    // 递增任务代数并丢弃尚未开始的旧任务，运行中的任务由 generation 校验隔离。
    ++m_frustaGeneration;
    m_thumbnailThreadPool.clear();

    m_lineVertices = lineVertices;
    m_lineColors = lineColors;

    releaseFrustaTextures();
    m_frustaPendingTextures.clear();
    m_frustaQuadCorners = quadCorners;
    m_frustaImagePaths = imagePaths;
    m_quadVertices.clear();

    const int cameraCount = quadCorners.size() / 4;
    const bool hasImages = cameraCount > 0 && imagePaths.size() == cameraCount;
    if (hasImages) {
        m_frustaTextures.resize(cameraCount);
        m_frustaTextures.fill(nullptr);
        m_quadVertices.reserve(cameraCount * 6 * 5);

        // 交错顶点 pos(3)+uv(2)，每相机 6 顶点；成员存储保证惰性上传时数据有效。
        const float uv[6][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 1}};
        for (int i = 0; i < cameraCount; ++i) {
            const QVector3D c[4] = {
                quadCorners.at(i * 4 + 0), quadCorners.at(i * 4 + 1),
                quadCorners.at(i * 4 + 2), quadCorners.at(i * 4 + 3)};
            const QVector3D verts[6] = {c[0], c[1], c[2], c[0], c[2], c[3]};
            for (int v = 0; v < 6; ++v) {
                m_quadVertices << verts[v].x() << verts[v].y() << verts[v].z()
                               << uv[v][0] << uv[v][1];
            }
        }
    }
    m_quadVertexCount = m_quadVertices.size() / 5;
    recomputeFrustaBBox();
    updateLineBuffers();
    updateQuadBuffers();
    updateProjectionMatrix();

    if (hasImages)
        startThumbnailLoading();
    update();
}

void PointCloudWidget::clearCameraFrusta()
{
    ++m_frustaGeneration;
    m_thumbnailThreadPool.clear();
    releaseFrustaTextures();
    m_frustaPendingTextures.clear();
    m_lineVertices.clear();
    m_lineColors.clear();
    m_frustaQuadCorners.clear();
    m_frustaImagePaths.clear();
    m_quadVertices.clear();
    m_quadVertexCount = 0;
    recomputeFrustaBBox();
    updateLineBuffers();
    updateQuadBuffers();
    updateProjectionMatrix();
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

int PointCloudWidget::displayedPointCount() const
{
    return m_points.size() / 3;
}

void PointCloudWidget::updateQuadBuffers()
{
    m_quadBufferDirty = true;
}

void PointCloudWidget::releaseFrustaTextures()
{
    if (m_frustaTextures.isEmpty())
        return;

    const bool needsContext = m_glInitialized
                              && QOpenGLContext::currentContext() != context();
    if (needsContext)
        makeCurrent();
    qDeleteAll(m_frustaTextures);
    m_frustaTextures.clear();
    if (needsContext)
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
        m_thumbnailThreadPool.start([this, generation, index, path]() {
            QImageReader reader(path);
            reader.setAutoTransform(true);
            QSize decodedSize = reader.size();
            if (decodedSize.isValid()
                && (decodedSize.width() > kThumbMaxSize
                    || decodedSize.height() > kThumbMaxSize)) {
                decodedSize.scale(kThumbMaxSize, kThumbMaxSize, Qt::KeepAspectRatio);
                reader.setScaledSize(decodedSize);
            }
            QImage thumb = reader.read();
            if (thumb.isNull())
                return;
            // 少数图像插件不支持读取尺寸或解码期缩放，兜底保证纹理最长边不超过上限。
            if (thumb.width() > kThumbMaxSize || thumb.height() > kThumbMaxSize) {
                thumb = thumb.scaled(kThumbMaxSize, kThumbMaxSize,
                                     Qt::KeepAspectRatio, Qt::SmoothTransformation);
            }
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
    updateProjectionMatrix();
}

void PointCloudWidget::updateProjectionMatrix()
{
    const float aspect = float(std::max(1, width())) / float(std::max(1, height()));
    const float radius = sceneRadius();
    const float depthPadding = std::max(radius * 1.1f, 0.01f);
    const float minNear = std::max(m_cameraDistance * 0.001f, 0.0001f);
    const float nearPlane = std::max(m_cameraDistance - depthPadding, minNear);
    const float farPlane = std::max(m_cameraDistance + depthPadding,
                                    nearPlane + std::max(radius * 0.01f, 0.01f));

    m_projection.setToIdentity();
    m_projection.perspective(m_fov, aspect, nearPlane, farPlane);
}

void PointCloudWidget::recomputeFrustaBBox()
{
    m_frustaBBoxValid = false;
    auto includePoint = [this](float x, float y, float z) {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            return;

        if (!m_frustaBBoxValid) {
            m_frustaBBoxMin = QVector3D(x, y, z);
            m_frustaBBoxMax = m_frustaBBoxMin;
            m_frustaBBoxValid = true;
        } else {
            m_frustaBBoxMin.setX(std::min(m_frustaBBoxMin.x(), x));
            m_frustaBBoxMin.setY(std::min(m_frustaBBoxMin.y(), y));
            m_frustaBBoxMin.setZ(std::min(m_frustaBBoxMin.z(), z));
            m_frustaBBoxMax.setX(std::max(m_frustaBBoxMax.x(), x));
            m_frustaBBoxMax.setY(std::max(m_frustaBBoxMax.y(), y));
            m_frustaBBoxMax.setZ(std::max(m_frustaBBoxMax.z(), z));
        }
    };

    for (int i = 0; i + 2 < m_lineVertices.size(); i += 3)
        includePoint(m_lineVertices[i], m_lineVertices[i + 1], m_lineVertices[i + 2]);
    for (const QVector3D &corner : m_frustaQuadCorners)
        includePoint(corner.x(), corner.y(), corner.z());
}

// ── 点云数据接口 ──

void PointCloudWidget::updatePointCloud(const QVector<float> &points,
                                        const QVector<float> &colors,
                                        bool resetView)
{
    m_points.clear();
    m_colors.clear();
    m_bboxValid = false;
    m_points.reserve((points.size() / 3) * 3);
    m_colors.reserve((points.size() / 3) * 3);
    appendDisplayPoints(points, colors);
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
    appendDisplayPoints(points, colors);
    updateBuffers();

    if (resetView)
        fitToData();
    updateModelViewMatrix();
    update();
}

void PointCloudWidget::appendDisplayPoints(const QVector<float> &points,
                                           const QVector<float> &colors)
{
    const int sourcePointCount = points.size() / 3;
    m_points.reserve(m_points.size() + sourcePointCount * 3);
    m_colors.reserve(m_colors.size() + sourcePointCount * 3);

    for (int pointIndex = 0; pointIndex < sourcePointCount; ++pointIndex) {
        const int offset = pointIndex * 3;
        const float x = points[offset];
        const float y = points[offset + 1];
        const float z = points[offset + 2];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            continue;

        m_points << x << y << z;
        if (offset + 2 < colors.size()
            && std::isfinite(colors[offset])
            && std::isfinite(colors[offset + 1])
            && std::isfinite(colors[offset + 2])) {
            m_colors << colors[offset] << colors[offset + 1] << colors[offset + 2];
        } else {
            m_colors << 1.0f << 1.0f << 1.0f;
        }

        const QVector3D point(x, y, z);
        if (!m_bboxValid) {
            m_bboxMin = point;
            m_bboxMax = point;
            m_bboxValid = true;
        } else {
            m_bboxMin.setX(std::min(m_bboxMin.x(), x));
            m_bboxMin.setY(std::min(m_bboxMin.y(), y));
            m_bboxMin.setZ(std::min(m_bboxMin.z(), z));
            m_bboxMax.setX(std::max(m_bboxMax.x(), x));
            m_bboxMax.setY(std::max(m_bboxMax.y(), y));
            m_bboxMax.setZ(std::max(m_bboxMax.z(), z));
        }
    }
}

void PointCloudWidget::clearPoints()
{
    m_points.clear();
    m_colors.clear();
    m_bboxValid = false;
    updateBuffers();
    updateProjectionMatrix();
    update();
}

// ── 包围盒与自动适配 ──

float PointCloudWidget::bboxRadius() const
{
    if (!m_bboxValid)
        return 1.0f;

    const float radius = (m_bboxMax - m_bboxMin).length() * 0.5f;
    return (std::isfinite(radius) && radius >= 1e-6f) ? radius : 1.0f;
}

float PointCloudWidget::sceneRadius() const
{
    float radius = 0.0f;
    auto includeBounds = [this, &radius](const QVector3D &minimum,
                                         const QVector3D &maximum) {
        for (int mask = 0; mask < 8; ++mask) {
            const QVector3D corner((mask & 1) ? maximum.x() : minimum.x(),
                                   (mask & 2) ? maximum.y() : minimum.y(),
                                   (mask & 4) ? maximum.z() : minimum.z());
            radius = std::max(radius, (corner - m_pivot).length());
        }
    };

    if (m_bboxValid)
        includeBounds(m_bboxMin, m_bboxMax);
    if (m_frustaBBoxValid)
        includeBounds(m_frustaBBoxMin, m_frustaBBoxMax);

    return (std::isfinite(radius) && radius >= 1e-6f) ? radius : 1.0f;
}

void PointCloudWidget::fitToData()
{
    if (!m_bboxValid) {
        m_pivot = QVector3D(0.0f, 0.0f, 0.0f);
        m_cameraDistance = 5.0f;
        return;
    }

    m_pivot = (m_bboxMin + m_bboxMax) * 0.5f;

    const float aspect = float(std::max(1, width())) / float(std::max(1, height()));
    const float halfVerticalFov = m_fov * 0.5f * kPi / 180.0f;
    const float halfHorizontalFov = std::atan(std::tan(halfVerticalFov) * aspect);
    const float limitingHalfFov = std::max(0.01f,
                                           std::min(halfVerticalFov, halfHorizontalFov));
    m_cameraDistance = bboxRadius() / std::sin(limitingHalfFov) * 1.1f;
    m_cameraDistance = std::max(m_cameraDistance, 0.01f);
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

    if (event->button() == Qt::LeftButton) {
        m_rotating = true;
        update();
    } else if (event->button() == Qt::RightButton) {
        m_panning = true;
        update();
    }
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
    update();
}

void PointCloudWidget::wheelEvent(QWheelEvent *event)
{
    QPoint delta = event->angleDelta();
    if (delta.y() == 0)
        return;

    float steps = delta.y() / 120.0f;
    float factor = std::pow(0.9f, steps);
    m_cameraDistance *= factor;

    const float maxDistance = std::max(50000.0f, sceneRadius() * 100.0f);
    m_cameraDistance = std::clamp(m_cameraDistance, 0.01f, maxDistance);
    m_wheelInteracting = true;
    m_wheelIdleTimer.start();

    updateModelViewMatrix();
    update();
}