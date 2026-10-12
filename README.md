# SFM 3.0

Windows 单机桌面三维重建软件（C++17 / Qt Widgets / qmake）。主流程为两步：

1. **稀疏重建**：特征提取 → 匹配 → 增量式 SfM / BA → 稀疏点云  
2. **密集重建**：PatchMatch 立体匹配 → 深度融合 → 密集点云  

工程按 `sfm3.0.pro` 拆为 `src/core`（静态库）、`src/plugins`（描述符等插件）、`src/app`（主界面）、`src/tests`（测试）。另附 `SFM3-PluginSDK` 供第三方插件开发。

## 依赖

| 组件 | 建议版本 | 说明 |
|------|----------|------|
| Qt | 6.10.x（msvc2022_64） | Widgets / OpenGL |
| MSVC | Visual Studio 2022 | `win32-msvc` |
| OpenCV | 4.6（world，含 contrib） | 图像 / 特征 / 几何 |
| PCL | 1.15.x | 点云 IO 等（经 `SFM3_ALG_DEPS`） |
| Ceres | 2.2 | Bundle Adjustment（含 glog） |
| Open3D | 0.18 | 几何 / 点云相关 |
| ONNX Runtime | 1.26 | 仅学习型描述符插件链接 |

第三方根目录在 [`common.pri`](common.pri) 中配置，可用同名环境变量覆盖：

- `OPENCV_DIR`
- `ONNXRUNTIME_DIR`
- `PCL_ROOT`
- `CERES_ROOT`
- `OPEN3D_ROOT`

## 编译

1. 安装上述依赖，并按本机路径修改 `common.pri`（或设置环境变量）。  
2. 打开「x64 Native Tools」或先执行 `vcvars64.bat`。  
3. 在仓库根目录生成并编译（示例为 Release）：

```bat
mkdir build\release
cd build\release
qmake ..\..\sfm3.0.pro -spec win32-msvc "CONFIG+=release"
nmake
```

或用 Qt Creator 打开 `sfm3.0.pro`，选择 Desktop Qt 6.x MSVC 套件后构建。

产物大致位置：

- 主程序：`build/.../src/app/release/`
- 核心库：`build/.../src/core/sfmcore.lib`
- 插件：与 app 同级的 `plugins/`（由各插件 `.pro` 的 `DESTDIR` 决定）

可选测试：构建 `src/tests` 下各子工程（如 `ui_test`）后运行对应可执行文件。

## 说明

代码为纯AI生成
