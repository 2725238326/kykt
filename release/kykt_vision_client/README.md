# KYKT Vision Client 发布包

更新时间：2026-04-20

这个目录是为了避免每次去 Tauri 的 `target/release` 深层目录里找文件。

## 推荐打开方式

如果只是你自己在当前电脑上使用，直接双击：

```text
kykt_vision_client.exe
```

本轮 3R 路线版本也保留了一个同内容的便捷副本：

```text
kykt_vision_client_3r_roadmap.exe
```

包含样例库 / 测评矩阵入口的最新副本：

```text
kykt_vision_client_3r_samples.exe
```

包含后端性能和 Tauri 健康检查优化的最新副本：

```text
kykt_vision_client_perf_backend.exe
```

包含任务人工评分与远端部署摘要的最新副本：

```text
kykt_vision_client_eval_deploy.exe
```

如果要安装到开始菜单或发给别人测试，用：

```text
KYKT Vision Client_0.1.0_x64-setup.exe
```

MSI 版本也可用：

```text
KYKT Vision Client_0.1.0_x64_en-US.msi
```

## 当前打包方式

这版桌面程序会自动托管本地后端：

1. 启动时先检查 `127.0.0.1:8765`。
2. 如果 FastAPI 已经在运行，就直接复用。
3. 如果没有运行，就从 `E:\kykt\Coding\4.06\vision_ui` 找到 `.venv\Scripts\python.exe`，自动执行 `python -m uvicorn app:app --host 127.0.0.1 --port 8765`。
4. 后端日志写到 `E:\kykt\Coding\4.06\vision_ui\local_jobs\_desktop\backend.log`。

最新体验修正：

- 2026-04-20：桌面 app 已重新构建，包含 3R 模型路线面板、模型 catalog、MASt3R smoke / MonST3R 标准样例状态，以及新的任务摘要状态修复。
- 2026-04-20：`帮助与系统` 和 `工作台` 可以直接看到当前可运行模型、待接入模型、暂缓前沿模型。
- 2026-04-20：后端 `/api/bootstrap` 会提供完整 `model_catalog`，为后续模型对比页做准备。
- 2026-04-21：桌面 app 已再次重新构建，包含 `/api/samples` 样例库接口、样例/测评矩阵面板、远端 3R 部署检查脚本和最新 active deployment 记录。
- 2026-04-21：桌面 app 已加入后端性能优化：Tauri 健康检查改用轻量 `/api/health`，样例清单按 mtime 缓存，任务日志改为尾部读取，空闲轮询频率降低。
- 2026-04-21：任务详情页现在支持人工评分持久化；系统页支持读取远端 active 3R 部署状态摘要。
- release 版不再弹出黑色命令行窗口。
- 打开时如果后端还没完全启动，会显示“正在准备本地服务”，不会直接显示裸 `fetch` 报错。
- 首页已经收敛为简洁工作台，高级参数默认折叠。
- 桌面端改用专用后端端口 `8765`，避免误连到旧的 `8000` 调试服务。

## 如果换目录

当前不是完全便携版，它依赖现有项目目录和 `.venv`。

如果你把项目移动到别的位置，需要设置环境变量：

```powershell
$env:KYKT_BACKEND_ROOT = "新的 vision_ui 目录"
```

以后如果要交付给完全没有项目环境的人，需要继续做“内置 Python runtime + 后端资源”的完整便携包。
