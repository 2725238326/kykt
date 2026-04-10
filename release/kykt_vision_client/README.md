# KYKT Vision Client 发布包

更新时间：2026-04-11

这个目录是为了避免每次去 Tauri 的 `target/release` 深层目录里找文件。

## 推荐打开方式

如果只是你自己在当前电脑上使用，直接双击：

```text
kykt_vision_client.exe
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
