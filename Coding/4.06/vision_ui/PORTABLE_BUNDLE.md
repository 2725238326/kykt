# KYKT Vision Client — Portable Bundle Notes

Last updated: 2026-05-03

## 桌面端寻找后端的顺序（0.3.x 起）

Tauri 端启动时按以下顺序定位 FastAPI 后端目录：

1. 环境变量 `KYKT_BACKEND_ROOT`，必须包含 `app.py` 与 `job_store.py`。
2. 开发态：从 `client/src-tauri` 上推两级（即原 vision_ui 项目根）。
3. 可执行文件 `kykt_vision_client.exe` 的祖先目录中第一个含 `app.py + job_store.py` 的，或这些祖先里名为 `backend/` 的子目录。
4. Tauri 资源目录里的 `backend/` 或资源目录本身。

只有 `app.py + job_store.py` 同时存在的目录会被认作合法后端根。**不再要求项目内必须有 `.venv\Scripts\python.exe`**，Python 解释器路径独立解析。

## Python 解释器解析顺序

`spawn_backend` 用如下顺序找 Python：

1. 环境变量 `KYKT_BACKEND_PYTHON`，指向具体 `python.exe`。
2. 后端根目录下 `.venv\Scripts\python.exe`（开发者首选）。
3. 后端根目录下 `python\python.exe`（portable 嵌入式）。
4. 与 `kykt_vision_client.exe` 同级或祖先里的 `python\python.exe`。
5. 系统 PATH 上的 `python.exe`（最后兜底，需自行装 requirements）。

第 3、4 项允许下面这种 portable 目录布局：

```text
kykt-vision-portable/
├─ kykt_vision_client.exe
├─ python/                      # 嵌入式 Python（python.exe + Lib/site-packages）
│   ├─ python.exe
│   └─ ...
└─ backend/                     # 后端根
    ├─ app.py
    ├─ job_store.py
    ├─ requirements.txt
    └─ ...
```

执行 `kykt_vision_client.exe` 时：
- 步骤 3 找到 `backend/` → 后端根 OK。
- 步骤 4 找到 `python/python.exe` → 解释器 OK。
- FastAPI 用嵌入 Python 在 `127.0.0.1:8765` 启动，UI 直连。

## 制作 portable 嵌入式 Python（建议）

下载 Python 3.11 embeddable zip：

```
https://www.python.org/ftp/python/3.11.x/python-3.11.x-embed-amd64.zip
```

1. 解压到 `kykt-vision-portable/python/`。
2. 编辑 `python311._pth`，去掉 `#import site` 前的 `#`，让 site-packages 生效。
3. 下载 `get-pip.py` 并跑：
   ```
   python\python.exe get-pip.py
   ```
4. 安装项目依赖：
   ```
   python\python.exe -m pip install -r backend\requirements.txt
   ```
5. 把 React 构建产物放到 `backend/client/dist/`（让浏览器直连也能拿到 React UI）。
6. 把整个 `kykt-vision-portable/` 打 ZIP 即可分发。

## 兼容性说明

- 旧的 `KYKT_BACKEND_ROOT` 用法不变，老用户无需改动。
- 已存在的开发态项目（含 `.venv`）行为完全一致——优先级依旧是项目内 venv。
- 若同时设了 `KYKT_BACKEND_PYTHON`，会跳过其它解释器候选，方便诊断。

## 打包前检查

- [ ] `client\dist\index.html` 已存在并最新（`npm run build`）。
- [ ] `python\python.exe` 在嵌入式 Python 下能 `python -c "import uvicorn, fastapi"` 成功。
- [ ] `kykt_vision_client.exe` 能从 `kykt-vision-portable/` 目录启动并打开 `127.0.0.1:8765`。
- [ ] `backend\local_jobs\_desktop\backend.log` 能成功写入第一行 `=== KYKT desktop backend start (...) ===`。
