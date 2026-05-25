# KYKT Vision Platform - Agent Handoff v0.4

> 升级任务交接文档 | 2026-05-25

---

## 项目概述

**KYKT Vision** 是一个基于 Tauri 2 + React + FastAPI 的桌面工作台，用于 3D 重建模型（DUSt3R/MASt3R/MonST3R 等）的执行、对比和评估。

### 技术栈

| 层 | 技术 |
|---|------|
| 前端 | React 18 + TypeScript + Vite |
| 桌面 | Tauri 2 (Rust) |
| 后端 | FastAPI + Python 3.10+ |
| 通信 | REST API + 轮询 (待升级 WebSocket) |

### 代码位置

```
E:\kykt\Coding\4.06\vision_ui\
├── client/                 # 前端 Tauri 应用
│   ├── src/               # React 源码
│   ├── src-tauri/         # Rust 桌面壳
│   └── package.json
├── app.py                 # FastAPI 后端
├── BACKEND_ROADMAP_v0.4.md  # 后端功能规划
└── local_jobs/            # 任务数据存储
```

---

## 当前状态 (v0.3.0)

### ✅ 已完成

1. **UI 专业化重构**
   - 设计系统: Figma/Linear 风格 Token
   - 图标系统: lucide-react SVG
   - 布局: 紧凑信息密集型

2. **Queue 工作区**
   - 统计条 (待派发/运行中/已完成/失败)
   - 双车道布局 (运行中 + 待派发卡片)
   - 快速派发/取消按钮

3. **Create 工作区**
   - 步骤式引导 (1→2→3→4)
   - 模型选择卡片
   - 批量对比模式

4. **System 工作区**
   - 卡片式信息展示
   - 部署状态表格

### 🔄 进行中

参见 `BACKEND_ROADMAP_v0.4.md`

---

## 下一轮任务 (P0)

### 任务 1: WebSocket 实时推送

**目标**: 替换 8 秒轮询，实现任务状态实时推送

**后端改动** (`app.py`):

```python
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        manager.active_connections[job_id].remove(websocket)
```

**前端改动** (`App.tsx`):

```typescript
useEffect(() => {
  if (!selectedJobId) return;
  const ws = new WebSocket(`ws://127.0.0.1:8000/ws/jobs/${selectedJobId}`);
  ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    // 更新任务状态
  };
  return () => ws.close();
}, [selectedJobId]);
```

---

### 任务 2: 批量操作 API

**后端新增端点**:

```python
@app.post("/api/jobs/batch-dispatch")
async def batch_dispatch(payload: BatchJobsRequest):
    results = []
    for job_id in payload.job_ids:
        try:
            await dispatch_job(job_id)
            results.append({"job_id": job_id, "success": True})
        except Exception as e:
            results.append({"job_id": job_id, "success": False, "error": str(e)})
    return {"results": results}

@app.post("/api/jobs/batch-cancel")
async def batch_cancel(payload: BatchJobsRequest):
    # 类似实现
```

**前端改动**:
- Queue 工作区添加多选模式
- 新增批量操作按钮

---

### 任务 3: 统计 API

```python
@app.get("/api/stats/overview")
async def get_stats_overview():
    jobs = load_all_jobs()
    return {
        "total": len(jobs),
        "by_status": Counter(j["status"] for j in jobs),
        "by_model": {
            model: {
                "count": count,
                "success_rate": calculate_success_rate(model),
                "avg_duration": calculate_avg_duration(model)
            }
            for model, count in Counter(j["model"] for j in jobs).items()
        }
    }
```

---

## 开发规范

### 文件命名

- 组件: `PascalCase.tsx` (e.g., `QueueWorkspace.tsx`)
- 工具函数: `camelCaseHelpers.ts`
- 样式: 统一在 `styles.css`

### CSS 变量

使用 `:root` 中定义的设计 Token:

```css
--bg-surface: #FFFFFF;
--text-primary: #111827;
--accent-primary: #2563EB;
--radius-md: 6px;
```

### 组件模式

```tsx
// 优先使用 section-card 布局
<div className="section-card">
  <div className="section-card-header">
    <h3 className="section-card-title">标题</h3>
    <span className="section-card-badge">徽章</span>
  </div>
  {/* 内容 */}
</div>
```

---

## 构建命令

```bash
# 开发
cd E:\kykt\Coding\4.06\vision_ui\client
npm run dev

# 构建前端
npm run build

# 构建桌面应用
npm run tauri build

# 后端
cd E:\kykt\Coding\4.06\vision_ui
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

---

## 验收标准

1. **功能验证**
   - [ ] WebSocket 连接成功，任务状态实时更新
   - [ ] 批量派发/取消功能正常
   - [ ] 统计数据正确显示

2. **性能验证**
   - [ ] 无轮询请求 (Network 面板检查)
   - [ ] 界面响应流畅

3. **兼容性**
   - [ ] Windows 10/11 桌面应用正常运行
   - [ ] 开发模式 (浏览器) 正常运行

---

## 已知问题

1. **SampleMatrixPanel** - 部分旧样式 class 可能需要清理
2. **模态框** - 需确保 `.settings-modal-backdrop` 样式完整
3. **Advisor 配置** - 已修复白屏问题 (全局 transition)

---

## 联系方式

如有问题，请查阅:
- `BACKEND_ROADMAP_v0.4.md` - 后端功能详细规划
- `UI_DESIGN_UPGRADE_PLAN.md` - UI 设计规范
- `BATCH_MODEL_COMPARE_ROADMAP.md` - 批量对比功能路线图

---

*Handoff Version: v0.4.0 | Date: 2026-05-25*
