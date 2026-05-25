# KYKT Vision Backend Roadmap v0.4

> 下一轮迭代规划 (2026-05)

---

## 一、核心能力增强

### 1.1 实时通信层 (P0)
| 功能 | 当前状态 | 目标 |
|------|----------|------|
| 任务状态推送 | 8s 轮询 | WebSocket 实时推送 |
| 日志流 | 静态读取 | SSE 实时流 |
| 进度更新 | 定时拉取 | 增量推送 |

**实现方案**:
```python
# app.py 新增
@app.websocket("/ws/jobs/{job_id}")
async def job_status_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    async for event in job_status_stream(job_id):
        await websocket.send_json(event)
```

### 1.2 批量操作 API (P0)
```
POST /api/jobs/batch-dispatch
POST /api/jobs/batch-cancel
POST /api/jobs/batch-retry
```

**请求体**:
```json
{
  "job_ids": ["job_001", "job_002"],
  "options": { "skip_validation": false }
}
```

### 1.3 统计与分析 API (P1)
```
GET /api/stats/overview
GET /api/stats/models/{model_id}
GET /api/stats/history?range=7d
```

**响应示例**:
```json
{
  "total_jobs": 156,
  "by_status": { "finished": 120, "running": 3, "failed": 8, "created": 25 },
  "by_model": {
    "dust3r": { "count": 45, "success_rate": 0.93, "avg_duration_sec": 127 }
  },
  "recent_24h": { "created": 12, "finished": 8 }
}
```

---

## 二、模型执行优化

### 2.1 任务队列调度器 (P1)
- **并发控制**: 限制同时运行任务数 (基于 GPU 显存)
- **优先级队列**: `high` / `normal` / `low`
- **自动重试**: 失败任务自动重试 (可配置次数)

```python
class JobScheduler:
    max_concurrent: int = 2
    retry_policy: RetryPolicy = RetryPolicy(max_retries=2, backoff="exponential")
    
    async def enqueue(self, job_id: str, priority: str = "normal"):
        ...
```

### 2.2 执行时间预估 (P1)
```
GET /api/models/{model_id}/estimate?input_size=10MB&file_count=5
```

**响应**:
```json
{
  "estimated_duration_sec": 180,
  "confidence": 0.85,
  "based_on_samples": 23
}
```

### 2.3 资源监控 (P2)
```
GET /api/system/resources
```

```json
{
  "gpu": { "name": "RTX 4090", "memory_used_gb": 12.3, "memory_total_gb": 24 },
  "cpu_percent": 45,
  "disk_free_gb": 128
}
```

---

## 三、对比与评估增强

### 3.1 评估指标自动计算 (P1)
- 深度图质量评估 (RMSE, AbsRel, δ<1.25)
- 点云密度/覆盖率
- 相机轨迹精度 (ATE, RPE)

```
GET /api/compare/samples/{sample_id}/metrics
```

### 3.2 可视化产物增强 (P2)
- 自动生成对比 GIF
- 深度图差异热力图
- 点云配准可视化

### 3.3 报告导出 (P1)
```
GET /api/compare/samples/{sample_id}/export?format=pdf
GET /api/compare/samples/{sample_id}/export?format=html
```

---

## 四、部署与分发

### 4.1 一键部署检查 (P0)
```
POST /api/deployment/check
```

**响应**:
```json
{
  "ready_models": ["dust3r", "mast3r"],
  "missing_dependencies": {
    "monst3r": ["depth-anything-v2 权重缺失"]
  },
  "recommendations": ["建议升级 PyTorch 至 2.2+"]
}
```

### 4.2 远程执行增强 (P2)
- SSH 连接池管理
- 断点续传支持
- 多节点负载均衡

### 4.3 配置导入/导出 (P1)
```
GET /api/config/export
POST /api/config/import
```

---

## 五、API 标准化

### 5.1 统一响应格式
```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": { "request_id": "req_xxx", "duration_ms": 45 }
}
```

### 5.2 分页规范
```
GET /api/jobs?page=1&page_size=20&sort=-created_at&status=running
```

### 5.3 错误码体系
| Code | 含义 |
|------|------|
| 1001 | 模型未部署 |
| 1002 | 输入文件无效 |
| 1003 | GPU 内存不足 |
| 2001 | 任务不存在 |
| 2002 | 任务状态冲突 |

---

## 六、优先级排序

| 优先级 | 功能 | 预估工时 |
|--------|------|----------|
| **P0** | WebSocket 实时推送 | 4h |
| **P0** | 批量操作 API | 2h |
| **P0** | 一键部署检查 | 2h |
| **P1** | 统计分析 API | 3h |
| **P1** | 任务队列调度器 | 6h |
| **P1** | 评估指标自动计算 | 8h |
| **P1** | 报告导出 | 4h |
| **P1** | 配置导入/导出 | 2h |
| **P2** | 资源监控 | 2h |
| **P2** | 可视化产物增强 | 6h |
| **P2** | 远程执行增强 | 8h |

---

## 七、前端适配要点

完成后端功能后，前端需同步更新：

1. **WebSocket 客户端** - 替换轮询逻辑
2. **批量选择 UI** - 任务列表支持多选
3. **统计仪表盘** - 新增 Dashboard 组件
4. **进度预估显示** - 任务卡片显示剩余时间
5. **错误码映射** - 友好错误提示

---

*文档版本: v0.4.0 | 更新时间: 2026-05-25*
