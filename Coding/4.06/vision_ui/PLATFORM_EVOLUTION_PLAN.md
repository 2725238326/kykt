# KYKT Vision Platform 演进规划

> 2026-05-26 · 基于当前代码库和路线图整理

---

## 一、后端现存可优化项

### 1.1 已完成 vs 待完善

| 功能 | 状态 | 待优化点 |
|------|------|----------|
| WebSocket 实时推送 | ✅ 已实现 | 前端仍有轮询逻辑未全部替换；日志流未走 SSE |
| 批量操作 API | ✅ 已实现 | 缺 `batch-retry`；无原子性回滚 |
| 统计分析 API | ✅ 基础 | 缺历史趋势 `/api/stats/history`、模型维度 `/api/stats/models/{id}` |
| 任务调度器 | ✅ 已实现 | 未与 `dispatch_job_api` 真正串联；缺动态并发调整（按 GPU 显存） |
| 评估指标 | ✅ 框架 | 缺 GT 对比路径；点云配准指标（Chamfer/F-Score）未实现 |
| 报告导出 | ✅ HTML/PDF | PDF 依赖 weasyprint/pdfkit 未在 requirements.txt 声明 |
| 资源监控 | ✅ 本地 | 仅监控本地，未监控远程 GPU 服务器 |
| 可视化产物 | ✅ 基础 | 缺点云配准渲染、3D 轨迹可视化 |
| 配置导入/导出 | ❌ 未实现 | Roadmap P1，可一次性做完 |
| API 标准化 | ❌ 部分 | 响应格式不统一，缺统一错误码 |
| SSH 连接池 | ❌ 未实现 | 每次操作新建连接，高频场景慢 |
| 执行时间预估 | ❌ 未实现 | 可基于历史数据做简单回归 |

### 1.2 具体优化建议（按优先级）

**高优先级**：
1. **调度器串联** — `dispatch_job_api` 应走调度器入队而非直接派发，实现真正的排队+并发控制
2. **远程资源监控** — 通过 SSH 定期拉取远程 `nvidia-smi` + `df`，展示服务器 GPU 状态
3. **SSH 连接复用** — 用 `ControlMaster` 或 `paramiko` 连接池减少握手开销
4. **统一响应格式** — 所有 API 包装为 `{ok, data, error, meta}` 结构

**中优先级**：
5. **requirements.txt 完善** — 声明 `weasyprint`、`psutil`、`numpy` 等可选依赖
6. **配置导入导出** — 单一 API 序列化/恢复 `advisor.json` + `samples_manifest.json` + SSH 配置
7. **历史统计** — 基于 `job.json` 的 `created_at` / `finished_at` 做时间序列聚合
8. **Align3R / CUT3R runner** — 环境已就绪，写 runner 是当前模型集成的最直接产出

---

## 二、AI 接入的改善与优化方向

当前 AI 层（`advisor.py`）是基础的 OpenAI-compatible 调用，只做了"结果评估 + 建议生成"。以下是可以发挥更大价值的方向：

### 2.1 AI 可以帮到的核心场景

| 场景 | 当前 | AI 增强后 |
|------|------|-----------|
| **结果评估** | 人工看图 + 手动评分 | AI 自动从深度图/点云/轨迹抽特征，给出结构化评分和诊断 |
| **实验规划** | 手动选模型/参数 | AI 根据输入特征（场景类型/运动幅度/纹理复杂度）推荐最优模型和参数组合 |
| **失败诊断** | 看日志猜原因 | AI 解析 runner.log，定位 OOM/nan/CUDA 错误，给出修复建议 |
| **报告撰写** | 手动写总结 | AI 基于多模型对比数据自动生成对比分析报告 |
| **环境排障** | 逐个检查依赖 | AI 解析 conda list/pip list/错误日志，输出修复命令序列 |
| **文献关联** | 手动查论文 | AI 将实验结果与模型论文声称的指标对比，指出差距和可能原因 |

### 2.2 具体实现路径

```
Phase 1: 智能诊断（当前可做）
├── AI 日志分析器 — 解析 runner.log 的错误模式
├── AI 参数推荐 — 根据输入图片/视频特征推荐预设
└── AI 对比报告 — 多模型结果的自动对比分析文本

Phase 2: 实验编排（中期）
├── AI 实验计划生成 — "给我跑这组图的全模型对比"
├── AI 超参搜索建议 — 基于已有结果推荐下一组参数
└── AI 质量门控 — 结果不达标自动建议重跑或换参数

Phase 3: 知识积累（长期）
├── 实验知识库 — 所有跑过的实验结果结构化存储
├── 经验推理 — "这个场景上次用 MonST3R 512 效果最好"
└── 论文对标 — 自动与公开 benchmark 数据对比
```

### 2.3 advisor.py 的具体改进

```python
# 新增能力
class AdvisorCapabilities:
    analyze_log: bool      # 日志智能分析
    recommend_params: bool # 参数推荐
    compare_report: bool   # 对比报告生成
    diagnose_env: bool     # 环境诊断
    plan_experiment: bool  # 实验计划生成
```

---

## 三、一键搭建 Agent 路线

### 3.1 核心理念

> 将"SSH 登录 → 建目录 → clone 仓库 → 创建 conda env → 装依赖 → 下载权重 → 编译扩展 → 跑 smoke test" 这个完整流程封装为 **一键 Agent**。

### 3.2 Agent 架构设计

```
┌─────────────────────────────────────────────┐
│              Experiment Agent                 │
├─────────────────────────────────────────────┤
│  Model Registry (模型元信息)                   │
│    ├── repo_url, branch, commit              │
│    ├── conda_env_spec (python, torch, cuda)  │
│    ├── pip_requirements                      │
│    ├── checkpoints [{name, url, size, hash}] │
│    ├── build_steps [{cmd, cwd, env}]         │
│    └── smoke_test {script, expected_outputs}  │
├─────────────────────────────────────────────┤
│  Environment Builder                         │
│    ├── SSH 连接管理                            │
│    ├── 依赖解析（conda/pip 冲突检测）            │
│    ├── 权重下载/上传调度                        │
│    ├── 扩展编译（curope 等）                    │
│    └── 健康检查（import/CUDA/kernel test）      │
├─────────────────────────────────────────────┤
│  Experiment Orchestrator                     │
│    ├── 样例集管理                              │
│    ├── 参数网格生成                             │
│    ├── 任务批量派发                             │
│    ├── 结果自动收集                             │
│    └── 指标计算 + 对比报告                      │
└─────────────────────────────────────────────┘
```

### 3.3 每个模型的声明式配置

```yaml
# model_specs/monst3r.yaml
name: MonST3R
repo:
  url: https://github.com/Junyi42/monst3r
  branch: main
  commit: 574cc77
  submodules: [croco, viser]

environment:
  conda_clone_from: dust3r  # 或独立 spec
  python: "3.11"
  torch: "2.5.1+cu121"
  extra_pip:
    - gradio==4.44.1
    - pyglet==2.0.20
  exclude_pip: [torch, torchvision]

checkpoints:
  - name: MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth
    path: checkpoints/
    source: huggingface  # or url
    size_gb: 2.1
  - name: Tartan-C-T-TSKH-spring540x960-M.pth
    path: third_party/RAFT/models/
  - name: sam2.1_hiera_large.pt
    path: third_party/sam2/checkpoints/

build_steps:
  - cmd: "cd croco/models/curope && python setup.py build_ext --inplace"
    env: {TORCH_CUDA_ARCH_LIST: "7.5", CUDA_HOME: "/usr/local/cuda-12.6"}

smoke_test:
  script: "python -c 'from dust3r.model import AsymmetricCroCo3DStereo; print(\"OK\")'"
  expected: "OK"

runner:
  script: runners/monst3r_runner.py
  default_params:
    image_size: 512
    num_frames: 48
```

### 3.4 实现阶段

| 阶段 | 内容 | 产出 |
|------|------|------|
| Phase 0 | 提炼现有 6 模型的 setup 经验为 YAML spec | 6 个 model_specs/*.yaml |
| Phase 1 | SSH-based Environment Builder — 读 spec → 自动执行 | `env_builder.py` |
| Phase 2 | Smoke Test Runner — 自动验证环境 | `smoke_runner.py` |
| Phase 3 | Experiment Orchestrator — 批量跑实验 | `experiment_agent.py` |
| Phase 4 | AI-assisted 故障修复 — 编译/依赖失败时 AI 诊断 | `ai_env_doctor.py` |

---

## 四、项目拆分方案

### 4.1 为什么拆分

| 维度 | kykt (研究主仓) | vision_platform (独立项目) |
|------|-----------------|---------------------------|
| **定位** | 论文/研究/Dream 架构 | 3R 模型管理平台工具 |
| **受众** | 个人研究 | 可复用的通用工具 |
| **迭代节奏** | 随研究进展 | 功能驱动，定期发版 |
| **体积** | 包含大量数据/权重引用 | 纯代码，轻量 |

### 4.2 拆分清单

```
新项目: kykt-vision-platform/  (或 vision-workbench/)
├── README.md                    # 新项目说明
├── CHANGELOG.md                 # 版本变更记录
├── LICENSE
│
├── backend/                     # 从 vision_ui/ 迁移
│   ├── app.py
│   ├── job_store.py
│   ├── job_scheduler.py
│   ├── resource_monitor.py
│   ├── metrics_calculator.py
│   ├── report_exporter.py
│   ├── visual_artifacts.py
│   ├── advisor.py
│   ├── ssh_runner.py
│   ├── model_registry.py
│   ├── model_contracts.py
│   ├── development_store.py
│   ├── requirements.txt
│   └── settings/
│
├── client/                      # 从 vision_ui/client/ 迁移
│   ├── src/
│   ├── src-tauri/
│   ├── package.json
│   └── ...
│
├── runners/                     # 远程执行脚本
│   ├── dust3r_runner.py
│   ├── mast3r_runner.py
│   ├── monst3r_runner.py
│   ├── spann3r_runner.py
│   ├── fast3r_runner.py
│   ├── align3r_runner.py        # 待写
│   └── cut3r_runner.py          # 待写
│
├── agent/                       # 一键搭建 Agent（新模块）
│   ├── model_specs/             # 声明式模型配置
│   ├── env_builder.py           # 环境自动搭建
│   ├── smoke_runner.py          # Smoke test 自动化
│   ├── experiment_agent.py      # 实验编排
│   └── ai_env_doctor.py         # AI 辅助环境诊断
│
├── docs/                        # 文档整合
│   ├── architecture.md
│   ├── deployment.md
│   ├── model-integration.md
│   ├── api-reference.md
│   └── portable-bundle.md
│
├── tools/                       # 辅助脚本
│   ├── check_3r_remote.ps1
│   └── ...
│
└── releases/                    # 版本归档
    ├── v0.1.0/                  # 最初 DUSt3R-only 版本
    ├── v0.2.0/                  # MonST3R + MASt3R
    ├── v0.3.0/                  # 6 模型 + 样例矩阵
    └── v0.4.0/                  # 调度器 + 指标 + 报告 + Agent
```

### 4.3 版本归档策略

**保留的重要版本节点**：

| 版本 | 时间点 | 里程碑 |
|------|--------|--------|
| v0.1.0 | 2026-04-06 | DUSt3R 基础版 — 首次 SSH 远程派发 + 本地缓存 + 中文 UI |
| v0.2.0 | 2026-04-13 | MonST3R + MASt3R + AI 助手 + 工作区布局 |
| v0.3.0 | 2026-04-21 | 6 模型集成 + 样例矩阵 + 部署状态 + Tauri 桌面端 |
| v0.3.1 | 2026-05-03 | React 默认前端 + curope 解锁 + 取消/恢复加固 |
| v0.4.0 | 2026-05-25 | 调度器 + 指标 + 报告 + 资源监控 + 可视化 |
| v0.5.0 | 计划 | Agent 模块 + AI 增强 + 独立项目化 |

**操作步骤**：
1. 在 `E:\kykt\Coding\4.06\vision_ui` 打 git tag `v0.4.0`
2. 创建独立仓库 `kykt-vision-platform`
3. 迁移代码（保留 git history 用 `git filter-branch` 或 `git subtree split`）
4. 原 `kykt` 仓库中保留轻量引用（子模块或文档链接）
5. 旧 `.md` 文档整合进 `docs/` 目录

### 4.4 kykt 主仓保留什么

```
E:\kykt/
├── Dream/                  # 研究主线（不动）
├── Coding/
│   ├── 3.16/              # MVSNet 历史（保留）
│   ├── 3.23/              # SfMLearner 历史（保留）
│   ├── 3.30/              # DUSt3R 历史（保留）
│   └── 4.06/vision_ui/    # → 改为 git submodule 指向新仓库
├── KYKT.md                 # 更新引用
├── PROJECT_PROGRESS_*.md   # 保留
└── ...
```

---

## 五、文档整理计划

### 5.1 当前文档问题

vision_ui 根目录有 **16 个 .md 文件**，存在：
- 时间线交叉（多个 PLAN/ROADMAP/HANDOFF 共存）
- 部分内容已过时（如 `SERVER_PREPARATION.md` 中 MonST3R 环境已就绪）
- 缺乏统一入口

### 5.2 整合方案

| 当前文件 | 处理 |
|----------|------|
| `README.md` | → `docs/` 重写为新项目 README |
| `DESIGN.md` | → `docs/architecture.md` |
| `BACKEND_ROADMAP_v0.4.md` | → 合并入新 CHANGELOG + 更新为 v0.5 |
| `AGENT_HANDOFF_v0.4.md` | → `docs/dev-guide.md` |
| `THREER_MODEL_ROADMAP.md` | → `docs/model-integration.md` |
| `ACTIVE_MODEL_INTEGRATION_PLAN.md` | → 合并入上文 |
| `MODEL_DEPLOYMENT_STATUS.md` | → `docs/deployment.md` |
| `MONST3R_MAINLINE_PLAN.md` | → `docs/models/monst3r.md` |
| `PORTABLE_BUNDLE.md` | → `docs/portable-bundle.md` |
| `APP_ARCHITECTURE_OPTIMIZATION.md` | → `docs/architecture.md` 附录 |
| `BATCH_MODEL_COMPARE_ROADMAP.md` | → 合并入 CHANGELOG |
| `UI_DESIGN_UPGRADE_PLAN.md` | → 归档 |
| `*_AGENT_PROMPT.md` | → 归档或删除（一次性交接文档） |
| `CLIENT_REBUILD.md` | → 归档（已完成） |
| `SERVER_PREPARATION.md` | → 合并入 deployment.md |

---

## 六、任务排期建议

### 第一阶段：整理与拆分（1-2 天）

- [ ] 打 `v0.4.0` tag
- [ ] 创建独立仓库结构
- [ ] 迁移代码，整合文档
- [ ] 更新 `kykt` 主仓引用

### 第二阶段：后端补齐（2-3 天）

- [ ] 调度器与 dispatch 串联
- [ ] Align3R / CUT3R runner
- [ ] SSH 连接复用
- [ ] 远程 GPU 监控
- [ ] requirements.txt 完善
- [ ] 统一 API 响应格式

### 第三阶段：AI 增强（2-3 天）

- [ ] advisor 日志分析能力
- [ ] advisor 参数推荐
- [ ] advisor 对比报告生成
- [ ] advisor 环境诊断

### 第四阶段：Agent 模块（3-5 天）

- [ ] 6 模型 YAML spec 编写
- [ ] env_builder.py — SSH 自动搭建
- [ ] smoke_runner.py — 自动验证
- [ ] experiment_agent.py — 批量实验
- [ ] 前端 Agent 面板

### 第五阶段：打磨发版（1-2 天）

- [ ] 全量 Smoke 验证
- [ ] 新 EXE 构建
- [ ] 文档最终审校
- [ ] v0.5.0 发版

---

*总预估: 9-15 天（可并行）*
