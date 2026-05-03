# KYKT 项目根目录

Last updated: 2026-05-04

## 项目当前形态

KYKT 已经从早期的"单论文复现"演进为研究 + 工程 + 教学汇报三层并行的项目结构：

- **研究主线（Phase 1.5: Research Workflow Deployment）**：在 `Dream/` 目录下，架构优先的 3R / 空间智能研究引擎。当前等待用户对 cycle 009 启动的授权。
- **工程主线（3R 模型工作台）**：在 `Coding/4.06/vision_ui/`，本地 React + FastAPI + Tauri 应用，已接入 6 个 3R 模型；MASt3R / MonST3R / Spann3R / Fast3R 平台 smoke 通过，Align3R / CUT3R 的 curope 已修，runner 待写。
- **历史复现线（已交付 / 暂停）**：`Coding/3.2 ~ 3.30` 下的 MVSNet / SfMLearner / DUSt3R 实验记录，作为基线和阶段汇报材料。

下面"研究主线"、"工程主线"、"历史复现线"三节给出每条线的当前进度与遗留 gap。

## 必读入口（按用途分流）

不同任务请按下表分流，避免一次把所有文档拉进上下文，否则容易触发 32 MB 请求上限（详情见 `Dream/TASK_SNAPSHOT.md` 的 F-001 与 "Working rules to avoid F-001"）：

| 任务类型 | 第一站 | 第二站 |
|---|---|---|
| 接手 / 恢复研究主线（Dream） | `Dream\TASK_SNAPSHOT.md` | `Dream\AGENT_MASTER_PROMPT.md`（其 mandatory load protocol 把 `TASK_SNAPSHOT.md` 列为 #1） |
| 接手 / 推进工程主线（vision_ui） | `KYKT.md` | `PROJECT_PROGRESS_2026-05-03.md` |
| 模型接入 / 远端部署 | `Coding\4.06\vision_ui\ACTIVE_MODEL_INTEGRATION_PLAN.md` | `Coding\4.06\vision_ui\MODEL_DEPLOYMENT_STATUS.md` |
| 桌面打包 | `Coding\4.06\vision_ui\PORTABLE_BUNDLE.md` |  |
| 本地架构与最近修复 | `Coding\4.06\vision_ui\APP_ARCHITECTURE_OPTIMIZATION.md` | `WORK.md` |
| 阅读历史复现成果（MVSNet 等） | 本文件"历史复现线状态"节 | 各子目录下 README |

如果不确定任务属于哪条线，先读 `Dream\TASK_SNAPSHOT.md` 与 `KYKT.md` 的 "Project Lines"，再决定。

## 仓库总体布局

```text
E:\kykt
├─ .claude\                           Claude Code 配置
├─ .git\                              Git 仓库元数据
├─ .gitignore
├─ .idea\                             JetBrains 配置
├─ .omx\
├─ Coding\                            代码与实验
│  ├─ 3.2\, 3.9\, 3.16\, 3.23\        早期复现：MVSNet / SfMLearner / DTU 等
│  ├─ 3.30\dust3r-main\               DUSt3R 静态重建实验
│  ├─ 4.06\vision_ui\                 当前主工程：本地 3R 工作台（Tauri + FastAPI + React）
│  └─ external_sources\               官方源码副本（monst3r、mast3r 等）
├─ Dream\                             研究引擎工作区（Phase 1.5）
├─ Files\                             过程文档、原始要求等
├─ KYKT.md                            工程主线汇总（路线、模型、平台、gap、agent 指引）
├─ LICENSE
├─ PROJECT_PROGRESS_2026-04-23.md     历史进度快照
├─ PROJECT_PROGRESS_2026-05-03.md     最新进度快照（六模型当前状态、平台修复清单）
├─ README.md                          本文件
├─ WORK.md                            具体工作日志（最新一轮平台改动）
├─ model_uploads\                     上传到服务器的权重 staging（raft、monst3r 等）
├─ ppt\                               汇报 / 演示材料
├─ release\
├─ tmp\                               临时文件（如 curope 探针）
└─ tools\                             辅助脚本（如 build_curope.sh）
```

## 研究主线状态（Dream）

阶段：`Phase 1.5: Research Workflow Deployment`

到 2026-05-04 为止已完成：

- 4 份 finalist L1 spec 起草（详见 `Dream\specs\`）
- 跨 spec 信号契约 v1（`Dream\paradigm\CROSS_SPEC_SIGNAL_CONTRACT.md`）
- 文献板 v1（`Dream\literature\INDEX.md` + 子文件）
- planning 层（`Dream\planning\`）已对齐到四 finalist 姿态：BRANCH_COMPARISON_MATRIX / MULTI_TRACK_RESEARCH_CANVAS / ACTION_TAXONOMY_AND_PROXY_METRICS / RESEARCH_GRAPH_AND_PAPER_START
- 注册表与库存（`Dream\registry\` / `Dream\sources\` / `Dream\units\REPRODUCTION_READINESS_MATRIX.md` / `Dream\logs\QUESTION_LOG.md`）已与 SPINE Anchor Map + cycle 008.5 dormancy + Round 10 同步
- 引入 `Dream\TASK_SNAPSHOT.md` 作为最高优先级的恢复入口；`AGENT_MASTER_PROMPT.md` 的 mandatory load protocol、`README.md` / `INDEX.md` / `WORKFLOW_STATUS.md` 的入口指针 + Sync Rule 链都已接进去

阻塞推进的用户决策（4 项；canonical 文本在 `Dream\WORKFLOW_STATUS.md` "Recommended Next User Decision"）：

1. **cycle 009 顺序**：Composer case card 与 Critic 并行（默认；走 cross-spec contract 测试路径）/ 等 Critic 第一张 case card 落地后串行
2. **Composer 能力卡来源**：仅 paper-derived（默认，速度快）/ paper + KYKT job-derived（更慢；默认下推迟到 cycle 010）
3. **`Dream\paradigm\TEACHER_AUDIENCE_PROFILE.md`** 的内容（用户输入；agent 不会自造字段，这一项不解决会卡 D3 教学 demo 目标）
4. **cycle 009 启动授权**：在用户 "go" 前不进入 case-card 填写

未授权动作（agent 不得擅自做）：reproduction、checkpoint 下载、训练、KYKT 导航变更、前端实现、最终 thesis 选定、退出任何非 finalist 路线、宣告 teacher demo 就绪。

## 工程主线状态（3R 模型工作台）

应用：`Coding\4.06\vision_ui`，本地 FastAPI 后端 + React 前端，桌面端 Tauri 包装；服务器侧通过 SSH 调度训练 / 推理。

当前模型状态（节选自 `PROJECT_PROGRESS_2026-05-03.md`）：

| 模型 | env / 路径 | 平台 smoke job | 状态 |
|---|---|---|---|
| MASt3R | 服务器 mast3r env | `20260420-222729` | 通过；`matches.png` + `pointcloud.ply` + `scene_meta.json` 等输出齐 |
| MonST3R | 服务器 monst3r env | `20260420-222928` | 通过；`scene.glb` + 48 帧预览 + 96 张动态 mask + 96 个 confidence array |
| Spann3R | `/hdd3/kykt26/code/spann3r` | `20260425-113227` | 通过；curope 已编译；E2E 输入用 6 张 MonST3R 帧预览 |
| Fast3R | `/hdd3/kykt26/code/fast3r` | `20260425-113002` | 通过；TITAN RTX 上必须 `attention_backend=pytorch_naive`，已写进 runner 默认 |
| Align3R | `/hdd3/kykt26/code/align3r` | 待 smoke | 2026-05-03 解锁；curope 已重编（**真因是 GLIBC，不是旧文档说的 CUDA mismatch**）；缺 `align3r_runner.py` |
| CUT3R | `/hdd3/kykt26/code/cut3r` | 待 smoke | 2026-05-03 解锁；curope 已编译；缺 `cut3r_runner.py` |

暂缓模型：DUSt3R multi-image / Pi3X / ZipMap / LingBot-Map（暂不进入工作台）。

平台层 2026-05-03 主要修复：

- 默认前端切到 React（`client/dist/index.html` 存在时直接 serve），Jinja 模板降级保留
- Tauri 端 `is_backend_root` 不再要求 `.venv`，Python 解释器解析顺序：`KYKT_BACKEND_PYTHON` → `<root>/.venv/Scripts/python.exe` → `<root>/python/python.exe` → 相邻 `python/` → 系统 PATH
- 远端取消硬化：`_kill_remote_job_processes` 增加 align3r / cut3r runner，SIGTERM → 2s grace → SIGKILL → 1s verify，按 PID 报告残留
- 孤儿任务自愈：FastAPI startup hook 把后端重启后残留的 `status="running"` 标 `failed` 并附重试提示
- `tools/build_curope.sh` 与 `tools/probe_curope.py` / `tools/verify_curope2.py` 备好，后续别的 env 重编可复用

App 当前能力（节选）：模型路线面板 / 样例库 / 测评矩阵 / 人工评分 + AI Advisor / 任务结果详情分组 / ZIP bundle 导出 / Tauri 自动监管 FastAPI / 远端部署摘要 / React 默认 UI / PID 级取消报告 / 后端重启后孤儿任务自愈。

主要 gap：

1. Align3R / CUT3R runner + 第一个平台 smoke
2. MASt3R / MonST3R / Spann3R / Fast3R 的更高质量样例（用于横向对比矩阵）
3. 按 `sample_id` 聚合的模型对比视图
4. portable bundle 一键发布脚本（嵌入式 Python + React dist + exe + backend）的打包测试
5. Windows 侧旧 uvicorn / ssh 进程的恢复仍偏手动
6. 服务器端写入的精确进度替换当前的本地 + 远端混合估算

## 历史复现线状态

均为已交付 / 暂停状态，作为基线参考。新工作不应改写这些子目录，除非用户明确要求：

- `Coding\3.16\MVSNet`：跑通 DTU 训练 → 推理 → 融合，得到一版可展示的 `.ply` 点云。剩余：DTU 官方定量评测、更长轮次 / 更大规模训练、融合模块系统优化、无监督 / 弱监督方案探索。
- `Coding\3.23\SfmLearner-Pytorch-master`：SfMLearner 对当前 PyTorch 环境的兼容性修复版本（投影、重采样、loss 接口、单卡训练流程）。
- `Coding\3.30\dust3r-main`：DUSt3R 静态重建实验。

历史进度文档（按时间倒序，详细程度递减）：

- `PROJECT_PROGRESS_2026-05-03.md`（最新；六模型当前状态 + 平台修复清单 + agent 指引）
- `PROJECT_PROGRESS_2026-04-23.md`
- `Coding\3.16\近期工作历程.md`（3.2 - 3.23 实验记录）

## 工作流约定（agent + 人都遵守）

1. **触碰研究主线之前先读 `Dream\TASK_SNAPSHOT.md`**。如果 status 是 `in_progress` 或 `blocked`，按其 "If interrupted, resume from" 块续上去，不开新工作。`AGENT_MASTER_PROMPT.md` 的 mandatory load protocol 也已经把它列为 #1。
2. **触碰工程主线之前先读 `KYKT.md` + `PROJECT_PROGRESS_2026-05-03.md`**。不要相信旧文档里"Align3R / CUT3R 是 CUDA mismatch"这一类已经被推翻的诊断。
3. **大文件读法**：优先 `Grep -n` 加 `-C` / `-A` / `-B` 做精准定位；用 `Read` 的 `offset` + `limit` 切片；不重复 Read 已经读过的文件；优先 `Edit` 而不是 `Write`。完整规则在 `Dream\TASK_SNAPSHOT.md` "Working rules to avoid F-001" 章节，违反这条就会复现今天发生过两次的 32 MB 请求上限失败。
4. **同步顺序**：Guidance File Sync Rule 链中 `TASK_SNAPSHOT.md` 优先更新，避免中断后没有 resume 指针。
5. **未授权动作（同 Dream 硬规则）**：reproduction、checkpoint 下载、训练、KYKT 导航变更、前端实现、最终 thesis 选定、retire 任何非 finalist、宣告 teacher demo 就绪 —— 都需要用户在当前对话里明示同意。
6. **Commit message** 短即可；当前历史提交多为 "1"，新提交可以更具描述性，但不必过长。
