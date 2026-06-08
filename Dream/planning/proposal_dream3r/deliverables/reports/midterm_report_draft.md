# 中期报告草稿

## 题目

面向前馈式三维重建模型的聚合管理平台设计与实现

## 写作口径

本报告只记录中期阶段已经完成的工程任务、代码改动和验证结果，不展开研究背景。开题阶段的内容只作为目标来源，不重复写“为什么要做三维重建平台”。

## 一、阶段目标

中期阶段的目标是把开题方案中的平台原型落到可运行状态。具体目标包括：

1. 将 Agent 模型蓝图和环境构建能力接入后端 API；
2. 在前端增加 Agent 工作台，能查看模型状态、校验结果和构建任务；
3. 完成发布检查脚本，统一验证版本、测试、前端构建、Docker 静态配置和正式产物；
4. 跑通至少一个代表模型的远端健康检查、烟雾测试和环境构建流程；
5. 完成 Windows 桌面正式打包所需的后端侧车和 Tauri 安装包；
6. 补充文档和测试，减少后续接手成本。

## 二、本阶段完成的主要任务

### 1. Agent 后端接口接入

已将 Agent 模块从命令行能力扩展为后端 API。后端新增接口包括：

| 接口 | 作用 |
| --- | --- |
| `GET /api/agent/registry` | 返回全部模型蓝图摘要 |
| `GET /api/agent/registry/{model}` | 返回单个模型蓝图详情 |
| `GET /api/agent/validate` | 校验全部或指定模型蓝图 |
| `GET /api/agent/builds` | 查看环境构建任务列表 |
| `GET /api/agent/builds/{task_id}` | 查看单个构建任务状态 |
| `POST /api/agent/build/{model}` | 启动异步环境构建任务 |

这部分改动让平台可以在桌面端直接调用模型注册、蓝图校验和环境构建流程，不再只能手动运行 `python -m agent ...`。

### 2. 前端 Agent 工作台

前端新增 `AgentWorkbench` 工作台，并接入侧边栏和页面状态。工作台覆盖以下内容：

- 查看 7 个模型的蓝图状态；
- 查看模型类型、输入输出、环境和接入状态；
- 运行蓝图校验并展示摘要；
- 发起后台环境构建任务；
- 轮询构建任务状态，展示任务结果。

该功能对应开题报告中的“模型注册”和“任务工作台”，目前已经从规划变成可操作界面。

### 3. 发布检查脚本

新增 `tools/release_check.py`，用于发布前集中检查。检查内容包括：

- 项目版本号是否一致；
- Agent 蓝图是否全部通过校验；
- Python 测试是否通过；
- 前端是否可以完成正式构建；
- Dockerfile 和 Docker Compose 静态配置是否有效；
- 是否存在正式发布产物；
- Docker CLI 存在时再检查 Compose 配置。

这个脚本的作用是把发布前的零散检查收敛成固定流程，避免只靠人工记忆。

### 4. Docker 与部署配置修正

Docker 相关配置已按生产运行方式整理：

- Docker 镜像改为 Python 3.11；
- 运行时保留 `backend/`、`agent/`、`runners/`、`samples/`、`tools/`；
- 健康检查统一到 `/api/health`；
- 容器数据根设置为 `KYKT_DATA_ROOT=/app/data`，数据写入挂载目录；
- Docker Runner 和 Online API Runner 只作为可选后备，不作为默认 UI 能力展示。

当前机器没有 Docker CLI，因此没有做容器真实启动测试；静态配置检查已通过。

### 5. 后端生命周期与运行稳定性修正

后端 FastAPI 启动/关闭逻辑已迁移到 lifespan，去掉旧的 `@app.on_event` 用法。这样可以减少新版 FastAPI 下的弃用问题。

同时处理了可选 runner 的导入问题：在 Docker runner 不可用时，`/api/runners/availability` 返回 `docker: false`，不再因为缺少可选模块抛 500。

### 6. Agent CLI 与诊断修正

本阶段修正了几个实际会影响使用的问题：

- Agent CLI 状态输出改为 ASCII 标记，避免 Windows GBK 控制台编码失败；
- `agent smoke` 命令改为读取 `SmokeReport.ready` 和 `smoke_output`，修复远端 smoke 成功后仍因字段名错误崩溃的问题；
- `HealthDoctor` 可以从诊断文本中提取反引号命令作为 `fix_command`；
- `{env_name}` 占位符替换逻辑已补齐；
- 常见可修复错误可以被归类为 fixable。

这些改动不是展示功能，但会直接影响远端验证和接手使用。

### 7. Windows 桌面打包

已完成 Windows 正式版打包所需的两类产物：

- 后端 PyInstaller 侧车：`dist/3r-backend.exe`；
- Tauri NSIS 安装包：`client/src-tauri/target/release/bundle/nsis/3R All-in-One_0.5.0_x64-setup.exe`。

PyInstaller spec 已补齐 Agent 蓝图、根级 runner 脚本和 React 构建产物，避免正式包里缺运行资源。

### 8. 文档更新

已更新 README、部署文档、Docker 部署文档、API 参考、Agent 使用指南、CHANGELOG 和交接文档。文档重点不是宣传，而是让后续能知道：

- 项目当前版本；
- 后端和前端如何运行；
- Agent CLI 怎么用；
- API 端点有哪些；
- 发布前怎么检查；
- Windows 桌面版怎么构建；
- Docker 方式还缺哪一步验证。

## 三、当前平台状态

### 1. 模型池状态

当前模型蓝图共 7 个：

| 模型 | 状态 | 说明 |
| --- | --- | --- |
| DUSt3R | integrated | 已接入，远端 health/smoke/build 已验证 |
| MASt3R | integrated | 已接入 |
| MonST3R | integrated | 已接入 |
| Spann3R | integrated | 已接入 |
| Fast3R | integrated | 已接入 |
| Align3R | env_ready | 环境准备状态 |
| CUT3R | env_ready | 环境准备状态 |

### 2. 平台结构

当前平台由四部分组成：

- Tauri 2 + React 前端；
- FastAPI 后端；
- Agent 蓝图、环境构建、健康检查和烟雾测试模块；
- 远端 runner 脚本和 SSH/SCP 调度流程。

这和开题阶段的“桌面前端、本地后端、远端调度、模型执行器”分层一致。

## 四、验证结果

本阶段已完成以下验证：

| 验证项 | 结果 |
| --- | --- |
| Python 测试 | `pytest -q`：130 passed，1 skipped，1 warning |
| 发布检查 | `python tools/release_check.py --require-artifacts`：PASS |
| Agent 蓝图校验 | 7/7 valid |
| 前端正式构建 | PASS |
| Docker 静态配置 | PASS |
| Docker 真实启动 | 未运行，原因是当前机器未安装 Docker CLI |
| 远端 SSH 连通性 | `KYKT-UI` 可访问 |
| DUSt3R health | PASS |
| DUSt3R smoke | PASS |
| DUSt3R build | PASS，conda env、pip install、smoke test 均通过 |
| 后端侧车 smoke | `/api/health` 与 `/api/agent/registry` 可访问 |
| Tauri 安装包 | NSIS 安装包已生成 |

测试中保留 1 个 warning，来源是 FastAPI TestClient / Starlette / httpx2 兼容提示，不影响当前测试通过。

## 五、已经解决的问题

1. 后端 Agent 能力只能靠命令行调用的问题：已通过 API 接入解决。
2. 前端缺少模型蓝图查看和构建入口的问题：已新增 Agent 工作台。
3. 发布前检查分散的问题：已新增 release check 脚本。
4. Windows 控制台下 Agent 输出编码失败的问题：已改为 ASCII 状态输出。
5. smoke 成功但 CLI 因字段名错误崩溃的问题：已修复。
6. Docker 可选模块缺失导致接口报错的问题：已改为可用性返回。
7. 正式打包缺少 Agent 和 runner 运行资源的问题：已补齐 PyInstaller 打包资源。

## 六、当前产物

当前可作为中期材料的产物包括：

- 可运行的本地桌面平台；
- FastAPI 后端 Agent API；
- React 前端 Agent 工作台；
- 7 个模型蓝图；
- DUSt3R 远端 health、smoke、build 验证记录；
- 130 个通过的 Python 测试；
- 发布检查脚本；
- PyInstaller 后端侧车；
- Tauri NSIS 安装包；
- README、API、部署、Docker、Agent 指南和 CHANGELOG。

## 七、剩余问题与下一步

下一阶段不再写泛背景，主要补证据和补功能：

1. 补 Docker 真实启动验证。当前只完成静态配置检查，需在有 Docker CLI 的机器上跑一次容器启动和 `/api/health`。
2. 对 MASt3R、MonST3R、Spann3R、Fast3R 继续补 health/smoke/build 记录。现在完整跑通的是 DUSt3R。
3. 补中期 PPT 所需截图：Agent 工作台、任务队列、API 返回、发布检查结果、安装包路径、远端验证输出。
4. 完善结果对比视图，把同一输入下的多模型输出、日志和元数据放到一页里。
5. 将验证记录整理成固定表格，便于最终报告复用。
6. 清理当前文档中的版本和路径表述，保持与 v0.5.0 一致。

## 八、中期结论

中期阶段已经完成从“平台方案”到“可运行正式版雏形”的落地。当前重点成果不是算法指标，而是平台工程能力：模型蓝图可校验，Agent 能通过 API 和前端调用，远端 DUSt3R 流程已跑通，测试和发布检查已形成固定流程，Windows 安装包已生成。后续工作应继续补齐更多模型的真实运行记录、Docker 启动验证和中期 PPT 截图证据。
