# KITTI 服务器任务安排

## 当前 3.23 目录情况

当前 `3.23` 目录下的主要内容是：

- `SfmLearner-Pytorch-master/`

进一步检查后可以确认：

1. 这是一个较完整的 `SfMLearner-Pytorch` 仓库。
2. 仓库自带 KITTI 数据加载器、数据预处理脚本和深度评测脚本。
3. 训练不能直接使用原始 KITTI 数据，必须先用 `data/prepare_train_data.py` 转成它自己的顺序数据格式。
4. 当前这份代码是偏旧版本实现，README 中说明主要是在较老的 PyTorch 环境下开发和测试过，因此正式上服务器前要先做兼容性自检。

---

## 我们现在已经确认的关键点

### 1. 训练入口

核心训练脚本是：

- `train.py`

训练时需要传入的不是原始 KITTI 根目录，而是 **预处理后的格式化数据目录**。

### 2. 数据预处理入口

核心预处理脚本是：

- `data/prepare_train_data.py`

对 KITTI Raw 的标准用法是：

```bash
python data/prepare_train_data.py /path/to/raw/kitti \
  --dataset-format kitti_raw \
  --dump-root /path/to/formatted/kitti \
  --width 416 \
  --height 128 \
  --num-threads 4 \
  --with-depth \
  --with-pose
```

它会生成：

- 各个场景子目录
- 每个场景下的 `.jpg`
- `cam.txt`
- 可选的深度 `.npy`
- 可选的 `poses.txt`
- 根目录下的 `train.txt`
- 根目录下的 `val.txt`

### 3. KITTI 类型判断

当前仓库同时支持：

- `kitti_raw`
- `kitti_odometry`

其中：

- **训练深度模型** 时，优先使用 `KITTI Raw`
- **位姿评测** 时，通常额外需要 `KITTI Odometry`

如果你说服务器上已经下好了 100 多 G 的 KITTI，按体量判断，大概率是 **KITTI Raw**，这正适合拿来做 SfMLearner 的主训练集。

### 4. 评测入口

深度评测脚本：

- `test_disp.py`

位姿评测脚本：

- `test_pose.py`

也就是说，这个仓库本身已经把：

- 数据预处理
- 训练
- 深度评测
- 位姿评测

这四条链路都准备好了。

---

## 下一步任务安排

下面按“必须先做”和“可以后做”来排顺序。

### 第一阶段：服务器准备与摸底

这是最优先的一步，目标是确认环境和数据都能对上。

#### 任务 1：确认服务器上的 KITTI 真实路径和数据类型

需要确认：

- 原始 KITTI 数据具体放在哪个目录
- 是 `KITTI Raw` 还是 `KITTI Odometry`
- 是否包含彩色图像、标定文件、oxts、velodyne

建议输出：

- KITTI 根目录绝对路径
- 目录树前两层截图或文本

#### 任务 2：把 `SfMLearner-Pytorch` 上传或同步到服务器

建议放在类似下面的位置：

```text
/hdd3/kykt26/code/SfmLearner-Pytorch
```

当前本地目录有双层嵌套：

```text
SfmLearner-Pytorch-master/SfmLearner-Pytorch-master
```

上传到服务器前建议整理成单层目录，避免路径过深和命令易错。

#### 任务 3：在服务器上创建单独环境

建议不要直接复用 MVSNet 那套环境，而是单独建一个环境：

```bash
conda create -n sfm python=3.10 -y
conda activate sfm
```

然后安装依赖：

```bash
pip install -r requirements.txt
```

如果报缺包，再按需补装。

#### 任务 4：先做代码自检，不直接开训

先跑下面这些命令看是否能正常打印帮助：

```bash
python train.py -h
python data/prepare_train_data.py -h
python test_disp.py -h
```

这一步的目标不是训练，而是先确认：

- 代码能 import 成功
- 依赖基本齐全
- 当前 PyTorch 不会在启动阶段直接报兼容性错误

---

### 第二阶段：KITTI 数据预处理

这一阶段是正式训练前最关键的一步。

#### 任务 5：先做一个小规模预处理 smoke test

不要一上来处理完整 100 多 G 数据，先抽一个小目录做验证，确认流程能跑通。

目标是验证：

- `prepare_train_data.py` 能正常运行
- 生成的格式符合训练集要求
- 会正确产生 `train.txt`、`val.txt` 和 `cam.txt`

#### 任务 6：再做全量预处理

如果 smoke test 没问题，再跑全量 KITTI Raw 预处理。

建议输出目录单独放：

```text
/hdd3/kykt26/data/kitti_formatted_sfm
```

推荐命令模板：

```bash
python data/prepare_train_data.py /path/to/kitti_raw \
  --dataset-format kitti_raw \
  --dump-root /hdd3/kykt26/data/kitti_formatted_sfm \
  --width 416 \
  --height 128 \
  --num-threads 8 \
  --with-depth \
  --with-pose
```

做完之后要检查：

- 是否生成 `train.txt`
- 是否生成 `val.txt`
- 子目录中是否有 `.jpg`
- 子目录中是否有 `cam.txt`

---

### 第三阶段：训练 smoke test

在全量训练前，先做一轮非常短的 smoke test。

#### 任务 7：短训练验证

目标：

- 确认 dataloader 能正常读格式化后的 KITTI
- 确认模型前向和 loss 正常
- 确认日志和 checkpoint 能输出

建议只跑 1 个 epoch，且把 `epoch-size` 压小：

```bash
python train.py /hdd3/kykt26/data/kitti_formatted_sfm \
  --dataset-format sequential \
  -b 4 \
  -m 0.2 \
  -s 0.1 \
  --epoch-size 200 \
  --sequence-length 3 \
  --epochs 1 \
  --with-gt \
  --log-output
```

这一步主要看：

- 是否能开始迭代
- 是否会出现 PyTorch 兼容性报错
- 是否会生成 `checkpoints/`
- TensorBoard 是否有事件文件

---

### 第四阶段：正式训练与评测

在 smoke test 稳定后，再进入正式阶段。

#### 任务 8：正式训练

可先沿 README 推荐参数跑一版：

```bash
python train.py /hdd3/kykt26/data/kitti_formatted_sfm \
  --dataset-format sequential \
  -b 4 \
  -m 0.2 \
  -s 0.1 \
  --epoch-size 3000 \
  --sequence-length 3 \
  --log-output \
  --with-gt
```

#### 任务 9：深度评测

训练完后，用：

- `test_disp.py`

对 KITTI depth 做评测。

#### 任务 10：如条件允许，再做 pose 评测

如果服务器上还有 KITTI odometry 数据，再用：

- `test_pose.py`

做位姿误差评测。

---

## 建议你马上做的 3 件事

如果只看“下一步最应该做什么”，我建议按这个顺序来：

1. 确认服务器上 KITTI 的绝对路径和目录结构。
2. 在服务器上建 `sfm` 环境，并跑 `train.py -h` / `prepare_train_data.py -h` 做兼容性检查。
3. 先做一个小规模 KITTI 预处理 smoke test，再决定是否全量预处理。

---

## 当前判断

基于 `3.23` 文件夹当前内容，我的判断是：

- 现在还不应该直接开正式训练。
- 当前最合理的推进方式是：
  - 先做环境和数据格式验证
  - 再做小规模预处理验证
  - 然后做短训练 smoke test
  - 最后再上完整 KITTI 训练

这样能最大程度避免像 MVSNet 一开始那样，花很长时间后才发现环境或数据格式问题。
