# SUGAR CarryBox 复现记录

本文只记录当前仍可执行的官方 SUGAR CarryBox 基线、方法结构、输入输出、运行环境和
最短复现路径。日期化故障日志、model-5000 中间阶段、旧 RGB/Plan-10/Plan-13 研究和
重复渲染说明已迁入 ignored `legacy/`。

## 1. 结论边界

官方 SUGAR 已在本工作区完成 CarryBox 推理、Refiner 训练/rollout 和下游数据链路
验证。当前 Plan-15 使用本地保留的官方 Refiner `model-10000` 作为冻结 teacher，并
使用官方 Tracker checkpoint 初始化 student。

必须明确：官方发布的机器人 policy 是 state-based，不在每个 control step 直接读取
原始 RGB。项目名称中的 human-video-driven 指训练目标来自人类物体交互视频处理后的
3D 人体/物体轨迹；官方仓库当前没有发布从原始 RGB-D 视频到该训练数据的处理流水线。
因此不能把“视频提供任务数据”写成“policy 的输入是 RGB”。

官方 ContactSensor 的 `hands_contact_label` 只是阈值化接触 proxy，也不是空间触觉。
Plan-15 新增的 TacSL/R15 整手触觉属于本仓库扩展，不应倒写为官方 SUGAR 原有能力。

## 2. 官方方法结构

官方训练顺序固定为：

```text
processed human/object motion
        │
        ▼
Refiner policy ── rollout ──► processed RL dataset
        │                           │
        │ frozen teacher            ▼
        └────────────────────── Tracker policy
                                    │
                                    ├─ rollout ──► processed IL dataset
                                    │
                                    ▼
                       diffusion Transformer Generator
                                    │
                                    ▼
                       generated motion + Tracker control
```

### 2.1 Refiner

Refiner 在 IsaacLab/PhysX 中跟踪一条给定的人体到 G1 的机器人/物体参考轨迹。

- 训练数据：`SUGAR/data/CarryBox` 中处理后的参考 motion；不是原始视频帧。
- actor/critic：RSL-RL PPO，ELU MLP，hidden dimensions `512/256/128`。
- action：完整 G1 的 29-D joint-position target。
- policy observation：官方配置把 policy 和 critic 都设为 privileged state group；
  CarryBox 当前展开为约 `890-D`。
- privileged fields：未来 joint/object reference，机器人 body pose/orientation，base
  velocity，joint position/velocity，上一动作，以及 measured object pose、orientation、
  linear velocity 和 angular velocity。
- 输出：`model_<iteration>.pt`；rollout 后保存完整成功 trajectory。

这意味着官方 Refiner 是仿真特权上界，不能直接描述为仅依靠真实机器人可获得输入的
部署 policy。

### 2.2 Tracker

Tracker 从 Refiner 成功 rollout 学习跟踪由 motion command/Generator 提供的计划。

- 训练数据：处理后的 Refiner rollout；原始人类 motion 仍提供 teacher reference。
- 优化：仓库 BCPPO；冻结 Refiner 提供 teacher action，task PPO 负责物理闭环行为。
- actor/critic：同样为 `512/256/128` ELU MLP，29-D joint-position action。
- 官方 Tracker actor observation：reference joint position、root velocity、contact
  label、五帧 proprio/action/gravity history，以及 measured object pose/orientation；
  当前 CarryBox 路径约 `510-D`。
- critic/teacher：仍使用 privileged object/robot state和未来 reference。
- 输出：Tracker checkpoint；rollout 处理后生成 Generator 的 IL dataset。

Plan-15 没有照搬官方 Tracker actor 中的 measured object pose。它固定使用重新整理的
`504-D` Tracker-command/proprioception contract，明确排除 measured object state；
这是当前触觉实验的公平性修改，不是官方 SUGAR 原始 observation。

### 2.3 Generator

Generator 是 state-conditioned diffusion Transformer，不是 Refiner/Tracker 使用的
MLP，也不是 RGB encoder。

- 数据：处理后的 Tracker rollout。
- observation encoder：object position + 6-D orientation、上一 36-D reference action，
  CarryBox 还可加入 target object position + 6-D orientation；每部分映射到 256-D token。
- predicted sequence：8-step horizon 的 36-D reference vector，由 Generator dataset
  定义；它是给 motion command/Tracker 使用的计划表示，不是 29-D motor torque。
- backbone：12-layer、8-head action-diffusion Transformer。
- diffusion：DDIM，50 个训练 noise timesteps，默认 16 个 inference steps。
- 输出：`generator.ckpt`，推理时与 Tracker checkpoint 一起加载。

### 2.4 训练与测试的区别

训练时，Refiner 看到给定 motion 的完整特权状态；Tracker 使用 Refiner teacher 和
仿真任务 reward；Generator 对成功 Tracker rollout 做 diffusion imitation。测试/推理
时不再提供逐帧 ground-truth 人体 trajectory loss，而是 Generator 产生 reference，
Tracker 根据当前 state 和计划闭环输出 29-D action。

官方 simulator 会随机化 robot/object friction、object mass 和初始状态，但具体任务
是否对材料、质量或新物体真正泛化，必须由冻结 checkpoint 的独立物理评估回答；不能
仅因训练配置存在 randomization 就声称泛化或 sim-to-real。

## 3. 工作区软件与资产

当前官方基线使用：

- Isaac Sim 5.1；
- official SUGAR 所要求的 IsaacLab v2.3.0 基线代码；
- Python 3.11 环境
  `/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python`；
- `SUGAR/data/CarryBox` 的处理 motion；
- `SUGAR/demo_ckpts/CarryBox/` 的官方 Tracker/Generator released checkpoints；
- `SUGAR/descriptions/robots/g1/` 与 `SUGAR/descriptions/objects/small_box/`。

本地保留的 Refiner teacher：

```text
experiments/sugar_reproduction/outputs/final/official_sugar/baseline/
└── ckpts/refiner_model10000.pt
```

Plan-15 的官方 TacSL/R15 calibration 位于：

```text
experiments/sugar_reproduction/assets/official_tacsl/
```

`experiments/` 整体被 Git 忽略。checkpoint、dataset、trace、video 和 runtime log 不
进入代码提交。

## 4. 当前保留结果

官方基线目录保留以下小型证据：

```text
experiments/sugar_reproduction/outputs/final/official_sugar/baseline/
├── ckpts/refiner_model10000.pt
└── visualizations/
    ├── refiner_model10000_full_rollout_summary.json
    ├── refiner_model10000_full_rollout_summary.png
    ├── refiner_model10000_rollout_summary.json
    ├── refiner_model10000_rollout_summary.png
    ├── refiner_model10000_rollout_video.mp4
    ├── refiner_training_curves.png
    └── tracker_training_curves.png
```

这些文件证明 checkpoint 能够进入官方 rollout、产生完整 CarryBox 行为并支持当前
teacher handoff。它们不是 Plan-15 Z/P/PS 的触觉收益结果。

## 5. 环境验证与 released inference

命令必须在 GPU compute node 中运行：

```bash
cd /public/home/yanhongru/Curiosity
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
export DISPLAY=
export OMNI_KIT_ACCEPT_EULA=Y

PYTHON_BIN="$PYTHON_BIN" bash scripts/sugar/preflight_official_sugar_env.sh
```

运行官方 released CarryBox inference：

```bash
PYTHON_BIN="$PYTHON_BIN" NUM_ENVS=16 VIDEO_LENGTH=200 \
  bash scripts/sugar/run_official_sugar_carrybox_inference.sh
```

该 wrapper 检查 `data/CarryBox`、Tracker、Generator、G1 和 small-box asset，然后运行：

```text
Sugar-G129dof-CarryBox-Inference
```

输出写入 ignored `experiments/sugar_reproduction/`。

## 6. 从头执行官方训练流水线

最短入口：

```bash
PYTHON_BIN="$PYTHON_BIN" \
TASK_NAME=CarryBox \
OUTPUT_DIR="$PWD/experiments/sugar_reproduction/outputs/CarryBox_reproduction" \
  bash scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh
```

wrapper 严格执行七个阶段：

1. `refiner_train`：`Sugar-G129dof-CarryBox-Refiner`；
2. `refiner_rollout`：保存成功 Refiner trajectories；
3. `process_refiner_rollout`：生成 Tracker RL dataset；
4. `tracker_train`：以冻结 Refiner 为 teacher 训练 BCPPO Tracker；
5. `tracker_rollout`：保存成功 Tracker trajectories；
6. `process_tracker_rollout`：生成 Generator IL dataset；
7. `generator_train`：训练 diffusion Transformer Generator。

默认正式预算是 Refiner 30001 iterations、Tracker 30001 iterations、Generator
1001 epochs。某阶段完成后可用 `START_STAGE` 从下一阶段继续；精确恢复 Refiner 或
Tracker 时使用 `REFINER_RESUME_CHECKPOINT` 或 `TRACKER_RESUME_CHECKPOINT`，并使
`*_MAX_ITERATIONS` 表示本次还要执行的迭代数、`*_FINAL_ITERATION` 表示期望终点。

示例：只跑到 Refiner rollout：

```bash
PYTHON_BIN="$PYTHON_BIN" TASK_NAME=CarryBox \
OUTPUT_DIR="$PWD/experiments/sugar_reproduction/outputs/CarryBox_refiner_repro" \
STOP_AFTER_STAGE=refiner_rollout \
  bash scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh
```

示例：已有 Refiner dataset 后从 Tracker 开始：

```bash
PYTHON_BIN="$PYTHON_BIN" TASK_NAME=CarryBox \
OUTPUT_DIR="$PWD/experiments/sugar_reproduction/outputs/CarryBox_reproduction" \
START_STAGE=tracker_train \
  bash scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh
```

## 7. H200 离屏渲染经验

H200 已在本项目实际完成 IsaacLab world-camera CarryBox 渲染；不能仅凭通用支持表将
H200 判为不支持。当前稳定路径是：

- compute node 上保持 `DISPLAY` 为空；
- 使用 Isaac Sim AppLauncher/headless recorder；
- 使用本地 ground-plane USD 和本地 frame marker；
- 对重复运行使用已转换 G1 USD，避免 scene construction 中重新导入 URDF；
- H.264 由 imageio/FFmpeg 完整写出并做全文件解码。

曾出现跨节点 `VK_ERROR_DEVICE_LOST`，但同一软件栈之后通过官方 AppLauncher 恢复。
当前 runtime 可用时不要重新开始通用 Vulkan 排障；先运行具体 collector/preflight，
只在它再次于 SimulationApp construction 之前失败时记录为环境故障。

## 8. 与 Plan 15 的连接

Plan-15 trainer/evaluator 复用：

- 官方 Tracker warm start；
- 本文保留的 Refiner teacher；
- 官方 motion 45/frame 0；
- 同一 G1 29-D action 和 SUGAR `512/256/128` actor；
- repository BCPPO。

新增部分只有：不含 measured object state 的 504-D deployed actor contract、双手
27-patch online TacSL、causal slip、live teacher handoff、在线 mass/inertia event 和
matched Z/P/PS evaluation。完整命令与当前数值结论见仓库根
[README](../README.md)。

## 9. 已知限制

- 官方仓库没有发布 raw RGB-D human video 到训练 motion 的处理代码；本记录只能从已
  处理 motion 开始复现。
- 当前 Refiner teacher 是仿真 privileged policy，不是可直接部署的真实机器人 actor。
- official hand-contact label 是 binary proxy，不是 tactile。
- TacSL 结果是高保真模拟触觉，未完成实体 GelSight 标定。
- 任何触觉收益必须来自 matched frozen physical evaluation；训练 loss、gradient 或
  单条视频不够。
