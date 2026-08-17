# Curiosity

本仓库当前只执行一条主线：在 IsaacLab/PhysX 中验证在线整手触觉能否帮助完整
SUGAR G1 在已经抬起 CarryBox 后，应对不改变几何和外观的突然质量变化。当前唯一计划
是 [Plan 15](PLAN/15_online_patch_tactile_mass_adaptation/plan.md)，任务状态见
[Plan 15 TODO](TODO/15_online_patch_tactile_mass_adaptation/todo.md)。RGB、demo following、
ICM/Curiosity、Newton 仿真和软体训练均不属于当前执行队列。

## 系统与实验问题

官方 SUGAR 提供三阶段方法：Refiner 用人体物体轨迹和仿真特权状态优化完整 G1
动作；Tracker 从 Refiner rollout 学习可部署的运动跟踪策略；Generator 对 Tracker
rollout 建模并在推理时产生运动计划。Plan 15 不重写这个方法：使用官方 Tracker
checkpoint 初始化 student，使用冻结的官方 Refiner 作为训练期 teacher，并沿用仓库
BCPPO、`512/256/128` actor 和 29-D action。官方结构、输入输出与从头复现见
[SUGAR CarryBox 复现记录](DOCS/sugar_carrybox_reproduction_full_record.md)。

部署 actor 只读取不含 measured object state 的 `504-D` Tracker-command/
proprioception。官方 Refiner 的 `890-D` 特权 observation 只进入冻结 teacher 和训练期
critic。由于 `joint_pos/joint_vel` 会在手臂受载后变化，本项目检验的是“触觉相对本体
感受是否有增量帮助”，不能声称只有触觉能感知重量。

每只手固定 27 个物理解剖 patch：掌心 `4 x 3`，拇指、食指、中指、无名指和小指各
有 proximal/middle/distal 三段。每个 patch 是 policy token，包含：

- 接触状态、法向载荷和平均压力；
- signed local-XY shear 和 friction utilization；
- causal slip evidence，以及 `NO_CONTACT/STICK/INCIPIENT/GROSS` 状态。

底层每个 patch 使用 IsaacLab v2.3.2 官方 TacSL `VisuoTactileSensor` 和 R15 taxel
阵列。taxel 是传感面内的原始采样点，R15 是官方 GelSight R15 传感器配置；policy
不会把 taxel 当成独立输入单元。所有训练信号都在当前 IsaacLab rollout 内、下一次
actor 调用之前在线生成，不读取离线 trace、物体速度、质量标签或未来帧。

## 固定实验设计

同一 PhysX episode 中，冻结 Refiner 从 motion 45/frame 0 控制机器人，直到箱子连续
10 个 control frame 抬升至少 `0.05 m`。随后无 reset、teleport 或 replay 地交给
student；再等待匹配的 `10--50` 帧后，将 CarryBox 的质量和 inertia 从
`0.3023375869 kg` 同步改为 `1.5x/3x/6x/10x`。`1.0x` 是 no-jump control。

三个正式分支除 tactile observation 外完全匹配：

- `Z`：patch/slip tensor 精确为零，actor 与质量调度器都不读取 TacSL；
- `P`：在线 contact/load/pressure/shear/friction，slip fields 精确为零；
- `PS`：与 P 相同，并调用 causal、batch-stateful `PatchSlipDetector.update(...)`。

每个分支固定训练 seeds `151014/151015/151016`，每个 seed 恰好 3000 updates。
BCPPO 的 0--499 为纯 distillation，500--999 加 critic warmup，1000--1999 提升 PPO
权重，2000--2999 为 steady full-PPO；三分支共同使用 `0.25` stage-3 distillation
floor。冻结评估一一配对 `151014->152014`、`151015->152015`、
`151016->152016`，每对在五个质量条件各运行 20 profiles，即每个分支 300 条。

## 主要结论与当前进展

截至 2026-08-17：

- 在线 sensing 已打通。`3 seeds x 5 mass factors` 的泄漏 sweep 全部完成，15 条轨迹
  的 action/event 匹配、质量读回、jump 前双手接触和 54-patch sensor clock 均通过。
  jump 当帧 contact binary 不变，但连续 patch load/pressure 与 `504-D` proprio 都
  响应。探索性 leave-one-seed-out probe 首次稳定区分质量的时间为 patch tactile
  约 13 帧、proprio 约 35 帧；这只说明提前信息窗口，不是 policy 收益。
- causal slip 已校准。受控官方 R15 轨迹将 fixed contact、`0.006 m/s` slow slide、
  `0.03 m/s` fast slide、`0.01 m/s` return 分为 STICK/INCIPIENT/GROSS/INCIPIENT；
  状态正确数为 `109/111`、`109/109`、`19/20`。在完整 G1 CarryBox 3x 在线轨迹中，
  对 evaluation-only active-taxel velocity 的 precision 为 `1.0`、recall 为 `0.9971`，
  中位延迟 0 帧、p95 1 帧。
- Z/P/PS live handoff preflight 均完成一次真实 BCPPO update。每项有 1440 transitions、
  4 次连续物理 handoff 和 2 次真实 mass event；Z 为 0 TacSL reads，P/PS 各有
  `361 x 54 = 19,494` 个官方 patch reads，P 为 0 slip calls，PS 为 361 次 causal
  slip calls。
- Z 三个正式 seed 已完成，终点均为 `model_2999.pt`。P 三个正式 seed 及其冻结评估
  也已完成。共同 eligible 分母为 59 时，3x hold 为 P `49/59`、Z `52/59`，3x drop
  为 P `8/59`、Z `2/59`；3x hold 的 P-Z 均值为 `-0.0508`，分层配对 bootstrap
  95% 区间 `[-0.2881, 0.1552]`。因此 P 没有证明触觉收益，且当前呈更差趋势。
- PS seed `151014` 当前最新完整恢复点是 `model_2750.pt`，剩余 249 updates。五天
  retained job `243374/server60` 已获得独占 pipeline lock，并从 iteration 2751
  继续；其他已获批 allocation 保留但不允许并发写正式 seed。PS seeds
  `151015/151016` 及其 300-rollout 冻结评估尚未完成，所以 Z/P/PS 最终比较尚无结论。
- PS 完成后将独立固定 CarryBox static/dynamic friction 为 `0.5/1.0/1.5/2.0`，在
  `6x/10x` 上运行冻结 Refiner feasibility sweep。该 sweep 用来区分控制能力和摩擦
  上限，不覆盖原 Z/P/PS 比较。

现有整手触觉可视化还给出三条重要物理结论：普通平面 CarryBox 抬升 `0.548 m`，但
主要由突出掌面的指端承载；`0.5 kg` 掌形贴合刚体抬升 `0.577 m`，双掌接触覆盖达到
左 `9/12`、右 `12/12`；相同动作在主动松手或物体增至 `2.0 kg` 后出现真实下落。
这些是高保真模拟触觉，不是实体 GelSight 标定或 sim-to-real 结果。

## 保留的实验

`experiments/` 是本地输出目录并被 Git 忽略。活动树只保留：

```text
experiments/
├── online_patch_tactile_mass_adaptation/
│   ├── leakage_sweep_v1/
│   ├── slip_calibration_force_v5/
│   ├── live_carrybox_slip_v4_seed150814_3x/
│   ├── training_handoff/                 # Z/P/PS live preflights
│   ├── training_handoff_anchor025/       # formal Z/P endpoints and active PS
│   ├── frozen_evaluation_handoff/        # formal Z/P, future PS
│   ├── frozen_reaction_window_v2/
│   ├── runtime_assets/
│   └── runtime/                          # active retained-job records only
├── isaaclab_g1_anatomical27_object_demos/
│   ├── carrybox_plain_longx1p6_native_v1/
│   ├── palm_grip_free_lift_native_v2/
│   ├── palm_grip_heavy_2kg_native_v1/
│   └── palm_grip_release_failure_native_v1/
└── sugar_reproduction/                   # Refiner checkpoint and TacSL assets
```

失败、重复、旧 frame-zero、Vulkan 排障、中间 checkpoint 和历史 PPT 已移入仓库根
`legacy/`；它被 Git 忽略，不是活动结果，也不得作为当前复现入口。

## 运行环境

所有 IsaacLab 命令必须在已分配 GPU 的 compute-node shell 中运行：

```bash
cd /public/home/yanhongru/Curiosity
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
export DISPLAY=
export OMNI_KIT_ACCEPT_EULA=Y
```

长任务使用 `scripts/sugar/native_tactile/launch_retained_child.sh` 单独记录 child
PID/PGID。不要退出 retained allocation；需要停止任务时只终止记录的 child process
group，不能向 tmux allocation shell 发送无目标 `Ctrl+C`。

## 最短复现指南

### 1. 官方 SUGAR CarryBox

已有官方 released checkpoint 时，先验证环境，再运行官方 inference：

```bash
PYTHON_BIN="$PYTHON_BIN" bash scripts/sugar/preflight_official_sugar_env.sh
PYTHON_BIN="$PYTHON_BIN" NUM_ENVS=16 VIDEO_LENGTH=200 \
  bash scripts/sugar/run_official_sugar_carrybox_inference.sh
```

从头执行 Refiner rollout、Tracker 和 Generator 的完整顺序：

```bash
PYTHON_BIN="$PYTHON_BIN" TASK_NAME=CarryBox \
OUTPUT_DIR="$PWD/experiments/sugar_reproduction/outputs/CarryBox_reproduction" \
  bash scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh
```

### 2. 完整 G1 双手 27-patch 触觉视频

普通 CarryBox：

```bash
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox
```

整掌贴合和 PickBottle：

```bash
bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_05kg 0.5

bash scripts/sugar/native_tactile/run_pickbottle_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_pickbottle 12 319
```

每个输出包含原始 `whole_hand_trace.npz`、`summary.json`、世界相机 H.264 和双手
27-patch 同钟 H.264。触觉来自官方 sensor tensor；物体位姿、PhysX contact force 和
相对速度只作为评价字段。

### 3. 在线质量泄漏与 slip

```bash
bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/online_patch_tactile_mass_adaptation/runtime/leakage_repro.process \
  --status experiments/online_patch_tactile_mass_adaptation/runtime/leakage_repro.status \
  --log experiments/online_patch_tactile_mass_adaptation/runtime/leakage_repro.log \
  --tag plan15-leakage-repro --foreground -- \
  "$PYTHON_BIN" scripts/sugar/native_tactile/run_online_mass_leakage_sweep.py \
    --output-root experiments/online_patch_tactile_mass_adaptation/leakage_reproduction \
    --device cuda:0
```

输出包含 15 条 paired live trace、`leakage_audit.json`、`slip_evaluation.json` 和
`patch_channel_scales.json`。relative tangential velocity 只在 rollout 保存后用于评价
slip，不进入 detector 或 actor。

### 4. Z/P/PS 正式训练

三个 task ID 分别为：

```text
Sugar-G129dof-CarryBox-OnlineMass-Patch-Z-BCPPO
Sugar-G129dof-CarryBox-OnlineMass-Patch-P-BCPPO
Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-BCPPO
```

单个 seed 的标准命令如下；把 `BRANCH` 替换为 `Z`、`P` 或 `PS`：

```bash
BRANCH=Z
SEED=151014
SCALE="$PWD/experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json"
OUT="$PWD/experiments/online_patch_tactile_mass_adaptation/reproduction/${BRANCH,,}_seed${SEED}"

"$PYTHON_BIN" -u SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py \
  --task "Sugar-G129dof-CarryBox-OnlineMass-Patch-${BRANCH}-BCPPO" \
  --patch-scale-file "$SCALE" --seed "$SEED" --log_dir "$OUT" \
  --headless --device cuda:0
```

从完整 checkpoint 精确恢复时只增加：

```bash
--resume_checkpoint_path "$OUT/model_2750.pt"
```

launcher 固定 3000-update 总预算；resume 不会重新计算已保存 updates。三个分支必须
串行执行相同三个 seeds，不得只给某一分支增加预算或 profiles。

### 5. 冻结评估、三分支比较与反应窗口

一个完整分支的三 seed sweep：

```bash
bash scripts/sugar/native_tactile/run_plan15_frozen_sweep.sh Z \
  experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/z_seed151014/model_2999.pt \
  experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/z_seed151015/model_2999.pt \
  experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/z_seed151016/model_2999.pt \
  experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_reproduction \
  cuda:0
```

P/PS 只替换 branch、checkpoint 和 output root。三个分支完成后：

```bash
"$PYTHON_BIN" SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py \
  --z-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151014 \
           experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151015 \
           experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151016 \
  --p-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/p_anchor025_formal_seed151014 \
           experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/p_anchor025_formal_seed151015 \
           experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/p_anchor025_formal_seed151016 \
  --ps-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/ps_anchor025_formal_seed151014 \
            experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/ps_anchor025_formal_seed151015 \
            experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/ps_anchor025_formal_seed151016 \
  --output experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_p_ps_comparison.json
```

现有 Z 三 seed 的 event-aligned reaction window 可离线复算：

```bash
"$PYTHON_BIN" scripts/sugar/native_tactile/analyze_frozen_mass_reaction_window.py \
  --seed-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151014 \
  --seed-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151015 \
  --seed-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151016 \
  --scale-file experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json \
  --output experiments/online_patch_tactile_mass_adaptation/frozen_reaction_window_reproduction/summary.json
```

## 代码与文档入口

- 整手传感、字段和视频：[scripts/sugar/native_tactile/README.md](scripts/sugar/native_tactile/README.md)
- 当前实验索引：[experiments/README.md](experiments/README.md)
- 当前执行计划：[PLAN/15_online_patch_tactile_mass_adaptation/plan.md](PLAN/15_online_patch_tactile_mass_adaptation/plan.md)
- 当前任务状态：[TODO/15_online_patch_tactile_mass_adaptation/todo.md](TODO/15_online_patch_tactile_mass_adaptation/todo.md)
- 官方 SUGAR 复现：[DOCS/sugar_carrybox_reproduction_full_record.md](DOCS/sugar_carrybox_reproduction_full_record.md)

实验 trace、视频、checkpoint、PPT 和 runtime log 都不进入 Git。代码提交只包含可复现
入口、配置、测试和文档。
