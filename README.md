# Curiosity

当前唯一研究主线是在 IsaacLab/PhysX 中检验：完整 SUGAR G1 已经抬起
CarryBox 后，在线整手触觉能否帮助策略应对几何与外观不变、质量突然增加的
情况。活动设计见 [Plan 15](PLAN/15_online_patch_tactile_mass_adaptation/plan.md)，
执行清单见 [TODO 15](TODO/15_online_patch_tactile_mass_adaptation/todo.md)。RGB、demo
following、ICM/Curiosity、Newton simulator 和软体训练均为历史方向，不在当前队列。

## 方法与输入输出

官方 SUGAR 包含三部分：Refiner 依靠人体—物体参考轨迹和仿真特权状态产生高质量
完整 G1 动作；Tracker 从 Refiner rollout 学习可部署的运动跟踪策略；Generator 对
Tracker rollout 建模并在推理时生成运动计划。本项目不替换官方方法：student 从官方
Tracker checkpoint 初始化，训练期使用冻结官方 Refiner teacher，并沿用 repository
BCPPO、`512/256/128` actor 和 `29-D` action。官方结构与完整复现记录见
[SUGAR CarryBox 复现](DOCS/sugar_carrybox_reproduction_full_record.md)。

部署 actor 只读取 `504-D` Tracker-command/proprioception，不读取 measured object
state、质量倍率、jump flag、RGB 或 future frame。官方 Refiner 的 `890-D` observation
只进入训练期 teacher/critic。由于质量变化也会通过关节下沉和跟踪误差泄漏到
proprioception，实验检验的是触觉相对 proprioception 的增量收益，不能声称“只有触觉
能感知重量”。

每只手固定 27 个物理解剖 patch：掌心 `4 x 3`，拇指、食指、中指、无名指和小指各
有 proximal/middle/distal 三段。每个 patch 是一个 policy token，包含 contact、法向
载荷、平均压力、signed local-XY shear、friction utilization，以及 PS 分支中的 causal
slip evidence/state。底层由 IsaacLab v2.3.2 官方 TacSL `VisuoTactileSensor` 和 R15
taxel 阵列计算，但 taxel 只作为物理采样和审计源，不是 policy 单元。所有输入都在
当前 rollout 内、下一次 actor 调用前在线生成。

## 固定实验设计

冻结 Refiner 在同一个 PhysX episode 中从 motion 45/frame 0 控制 G1，直到箱子连续
10 个 control frames 抬升至少 `0.05 m`。随后无 reset、teleport 或 replay 地把控制权
交给 student；再等待匹配的 `10--50` 帧，把质量和 inertia 从 `0.3023375869 kg`
在线改为 `1.5x/3x/6x/10x`。`1.0x` 是 no-jump control。

三个正式分支除触觉 observation 外完全匹配：

- `Z`：patch/slip tensor 精确为零，actor 和 scheduler 都不读取 TacSL；
- `P`：在线 contact/load/pressure/shear/friction，slip fields 精确为零；
- `PS`：与 P 相同，再调用 causal、batch-stateful `PatchSlipDetector.update(...)`。

每个分支固定训练 seeds `151014/151015/151016`，每个 seed 恰好 3000 updates。
BCPPO 的 0--499 为纯 distillation，500--999 加 critic warmup，1000--1999 提升 PPO
authority，2000--2999 为 steady full-PPO；三分支共同保留 `0.25` distillation floor。
冻结评估一一配对 `151014->152014`、`151015->152015`、`151016->152016`，每对在
五个质量条件各运行 20 profiles，因此每个完整分支恰好 300 条 rollout。

每个正式 seed 到 `model_2999.pt` 后必须停止。先审查 checkpoint finiteness、live
handoff、质量读回、80-frame 物理窗口、动作连续性和同步视频，再显式启动该 seed 的
冻结评估；不得自动开始下一个正式 seed。

## 主要结论与当前进展

截至 2026-08-18：

- 在线 sensing 与 slip 已打通。15 条 paired leakage traces 的质量读回、jump 前双手
  接触和 54-patch clock 均通过。continuous patch 信号约在 jump 后 13 帧稳定区分
  质量，proprioception 约为 35 帧；这是信息窗口，不是策略收益。
- 在 119 条重箱下落中，continuous patch change 全部早于 drop，中位提前 21 帧；
  contact binary 中位只提前 15 帧。对 133 条至少下沉 `0.02 m` 的轨迹，continuous
  patch change 覆盖 `133/133`，contact binary 只覆盖 `81/133`。这证明连续压力/剪切
  相对 binary contact 有信息优势，但仍不证明训练收益。
- causal slip 的受控 R15 状态正确数为 STICK `109/111`、INCIPIENT `109/109`、GROSS
  `19/20`。完整 G1 CarryBox 3x trace 对 held-out active-taxel velocity 的 precision
  `1.0`、recall `0.9971`，中位延迟 0 帧、p95 1 帧。
- Z 三 seed 的 eligible physical holds 为
  `59/59, 59/59, 52/59, 1/59, 0/59`，drops 为
  `0/59, 0/59, 2/59, 58/59, 59/59`，顺序均为
  `1x/1.5x/3x/6x/10x`。
- P 三 seed 的 holds 为 `59,59,49,0,0`，drops 为 `0,0,8,59,59`。3x 的 P-Z
  hold interval 跨过零，因此 P 没有证明收益，并呈更差趋势。
- PS 三 seed 的 holds 为 `59,58,33,0,0`，drops 为 `0,1,22,59,59`。严格比较中，
  3x PS-P hold 差值为 `-0.2712`，paired hierarchical-bootstrap 95% CI
  `[-0.4655,-0.0667]`；drop 差值为 `+0.2373`，CI `[0.1053,0.3833]`。因此当前
  PS 不但没有证明收益，还在 3x 显著劣于 P。连续触觉有更早的信息，并不等于当前
  BCPPO 已学会利用它。
- PS-151016 的 endpoint 已完成 finite checkpoint、live no-reset handoff、质量读回、
  450-frame outcome window 与视频审查。camera-enabled 证据包含完整 G1/CarryBox、
  同钟双手 27-patch：
  `experiments/online_patch_tactile_mass_adaptation/visualizations/`
  `ps_seed151016_1p5x_endpoint_review_single_env/ps_seed151016_1p5x_world_bilateral27.mp4`
  显示自己的 1.5x hold；同目录体系下的 3x 视频显示自己的真实 drop。
- PS-151015 的 camera-enabled 3x 证据也包含完整 450 帧、同钟 G1/CarryBox 和双手
  27-patch：
  `experiments/online_patch_tactile_mass_adaptation/visualizations/`
  `ps_seed151015_3x_endpoint_review_single_env/ps_seed151015_3x_world_bilateral27.mp4`。
  视频证明它自己的 camera rollout 持箱，不冒充 camera-free formal trace 的逐帧 replay。
- 普通平面 CarryBox 样本主要由突出掌面的指端承载；独立的 `0.5 kg` 掌形贴合样本
  达到左掌 `9/12`、右掌 `12/12` 接触，并完成抬升。主动松手和 `2.0 kg` 相同动作均
  出现真实下落。这些是高保真模拟触觉，不是实体 GelSight 标定或 sim-to-real。

三分支 exact comparison 已完成，每分支恰好 300 rollouts。随后单独运行
static/dynamic friction `0.5/0.5、1.0/1.0、1.5/1.5、2.0/2.0` 的 `6x/10x`
feasibility sweep。八条的材料/质量读回、jump 前双手接触和 outcome window 均通过。
6x 高度损失依次为 `0.5589/0.5429/0.02636/0.06596 m`，只有 `mu=1.5` 达到 5-cm
hold；10x 四个条件均 drop。这证明 6x 在现有 controller 下并非物理不可能，但不是
单调摩擦曲线，因为摩擦同时改变 pickup dynamics 和 jump timing。该实验不与原始
Z/P/PS 统计混合。

成功的 `6x, mu=1.5` camera-enabled rollout 也通过自身审查：实际质量读回
`1.8140255 kg`、jump 后 124 帧、最大高度损失 `0.02552 m`、hold=true、drop=false。
450 帧完整 G1/CarryBox 和双手 27-patch H.264 位于：
`experiments/online_patch_tactile_mass_adaptation/visualizations/`
`official_refiner_mu1p5_6x_friction_hold_single_env/official_refiner_mu1p5_6x_world_bilateral27.mp4`。
这是相机 rollout 自己的结果，不冒充 camera-free trace 的逐帧 replay。

## 活动实验目录

`experiments/` 完全被 Git 忽略，只保留三类不可替代的本地输出：

```text
experiments/
├── online_patch_tactile_mass_adaptation/
│   ├── leakage_sweep_v1/              # paired sensing traces and scales
│   ├── slip_calibration_force_v5/      # controlled R15 slip
│   ├── live_carrybox_slip_v4_seed150814_3x/
│   ├── training_handoff/               # Z/P/PS one-update reports
│   ├── training_handoff_anchor025/     # formal endpoint checkpoints only
│   ├── frozen_evaluation_handoff/      # formal 100-rollout seed results
│   ├── frozen_reaction_window_v2/
│   ├── physics_feasibility_baseline/
│   ├── friction_feasibility_after_ps/
│   ├── visualizations/
│   ├── runtime_assets/
│   └── runtime/                        # current child records only
├── isaaclab_g1_anatomical27_object_demos/
│   ├── carrybox_plain_longx1p6_native_v1/
│   ├── palm_grip_free_lift_native_v2/
│   ├── palm_grip_heavy_2kg_native_v1/
│   └── palm_grip_release_failure_native_v1/
└── sugar_reproduction/                 # official Refiner and TacSL assets
```

失败、重复、旧 frame-zero、Newton simulator、Vulkan 排障、中间 checkpoint、历史 PPT
和 runtime 流水账均移入根 `legacy/`；该目录被 Git 忽略，不是活动复现入口。

## 运行环境

所有 IsaacLab 命令必须在保留的 GPU compute-node shell 中运行：

```bash
cd /public/home/yanhongru/Curiosity
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
export DISPLAY=
export OMNI_KIT_ACCEPT_EULA=Y
```

长任务用 `scripts/sugar/native_tactile/launch_retained_child.sh` 记录独立 PID/PGID。
任务结束不退出 allocation；需要换任务时只终止记录的 child process group。

## 最短复现路径

### 1. 官方 SUGAR CarryBox

```bash
PYTHON_BIN="$PYTHON_BIN" bash scripts/sugar/preflight_official_sugar_env.sh
PYTHON_BIN="$PYTHON_BIN" NUM_ENVS=16 VIDEO_LENGTH=200 \
  bash scripts/sugar/run_official_sugar_carrybox_inference.sh
```

从头执行 Refiner rollout、Tracker 和 Generator：

```bash
PYTHON_BIN="$PYTHON_BIN" TASK_NAME=CarryBox \
OUTPUT_DIR="$PWD/experiments/sugar_reproduction/outputs/CarryBox_reproduction" \
  bash scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh
```

### 2. 完整 G1 双手 27-patch 触觉视频

```bash
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox

bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_05kg 0.5

bash scripts/sugar/native_tactile/run_pickbottle_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_pickbottle 12 319
```

每个输出包含 raw trace、summary、世界相机 H.264 和双手 27-patch 同钟 H.264。
复现主动松手失败时仍使用同一个入口：

```bash
PALM_GRIP_SCENARIO=release_failure \
bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_release 0.5
```

### 3. 在线质量泄漏与 slip

```bash
bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/online_patch_tactile_mass_adaptation/runtime/r15_slip.process \
  --status experiments/online_patch_tactile_mass_adaptation/runtime/r15_slip.status \
  --log experiments/online_patch_tactile_mass_adaptation/runtime/r15_slip.log \
  --tag plan15-r15-slip --foreground -- \
  "$PYTHON_BIN" scripts/sugar/native_tactile/run_isaaclab_r15_capsule_slip.py \
    --output-root experiments/online_patch_tactile_mass_adaptation/slip_reproduction \
    --frames 240 --force-only --headless --device cuda:0

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/online_patch_tactile_mass_adaptation/runtime/leakage.process \
  --status experiments/online_patch_tactile_mass_adaptation/runtime/leakage.status \
  --log experiments/online_patch_tactile_mass_adaptation/runtime/leakage.log \
  --tag plan15-leakage --foreground -- \
  "$PYTHON_BIN" scripts/sugar/native_tactile/run_online_mass_leakage_sweep.py \
    --output-root experiments/online_patch_tactile_mass_adaptation/leakage_reproduction \
    --device cuda:0
```

### 4. Z/P/PS 单 seed 正式训练

```bash
BRANCH=PS                     # Z, P, or PS
SEED=151016
SCALE="$PWD/experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json"
OUT="$PWD/experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/${BRANCH,,}_seed${SEED}"

"$PYTHON_BIN" -u SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py \
  --task "Sugar-G129dof-CarryBox-OnlineMass-Patch-${BRANCH}-BCPPO" \
  --patch-scale-file "$SCALE" --seed "$SEED" --log_dir "$OUT" \
  --headless --device cuda:0
```

从完整 numbered checkpoint 恢复时增加
`--resume_checkpoint_path "$OUT/model_2750.pt"`。总终点仍固定为 3000 updates。

### 5. 冻结评估与三分支比较

单个 PS endpoint 先人工审查，再显式开放评估：

```bash
PLAN15_ALLOW_PS_ENDPOINT_EVALUATION=1 \
bash scripts/sugar/native_tactile/run_plan15_frozen_seed.sh \
  PS \
  experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/ps_seed151016/model_2999.pt \
  151016 152016 \
  experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/ps_anchor025_formal_seed151016 \
  cuda:0
```

三个 PS seed 完成后运行正式比较：

```bash
"$PYTHON_BIN" SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py \
  --z-root \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151014 \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151015 \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151016 \
  --p-root \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/p_anchor025_formal_seed151014 \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/p_anchor025_formal_seed151015 \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/p_anchor025_formal_seed151016 \
  --ps-root \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/ps_anchor025_formal_seed151014 \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/ps_anchor025_formal_seed151015 \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/ps_anchor025_formal_seed151016 \
  --output experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_p_ps_formal_comparison_v1.json
```

### 6. 比较后独立摩擦可行性

该入口先要求上一步 exact comparison 已存在，再运行
`mu=0.5/1.0/1.5/2.0 × 6x/10x`；每条必须有真实材料/质量读回、jump 前双手接触和
完整 80-frame outcome window：

```bash
bash scripts/sugar/native_tactile/run_plan15_friction_feasibility.sh cuda:0
```

代码入口与字段说明见
[native tactile README](scripts/sugar/native_tactile/README.md)；本地实验索引见
[experiments README](experiments/README.md)。checkpoint、trace、视频和日志均不得
提交或推送。

---

## 正确性审计 / Correctness audit — 2026-08-19

以上 Z/P/PS 结论在 2026-08-19 经过一次完整代码审计。**审计结果：当前 null result 不能
作为"触觉无用"的结论报告。** 完整条目（115 条，含 file:line）见
[`claude_context/findings.md`](claude_context/findings.md)，可浏览页面为
`claude_context/index.html`（`python3 claude_context/serve.py`）。以下为英文摘要。

*Written in English for precision; the findings log is the authoritative record.*

### Verified sound — do not rebuild

- **Branch matching is real.** The 54 TacSL sensor bodies do **not** change Z's physics:
  the sensors create read-only PhysX views and never write forces back, neither
  observation term consumes RNG, and the jump/handoff schedules are deterministic. Z's
  trajectory is matched to P/PS at the same seed. This was the single biggest risk to the
  design and it is clean.
- **The mass event** is written at the action boundary with inertia scaled by exactly
  `target/default` and both values read back, raising past `rtol=1e-6`.
- **Z's gradient isolation** was measured, not argued: all 41 encoder parameters are in
  the optimizer and every gradient is exactly zero.
- **The eligibility gate cannot bias the comparison.** 59 = 60 − one seed-151014 profile
  whose jump never lands by frame 370; because the jump is scheduled off the *frozen
  teacher's* pickup and fires unconditionally, that exclusion is branch- and
  factor-invariant. Confirmed arithmetically: −16/59 = −0.27119 = the reported −0.2712.
- Every dimension in the contract (504 / 890 / 632 / 1944 / 128 / 29) holds and is
  asserted at runtime.

### Blocking defects

1. **The reward penalises grasping; the term meant to reward it is dead.** `hoi_contact`
   (+1.0) reads ContactSensors on `left/right_rubber_hand`, whose collision subtrees
   `_disable_original_hand_collision_owner` deactivates at spawn — so `is_contact` is
   permanently `False` and the term reduces to `(False == contact_label)`, a pure function
   of the reference clock with **no behavioural gradient**. `undesired_contacts` (−1.0)
   counts bodies over 0.1 N and matches all 54 elastomer patches (87 of 91 bodies), giving
   −0.02 per contacting body per step against a maximum achievable positive reward of
   5.125 — **six patches in contact cancel everything**. Nothing in the 21 terms asks the
   policy to hold the box up.
2. **`friction_utilization` is invariant to the object's friction** (divides by the
   sensor's fixed `mu = 0.5`, which TacSL already used to cap the numerator) — while
   `obj_physics_material` randomizes the box's friction over `U[0.2,0.8]` **during
   training**, giving a true contact coefficient of 0.35–0.65.
3. **Under stick the shear channel is a geometric leak of the normal force**, not
   traction: TacSL projects the *total* force into a per-taxel frame, so shear ≈
   `k_n·d·sinθ` and utilization ≈ `2·tanθ`. Measured at exactly zero relative velocity,
   17.2° of off-centre contact gives utilization **0.622 — past the 0.60 incipient-slip
   trigger with no slip at all**. Any utilization above 1.0 is geometric contamination by
   construction.
4. **Trained on motions 0–3, evaluated only on motion 45.** Every headline number is
   out-of-distribution.
5. **The slip detector's reset is silently swallowed between evaluation batches.**
   `_online_patch_slip_history` returns early on a `common_step_counter` match *before*
   calling `detector.update`, and `env.reset()` does not increment that counter — so the
   detector's `previous_*` buffers and its GROSS latch survive the episode boundary.
   **P has no differencing state, so this bug is asymmetric and favours P over PS**, on
   the evaluation path that produced the −0.2712. Best single candidate for PS < P.

### Weaker than reported

- Every number came from `--physical-outcome-view`, which suppresses all six SUGAR
  terminations, so `eligible` means only "the jump landed by frame 370". The stricter
  `strict_sugar_hold_success` labels exist in every summary and have never been reported.
- `1×` writes no mass at all — it is a no-perturbation control, not a 1× jump.
- The 95 % CI is a *percentile* two-level bootstrap over **3 seed clusters**, no BCa, no
  correction for the 180 intervals emitted per run. Strong point estimate, optimistic
  interval.
- The 20 "profiles" per run are near-replicates: same motion 45 frame 0, no push or
  observation randomization, differing only in a deterministic jump delay and four per-env
  friction draws that repeat across batches.
- `hold + drop + safe_lower` does not sum to the eligible count — Z at 3× leaves 5 of 59
  profiles in no label at all.

### Remediation order

1. Reward: exclude `.*_anatomical_.*` from `undesired_contacts`, repoint `hoi_contact`,
   consider a hold term. **Cheapest, furthest upstream.**
2. Hoist the reset out of the step-counter guard in `_online_patch_slip_history`.
3. Measure per-channel variance and mass/friction MI during the *hold* phase on saved
   traces before any retrain. No GPU needed.
4. Rebuild the sensor reduction in the **contact frame** with a minimum-load gate and the
   object's real material friction. The per-taxel frame causing the leak is a **local**
   modification (upstream v2.3.2 uses one constant quaternion per sensor), so this needs no
   vendor change; only a tangential spring with stick memory would.
5. Close the train/eval motion split; report strict-view numbers alongside.

**既有 checkpoint 不能跨 sensing 修改复用** — channel scales 存在 encoder 的 persistent
buffer 里、进了每个 checkpoint，且没有任何机制把 scale 文件绑定到产生它的 channel 定义上。
