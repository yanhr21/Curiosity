# IsaacLab native whole-hand tactile

本目录只保留当前 IsaacLab/PhysX 整手传感、Plan-15 在线质量实验和可视化入口。
Newton runner、旧 Plan-13 tactile training、teacher-residual、authority-curve 和节点
Vulkan 排障脚本已迁入 ignored `legacy/`。

## 传感合同

每只 G1 手固定 27 个物理解剖 TacSL patch：掌心 `4 x 3`，五指各
proximal/middle/distal 三段。每个物理 patch 使用官方
`VisuoTactileSensor`/`GELSIGHT_R15_CFG`，底层保留 raw taxel：

- taxel position、`xyzw` orientation 和 penetration；
- signed local-Z normal force 与 signed local-XY shear；
- force sequence、timestamp、dt；
- optical 可用时独立保存 GelSight RGB/depth 及其时钟。

policy 不使用 taxel 作为 token。每个 control step 将 raw taxels 在线归约成左右
`2 x 27` 个 patch records：contact、normal load、mean pressure、signed XY shear、
friction utilization。Plan-15 使用四帧历史；PS 额外加入 causal slip evidence/state。
物体位姿、质量、jump flag、相对速度、PhysX 合力和 outcome 只能作为 evaluation
字段，不能进入 tactile callable 或 deployed actor。

## 主要入口

- `collect_sugar_whole_hand_carrybox.py`：完整 G1、自由物体和 54 个官方 sensor 的
  同钟 collector；
- `run_plain_carrybox_whole_hand_visualization.sh`：普通 CarryBox 一条命令采集、渲染和
  H.264 解码；
- `run_palm_grip_whole_hand_visualization.sh`：大面积掌面接触、质量变化和物理失败；
- `run_pickbottle_whole_hand_visualization.sh`：官方 PickBottle Tracker motion；
- `run_isaaclab_r15_capsule_slip.py`：受控官方 R15 slip 校准；
- `run_online_mass_leakage_sweep.py`：Plan-15 三 seed、五质量 paired live sweep；
- `preflight_online_patch_mass_jump.py`：冻结 Refiner 的在线质量/摩擦物理 feasibility；
- `run_plan15_frozen_seed.sh` / `run_plan15_frozen_sweep.sh`：固定 seed pairing 的正式
  frozen evaluation；
- `analyze_frozen_mass_reaction_window.py`：保存 trace 上的 event-aligned 提前量分析；
- `render_online_patch_mass_jump.py`：同一 rollout 的 G1/CarryBox 与左右 27-patch 视频；
- `launch_retained_child.sh`：在 retained Slurm shell 中启动独立 child process group。

正式训练入口位于：

```text
SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py
SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py
SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py
```

## 一条命令复现视频

在 GPU compute-node shell 中：

```bash
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox
```

输出目录包含 `whole_hand_trace.npz`、`summary.json`、world-camera H.264 和双手
27-patch 同钟 H.264。主视频上方显示完整 G1/物体，下方按 `4 x 3` palm 与五指三段
排列左右手；未接触 patch 保持空白，不插值或填充。

整掌贴合与 PickBottle：

```bash
bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_05kg 0.5

bash scripts/sugar/native_tactile/run_pickbottle_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_pickbottle 12 319
```

## 当前边界

- 普通 CarryBox 受固定手型影响，主要由指端承载；它证明 sensor 在线工作，不证明
  整掌承载。
- 单独的掌形贴合自由刚体证明掌面覆盖，不能冒充普通箱子的托底动作。
- TacSL 是 SDF penalty tactile model，不是完整软体 FEM；现有结果必须称为高保真
  模拟触觉。
- 未完成实体 GelSight 标定前，不能声称硬件触觉或 sim-to-real。
- 训练收益只能由冻结策略的 matched physical behavior 得出；非零 tensor、gradient、
  action difference 或单条视频都不是收益结论。

固定实验设计、Z/P/PS task ID 和完整复现命令见仓库根
[README](../../../README.md)。实验输出只写入 ignored `experiments/`；不要提交
checkpoint、trace、视频、PPT 或常规 hash 清单。
