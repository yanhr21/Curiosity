# IsaacLab native whole-hand tactile

本目录只保留 IsaacLab/PhysX 整手传感、Plan-15 corrected 源码和可视化入口。旧
Plan-15 Z/P/PS 结果已在 2026-08-20 invalidity audit 后撤回；corrected tactile-only
diagnostic 也没有形成有效 matched Z/P/PS 结论，因此当前整条训练线冻结。
Newton simulator adapter、旧 Plan-13 training、teacher-residual、authority-curve、旧
bundle renderer 和节点 Vulkan 排障脚本均已迁入 ignored `legacy/`。

## 传感合同

每只 G1 手固定 27 个物理解剖 TacSL patch：掌心 `4 x 3`，五指各
proximal/middle/distal 三段。每个物理 patch 使用官方
`VisuoTactileSensor`/`GELSIGHT_R15_CFG`，底层保留 raw taxel：

- taxel position、`xyzw` orientation 和 penetration；
- nonnegative penalty normal load、friction-only signed local-XY shear 和完整 friction
  magnitude；
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
- `universal.py`：IsaacLab TacSL 原始 frame contract 与官方 sensor-data adapter；
- `run_plain_carrybox_whole_hand_visualization.sh`：普通 CarryBox 一条命令采集、渲染和
  H.264 解码；
- `run_palm_grip_whole_hand_visualization.sh`：大面积掌面接触、质量变化和物理失败；
- `run_pickbottle_whole_hand_visualization.sh`：官方 PickBottle Tracker motion；
- `run_isaaclab_r15_capsule_slip.py`：受控官方 R15 slip 校准；
- `run_online_mass_leakage_sweep.py`：Plan-15 三 seed、五质量 paired live sweep；
- `preflight_online_patch_mass_jump.py`：冻结 Refiner 的在线质量/摩擦物理 feasibility；
- `run_plan15_friction_feasibility.sh`：正式比较后独立运行
  `mu=0.5/1.0/1.5/2.0 × 6x/10x`，并汇总完整 outcome window；
- `run_plan15_frozen_seed.sh` / `run_plan15_frozen_sweep.sh`：冻结保留的 motion45-only、
  strict-termination evaluation 入口；未经明确恢复不得运行；
- `run_plan15_corrected_gate.sh`：重新采集 corrected-force motion45 scale，并串行执行
  Z/P/PS 三个 live runtime preflight；
- `run_plan15_corrected_overfit.sh`：固定 motion45、3x、20-frame delay 的 serious PS
  overfit，到 update 1499 后停止；
- `run_plan15_corrected_overfit_review.sh`：使用同一个 20-frame delay 和零 reset-pose
  noise 做 4-profile strict frozen review，禁止误用 formal 随机评测分布；
- `run_plan15_corrected_formal_seed.sh`：冻结的单分支 formal 入口；保留其严格 endpoint、
  resume 和 pipeline-lock 合同，但当前不得启动；
- `analyze_frozen_mass_reaction_window.py`：保存 trace 上的 event-aligned 提前量分析；
- `render_frozen_mass_reaction_window.py`：把该分析渲染为事件对齐的信号证据；
- `render_online_patch_mass_jump.py`：同一 rollout 的 G1/CarryBox 与左右 27-patch 视频；
- `launch_retained_child.sh`：在 retained allocation 的 `srun` compute step 中启动独立 child
  process group，并记录 Slurm step、compute host、PID/PGID；缺少 `SLURM_STEP_ID` 或位于登录
  节点时拒绝运行。

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

- 旧 Z/P/PS 同时受 reward、force semantics、reset、motion mismatch、termination 和
  inference 问题影响，不能证明 tactile benefit 或 harm。corrected model1100 为
  `14/20` physical hold、`6/20` strict success；仍无 corrected matched Z/P/PS 结论。
- 普通 CarryBox 受固定手型影响，主要由指端承载；它证明 sensor 在线工作，不证明
  整掌承载。
- 单独的掌形贴合自由刚体证明掌面覆盖，不能冒充普通箱子的托底动作。
- TacSL 是 SDF penalty tactile model，不是完整软体 FEM；现有结果必须称为高保真
  模拟触觉。
- 未完成实体 GelSight 标定前，不能声称硬件触觉或 sim-to-real。
- 训练收益只能由冻结策略的 matched physical behavior 得出；非零 tensor、gradient、
  action difference 或单条视频都不是收益结论。

完整复现、失效审计、冻结设计和未来重启 gate 见
[reproducibility](../../../DOCS/reproducibility.md)。实验输出只写入 ignored `experiments/`；不要提交
checkpoint、trace、视频、PPT 或常规 hash 清单。
