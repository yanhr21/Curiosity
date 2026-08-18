# Local experiment index

`experiments/` 只保存当前主线、正式终点和不可替代的人眼证据。整个目录被 Git
忽略；checkpoint、trace、视频和 runtime 日志不会 push。

## Plan 15

`online_patch_tactile_mass_adaptation/` 保留：

- `leakage_sweep_v1/`：三 seeds、五质量的 paired live traces、泄漏报告和公共尺度；
- `slip_calibration_force_v5/`：受控官方 R15 slip 校准；
- `live_carrybox_slip_v4_seed150814_3x/`：完整 G1 CarryBox 在线 slip 复核；
- `training_handoff/`：Z/P/PS one-update preflight 报告，不保留 preflight 模型；
- `training_handoff_anchor025/`：正式 seed 目录只保留 `model_2999.pt` 和训练元数据；
- `frozen_evaluation_handoff/`：每个已完成 seed 的五质量 × 20 profiles 结果；
- `frozen_reaction_window_v2/`：event-aligned sensing 提前量；
- `physics_feasibility_baseline/`：原摩擦合同下的 Refiner 和固定 squeeze/lower 边界；
- `visualizations/`：代表性同步 G1/CarryBox/双手 27-patch 视频；
- `runtime_assets/`：避免重复导入的官方 G1 USD；
- `runtime/`：只写当前 retained-job child 记录。

当前正式状态：Z、P、PS 各三 seed 的 endpoint 与冻结评估全部完成，exact
300-profile-per-branch comparison 已生成。P 未证明优于 Z；PS 在 3x 显著劣于 P。
独立高摩擦 `6x/10x` sweep 已完成且不并入原比较。`6x, mu=1.5` 的完整窗口高度损失
`0.02636 m`，是唯一满足 5-cm hold 的条件；10x 四个条件均 drop。该成功条件的
camera-enabled rollout 也 hold（高度损失 `0.02552 m`），同步视频保存在
`visualizations/official_refiner_mu1p5_6x_friction_hold_single_env/`。

## 原生整手触觉证据

`isaaclab_g1_anatomical27_object_demos/` 只保留四个 IsaacLab/PhysX 包：

- `carrybox_plain_longx1p6_native_v1/`：普通 CarryBox，主要指端承载；
- `palm_grip_free_lift_native_v2/`：`0.5 kg` 大面积掌面接触并抬升；
- `palm_grip_heavy_2kg_native_v1/`：相同动作在 `2.0 kg` 下失败；
- `palm_grip_release_failure_native_v1/`：主动松手后的真实下落。

每个包包含 raw `whole_hand_trace.npz`、`summary.json`、世界相机 H.264 和同步触觉
H.264。它们是模拟证据，不是硬件 GelSight 标定。

## 官方依赖

`sugar_reproduction/` 只保留官方 Refiner checkpoint、必要 summary/video、TacSL R15
calibration/USD 和当前 collector 所需的本地 Isaac assets。

失败、重复、过时 preflight、中间 checkpoint、历史日志和一次性排障均在根
`legacy/`，不属于当前结果。最短复现命令见 [`../README.md`](../README.md)。
