# Local experiment index

`experiments/` 只保存当前可复现主线和不可替代的人眼证据；整个目录被 Git 忽略，
checkpoint、trace、视频和日志不会 push。

## Plan 15：在线质量变化

`online_patch_tactile_mass_adaptation/` 保留：

- `leakage_sweep_v1/`：三个 seeds、五个质量条件的 paired live trace、泄漏审查、
  slip 评价和九通道尺度；
- `slip_calibration_force_v5/`：最新受控 R15 STICK/INCIPIENT/GROSS 校准；
- `live_carrybox_slip_v4_seed150814_3x/`：完整 G1 CarryBox 在线 slip 复核；
- `training_handoff/`：Z/P/PS live one-update preflight；
- `training_handoff_anchor025/`：Z/P 三 seed 的 `model_2999.pt` 正式终点和当前 PS；
- `frozen_evaluation_handoff/`：Z/P 三 seed 正式 frozen evaluation；PS 完成后写入同一根；
- `frozen_reaction_window_v2/`：三-seed event-aligned 提前量汇总；
- `feasibility_online_refiner_*` 与 `feasibility_fixed_squeeze_lower_*`：质量难度和简单
  固定响应的物理边界；
- `runtime_assets/`：避免重复 URDF 导入的已转换官方 G1 USD；
- `runtime/`：只保留当前 PS retained-job 恢复脚本与日志。

Z/P 六个已完成 seed 各只保留最终 `model_2999.pt`；旧 frame-zero 训练、中间
checkpoint、Vulkan 排障、重复 preflight、临时视频和早期 detector 版本均已迁入根
`legacy/experiments/repo_cleanup_20260817/`。

## 原生整手触觉证据

`isaaclab_g1_anatomical27_object_demos/` 只保留四个原始 IsaacLab/PhysX 包：

- `carrybox_plain_longx1p6_native_v1/`：完整 G1 抬升普通 CarryBox，主要指端承载；
- `palm_grip_free_lift_native_v2/`：`0.5 kg` 大面积掌面接触并抬升；
- `palm_grip_heavy_2kg_native_v1/`：相同动作在 `2.0 kg` 下物理失败；
- `palm_grip_release_failure_native_v1/`：主动松手后的真实下落。

每个包保留 raw `whole_hand_trace.npz`、`summary.json` 和世界相机 H.264。历史 PPT 和
重复视频已迁入 ignored legacy；原始证据没有删除。

## 官方 SUGAR 与 TacSL 依赖

`sugar_reproduction/` 只保留：

- `outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt`；
- 官方 Refiner 的小型 summary/video/curve；
- `assets/official_tacsl/` 下的 R15 calibration 和 USD；
- 当前 collector 仍需要的本地 Isaac 资产。

最短命令、固定 seeds、Z/P/PS task ID 与冻结评估设计见
[`../README.md`](../README.md)。
