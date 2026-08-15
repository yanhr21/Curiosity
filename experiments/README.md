# Local experiment index

`experiments/` 只保存当前 IsaacLab G1 触觉证据和最小 SUGAR 运行依赖；整个目录被 Git 忽略，输出不会被 push。

Plan 15 开始运行后，新的正式输出统一写入
`online_patch_tactile_mass_adaptation/`。只有 leakage audit、`Z/P/PS` 正式训练、
frozen evaluation 和同步视频可以进入该根；失败的实现版本和不再需要的中间
输出移入仓库根 `legacy/experiments/`，不得继续堆在活动实验根。

当前 Plan-15 最重要的活动证据为：

- `online_patch_tactile_mass_adaptation/leakage_sweep_v1/`：在线质量泄漏与公共尺度；
- `online_patch_tactile_mass_adaptation/training_handoff_anchor025/`：带 `0.25` BC anchor
  的三个固定 3000-update Z checkpoints；
- `online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151014/`
  到 `z_anchor025_formal_seed151016/`：三个 checkpoint 与 disjoint evaluation seed
  一一配对的 300 条正式 camera-free frozen rollout；
- `online_patch_tactile_mass_adaptation/frozen_reaction_window_v2/`：三-seed event-aligned
  reaction-window 汇总。已有同步视频保留在各 formal seed 的 `videos/` 和历史
  `z_anchor025_endpoint_audit_seed151015/videos/` 下。
- `online_patch_tactile_mass_adaptation/training_handoff_anchor025/p_seed151014/` 与
  `p_seed151015/`：live-patch、zero-slip P 正式训练。`151014` 已严格停在
  `model_2999.pt`；`151015` 在 `240922/server07` 外部取消后，当前从最后完整
  `model_250.pt` 在 retained `241217/server59` 的 iteration251 接续。
- `online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/p_anchor025_formal_seed151014/`：
  P 首个 endpoint 与 evaluation seed `152014` 配对的五质量、每项20 profiles正式
  camera-free rollout。

三个 Z endpoint 与 P/seed151014 均已冻结在 `model_2999.pt`，不得续训。当前只运行
`P/seed151015`；PS 尚未启动。

## 保留的实验

`isaaclab_g1_anatomical27_object_demos/` 目前只保留：

- `carrybox_plain_longx1p6_native_v1/`：普通平面 CarryBox，主要由指端承载；
- `palm_grip_free_lift_native_v2/`：`0.5 kg` 整掌贴合自由抬升；
- `palm_grip_heavy_2kg_native_v1/`：同动作 `2.0 kg` 物理失败；
- `palm_grip_release_failure_native_v1/`：同场景主动松手后的物理下落；
- `report_isaaclab_native_tactile_20260813/`：中文白底 PPT、全部可播放视频和构建脚本。

PickBottle、低摩擦、`1.0 kg` 和掌面压合的原始大 trace 已归档；正式报告仍保留对应 H.264，人眼证据没有丢失。当前脚本可重新采集 PickBottle、普通 CarryBox 和整掌贴合物体。

## 最小运行依赖

`sugar_reproduction/` 仅保留：

- 官方 Refiner checkpoint；
- 官方 TacSL/R15 calibration 与必要资产；
- 小型官方复现可视化。

451 MB 的历史 rollout dataset 和旧渲染环境已迁出工作区。

## 归档

本轮迁出的实验位于：

```text
/public/home/yanhongru/Curiosity_archive/repo_cleanup_20260814/
```

归档包含旧 demo/ICM/触觉训练、Newton 运行、独立刚体/软体 fixture、被否决的底托候选以及不再需要的原始 trace。不要在仓库旁创建新的 archive 根。

最短复现命令见 [`../README.md`](../README.md)。
