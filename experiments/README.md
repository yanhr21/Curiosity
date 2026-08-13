# Local experiment index

`experiments/` 只保存当前 IsaacLab G1 触觉证据和最小 SUGAR 运行依赖；整个目录被 Git 忽略，输出不会被 push。

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
