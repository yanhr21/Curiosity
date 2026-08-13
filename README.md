# Curiosity

当前主线是在 **IsaacLab/PhysX** 中为完整 SUGAR G1 构建可复用的原生整手触觉演示。每只手固定安装 27 个官方 TacSL/R15 patch：掌心 `4 x 3`，拇指、食指、中指、无名指和小指各 3 段。Newton 只允许提供 USD/mesh 资产，不再作为新演示的模拟器；当前不训练策略。

## 当前结果

- 普通平面 CarryBox：完整 G1 将自由刚体抬升 `0.548 m`，`76/80` 帧双手有原生触觉。由于 G1 固定手型的指尖比掌心突出约 `2–3 cm`，这个物体主要由指端受力，不能称为整掌承载。
- 整掌贴合刚体：`0.5 kg` 自由物体抬升 `0.577 m`，`80/80` 帧双手有触觉，`79/80` 帧双掌接触；峰值掌区覆盖为左 `9/12`、右 `12/12`。
- 物理失败：相同动作在主动松手后物体从峰值下落 `0.394 m`；质量改为 `2.0 kg` 后最终高度比初始低 `0.179 m`。
- PickBottle：官方 motion 12 完成双手抬升并持续接触；motion 17 接触后发生弹道释放。两者使用官方 510-D Tracker 输入和 29-D 动作，不重放 CarryBox 动作。
- 当前未完成：普通箱子的“左手托底、右手扶侧”持续受力样本；TacSL 数值与 PhysX 支撑力之间的绝对标定。

这些结果只能称为**高保真模拟触觉**，不是硬件 GelSight 标定，也不是 sim-to-real。

## 最短复现路径

在已保留的 H200 Slurm shell 中，从仓库根目录运行：

```bash
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox
```

脚本完成一次 80 帧完整 G1 CarryBox 采集和渲染，输出：

```text
experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox/
├── whole_hand_trace.npz
├── summary.json
├── world_carrybox.mp4
└── videos/plain_carrybox_world_bilateral_taxels.mp4
```

主视频上方是同一时钟下的 G1/物体世界画面，下方是左右手各 27 个解剖 patch。原始 trace 保留逐 taxel penetration、signed local-Z normal force、signed local-XY shear、位姿和时间戳；渲染值不是 contact label、刚体合力或手工生成热图。

复现整掌贴合物体和非箱形物体：

```bash
bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_05kg 0.5

bash scripts/sugar/native_tactile/run_pickbottle_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_pickbottle 12 319
```

运行依赖：

- `/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python`；
- 本仓库中的 `IsaacLab/`、`SUGAR/` 和官方 SUGAR 数据；
- `experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt`；
- `experiments/sugar_reproduction/assets/official_tacsl/calibration/`。

脚本要求在已分配 GPU 的 Slurm shell 内运行，主动拒绝覆盖已有输出。不要退出或释放可用的保留分配；若要停止子任务，只终止记录的子进程组。

## 结果与代码入口

- 正式中文白底汇报（含 15 个可播放 H.264）：
  `experiments/isaaclab_g1_anatomical27_object_demos/report_isaaclab_native_tactile_20260813/IsaacLab原生触觉与跨场景验证_20260813.pptx`
- 当前实验索引：[experiments/README.md](experiments/README.md)
- 采集、渲染与字段说明：[scripts/sugar/native_tactile/README.md](scripts/sugar/native_tactile/README.md)
- 当前唯一执行计划：[PLAN/14_newton_isaaclab_universal_tactile/plan.md](PLAN/14_newton_isaaclab_universal_tactile/plan.md)
- 当前任务状态：[TODO/14_newton_isaaclab_universal_tactile/todo.md](TODO/14_newton_isaaclab_universal_tactile/todo.md)
- 官方 SUGAR 复现记录：[DOCS/sugar_carrybox_reproduction_full_record.md](DOCS/sugar_carrybox_reproduction_full_record.md)

`experiments/` 被 Git 忽略，实验 trace、视频、checkpoint 和 PPT 不进入提交。历史实验和旧文档统一归档在 `/public/home/yanhongru/Curiosity_archive/`，不在仓库内建立第二个归档根。
