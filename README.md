# Curiosity

当前最高优先级是在 **IsaacLab/PhysX** 中验证在线整手触觉能否帮助完整 SUGAR G1 应对持物过程中的突然质量变化。每只手固定安装 27 个官方 TacSL/R15 patch：掌心 `4 x 3`，拇指、食指、中指、无名指和小指各 3 段。policy 以 patch 为单元，不以 taxel 为单元；Newton 不参与新实验。

## 当前活动实验

在 G1 已抬起 `0.3023375869 kg` CarryBox 后，保持几何和视觉不变，将质量在线改为 `1.5x/3x/6x/10x`。正式比较三个同架构分支：proprio-only exact-zero tactile、online patch tactile、online patch tactile 加 causal slip。

官方 Refiner `890-D` observation 中的 `obj_lin_vel_b` 不能进入部署 actor；正式 actor 使用不含 measured object state 的 `504-D` Tracker-command/proprioception contract。`joint_pos/joint_vel` 仍可能泄漏负载，因此训练前必须先完成 time-resolved leakage audit，最终只按证据声称“触觉独有”或“在本体感受上的增量帮助”。

- 当前计划：[PLAN/15_online_patch_tactile_mass_adaptation/plan.md](PLAN/15_online_patch_tactile_mass_adaptation/plan.md)
- 当前 TODO：[TODO/15_online_patch_tactile_mass_adaptation/todo.md](TODO/15_online_patch_tactile_mass_adaptation/todo.md)
- 当前状态：真实 Plan-15 runtime 已恢复。`3 seeds x 5 mass factors` 的 15 条
  full-G1/54-patch 在线轨迹已完成；所有 paired action/event 完全一致，质量读回、
  jump 前双手接触和逐帧 TacSL 时钟均通过。质量变化当帧 contact binary 完全不变，
  patch load/pressure 与 `504-D` proprio 都发生变化，因此正式问题是“触觉在本体
  感受之上是否带来增量帮助”，不是“只有触觉知道质量”。受控官方 R15 滑动轨迹
  已将 callable 的 STICK/INCIPIENT/GROSS 区分校准通过，同一 callable 也已完成
  420 帧 full-G1 CarryBox 在线复核；正式训练尚未开始，下一步是串行 Z/P/PS
  one-update preflight。

## 当前结果

- 普通平面 CarryBox：完整 G1 将自由刚体抬升 `0.548 m`，`76/80` 帧双手有原生触觉。由于 G1 固定手型的指尖比掌心突出约 `2–3 cm`，这个物体主要由指端受力，不能称为整掌承载。
- 整掌贴合刚体：`0.5 kg` 自由物体抬升 `0.577 m`，`80/80` 帧双手有触觉，`79/80` 帧双掌接触；峰值掌区覆盖为左 `9/12`、右 `12/12`。
- 物理失败：相同动作在主动松手后物体从峰值下落 `0.394 m`；质量改为 `2.0 kg` 后最终高度比初始低 `0.179 m`。
- PickBottle：官方 motion 12 完成双手抬升并持续接触；motion 17 接触后发生弹道释放。两者使用官方 510-D Tracker 输入和 29-D 动作，不重放 CarryBox 动作。
- 当前未完成：普通箱子的“左手托底、右手扶侧”持续受力样本；TacSL 数值与 PhysX 支撑力之间的绝对标定。

这些结果只能称为**高保真模拟触觉**，不是硬件 GelSight 标定，也不是 sim-to-real。

## Plan 15 最短复现路径

已完成的 paired live leakage sweep 位于
`experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/`。从 retained
GPU shell 复现时运行：

```bash
bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/online_patch_tactile_mass_adaptation/runtime/leakage_sweep_v1.process \
  --status experiments/online_patch_tactile_mass_adaptation/runtime/leakage_sweep_v1.status \
  --log experiments/online_patch_tactile_mass_adaptation/runtime/leakage_sweep_v1.log \
  --tag plan15-leakage-sweep -- \
  /public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/native_tactile/run_online_mass_leakage_sweep.py \
  --output-root experiments/online_patch_tactile_mass_adaptation/leakage_sweep_reproduction \
  --device cuda:0
```

该命令固定使用 leakage seeds `150814/150815/150816`。每个 seed 先在线生成
`1.0x` nominal action，再逐帧重放到 `1.5x/3x/6x/10x`；输出 exact Refiner
object-state、504-D proprio、54-patch tactile、causal slip 的时序泄漏结果，以及仅
从这些 live trace 拟合的 9-channel 公共尺度。simulator relative tangential
velocity 只在 collection 后评价 slip precision/recall/detection delay，绝不进入
detector 或 actor。若 jump 前没有连续 10 帧 bilateral
TacSL contact，命令直接判定该 rollout 不可用于训练。

公共归一化尺度固定在
`experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json`。
先用 `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py` 串行运行 Z、P、PS 的
one-update preflight；只有各分支 live report 通过后才运行对应 3000-update 正式
任务。BCPPO 在 update 1000 前不会给 actor task-reward PPO，所以旧的 512-update
草案不能用于该科学问题。具体 task 名、seed、观察合同与停止条件以当前 Plan/TODO
为准。

## 历史整手可视化最短复现路径

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
- 当前唯一执行计划：[PLAN/15_online_patch_tactile_mass_adaptation/plan.md](PLAN/15_online_patch_tactile_mass_adaptation/plan.md)
- 当前任务状态：[TODO/15_online_patch_tactile_mass_adaptation/todo.md](TODO/15_online_patch_tactile_mass_adaptation/todo.md)
- 官方 SUGAR 复现记录：[DOCS/sugar_carrybox_reproduction_full_record.md](DOCS/sugar_carrybox_reproduction_full_record.md)

`experiments/` 被 Git 忽略，实验 trace、视频、checkpoint 和 PPT 不进入提交。被否决或不再活动的本地工作可移入对应 `legacy/`；所有 `legacy/` 均由 Git 忽略。既有大体积历史包继续保存在 `/public/home/yanhongru/Curiosity_archive/`。
