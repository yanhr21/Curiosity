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
  420 帧 full-G1 CarryBox 在线复核。Z 已完成一次 360-step BCPPO update：`364`
  次 exact-zero observation 且 `0` 次 TacSL read；P 已完成 `361` 次 online feature
  update 和 `19,494 = 361 x 54` 次官方 patch read，slip call 为 0；PS 同样完成
  `19,494` 次 patch read，并执行 `361` 次 causal slip callable。三个 one-update
  training-path preflight 均通过。正式 Z 的 seed `151014/151015` 已使用完全相同的
  4-env、24-step、3000-update 配置运行，P/PS 仍未启动；两者均已越过 update
  500，各自 checkpoint 可完整读取并已从纯 distillation 进入 critic warmup。
  matched frozen-Refiner feasibility 中，`1.5x` jump 后仍持续双手接触并继续抬升，
  `3x/6x/10x` 均失持掉落；这提供了可恢复温和条件和极端失败条件，但仍不代表
  触觉已经改善行为。固定的双肩夹紧+降姿响应也未恢复 `6x/10x`。seed `151014`
  的原 allocation 被调度器终止后，已从最后完整 update-500 checkpoint 精确恢复到
  BCPPO iteration 501；到达 update 750 后已迁移到五天 `238250`/`server23` 并从
  iteration 751 接续，总 endpoint 仍是 3000。seed `151015` 的 allocation 也在
  update 784 后被调度器终止，最后完整 checkpoint 为 750；现已在
  `238355`/`server07` 精确恢复到 BCPPO/runner iteration 751，总 endpoint 仍是
  3000。两个 resumed Z seed 均已生成可完整读取的 update-1000 checkpoint，并进入
  task-reward PPO authority ramp；两者均已到 update 2000 并进入 steady full-PPO。
  seed `151015` 现已正常完成全部 3000 次训练循环，实际零基终点文件为
  `model_2999.pt`；seed `151014` 也已正常完成同一终点，两个 checkpoint 均含
  59 个模型张量与 58 项 optimizer state。两个终点的 `1.5x` frozen check 共
  `8/8` profiles 在箱体接触前 fall，contact 与 mass event 均为零；seed `151014`
  的 `1.0x` 四条又与其 `1.5x` 完全同帧终止。这证明旧路径失败在 frame-zero 抓取
  入口，而不是突然变重。第三 seed `151016` 已在 iteration 226 仅停止记录的 child
  PGID，server07 allocation 保留。

  当前实现目标已改为：official frozen Refiner 从 motion 45/frame 0 在线控制到
  连续 10 帧 lift `>=0.05 m`，随后在同一 PhysX episode 无 teleport、无 replay 地
  交给 Z/P/PS actor，再延迟 `10--50` 帧增重。交接前数据不计 PPO surrogate/value/
  entropy credit，handoff mask 与 teacher/object state 不进入 actor；P/PS 的四帧
  patch/slip history 必须在 teacher 前缀中在线形成。该 replacement 实现及 Z/P/PS
  三个 one-update preflight 均已通过：每项完成 4 次 live handoff、2 次真实 mass
  change，并只给 142/143 个 post-handoff transitions PPO credit；Z 保持 0 次 TacSL
  read，P/PS 各完成 19,494 次官方 patch read，PS 实际执行 361 次 causal slip
  update。它们只准入新的 formal Z，不是触觉收益结论，也没有启动 P/PS formal。
  replacement handoff-Z 正式训练随后开始：seed `151014` 的完整恢复点为
  `model_1500.pt`，seed `151015` 为 `model_2250.pt`；两份均含 59 个模型张量、58
  项 optimizer state 且数值有限。对应 allocations `238253/238620` 分别在打印
  iteration 1711/2339 后被调度器外部 `CANCELLED by 0`，没有训练 Traceback/OOM；
  未保存迭代不计。五天恢复 jobs `238934/239098` 和 8 小时 backfill jobs
  `239105/239106` 已排队，总 endpoint 仍为 3000。P/PS formal 仍未启动。
  Frozen evaluator 已修正为
  motion 45/frame 0 物理状态与 reference command buffer 同步起步；update-1000
  中间策略仍在接触箱子前的 frame 63 终止，所以不能提前作为质量适应结果。双手
  27-patch 可视化布局和 H.264 编码已验证，但当前 H200 Kit/Vulkan
  camera start 仍会在场景构建前 `ERROR_DEVICE_LOST`，所以尚未把离线布局测试冒充
  真实同钟 world+tactile 视频；无相机在线训练正常继续。

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

新的 handoff 分支有三个 update-3000 checkpoint 后，在 retained validation GPU shell
中执行完整的 300-profile frozen sweep：

```bash
scripts/sugar/native_tactile/run_plan15_frozen_sweep.sh Z \
  experiments/online_patch_tactile_mass_adaptation/training_handoff/z_seed151014/model_2999.pt \
  experiments/online_patch_tactile_mass_adaptation/training_handoff/z_seed151015/model_2999.pt \
  experiments/online_patch_tactile_mass_adaptation/training_handoff/z_seed151016/model_2999.pt \
  experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z
```

该入口固定 checkpoint/evaluation-seed 一一配对并运行 5 个质量条件、每项 20
profiles。结束后用 `SUGAR/scripts/sugar_rl/summarize_online_patch_mass_sweep.py`
汇总各质量条件；P、PS 只替换 branch 和对应 checkpoint/output 路径。

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
