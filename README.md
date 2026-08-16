# Curiosity

当前最高优先级是在 **IsaacLab/PhysX** 中验证在线整手触觉能否帮助完整 SUGAR G1 应对持物过程中的突然质量变化。每只手固定安装 27 个官方 TacSL/R15 patch：掌心 `4 x 3`，拇指、食指、中指、无名指和小指各 3 段。policy 以 patch 为单元，不以 taxel 为单元；Newton 不参与新实验。

## 当前活动实验

在 G1 已抬起 `0.3023375869 kg` CarryBox 后，保持几何和视觉不变，将质量在线改为 `1.5x/3x/6x/10x`。正式比较三个同架构分支：proprio-only exact-zero tactile、online patch tactile、online patch tactile 加 causal slip。

正式训练质量点是离散的 `1x/1.5x/3x/6x/10x`，所以当前五质量冻结测试全部是
训练分布内测试，不得称为 OOD。

官方 Refiner `890-D` observation 中的 `obj_lin_vel_b` 不能进入部署 actor；正式 actor 使用不含 measured object state 的 `504-D` Tracker-command/proprioception contract。`joint_pos/joint_vel` 仍可能泄漏负载，因此训练前必须先完成 time-resolved leakage audit，最终只按证据声称“触觉独有”或“在本体感受上的增量帮助”。

- 当前计划：[PLAN/15_online_patch_tactile_mass_adaptation/plan.md](PLAN/15_online_patch_tactile_mass_adaptation/plan.md)
- 当前 TODO：[TODO/15_online_patch_tactile_mass_adaptation/todo.md](TODO/15_online_patch_tactile_mass_adaptation/todo.md)
- 当前状态（2026-08-16 06:07）：anchored Z seeds `151014/151015/151016` 均已
  严格停在 3000 updates 的 `model_2999.pt`，没有任何更晚 checkpoint。第三 seed
  `151016` 在 retained job `239098` 被调度器外部取消后，从最后完整
  `model_2750.pt` 恢复到 iteration 2751，并在 `240173/server07` 正常完成；终点
  含 59 个有限模型张量和 58 项 optimizer state。其正式 camera-free frozen audit
  也已完成 100 条：`1.0x/1.5x/3x/6x/10x` hold 为
  `20/20,20/20,20/20,0/20,0/20`，drop 为
  `0/20,0/20,0/20,20/20,20/20`，robot fall 全为 0。五个 trace 均为
  `450 x 20`，全部有限；100/100 mass event、质量读回和 jump 前十帧 bilateral
  patch contact 均通过。

  三个 checkpoint 与 disjoint evaluation seed 一一配对后的 eligible 分母均为
  `59`：合并 hold=`59/59,59/59,52/59,1/59,0/59`，drop=
  `0/59,0/59,2/59,58/59,59/59`。因此 Z baseline 已完整形成 mild/boundary/heavy
  区间；这不是触觉收益，但行为有效且不含糊，所以当前不做 Z overfit。

  正式 `P/seed151014` 已严格完成 3000 updates 并停在有限的 `model_2999.pt`，没有
  更晚 checkpoint。训练跨越三段保留 allocation：`240170/server44` 到
  `model_1500.pt`、`231256/server64` 到 `model_2250.pt`，两次均被调度器外部
  `CANCELLED by 0`；最终在 `240922/server07` 从 iteration 2251 精确恢复并正常退出。
  配对 `151014->152014` 的正式 camera-free frozen evaluation 已完成五质量各20条；
  每项有19条通过 handoff 的 eligible profile。P 的 hold 为
  `19/19,19/19,17/19,0/19,0/19`，drop 为
  `0/19,0/19,2/19,19/19,19/19`。同一 profile 的 Z seed `151014` hold 为
  `19/19,19/19,16/19,1/19,0/19`，drop 为
  `0/19,0/19,2/19,18/19,19/19`。因此首个 P seed 在3x只有轻微迹象，在6x没有收益，
  不能单独支持“触觉帮助训练”。`P/seed151015` 经 `model_250.pt` 与
  `model_1750.pt` 两次精确恢复后，已在 `241298/server59` 正常完成全部3000 updates；
  有限的 `model_2999.pt` 含 59 个模型张量、42 个 patch-encoder 张量和 58 项
  optimizer state，没有更晚 checkpoint。配对 `151015->152015` 的五质量各20条
  frozen evaluation 也已完成：P hold=`20,20,18,0,0`、drop=`0,0,0,20,20`、
  robot fall=`0,0,0,0,0`；Z hold=`20,20,16,0,0`、drop=`0,0,0,20,20`、
  robot fall=`0,0,0,4,3`。3x 成对 discordance 为 P-only `4`、Z-only `2`；这是
  第二个 seed 的温和迹象，不是收益证明。两个 P seeds 合并后的39个 eligible
  profiles 在3x为 P `35/39` hold、Z `32/39`，仍需第三个 seed。

  三-seed reaction-window 复算覆盖 119 条 drop：continuous
  patch 变化 `119/119` 早于 drop，中位 lead 21 帧；normal load 和 pressure 也均为
  `119/119`、中位 lead 20 帧；binary 中位 lead 15 帧，slip 为 `118/119`、中位
  lead 11 帧。133 条至少下沉 2 cm 的轨迹中，continuous 为 `133/133` 提前、
  中位 lead 7 帧，binary 为 `81/133`、中位 lead 3 帧。Z action 在有可检测 onset
  的 117 条 drop 中有 111 条早于 drop，继续证明正式问题是相对 proprioception 的
  增量收益，而非触觉独占质量信息。结果位于
  `experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/`
  `z_anchor025_formal_seed151014/`、`151015/`、`151016/`，reaction audit 位于
  `frozen_reaction_window_v2/summary.json`。当前 retained job `241811/server28`
  正在训练 `P/seed151016`；有限的 `model_500.pt` 已含 59 个模型张量、42 个
  patch-encoder 张量和 58 项 optimizer state，学习率为 `1e-5`，该 seed 已进入
  critic warmup，固定终点仍为 `model_2999.pt`。PS 尚未启动。`241298/server59`
  在完成训练后还生成了两条450帧、单 G1、同钟双手27-patch H.264：`1.5x` physical hold 与
  `3x` drop，均为 frozen P policy 的真实 camera-enabled rollout。活动文件位于
  `p_anchor025_formal_seed151014/videos/` 的
  `train151014_eval152014_1p5x_singleenv_profile0_camera_v1/*final_v2.mp4` 和
  `train151014_eval152014_3p0x_singleenv_profile2_camera_v1/*final.mp4`。
  seed `151016` 的两条 450-frame H.264 人眼证据也已完成并全帧解码：3x profile 0
  持箱、6x profile 0 下落 `0.562 m`；两条都在同一时钟显示完整 G1/CarryBox 和左右
  各 27 patch。路径分别为 `z_anchor025_formal_seed151016/videos/` 下
  `train151016_eval152016_3p0x_profile0_hold_camera_v1/*final.mp4` 与
  `train151016_eval152016_6p0x_profile0_drop_camera_v1/*final.mp4`。它们是各自的
  camera-enabled rollout，不冒充正式 camera-free trace 的逐帧 replay。

- 先前两-seed审查明细（已由上面的三-seed正式结果取代）：anchored Z seed
  `151014/151015` 均已严格停在
  3000 updates（`model_2999.pt`），没有继续加训；P/PS 与第三个 Z seed 均未自动
  启动。seed `151015` 的五质量、每项 20 profiles 正式冻结审查已完成，共 100 条
  live PhysX rollout。`1.0x/1.5x/3x/6x/10x` 的物理 hold 为
  `20/20,20/20,16/20,0/20,0/20`，drop 为
  `0/20,0/20,0/20,20/20,20/20`，robot fall 为
  `0/20,0/20,0/20,4/20,3/20`。3x 的另外 4 条下沉约 `5.4--7.8 cm`，但尚未达到
  drop；因此它是边界条件。原始 SUGAR reference termination 继续只作为 label，
  不能代替物理结果。五个 trace 均为 `450 x 20`，全部有限；100 条均正确读回质量，
  并在 jump 前保持 10 帧双手 patch contact。每个 profile 的 handoff 后随机延迟在
  五个质量条件中逐项一致；live PhysX 的绝对 handoff/event 时刻最多相差 4 帧。
  该端点已经形成温和成功、边界下沉和重质量掉落三个可解释区域，当前无需 overfit；
  训练继续冻结。若后续人眼证据否定行为，再先做单条件 serious overfit。这是一个 Z
  checkpoint 的正式端点审查，仍不能证明触觉收益。正式 summary 与 trace 位于
  `experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/`
  `z_anchor025_formal_seed151015/`。
  450-frame 同钟 H.264 人眼审查也已完成：`1.5x` 持稳；`3x` profile 8
  保持双手接触但下沉约 `0.052 m`；`6x` profile 0 掉箱，camera replay 后续还出现
  robot destabilization；`10x` 掉箱。每条视频上方只保留目标 G1/CarryBox，下方同时
  显示左右各 27 patch，均已完整解码。证据位于
  `experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/`：
  `z_anchor025_endpoint_audit_seed151015/videos/` 下的 `1p5x` 与 `10p0x` final 视频，
  以及 `z_anchor025_formal_seed151015/videos/` 下的
  `3p0x_profile8_boundary_reproduce_v1/*single_g1*final_v2.mp4` 和
  `6p0x_profile0_drop_reproduce_v1/*single_g1*final.mp4`。3x camera replay 与正式
  profile 的失败类别和下沉量一致；6x replay 复现掉箱类别，但掉箱后的 robot-fall
  label 对渲染引入的微小闭环扰动敏感，因此不写成逐帧复现正式无相机轨迹。
  当时 retained `239098/server44` 在正式审查完成后继续保留用于数值复核和渲染；没有
  释放 allocation，也没有自动启动任何训练。

  seed `151014` 的匹配正式 frozen audit 也已完成，共 `5 x 20 = 100` 条。每个质量
  条件都重复出现同一个 profile 1 的 handoff 前 Refiner 失败；它不进入 student 的
  post-jump 分母。其余每项 19 条的 `1.0x/1.5x/3x/6x/10x` hold 为
  `19/19,19/19,16/19,1/19,0/19`，drop 为
  `0/19,0/19,2/19,18/19,19/19`。两个已完成 checkpoint/seed pair 合并后的 eligible
  hold 为 `39/39,39/39,32/39,1/39,0/39`，drop 为
  `0/39,0/39,2/39,38/39,39/39`。这进一步确认 Z 已有可用的温和、边界和重质量测试
  区域；不需要继续训练或为了全成功而 overfit。

  临界轨迹对 world-camera 有闭环敏感性：seed `151014` 的 `6x` profile 7 在无相机
  正式轨迹及独立无相机复测中都持稳，但开启相机后下落 `0.279 m`；`3x` profile 0
  则从无相机正式轨迹的 drop+robot-fall 变为相机轨迹中的 hold。新的 3x 无相机
  四-profile 复测已正常退出；其 23 个 trace 字段与正式结果的前四条全部逐值相同，
  profile 0 精确复现 `297/337` handoff/jump、`0.452212 m` 下落、drop 和 robot fall。
  正式计数因此只采用 camera-free trace；视频仅代表其自身带相机 rollout，不再写成
  正式轨迹的逐帧复现。
  对应同钟视频为 `z_anchor025_formal_seed151014/videos/` 下的
  `train151014_eval152014_6p0x_profile7_rare_hold_reproduce_v1/`
  `seed151014_6p0x_profile7_camera_replay_drop_single_g1_bilateral_27patch_final.mp4`
  与 `train151014_eval152014_3p0x_profile0_drop_reproduce_v1/`
  `seed151014_3p0x_profile0_camera_replay_hold_single_g1_bilateral_27patch_final.mp4`。

  两个 Z endpoint 的配对 frozen reaction-window audit 已完成。每条重质量轨迹都与
  同 seed、同 profile 的 1x 轨迹按 jump 对齐，并用 jump 前十帧差异标定变化阈值。
  连续 load/pressure/signed-shear/friction 与 contact binary 分开计算。79 条实际 drop
  中，连续 patch 变化 `79/79` 早于 drop，中位提前 20 帧（`0.40 s`），contact binary
  中位只提前 12 帧，slip-state `78/79` 提前、中位 10 帧（`0.20 s`）。normal load
  和 pressure 单独计算也都是 `79/79` 提前，中位 lead 均为 19 帧，因此不是只靠
  friction 或 contact-bit 变化。在 93 条至少
  下沉 2 cm 的轨迹中，连续 patch `93/93` 提前变化、中位 lead 7 帧；binary 只有
  `47/93` 在下沉前变化，中位 lead 为 0。`6x` 的连续/binary/slip/drop 中位 offset 为
  `1/9/11/24.5` 帧；`10x` 为 `2/9/10/16` 帧。
  但 Z action 也有 `73/79` 在 drop 前已经分叉，说明 actor-visible proprio/闭环动力学
  同样提供了反应信号。结论只是“触觉存在可利用的提前窗口”，不是“触觉已经帮助
  policy”；当前只证明连续触觉比 binary contact 更早，正式比较仍必须证明它相对
  proprio-only Z 的增量收益。分析入口为
  `scripts/sugar/native_tactile/analyze_frozen_mass_reaction_window.py`，结果位于
  `experiments/online_patch_tactile_mass_adaptation/frozen_reaction_window_v1/summary.json`。
  对应的 450-frame H.264 为同目录下
  `videos/6x_camera_and_formal_reaction_window_v1.mp4`：左侧是完整 G1、CarryBox 和双手
  27-patch 的带相机 6x drop，右侧动态显示 38 条正式无相机 6x drop 的 continuous/
  binary/slip/sag/drop 分布。视频内部明确标注两侧不是同一 rollout，正式数字只来自
  camera-free trace。

- 此前执行记录：anchored Z seed `151014` 已严格停在 3000 次 iteration
  的 `model_2999.pt`，没有继续加训；checkpoint 的 59 个模型张量和 58 项 optimizer
  state 均有限。正式 frozen horizon 已从不足的 420 修正到 450，因为 frame-297
  handoff、最多 50 帧 delay 和 80 帧结果窗口至少要求覆盖到 frame 427。四条
  `1.5x` profiles 现在有 2 条完整通过 80 帧；另 1 条 jump 后 39 帧触发 `obj_pos`，
  1 条 handoff 前触发 `ee_body_pos`。两条成功 profile 后续才在 frame 423/410 触发
  `obj_ori`。BC anchor 将 handoff 前十帧
  student/teacher action L2 从撤回终点的约 `5.5` 降到约 `1.0--1.1`。同一 checkpoint
  的 eligible profile 3 已生成 420-frame 同钟 H.264：frame 308 质量从约 `0.302`
  变为 `0.454 kg`，世界画面与左右 27-patch load/pressure/signed-shear/slip 同屏。
  匹配 camera profile 0 也没有物理掉箱：jump 后持箱 59 帧，在 lift 约 `+0.823 m`
  时因 reference-tracking 偏差终止，而不是 drop。仅关闭物体 reference termination
  的诊断仍得到 2 条通过；profile 0 随后因 `anchor_pos` 终止但仍双手持箱。这说明 handoff forgetting
  已修复，但跨 profile 轨迹稳定性仍不足，也仍不能证明触觉收益。下一正式 seed 不会自动
  启动；先完成人眼正/负行为审查，如不对则先做单一 `1.5x` 条件的 serious overfit
  诊断。P/PS formal 仍未开始。

  同一 anchored Z endpoint 的 450-frame 四-profile 跨质量审查也已完成：`1.0x/
  1.5x/3x/6x/10x` 的 eligible hold 为 `1/4, 2/4, 2/4, 0/4, 0/4`；`6x` 已出现
  `0.194 m` drop，`10x` 出现 `0.165/0.216 m` drops。说明 Z 在温和条件可持箱、
  重质量条件有明确失败空间，当前不需要 overfit Z 来制造全成功。该四-profile
  结果只用于准入和选择困难度，不能当作触觉收益或单调质量曲线。

  历史与支撑状态：真实 Plan-15 runtime 已恢复。`3 seeds x 5 mass factors` 的 15 条
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
  未保存迭代不计。seed `151014` 已在保留 job `239105`/`server35` 从 iteration
  1501 正确恢复，checkpoint iteration、BCPPO update step 和 optimizer learning
  rate 均已同步；seed `151015` 也已在 job `239106`/`server44` 从 iteration 2251
  正确恢复并通过相同同步，随后正常完成 `model_2999.pt`；该 endpoint 的 59 个模型
  张量与 58 项 optimizer state 均有限。同一 retained job 已从官方 Tracker、
  iteration 0 启动 seed `151016`；seed `151014` 已越过有限的 `model_2000.pt` 并
  继续 steady full-PPO。后续冻结 gate 已证明 zero-floor 的最后 1000 updates 会
  遗忘 Refiner 持箱行为：seed `151014` 的 distillation loss 从 update 2000 的
  `0.3404` 升到 endpoint 的 `35.8202`，交接动作差从约 `0.9` 恶化到 `5.4--5.9`。
  endpoint 在 mass event 前失败；相同 update-2000 checkpoint 则完成三次真实
  `1.5x` jump，并维持 `65/38/74` 个 post-jump frames，但仍不足固定 80 帧。
  因此 Z/P/PS 现统一使用仓库已有 `stage3_distill_weight_floor=0.25`，保持 full PPO
  authority 和 3000-update endpoint；三个 Z 从 update 2000 重跑最后 1000 updates。
  该阶段曾申请五天恢复资源；当时只有 `239098`/`server44` 在保留，P/PS formal
  仍未启动。
  Plan-15 training launcher 已与 frozen evaluator 统一绑定本地 ground-plane USD 和
  预转换 G1 USD；同一环境合同在 task import 前关闭远端 debug marker，并固定相同
  TacSL/PhysX 参数，不再依赖当前不可用的 ground/marker 远端 assets。
  Anchored Z 当时保存了151014/update2250与151016/update1000；两份 checkpoint 均
  有限。jobs239105/239106随后均被调度器外部取消，未保存更新不计。该调度快照已被
  当前 `151014/151015` 的完整 anchored endpoints 取代；P/PS 仍禁止自动启动。
  Frozen evaluator 已修正为
  motion 45/frame 0 物理状态与 reference command buffer 同步起步；update-1000
  中间策略仍在接触箱子前的 frame 63 终止，所以不能提前作为质量适应结果。双手
  27-patch 可视化布局和 H.264 编码已验证。Frozen evaluator 现有一个独立的
  one-profile endpoint-video 入口，可在同一 rollout 直接记录 world camera、
  Refiner/policy handoff、质量事件和 54-patch/slip trace；该入口仍须等 replacement
  Z endpoint 后运行验证，尚未把布局测试冒充正式同钟行为视频。

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
scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/online_patch_tactile_mass_adaptation/runtime/z_frozen_sweep.process \
  --status experiments/online_patch_tactile_mass_adaptation/runtime/z_frozen_sweep.status \
  --log experiments/online_patch_tactile_mass_adaptation/runtime/z_frozen_sweep.log \
  --tag plan15-z-frozen-sweep --foreground -- \
  scripts/sugar/native_tactile/run_plan15_frozen_sweep.sh Z \
    experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/z_seed151014/model_2999.pt \
    experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/z_seed151015/model_2999.pt \
    experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/z_seed151016/model_2999.pt \
    experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_reproduction
```

该入口固定 checkpoint/evaluation-seed 一一配对并运行 5 个质量条件、每项 20
profiles。结束后用 `SUGAR/scripts/sugar_rl/summarize_online_patch_mass_sweep.py`
汇总各质量条件；P、PS 只替换 branch 和对应 checkpoint/output 路径。

当前三个已完成 Z endpoint 的反应窗口可直接离线复算，不启动仿真或训练：

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/native_tactile/analyze_frozen_mass_reaction_window.py \
  --seed-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151014 \
  --seed-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151015 \
  --seed-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_formal_seed151016 \
  --scale-file experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json \
  --output experiments/online_patch_tactile_mass_adaptation/frozen_reaction_window_v2/summary.json
```

从统计结果中选定一个 profile 后，用同一个 frozen evaluator 重跑它所在的单个
4-profile batch，并通过 `--record-profile-index` 只录目标 profile；这样视频与数值
gate 使用相同的 batch 随机条件。当前 anchored-Z 正例 profile 3 的最短复现为：

```bash
scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/online_patch_tactile_mass_adaptation/runtime/z_anchor025_profile3_reproduce.process \
  --status experiments/online_patch_tactile_mass_adaptation/runtime/z_anchor025_profile3_reproduce.status \
  --log experiments/online_patch_tactile_mass_adaptation/runtime/z_anchor025_profile3_reproduce.log \
  --tag plan15-z-anchor025-profile3 --foreground -- \
  /public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
    SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py \
    --branch Z \
    --checkpoint experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/z_seed151014/model_2999.pt \
    --patch-scale-file experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json \
    --output-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_endpoint_video/profile3_reproduce \
    --training-seed 151014 --seed 152014 --mass-factor 1.5 \
    --motion-id 45 --profiles 4 --num-envs 4 --max-steps 450 \
    --post-jump-window 80 --physical-outcome-view \
    --record-world --record-profile-index 3 \
    --headless --device cuda:0

/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/native_tactile/render_online_patch_mass_jump.py \
  --run-root experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_endpoint_video/profile3_reproduce \
  --scale-file experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json \
  --output experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/z_anchor025_endpoint_video/profile3_reproduce/world_bilateral_27patch.mp4 \
  --profile-index 3 \
  --world-crop-left 320 \
  --title "Plan 15 anchored Z: Refiner handoff and 1.5x mass"
```

视频中的 mass/jump 文本仅为 evaluator overlay，明确不进入 actor；统计 sweep 不
启用相机，避免改变正式 300-rollout 设计。

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
