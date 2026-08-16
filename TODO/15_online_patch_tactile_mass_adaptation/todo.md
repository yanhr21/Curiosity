# TODO 15: Online Whole-Hand Patch Tactile for Sudden-Mass Adaptation

## A. 文档与冻结合同

- [x] 将 Plan/TODO 15 设为唯一活动队列，并把 Plan/TODO 14 移入各自 `legacy/`。
- [x] 在 `AGENTS.md` 顶部写明 online、54-patch、causal slip、leakage audit 和
  serious matched-training 规则。
- [x] 固定 nominal mass `0.3023375869 kg` 与 `1.5x/3x/6x/10x` sweep。
- [x] 固定 actor/teacher/critic 边界：`504-D` deployable actor 禁止 measured
  object state；官方 `890-D` Refiner 只能作为 training-only teacher/critic。
- [x] 固定 policy 的空间单元为每手 27 patches；taxel 只留在官方 sensor 内部
  和 sensor debug，不得作为 actor 单元。

## B. IsaacLab live mass jump

- [x] 在完整 G1 CarryBox env 中加入 runtime mass/inertia change event。
- [x] 代码路径保证 jump 不改变 object geometry、material、RGB、pose、reference、sensor
  history 或上一动作。
- [x] 实现每个 jump 后的实际 mass/inertia readback 和 event timestamp 评价字段。
- [x] 实现从 frame 0 连续运行、稳定抬升 10 帧后随机等待 10--50 帧再 jump；
  scheduler 不读取 TacSL，双手连续接触 10 帧由 live trace 独立检查。
- [x] 为 `1.0x` no-jump 条件加入 matched placebo event clock；不写 PhysX mass/
  inertia，并单独记录 `mass_changed=false`。
- [x] 在 official AppLauncher/H200 上完成完整 G1、54 个 physical patches、GPU
  PhysX 和全部 manager 初始化，并真实推进一个 control frame；54 个 official
  TacSL source clocks 在该帧同步为 `0.02 s`。
- [x] 420 帧 live preflight 确认 lift-gated mass/inertia write/readback：第 299 帧
  `0.3023376 -> 0.9070128 kg`，jump 前连续 10 帧双手接触，54-patch clock 零偏差
  且逐帧严格前进；箱子最高抬升 `0.7469 m` 后物理失持。
- [x] 完成 no-jump、`1.5x/3x/6x/10x` 的三 seed fixed-action 无学习 sweep；15 条
  live trace 均通过 mass readback、paired action/event、jump 前双手接触和 54-patch
  clock 检查。该 sweep 量化 nominal action 下的失败严重度，不把它误写成改变动作后
  仍不可恢复。
- [x] 用同一 frozen Refiner closed-loop controller 补齐同 clock 的 feasibility：
  `1.5x` jump 后 `118/121` 帧仍保持双手 contact 且继续抬升；`3x/6x/10x` 分别在
  frame `354/321/315` 后失持并下落。该结果证明温和条件有恢复窗口，并只把
  `3x+` 标为当前 controller 的失败区。
- [x] 用预先固定的 stronger-grip/lower-posture action 检查 `6x/10x`：在绝对
  frame 300 平滑叠加相同双肩内夹和双腿降姿目标，不读取 mass ID、jump、tactile
  或 object state。两者仍失持；`6x/10x` 双手接触结束于 frame `320/315`，物体
  下降 `0.171/0.213 m`。结论是该简单固定响应不足，不是任何策略都不可恢复。

## C. 54-patch online observation

- [x] 从官方 bilateral 27-patch `VisuoTactileSensor` 当前帧在线读取 raw tensor。
- [x] 在 GPU 上逐 patch 归并 `contact`、`normal_load_n`、`mean_pressure_pa`、
  `shear_x_n`、`shear_y_n` 和 `friction_utilization`。
- [x] 固定 `[B,4,2,27,9]` contract、anatomical order、单位和符号；公共归一化
  尺度须等 live sweep 后冻结。
- [x] 从 15 条 live mass-sweep trace 冻结统一 9-channel 公共尺度；正式启动器只读取
  `leakage_sweep_v1/patch_channel_scales.json`，不使用猜测常数。
- [x] live collector 逐帧保存 54 个 official TacSL source timestamps；15 条正式
  sweep 全部确认同帧同步和逐 control-frame 严格前进。
- [x] 保证 actor observation 中不存在 20x25 taxel 维度、普通 ContactSensor、
  `hands_contact_label` 或 object-state proxy。
- [x] 实现 exact-zero no-sensor-read observation，保证 zero encoder output 也为零。
- [x] 用 synchronized patch visualization 检查压力/剪切变化与世界接触同钟对应。
  seed `151015` endpoint 已生成并完整解码两段 450-frame H.264：`1.5x` 真实持箱与
  `10x` 真实掉箱。每段上方只显示目标完整 G1/CarryBox，下方同步显示左右各 27 个
  patch 的 load、pressure、signed XY shear 和 slip；质量 overlay 明确不进入 actor。

## D. IsaacLab patch slip callable

- [x] 实现 batch-stateful `PatchSlipDetector.update(...)` 与 reset mask。
- [x] 输入仅限当前/历史 patch contact、pressure、signed shear、friction utilization
  和 timestamp。
- [x] 输出每个 patch 的 `NO_CONTACT/STICK/INCIPIENT/GROSS`、`slip_score`、
  `incipient_slip` 和 `gross_slip`。
- [x] 将 callable 接入 IsaacLab observation term，并在 15 条 live GPU/PhysX
  CarryBox trace 的每一帧实际调用。
- [x] 实现独立 evaluation-only oracle：逐 patch 最大 active-taxel simulator
  tangential speed，并输出 precision、recall 和 onset delay；oracle 与 detector/
  actor/mass scheduler 隔离。
- [x] velocity oracle 只在 current-contact samples 上评价滑动；有载 contact-loss
  gross alert 单独报告，避免把接触结束后不存在的 taxel velocity 当 false positive。
- [x] 在 controlled stick-to-slide 与 CarryBox jump/slip 中评价 precision、recall、
  false positives 和 detection delay；relative velocity 仅作标签。
- [x] CarryBox 3x jump trace 的 contact-supported velocity-oracle 评价为 precision
  `1.000`、recall `0.9909`、median onset delay `0` 帧；该轨迹没有 incipient-oracle
  样本且多数接触已 gross sliding，不能替代 controlled stick-to-slide 校准。
- [x] 完整 15-trace 评价为 precision/recall `0.9992/0.9904`、median delay `0`，但
  14 个 oracle STICK samples 全被判为 GROSS；在 controlled calibration 修复状态
  饱和前禁止 PS 训练。
- [x] 用 240 帧 official R15 controlled trace 修正旧阈值：friction utilization
  只触发 INCIPIENT，GROSS 需要连续两个高 shear-rate/pressure-drop sample；静止、
  慢滑、快滑、回程分别得到 STICK/INCIPIENT/GROSS/INCIPIENT，state 正确数为
  `109/111`、`109/109`、`19/20`，incipient/gross onset delay 为 `0/1` 帧。
- [x] 更新后的 callable 已在独立 420 帧 full-G1 CarryBox `3x` live rollout 中逐帧
  执行；contact-supported precision/recall `1.0/0.9971`，median/p95 delay `0/1`
  帧，28 次 loaded contact loss 全部报警，54-patch official clock 仍严格在线。
- [x] 确认 callable 与 live actor path 没有 offline replay、future frame、mass/
  jump flag、object motion 或 simulator relative velocity 输入。

## E. 质量信息泄漏审计

- [x] 实现串行 paired sweep 入口：每个预定 seed 先采集 nominal `1.0x` 动作，
  再把同一动作逐帧重放到 `1.5x/3x/6x/10x`，完成后运行 leakage analyzer 与
  live scale fitter；15 条真实 IsaacLab trace 已全部完成。
- [x] 冻结 leakage seeds `150814/150815/150816`、training seeds
  `151014/151015/151016` 与 frozen-evaluation seeds
  `152014/152015/152016`。
- [x] 先记录 nominal controller action sequence，再开环重放同一 sequence 采集
  paired no-jump 与四倍率 jump；最大 action 误差与 event-frame 误差均为 `0`。
- [x] 分别导出 object-state、proprio-only、patch-tactile、patch-tactile+slip 信号组。
- [x] 在 jump 前 0.5 s/后 1.0 s 窗口报告原始变化、mass-factor linear-probe
  balanced accuracy 和 change-onset latency。
- [x] 明确证明 deployed actor 没有 `obj_lin_vel_b`、mass、jump flag、RGB 或
  simulator contact velocity。
- [x] 根据审计结果固定结论边界为“在 proprioception 上的增量帮助”；proprio 在
  event 当帧已有非零变化，禁止再写“触觉独有感知”。
- [x] 三 seed 所有质量 event 当帧的 contact binary 相对 nominal 完全相同，而
  patch load/pressure 已改变；信号可进入 P preflight，但不把非单调响应称为质量计。

## F. Serious matched training implementation

- [x] 实现 shared anatomical patch-token encoder：9->128 projection、hand/patch/
  time embedding、3-layer 4-head Transformer、128-D pooled output。
- [x] 接入已有 SUGAR `512/256/128` actor、29-D action 和官方 510-D Tracker warm
  start；H200 structural preflight 已确认 zero-patch action error `1.31e-6`、encoder
  gradient 非零。
- [x] 将新 policy class 注册到 frozen Refiner teacher 和 repository-native BCPPO
  runner；启动器在 scale JSON 缺失时拒绝训练。
- [x] 定义三个共享 policy/runner 配置的 process-local 分支：`Z`、`P`、`PS`，
  并分别提供 one-update preflight 与 3000-update formal task。
- [x] 在配置层保持 critic、teacher、optimizer、reward、physics、mass sampling
  和 3000-update budget 一致；live runner 已在 H200/IsaacLab 中完成 Z/P/PS
  one-update preflight，并已启动正式 Z。
- [x] launcher 自动绑定 official Refiner teacher 和 official Tracker warm start，
  formal run 只接受冻结 seeds `151014/151015/151016`；resume 按总预算计算剩余
  updates，不重复或延长到 3000 之外。
- [x] mass-factor/delay assignment 继承对应 env seed，且每个 env 的连续五个
  episodes 各覆盖一次 `1.0x/1.5x/3x/6x/10x`。
- [x] Plan-15 runner 禁用通用的 random episode-length initialization，确保
  one-update preflight 和 formal episodes 都从 motion frame 0 连续进入抓取。
- [x] one-update preflight 自动记录并检查 Z 的零 TacSL read、P/PS 的在线
  54-patch read 及 PS callable execution；它同时如实记录双手接触、非零负载和
  mass event，但不在尚未进入抓箱窗口的早期 rollout 中伪造这些信号。非零双手
  tactile 和质量/惯量 event 已由连续动作 full-G1 collector 独立准入。
- [x] 保持三个分支的官方 SUGAR CarryBox reward 完全一致；mass ID、jump flag、
  patch/slip state 均不直接成为 reward 或 actor 答案。
- [x] 冻结 3 个 formal training seeds，以及 3 个未参与训练的 evaluation seeds；
  固定一一配对 `151014->152014`、`151015->152015`、`151016->152016`，每对每个
  factor 跑 20 profiles，共 300 matched rollouts/arm，而不是 900。

## G. 在线持箱交接与串行训练

- [x] 冻结旧 frame-zero Z 为结构性负结果，不再续跑。seed `151014/151015` 均完成
  3000 次训练，实际零基终点为 `model_2999.pt`；其 `1.5x` frozen check 共 `8/8`
  profiles 在接触箱体前 fall，bilateral contact 与 jump 都为零。`151014` 的
  `1.0x` 四条又与其 `1.5x` 四条完全同帧终止。seed `151016` 因此在 iteration 226
  仅终止记录的 child PGID，`238355` allocation 保留。旧 checkpoint 不进入触觉
  收益比较。
- [x] 实现 live official-Refiner handoff：每次 reset 后由 frozen Refiner 从 motion
  45/frame 0 在线控制；连续 10 帧 lift `>=0.05 m` 后，在同一 PhysX episode、无
  teleport/无 replay/不清空 tactile history 的下一动作边界交给 actor。
- [x] 加入仅供训练算法使用的 handoff mask：交接前 transition 不进入 PPO
  surrogate/value/entropy；mask、teacher observation、mass/jump 均不进入 actor。
- [x] 让 mass scheduler 只在 handoff 后开始 `10--50` 帧 delay，并确认 P/PS 在
  teacher prefix 中继续实时形成四帧 patch/slip history；Z 保持零 TacSL read。
- [x] 重新运行 handoff-Z one-update preflight：必须真实持箱、交接、student 控制并
  触发至少一个 mass event；不满足则不开始 formal Z。
- [x] 重新运行 handoff-P one-update preflight：除上述条件外必须出现双手在线 patch
  contact/load，不能读取离线 trace。
- [x] 重新运行 handoff-PS one-update preflight：除 P 条件外必须实际调用 causal
  `PatchSlipDetector.update(...)`。
- [x] 三个 replacement preflight 均在 retained `238253`/`server59` 通过。每项均为
  1440 transitions、4 次 live handoff 和 2 次真实 mass change；Z 的 TacSL read
  为 0。P/PS 各有 `361` 次在线 feature update、`19,494 = 361 x 54` 次官方 patch
  read 和 363 个 bilateral-contact env-samples；P 的 slip call 为 0，PS 的 causal
  slip call 为 361。handoff mask 与 wrapper 逐步计数完全一致，只把 142/143 个
  post-handoff transitions 计入 PPO，并屏蔽 1298/1297 个 teacher transitions。
- [x] 新 `Z`：三个冻结 seed 各完成匹配 3000 updates 与 endpoint frozen evaluation；
  只有存在 eligible post-jump profiles 才允许进入 P。
- [x] anchored Z seed `151014` 严格完成到固定 `model_2999.pt` 后停止；checkpoint 为
  59 个有限模型张量与 58 项 optimizer state，未运行 3000 之后的 update。
- [x] 完成该 endpoint 的首轮 420-frame 四-profile `1.5x` 诊断：表面为 `1/4`
  eligible，另有 76-frame horizon 截断、39-frame post-jump termination 和 1 条
  pre-handoff Refiner fall；交接动作差已从 zero-floor 的约 `5.5` 降到约
  `1.0--1.1`。该 `1/4` 后续因 horizon 不足而撤回。
- [x] 修正 frozen horizon：handoff 约 frame 297、最大 delay 50、结果窗口 80，
  因此 420 帧不足，正式 sweep 统一改为 450 帧。精确 termination audit 得到 `2/4`
  eligible：profile 0=`obj_pos@376`，profile 1=`ee_body_pos@220`，profile 2/3 完成
  80 帧后分别为 `obj_ori@423/410`。撤回旧 `1/4` 作为正式 endpoint 计数。
- [x] 完成 no-learning physical-continuation 诊断：仅关闭 `obj_pos/obj_ori` 后仍为
  `2/4` eligible；profile 0 改在 `anchor_pos@383` 终止但仍双手持箱，证明主要缺陷
  是参考/整机漂移而非物理掉箱。该诊断不计入正式分支比较。
- [x] 完成 anchored seed-151014 的 450-frame 四-profile 跨质量小审查：`1.0x/
  1.5x/3x/6x/10x` eligible hold=`1/4,2/4,2/4,0/4,0/4`；`6x` 有 1 条物理 drop，
  `10x` 有 2 条物理 drop。该结果确认 mild-pass/heavy-failure 动态区间，不触发
  overfit，也不替代三 seed、20 profiles/factor 的正式比较。
- [x] 检查跨 factor jump 前匹配：event frames 全部一致；`1.5x/3x/6x` action 和
  object position exact；`10x` profile 2 仅最后两帧出现 `0.0146` action、`0.23 mm`
  position 闭环非确定性，且 event 前质量仍为 nominal。
- [x] 录制并完整解码 eligible profile 3 的 420-frame 同钟 H.264：世界画面、
  Refiner/policy 交接、frame-308 `0.302->0.454 kg` 质量变化和左右各 27 patch 均在
  同一视频中。
- [x] 录制匹配 profile 0 并区分物理掉落与 reference termination：它在 frame 337
  增重后继续持箱到 frame 396，终止前 lift 约 `+0.823 m`，没有 drop/robot fall；
  59-frame 后终止与 `0.208 m/0.806 rad` 最大 reference 误差相伴。
- [x] 完成 seed `151014` endpoint 的正/负视频审查和失败原因判定；它只准入一个额外
  Z endpoint，不准入 P/PS。
- [ ] 若 endpoint 行为仍无效或含糊，先做一个固定 `1.5x` profile 的 serious overfit
  诊断；必须保留完整 SUGAR actor、live Refiner handoff 和在线物理，不得用 toy
  model，且不得把 overfit 计入正式 Z/P/PS 结果。
- [x] 定位 zero-floor endpoint 的 handoff 退化：seed `151014` update-2999 gate 为
  `0/4` eligible，三条 handoff 后约 7 帧且 mass event 前失败；同 profile 的
  update-1000/update-2000 均产生三次真实 `1.5x` jump，update-2000 post-jump 生存
  `65/38/74` 帧。distillation loss 从 update-2000 的 `0.3404` 升至 update-2999 的
  `35.8202`，证明最后 1000 updates 出现 Refiner behavior forgetting。
- [x] 三个 Z 使用共享 `stage3_distill_weight_floor=0.25` 到固定 update-2999 endpoint；
  `151014/151015/151016` 均已完成且停在 3000 updates。P/PS 必须使用完全
  相同的 floor、BCPPO、optimizer、seed 和预算，不得选取中间 checkpoint 作为正式结果。
- [x] anchored seed `151015` 严格完成到 `model_2999.pt`：59 个模型张量与 58 项
  optimizer state 均有限，没有多跑 update，也没有自动启动 `151016`、P 或 PS。
- [x] 完成 seed `151015` 的四-profile、五质量 physical-outcome endpoint 审查：
  `1.0x/1.5x/3x/6x/10x` 物理 hold=`4/4,4/4,4/4,0/4,0/4`，drop=
  `0/4,0/4,0/4,4/4,4/4`，`10x` robot fall=`1/4`；严格 SUGAR reference hold=
  `0/4,1/4,3/4,0/4,0/4`。reference termination 与物理结果现分开报告。
- [x] 该 endpoint 已有明确 mild-pass/heavy-failure 区间，当前不做 overfit；冻结训练并
  人眼审查同步 H.264。若审查发现行为无效或含糊，再先做固定条件 serious overfit。
- [x] 在 retained `239098/server44` 完成 seed `151015` 的正式冻结评测：五个质量条件
  各 20 profiles，共 100 条 live PhysX rollout。`1.0x/1.5x/3x/6x/10x` 的 hold=
  `20/20,20/20,16/20,0/20,0/20`，drop=
  `0/20,0/20,0/20,20/20,20/20`，robot fall=
  `0/20,0/20,0/20,4/20,3/20`。五个 `450 x 20` trace 全部有限，100 条质量读回、
  jump 前十帧双手 patch contact 和逐 profile 匹配 delay 均通过；绝对 handoff/event
  的 live PhysX 抖动不超过 4 帧。该任务只运行 frozen actor，没有更新权重，也没有
  自动启动 `151016`、P 或 PS。
- [x] 修复并验证 frozen evaluator 的多 batch 合同：reset 使用 inference mode，每批
  清除 latched termination reason，main 异常保持非零退出，并在进入下一个质量条件前
  检查 summary/trace 均已写出。两批 8-profile 诊断与正式 100 条均无 frame-0 假终止。
- [x] 完成 seed `151015` endpoint 的四级同钟视频审查：`1.5x` 持稳、`3x` profile 8
  持续双手接触但下沉约 `0.052 m`、`6x` profile 0 掉箱并随后整机失稳、`10x` 掉箱。
  四条均为 450-frame H.264、单一目标 G1/CarryBox、左右各 27 patch，且全帧解码通过。
  3x replay 与正式边界类别和下沉量相符；6x replay 与正式轨迹都为 drop，但只把
  post-drop robot-fall 差异报告为 live camera replay 的闭环敏感性，不冒充逐帧一致。
- [x] 依据数值和人眼审查维持训练冻结：不运行 update 3000 之后的训练；第三 Z seed
  后续只按相同合同完成，当前不启动 P/PS，也不触发 overfit。只有后续证据否定当前 endpoint 时才先做
  已声明的固定条件 serious overfit。
- [x] 完成 anchored seed `151014` 的正式 frozen audit：五个质量各 20 profiles；
  每项同一个 profile 1 在 handoff 前发生 Refiner `ee_body_pos@220`，其余 19 条的
  hold=`19,19,16,1,0`、drop=`0,0,2,18,19`。所有 real event 的质量读回、jump 前
  十帧 bilateral patch contact 和 matched delay 均通过。
- [x] 完成 anchored seed `151016`：job `239098` 外部取消后从完整 `model_2750.pt`
  精确恢复，从 iteration 2751 正常完成到 `model_2999.pt`；59 个模型张量和 58 项
  optimizer state 均有限且没有更晚 checkpoint。快速 5x4 审查通过后，正式 100 条
  frozen audit 的 hold=`20,20,20,0,0`、drop=`0,0,0,20,20`，robot fall 全为 0；
  五个 trace 均有限，100/100 mass readback 与 pre-event bilateral-contact gate 通过。
- [x] 完成 seed `151016` 代表性同步视频：3x profile 0 持箱、6x profile 0 下落
  `0.5619 m`；两条均为 450-frame H.264、完整 G1/CarryBox 和左右 27 patch，同钟且
  全帧解码。视频按自身 camera rollout 报告，不替代正式 camera-free 计数。
- [x] 汇总三个完成的 Z pair：eligible hold=`59/59,59/59,52/59,1/59,0/59`，
  drop=`0/59,0/59,2/59,58/59,59/59`。冻结全部训练；当前不触发 overfit，不启动 P/PS。
- [x] 审查相机扰动：`6x` profile 7 的 camera-free 正式轨迹与 repeat 都持稳，但
  camera rollout 下落 `0.279 m`；`3x` profile 0 从 camera-free drop+fall 变为 camera
  hold。新增 3x camera-free 四-profile repeat 正常完成，其 23 个 trace 字段与正式
  trace 的前四条逐值相同，profile 0 精确重复 `0.452212 m` drop+fall。正式计数固定
  使用 camera-free trace，视频只按自身 rollout 标注。
- [x] 完成三个 Z endpoint 的 paired reaction-window audit：每个重质量 profile 与同
  seed/profile 的 1x trace event-align，onset 用 jump 前十帧 paired delta 标定，并把
  连续 load/pressure/shear/friction 与 binary contact 分开。119 条 drop 中 continuous
  patch `119/119` 提前、中位 lead 21 帧，binary lead 15 帧，slip `118/119`、lead 11 帧；
  normal load 与 pressure 单独也均为 `119/119` 提前、中位 lead 20 帧；
  133 条 2 cm sag 中 continuous `133/133` 提前而 binary 只有 `81/133`。Z action 在
  117 条有可检测 onset 的 drop 中有 111 条先分叉。结论固定为连续触觉优于 binary 的提前窗口，但只能检验相对
  proprio-only 的增量收益。
- [x] 生成并人眼检查 6x reaction-window H.264：450 帧、完整 G1/CarryBox 与双手
  27-patch 在左，38 条正式 camera-free drop 时序在右；视频内明确标注两侧为不同
  rollout，且已全帧解码。
- [x] 训练 launcher 与冻结 evaluator 统一使用本地 ground-plane USD 和已转换 G1
  USD；双节点同时失败后，单节点复现证明远端默认 ground asset 返回空 Plane prim，
  不是 BCPPO floor 或 TacSL 失败。
- [x] 将本地 asset、TacSL/PhysX 参数和 `SUGAR_DISABLE_TRAIN_DEBUG_VIS` 环境合同移到
  task registration import 之前，避免 task 在开关生效前创建远端 debug marker，
  并保证正式训练与 frozen evaluator 使用同一物理配置。
- [x] replacement handoff-Z 的恢复链已完成：`151014/151015/151016` 均严格停在有限的
  `model_2999.pt`。历史 jobs 的未保存区间均不计；这些 Z jobs 已由调度器结束，当前
  retained `241811/server28` 用于 P 训练。
- [ ] 新 `P`：完成三个匹配 3000-update seeds。seed `151014` 已严格完成并停在有限
  `model_2999.pt`；`240170/server44` 和 `231256/server64` 的外部取消均只从最后完整
  `model_1500.pt/model_2250.pt` 恢复，最终在 `240922/server07` 正常退出。其配对
  100-rollout frozen evaluation 已完成：P hold=`19,19,17,0,0`、drop=
  `0,0,2,19,19`，单 seed 尚不证明收益。`P/seed151015` 在 `240922/server07` 从零
  启动后遭调度器外部取消，最后完整点为 `model_250.pt`；在 `241217/server59` 从
  iteration251恢复后，该 allocation 于打印 iteration1961后外部结束，最后完整点为
  `model_1750.pt`。随后在 `241298/server59` 从 iteration1751精确恢复并正常完成有限
  `model_2999.pt`。其100-rollout评估完成：P hold=`20,20,18,0,0`、drop=
  `0,0,0,20,20`、fall全零；配对 Z hold=`20,20,16,0,0`、drop相同、fall=
  `0,0,0,4,3`。单 G1 的1.5x physical-hold与3x-drop视频均已完成并全帧解码。
  `P/seed151016` 已生成有限的 `model_500.pt`：59个模型张量、42个 patch-encoder
  张量、58项 optimizer state，学习率为 `1e-5`。`241811/server28` 随后在打印
  iteration747后由调度器外部取消，未保存的501--747不计；tmux-held replacement
  jobs `242229/242239/242242` 正在排队，只能从 iteration501恢复。固定 endpoint 仍为
  `model_2999.pt`。
- [ ] 新 `PS`：完成匹配 3000 updates。
- [ ] 每个分支完成后保留 GPU allocation；停止/失败时只终止记录的 child PGID。
- [ ] 不在三个分支之间修改架构、reward、seed、mass 分布或训练预算。

## H. Frozen evaluation 与报告

- [x] 在观察分支结果前固定 80-frame 判据：hold 最大下降 `<=0.05 m`；drop 下降
  `>=0.15 m` 或回到初始高度 `+0.03 m`；safe lower 要求无 drop/fall、受控下降到
  初始 `+0.08 m`、向下速度不超过 `0.35 m/s`、reference orientation error
  不超过 `0.8 rad`。event 未发生或窗口不足的 profile 不进入三项分母。
- [x] 固定 frozen evaluation 为官方 CarryBox motion 45、frame 0 连续开始，并在
  第一帧 policy observation 前按官方路径同步 motion-relative command buffer。
  默认随机 motion-time 和只写物理状态但未刷新 command buffer 的两次无效预检均已
  移入根目录 `legacy/`。修正后的 update-1000 Z 单 profile 在 frame 63 才终止，
  但仍在接触箱子前失败；它只验证起点修复，正式评估必须等待 update-3000 endpoint。
- [x] 提供单一 frozen-sweep 入口，按固定的一一 seed 配对依次执行 5 个 mass
  conditions，并提供按 factor 汇总 hold/drop/safe-lower 与连续触觉指标的脚本。
- [x] Frozen evaluator 锁存首次 termination 并保存 `valid_frame`；update-1250 Z
  live 复核的有效帧为 `0--63`，终止后 action exact zero 且全为有限值。旧的无效
  termination 后缀轨迹已移入根目录 `legacy/`。
- [x] 用正式 `num_envs=4` 形状完成在线结构预检：四个 profile 的 termination
  frame 为 `63/46/90/70`，逐 profile mask 和 `[420,4,...]` 拼接正确，终止后 action
  全零。活动目录只保留该 4-env 证据。
- [x] 固定检查 seed `151015` 的 update-2000：四个相同 profiles 的 termination 为
  `96/48/201/194`，但仍为零箱体 contact、零 mass event。继续完成 update 3000，
  不把中间生存时长改善当作正式结果。
- [x] 完成旧 frame-zero endpoint 判定：两个 seed 的 `1.5x` 共 `8/8` profiles 与
  seed `151014` 的 `1.0x` 四条均在 contact/jump 前 fall；旧 frozen sweep 到此停止，
  不再为无效入口补齐 300 profiles。
- [x] 更新 frozen evaluator 使用同一个 live official-Refiner handoff，并从 handoff
  后而不是 frame 0 统计 policy-controlled behavior；world/tactile 视频仍显示完整
  frame-zero teacher pickup、交接、增重和后续行为。
- [x] 在 endpoint 结果出现前固定 paired hierarchical bootstrap：先重采样三个
  seed pairs，再重采样每 seed 的 matched profiles，`10,000` 次、analysis seed
  `153015`；比较入口要求 Z/P/PS 各自正好 300 profiles。
- [ ] 对 no-jump 和每个 mass factor 分别比较 hold success、drop/fall、height loss、
  orientation、recovery latency 和 safe-lower outcome。
- [x] 完成 P seed `151014` 的五质量、每项20 profiles正式冻结评测；当前只报告单
  seed 配对结果，不提前形成触觉收益结论。
- [ ] 比较 `P-Z`、`PS-P` 和主要的 `PS-Z` paired 95% confidence intervals。
- [ ] 检查 action divergence 只发生在 jump 后 live observation 更新之后。
- [ ] 报告 nominal no-jump 是否因无条件强握/降姿而退化。
- [ ] 为每个正式 factor 制作 `Z/P/PS` 同钟 H.264：完整 G1 世界画面、左右 27
  patch contact/pressure/shear/slip、jump overlay 和物理结果。
- [ ] 视频主图只使用 patch 单元；taxel detail 只能进入独立 debug 视频。
- [ ] 只有 frozen physical behavior 改善才能写“触觉帮助训练”；gradient/loss/
  action difference 只能写“触觉被使用”。
