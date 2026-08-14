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
- [ ] 用 synchronized patch visualization 检查压力/剪切变化与世界接触同钟对应。
  27-patch 双手布局和 H.264 全解码已通过离线布局测试；`238354`/`server38` 的
  actual world-camera 采集在场景构建前遇到当前已知 Kit/Vulkan
  `ERROR_DEVICE_LOST`，因此该布局测试不是科学视频，必须等真实同钟 world frame
  成功后再勾选。

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
  每个 evaluation seed/factor 固定 20 个 profile，共 300 matched rollouts/arm。

## G. 串行训练

- [x] `Z` one-update preflight：完成 360 个 live steps 和一次 BCPPO update；
  `364` 次 exact-zero observation、`0` 次 TacSL read，report v2 `overall_pass=true`。
  当前随机 warm-start rollout 在 lift 前终止，因此 mass event 如实为 0；其物理准入
  使用已完成的连续动作 full-G1 collector，不提前伪造 jump。
- [ ] `Z`：完成三个冻结 seed 的 3000 updates。seed `151014/151015` 已按完全相同
  配置运行，均已生成并越过可完整读取的 update-500 checkpoint，从纯 distillation
  进入 critic warmup。该里程碑证明可恢复训练和阶段切换链路，不是最终 Z endpoint，
  也不是触觉收益证据。`151014` 的原 allocation 在 update 651 被调度器终止，现已
  从最后完整 `model_500.pt` 精确恢复：BCPPO counter 与下一迭代均为 501，总预算
  保持 3000；到达完整 update 750 后又迁移到五天 `238250`/`server23`，并从
  BCPPO/runner iteration 751 接续。`151015` 也被调度器在打印 update 784 后外部
  终止；最后完整 checkpoint 为 update 750，现已在 `238355`/`server07` 精确恢复到
  BCPPO/runner iteration 751，固定总 endpoint 3000、剩余 2249 updates。两个
  resumed seed 现均已生成可完整读取的 update-1000 checkpoint，并进入 task-reward
  PPO authority ramp；仍需完成到 3000，不能把该阶段切换写成触觉收益。
- [x] `P` one-update preflight：`361` 次在线 feature update、`19,494 = 361 x 54`
  次官方 patch sensor read、`0` 次 slip call，并完成一次 BCPPO update。当前
  warm-start policy 尚未进入抓箱窗口，所以 contact/load 如实为 0；非零在线信号
  由已准入的 continuous full-G1 collector 提供。
- [ ] `P`：完成匹配 3000 updates。
- [x] `PS` one-update preflight：`361` 次在线 feature update、`19,494 = 361 x 54`
  次官方 patch sensor read、`361` 次 causal slip callable，并完成一次 BCPPO update。
- [ ] `PS`：完成匹配 3000 updates。
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
- [ ] 对 no-jump 和每个 mass factor 分别比较 hold success、drop/fall、height loss、
  orientation、recovery latency 和 safe-lower outcome。
- [ ] 比较 `P-Z`、`PS-P` 和主要的 `PS-Z` paired 95% confidence intervals。
- [ ] 检查 action divergence 只发生在 jump 后 live observation 更新之后。
- [ ] 报告 nominal no-jump 是否因无条件强握/降姿而退化。
- [ ] 为每个正式 factor 制作 `Z/P/PS` 同钟 H.264：完整 G1 世界画面、左右 27
  patch contact/pressure/shear/slip、jump overlay 和物理结果。
- [ ] 视频主图只使用 patch 单元；taxel detail 只能进入独立 debug 视频。
- [ ] 只有 frozen physical behavior 改善才能写“触觉帮助训练”；gradient/loss/
  action difference 只能写“触觉被使用”。
