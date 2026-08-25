# Curiosity

本仓库研究两个问题：示范反馈能否真正改变并约束 humanoid 操作策略，以及在线整手触觉
能否在视觉不变、接触 binary 不变但物体动力学改变时提供额外适应能力。仿真统一使用
IsaacLab/PhysX；Newton 只作为 asset 来源，不作为执行后端。

## 当前结论

### Demo following：可执行双技能路由成立，任意 demo following 尚未成立

最新实验保留 released CarryBox/KickBox 两个 official Tracker actor 的完整参数和
`510-D -> 512/256/128 -> 29-D` 闭环结构，只训练读取冻结 predictor `798-D` causal
selected-demo condition 的 router。审计确认官方 Carry/Kick 推理环境除 SMALLBOX/BIGBOX
资产外代码相同，但 Tracker 的 510-D 输入包含各任务独立 Generator 输出的 36-D command；
因此有效技能单元必须是完整 `Generator + Tracker` pair，不能只换 Tracker。

最终 matched frozen physics 在每个域内使用 bitwise-identical initial state、post-prefix state、
prefix action 和 seed。SMALLBOX/Carry 域选择 Carry45 完整技能得到 `18/20` 搬运、平均最大
抬升 `0.43204 m`、`0/20` 跌倒；只改变 causal demo condition 并联合选择 Kick21 完整技能后，
得到 `19/20` 踢动、`0/20` 搬运、平面净位移 `0.49079 m`、`0/20` 跌倒。两臂的 36-D
Generator command 和实际 29-D action 的 mean absolute difference 分别为 `0.27559` 和
`1.00992`，而不是仅改变一个显示分数。

联合路由直接修复了此前只换 Tracker 的失控：Carry→Kick 的 raw action 最大值从
`5.87e11` 降到 `5.1712`，physical fall 从 `9/20` 降到 `0/20`。BIGBOX/Kick 域的 matched
Kick21 为 `20/20` 踢动、`0/20` 跌倒、平面净位移 `1.06092 m`；反向选择 Carry45 只有
`8/20` 搬运，raw action 最大 `68.437`，因此被自动行为/envelope gate 拒绝。这证明完整控制链
路由在 SMALLBOX 上已经形成稳定 Carry/Kick 语义切换，同时也明确暴露了 BIGBOX 资产尺寸、
目标姿态和技能初始分布的反向兼容性边界。

三个严格 `2 x 2` 因子实验进一步把该边界定位为几何/初始化分布耦合：SMALLBOX 在
small/big nominal mass 下均可执行 Carry45，BIGBOX 在两种质量下均失败，所以 `1.5x` 质量
不是原因；Carry 初始化对 Carry/Kick 最终目标都可搬运，Kick 初始化对两种目标都不可靠，
所以只换 goal 也不能修复。asset × motion context 的结果是 crossover，而不是“BIGBOX
永远失败”。完整复现与四格视频命令见 [reproducibility](DOCS/reproducibility.md)。

进一步的安全切换审计否决了两个看似直接的办法。在线 shadow `Generator + Tracker` 已在
兼容 CarryBox→Kick21 上做到 650 步 command/action/object/robot trace 全部 bitwise equal，
因此实现本身已对齐官方链路；但动作幅值阈值直到状态已经失稳后才触发，候选/回退动作分别
爆到 `2.17e5/3.85e5`，没有阻止跌倒。官方 Generator 自带的训练 min/max normalizer 也不能
提前区分：前 100 帧中，不兼容 Carry-on-BIGBOX 的越界比例均值 `0.00132`，反而低于成功
Kick-on-SMALLBOX 的 `0.00625`。这两条 threshold 路线不再继续调参。

随后完成了 serious causal transition-risk Transformer，而不是用简单 MLP 占位。模型为
6 层、384-D、8 heads、11.012M 参数，部署输入严格是过去 `10 x 539`：官方 Tracker 当前
510-D observation 与当前 Carry candidate 的 29-D action；未来是否跌倒、是否成功和动作是否
越界只作为整条 profile 的训练标签。train/validation profile 分离，test 使用独立 seed/context。
固定 500-step 小样本 overfit 先通过；正式冻结模型的 first-50 held-out AUROC 为 `0.7430`、
balanced accuracy `0.6536`、risk-safe probability gap `0.2655`，阈值 `0.715` 只由 validation
first-50 选择。因此它确实能做离线早期风险排序，但这还不是安全控制证明。

严格在线对照否决了“第 49 帧二分类后硬切 expert”。direct 与 composition 的初始状态、一步
prefix 和前 50 帧 Carry candidate action 完全一致；风险模型用 9 个因果样本把 `10/20`
profiles 锁定到 official Kick fallback。composition 仍出现一次 physical fall，并在第 `447`
帧首次产生非有限 Tracker transition。失效的 env0 风险为 `0.8885 > 0.715`，而且早已处于
fallback，所以不是 classifier 漏检；问题是先走 50 帧 Carry 后突然切到 Kick，已经把官方
Kick expert 放入不稳定状态分布。当前 transition-risk 只能称为离线风险信号，不能称为安全
切换器。同步 H.264 失败证据位于
`experiments/demo_following/official_transition_risk_v1/online_fallback_seed171627_v2/`
`videos_exact_failure_v2/direct_vs_causal_risk_fallback.mp4`。下一步先用 validation-only 规则
审计最早可用的 frame-9 决策。该审计现已完成：test AUROC `0.7239`、balanced accuracy
`0.6553`、risk-safe gap `0.3564`，但 validation-only 阈值 `0.84` 下 test Brier `0.2773`，
差于 prevalence baseline `0.2331`。因此不启动另一个 hard-switch rollout；下一阶段训练连续的
causal transition/recovery controller，不再调在线失败阈值。

第一项连续 recovery 诊断现已完成。每个 episode 都在 IsaacLab/PhysX 中先执行 1 帧 official
Kick 对齐，再执行 9 帧 official Carry `Generator + Tracker`，这些前缀帧不进入 PPO；随后从
released Kick Tracker 的 actor/critic 权重出发，用仓库 BCPPO、固定 `1e-5` 学习率和 frozen
Kick action-mean anchor 训练到 update 64。冻结 seed181629 的 baseline/update64 初始 robot、
joint、box 和 510-D observation 逐元素一致。两者都是 `20/20` Kick、`0/20` 跌倒；训练后平均
箱体平面净位移从 `0.17136` 增至 `0.18634 m`，足—箱接触比例从 `0.0632` 增至 `0.0674`，
每帧 reward 从 `0.072629` 增至 `0.073203`。因此这是稳定的小幅局部增益，但 Carry-9 本身已
被 official Kick 完全解决，不能声称困难跨技能恢复已经完成。完整对照视频位于
`experiments/demo_following/cross_skill_recovery_v1/bcppo_frozen_eval_seed181629/videos/`
`matched_baseline_vs_update64_actual_world.mp4`。

尝试延长到 128 updates 时，update 67 后出现 timeout-only 大并行环境的稀有 outlier 并污染
critic；该延长无效，不作为结果，也不从坏 checkpoint 续训。下一步先冻结 official Kick，
无训练扫描 Carry 前缀长度，找到“状态仍有限但 Kick 已不再成功”的难度前沿，再在一个固定
前沿条件上训练匹配 recovery，而不是继续给已经 20/20 的 Carry-9 条件加训练步数。

该困难前沿实验现已完成。冻结 official Kick 在 Carry prefix
`9/17/25/33/41/49/57/65/73/81/89/97` 上的 safe-success 为
`18/20/17/19/14/11/14/16/16/17/16/16`；不存在“handoff 仍直立且少于 10/20 成功”的严格点。
prefix41 是最大直立失败边界：最低 root height `0.6680 m`，所有 state/action 有限，
`14/20` safe Kick、`6/20` fall。prefix49 虽然更差，但最低 root 已低于预设 `0.65 m`。

在 prefix41 上完成了两组 `171631 -> 181631`、update64 matched recovery。无安全 penalty 的
训练把位移 `0.1885 -> 0.3013 m`、reward `-1.6766 -> -1.2137`，但 safe Kick 仍为 `17/20`，
fall 从 `2/20` 恶化到 `3/20`。加入 standard physical-invalid termination penalty 后，fall
保持 `2/20`，contact fraction `0.0726 -> 0.0966`、位移到 `0.2015 m`，但 safe Kick 仍为
`17/20`。因此两者都没有真实恢复增益；后续不再增加 update 或调 penalty/reward scale，必须
转向 serious shared skill prior 和 state-aware transition objective。

准确结论是：causal demo condition 能可靠选择两个已发布完整技能，并在同一 SMALLBOX
物理场景中产生可执行的 Carry/Kick 行为分叉；它仍不是任意视频生成新技能，也不是连续技能
latent 或安全跨资产 transition policy。最终四视频位于
`experiments/demo_following/official_tracker_router_v1/seed161610/`
`videos_joint_reference_actual_final/`。
此前单 MLP 的完整 `510-D` offline BC 虽把 MSE 降到 `0.00682`，仍在 Carry 闭环中
`0/20` 成功、`20/20` 跌倒；三阶段 `beta=0/0.5/0.9` DAgger 最终也只有 `6/20` Carry、
`14/20` 跌倒。继续增加离线训练步数不是当前路线。

最新的固定 overfit 诊断进一步定位了瓶颈。它保留同一个 serious SUGAR `512/256/128`
shared actor、冻结 11.386M predictor 和 CarryBox45 Refiner 执行基线，只训练 actor residual：
correct 条件目标为零残差，unrelated 条件目标为 official
`KickBox21 Tracker action - CarryBox45 Tracker action`。同一状态同时配对两种条件，未来 action
只作为训练 label，冻结评估时 actor 仍只读取当前因果状态和 selected-demo condition。3000
optimizer steps 后，训练 MSE 从 `1.48955` 降到 `0.10764`，同一状态切换条件的动作 mean/max
差为 `0.98783/12.1281`。

同一个 step-3000 checkpoint 在 20 个完全相同的 Carry 初始状态上只切换条件后，correct
保持稳定搬运：平均最大抬升 `0.68792 m`、双手接触比例 `0.84006`、physical fall `0/20`。
unrelated 则平均只抬升 `0.00267 m`、双手接触比例为 `0`、ground-transport 为 `0.99764`、
足—箱接触比例从 `0.00289` 增至 `0.03646`，但 physical fall 达 `15/20`。视频中可看到抬腿和
绕箱倾向，也可看到摔倒；它不是成功 KickBox。该结果首次证明 action-direction supervision
足以让同一 actor 离开 Carry 解，但也证明“行为分叉”不等于“语义正确的 demo following”。
固定 Carry Refiner 基线和 Carry frame-197 初始化不提供可稳定执行的 Kick 状态分布，下一步
必须在 official Carry/Kick 两种物理 rollout 上训练共享条件 actor，再做同 checkpoint 条件交换。

上一阶段已经消除了“两个 demo 对应两个 checkpoint”的混杂。一个共享的 serious SUGAR
actor 在同一批 20 个 CarryBox 环境中训练 64 updates：10 个环境读取 Carry45 条件，10 个读取
Kick21 条件，teacher、任务和物理全部仍为 CarryBox45。actor 每步读取冻结 11.386M causal
predictor 提供的 798-D selected-demo/当前轨迹条件；未来事件和 GT 轨迹不进入 actor。

冻结评估两次加载完全相同的 `policy.pt` 和初始状态，只交换 demo 条件。残差动作的平均/最大
绝对差为 `0.01319/0.37943`，证明测试时 demo 条件确实能调制同一策略。20 profiles 中，
correct/unrelated 的平均最大抬升为 `0.68367/0.66868 m`，双手接触帧为 `294.3/287.4`，
physical falls 为 `0/20` 与 `1/20`；unrelated 在预登记行为方向上达到 `3/4`，但两边仍主要
是双手搬箱，没有形成完整 Kick 接触结构。当前准确结论是“same-policy conditioning 有因果
作用，但 64 updates 尚未产生语义级策略切换”。

当前有效实验固定同一个 CarryBox45 official Refiner teacher、相同初始化、物理、seeds、
优化器、reward weights 和 64-update budget，只改变 internal reward 读取的 selected demo：

- `correct`：CarryBox45；
- `unrelated`：KickBox21，任务仍然是 CarryBox。

teacher-only zero-residual gate 在 20/20 profiles 中实现双手接触并抬升至少 5 cm，平均
最大抬升 0.7128 m。三组独立训练 seed 的 correct/unrelated frozen success 分别为
`16/18`、`18/17`、`16/17`（每臂 20 profiles），physical falls 分别为 `2/2`、`1/1`、
`2/1`。checkpoint 和 residual action 均发生可测差异，因此 selected-demo reward 确实进入
优化并改变策略；task success 没有稳定的 correct-demo 优势。

predictor-independent 行为审计进一步排除了“只是分数没显示出来”的解释。审计不读取
predictor loss、demo reward 或训练 loss，只读取机器人/箱子状态、明确过滤到箱子的手脚
接触和运动学。CarryBox45 reference 最大抬升 `0.7639 m`，KickBox21 最大仅 `0.0304 m`。
三个 training seeds 上，unrelated 减 correct 的 lifted-frame delta 为
`+0.0350/+0.0179/-0.0058`，lifted-transport delta 为
`+0.0323/+0.0132/-0.0277`，orbit-rate delta 为
`-0.0050/-0.0294/-0.0115 rad/s`。预注册 Kick-like 方向在 lift/transport 上仅 `1/3`
seeds，在 orbit 上为 `0/3`；新增脚—箱接触也接近零。seed161585 有部分 `3/4` 方向变化，
但另外两个 seeds 为 `0/4`。因此三种子结果不支持稳定 semantic demo following，只支持
selected reward 改变 Carry 解族内行为。

随后完成固定物理的 teacher-authority learnability diagnostic：两臂都从各自 update-64
端点继续 64 updates，共同 CarryBox45 teacher 从 `1.0` 线性降至 `0.25`，只有 selected
reward demo 不同。两臂训练 proof 和 20-profile frozen evaluation 都通过，但行为发生坍塌：
correct 与 unrelated 的双手接触率、5 cm 抬升率和 lifted transport 均为 `0`，足—箱接触也
均为 `0`，四个预登记 Kick-like 方向为 `0/4`。因此不能进入多 seed；降低 teacher authority
既没有保住 Carry，也没有使 trajectory-only reward 产生 Kick 接触语义。

自动转入 contact/event reward redesign 后，官方 reference corpus 审计已覆盖 100 条
CarryBox 和 99 条 KickBox。binary contact proxy 仅用作示范事件标签，不作触觉力：Carry
接触帧最近效应器为手的比例均值为 `95.46%`，Kick 接触帧最近效应器为脚的比例均值为
`99.78%`；Carry 中位 lifted-moving fraction 为 `40.85%`，Kick 为 `0%`。这证明 reference
中有清晰可分的接触角色和物体运动 regime。

actual-rollout redesign 已完成。official Tracker 在 IsaacLab/PhysX 中为每条
source motion 采集 700 个同钟帧，实际 target 来自分别过滤到箱子的左右手/左右脚
`force_matrix_w_history`，而不是 reference binary。完整 corpus 覆盖 100 条 CarryBox 和
99 条 KickBox、无重复和 reset。Carry 的中位双手同时接触率为 `32.93%`、最长手部事件
`4.60 s`、最大抬升 `0.490 m`；Kick 的中位足部接触率为 `4.14%`、最长足部事件
`0.22 s`、足力峰值 `60.23 N`、最大抬升仅 `0.0066 m`。

第一版 predictor 虽通过 held-out MAE，却不能作为 reward：它读取不可直接接入 actor 的
510-D Tracker observation，并允许每个时刻从 32 个 demo windows 中自由选择最小误差。
直接审计发现该规则会让 Kick 轨迹错误偏好 Carry45；原因不是 uncertainty，而是任意跳到
静止 demo 片段的 phase loophole。该版本已降为失败诊断。

正式版本保留 serious 6-layer、384-D causal Transformer，具有 `11,386,010` 个参数，输入为
过去 `10 x 121` 的部署侧核心观测、固定 numeric selected demo 和 `[0,1]` 因果归一化时钟
phase；未来 actual events 只作 13-D label。phase 固定后，真实 mismatch 在 validation/test
同时恢复 Carry→Carry45、Kick→Kick21 的正确方向。formal seed271303 冻结 epoch 20，
validation/test normalized MAE 为 `0.1771/0.1560`，优于 constant `0.2803/0.2566`；
zero-demo 为 `0.2945/0.2766`，permuted-demo 为 `0.2018/0.1761`，median Spearman 为
`0.677/0.694`，12/12 gates 通过。

冻结 checkpoint 后，仅用 validation 拟合 uncertainty；90% 区间在 validation/test 的平均
覆盖率为 `97.13%/97.77%`，test 最低单目标为 `91.86%`。固定 Carry45/Kick21 的完整
reward-scale audit 通过全部 10 项门槛，validation/test 都双向偏好匹配任务。dense feedback
固定为 `eta * (exp(-calibrated_event_risk) - train baseline)`，`eta=0.2427623309`、clip
`0.1431077421`，平均绝对幅度为既有 task/constraint reward 的 25%。冻结 runtime 已验证
121-D 输入、9-transition warmup、reset-safe history、phase 输入、eval mode 和零可训练参数。
这建立了可接入策略的因果语义奖励，不等于 policy 已经遵循 demo。

该 reward 已接入正式 SUGAR rollout boundary：base task/SMP/original ICM 保持不变，冻结
predictor 只把 dense feedback 加到 policy reward，并记录 risk、uncertainty、ready 和 phase。
correct/unrelated 两臂的同 teacher 协议、update 32/64 checkpoint、冻结评估、独立行为审计和
最终双视频入口均已通过 dry-run/CPU 回归。
随后在 retained H200 job257762 上通过正式内层 runner 的两臂 admission：Isaac Sim/Vulkan
启动成功，correct 解析为 CarryBox45，unrelated 解析为 KickBox21，均为 121-D、clock-phase、
0 trainable parameter、0 PPO update。该 job 为 5 天 allocation，当前已恢复 GPU hold。

随后 online-rollout 准入发现一个实质性混杂：`explicit_zero_control` 虽把 actor/ICM 的触觉
tensor 置零，旧配置仍会实例化双 R15 TacSL scene。现已改为原始 SUGAR G1/CarryBox scene，
完全不创建 TacSL sensor。2026-08-24 在 fresh H200 job257815/server54 上，最小
`SimulationApp` canary 通过，correct 与 unrelated 的 24-step、零 optimizer online smoke
也依次通过：真实执行 actor、冻结 Refiner、SMP、original ICM、phase-aware reward 和 rollout
storage，同时 policy/ICM 参数及 optimizer counter 全部不变。此前 server60/server45 的
`ERROR_DEVICE_LOST` 属于已损坏的 GPU runtime 状态，不是 reward、teacher 或 policy 失败。
两臂未优化时的动作与 base reward 逐步完全相同；history ready 后 16 步的 mean demo
reward 为 correct `0.04013`、unrelated `0.01734`，对应 mean risk `0.36048/0.50554`。因此
selector 确实在同一物理 rollout 上读到了不同 demo，而不是由行为差异伪造 reward 差异。
这一步是正式优化前的在线接入证据。

同一 fresh H200 上还完成了正式训练前的 frozen teacher-only 门禁。旧 evaluator 虽然把触觉
tensor 置零，却仍构造 TacSL scene；现已与 training/smoke 统一为原始无 TacSL 的 SUGAR
G1/CarryBox scene，并直接写入、回读 nominal object/robot mass、inertia 和 `0.5/0.5`
friction。20 个 profile、400 control steps、exact-zero residual 全部通过：最大抬升范围
`0.6854--0.7224 m`，双手刚体接触 `153--156` 帧，物理跌倒 `0/20`。因此 matched
experiment 的共同 CarryBox45 teacher 起点本身具备稳定抓取和抬升能力。

进一步审计发现：无 TacSL scene 仍继承官方 SUGAR 的 startup mass/material randomization；若只
保存 seed 而不保存实际 PhysX readback，冻结评估不能严格恢复训练物理。现已在每个正式 proof
记录逐环境 object/robot material、object mass、inertia 和 COM，评估前逐项恢复并回读。重新
执行的 correct/unrelated online smoke 证明两臂 startup physics 完全相同，且 action/base
reward 仍逐步完全一致，只有 selected-demo reward 保持 `0.04013/0.01734` 的差异。

训练前的 reward-to-gradient admission 也已在同一 fresh H200 上通过。它在每条相同的
24-step rollout 上保留正式 total reward，再仅减去 selected-demo feedback 构造 counterfactual
base reward，分别运行同一个 PPO GAE/advantage 计算，并对精确 clipped actor surrogate 求梯度；
全程不调用 optimizer。correct 的 return/normalized-advantage 最大变化为
`0.45412/0.25342`，actor-gradient delta L2 为 `0.07804`；unrelated 分别为
`0.23489/0.16923/0.04430`。两臂均保持 `0` policy update 和参数不变。因此 demo feedback
不只进入日志或 reward tensor，确实改变了 PPO 将要使用的 actor 学习方向；这仍不是训练后
行为遵循的证据。

同一门禁还排除了 fixed teacher 遮蔽 student 的疑问。正式动作公式是
`executed = 1.0 * teacher + 1.0 * residual`；correct/unrelated 两臂中，29-D sampled residual
最大绝对值均为 `3.72674`，wrapper 公式和 ActionManager raw input 都逐元素精确，joint
scale/offset 逆变换的最大 float32 误差为 `4.77e-7`，低于既有 `2e-6` 容差。因此 student
residual 确实到达环境，固定 CarryBox45 teacher 只是共同基线，不会把 student action 乘零。

同时修复了 probe 的失败语义：Isaac Sim 关闭阶段可能掩盖内层 Python 非零退出状态。外层
runner 现在必须读取独立 machine-readable result，并核对 protocol、`passed=true` 和
`policy_updates_executed=0`；缺失或无效结果一律失败，不能再只凭 subprocess return code 放行。

2026-08-24 已在 fresh H200 上完成新的严格 matched policy experiment。两臂使用完全相同的
CarryBox45 official Refiner teacher、seed/action seed `161587/161588`、20 个环境、startup
PhysX readback、PPO/ICM/SMP、reward weights 和 64 updates；唯一变量是 selected demo：
CarryBox45 或 KickBox21。两臂 proof 均通过 65/65 checks，update 32/64 checkpoint 均有限且
可精确 reload。冻结评估使用 seed `171587`，每个 checkpoint 20 个相同 physics profiles。

update 32 和 update 64 的独立行为审计均观察到 `3/4` 个预登记方向，但两臂仍是稳定的
Carry 解。update 64 的 correct/unrelated 平均最大抬升为 `0.69332/0.69666 m`，双手同时
接触比例为 `0.83335/0.83447`，lifted-frame 比例为 `0.61142/0.61644`，lifted transport
为 `0.94514/0.94141`，ground transport 为 `0.05486/0.05859`，orbit rate 为
`0.37166/0.37547 rad/s`，均为 `0/20` physical fall。unrelated arm 因此减少了空中运输占比、
增加了地面运输和绕箱运动，但没有减少 lifted-frame time；实际足—箱接触仍约为每个 episode
一帧。update 32 则在 lifted time、lifted/ground transport 三个方向移动，但 orbit 未移动。
这比旧错误时钟实验更强：selected reward 确实把行为推向部分 Kick-like 方向；但它仍未产生
KickBox21 的脚部接触结构，而且只有一个训练 seed，不能声称完整或可复现的 semantic following。

独立复现 `161589/161590 -> 171589` 已完成。update 64 再次得到相同的 `3/4` 方向：两组 seed
的 unrelated-minus-correct lifted-transport delta 为 `-0.00372/-0.00386`，ground-transport
delta 为 `+0.00372/+0.00386`，orbit-rate delta 为 `+0.00381/+0.00369 rad/s`；两组都没有
减少 lifted-frame time。update 32 则为 `3/4` 与 `1/4`，不稳定。第二组 update 64 的
correct/unrelated 最大抬升为 `0.69619/0.70516 m`，双手接触率为 `0.83439/0.83226`，两臂
physical fall 都是 `0/20`，足—箱接触仍约一帧。因此当前最准确结论是：64 updates 后可重复
出现小幅行为方向变化，但没有形成 Kick 接触结构或完整语义遵循。

固定 4x reward-strength overfit 也已完成。它复用第二组 seed，并只把 `eta` 与
`reward_clip` 乘四；两臂 training proof、冻结评估和视频全部通过。结果没有改善：update 64
从 baseline `3/4` 降为 `1/4`，unrelated-minus-correct ground transport 从 `+0.00386` 反转为
`-0.02801`，orbit 从 `+0.00369` 反转为 `-0.03546 rad/s`，足—箱接触优势为 `-0.00535`。
unrelated 累计 feedback 从 `-11.99` 放大到 `-48.64`，predicted loss 仅改善 `0.00320`，实际
仍搬箱。结论不是“再加大 reward”，而是标量 mismatch reward 缺少产生新接触拓扑的可执行
方向；继续 scale sweep 没有科学依据。

冻结评估还暴露出比“训练不够久”更关键的问题：在实际 Refiner+residual Carry 轨迹上，correct
arm 的平均 Carry45/Kick21 predicted mismatch 为 `0.96986/0.89087`，即 predictor 反而认为
这个明显的搬箱轨迹更接近 Kick。严格 scorer-only 审计现已保存逐帧 exact `121-D` 输入，并在
同一 frozen trajectory、同一 frozen 11.386M predictor 上只改变初始 phase。旧的 phase-0
运行在 correct/unrelated、update 32/64 四个 block 中均得到约 `-0.080~-0.082` 的
`Kick risk - Carry risk`，且 `0/20` profiles 偏好 Carry；从真实恢复的 CarryBox45 reference
frame `197` 起钟后，margin 变为 `+0.324~+0.328`，Carry-preferred frame 达
`85.7%~86.1%`，四个 block 均为 `20/20` profiles 偏好 Carry。旧 runtime 的 phase、ready、
reward、risk 和 uncertainty 均在 float32/模型容差内复现。因此在线语义倒置的直接原因是
初始 phase 错位；正式 scorer/runner/evaluator 已改为从 reset reference frame 起钟。

Tracker-to-Refiner rollout shift 仍真实存在，但不再是解释当前 Carry 语义倒置所必需的原因：
official Tracker test 的 normalized state `mean|z|/p95/p99` 为
`0.668/1.923/2.882`，当前 frozen Carry rollout 为约 `1.035/3.212/5.420`，主要来自 joint
position、projected gravity、box velocity 和 previous action。此时结果只通过了 Carry-domain
必要门槛，也尚未用修正后的 reward 重新训练策略，所以仍不能声称 policy 已语义遵循 demo。

随后完成 motion-disjoint official Generator/Tracker Kick-domain gate。使用 predictor test
split 的 KickBox motions `9/19/.../89`，9/9 轨迹都有足—箱接触，9/9 都让箱子平移至少 1 cm；冻结
predictor 在部署用 fixed-650 clock 下得到 mean `Kick risk - Carry risk=-0.06508`，ready frames
中 `50.50%` 偏好 Kick，`8/9` motion profiles 整体偏好 Kick21。motion29 仍错误偏好 Carry45，
所以该结果证明 official inference Kick 域上的多数 motion transfer，不证明 universal transfer，
也不替代 Refiner-plus-residual Kick rollout。

2026-08-24 随后在 retained H200 job258074 的 fresh physical GPU7 上完成正式修正门槛。correct
和 unrelated 都运行 24 个真实环境 control steps，参数与 optimizer counters 不变、policy
updates 为 0，且 frozen scorer 均明确从 episode step `197` 起钟。相同未优化 Carry rollout 上，
ready-step mean reward/risk 为 correct `+0.04804/0.31539`、unrelated
`-0.00338/0.65776`。因此修正后的在线 reward 已恢复正确方向。

同一 GPU 上的 frozen Carry evaluation 也通过：correct/unrelated 的 update 32/64 四个 block
均为 `20/20` profiles 偏好 Carry45，mean `Kick risk - Carry risk` 为
`+0.32437/+0.32724/+0.32451/+0.32787`，Carry-preferred ready-frame fraction 为
`85.77%/86.10%/85.71%/86.12%`。这与 exact-prefix scorer-only 预测一致，说明 phase 修复已在
真实 evaluator 中落地；它仍不等于旧 policy 自动获得 semantic following，因为旧 policy 是
在错误 phase 下训练的。

官方 MimicKit TinyMDM 目前只是 generic motion prior。两个 official single-clip prior 能
完美识别各自训练 clip，但 CarryBox96/KickBox22 的独立同任务扩展没有通过。因此没有把
任意 Transformer hidden state 冒充 SMP latent，现有证据也不支持 selected-demo SMP
policy integration。

### 在线整手触觉：实现保留，收益结论冻结

每只手有 27 个物理解剖 patch：掌心 `4 x 3`，拇指、食指、中指、无名指和小指各有
proximal/middle/distal 三段。policy unit 是 patch；TacSL/R15 taxel 只作为每个 patch 内部
的物理采样与审计后端。每个在线 patch record 包含 contact、normal load、mean pressure、
signed local-XY shear 和 friction utilization；PS 额外使用 causal batch-stateful slip
callable。

2026-08-20 审计发现并修复了接触负奖励、dead contact sensor、缺失 hold reward、
normal/shear 混合、摩擦不一致、slip reset 丢失、训练/评测 motion 不匹配、评测关闭终止
以及统计多重比较问题。旧 Z/P/PS 数字全部撤回。

修正后的 tactile-only diagnostic 在 model1100 的 20-profile 评估为 `14/20` physical hold、
`6/20` strict success、0 drop、0 physical fall 和 10 reference deviations。它证明在线触觉
可以进入 actor 并改变参数，但没有证明触觉改善重量突变后的物理行为。Plan 15 因此冻结，
不得继续盲训或把历史结果称为触觉增益。

独立 frozen-Refiner friction sweep 表明 6x、`mu=1.5` 可满足 5 cm hold；10x 在
`mu=0.5/1.0/1.5/2.0` 下全部掉落。这是物理可行性结论，不是触觉策略收益。

## 明确贡献

本仓库新增并验证的贡献是：

1. 将官方 SUGAR CarryBox 的 Refiner、Tracker、Generator 输入输出和 state-based policy
   边界整理为可复现基线；
2. 在 IsaacLab/PhysX 中为完整 G1 建立双手 54 个在线物理解剖 TacSL patch，并提供同钟
   world/双手 27-patch 可视化；
3. 建立不读取物体速度、质量、jump flag 或未来帧的 causal `PatchSlipDetector.update`；
4. 建立 teacher handoff 后在线改变 mass/inertia 的 matched Z/P/PS 协议及本体感受泄漏
   审计；
5. 实现 11.386M phase-aware causal trajectory/contact/duration/regime mismatch predictor，
   修复自由窗口静止片段漏洞，通过 motion-disjoint、zero/permuted-demo 与双向语义检查，并
   用 exact 121-D frozen-policy 重评分定位并修复 nonzero-reference phase 初始化错误；
6. 建立 fixed-teacher、只改变 selected-demo reward 的因果实验，排除 teacher replacement
   混杂；
7. 建立 predictor/reward-independent、以训练 seed 为重复单位的 Carry/Kick 行为审计，
   分离 task success、reward use 与 semantic obedience；
8. 对 official single-clip TinyMDM 做 exact-identity 与 independent semantic-extension
   分离测试，得到“记住 clip 但尚未形成可靠语义空间”的负结果；
9. 建立 causal demo-conditioned official-skill router：一个 checkpoint 内保留参数完全不变的
   Carry/Kick released Tracker，并用 matched/counterfactual frozen physics 分离“可执行技能
   选择”“task generator 耦合”和“跨技能域外失控”。

官方 SUGAR、IsaacLab TacSL 和 MimicKit TinyMDM 本身不是本仓库的原创方法；本仓库的
贡献是忠实接入、实验协议、在线传感扩展、因果隔离与失效审计。

## 最短入口

完整环境、资产、命令、输出合同和结果核验见
[可复现性与证据记录](DOCS/reproducibility.md)。常用入口如下。

```bash
cd /public/home/yanhongru/Curiosity
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python

# retained GPU：训练一个 checkpoint 内的 causal Carry/Kick official-skill router
$PYTHON_BIN scripts/sugar/demo_following/train_official_tracker_router.py

# retained GPU：四臂冻结评估入口；同域两臂必须使用相同 seed
$PYTHON_BIN scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py \
  --domain CarryBox --selected-demo-option correct \
  --shared-checkpoint experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/policy.pt \
  --training-proof experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/proof.json \
  --output-dir "$PWD/experiments/demo_following/router_reproduction/carry_correct" \
  --num-envs 20 --steps 650 --seed 171610 --headless --device cuda:0

# 无仿真：检查 phase-aware matched 配置和下一条训练命令
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --dry-run

# retained GPU：正式内层 runner/model admission，明确不创建环境、不执行 PPO
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --runner-admission-only

# retained GPU：真实环境执行一个 24-step online rollout，但不调用 optimizer/update
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --runner-rollout-smoke-only

# scorer-only：重采 exact 121-D frozen trace，并比较 phase-0 与 reference-aware phase
OUTPUT_ROOT="$PWD/experiments/demo_following/reproduce_phase_transfer" \
  bash scripts/sugar/demo_following/run_phase_event_scorer_transfer_audit.sh

# 从全新目录串行运行 phase-corrected matched pair；脚本完成后停止
OUTPUT_ROOT="$PWD/experiments/demo_following/matched_phase_event_reward_reference_aware_v2" \
  bash scripts/sugar/demo_following/run_reference_aware_phase_event_pair.sh

# 两臂 endpoint proof 通过后：复现冻结评估 update 32/64、独立行为审计和完整双视频
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 phase_event_reward_only \
  "$PWD/experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587"

# 无 GPU：Vulkan 相机不可用时，从已通过的冻结 PhysX trace 生成精确行为视频
$PYTHON_BIN scripts/sugar/demo_following/render_frozen_trace_behavior.py \
  --correct-trace "$PWD/experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/evaluation_update0064/correct/TRACE.npz" \
  --unrelated-trace "$PWD/experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/evaluation_update0064/unrelated/TRACE.npz" \
  --output-dir "$PWD/experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/videos_update0064_trace_exact"

# 无 GPU：从现有 traces 重算独立行为审计
$PYTHON_BIN scripts/sugar/demo_following/analyze_behavior_adherence.py

# retained GPU：固定 3000-step official-action direction overfit，然后串行冻结评估并渲染
$PYTHON_BIN scripts/sugar/demo_following/train_shared_topology_distillation.py
bash scripts/sugar/demo_following/evaluate_shared_topology_distillation_pair.sh

# 无 GPU：汇总三个独立训练 seeds；20 physics profiles 只作 seed 内变化
$PYTHON_BIN scripts/sugar/demo_following/aggregate_behavior_adherence.py

# GPU compute node：复现 teacher 1.0 -> 0.25 的单 seed 诊断、冻结评估和视频
bash scripts/sugar/demo_following/run_teacher_floor_overfit_pair.sh

# 无 GPU：审计 199 条 official Carry/Kick reference 的 contact/event 标签可分性
$PYTHON_BIN scripts/sugar/demo_reward/audit_contact_event_reference_corpus.py

# GPU compute node：采集部署侧 121-D corpus；脚本串行覆盖 100 Carry + 99 Kick motions
bash scripts/sugar/demo_reward/collect_deployable_goal_core_corpus.sh \
  experiments/demo_following/contact_event_reward_redesign_v1/reproduction_goal_core_corpus

# GPU compute node：构建 phase-aware targets、训练、校准并冻结 dense reward scale
$PYTHON_BIN scripts/sugar/demo_reward/build_actual_contact_event_predictor_dataset.py \
  --corpus-root experiments/demo_following/contact_event_reward_redesign_v1/reproduction_goal_core_corpus \
  --output-dir experiments/demo_following/contact_event_reward_redesign_v1/reproduction_phase_dataset \
  --policy-observation-key goal_policy_core_observation --alignment-mode clock_phase
$PYTHON_BIN scripts/sugar/demo_reward/train_actual_contact_event_predictor.py \
  --dataset-root experiments/demo_following/contact_event_reward_redesign_v1/reproduction_phase_dataset \
  --output-dir experiments/demo_following/contact_event_reward_redesign_v1/reproduction_phase_seed271303 \
  --epochs 20 --seed 271303 --device cuda:0
$PYTHON_BIN scripts/sugar/demo_reward/calibrate_actual_contact_event_predictor.py \
  --dataset-root experiments/demo_following/contact_event_reward_redesign_v1/reproduction_phase_dataset \
  --predictor-dir experiments/demo_following/contact_event_reward_redesign_v1/reproduction_phase_seed271303 \
  --device cuda:0
$PYTHON_BIN scripts/sugar/demo_reward/audit_deployable_demo_event_reward.py \
  --corpus-root experiments/demo_following/contact_event_reward_redesign_v1/reproduction_goal_core_corpus \
  --dataset-root experiments/demo_following/contact_event_reward_redesign_v1/reproduction_phase_dataset \
  --predictor-dir experiments/demo_following/contact_event_reward_redesign_v1/reproduction_phase_seed271303 \
  --output-dir experiments/demo_following/contact_event_reward_redesign_v1/reproduction_reward_scale \
  --unrelated-motion-id 21 --device cuda:0

# GPU compute node：复现完整 G1 CarryBox 与双手 27-patch 在线触觉视频
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox
```

当前保留视频（标为“旧 phase-0”的两条是时钟错位负结果，不是 corrected policy endpoint）：

- [Prefix41 安全约束：baseline 与 update64 真实世界对照](experiments/demo_following/cross_skill_recovery_prefix41_safe_v1/videos/matched_baseline_vs_learned_recovery_actual_world.mp4)；
- [Prefix41 无安全 penalty：位移增加但跌倒恶化](experiments/demo_following/cross_skill_recovery_prefix41_v1/videos/matched_baseline_vs_learned_recovery_actual_world.mp4)；
- [Carry-9→Kick：零更新与 update64 真实世界对照](experiments/demo_following/cross_skill_recovery_v1/bcppo_frozen_eval_seed181629/videos/matched_baseline_vs_update64_actual_world.mp4)；
- [official router：Carry45 reference 与 matched Carry](experiments/demo_following/official_tracker_router_v1/seed161610/videos_reference_actual_final/01_carry_domain_carry45_condition.mp4)；
- [official router：Kick21 condition 在 Carry 域的 action-limit failure](experiments/demo_following/official_tracker_router_v1/seed161610/videos_reference_actual_final/02_carry_domain_kick21_condition.mp4)；
- [official router：Carry45 condition 在 Kick 域仍受 generator 主导](experiments/demo_following/official_tracker_router_v1/seed161610/videos_reference_actual_final/03_kick_domain_carry45_condition.mp4)；
- [official router：Kick21 reference 与 matched Kick](experiments/demo_following/official_tracker_router_v1/seed161610/videos_reference_actual_final/04_kick_domain_kick21_condition.mp4)；
- [action-direction 诊断：Carry45 条件与稳定搬运](experiments/demo_following/shared_topology_distillation_v1/seed161593/videos_fixed_carry_teacher_step3000/01_correct_demo_and_actual_behavior.mp4)；
- [action-direction 诊断：Kick21 条件与失败/摔倒行为](experiments/demo_following/shared_topology_distillation_v1/seed161593/videos_fixed_carry_teacher_step3000/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [共享 checkpoint：Carry45 输入与实际行为](experiments/demo_following/shared_actionable_demo_conditioning_v1/seed161591/videos_same_checkpoint_update0064/01_correct_demo_and_actual_behavior.mp4)；
- [共享 checkpoint：Kick21 输入与实际行为](experiments/demo_following/shared_actionable_demo_conditioning_v1/seed161591/videos_same_checkpoint_update0064/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [新 reference-aware correct demo 与冻结实际行为](experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/videos_update0064_trace_exact/01_correct_demo_and_actual_behavior.mp4)；
- [新 reference-aware unrelated Kick demo 与冻结实际行为](experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/videos_update0064_trace_exact/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [独立复现 correct demo 与冻结实际行为](experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161589/videos_update0064/01_correct_demo_and_actual_behavior.mp4)；
- [独立复现 unrelated Kick demo 与冻结实际行为](experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161589/videos_update0064/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [4x correct demo 与冻结实际行为](experiments/demo_following/matched_phase_event_reward_reference_aware_4x_overfit_v1/seed161589/videos_update0064/01_correct_demo_and_actual_behavior.mp4)；
- [4x unrelated Kick demo 与冻结实际行为](experiments/demo_following/matched_phase_event_reward_reference_aware_4x_overfit_v1/seed161589/videos_update0064/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [旧 phase-0 correct Carry45 demo 与实际 Carry 行为](experiments/demo_following/matched_phase_event_reward_v1/seed161587/videos_update0064/01_correct_demo_and_actual_behavior.mp4)；
- [旧 phase-0 unrelated Kick21 demo 与实际 Carry 行为](experiments/demo_following/matched_phase_event_reward_v1/seed161587/videos_update0064/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [correct CarryBox demo 与实际行为](experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161581/videos_update0064/01_correct_demo_and_actual_behavior.mp4)；
- [unrelated KickBox demo 与实际行为](experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161581/videos_update0064/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [第三个 seed 的 correct demo 与实际行为](experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161585/videos_update0064/01_correct_demo_and_actual_behavior.mp4)；
- [第三个 seed 的 unrelated demo 与实际行为](experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161585/videos_update0064/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [teacher-floor correct demo 与坍塌行为](experiments/demo_following/teacher_floor_overfit_v1/seed161581/videos_update0128/01_correct_demo_and_actual_behavior.mp4)；
- [teacher-floor unrelated demo 与坍塌行为](experiments/demo_following/teacher_floor_overfit_v1/seed161581/videos_update0128/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [6x、mu=1.5 的 G1/CarryBox 与双手 27-patch](experiments/online_patch_tactile_mass_adaptation/visualizations/official_refiner_mu1p5_6x_friction_hold_single_env/official_refiner_mu1p5_6x_world_bilateral27.mp4)。

## 活动目录

```text
README.md                         当前结论、贡献和最短入口
DOCS/reproducibility.md           单一完整复现与证据记录
PLAN/README.md                    当前 demo-following 决策与下一实验
PLAN/15_.../plan.md               冻结的触觉/质量突变协议
TODO/README.md                    当前执行队列
TODO/15_.../todo.md               冻结的 Plan 15 历史清单
scripts/sugar/demo_following/     当前 same-teacher 训练、评估和视频入口
scripts/sugar/native_tactile/     在线整手触觉、泄漏和物理可行性入口
experiments/                      本地最小证据包，不提交
legacy/                           失败、混杂、重复和过期内容，不提交
```

`experiments/`、checkpoint、trace、dataset、视频和 runtime log 均为本地证据，不进入
Git。当前实验目录索引见 [experiments README](experiments/README.md)。

## 下一步

冻结 Carry-prefix 扫描以及 prefix41 的两组 matched recovery 已完成，并否决现有 endpoint
BCPPO fine-tuning 路线。当前下一项是保留 released Carry/Kick endpoint 能力，学习 serious
shared skill prior 与 state-aware transition objective。不得继续增加 optimizer steps、调同一
penalty/reward scale，也不得用 hand-written toy latent/world model 代替官方实现。

旧 matched 64-update comparison 已完成，结果为稳定 Carry、无 semantic separation；exact-prefix
scorer ablation 已证明在线语义倒置由 nonzero-reference phase 初始化错误直接造成，并已修复
正式 runtime 接口。修正后的 online smoke 与 frozen Carry gate 已正式通过；motion-disjoint
official Generator/Tracker Kick gate 也以 `8/9` profile preference 通过，但 motion29 是明确
反例。官方发布物当前只有 KickBox
`generator.ckpt + tracker.pt`，没有 frozen Kick Refiner checkpoint；因此现有 gate 已覆盖最强
可忠实复现的官方 inference 路径，但不冒充 Refiner 结果。第一组从零 reference-aware matched
pair、独立 seed 复现和固定 4x 诊断均已完成。1x 在 update 64 的同一 `3/4` 小幅变化可重复，
4x 却退化为 `1/4`，所以问题不是单纯 reward 太小。共享 MLP 的 full-510D BC 和三阶段
DAgger 也未保持闭环稳定。新的 official-skill router 已让一个 checkpoint 在 matched 域中
分别达到 Carry `18/20`、Kick `20/20` 且零跌倒，但 condition-only counterfactual 证明它仍有
task-generator 耦合和跨域动作爆炸。下一步应使用 serious shared skill prior/latent 与
state-aware safe transition policy，在保留 released expert 执行能力的同时逐步解除 Generator/
Tracker 任务耦合。不得继续 reward-scale/optimizer-step sweep，不得把 future action label 放入
deployed actor，也不得用 toy teacher/MLP 替代 official SUGAR 组件。SMP 只有在官方实现和
selected-demo latent deviation 真正接通后才能称为集成。
