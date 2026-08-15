# Plan 15: Online Whole-Hand Patch Tactile for Sudden-Mass Adaptation

## 1. 核心问题

在完整 SUGAR G1 已经抬起 CarryBox 并进入稳定持物阶段后，保持箱子的
几何、外观、位姿参考和接触目标不变，在 IsaacLab/PhysX 中在线提高箱子
质量。接触 binary 可能仍然全为 `1`，RGB 在质量切换瞬间没有变化；新的
整手触觉应通过每个 patch 的压力、剪切/摩擦负载及滑动状态更早反映负载
变化。要回答的不是“网络是否读到了触觉”，而是：

> 在线整手触觉是否使同一个 serious SUGAR policy 在突然变重后更快采取
> 有效动作，并提高持稳、恢复或安全放下的物理结果？

Plan 15 是唯一活动实验计划，优先级高于 Plan 14 及所有历史训练、demo、
ICM、RGB、Newton 和软体任务。实现和实验必须严格按本文的串行顺序进行。

## 2. 必须先修正的科学前提：质量并非天然“只有触觉可见”

官方 Refiner 的 `890-D` observation 包含 measured object state，例如
`obj_lin_vel_b`，也包含 `joint_pos`、`joint_vel`。质量改变后，箱子速度、手臂
下沉和关节跟踪误差都可能泄漏质量信息。因此不能把实验预设成“只有触觉能
感知变重”。处理方式如下：

1. **部署 actor 禁止 measured object state。** 正式 actor 使用已有的
   `504-D` deployable Tracker-command/proprioception contract：`35-D` 官方命令，
   五帧机器人本体感受/上一动作历史，当前 base linear velocity 和 phase。
   `obj_pos/ori/lin_vel/ang_vel`、真实质量、质量倍率、jump flag、RGB、刚体
   ContactSensor 和 simulator relative velocity 均不得进入 actor。
2. **`890-D` 仅限训练期。** 官方 Refiner 可作为 frozen teacher，privileged
   critic 可使用官方 object state；二者在三个训练分支中完全相同，不能被
   描述成部署输入。
3. **保留真实机器人需要的 proprioception。** 不删除 `joint_pos/joint_vel`
   来人为制造“触觉独占”。正式结论比较 tactile 相对于 proprio-only baseline
   的增量收益和反应时延。
4. **训练前做时间分辨率泄漏审计。** 对固定动作、相同初态和相同 jump 的
   rollout，分别从 measured-object、robot-proprio、patch-tactile 和 slip
   四组信号预测质量倍率/变化时刻，并报告每组信号在 jump 后何时首次可靠
   可分。linear probe 只作为泄漏诊断，不是 policy，也不算触觉收益。

如果 proprioception 与触觉同样早地暴露质量，实验仍可继续，但只能声称
“触觉在已有本体感受之上是否提供增量帮助”；不得声称质量只有触觉可见。
如果 measured object state 意外进入 actor，必须先修正 observation contract，
不得开始正式训练。

## 3. 在线质量突变场景

### 3.1 物理条件

- backend：仅 IsaacLab/PhysX；
- robot：完整 SUGAR G1；
- object：当前官方 CarryBox，自由动态刚体；
- tactile：每手 27 个 physical anatomical TacSL patches，共 54 个；
- nominal mass：`m0 = 0.3023375869 kg`；
- sweep：
  - `1.5x = 0.4535063804 kg`；
  - `3x = 0.9070127607 kg`；
  - `6x = 1.8140255214 kg`；
  - `10x = 3.0233758690 kg`。

先用无学习 rollout 检查四个倍率的物理可恢复性。`10x` 如果在任何可执行
动作生效前就必然失稳，仍保留为极端失败条件，但主要正向结论只能基于物理
上可恢复的倍率；极端条件评价安全响应或受控放下，不能把必然失败算成感知
失败。

### 3.2 在线 Refiner 持箱交接与 jump 顺序

旧版从 frame 0 立即让 student 控制整段抓取。两个独立 3000-update Z endpoint 的
8 条 `1.5x` profiles 全部在箱体接触前终止；额外 `1.0x` nominal 四条与同 seed 的
`1.5x` 逐帧终止点完全相同，均为 `68/68/65/85`。因此这是抓取入口失败，不是质量
适应失败，旧 checkpoint 不再进入 Z/P/PS 收益比较。

新版每个 episode 仍从 motion 45/frame 0 在线连续运行，但先由 exact frozen
official Refiner 控制同一个完整 G1。箱子相对 reset 高度连续 10 个 control frames
保持至少 `0.05 m` lift 后，下一次控制边界把动作权交给待训练 actor。交接前后必须
是同一个 IsaacLab/PhysX episode、同一个机器人和箱子、同一份连续更新的触觉历史；
禁止 teleport、中段状态恢复、离线 replay 或重置 sensor/slip/上一动作历史。

交接是实验初始条件生成器，不是部署 actor 输入：teacher 的 `890-D` privileged
observation、`handoff_active` 和 hold gate 都不得进入 actor。交接前 transition 不参与
PPO surrogate、value 或 entropy credit，避免把“actor 采样但 teacher 执行”的区间
误作 on-policy 数据；官方 teacher distillation target 保持不变。P/PS 在 teacher 前缀
中照常在线更新四帧 patch/slip 历史，Z 全程保持零 sensor read。

mass scheduler 从 hold qualification/actor handoff 后再随机等待 `10--50` 个 control
frames。双手 TacSL contact 不参与触发；P/PS live trace 仍须独立证明 jump 前连续
10 帧双手都有 patch contact。随后：

1. actor 在 nominal mass 下输出当前动作；
2. 在两个 control actions 之间，用 PhysX runtime API 同步更新 box mass 和
   inertia，并读回实际质量；
3. 用新质量完成后续 physics substeps；
4. 官方 TacSL sensor 在 physics 后在线更新；
5. 下一次 actor call 读取新时间戳的 patch observation 和 slip state。

质量倍率、jump 时间和 jump flag 只写诊断记录，绝不进入 actor。对象 USD、
mesh、材质、颜色、相机参数和 reference motion 均不改变；不重置物体、机器人、
触觉历史或上一动作。随机 jump 时间防止 policy 只根据 phase 提前背答案。
训练条件同时保留 no-jump nominal episodes，避免所有策略无条件用最大握力。
为使离线泄漏分析拥有完全相同的对齐时钟，`1.0x` 条件在相同阶段记录一个
placebo event；该事件不调用 PhysX mass/inertia write，实际质量始终保持 nominal，
并以独立的 `mass_changed=false` 诊断字段与真实 jump 区分。

## 4. Policy 的 54-patch 在线触觉合同

### 4.1 Patch 是唯一 policy 单元

官方 `VisuoTactileSensor` 仍以 R15 taxel 计算真实 penetration、normal force 和
signed XY shear；raw taxel tensor 可以保留用于传感器正确性与可视化审计。
但是 **taxel 不得成为 policy 维度、token 或空间单元**。每个 control step 在
GPU 上将每个 physical patch 的 raw taxels 在线归并成一个 patch record。

固定顺序为左右手各 27 个：

- palm `4 x 3`：12 patches；
- thumb/index/middle/ring/little：每指 proximal/middle/distal 3 patches。

每个 patch 的基础特征固定为：

1. `contact`：该 patch 内是否存在官方 TacSL 的正 penetration/有效接触；
2. `normal_load_n`：方向校正后的 compressive normal force 总和 `[N]`；
3. `mean_pressure_pa`：`normal_load / physical_patch_area` `[Pa]`；
4. `shear_x_n`：patch-local X 的 signed shear 总和 `[N]`；
5. `shear_y_n`：patch-local Y 的 signed shear 总和 `[N]`；
6. `friction_utilization`：`||shear_xy|| / (mu * normal_load + eps)`。

`contact` 是由官方 TacSL patch 内部信号在线派生的 patch feature，不得换成
SUGAR 原有 `hands_contact_label` 或普通刚体 ContactSensor binary。压力与剪切
保留物理单位和符号；归一化使用训练前冻结的公共尺度，三个分支完全一致。

### 4.2 Slip 输出

slip callable 为每个 patch 再提供：

7. `slip_score`：连续、因果的滑动证据；
8. `incipient_slip`：是否进入初始滑动；
9. `gross_slip`：是否进入明显滑动。

因此 actor 的触觉张量为：

```text
[batch, history=4, hand=2, patch=27, channel=9]
```

总共是四帧、54 个 patch token、每 token 9 个值；不是
`54 x 20 x 25` taxel grid。所有分支使用相同张量宽度。zero 分支不读取 sensor，
并在 observation 和 encoder output 两处保持 exact zero。

### 4.3 Patch encoder

使用一个正式的 anatomical patch-token encoder，而不是把 1944 个标量直接
塞入临时小 MLP：

- 共享的 `9 -> 128` patch projection；
- 固定 anatomical patch identity、left/right hand 和四帧 time embeddings；
- 3 层 Transformer encoder，`d_model=128`、4 heads、FFN width 256；
- masked attention pooling 得到 `128-D` tactile embedding；
- embedding 接入已有 SUGAR `512/256/128` actor，输出原有 `29-D` action。

结构在任何 policy outcome 出现前冻结，三个分支完全相同。全零历史必须显式
映射为 exact-zero tactile embedding，避免 token embedding/bias 让 zero arm 获得
伪信号。不得用 taxel CNN、离线 classifier 或小型替代 policy 冒充正式实验。

## 5. IsaacLab causal slip callable

实现一个可在 IsaacLab observation term 中直接调用的 stateful function：

```python
PatchSlipDetector.update(
    contact,
    normal_load_n,
    mean_pressure_pa,
    shear_xy_n,
    friction_utilization,
    timestamp_s,
    reset_mask,
) -> PatchSlipOutput
```

要求：

- 输入/输出 batch shape 始终以 `2 x 27 patches` 为空间单位；
- 只读取当前和过去的 patch contact、压力、signed shear、摩擦利用率和时间戳；
- 根据 shear/normal 比、压力下降、shear/pressure 时间变化和 contact loss 维护
  `NO_CONTACT/STICK/INCIPIENT/GROSS` 状态；
- episode reset 必须清除相应 env 的历史，不能串 episode；
- 不读取 object pose/velocity、relative contact velocity、SDF normal、reward、
  mass factor、jump flag 或未来帧；
- simulator relative tangential velocity 只能用于离线评价 detector precision、
  recall 和 detection delay，不能进入 callable。

训练前必须在 live IsaacLab stick-to-slide 与 CarryBox post-jump rollout 上调用同一
function，确认 state、timestamp 和 patch 位置都对应。保存的 trace 只用于复查；
正式 policy observation 必须来自当前 simulation step，禁止离线 replay。

## 6. 泄漏审计

先用 frozen nominal controller 记录一条 nominal action sequence，再把同一 sequence
开环重放到 no-jump 和四种 mass jump；保持动作、初态、jump 时刻和随机种子配对，
避免 privileged controller 根据 object state 改动作而污染泄漏比较。以 jump 前
`0.5 s` 到 jump 后 `1.0 s` 为窗口，分别检查：

- `object-state`：直接记录 Refiner privileged contract 中的 `obj_pos_b`、
  `obj_ori_b`、`obj_lin_vel_b` 和 `obj_ang_vel_b`，仅作为泄漏上界；
- `proprio-only`：正式 actor 的 robot state/history，不含触觉；
- `patch tactile`：54-patch 基础特征；
- `patch tactile + slip`：基础特征和 callable 输出。

每组报告原始变化幅度、质量倍率 linear-probe balanced accuracy 和 change-onset
latency。训练准入条件只有三个：

1. actor tensor 中 measured object state、mass 和 jump flag 为零项；
2. 在线 patch normal load/pressure 对可恢复 mass factor 呈可解释变化，而不是
   仍然只有 binary contact；
3. slip callable 在真实滑动区间有非零、时序正确的响应。

泄漏审计决定最终 claim 的措辞，不用于挑选有利 seed。若 tactile 在 live jump 后
完全无响应，先修 sensor/归并/物理，不得靠训练掩盖。

## 7. 匹配训练实验

### 7.1 三个分支

严格串行运行，每次只训练一个分支：

1. `Z / proprio-only`：patch 与 slip tensor exact zero，且不读取 sensor；
2. `P / patch tactile`：读取 live contact/normal pressure/signed shear/friction，
   slip 三通道 exact zero；
3. `PS / patch tactile + slip`：读取相同 live patch signal，并调用 causal slip。

三者共享：actor/critic/teacher 架构、初始化、optimizer、BCPPO schedule、PPO reward、
mass/no-jump sampling、seed、physics、episode length 和 update budget。正式 actor
均为 `504-D base + 128-D patch embedding`；zero arm 的 embedding exact zero。

### 7.2 Serious SUGAR training

- 使用已有官方 Tracker warm start、SUGAR `512/256/128` actor、官方 Refiner
  checkpoint 和 repository-native BCPPO；
- frozen Refiner teacher 与 privileged critic 只在训练中使用；
- task reward 可以读取 simulator object state 评价持稳、跌落、朝向和机器人
  稳定性，但不得把 mass ID/jump flag 作为 actor observation 或直接奖励答案；
- 三个分支原样共享官方 SUGAR CarryBox reward，包括机器人/物体 reference tracking、
  action/torque regularization 和 training-only hand-contact term。后者是相同的
  mass-independent task reward，不是 actor tactile input；Plan 15 不新增按质量给分
  或按 slip state 给分的 reward；
- 不加入 RGB、demo internal reward、ICM、T-Rex、actor contact proxy 或离线 tactile；
- full 29-DoF action 保持可用，policy 可以加强握持、降低身体、改变双手受力或
  安全放下，具体反应不手工脚本化。

每个分支先做一次 live one-update preflight，随后使用相同的 `3000` update budget。
非零双手触觉和质量/惯量 event 的物理准入由已完成的连续动作 full-G1 collector
提供；one-update preflight 只要求 Z 的 exact-zero/no-read，或 P/PS 的在线 54-patch
read 以及 PS callable。它仍如实记录 contact/load/event；早期 policy 尚未进入抓箱
窗口时不伪造接触或提前 jump。preflight 与正式训练都原样读取 released Tracker 的
exploration standard deviation。
这是 repository-native BCPPO schedule 所需的完整阶段：updates `0--499` 为 teacher
distillation，`500--999` 加入 critic warmup，`1000--1999` 将 task-reward PPO
authority 从 0 线性升到 1，`2000--2999` 保持 full PPO authority，同时三个分支
统一保留 `stage3_distill_weight_floor=0.25` 的 Refiner BC anchor。这个 floor 是仓库
已有 BCPPO 参数，只约束训练期 student 不遗忘持箱行为，不把 teacher action、物体
状态或质量信息加入部署 actor。原先的 512-update
草案在 actor 收到 task-reward PPO 之前就结束，不能回答触觉是否帮助训练，已撤销。
泄漏审计固定使用 `150814/150815/150816`；正式训练固定使用
`151014/151015/151016`；frozen evaluation 固定使用
`152014/152015/152016`。不得看到结果后延长单一分支或更换 seed。训练分布平衡
采样 no-jump、`1.5x/3x/6x/10x` jump；如果 feasibility 阶段
确认某倍率物理不可恢复，该倍率仍保留为 safe-failure evaluation，但不主导
hold-success reward。

Frozen evaluation 将三个训练 checkpoint 与三个未参与训练的 seed 一一固定配对：
`151014 -> 152014`、`151015 -> 152015`、`151016 -> 152016`。每对 checkpoint/seed
在每个 factor 上跑 20 个连续 profile；factor 固定而 jump delay 由对应 evaluation
seed 在 `10--50` 帧内确定，因而每支共 `3 x 5 x 20 = 300` 个 matched rollouts。
不得让每个 checkpoint 再遍历全部三个 evaluation seeds，否则会无意变成每支 900
个 rollouts；任何分支也不得单独补 profile。
所有 profile 固定使用官方 CarryBox motion 45 并从 frame 0 连续开始；默认 play
reset 的随机 motion time 不得进入该比较。

代码入口固定为三个 process-local Z/P/PS preflight task 和三个对应 formal task。
所有入口复用同一个 runner 配置；启动器必须读取 live sweep 生成的 9-channel
scale JSON，禁止用猜测常数代替真实在线归一化尺度。该接线完成只代表训练路径
已准备好，不越过 leakage/slip/live-physics 准入顺序。启动器自动绑定 official
Refiner teacher 和 official Tracker warm start，formal seed 只能来自冻结的三项；
中断恢复以 3000 为总 endpoint 计算 remaining updates，而不是重新多跑 3000。

## 8. Frozen-policy 测试与判据

训练结束后冻结 policy，在未参与训练 rollout 的 paired seeds 和随机 jump 时间上
测试。每个 mass factor 单独报告，不只给总平均。

主要物理指标：

- jump 后指定窗口内的 hold success、drop rate 和 robot fall rate；
- 箱子高度损失、orientation error、恢复时间和安全放下成功率；
- actor action 首次显著改变的 causal latency；
- patch pressure/shear 重分配、incipient/gross slip duration；
- no-jump nominal performance，检查触觉策略是否无条件过度用力。

冻结判据使用 jump 后连续 `80` 个 control frames（`1.6 s`），在看到分支结果前固定：

- `hold_success`：该窗口完整存在，box 相对 jump 高度的最大下降不超过 `0.05 m`，
  且没有 robot `anchor_ori/anchor_pos/ee_body_pos` failure；
- `drop`：相对 jump 高度下降至少 `0.15 m`，或 box 高度回到 episode 初始高度
  `+0.03 m` 以内；
- `safe_lower`：不是 hold、不是 drop、没有 robot failure，box 受控下降到初始高度
  `+0.08 m` 以内，最大向下速度不超过 `0.35 m/s`，且相对官方 reference 的最大
  orientation error 不超过 `0.8 rad`；
- 如果 mass/placebo event 没发生，或剩余轨迹不足 80 帧，该 profile 保留并报告，
  但不进入 hold/drop/safe-lower 分母。双手 patch-contact fraction、gross-slip
  fraction、reference pose error 和 reward 作为连续量同时报告。

主要对比为 `PS - Z`；`P - Z` 判断 patch load 本身是否有用，`PS - P` 判断 slip
callable 是否提供额外帮助。正向结论要求 frozen physical behavior 的 paired
95% confidence interval 支持改善，并且 nominal no-jump 没有相应崩坏。encoder
gradient、训练 loss、reward predictor 或单个好视频都不能证明触觉有帮助。

统计方法在 endpoint 出现前固定为 paired hierarchical percentile bootstrap：先对
三个 training/evaluation seed pairs 有放回采样，再在被选中的 seed 内对 20 个
matched profiles 有放回采样，固定 `10,000` 次和 analysis seed `153015`。逐 factor
报告 event-window eligibility、hold-or-safe-lower、hold、drop、safe-lower、robot
fall、height loss、双手 contact fraction 和 gross-slip fraction；差值方向始终写为
前一分支减后一分支。需要 event 后完整窗口的 outcome 只在两支都 eligible 的 paired
profiles 上比较，同时单独报告 eligibility 差异，避免把接触前跌倒悄悄排除。

如果 PS 只在 action 上变化而不改善持稳/安全结果，结论是“policy 使用了触觉，
但未证明帮助”。如果 P 改善而 PS 不再改善，则分别报告 patch tactile 有效、当前
slip detector 无额外收益；不得把两者合并包装成正向结果。

## 9. 必须提供的可视化

每个正式 mass factor 至少给出一组同步 H.264，对比 `Z/P/PS` frozen rollout。视频
必须显示：

- 完整 G1 和 CarryBox 世界画面；
- 左右手各 27 个 patch 单元，显示 contact、pressure、signed shear 和 slip state；
- mass jump 的真实时刻与倍率作为**评价 overlay**，明确注明 actor 看不到；
- box height/orientation、action-response latency 和 drop/fall 状态。

## 10. 2026-08-14 当前执行状态

- mass/inertia action-boundary controller、54-patch online reducer、causal slip
  callable、anatomical Transformer 和 Z/P/PS BCPPO 入口已经实现；相关 39 个
  非仿真单元/结构测试通过。live collector 逐帧保存官方 TacSL
  `SensorBase._timestamp_last_update` 的双手 54-patch 时钟，并要求同帧同步且每个
  control frame 严格前进。
- H200 上已确认 official Tracker 的 zero-patch action 映射误差为
  `1.31e-6`，live synthetic patch 可反向传播到 encoder；这只说明结构与梯度，
  不说明传感器在线、slip 正确或触觉有训练收益。
- 2026-08-14 已恢复真实 runtime：复用同一官方 G1 转换 USD，避免在场景构建中
  重复启动 URDF importer/renderer，随后通过 official AppLauncher 在 server13
  H200 上完成完整 G1、54 个 physical patches、GPU PhysX 和全部 manager 初始化。
  首个真实 control frame 已推进，54 个 official TacSL source timestamps 同步为
  `0.02 s`；这排除了“当前 TacSL/54-patch 场景根本不能运行”的解释。随后完整
  420 帧在线 preflight 通过：第 299 帧质量从 `0.3023376 kg` 变为
  `0.9070128 kg`，jump 前连续 10 帧双手均有 patch contact，54-patch 最大时钟偏差
  为零且每帧严格前进，箱子最高抬升 `0.7469 m` 后失持。CarryBox velocity-oracle
  对当前 slip callable 的 contact-supported precision/recall 为
  `1.000/0.9909`，median onset delay 为 0 帧；但该轨迹没有 incipient-oracle 样本，
  且多数接触已处于 gross sliding，因此它不能替代后续 controlled stick-to-slide
  校准。
- `3 seeds x 5 factors` 的 fixed-action paired leakage sweep 已完整结束。15 条
  轨迹的 applied action 最大误差为 `0`，paired event frame 最大误差为 `0`；所有
  mass readback 分别正确落在 `0.3023/0.4535/0.9070/1.8140/3.0234 kg`，每条轨迹
  event 前 10 帧均为双手接触，54-patch clock 全程同步并在线前进。三个 seed 的
  nominal 最大抬升均值为 `0.7464 m`，随后 `1.5x/3x/6x/10x` 分别降为
  `0.6662/0.6360/0.6297/0.6276 m`；这是相同 nominal action 下的失败严重度，不能
  单独证明 `6x/10x` 在改变动作后仍不可恢复。
- 在完全相同的 seed、frame-299 jump、420-frame live collector 下，training-only
  frozen Refiner 的 closed-loop feasibility 形成清晰分界。`1.5x` 在 jump 后
  `118/121` 帧仍有双手 patch contact，物体没有下沉且继续抬升；`3x/6x/10x`
  分别在 frame `354/321/315` 后失去双手 contact，并从 jump 高度下降
  `0.213/0.222/0.226 m`。因此 `1.5x` 已证明存在物理可恢复窗口，`3x+` 是该
  frozen Refiner 的失败区，但仍不能据此断言任何更强动作都不可恢复。预先固定的
  stronger-grip/lower-posture 响应随后也在绝对 frame 300 执行：双肩向内各
  `0.10 rad`，双髋/膝/踝使用完全相同的轻微降姿目标，且不读取 mass、jump、
  tactile 或 object state。它没有恢复 `6x/10x`；双手接触结束帧为 `320/315`，
  相对无响应的 `321/315` 没有改善，箱子仍分别下降 `0.171/0.213 m`。因此这个
  简单固定策略不足，但训练 policy 仍可学习其他 29-DoF 响应。
- 质量 event 当帧，三个 seed 的所有倍率相对 nominal 都保持完全相同的 patch
  contact binary，但连续 patch load/pressure 已经变化。这直接验证了 binary 无法
  表达的负载信号；该响应并非随质量严格单调，因此它是 policy feedback，不应被
  宣称为直接质量计。与此同时 `504-D` proprio 当帧也已变化，故最终科学问题固定
  为“whole-hand patch tactile 在本体感受之上是否带来增量帮助”，不再使用“只有
  触觉能感知变重”的表述。
- 三 seed leave-one-seed-out linear probe 的首个连续可靠质量分类时刻为：evaluation-
  only object state `11` 帧、proprio `35` 帧、patch tactile `13` 帧、patch tactile +
  slip `11` 帧（50 Hz）。这提示触觉可能比 proprio 更早提供可分信息，但样本只有
  三个 paired seeds，只作为训练前泄漏诊断，不是触觉收益结论。
- 完整 6300 帧 CarryBox slip 评价表面上得到 contact-supported precision/recall
  `0.9992/0.9904` 和 median delay `0`，但旧阈值把 oracle 的 14 个 STICK samples
  全判成 GROSS。受控官方 R15 trace 证明原因是正常加载运动时 friction utilization
  已饱和，不能单独作为 GROSS 条件。修正后的 callable 仍只读取当前/历史 patch
  pressure、signed shear、friction 与时钟：friction utilization 只触发 INCIPIENT，
  GROSS 需要连续两个高 shear-rate 或 pressure-drop sample，有载接触丢失仍为 gross
  alert。240 帧独立物理 trace 对静止、`0.006 m/s` 慢滑、`0.03 m/s` 快滑和
  `0.01 m/s` 回程分别得到 STICK/INCIPIENT/GROSS/INCIPIENT；state confusion 为
  STICK `109/111`、INCIPIENT `109/109`、GROSS `19/20`，incipient 零延迟，gross
  延迟一帧（50 Hz 下 `0.02 s`）。simulator relative speed 只用于评价标签。该
  callable 随后完成独立 420 帧 full-G1 CarryBox `3x` live rollout：共有 107 帧
  双手接触，frame 328 mass event 前连续十帧双手接触，54-patch clock 零偏差且
  严格逐帧前进；contact-supported slip precision/recall 为 `1.0/0.9971`，median/
  p95 onset delay 为 `0/1` 帧，28 次有载接触丢失均触发 gross alert。真实 CarryBox
  中绝大多数 active contact 已经是 gross sliding，因此 fine-grained state 的依据仍
  以受控 trace 为主；binary slip 检测与在线接入已通过，可进入 Z/P/PS one-update
  preflight。Z 已完成一次 360-step BCPPO update：`364` 次 exact-zero observation、
  `0` 次 TacSL read，training-path report 通过。该 stochastic rollout 平均约 37 帧
  即因姿态偏差终止，未达到 lift gate，因此 event 仍为 0；没有通过取消 lift gate
  或提前改质量来制造通过。P 随后完成 `361` 次 online feature update、
  `19,494 = 361 x 54` 次官方 patch sensor read、`0` 次 slip call 和一次 BCPPO
  update，training-path report 通过；contact/load 仍如实为 0。PS 也完成 `361`
  次 online feature update、`19,494` 次 patch read、`361` 次 causal slip call 和
  一次 BCPPO update。三个 training-path preflight 均通过；失败的中间版本已移到
  根目录 `legacy/experiments/`。正式 Z 已按冻结 seeds 启动，P/PS 未启动。seed
  `151014` 与 `151015` 使用相同 4-env、24-step、3000-update 配置并行运行；这是
  同一 Z 分支内的 seed 并行，不是跨分支并行。两个 seed 均已越过 update 500；
  各自的 `model_500.pt` 均可完整读取，包含 patch Transformer、SUGAR actor/critic
  和 58 项 Adam optimizer state，且训练已经从纯 distillation 进入 critic warmup。
  该 checkpoint 只证明正式训练、阶段切换与中断恢复路径成立，尚不能比较触觉
  收益。seed `151014` 随后在 update 651 被调度器以 `CANCELLED by 0` 终止，
  不是训练异常或主动释放；最后完整 checkpoint 为 update 500。它已在另一保留
  allocation 中从 `model_500.pt` 恢复，runner 明确重建 BCPPO `update_step=501`、
  从 iteration 501 开始。到达下一个完整 update-750 checkpoint 后，该训练子进程
  已精确停止并迁移到五天 retained job `238250`/`server23`；新的 runner 恢复
  BCPPO/learning iteration 751，总 endpoint 仍为 3000。
  seed `151015` 的 allocation 随后也被调度器在打印 update 784 后以
  `CANCELLED by 0` 终止；最后完整 checkpoint 是 update 750。训练日志没有
  Traceback/OOM。该 seed 已在 retained job `238355`/`server07` 从该 checkpoint
  精确恢复：BCPPO `update_step=751`、runner 从 iteration 751 开始、总 endpoint
  仍为 3000，剩余 2249 updates；未保存的 751--784 不计入完成进度。
- 两个 resumed Z seed 都已生成可完整读取的 `model_1000.pt`：iteration 为 1000，
  含 42 个 patch-encoder tensors、58 项 optimizer state，learning rate 为 `1e-5`。
  两者现已进入 update 1000--1999 的 task-reward PPO authority ramp；这只证明正式
  物理任务优化阶段已经开始，不是触觉收益证据。
- Frozen evaluator 现在按官方路径在 reset 后刷新 motion-relative command buffer，
  再把第一帧 policy observation 固定到 CarryBox motion 45/frame 0。用 Z 的
  update-1000 中间 checkpoint 做单 profile 预检时，策略到 frame 63 才因机器人
  姿态终止，说明旧的 frame-0 command-buffer 错位已修复；但它尚未接触箱子或触发
  mass event。因此该结果只验证 evaluator 起点，不能提前代表 update-3000 Z 结果。
- Evaluator 同时锁存每个 profile 的首次 termination：终止后的 actor observation
  不再进入有效统计，action 固定为零，并保存 `valid_frame`。update-1250 Z 的在线
  复核确认有效区间严格为 frame `0--63`、其后 action 全零且有限；该中间策略仍未
  接触箱子。下一次阶段检查固定在 PPO authority ramp 结束的 update 2000，避免
  形成中间 checkpoint 版本梯子。
- 与正式 sweep 相同的 `num_envs=4` 结构预检也已通过：四个 profile 的首次
  termination 为 frame `63/46/90/70`，对应有效帧独立锁存，termination 后 action
  全部 exact zero，输出仍保持 `[420,4,...]` 便于批次拼接。四条都没有到达箱子
  contact；这仍是 update-1250 阶段诊断，不是 Z endpoint 结果。
- seeds `151014/151015` 的 update-2000 checkpoints 均已完整可读，并进入最后 1000
  次 steady full-PPO。seed `151015` 随后正常完成全部 3000 次训练循环；训练器按
  零基 iteration 保存的正式终点是 `model_2999.pt`，其中有 59 个模型张量和 58 项
  optimizer state。seed `151014` 也已正常完成相同终点和相同状态计数；P/PS 未
  启动。固定的相同四个
  motion-45/1.5x profiles 在 `151015` update-2000 checkpoint 的 termination
  frame 为 `96/48/201/194`；相对 update-1250 的 `63/46/90/70`，两条轨迹明显延长，
  但四条仍为零箱体接触、零 mass event。后续 endpoint 负结果已经取代这项中间
  观察；不得把生存时间延长写成触觉或质量适应结果。
- 旧 frame-zero Z 的两个 3000-update endpoints 已完成冻结检查并共同给出结构性
  负结果：`1.5x` 共 `8/8` profiles 在箱体接触前 robot fall，bilateral patch contact
  与 jump event 都为零；seed `151014` 的 `1.0x` 四条与其 `1.5x` 四条终止帧完全
  相同。第三 seed `151016` 因此在 iteration 226 停止，仅终止记录的 child PGID，
  allocation 保留。下一步不是继续旧 Z 或启动 P，而是实现上述 live official-
  Refiner hold handoff，并重新完成三分支 one-update preflight。
- replacement handoff 实现和 Z/P/PS 三个 one-update preflight 已完成。每项均从
  motion 45/frame 0 由 frozen Refiner 在线拿箱，在同一 episode 完成 4 次无重置
  handoff、交接后 student control 和 2 次真实 mass change。Z 的 TacSL read 为 0；
  P/PS 各有 361 次在线 feature update、19,494 次官方 patch read 和 363 个双手接触
  env-samples；PS 实际执行 361 次 causal slip update。BCPPO mask 与 wrapper 计数
  逐步吻合，只给 142/143 个 post-handoff transitions PPO credit。该结果只准入新的
  formal Z，不是触觉收益结论，也不准入 P formal。
- replacement handoff-Z 的正式训练已开始。seed `151014` 的恢复
  完整文件为 `model_1500.pt`，其 allocation `238253` 在打印 iteration 1711 后被
  调度器外部 `CANCELLED by 0`；seed `151015` 的最后完整文件为 `model_2250.pt`，
  allocation `238620` 在 iteration 2339 后同样被外部取消。两份 checkpoint 均为
  59 个模型张量、58 项 optimizer state 且数值有限；未保存区间不计。8 小时
  backfill jobs `239105/server35` 与 `239106/server44` 已分别从 iteration
  1501/2251 恢复，checkpoint、BCPPO stage 和 optimizer learning-rate 同步均通过。
  seed `151015` 随后正常完成 `model_2999.pt`，59 个模型张量与 58 项 optimizer
  state 均有限；同一 retained job 已从官方 Tracker、iteration 0 启动 replacement
  seed `151016`。seed `151014` 当时已越过有限的 `model_2000.pt`；这是历史训练里程碑，
  已被下文 endpoint 审查取代，不构成质量适应结果，也不准入 P/PS formal。
- 2026-08-15 的 replacement-Z endpoint gate 发现 zero-floor BCPPO 会在最后 1000
  updates 遗忘 Refiner 持箱行为。seed `151014` 的 distillation loss 从 update 2000
  的 `0.3404` 升到 update 2999 的 `35.8202`；冻结时交接前 teacher/student action
  L2 从约 `0.9` 恶化到 `5.4--5.9`，三条成功 handoff 轨迹均约 7 帧后、mass event
  前失败。相同四个 profile 的 update-1000 checkpoint 能产生三次真实 `1.5x` jump，
  post-jump 生存 `18/21/60` 帧；update-2000 则为 `65/38/74` 帧。两者仍未达到冻结的
  80-frame eligibility，但证明当前失败是训练后期 behavior forgetting，不是 TacSL、
  mass scheduler 或 handoff wrapper 失效。正式设计因此在三个分支统一启用仓库已有
  `stage3_distill_weight_floor=0.25`，保留完整 3000-update budget 和 update 2000 后
  full PPO authority；Z 从各自 update-2000 checkpoint 重跑最后 1000 updates，旧的
  zero-floor update-2999 endpoint 不进入正式比较。
- Anchored-Z 首次双节点并发恢复在 runner 加载前均因远端默认 ground-plane USD
  没有生成 Plane prim 而退出；单节点复现了同一 `Stage.GetPrimAtPath(NoneType)`，
  排除 BCPPO floor。正式 training launcher 现与已通过的 frozen evaluator 一致，
  显式使用仓库本地 `sugar_ground_plane.usda` 和已转换的官方 G1 USD。该修改只消除
  外部 asset 加载失败，不改变 physics、observation、reward 或训练预算。
- 本地 ground plane 生效后的下一次启动继续暴露了 task import 时序：训练入口在
  设置 `SUGAR_DISABLE_TRAIN_DEBUG_VIS=1` 前已经导入 task，因而仍尝试下载远端
  `frame_prim.usd`。环境合同现已移到 task registration 之前，并与 frozen evaluator
  统一固定 TacSL/PhysX 参数、本地 calibration、关闭 debug marker 和 anatomy audit；
  这既移除远端 marker 依赖，也保证训练和冻结评测使用同一传感器物理合同。
- 修正后的 anchored runtime 已真实进入训练。seed `151014` 从 update 2000 恢复，
  日志确认 BCPPO step 2001、optimizer LR `1e-5`、distill floor `0.25`，并写出有限
  `model_2250.pt`（59 model tensors、58 optimizer states）；seed `151016` 从 750
  恢复并写出有限 `model_1000.pt`。job `239105` 在151014打印 iteration 2337后被
  调度器外部取消。随后只终止239106中151016记录的 child PGID，在同一 allocation
  优先恢复151014；该 job 又在 iteration 2304后被外部取消。未保存更新全部舍弃，
  当时的精确恢复点为 anchored151014/update2250 与151016/update1000；该调度快照已被
  下文完成的 `151014/151015` anchored endpoints 取代，不能再作为当前进度。
- anchored Z seed `151014` 已严格停在固定 `model_2999.pt`，共 3000 次 iteration，
  checkpoint 的 59 个模型张量和 58 项 optimizer state 均有限；末次 distillation
  loss 为 `2.0839`，distillation weight 为 `0.25`。评测 horizon 已从错误的 420
  修正到 450：handoff 约 frame 297、最大 delay 50、结果窗口 80，最晚必须覆盖到
  frame 427。修正后 camera-free `1.5x` 四条冻结 gate 得到 `2/4` 个完整 80-frame
  eligible hold：profile 0 在 jump 后 39 帧触发 `obj_pos`，profile 1 在 handoff 前
  触发 `ee_body_pos`，profile 2/3 完成窗口后分别在 frame 423/410 触发 `obj_ori`。
  旧 `1/4` 是 horizon 截断，不是 profile 2 的策略失败。交接前十帧 student/teacher action
  L2 为 `1.04--1.12`，显著低于撤回的 zero-floor endpoint `5.51--5.86`。这说明 BC
  anchor 修复了后期 handoff forgetting，但还不是触觉收益结论。
- 已用同一 checkpoint、同一 seed 和同一四-profile 批次录制 eligible profile 3。
  420 帧 H.264 可完整解码；frame 308 PhysX 质量从约 `0.302` 变为 `0.454 kg`，frame
  400 仍由 frozen policy 持续抬箱，并同步显示左右各 27 patch 的 load、pressure、
  signed shear 和 slip。该结果完成了 seed `151014` 的 endpoint 审查，并只准入一次
  额外 Z endpoint；没有准入 P/PS。若后续 endpoint 行为不满足人眼或数值要求，先用
  相同 serious SUGAR actor、live handoff 和在线物理做单一固定条件 overfit 诊断，
  再决定是否花费下一个 3000-update 正式预算。overfit 不计入正式 Z/P/PS 比较。
- 匹配 profile 0 负例也已录制并完整解码。它在 frame 337 增重，frame 396 终止前
  仍双手持箱、lift 约 `+0.823 m`，没有 drop 或 robot fall；未通过 80-frame gate
  是因为 59 帧后发生 reference-tracking termination，伴随最大位置/姿态误差约
  `0.208 m/0.806 rad`。因此当前缺陷是轨迹跟随和跨 profile 稳定性不足，不能写成
  “1.5x 增重后普遍拿不住”。
- 单独的 no-learning physical-continuation 诊断只关闭 `obj_pos/obj_ori` reference
  termination，仍保留 robot `anchor/ee` fall 条件。profile 2/3 均通过 80 帧；
  profile 0 在 frame 383 触发 `anchor_pos` 时仍双手接触且箱子无下降。这进一步说明
  当前弱点是整机/参考轨迹漂移，而不是增重瞬间抓力失效；该诊断明确不是正式结果。
- 人眼主视频对保存的原始 world frame 做固定裁剪，只移除同批相邻 env；目标 profile
  的物理、trace、时钟和数值均未重算。正例与负例主视频现在都只显示一台完整 G1。
- anchored seed `151014` 的 450-frame 四-profile 跨质量审查已完成：`1.0x/1.5x/3x/
  6x/10x` 的 eligible hold 分别为 `1/4, 2/4, 2/4, 0/4, 0/4`。`6x` 的 profile 2
  下落 `0.194 m`；`10x` 的 profile 0/2 下落 `0.165/0.216 m`。这给后续 P/PS 留下
  明确的重质量失败区间，因此不需要为了让 Z 全成功而做 overfit。四条样本只用于
  endpoint 准入和选择测试区间，不能写成单调质量效应或触觉收益。
- 五个条件的 jump frames 都为 `[337, none, 344, 308]`。`1.5x/3x/6x` 相对 nominal
  的 jump 前 action/object position 完全一致；`10x` profile 2 在 event 前最后两帧
  出现最大 `0.0146` action 与 `0.23 mm` object-position 闭环数值分歧，且质量读回仍
  为 nominal。这是独立 GPU rollout 的微小闭环非确定性，正式报告不得声称所有
  policy evaluation 都逐位相同。
- 同步可视化已固定为上方完整 G1/CarryBox world、下方左右各 27 个 patch；patch
  显示 pressure、signed XY shear、load 和 causal slip，不显示 taxel grid。Frozen
  evaluator 已加入指定 batch-profile 的同钟 world-camera 录制和 handoff overlay；
  anchored endpoint profile 3 的 420 帧 H.264 已完整解码并完成关键帧人眼检查。
- anchored Z seed `151015` 已严格完成并停在 `model_2999.pt`；59 个模型张量与 58 项
  optimizer state 均有限，没有运行第 3000 次之后的 update，也没有自动启动下一 seed。
  该 endpoint 的四-profile、450-frame 审查现在从同一 PhysX rollout 同时报告两类结果：
  原始 SUGAR reference termination 只作为 label；物理轨迹继续到 80-frame outcome
  window。`1.0x/1.5x/3x/6x/10x` 的物理 hold 为 `4/4,4/4,4/4,0/4,0/4`，物理 drop
  为 `0/4,0/4,0/4,4/4,4/4`，`10x` 另有 `1/4` robot fall；同批 camera-free 严格
  reference hold 为 `0/4,1/4,3/4,0/4,0/4`。这证明当前 Z 有清楚的温和成功/重质量
  失败区间，同时证明 reference 偏离不能等同物理失败。它仍只是单 checkpoint 的小
  审查，不是触觉收益。该审查随后准入并只准入最后一个 Z seed；P、PS 仍不自动
  启动。若人眼或数值审查后来否定行为，再先做固定条件 serious overfit。
- anchored Z seed `151015` 的正式冻结审查已完成：`1.0x/1.5x/3x/6x/10x` 各 20
  profiles，共 100 条 live PhysX rollout。物理 hold 分别为
  `20/20,20/20,16/20,0/20,0/20`，drop 为
  `0/20,0/20,0/20,20/20,20/20`，robot fall 为
  `0/20,0/20,0/20,4/20,3/20`。3x 中 4 条未达到 hold 的轨迹下沉约
  `0.054--0.078 m`，但没有达到 drop，因此把 3x 视为边界区。五个 `450 x 20`
  trace 全部有限，100 条质量读回和 jump 前连续十帧双手 patch contact 均通过。
  每个 profile 的 handoff 后延迟序列在五个条件中完全一致；live PhysX 的绝对
  handoff/event 时刻最多抖动 4 帧，不写成逐位相同闭环轨迹。该正式结果仍只属于 Z
  端点，不能证明触觉收益；它已提供温和/边界/重质量三个有效区域，所以当前不做
  overfit。第三 Z seed 后续按同一合同完成；P/PS 仍不自动启动。若后续人眼检查否定
  行为，再执行已声明的单条件 serious overfit。
- 正式评测前修复了多 batch evaluator：reset 在 inference mode 内执行；每批 reset 后
  清除 IsaacLab 遗留的 termination-reason latch；main 异常不再被 SimulationApp close
  吞成 exit code 0；每个质量条件必须同时写出 summary 和 trace 才能进入下一个条件。
  两批 8-profile 诊断和随后正式 100 条均未再出现第二批 frame-0 假终止。
- 当前只保留 job `239098`/`server44` 用于审查与渲染；`238934` 已不在运行，不能再写成
  retained。完成本轮审查不授权退出 `239098`。
- seed `151015` endpoint 的四级人眼证据已完成，且没有继续训练：`1.5x` 为完整持箱，
  `3x` profile 8 为持续双手接触但最大下沉约 `0.052 m` 的边界样本，`6x` profile 0
  为掉箱并随后整机失稳，`10x` 为明确掉箱。四条都是 450-frame H.264，上方只显示
  目标完整 G1/CarryBox，下方同钟显示左右各 27 patch，并已完成全帧解码。3x camera
  replay 与正式 profile 8 的 `0.0538 m` 下沉一致；6x camera replay 与正式 profile 0
  均为 drop，但掉箱后的 robot-fall label 因 live camera replay 的闭环扰动而不同，
  只报告稳定的物理类别，不声称逐帧确定性。这轮审查没有否定 endpoint，因此不触发
  overfit；继续冻结 seed `151016`、P 和 PS。
- anchored seed `151014` 的正式 frozen audit 也已完成，每个质量 20 profiles。每个
  factor 都有同一个 profile 1 在 handoff 前触发 `ee_body_pos@220`，这是 Refiner
  prefix 覆盖失败，不属于 student 的 post-jump 结果。其余 19 条的
  `1.0x/1.5x/3x/6x/10x` 物理 hold 为 `19/19,19/19,16/19,1/19,0/19`，drop 为
  `0/19,0/19,2/19,18/19,19/19`，eligible robot fall 为
  `0/19,0/19,3/19,0/19,3/19`。所有 real event 的 mass readback、jump 前十帧双手
  contact 和跨 factor 匹配 delay 均通过。
- anchored Z seed `151016` 在 job `239098` 外部取消后只从完整 `model_2750.pt`
  恢复，严格从 iteration 2751 运行到固定 `model_2999.pt` 并正常退出；终点 59 个
  model tensors、58 项 optimizer state 均有限，没有更晚 checkpoint。其 100 条正式
  camera-free frozen audit 的 `1.0x/1.5x/3x/6x/10x` hold 为
  `20/20,20/20,20/20,0/20,0/20`，drop 为
  `0/20,0/20,0/20,20/20,20/20`，robot fall 全为 0。五个 trace 全部有限，100/100
  event 的质量读回和 jump 前十帧 bilateral contact 均通过。
- seed `151016` 的同步人眼审查已补齐：3x profile 0 camera rollout 完整持箱，6x
  profile 0 下落 `0.5619 m`；两条均为 450-frame H.264，上方完整 G1/CarryBox、下方
  左右各 27 patch，已全帧解码并检查 handoff/jump/drop 关键帧。视频只代表自身
  camera-enabled rollout，正式统计仍来自 camera-free trace。
- 三个完成 Z pair 合并后的 eligible hold 为
  `59/59,59/59,52/59,1/59,0/59`，drop 为
  `0/59,0/59,2/59,58/59,59/59`。该结果已经完整固定 mild/boundary/heavy 测试区间；
  训练停在 `model_2999.pt`，不额外加训，也不因重质量失败而 overfit。只有后续审查
  证明行为本身无效或含糊，才先做单一固定条件 serious overfit 诊断。
- world camera 会扰动临界闭环 outcome。seed `151014` 的 `6x` profile 7 在正式
  camera-free 轨迹与独立 camera-free repeat 中都以相同 `297/307` handoff/jump 持稳，
  但开启相机后下落 `0.279 m`；`3x` profile 0 则从正式 camera-free 的 drop+fall
  变为带相机 rollout 的 hold。3x 的新 camera-free 四-profile repeat 已以 exit code
  0 完成，23 个 trace 字段与正式 20-profile trace 的前四条全部逐值相同，profile 0
  精确重复 `0.452212 m` drop+fall。正式统计以无相机 trace 为准；同步视频只证明
  自身带相机运行，不再冒充正式轨迹的逐帧重放。
- 三个 Z endpoint 的 paired frozen reaction-window audit 已完成。每个 mass profile
  与同 seed/profile 的 1x trace 按 jump 对齐；patch 先按固定 channel scale 归一化，
  onset 定义为连续两帧超过 jump 前十帧 paired delta 上界。59 条 1.5x 均无 2 cm sag；
  3x 有 16 条达到 2 cm sag、2 条 drop；6x/10x 分别有 58/59 条 drop。全部 119 条 drop
  都先出现连续 load/pressure/shear/friction divergence，中位提前 21 帧；contact binary
  中位提前 15 帧，slip 为 118/119、中位提前 11 帧。normal load 与 pressure 分别都是
  119/119 早于 drop，中位 lead 20 帧，排除仅由 friction/contact-bit 驱动结论。在 133 条
  至少下沉 2 cm 的轨迹中，
  连续 patch 133/133 提前、中位 lead 7 帧，而 binary 只有 81/133 提前且中位 lead 为 3。
  因此连续触觉相对 binary contact 有真实提前窗口，但 Z action 在 117 条有可检测
  onset 的 drop 中已有 111 条先分叉，证明 proprio/闭环泄漏也足以改变行为。该 audit 只准入后续
  增量比较，不证明 P 或 PS 会成功，也不触发额外 Z 训练。
- reaction-window 主视频已经生成并全帧解码：450 帧、50 Hz、H.264/yuv420p、
  `1920 x 1080`。左侧是带相机的完整 G1/CarryBox/双手 27-patch 6x drop，右侧按同一
  时间游标激活 38 条正式 camera-free 6x drop 的 continuous/binary/slip/sag/drop
  marker。页脚明确说明两侧为不同 rollout，不能把左侧 world 画面冒充右侧正式 trace。

主图不得恢复为 20x25 taxel heatmap；taxel detail 只能作为单独 sensor debug。
所有分支使用相同视频尺寸、时钟、固定颜色尺度和 episode 区间。

## 11. 串行执行顺序

1. 实现 live mass/inertia jump 与 readback；
2. 实现 54-patch online reducer；
3. 实现并验证 IsaacLab `PatchSlipDetector` callable；
4. 完成 leakage audit，明确 actor/teacher/critic 边界；
5. 固定 task、reward、patch encoder、seeds 和 frozen evaluation；
6. `Z` one-update preflight 与 3000 updates；
7. `P` one-update preflight 与 3000 updates；
8. `PS` one-update preflight 与 3000 updates；
9. 三分支 frozen evaluation、同步视频和结论报告。

不得并行启动后续分支，不得在 leakage audit 前训练，不得恢复旧 RGB/demo/ICM/
Newton/soft-body 队列。GPU allocation 按 `AGENTS.md` 保留规则管理；结束一个子进程
不等于释放 allocation。

截至 2026-08-15 20:31，Z 三 seed 及其正式冻结评估均已完成。用户明确要求继续未完成
训练后，`P/seed151014` 已在 retained `240170/server44` 从官方 Tracker 启动；正式任务
为 `Sugar-G129dof-CarryBox-OnlineMass-Patch-P-BCPPO`，读取同 episode 的 live 54-patch
历史、保持 slip 字段为零，并固定在 3000 updates 停止。该 seed 完成后先做 endpoint
物理审查，不自动串联 `151015` 或 PS。

该运行在打印 iteration 1734 后由调度器把 job `240170` 标为 `CANCELLED by 0`；进程
没有训练异常，最后完整 checkpoint 为有限的 `model_1500.pt`。未保存 1501--1734
全部舍弃。随后在新获批五天 retained `231256/server64` 从该文件恢复：optimizer LR
为 `1e-5`，BCPPO `update_step=1501`，runner 从 iteration 1501 开始，固定总预算仍为
3000、remaining=1499。恢复不改变 P 的传感器、质量、reward、seed 或架构合同。
