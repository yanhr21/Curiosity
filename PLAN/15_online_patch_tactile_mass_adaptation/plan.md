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

### 3.2 jump 顺序

每个 episode 从 motion frame 0 连续运行，禁止把带 elastomer skin 的手直接
teleport 到中段接触状态。为了让 `Z` 分支完全不读取 TacSL，mass scheduler 只在
评价侧检查“箱子已连续抬升 10 个 control frames”，再随机等待 `10--50` 个
control frames。双手 TacSL contact 不参与触发；live preflight 必须独立确认 jump
前连续 10 帧双手都存在 patch contact，否则该 rollout 不准进入训练。随后：

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
这是 repository-native BCPPO schedule 所需的完整阶段：updates `0--499` 为 teacher
distillation，`500--999` 加入 critic warmup，`1000--1999` 将 task-reward PPO
authority 从 0 线性升到 1，`2000--2999` 为 steady full PPO。原先的 512-update
草案在 actor 收到 task-reward PPO 之前就结束，不能回答触觉是否帮助训练，已撤销。
泄漏审计固定使用 `150814/150815/150816`；正式训练固定使用
`151014/151015/151016`；frozen evaluation 固定使用
`152014/152015/152016`。不得看到结果后延长单一分支或更换 seed。训练分布平衡
采样 no-jump、`1.5x/3x/6x/10x` jump；如果 feasibility 阶段
确认某倍率物理不可恢复，该倍率仍保留为 safe-failure evaluation，但不主导
hold-success reward。

Frozen evaluation 对每个 Z/P/PS checkpoint 使用 seeds
`152014/152015/152016`。每个 seed、每个 factor 各跑 20 个连续 profile；factor
固定而 jump delay 由相同 seed 在 `10--50` 帧内确定，因而每支共
`3 x 5 x 20 = 300` 个 matched rollouts。任何分支都不得单独补 profile。

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

主要对比为 `PS - Z`；`P - Z` 判断 patch load 本身是否有用，`PS - P` 判断 slip
callable 是否提供额外帮助。正向结论要求 frozen physical behavior 的 paired
95% confidence interval 支持改善，并且 nominal no-jump 没有相应崩坏。encoder
gradient、训练 loss、reward predictor 或单个好视频都不能证明触觉有帮助。

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
  callable、anatomical Transformer 和 Z/P/PS BCPPO 入口已经实现；相关 31 个
  非仿真单元/结构测试通过。
- H200 上已确认 official Tracker 的 zero-patch action 映射误差为
  `1.31e-6`，live synthetic patch 可反向传播到 encoder；这只说明结构与梯度，
  不说明传感器在线、slip 正确或触觉有训练收益。
- 真正的 paired rollout/leakage audit 仍未开始。两块不同的 server13 H200 上，
  原始 collector、force-only 路径和曾成功的 camera/rendering 命令都在场景创建前
  遇到相同 `VK_ERROR_DEVICE_LOST`。CUDA 计算正常，故当前证据指向 Kit/Vulkan
  runtime 状态，而不是 patch、mass 或 slip 实现。全新 portable root 和显式关闭
  renderer multi-GPU 也复现同一错误。进一步将完整 `25.7 GB` Python/Isaac runtime
  复制到 server13 本地磁盘、排除共享文件系统读取后，最小 SimulationApp 仍在同一
  `Simulation App Starting` 边界报错，因此共享文件系统不是该 Vulkan 崩溃的根因。
  保留 jobs `238022/238055`；server01 job
  `238054` 和 server38 job `238092` 正在排队，任一启动后先跑跨节点 canary；在
  真实 physics step 恢复前不启动训练。

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
