# Plan 15: Online Whole-Hand Patch Tactile for Sudden-Mass Adaptation

> **Frozen 2026-08-22.** All listed source bugs were corrected, but the corrected tactile-only
> P continuation did not yield a valid endpoint: model1100 gives `14/20` physical holds but only
> `6/20` strict successes, and model1250 regresses. No matched corrected Z/P/PS comparison was
> completed. Preserve the artifacts, make no tactile benefit/harm claim, and keep this line outside
> the active queue while demo following remains the current evidence-selected priority.

## 1. 科学问题

在完整 SUGAR G1 已经抬起 CarryBox 后，保持几何、外观、参考轨迹和接触目标不变，
在线提高 PhysX mass/inertia。检验 live whole-hand tactile 是否在可部署 proprioception
之上带来更早、有效的控制反应，并提高持稳、恢复或安全放下的物理结果。

质量变化不是天然“只有触觉可见”：官方 Refiner `890-D` observation 包含 measured
object state，部署所需的 `joint_pos/joint_vel` 也会在受载后变化。因此正式 claim 只能是
触觉相对 `504-D` Tracker-command/proprioception 的增量收益。measured object state、
真实质量、mass factor、jump flag、RGB 和 future frame 不得进入 actor。

Plan 15 曾在 2026-08-20 invalidity audit 后从零重跑，现已冻结；当前优先级是
demo-following，不再自动继续 corrected comparison。

## 2. 物理场景与时序

- backend：IsaacLab/PhysX；
- robot：完整 SUGAR G1，29-DoF action；
- object：官方 CarryBox，自由动态刚体；
- nominal mass：`0.3023375869 kg`；
- conditions：`1.0x/1.5x/3x/6x/10x`；
- motion：official CarryBox motion 45；
- control rate：50 Hz。

每个 episode 的顺序固定：

1. exact frozen Refiner 从 frame 0 在线控制同一个机器人与箱子；
2. 箱子连续 10 个 control frames 抬升至少 `0.05 m` 后，下一控制边界交给 student；
3. 不 reset、teleport、replay 或清空 tactile/slip history；
4. handoff 后等待 matched `10--50` frames；
5. 在两个 actor calls 之间同步写入 mass/inertia，并读回实际质量；
6. 用新质量完成 physics substeps，更新 TacSL，再供下一 actor call 使用。

event clock 不读取 tactile。P/PS trace 另行要求 jump 前 10 帧持续双手 patch contact；
Z 全程不读取 TacSL。`1.0x` 使用匹配 placebo event，但不写质量。

## 3. 双手 54-patch 在线触觉合同

每只手固定 27 个 physical anatomical patches：

- palm `4 x 3`：12；
- thumb/index/middle/ring/little：各 proximal/middle/distal 3 个，共 15。

官方 TacSL R15 taxels 计算 penetration 和 penalty forces；每个 control step 在 GPU 上
在线归约成一个 patch record。修正后的法向载荷只来自 `F_normal`，signed XY shear 只
来自 `F_friction`，friction utilization 使用未丢失方向分量的 `|F_friction|` 幅值。
Plan 15 同时把 PhysX box/pad 与 TacSL 的摩擦系数固定为 `0.5`。独立 PhysX audit
把法向力和摩擦力分开读取。旧 `80621.56402207233/45` 标定只在一个 nominal 十帧聚合
窗口碰巧得到 `0%/2.90%` 误差；model-750 动态窗口的 TacSL/PhysX normal 总量中位比为
`2.01`，并漏掉 `18.5%` 的 PhysX pad force，因此该标定和对应 scale 已撤销。

进一步发现 released TacSL 的正方 taxel-spacing 规则映射到不同比例的 anatomical patch
后，仅覆盖各 patch 中央约 `44--66%`。本地 adapter 保留官方默认行为，但让 54 个
anatomical sensor 的 20x25 网格分别铺满两轴，并仅在边缘 ray miss 时自动增加 inset；
几何覆盖约提高到 `96--99%`。SDF 审计显示剩余 false-negative 都位于表面外
`0.001--0.505 mm`，因此 anatomical adapter 固定使用 `0.3 mm` compliant-layer offset。
最终 CarryBox 参数为 `kn=7294.8755, kt=9, mu=0.5`。nominal contact precision/recall 为
`0.910/0.967`、漏掉 PhysX pad force `0.86%`；独立 seed 的 3x post-jump recall 为 `0.892`、
漏力 `6.5%`，且 TacSL normal 总量约偏高 30%。这是已标定的模拟触觉，不是逐帧完全等于
PhysX 的力传感器。policy 单元永远是 patch，不是 taxel。

v3 gate 已完成 3 seeds x 5 mass factors 的 15 条 online traces。全部 jump 位于 frame 326，
质量读回正确且 jump 前十帧双手持续接触；slip precision/recall 为 `0.9914/0.9856`，507 个
onset 检出 501 个，median/p95 delay 为 `0/1` 帧。Z/P/PS runtime preflight 全部通过：Z
零 TacSL read，P/PS 各 `19,494` 次 live patch read，P 零 slip update，PS 361 次。v3 scale
为 `[1, 60.3778, 212561.67, 13.7072, 13.7072, 1, 2, 1, 1]`。

每个 patch 的 9 个通道为：

1. `contact`；
2. `normal_load_n`；
3. `mean_pressure_pa`；
4. `shear_x_n`；
5. `shear_y_n`；
6. `friction_utilization`；
7. `slip_score`；
8. `incipient_slip`；
9. `gross_slip`。

actor 输入形状为 `[batch, history=4, hand=2, patch=27, channel=9]`。公共尺度由训练前
的 live leakage sweep 冻结。不得使用 `hands_contact_label`、普通 ContactSensor、
离线 trace、生成值或 object state 替代。

## 4. Causal slip callable

唯一训练期 slip 接口为：

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

它只读取当前/过去的 54-patch 信号、时间戳和 reset mask，维护
`NO_CONTACT/STICK/INCIPIENT/GROSS`。object pose/velocity、relative contact
velocity、mass、jump flag、reward 和 future frames 只能作为 evaluation labels。

## 5. Policy 与训练

三个分支共享相同 serious architecture：

- official Tracker warm start；
- frozen official Refiner teacher；
- `504-D` deployable actor base；
- anatomical patch-token encoder：`9->128` projection、hand/patch/time embeddings、
  3-layer Transformer、4 heads、FFN 256、masked pooling 得到 `128-D`；
- existing SUGAR `512/256/128` actor，输出 `29-D` action；
- official `890-D` privileged critic，仅训练期可见；
- repository BCPPO、optimizer、physics 和 mass scheduler；
- 修正后的共同 reward 额外包含 post-handoff box lift/hold outcome term，teacher prefix
  上严格为零。

分支只在在线输入上不同：

- `Z`：patch/slip exact zero，zero TacSL reads；
- `P`：live contact/load/pressure/shear/friction，slip fields exact zero；
- `PS`：与 P 相同，加 causal slip callable。

训练 seeds 固定 `151014/151015/151016`，每个恰好 3000 updates：

- 0--499：pure distillation；
- 500--999：critic warmup；
- 1000--1999：PPO authority ramp；
- 2000--2999：steady full PPO；
- stage 3 保留共同 `distill_weight_floor=0.25`。

未来训练与评测都只加载 `data/CarryBox/data_045`，不能再用四个训练环境隐式选择
motion 0--3。teacher prefix transition 不进入 PPO surrogate/value/entropy credit。不得写 toy MLP、
offline tactile replay、taxel CNN、替代 teacher 或新 reward 来改变实验问题。

## 6. Endpoint 与冻结评估

每个 formal seed 到 `model_2999.pt` 后立即停止，不得延长或自动启动下一 seed。按顺序
检查：

1. model/optimizer tensor finiteness 与 iteration；
2. live Refiner handoff，无 reset；
3. requested/readback mass 一致；
4. jump 前 10 帧双手 contact；
5. 至少 450 帧覆盖 handoff、最大 50 帧 delay 和 80-frame outcome window；
6. handoff/jump action continuity；

集群约四小时强制撤销 H200 allocation；2026-08-21 的 P seed151014 先后在 update1259、
update2140 被撤销。matched formal 统一采用显式资源边界：先到 `model_1250.pt`，恢复到
`model_2000.pt`，再到 `model_2500.pt`，最后恢复到 2999。每次恢复必须加载 model、
optimizer、adaptive-KL learning rate，并从 checkpoint 的下一 BCPPO update 继续；模拟器/
RNG 状态不在 RSL-RL checkpoint 中，故每一臂都在相同三个边界以相同配置 seed 重新创建
环境。不得按各臂实际中断位置选择 checkpoint。
7. 同钟 G1/CarryBox 和双手 27-patch H.264。

通过后才显式运行 seed pairing：

- `151014 -> 152014`；
- `151015 -> 152015`；
- `151016 -> 152016`。

每对在五质量各 20 profiles，因此每分支 300 rollouts。camera-free traces 是统计来源；
camera video 只证明自己的 camera-enabled rollout，不声称逐帧 replay。

主要 outcome：post-jump 80-frame physical hold、drop、robot fall、minimum object height、
recovery latency、action response 和 nominal no-jump behavior。positive result 必须来自
matched frozen-policy physical improvement；loss、gradient、nonzero action difference 或
单条视频只证明信号可能被读取。

## 7. 2026-08-20 invalidity audit

旧 Z/P/PS 的“完成”状态与数值比较全部撤回，原因不是单一 scale：

1. `undesired_contacts` 把 54 个 anatomical pads 计入负奖励；正向 hand-contact term 却
   读取已经停用碰撞的原 hand links；
2. 训练只有间接的 reference object tracking，没有明确 post-handoff lift/hold outcome；
3. TacSL 把 `F_normal + F_friction` 一起投影后称为 shear，静止斜面法向力也会进入
   shear；friction utilization 使用 TacSL 固定 `mu=0.5`，PhysX box friction 却独立随机；
4. same-step reset 会在 slip cache 命中时提前返回，reset mask 没有进入 detector；
5. 4 个训练环境固定看到 motion 0--3，评测固定 motion45；
6. 正式 shell 总是打开 continued physical-outcome view，失败后的状态仍被用于主要物理
   指标；
7. 旧 percentile hierarchical bootstrap 只有 3 个独立训练 seeds，却对 180 个区间没有
   multiplicity correction。

已经完成的源码修正：pad 不再进入 undesired-contact penalty；正向接触奖励使用 54 个
独立的一-pad-to-box filtered ContactSensors 并每手聚合 27 个（IsaacLab 不支持单个
filtered regex sensor 覆盖多 bodies）；加入 post-handoff lift reward；TacSL 正常力/摩擦力分离；
PhysX/TacSL 摩擦固定匹配；reset 优先于 cache return；训练/评测统一 motion45；正式汇总
只接收 strict SUGAR evaluation；统计改为 3-seed exact paired sign-flip 并对完整 family
做 Holm correction。三 seeds 下双侧检验最小 raw p-value 是 `0.25`，因此不能再声称
“显著更好/更差”。

这些仍只是源码修复，不是新 policy 结果；当时的 no-learning aggregate force
calibration 也已被后续动态逐-pad audit 撤销。旧 checkpoints、traces 和 videos 仅供
追溯，不得用于判断 tactile benefit、harm、sensing 或 slip validity。

## 8. 高摩擦 6x/10x 可行性

完成当前摩擦合同下的 PS 三 seed 和正式比较后，已独立固定 CarryBox static/dynamic
friction 为 `0.5/0.5、1.0/1.0、1.5/1.5、2.0/2.0`，分别测试 `6x/10x`。八条均保存
material readback、jump 后高度、双手 contact 和完整 outcome window。

6x 在 `mu=0.5/1.0/1.5/2.0` 下的最大高度损失分别为
`0.5589/0.5429/0.02636/0.06596 m`；只有 `mu=1.5` 满足 5-cm hold。10x 四个条件均
drop。该结果证明至少一个 6x 条件在现有官方 Refiner 下物理可行，但不是单调摩擦曲线：
不同摩擦改变 pickup dynamics 并使 jump frame 在 325--328 间移动。高摩擦结果不得
并入原 Z/P/PS 对比。既然 6x gate 已通过，不触发 stronger-grip/lower-posture overfit；
成功条件的 camera-enabled rollout 同样 hold，最大高度损失 `0.02552 m`。同步
G1/CarryBox/54-patch H.264 位于
`experiments/online_patch_tactile_mass_adaptation/visualizations/`
`official_refiner_mu1p5_6x_friction_hold_single_env/official_refiner_mu1p5_6x_world_bilateral27.mp4`。

## 9. Corrected rerun 固定顺序

2026-08-20 source/runtime gate 当时通过：三 seeds × 五质量的 paired sweep 中 action
最大误差为零、jump-frame 最大误差为零；actor 为 `504-D` 且不含 measured object state。
causal slip precision/recall 为 `0.99285/0.97938`，675 个 onset 漏 18 个，median/p95 delay
为 `0/1` 帧，无非 contact-loss 的无接触报警。Z/P/PS runtime preflight 均 overall pass：Z
零 TacSL read；P/PS 各有 `361 x 54 = 19494` 次在线 patch read；PS 有 361 次 slip update。
独立 force audit 已改为合法的 object-centric ContactSensor：box 对 27/54 个 exact pad
filters。当时得到 PhysX `56.964466/15.751346 N` 与 TacSL
`56.964466/16.207577 N`；unfiltered normal coverage error 仅 `4.77e-8 N`。后续动态
逐-pad audit 已证明这个十帧 aggregate match 不能作为 calibration admission，因此
该 gate 的 scale 部分已撤销；其 runtime wiring 结果仍有效。
额外的逐-pad reward runtime preflight 确认 `hoi_contact` 非零、`30/360` handoff 后
transitions 和 1 次 mass event，且不再出现 PhysX filter-count error。首轮纯蒸馏中途
`model_250` 的 4 条 strict 3x profile 均没有完整 80-frame post-jump window，已判废归档；
它不是 tactile-benefit 结果。随后发现两个已归档进程并未退出，其中一个与新 run 并发
写同一目录；受污染目录及其 model-0 评测全部撤销。启动器现用稳定 pipeline lock 阻止
复发，运行时 lock probe 已按预期以 exit 75 拒绝第二 writer。干净的 4-env fixed-3x PS
overfit 已从零重启。首批 model-0/250/500/750 review 错用了 formal evaluation 的随机
`10--50` 帧 mass delay 和随机 reset pose，与 overfit 训练固定的 20 帧、零位姿噪声不匹配。
它们仅保留为更难分布下的行为 diagnostic，不是 fixed-condition overfit gate。评测入口
现用 `--fixed-3x-overfit-gate` 强制训练同配置，正在不重训的情况下重评已有 checkpoint。
fixed-condition model-0/250/500/750/1000 均为 `0/4` strict hold、`0` robot fall；3x 后存活帧
约为 `12`、`19--20`、`19--22`、`66--72`、`19--21`。model-750 最好但仍未达到 80 帧，
model-1000 又因 object orientation 偏差退化。model-1250 严格评测达到 `3/4` 完整
80-frame hold、`0` drop、`0` fall，说明 serious policy 在固定条件下可学习；但它早于
taxel-grid 修复，只能作为行为 diagnostic，不能续训或支持 tactile benefit。
错配 review 中同一 camera-enabled profile 的 CarryBox 世界画面与双手 27-patch 同步视频为
`overfit_ps_model500_eval3x_camera_profile0_v3/model500_3x_world_bilateral27.mp4`；该视频只证明
自身 rollout，不代替 camera-free strict 统计。上游 TacSL pytest 依赖的远程 NVIDIA
R15 USD 在当前节点不可下载，因此不将其外层 exit code 当作通过证据。

1. 完成 extent-filled taxel grid 的动态 SDF/contact-offset 与逐 pad PhysX 对照，重做
   CarryBox TacSL calibration；旧 schema-v2 scale 也必须撤销；
2. 串行执行 Z/P/PS 360-frame runtime preflight，验证 zero-read、54-pad read、causal slip、
   handoff、mass readback 和 PPO mask；进入训练前，causal slip 对 held-out simulator
   tangential-velocity label 的 contact-supported precision/recall 均须至少 0.8，onset miss
   rate 不超过 20%，且不得在非 contact-loss 的无接触样本报警；
3. 重新生成尺度后，在固定 motion45、3x mass、20-frame delay、无 reset pose noise 下从
   official Tracker 运行新的 PS serious overfit；不得从 model-1250 续训；
4. 只有该 gate 通过才从 official Tracker 重新训练 matched Z/P/PS，旧 checkpoint 不续训；
5. 三分支使用相同 motion45、reward、摩擦、初始化、3000-update budget 和 seeds；
6. 严格评测终止即停止计分，continued physical outcome 只能作为明确标注的 diagnostic；
7. 统计以 training seed 为独立单位，报告每-seed effect 与 Holm-corrected exact test。

截至 2026-08-21，旧 Z seed151014/model2999 使用已撤销的 model1750 边界，不能与新协议
比较；其首轮 seed152014 五档评测也因 evaluator 为画图在 Z 上额外读取 TacSL 而撤回。评测器现将 Z patch/slip trace
固定为 exact zero，完全不调用 TacSL 或 slip detector，并显式记录 zero-read 合同。P
P seed151014 的 finite model2000 已审查，将从该 direct child 恢复到 model2500；新 Z
必须走相同边界并完成 endpoint 后才串行评测。formal evaluator 与 trainer 现共享同一个稳定 pipeline lock；训练活跃时评测入口
必须以 exit 75 拒绝。正式结论仍必须等待 matched P/PS 和三 seed 配对统计。
