# Plan 15: Online Whole-Hand Patch Tactile for Sudden-Mass Adaptation

## 1. 科学问题

在完整 SUGAR G1 已经抬起 CarryBox 后，保持几何、外观、参考轨迹和接触目标不变，
在线提高 PhysX mass/inertia。检验 live whole-hand tactile 是否在可部署 proprioception
之上带来更早、有效的控制反应，并提高持稳、恢复或安全放下的物理结果。

质量变化不是天然“只有触觉可见”：官方 Refiner `890-D` observation 包含 measured
object state，部署所需的 `joint_pos/joint_vel` 也会在受载后变化。因此正式 claim 只能是
触觉相对 `504-D` Tracker-command/proprioception 的增量收益。measured object state、
真实质量、mass factor、jump flag、RGB 和 future frame 不得进入 actor。

Plan 15 是唯一活动计划。RGB、demo following、ICM/Curiosity、Newton simulator、
deformable 和软体训练不进入本实验。

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

官方 TacSL R15 taxels 计算 penetration、normal force 和 signed XY shear；每个 control
step 在 GPU 上在线归约成一个 patch record。policy 单元永远是 patch，不是 taxel。

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
- repository BCPPO、optimizer、reward、physics 和 mass scheduler。

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

teacher prefix transition 不进入 PPO surrogate/value/entropy credit。不得写 toy MLP、
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

## 7. 已完成证据

### Sensing

- paired leakage sweep：3 seeds × 5 factors 完成，质量读回、event pairing、双手 contact
  和 54-patch clock 通过；
- continuous patch 约 13 帧稳定区分质量，proprio 约 35 帧；
- 119/119 drops 前 continuous patch change，中位提前 21 帧；
- 133/133 sag≥`0.02 m` 前 continuous patch change，binary contact 只覆盖 81/133；
- controlled R15 slip：STICK `109/111`、INCIPIENT `109/109`、GROSS `19/20`；
- CarryBox 3x slip：precision `1.0`、recall `0.9971`、median delay 0、p95 1 frame。

这些只证明信息和时序优势，不是 tactile-policy benefit。

### Formal Z/P

Z 三 seed holds=`59,59,52,1,0`，drops=`0,0,2,58,59`。

P 三 seed holds=`59,59,49,0,0`，drops=`0,0,8,59,59`。3x P-Z paired interval 跨零，
所以 P 没有证明增益，且当前趋势更差。

### Formal PS

- seed `151014`：19 eligible profiles/factor；holds=`19,19,10,0,0`，
  drops=`0,0,6,19,19`；
- seed `151015`：20 profiles/factor；holds=`20,20,16,0,0`，
  drops=`0,0,4,20,20`；
- seed `151016`：未开始。

PS 仍缺第三 seed，不能做最终 Z/P/PS 结论。

## 8. 高摩擦 6x/10x 可行性

完成当前摩擦合同下的 PS 三 seed 和正式比较后，独立固定 CarryBox static/dynamic
friction 为 `0.5/0.5、1.0/1.0、1.5/1.5、2.0/2.0`，分别测试 `6x/10x`。每条保存
material readback、jump 后高度、双手 contact 和 outcome。

高摩擦结果不得并入原 Z/P/PS 对比。某个 frozen controller 失败不等于物理不可能；
若 `mu<=2` 仍不能抬稳 6x，继续测试 stronger grip/lower posture 或相同 serious SUGAR
policy 的固定条件 overfit，直到至少一个 6x 条件出现可验证的完整持箱。最后生成同步
G1/CarryBox/54-patch H.264。

## 9. 串行执行顺序

1. 人工确认 PS-151015 endpoint 视频与数值审查；
2. 从 official Tracker 启动 PS-151016，严格停在 update 2999；
3. 审查 endpoint，再显式运行其 100-rollout frozen evaluation 和视频；
4. 运行 exact Z/P/PS 三 seed paired comparison；
5. 运行独立 friction sweep；
6. 若 6x 未成功，继续 stronger-grip/lower-posture 或 serious overfit；
7. 更新 README/TODO/AGENTS，只提交源码与文档。
