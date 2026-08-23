# Curiosity

本仓库研究两个问题：示范反馈能否真正改变并约束 humanoid 操作策略，以及在线整手触觉
能否在视觉不变、接触 binary 不变但物体动力学改变时提供额外适应能力。仿真统一使用
IsaacLab/PhysX；Newton 只作为 asset 来源，不作为执行后端。

## 当前结论

### Demo following：信号被使用，语义遵循尚未成立

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

actual-rollout redesign 现已完成第一阶段。official Tracker 在 IsaacLab/PhysX 中为每条
source motion 采集 700 个同钟帧，实际 target 来自分别过滤到箱子的左右手/左右脚
`force_matrix_w_history`，而不是 reference binary。完整 corpus 覆盖 100 条 CarryBox 和
99 条 KickBox、无重复和 reset。Carry 的中位双手同时接触率为 `32.93%`、最长手部事件
`4.60 s`、最大抬升 `0.490 m`；Kick 的中位足部接触率为 `4.14%`、最长足部事件
`0.22 s`、足力峰值 `60.23 N`、最大抬升仅 `0.0066 m`。

新的 internal reward predictor 保留 serious 6-layer、384-D causal Transformer，具有
`11,530,010` 个参数，并增加 trajectory、四肢 contact、event duration 和 motion-regime
共 13 个 mismatch targets。输入只有过去 10 帧 official Tracker 510-D observation 和固定
numeric selected-demo windows；未来 actual events 只作 label。motion-disjoint formal
seed271301 在 epoch 13 early-stop，冻结 best epoch 8。validation/test normalized MAE 为
`0.1878/0.1708`，优于 constant `0.2354/0.2192`；zero-demo 为 `0.2423/0.2279`，
permuted-demo 为 `0.1985/0.1778`，median Spearman 均约 `0.505`。test 中模型预测的
cross-task contact mismatch 为 `0.1495`，correct 为 `0.0188`。全部 12 个 predictor gates
通过。冻结 epoch-8 checkpoint 后，仅用 validation 拟合 13 个 variance multipliers；90%
区间在 validation/test 上的平均覆盖率为 `95.25%/95.90%`，test 最低单目标覆盖率为
`86.93%`。这证明它在 held-out motions 上读取并使用了 selected demo 的接触/运动语义，
且不确定性区间保守可用；仍不等于 policy 已经遵循 demo，下一步必须做 matched policy
experiment 和独立行为审计。

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
5. 实现 11.53M serious causal trajectory/contact/duration/regime mismatch predictor，并通过
   motion-disjoint held-out、zero-demo 与 permuted-demo 检查；
6. 建立 fixed-teacher、只改变 selected-demo reward 的因果实验，排除 teacher replacement
   混杂；
7. 建立 predictor/reward-independent、以训练 seed 为重复单位的 Carry/Kick 行为审计，
   分离 task success、reward use 与 semantic obedience；
8. 对 official single-clip TinyMDM 做 exact-identity 与 independent semantic-extension
   分离测试，得到“记住 clip 但尚未形成可靠语义空间”的负结果。

官方 SUGAR、IsaacLab TacSL 和 MimicKit TinyMDM 本身不是本仓库的原创方法；本仓库的
贡献是忠实接入、实验协议、在线传感扩展、因果隔离与失效审计。

## 最短入口

完整环境、资产、命令、输出合同和结果核验见
[可复现性与证据记录](DOCS/reproducibility.md)。常用入口如下。

```bash
cd /public/home/yanhongru/Curiosity
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python

# 无仿真：检查当前 same-teacher 配置和下一条训练命令
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design same_teacher_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --dry-run

# GPU compute node：复核现有冻结评估和完整视频
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 same_teacher_reward_only

# 无 GPU：从现有 traces 重算独立行为审计
$PYTHON_BIN scripts/sugar/demo_following/analyze_behavior_adherence.py

# 无 GPU：汇总三个独立训练 seeds；20 physics profiles 只作 seed 内变化
$PYTHON_BIN scripts/sugar/demo_following/aggregate_behavior_adherence.py

# GPU compute node：复现 teacher 1.0 -> 0.25 的单 seed 诊断、冻结评估和视频
bash scripts/sugar/demo_following/run_teacher_floor_overfit_pair.sh

# 无 GPU：审计 199 条 official Carry/Kick reference 的 contact/event 标签可分性
$PYTHON_BIN scripts/sugar/demo_reward/audit_contact_event_reference_corpus.py

# GPU compute node：从已采集的 actual physical rollouts 构建、训练并校准 predictor
$PYTHON_BIN scripts/sugar/demo_reward/build_actual_contact_event_predictor_dataset.py \
  --corpus-root experiments/demo_following/contact_event_reward_redesign_v1/actual_tracker_corpus \
  --output-dir experiments/demo_following/contact_event_reward_redesign_v1/reproduction_dataset_v1
$PYTHON_BIN scripts/sugar/demo_reward/train_actual_contact_event_predictor.py \
  --dataset-root experiments/demo_following/contact_event_reward_redesign_v1/reproduction_dataset_v1 \
  --output-dir experiments/demo_following/contact_event_reward_redesign_v1/reproduction_seed271301 \
  --epochs 20 --seed 271301 --device cuda:0
$PYTHON_BIN scripts/sugar/demo_reward/calibrate_actual_contact_event_predictor.py \
  --dataset-root experiments/demo_following/contact_event_reward_redesign_v1/reproduction_dataset_v1 \
  --predictor-dir experiments/demo_following/contact_event_reward_redesign_v1/reproduction_seed271301 \
  --device cuda:0

# GPU compute node：复现完整 G1 CarryBox 与双手 27-patch 在线触觉视频
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox
```

当前关键视频：

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

actual contact/event corpus 与 serious predictor gate 已完成。下一步是在不改变 teacher、
初始化、seed、物理、budget 和 task reward 的前提下，只把冻结的 event predictor potential
接到 selected-demo reward，重新做 correct Carry versus unrelated Kick matched policy
comparison。训练前需先固定 reward scale 与 stopping rule；最终结论只看 predictor-independent
冻结行为，不用 predictor 自己给自己判成功。SMP 仍不进入 selected-demo policy reward。
