# 可复现性与证据记录

本文是当前项目唯一的完整复现记录。它覆盖官方 SUGAR CarryBox 基线、same-teacher
demo-following、official TinyMDM 语义门槛、IsaacLab 在线整手触觉、质量泄漏审计、重箱
摩擦可行性、关键产物和结论边界。日期化排障记录、失败实验和重复渲染均在 ignored
`legacy/`，不作为复现入口。

## 1. 软件、后端与资产

所有仿真、训练和渲染统一使用 IsaacLab/PhysX：

- Python：`/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python`；
- Isaac Sim 5.1；
- 活动 IsaacLab：仓库根 `IsaacLab/`；
- 活动 SUGAR：仓库根 `SUGAR/`；
- official MimicKit checkout：仓库根 `MimicKit/`，固定 commit
  `2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69`；
- CarryBox motions：`SUGAR/data/CarryBox/`；
- official Tracker/Generator checkpoints：`SUGAR/demo_ckpts/CarryBox/`；
- frozen Refiner teacher：
  `experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt`；
- official R15 USD/calibration：`experiments/sugar_reproduction/assets/official_tacsl/`。

这些大资产和所有实验输出被 Git 忽略。一个 fresh clone 必须先恢复官方 SUGAR 数据、
released checkpoints、MimicKit checkout 和本地实验依赖，不能用自写 toy model 替代。

GPU 工作只能在 retained Slurm compute allocation 中执行。长任务通过
`scripts/sugar/native_tactile/launch_retained_child.sh` 启动；切换工作时只终止记录的 child
process group，不取消 allocation shell。

基础环境检查：

```bash
cd /public/home/yanhongru/Curiosity
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
export DISPLAY=
export OMNI_KIT_ACCEPT_EULA=Y

PYTHON_BIN="$PYTHON_BIN" bash scripts/sugar/preflight_official_sugar_env.sh
```

## 2. 官方 SUGAR CarryBox 基线

官方流水线为：

```text
processed human/object motion
        -> privileged Refiner policy
        -> successful Refiner rollout / Tracker RL dataset
        -> Tracker policy
        -> successful Tracker rollout / Generator IL dataset
        -> diffusion Transformer Generator
        -> generated reference + Tracker closed-loop control
```

Refiner 和 Tracker actor 使用 `512/256/128` ELU MLP，输出 29-D G1 joint-position target。
Refiner 的约 890-D observation 包含未来 reference、完整 robot/object privileged state；它是
仿真 teacher，不是可直接部署的真实机器人 policy。官方 Tracker actor 为约 510-D state
observation。Generator 是 12-layer、8-head diffusion Transformer，对 8-step、36-D
reference sequence 建模。官方 released policy 在 control step 读取 state，不读取原始 RGB；
human video 只在上游生成 processed 3D motion 数据。

运行 released CarryBox inference：

```bash
PYTHON_BIN="$PYTHON_BIN" NUM_ENVS=16 VIDEO_LENGTH=200 \
  bash scripts/sugar/run_official_sugar_carrybox_inference.sh
```

从头运行官方七阶段流水线：

```bash
PYTHON_BIN="$PYTHON_BIN" TASK_NAME=CarryBox \
OUTPUT_DIR="$PWD/experiments/sugar_reproduction/outputs/CarryBox_reproduction" \
  bash scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh
```

七个阶段依次是 `refiner_train`、`refiner_rollout`、
`process_refiner_rollout`、`tracker_train`、`tracker_rollout`、
`process_tracker_rollout` 和 `generator_train`。可用 `START_STAGE`、`STOP_AFTER_STAGE`、
`REFINER_RESUME_CHECKPOINT` 与 `TRACKER_RESUME_CHECKPOINT` 恢复，不得把中间诊断冒充完整
官方训练。

保留证据位于：

```text
experiments/sugar_reproduction/outputs/final/official_sugar/
├── baseline/ckpts/refiner_model10000.pt
├── baseline/visualizations/
└── released_inference/
```

## 3. Same-teacher demo-following

### 3.1 实验定义

两个分支共同使用：

- CarryBox45 fixed official Refiner teacher；
- sim/policy seed `161581`，action seed `161582`；
- 20 environments，24 steps/update，64 updates；
- 相同初始化、physics tuples、SUGAR native PPO、reward weights；
- frozen 11.9M future-mismatch predictor；
- 相同 generic official TinyMDM prior；
- exact-zero tactile control；
- teacher coefficient 全程为 1。

唯一差异是 selected reward demo：correct 使用 CarryBox45，unrelated 使用 KickBox21。
KickBox21 不替换 teacher，CarryBox 任务也不改变。

### 3.2 无 GPU 配置检查

```bash
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design same_teacher_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --dry-run

$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design same_teacher_reward_only --arm unrelated \
  --endpoint-updates 64 --stop-after-segment --dry-run
```

dry-run 必须显示两臂的 teacher path 相同，demo runtime config 不同。

### 3.3 从新目录串行训练两臂

在 retained GPU shell 中：

```bash
OUT=experiments/demo_following/reproduction_same_teacher_v1

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record "$OUT/correct.process" --status "$OUT/correct.status" \
  --log "$OUT/correct.log" --tag demo-correct-64 -- \
  "$PYTHON_BIN" -u scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design same_teacher_reward_only --arm correct --output-root "$OUT" \
  --endpoint-updates 64 --stop-after-segment
```

等 correct child 结束并检查 `seed161581/correct/update_0064/proof.json` 后，再运行：

```bash
bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record "$OUT/unrelated.process" --status "$OUT/unrelated.status" \
  --log "$OUT/unrelated.log" --tag demo-unrelated-64 -- \
  "$PYTHON_BIN" -u scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design same_teacher_reward_only --arm unrelated --output-root "$OUT" \
  --endpoint-updates 64 --stop-after-segment
```

两个 arm 不得并行训练，也不得有多个 writer 写同一个 seed 目录。

### 3.4 Teacher-only prerequisite gate

在 correct endpoint 存在后，用相同环境把 learned residual 精确置零：

```bash
TEACHER_ONLY_GATE=1 \
TEACHER_GATE_OUTPUT="$PWD/$OUT/seed161581/teacher_only_gate" \
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 same_teacher_reward_only "$PWD/$OUT/seed161581"
```

`RESULT.json` 必须满足所有 checks，并且 20/20 profiles 均有 bilateral contact 和至少
5 cm lift。当前保留 gate 的平均 bilateral-contact frames 为 155.05，平均最大抬升为
0.7128 m。

### 3.5 冻结评估和完整视频

```bash
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 same_teacher_reward_only "$PWD/$OUT/seed161581"
```

输出包含：

```text
seed161581/
├── correct/update_0064/{policy.pt,proof.json,protocol.json}
├── unrelated/update_0064/{policy.pt,proof.json,protocol.json}
├── evaluation_update0064/{correct,unrelated}/{RESULT.json,TRACE.npz}
└── videos_update0064/
    ├── 01_correct_demo_and_actual_behavior.mp4
    ├── 02_unrelated_kickbox_demo_and_actual_behavior.mp4
    └── RENDER_PROOF.json
```

现有正式结果为 correct `16 success / 2 fall / 2 unfinished`，unrelated
`18 success / 2 fall`。结果证明 reward demo 改变策略，但不证明 correct-demo superiority
或 semantic obedience；20 个 physics profiles 不是 20 个独立训练 seeds。

### 3.6 Predictor-independent behavior audit

该审计只读取 frozen trace 中的 robot/object state、lift height 和左右手刚体合力。它不会
加载 predictor，不读取 `demo_*`、reward term、policy loss 或 future mismatch：

```bash
$PYTHON_BIN scripts/sugar/demo_following/analyze_behavior_adherence.py
```

输出位于：

```text
experiments/demo_following/matched_reward_identity_same_teacher_v1/
└── seed161581/behavior_adherence_audit_v1/
    ├── RESULT.json
    ├── profiles.csv
    └── behavior_adherence.png
```

物理阈值固定为 lift `>=0.05 m`、每手刚体合力 `>0.1 N`，与冻结 evaluator 的定义一致。
CarryBox45 reference 最大抬升 `0.7639 m`，lifted transport fraction `0.8141`；KickBox21
最大抬升 `0.0304 m`，lifted transport fraction `0`，orbit rate 为 `0.5646 rad/s`，高于
CarryBox45 的 `0.3521 rad/s`。

实际两臂都高度 Carry-like。Kick-reward 减 correct-reward 的 paired-profile mean 为：

- lifted-frame fraction `+0.0350`；
- lifted-transport fraction `+0.0323`；
- ground-transport fraction `-0.0323`；
- orbit rate `-0.0050 rad/s`；
- bilateral hand-contact fraction `+0.0390`。

这些方向不支持 Kick 语义。lifted/ground transport 是互补量，不当作两个独立统计检验。
20 profiles 共享一个训练 seed，因此只作物理 profile 描述，不冒充多 seed 显著性。现有
TRACE 没有逐 body pose、foot-box contact 或 hand-box-only contact，不能追溯脚部踢击；
下一冻结 evaluator 必须补齐这些 evaluation-only 字段。

### 3.7 Predeclared multi-seed repeat

保持两臂 64 updates、CarryBox45 teacher coefficient 1 和所有 reward/physics 设置完全一致，
只改变 selected demo。三组 training/action seeds 为 `161581/161582`、`161583/161584`、
`161585/161586`；对应 frozen evaluation seeds 为 `171581/171583/171585`。已有第一组，
后两组必须串行执行，且未经明确授权不得启动。

每次只运行并完成一个 seed pair；脚本会串行执行 correct、unrelated、冻结评估、视频和
独立行为审计，任一 proof 失败都会停止：

```bash
# retained GPU shell
bash scripts/sugar/demo_following/run_predeclared_multiseed_pair.sh 161583

# 人工检查 seed161583 后才运行下一组
bash scripts/sugar/demo_following/run_predeclared_multiseed_pair.sh 161585

# 三个 seed 全部完成后汇总；训练 seed 才是 replication unit
$PYTHON_BIN scripts/sugar/demo_following/aggregate_behavior_adherence.py
```

主要行为判据不是 predictor score：correct arm 应有更多 lifted transport；unrelated arm 应
有更多 ground-level transport 和更高 orbit rate。task success、fall、双手接触和最大抬升
分别报告。三个 seed 均应显示相同方向，才把结果称为稳定的 demo-conditioned behavior；
否则结论仍是 reward 使用但语义遵循未成立。

研究依据是：DeepMimic 将 imitation objective 与 task objective 分开；PhysHOI 使用 contact
graph 防止错误 body-object interaction；InterMimic 同时约束 object deviation、joint-object
关系和 required-contact duration；CHORD 进一步用 object-centric contact wrench 衡量接触
如何驱动物体，而不只看接触位置。对应原始来源：

- https://arxiv.org/abs/1804.02717
- https://arxiv.org/abs/2312.04393
- https://arxiv.org/abs/2502.20390
- https://nvidia-isaac.github.io/video_to_data/chord/

## 4. Internal reward predictor

模型输入为过去 10 帧 policy state 和指定 demo condition，目标为未来 body、box position、
box rotation 6D 和 box velocity mismatch。policy 使用 predicted mismatch potential
difference；冻结测试时不把未来 GT trajectory 输入 actor。

保留文件：

```text
experiments/demo_following/predictor/
├── RESULT.json
├── FROZEN_ENDPOINT.json
└── validation_best.pt
```

artifact-only 检查：

```bash
jq '{status,scientific_gate_passed,trainable_parameter_count,
     test:{full:.test.full.mean_component_normalized_mae,
           permuted:.test.permuted_demo.mean_component_normalized_mae,
           zero:.test.zero_demo.mean_component_normalized_mae}}' \
  experiments/demo_following/predictor/RESULT.json
```

预期 full/permuted/zero normalized MAE 约为 `0.1873/0.3885/0.4481`。这证明 demo identity
被读取；它不是独立的行为语义评价器。

## 5. Official TinyMDM selected-demo gate

使用 official MimicKit `10 x 216`、50 Hz feature contract，CarryBox45 和 KickBox21 各训练
一个 2.836M 参数、50,000-iteration single-clip prior：

```bash
$PYTHON_BIN scripts/sugar/smp/run_selected_demo_tinymdm.py prepare
$PYTHON_BIN scripts/sugar/smp/run_selected_demo_tinymdm.py train --clip carry45
$PYTHON_BIN scripts/sugar/smp/run_selected_demo_tinymdm.py train --clip kick21
$PYTHON_BIN scripts/sugar/smp/run_selected_demo_tinymdm.py score
```

exact selected-clip pairwise preference 两个方向均为 1.0，但 independent CarryBox96/KickBox22
semantic extension 失败，`policy_integration_authorized=false`。该结果是严肃负结果，不是
执行失败；保留完整 priors、dataset 和 cross-score 于
`experiments/demo_following/selected_demo_smp_v1/`。

## 6. IsaacLab 在线整手触觉

每只手的 27 个物理 patch 排列为掌心 12 个加五指各三段。official R15 taxel 在 patch 内
产生法向、摩擦/切向与可选 optical 输出，在线归约后才进入 policy。训练用 slip 接口只有：

```python
PatchSlipDetector.update(
    contact,
    normal_load_n,
    mean_pressure_pa,
    shear_xy_n,
    friction_utilization,
    timestamp_s,
    reset_mask,
)
```

object motion、relative contact velocity、mass factor、jump flag、reward 和 future frame 只可
作为评估 label。

复现完整 CarryBox world + bilateral 27-patch 视频：

```bash
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox
```

复现大面积掌面接触、重物和释放失败样本：

```bash
bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_free 0.5

bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_heavy 2.0
```

保留的四个原生样本均含 `whole_hand_trace.npz`、`summary.json`、world video、双手 27-patch
H.264 和 render proof。普通 CarryBox 主要由指端承载；掌面样本只证明整掌覆盖，不冒充
CarryBox 托底动作。

## 7. 质量泄漏与 causal slip 审计

三 seeds、五质量 `1/1.5/3/6/10x` 使用 nominal action exact replay，隔离质量变化对
proprioception 和在线 patch channel 的影响：

```bash
$PYTHON_BIN scripts/sugar/native_tactile/run_online_mass_leakage_sweep.py \
  --output-root experiments/online_patch_tactile_mass_adaptation/reproduce_leakage_v1 \
  --motion-folder SUGAR/data/CarryBox/data_045 --motion-id 0 \
  --max-steps 420 --jump-delay-frames 30 --device cuda:0
```

输出必须包含 `leakage_audit.json`、`slip_evaluation.json`、
`patch_channel_scales.json` 和 15 条 online trace。已有审计证明 504-D actor 虽不含 measured
object state，质量仍会经关节状态/动力学泄漏到 proprioception，因此最终触觉结论必须是
相对 proprioception 的增益，而不是“只有触觉能感知变重”。历史 slip 数值只保留为审计，
Plan 15 的最终 sensing claim 仍冻结。

## 8. 重箱摩擦可行性

该 sweep 独立于无效的 Z/P/PS 比较，使用 frozen official Refiner：

```bash
OUTPUT_ROOT="$PWD/experiments/online_patch_tactile_mass_adaptation/reproduce_friction_v2" \
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/sugar/native_tactile/run_plan15_friction_feasibility.sh cuda:0
```

它运行 `mu=0.5/1.0/1.5/2.0 x mass=6x/10x` 八个条件并要求 exact material/mass readback、
跳变前 10 帧双手接触和完整 post-jump window。6x 的 height loss 依次为
`0.5589/0.5429/0.02636/0.06596 m`，只有 `mu=1.5` 通过 5 cm hold；10x 全部掉落。

## 9. 回归测试与静态核验

不启动仿真即可运行：

```bash
$PYTHON_BIN -m pytest -q \
  tests/native_tactile/test_plan15_invalidity_regressions.py \
  tests/native_tactile/test_online_patch_tactile.py \
  tests/native_tactile/test_online_mass_jump.py \
  tests/native_tactile/test_compare_online_patch_mass_sweeps.py
```

当前集中测试预期为 38 passed。关键 JSON 可直接检查：

```bash
jq '.final_update_aggregate' \
  experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161581/evaluation_update0064/correct/RESULT.json
jq '.final_update_aggregate' \
  experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161581/evaluation_update0064/unrelated/RESULT.json
jq '{hold_success_count,strict_sugar_hold_success_count,drop_count,robot_fall_count}' \
  experiments/online_patch_tactile_mass_adaptation/corrected_rerun_20260820/p151014_tactile_only_v3_checkpoint_sweep/model1100_profiles20/summary.json
jq '.runs' \
  experiments/online_patch_tactile_mass_adaptation/friction_feasibility_after_ps/aggregate_summary.json
```

## 10. 证据边界

- training loss、gradient、predicted reward、nonzero action difference 和单条有利视频只能证明
  signal use，不能证明任务收益；
- current predictor 的自评分不能替代独立 demo-adherence metric；
- one training seed 不能支持 correct/unrelated 的统计优劣；
- TacSL 是 SDF penalty high-fidelity simulated tactile，不是完整 soft-body FEM；
- 未完成实体 GelSight 标定前不能声称硬件触觉或 sim-to-real；
- 当前没有完成 corrected matched Z/P/PS，所以不能声称触觉有益或有害；
- 所有实验产物只在 ignored `experiments/`，不得 commit/push。
