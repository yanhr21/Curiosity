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

GPU 工作只能在 retained Slurm compute step 中执行。获得 `salloc` 后必须再用
`srun --jobid=<job_id> ... bash` 进入计算节点；仅有 `SLURM_JOB_ID` 的 `salloc` prompt 仍可能
停留在登录节点。长任务从 compute step 通过
`scripts/sugar/native_tactile/launch_retained_child.sh` 启动；记录中必须同时有 Slurm step、
compute host 和 child PID/PGID。切换工作时只终止记录的 child process group，不取消
allocation shell。

```bash
srun --jobid="$SLURM_JOB_ID" --exclusive --gres=gpu:1 --pty bash
test -n "$SLURM_STEP_ID"
hostname  # 必须是计算节点，不得是 mgmtserver/login
```

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
    ├── behavior_adherence.png
    └── reference_semantic_timeline.png
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
seed161581 的旧 TRACE 没有逐 body pose 或 foot-box contact，不能追溯脚部踢击。后续重复实验
的冻结 evaluator 已补充 `robot_body_position_w`、左右 `foot_box_contact_force_w` 和明确过滤到
箱子的左右手接触力；这些字段只用于 evaluation，不进入 actor、predictor 或 reward。

`reference_semantic_timeline.png` 同时画出 source clip 的 box lift、XY speed 和官方 binary
contact proxy。CarryBox45 的 hand-contact proxy 为一个 `4.90--10.82 s` 连续区间，5 cm lift
从 `5.72 s` 持续至 `10.16 s`，峰值在 `7.00 s`。KickBox21 有 14 段间歇 foot-contact proxy，
右脚踝在 `3.80 s` 最接近箱子中心，且全段不超过 5 cm。该图把下一实验的语义预期固定为
interaction event structure；binary proxy 不得被描述为触觉力。

### 3.7 Predeclared multi-seed repeat

保持两臂 64 updates、CarryBox45 teacher coefficient 1 和所有 reward/physics 设置完全一致，
只改变 selected demo。三组 training/action seeds 为 `161581/161582`、`161583/161584`、
`161585/161586`；对应 frozen evaluation seeds 为 `171581/171583/171585`。三组现已全部
串行完成。

每次只运行并完成一个 seed pair；脚本会串行执行 correct、unrelated、冻结评估、视频和
独立行为审计，任一 proof 失败都会停止：

```bash
# retained GPU shell
bash scripts/sugar/demo_following/run_predeclared_multiseed_pair.sh 161583

# 前一组 proof、冻结评估和行为审计全部通过后，脚本按固定顺序进入下一组
bash scripts/sugar/demo_following/run_predeclared_multiseed_pair.sh 161585

# 三个 seed 全部完成后汇总；训练 seed 才是 replication unit
$PYTHON_BIN scripts/sugar/demo_following/aggregate_behavior_adherence.py
```

主要行为判据不是 predictor score：correct arm 应有更多 lifted transport；unrelated arm 应
有更多 ground-level transport 和更高 orbit rate。task success、fall、双手接触和最大抬升
分别报告。三个 seed 均应显示相同方向，才把结果称为稳定的 demo-conditioned behavior。
实际 unrelated-minus-correct 结果为：

- lifted-frame fraction：`+0.0350/+0.0179/-0.0058`，预期负方向仅 `1/3`；
- lifted-transport fraction：`+0.0323/+0.0132/-0.0277`，预期负方向仅 `1/3`；
- ground-transport fraction：`-0.0323/-0.0132/+0.0277`，预期正方向仅 `1/3`；
- orbit rate：`-0.0050/-0.0294/-0.0115 rad/s`，预期正方向 `0/3`。

lifted/ground transport 是同一路径的互补视图，不算两个独立检验。task success 的
correct/unrelated 计数分别为 seed161581 `16/18`、seed161583 `18/17`、seed161585
`16/17`；physical falls 分别为 `2/2`、`1/1`、`2/1`。seed161583 和 seed161585 的新增
foot-to-box channel 显示接触为零或极少，未形成 Kick 接触角色。聚合结果为
`stable_semantic_following=false`。

聚合证据位于：

```text
experiments/demo_following/matched_reward_identity_same_teacher_v1/
└── multiseed_behavior_adherence_v1/
    ├── RESULT.json
    └── multiseed_behavior_deltas.png
```

### 3.8 Teacher-floor learnability diagnostic

该 fixed-profile pair 已完成。两臂分别从 seed161581 的 update-64 endpoint 恢复，只新增
64 updates；共同 CarryBox45 teacher 以同一 global schedule 从 `1.0` 降到 `0.25`，冻结评估
也保持 `0.25`，没有换成 teacher-free。固定物理为 mass `1.0x`、static/dynamic friction
`0.6/0.5`、COM-y `0`、pulse `0`。复现完整训练、proof、冻结评估和视频：

```bash
# retained GPU shell；脚本自动串行 correct、unrelated、evaluation、render 和行为审计
bash scripts/sugar/demo_following/run_teacher_floor_overfit_pair.sh
```

两臂 proof 均记录 `resume_update=64`、`updates_executed_this_process=64`、最终 teacher floor
`0.25`，且全部 checks 通过。20-profile frozen behavior 的结果不是语义分离，而是两臂坍塌：

- correct/unrelated bilateral-contact mean：`0/0`；
- correct/unrelated lifted-frame mean：`0/0`；
- correct/unrelated lifted-transport mean：`0/0`；
- correct/unrelated foot-to-box contact mean：`0/0`；
- 四个预登记 Kick-like 方向：`0/4`；
- episode duration 均约 `0.88 s`，correct 最大抬升为 `0`，unrelated 平均最大抬升约
  `0.00027 m`。

因此 teacher 降权没有暴露 trajectory predictor 的语义控制能力，而是先破坏了共同 Carry
解。该 schedule 不进入多 seed。视频与直接行为证据位于：

```text
experiments/demo_following/teacher_floor_overfit_v1/seed161581/
├── videos_update0128/01_correct_demo_and_actual_behavior.mp4
├── videos_update0128/02_unrelated_kickbox_demo_and_actual_behavior.mp4
├── behavior_adherence_audit_v1/RESULT.json
└── TEACHER_FLOOR_GATE.json
```

### 3.9 Contact/event reward redesign

第一项 reference 可行性审计已完成：

```bash
$PYTHON_BIN scripts/sugar/demo_reward/audit_contact_event_reference_corpus.py
```

它读取全部 100 条 CarryBox 和 99 条 KickBox official source。官方 binary contact label 只作
reference event proxy，并用 G1 named hand/foot 到箱子中心的距离确定角色；不声称是触觉力
或真实 contact wrench。Carry 接触帧 hand-nearest 比例均值/中位数为
`95.46%/97.09%`，Kick 的 foot-nearest 为 `99.78%/100%`；Carry lifted-moving fraction
中位数为 `40.85%`，Kick 为 `0%`。全部自动数据 checks 通过，证明示范中存在清楚的
contact-role、duration 和 motion-regime 标签。

actual rollout corpus 与 predictor gate 已完成。以下命令必须在 retained GPU compute shell
中串行执行；collector 内置项目已验证的 H200 Vulkan、local ground 和临时目录默认值。

```bash
ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
BASE=$ROOT/experiments/demo_following/contact_event_reward_redesign_v1
CORPUS=$BASE/reproduction_goal_core_corpus
DATASET=$BASE/reproduction_phase_dataset
PREDICTOR=$BASE/reproduction_phase_predictor_seed271303
SCALE=$BASE/reproduction_phase_reward_scale

cd "$ROOT"

bash scripts/sugar/demo_reward/collect_deployable_goal_core_corpus.sh "$CORPUS"

$PYTHON_BIN scripts/sugar/demo_reward/audit_actual_contact_event_corpus.py \
  --corpus-root "$CORPUS" \
  --output-dir "$CORPUS/audit_v1"

$PYTHON_BIN scripts/sugar/demo_reward/build_actual_contact_event_predictor_dataset.py \
  --corpus-root "$CORPUS" \
  --output-dir "$DATASET" \
  --policy-observation-key goal_policy_core_observation \
  --alignment-mode clock_phase

$PYTHON_BIN scripts/sugar/demo_reward/train_actual_contact_event_predictor.py \
  --dataset-root "$DATASET" --output-dir "$PREDICTOR" \
  --epochs 20 --batch-size 128 --num-workers 4 --early-stop-patience 5 \
  --seed 271303 --device cuda:0

$PYTHON_BIN scripts/sugar/demo_reward/calibrate_actual_contact_event_predictor.py \
  --dataset-root "$DATASET" --predictor-dir "$PREDICTOR" \
  --batch-size 256 --num-workers 4 --device cuda:0

$PYTHON_BIN scripts/sugar/demo_reward/audit_deployable_demo_event_reward.py \
  --corpus-root "$CORPUS" --dataset-root "$DATASET" --predictor-dir "$PREDICTOR" \
  --output-dir "$SCALE" --unrelated-motion-id 21 --device cuda:0
```

collector 的 actual contact 是左右手/左右脚 named body 对 `/Obj` 的 filtered
`force_matrix_w_history`，严格按 `0.1 N` 阈值生成；reference binary 只描述 selected-demo
事件。corpus 审计结果为 100 条 CarryBox、99 条 KickBox、8 shards、0 duplicate、0 reset。
Carry 中位 bilateral contact/最长手部事件/最大抬升为 `0.3293/4.60 s/0.490 m`；Kick 中位
foot contact/最长足部事件/足力峰值/最大抬升为
`0.0414/0.22 s/60.23 N/0.0066 m`。

dataset 固定按 source motion ID 做 `80/10/10`（Kick test 为 9）motion-disjoint split。每个
causal base row 只有过去 `10 x 121` 部署侧核心观测、固定 numeric demo 和 `[0,1]` normalized
clock phase；每行配 correct、same-task wrong 和 cross-task wrong。未来 actual contact、
duration 和 regime 只进入 13-D target。

必须使用 `clock_phase`。被否决的 `free_window` 版本在每个时刻独立选 32 个 demo windows 中
误差最小者，使不动的轨迹可以反复匹配静止片段。它虽然通过普通 MAE gate，却让 held-out
Kick 轨迹错误偏好 Carry45。将同一批真实 target 绑定到因果时钟后，validation/test 同时恢复
Carry→Carry45 与 Kick→Kick21 的正确方向。

formal V3 predictor 有 `11,386,010` 参数，seed271303 冻结 epoch 20。validation/test full
MAE 为 `0.1771/0.1560`，constant 为 `0.2803/0.2566`，zero-demo 为 `0.2945/0.2766`，
permuted-demo 为 `0.2018/0.1761`；median Spearman 为 `0.677/0.694`，12/12 gates 通过。
13 个 uncertainty scale 只由 validation residual 拟合，名义 90% interval 在
validation/test 的覆盖率为 `97.13%/97.77%`，test 最低单目标为 `91.86%`。

最终 scale audit 固定 CarryBox45/KickBox21 并在完整 motion-disjoint corpus 上逐帧评分。
validation/test 都更偏好匹配任务；held-out Carry 的 correct/unrelated mean feedback 分别为
`+0.01463/-0.01737`。冻结 baseline 为 `0.5322520137`，`eta=0.2427623309`，reward clip
为 `0.1431077421`，平均绝对 feedback 是现有 task/constraint reward 的 25%。reward 定义为
`eta * (exp(-calibrated_event_risk) - baseline)`，不是 potential difference；它有意改变策略
目标。runtime 前 9 个 transition 不足 10-frame history 时精确为零，模型为 eval mode 且
trainable parameter 为零。

这些结果证明 reward 在 official-Tracker corpus 上满足因果、部署输入、phase 和双向语义
门槛，不证明它能迁移到 Refiner-policy rollout，也不证明 policy semantic following。

策略侧接入已完成。`FrozenPhaseAwareDemoEventScorer` 在线截取 policy
observation 的前 121 维，维护每环境 10-state reset-safe history，并以
`(episode_step + 1) / 650` 提供因果 phase；不足十个状态或 benign terminal 时 reward 为零。
它在原有 task/SMP/original-ICM reward 完成后追加冻结 dense feedback，predictor 始终 eval、
零 trainable parameter，未来 contact/event target 不进入 runtime 或 actor。

无仿真 admission：

```bash
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --dry-run
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm unrelated \
  --endpoint-updates 64 --stop-after-segment --dry-run
```

两条命令固定同一个 CarryBox45 teacher、`161587/161588` sim/action seed、20 env、physics、
optimizer、reward mix 和 update budget；唯一科学变量是 `selected_option=correct/unrelated`。
checkpoint 为 update 32/64，训练时必须带 `--policy-training-authorized` 且两臂串行。两臂
endpoint proof 通过后执行：

```bash
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 phase_event_reward_only
```

该入口在 evaluation seed `171587` 下分别加载 update 32/64，每个 checkpoint 使用 20 个
matched physics profiles；随后按 checkpoint 独立计算不读取 predictor/reward 的 lift、
lifted/ground transport、orbit 和 hand/foot contact 审计。最终视频取 update-64 profile 0，
分别只显示 Carry45/Kick21 输入 demo 与对应实际策略行为。该训练、评估、审计和渲染已经完成；
结果见 3.10 节。

正式内层 runner admission 可在 retained GPU 上单独执行，不创建环境、不写 checkpoint、
不执行 PPO：

```bash
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --runner-admission-only
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm unrelated \
  --endpoint-updates 64 --stop-after-segment --runner-admission-only
```

2026-08-24 在 H200 job257762 上两臂均返回
`sugar_phase_event_policy_admission_only_v1/pass`：correct 为 CarryBox45 motion45/demo row37，
unrelated 为 KickBox motion21/demo row97；两者均为 121-D、10-state history、clock-phase、
eval mode、0 trainable parameter、0 environment、0 policy update。完整本地日志位于 ignored
`experiments/runtime_allocations/job257762/`。

正式 online reward gate 会创建原始 SUGAR G1/CarryBox 环境并执行恰好一个 24-step rollout，
调用 actor、frozen Refiner、SMP、original ICM、phase-aware scorer 和 rollout storage，但不调用
任何 optimizer 或 `update()`：

```bash
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --runner-rollout-smoke-only
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm unrelated \
  --endpoint-updates 64 --stop-after-segment --runner-rollout-smoke-only
```

输出协议为 `sugar_phase_event_online_rollout_gradient_authority_smoke_v3`。它要求 24 步全部完成、
10-state history 进入 ready、demo reward 非零、`policy_reward=base_reward+demo_reward`、
original ICM 逐元素不变、policy parameter 和 optimizer counter 不变、ICM optimizer 为零，
并证明 scene 中没有 R15/TacSL sensor 或 elastomer body。旧实现虽然把 tactile tensor 置零，
仍构建了双 R15 scene；`NoTactileGoalRobotEnvCfg` 已修正该混杂。

同一 probe 还在不调用 optimizer 的情况下验证 reward-to-gradient 路径：保存 rollout 的 total
reward，减去每步 selected-demo feedback 得到唯一 counterfactual base reward，分别调用同一个
PPO GAE/normalized-advantage 计算，再对精确 clipped actor surrogate 求梯度，最后恢复 total
storage。correct 的 return/advantage/actor-gradient delta 为
`0.454116/0.253416/0.0780353`；unrelated 为
`0.234895/0.169226/0.0442988`。这证明 feedback 会改变策略将要接收的学习方向，但没有执行
parameter update，也不是行为结果。

该门禁还逐步核对 residual authority：`teacher_coefficient=1.0`、`residual_scale=1.0` 时，正式
公式是 `executed=teacher+residual`，不是 `(1-teacher_coefficient)*residual`。两臂 sampled
residual maximum 均为 `3.726743`；wrapper 公式和 ActionManager raw input 逐元素精确。由
joint scale/offset 逆变换得到的 policy-unit readback 最大误差为 `4.768e-7`，通过既有
`2e-6` float32 round-trip 容差。因此 fixed CarryBox45 teacher 没有遮蔽 student action。

2026-08-24 的最终 gate 已在 fresh H200 job257815/server54 顺序通过。最小
`SimulationApp` canary 先以系统 NVIDIA ICD
`/etc/vulkan/icd.d/nvidia_icd.json` 完成五次 update 并正常关闭；随后 correct、unrelated
均返回 `sugar_phase_event_online_rollout_gradient_authority_smoke_v3/pass`。correct 的 24 步 mean demo reward
非零且 policy reward 逐步等于 base reward 加 demo reward；unrelated 通过相同不变量检查。
两臂均报告 no-TacSL scene、24 步完整执行、history ready、future labels hidden、frozen event
model、policy/ICM 参数不变以及 optimizer/update counter 为零。

由于 smoke 不执行优化，两臂的 24 步 action summary 和 base policy reward 逐步完全相同。
在 history ready 的 16 步中，correct/unrelated 的 mean demo reward 分别为
`0.0401299449/0.0173427592`，mean calibrated risk 为
`0.3604801279/0.5055444967`；correct-minus-unrelated mean reward delta 为 `0.0227871857`。
这是一项受控的 online selector-sensitivity 检查：差异来自 selected demo，不来自 rollout
行为或 task reward 差异。它不等于经过训练的策略已经服从 correct demo。

probe 的外层 launcher 还必须读取内层写出的独立 JSON result，并核对 protocol、pass 和零
policy update。Isaac Sim shutdown 可能掩盖内层 Python 异常并留下零 process return code，
因此 subprocess return code 单独不构成通过证据。

job257762/server60 与 job257794/server45 曾在 Vulkan `ERROR_DEVICE_LOST` 后留下 Kit 子进程；
按准确 PGID 清理后，同一 GPU 上的最小 canary 仍失败。不要把这种 allocation-local GPU
runtime 损坏解释为模型结果，也不要在同一块已 device-lost 的 GPU 上采集 Isaac 证据。
runner 现与通过的 canary 一致，显式设置系统 NVIDIA ICD。online gate 已满足，随后完成的
matched policy optimization 见 3.10 节。

训练前的 shared-teacher prerequisite 使用：

```bash
TEACHER_ONLY_GATE=1 \
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 same_teacher_reward_only
```

输出目录固定为 ignored
`experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161581/`
`teacher_only_gate_no_tactile_v2/`。该 evaluator 不再实例化 TacSL scene；RESULT 必须同时
通过 exact-zero residual、nominal PhysX readback、no-tactile scene、双手接触和至少 5 cm
抬升。job257815/server54 的 20 个 nominal profile、400 control steps 均通过：residual
absolute maximum `0`，最大抬升 `0.685389--0.722354 m`，双手接触 `153--156` 帧，物理跌倒
`0/20`。这证明共同冻结 teacher 是有效起点，不证明尚未训练的 selected-demo policy 行为。

正式 phase-event proof 还必须包含
`no_tactile_startup_physics`：逐环境保存 standard SUGAR startup randomization 产生的完整
object/robot material tensor 以及 object mass、inertia、COM。冻结 evaluator 不再尝试调用
TacSL-coupled latent event，而是从对应训练 proof 写回这些值并进行 PhysX readback。修正后的
job257815 correct/unrelated 24-step smokes 都通过
`no_tactile_startup_physics_recorded`，两臂保存的物理数组完全一致；动作与 base reward 的
逐步最大差仍为 `0`。缺少该记录的 phase-event checkpoint 不得进入冻结评估。

### 3.10 Phase-event matched policy result 与 transfer failure

正式结果根目录为：

```text
experiments/demo_following/matched_phase_event_reward_v1/seed161587/
├── correct/update_0064/
│   ├── policy_update32.pt
│   ├── policy.pt
│   └── proof.json
├── unrelated/update_0064/
│   ├── policy_update32.pt
│   ├── policy.pt
│   └── proof.json
├── evaluation_update0064/{correct,unrelated}/{RESULT.json,TRACE.npz}
├── behavior_adherence_update0032/RESULT.json
├── behavior_adherence_update0064/RESULT.json
└── videos_update0064/
    ├── 01_correct_demo_and_actual_behavior.mp4
    ├── 02_unrelated_kickbox_demo_and_actual_behavior.mp4
    └── RENDER_PROOF.json
```

两臂各执行 64 updates，proof 均通过 65/65 checks；actor maximum parameter delta 为
`0.0055189/0.0066973`，整个 policy delta 为 `0.0167299/0.0169497`。两臂都使用原始
no-TacSL SUGAR scene，startup mass/inertia/COM/material 完全相同，selected demo 分别是
CarryBox45 row37 与 KickBox21 row97。

冻结 evaluator 在一个 40-env scene 中同时装载 update 32/64，每个 update 占 20 profiles。
训练 proof 中 batch-shaped wrapper state 只有 `release_latched=false`、`release_progress=0`、
`teacher_coefficient=1` 三个不变量，验证后才允许扩展 20 到 40。20 个 startup physics
profiles 精确重复到两个 update slice；PhysX COM readback 只允许一个 float32 epsilon，其他
字段仍用 `1e-7`。source origin 不一定落在 env0，因此 first teacher action gate 使用离记录
source origin 最近的 replica；本次 canonical env 为 15，action error `6.56e-7 < 2e-6`。

update 64 的 predictor-independent mean behavior：

- correct/unrelated maximum lift：`0.694533/0.694187 m`；
- bilateral hand-contact fraction：`0.833484/0.832515`；
- lifted-frame fraction：`0.612924/0.612137`；
- lifted-transport fraction：`0.941271/0.942186`；
- ground-transport fraction：`0.058729/0.057814`；
- root orbit rate：`0.374611/0.371613 rad/s`；
- any foot-box contact：`0.002878/0.002877`；
- physical fall：两臂均 `0/20`。

update 32/64 分别只有 `2/4`、`1/4` 预登记 Kick-like 方向，而且主要差值只有 `1e-3` 量级。
两条最终 H.264 视频均完整解码，并各自只显示 official input demo 与 actual policy behavior。
结论是两臂都保留稳定 Carry 解，没有 semantic separation。

需要单独记录 scorer failure。correct update-64 actual rollout 的 mean Carry45/Kick21
predicted mismatch 是 `0.969861/0.890871`；unrelated arm 是 `0.970261/0.892214`。即使实际
行为明显是 Carry，scorer 在多数中段 frames 仍偏好 Kick21。训练时 feedback 已进入 PPO 并
改变 checkpoint，但错误的在线语义使“有梯度”不能推出“会按 demo 改行为”。

严格 scorer-only transfer audit 已完成。首先用同一 frozen checkpoints、同一 physics profiles
重采 exact `goal_policy_core_observation [401,40,121]`，以及每个 selected-demo scorer 的
phase/ready/risk/weighted-uncertainty/reward `[400,40]`。然后在不启动仿真、不更新 policy、
不更新 predictor 的条件下，用 frozen 11.386M predictor 逐帧复现旧 runtime，再只改变第一段
episode 的 phase 起点。复现最大绝对误差为 phase `2.98e-8`、reward `4.35e-8`、risk
`4.77e-7`、uncertainty `3.58e-7`，ready exact equal。

reset-zero 结果在 correct/unrelated 两臂的 update 32/64 四个 block 中均错误偏好 Kick：
`Kick risk - Carry risk = -0.08239/-0.08158/-0.08142/-0.08049`，Carry-preferred frame 为
`30.13%/30.36%/30.25%/30.59%`，每个 block 均为 `0/20` profiles。reference-aware 版本令
第一段 episode 从真实 source reference frame `197` 起钟，后续自然 reset 仍从 0 起；四个
margin 变为 `+0.32437/+0.32724/+0.32451/+0.32787`，Carry-preferred frame 为
`85.77%/86.10%/85.71%/86.12%`，每个 block 均为 `20/20` profiles。phase-only necessary
Carry gate 因此通过。正式 scorer、训练 runner 和 frozen evaluator 已统一从 command 的
reset reference frame 初始化 causal clock。

运行完整审计的入口是：

```bash
OUTPUT_ROOT="$PWD/experiments/demo_following/reproduce_phase_transfer" \
PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  bash scripts/sugar/demo_following/run_phase_event_scorer_transfer_audit.sh
```

关键本地证据为：

```text
experiments/demo_following/matched_phase_event_reward_v1/seed161587/
├── scorer_transfer_source_trace_v1/{correct,unrelated}/{RESULT.json,TRACE.npz}
└── scorer_transfer_phase_ablation_v1/{RESULT.json,SCORES.npz}
```

Tracker-to-Refiner distribution shift 仍被量化：official Tracker test 的 normalized state
`mean|z|/p95/p99` 为 `0.6679/1.9230/2.8818`，correct policy rollout 为
`1.0350/3.2120/5.4203`；偏移最大的组包括 joint position、projected gravity、box linear/
angular velocity 和 previous action。但 phase 修正本身已经足以恢复 Carry-domain 语义方向，
所以不能再把当前倒置归因于 domain shift。motion-disjoint official Generator/Tracker Kick
反向门槛已可从现有 121-D corpus 独立复现：

```bash
$PYTHON_BIN scripts/sugar/demo_following/audit_heldout_kick_tracker_scorer_transfer.py \
  --corpus-root experiments/demo_following/contact_event_reward_redesign_v1/deployable_goal_core_corpus_v1 \
  --runtime-config experiments/demo_following/contact_event_reward_redesign_v1/phase_aware_dense_feedback_scale_audit_v1/RUNTIME_CONFIG.json \
  --output-dir experiments/demo_following/reproduce_heldout_kick_gate \
  --device cpu
```

该 gate 强制回读 official `generator.ckpt + tracker.pt` provenance、0.1 N filtered physical
force threshold、无 reset、9/9 foot interaction 和 9/9 至少 1 cm 平移。部署 fixed-650 clock
得到 mean `Kick risk - Carry risk=-0.06508`、`8/9` profile preference 和 `50.50%` ready-frame
preference；motion29 是保留的反例。source-duration clock 的 `9/9` 仅为 evaluation diagnostic。
官方发布物中没有 frozen Kick Refiner/residual checkpoint，因此该结果不能冒充 Refiner transfer。
corrected online 与 frozen Carry gate 已随后在 retained H200 job258074 上正式通过，证据根目录为：

```text
experiments/demo_following/corrected_phase_runtime_gate_job258074_compute_v3/
├── online_smokes/{correct,unrelated}.json
├── frozen_carry/source_evaluations/{correct,unrelated}/{RESULT.json,TRACE.npz}
├── frozen_carry/scorer_audit/RESULT.json
└── heldout_kick_tracker/{RESULT.json,SCORES.npz}
```

两条 online smoke 都记录 `initial_episode_steps_min=max=197`、0 policy update 和参数不变；
ready-step mean reward/risk 为 `+0.04804/0.31539` 与 `-0.00338/0.65776`。frozen Carry 四个
arm/update blocks 均为 `20/20` profiles 偏好 Carry，margin 为
`+0.32437/+0.32724/+0.32451/+0.32787`。同一 pipeline 的 CUDA Kick gate 重现 `8/9` 结果并
返回总 `RC=0`。job258067 的物理 GPU0 曾在 Isaac 启动时发生 `ERROR_DEVICE_LOST`，因此没有
复用作 Isaac 证据；job258074 的独立物理 GPU7 完成全部门槛。两个 retained allocations 均在
门槛后保持 GPU hold。下一步只有在用户明确授权后才从新目录重跑一组 from-scratch
64-update matched policy experiment。预登记入口是：

```bash
# 仅在用户明确授权后，在 retained srun compute step 内执行；两臂串行且不会自动评估。
DEMO_POLICY_TRAINING_AUTHORIZED=YES \
OUTPUT_ROOT="$PWD/experiments/demo_following/matched_phase_event_reward_reference_aware_v2" \
  bash scripts/sugar/demo_following/run_reference_aware_phase_event_pair.sh
```

该入口固定 `161587/161588`、20 env、CarryBox45 common teacher、update 32/64、相同 physics/
optimizer/reward；唯一变量是 CarryBox45 或 KickBox21 selected demo。每个 arm 完成后必须通过
完整 proof，尤其是 reference-frame-197 causal phase 检查。脚本完成两臂后停止，不自动追加
更新、seed、冻结评估或渲染。

研究依据是：DeepMimic 将 imitation objective 与 task objective 分开；PhysHOI 使用 contact
graph 防止错误 body-object interaction；InterMimic 同时约束 object deviation、joint-object
关系和 required-contact duration；CHORD 进一步用 object-centric contact wrench 衡量接触
如何驱动物体，而不只看接触位置。对应原始来源：

- https://arxiv.org/abs/1804.02717
- https://arxiv.org/abs/2312.04393
- https://arxiv.org/abs/2502.20390
- https://nvidia-isaac.github.io/video_to_data/chord/

## 4. Earlier trajectory-only predictor

这是 contact/event redesign 之前保留的 11.9M trajectory-only 模型。输入为过去 10 帧
policy state 和指定 demo condition，目标只有未来 body、box position、box rotation 6D 和
box velocity mismatch。它证明 demo identity 被读取，但从未建立可靠语义遵循，也不是当前
reward。当前 phase-aware 13-target 模型与 dense feedback 见 3.9 节。

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
semantic extension 失败，`policy_integration_supported=false`。该结果是严肃负结果，不是
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
