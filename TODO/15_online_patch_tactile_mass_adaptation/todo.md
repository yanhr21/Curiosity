# TODO 15: Online Whole-Hand Patch Tactile for Sudden-Mass Adaptation

> **Frozen 2026-08-22.** Old checked runs record execution only and remain invalid evidence. The
> corrected line stopped without a valid matched Z/P/PS comparison and must not resume implicitly.

## 2026-08-20 audit and source correction

- [x] anatomical pads excluded from `undesired_contacts`.
- [x] reward-facing contact 改为 54 个 IsaacLab 支持的一-pad-to-box filtered ContactSensors，
  每手聚合 27 个；禁止单个 regex filtered sensor 覆盖多 pad。
- [x] explicit post-handoff box-lift/hold reward added; teacher prefix receives zero.
- [x] TacSL `F_normal` and `F_friction` outputs separated; patch utilization uses full friction
  magnitude rather than cancellation-prone signed XY sum.
- [x] Plan 15 PhysX box/pad and TacSL friction fixed to the same `0.5` contract.
- [x] same-step reset handling moved ahead of patch/slip cache returns.
- [x] training and evaluation motion source unified to `CarryBox/data_045`.
- [x] formal launcher no longer passes physical-outcome continuation; summarizer/comparator reject
  non-strict results.
- [x] percentile bootstrap removed; future inference uses exact paired training-seed sign flips
  with Holm familywise correction.
- [x] 用户已明确批准 corrected rerun。
- [x] schema-v2 scale、三分支 preflight、固定 3x PS overfit 和 fresh formal 单-seed 入口
  已建立；入口拒绝旧 scale 与既有 output directory。
- [x] 独立 PhysX audit 分开记录 normal/friction；修正了把 `force_matrix_w` 当作总力的
  审计错误。
- [x] 历史 nominal 十帧 aggregate 曾得到 PhysX `56.9645/15.7513 N` 与 TacSL
  `56.9645/16.2076 N`；动态逐-pad audit 已否定其 calibration admission，旧 scale 撤销。
- [x] force audit 改为单 box sensor body + 27/54 exact pad filters；unfiltered box normal 与
  54-pad filtered normal 的 L2 mean error 为 `4.77e-8 N`，旧 multi-body regex audit 禁用。
- [x] corrected motion45 scale sweep 与 Z/P/PS runtime preflight：15 条 paired traces、
  action/jump exact match、504-D no-object-state actor、slip precision/recall
  `0.99285/0.97938`，Z/P/PS 三分支 overall pass。
- [x] 54 个 reward sensors 的 live PS preflight：无 PhysX filter-count error，`hoi_contact`
  非零，`30/360` handoff 后 transitions，1 次真实 mass change。
- [x] 首轮 `model_250` strict review：`0/4` eligible，3 条 jump 前终止，唯一 jump 条只有
  9-frame post-jump window；该 run 因旧 reward sensor 运行时无效而归档。
- [x] 清除 3 个遗留并发 writer；受污染的 v2 checkpoint/evaluation 已归档。overfit 与
  formal launcher 增加跨目录稳定 pipeline lock，运行时第二 writer probe 返回 exit 75。
- [x] 发现首批 model-0/250/500/750 review 仍用 formal 随机 delay/pose，与 fixed overfit 不匹配；
  相关数字降级为 harder-distribution diagnostic，不再作为 overfit gate。
- [x] 增加 `--fixed-3x-overfit-gate`，强制 PS/3x/20-frame delay/零 reset-pose noise，并在
  summary 写入实际配置。
- [x] model-500 同一 rollout 的 CarryBox world + bilateral 27-patch H.264 同步视频已完整
  解码并人工抽帧检查；它仅是行为证据，不是 tactile-benefit 结论。
- [x] fixed gate 重评 model-0/250/500/750/1000：全部 `0/4` hold、`0` fall；model-750
  达到 `66--72` 帧后早停，model-1000 退回 `19--21` 帧，均未通过 80-frame gate。
- [x] model-1250 fixed review：`3/4` strict 80-frame hold、`0` drop、`0` fall；证明固定条件
  可学习，但该 checkpoint 早于 taxel-grid 修复，禁止续训或进入 formal。
- [x] 定位 anatomical taxel 中央裁剪：旧网格仅覆盖约 `44--66%` patch footprint；增加
  保留官方默认的 extent-fill adapter，理论几何覆盖约 `96--99%`。
- [x] 完成新网格 SDF/contact-offset、逐 pad recall、normal/friction 标定和独立 3x dynamic
  审计；固定 `0.3 mm / kn=7294.8755 / kt=9 / mu=0.5`，撤销旧参数与 schema-v2 scale。
- [x] 完成 3-seed x 5-mass v3 sweep、causal-slip gate 和 Z/P/PS runtime preflight；15 条
  jump/readback/contact 合同全部成立，slip precision/recall `0.9914/0.9856`。
- [x] 新 calibration 后从 official Tracker 重跑 fixed 3x PS overfit，不续训 model-1250；
  fresh model-750 strict review 为 `3/4` hold、`0` drop/fall、三条完整 98-frame window，
  learnability gate 已通过。
- [x] fresh corrected Z/P/PS training and evaluation 未完成即冻结；没有形成可比较的正式
  结果。旧 Z/model2999、P/PS 中间端点和违规读取 TacSL 的评测均不得进入未来 comparison。

## Current status (2026-08-22)

- [x] correct all nine audited source/evaluation/statistics defects and pass regression tests;
- [x] run a bounded tactile-only stability diagnostic: model1100 reaches `14/20` physical holds but
  only `6/20` strict successes; model1250 is worse;
- [ ] no valid corrected matched Z/P/PS result exists;
- [x] freeze this queue and return priority to demo following. Do not resume implicitly.

## A. 固定合同

- [x] 本实验 backend 固定为 IsaacLab/PhysX；该规则描述 Plan 15 执行期，不再定义当前
  仓库优先级。
- [x] actor 固定为 `504-D` deployable Tracker-command/proprioception，不含 measured
  object state、mass factor、jump flag、RGB 或 future frame。
- [x] 双手固定 `2 x 27` physical patches；taxel 只作为 TacSL backend/audit source。
- [x] 每 patch 在线输出 contact、normal load、pressure、signed XY shear、friction
  utilization；PS 再输出 causal slip score/state。
- [x] Z/P/PS 共用 official Tracker warm start、frozen Refiner、BCPPO、
  `512/256/128` actor、29-D action、physics、reward、seeds 和 3000-update budget。
- [x] stage-3 distillation floor 固定 `0.25`。
- [x] formal endpoint 固定 `model_2999.pt`；不得延长或自动启动下一 seed。

## B. 在线 sensing 与 slip

- [x] live mass/inertia event 在同一 episode、两个 actor calls 之间写入并读回。
- [x] Refiner pickup -> no-reset student handoff -> matched delay -> real mass event。
- [x] leakage sweep 完成 3 seeds × 5 mass factors；公共 patch scale 冻结。
- [x] jump 前 10 帧 bilateral contact gate、54-patch clock 和 event pairing 通过。
- [x] `PatchSlipDetector.update(...)` 为 causal、batch-stateful，reset mask 可清空历史。
- [x] controlled R15 slip 与完整 CarryBox 3x held-out velocity 评价完成。
- [x] frozen reaction-window audit 完成：continuous patch 相对 binary contact 有更早且
  更完整的信息覆盖。

## C. 正式训练与冻结评估

- [x] Z seed `151014`：endpoint + 100 rollouts。
- [x] Z seed `151015`：endpoint + 100 rollouts。
- [x] Z seed `151016`：endpoint + 100 rollouts。
- [x] P seed `151014`：endpoint + 100 rollouts。
- [x] P seed `151015`：endpoint + 100 rollouts。
- [x] P seed `151016`：endpoint + 100 rollouts。
- [x] PS seed `151014`：endpoint + 100 rollouts + synchronized video。
- [x] PS seed `151015`：endpoint + 100 rollouts + synchronized video。
- [x] PS seed `151016`：从 official Tracker 启动，严格训练到 `model_2999.pt` 后停止。
- [x] 审查 PS-151016 checkpoint finiteness、handoff、mass readback、80-frame window、
  action continuity 和同步 54-patch video。
- [x] 显式运行 `151016->152016` 五质量 × 20 profiles frozen evaluation。
- [x] 运行 exact three-seed Z/P/PS paired comparison；每分支恰好 300 profiles。

上述 endpoint/evaluation 全部属于旧失效 pipeline，只保留追溯，不再报告分支优劣或
置信区间。

## D. Endpoint 评审清单

每个新 endpoint 必须逐项完成后才进入下一 seed：

- [x] checkpoint iteration=2999，model/optimizer tensors finite；
- [x] live Refiner 在同一 PhysX episode 完成抬箱和 no-reset handoff；
- [x] requested/readback mass 与 inertia event 一致；
- [x] nonnominal profiles 在 event 前 10 帧有 bilateral patch contact；
- [x] 450-frame trace 完整覆盖 handoff、event 和 80-frame outcome；
- [x] applied action 与 saved policy action 对齐，无 gross handoff/jump discontinuity；
- [x] H.264 完整解码，画面包含 G1/CarryBox、左右各 27 patches 和 evaluator-only event
  标注；
- [x] camera video 只描述自己的 camera rollout，不冒充 camera-free formal trace。

PS-151015 与 PS-151016 均完成以上清单。

## E. 高摩擦 6x/10x

- [x] 仅在 Z/P/PS 正式比较完成后启动。
- [x] 分别测试 static/dynamic friction
  `0.5/0.5、1.0/1.0、1.5/1.5、2.0/2.0` × `6x/10x`。
- [x] 每条保存 PhysX material readback、mass readback、post-jump height/contact 和 outcome。
- [x] 不把高摩擦结果混入原 Z/P/PS comparison。
- [x] `6x, mu=1.5` 已满足 hold，因此 stronger-grip/lower-posture/overfit 条件分支未触发。
- [x] 为 `6x, mu=1.5` 成功条件生成并审查同步 G1/CarryBox/54-patch H.264；该
  camera-enabled rollout 高度损失 `0.02552 m`、hold=true、drop=false。

Camera-free 6x height loss 按 `mu=0.5/1.0/1.5/2.0` 为
`0.5589/0.5429/0.02636/0.06596 m`；只有 `mu=1.5` hold。10x 四个条件全部 drop。

## F. 交付

- [x] experiments 只保留正式 endpoint、冻结评估、关键 sensing 和人眼证据；中间
  checkpoint、旧 runtime、失败/重复实验已移入根 `legacy/`。
- [x] 旧 Plan-13、旧 bundle renderer 和 Newton simulator adapter 已从活动代码移出。
- [x] 完成 friction 结果后更新 README、Plan、TODO、AGENTS。
- [x] 最终 commit/push 只包含源码、测试和文档；checkpoint、trace、视频和日志保持
  在 ignored `experiments/`。

## G. 审计后重开项 / Reopened by the 2026-08-19 audit

A–F 的勾选反映"按当时的合同执行完毕"，仍然有效。以下条目是审计新开的，**在修复前
§7/§C 的结论不能作为科学结果发布**。完整证据见
[`claude_context/findings.md`](../../claude_context/findings.md)（115 条，含 file:line）。

### G1. 阻塞项 — 必须修复后才能重跑

- [ ] **Reward**: exclude `.*_anatomical_.*` from `undesired_contacts`. It currently
      counts all 54 patches at −0.02/body/step, and six in contact cancel the entire
      achievable positive reward of 5.125.
      **Where:** `carry_box_refiner_env_cfg.py:86-98` — the regex is **line 93**. Sensor is
      all bodies at `base_refiner_env_cfg.py:72`; reduction at
      `IsaacLab/.../envs/mdp/rewards.py:260-268`. **One regex.**
- [ ] **Reward**: repoint `hoi_contact` at the 54 patch bodies — the `*_rubber_hand`
      links it reads have their collision subtrees deactivated at spawn, so the term is
      dead and supplies no behavioural gradient.
      **Where:** term `carry_box_refiner_env_cfg.py:99-112`; impl `mdp/rewards.py:142-172`
      (**line 167**); cause `assets/robots/anatomical_whole_hand_tacsl_g1.py:1093`, disable
      at **1136**; dead sensors `base_refiner_env_cfg.py:93-108`.
- [ ] **Reward**: decide whether to add a term that actually rewards holding the box after
      the jump. There is none today; the objective is reference tracking end to end.
      **Where:** the full 21 terms are `base_refiner_env_cfg.py:298-399` +
      `carry_box_refiner_env_cfg.py:84-116`; hold/drop exist only in
      `evaluate_online_patch_mass_bcppo.py:339-360`.
- [ ] **Slip**: hoist the reset out of the `common_step_counter` guard in
      `_online_patch_slip_history` — the early return skips `detector.update`, so
      `env.reset()` (which does not bump the counter) is silently swallowed and the GROSS
      latch survives across evaluation batches. **Asymmetric: damages PS, not P.**
      **Where:** `online_patch_tactile.py:446` returns early, before `detector.update` at
      **460**. Counter only moves in `step()`: `IsaacLab/.../manager_based_rl_env.py:203`.
      Bypassing reset: `evaluate_online_patch_mass_bcppo.py:546`. **A few lines.**
- [ ] **Sensing**: resolve normal/tangential in the **contact frame** (the SDF normal is
      already computed) instead of the per-taxel frame; gate the utilization ratio on a
      minimum normal load; compute friction margin against the object's real material.
      Note the per-taxel frame is a *local* modification — upstream IsaacLab v2.3.2 uses
      one constant quaternion per sensor — so no vendor change is required for this.
      **Where:** `visuotactile_sensor.py:1018` sums normal+friction, **1023-1024** split the
      total; the per-taxel frame is **564-608**. The `mu` division is
      `online_patch_tactile.py:144-146` with the sensor's `mu` at **237**; the numerator cap
      is `visuotactile_sensor.py:1005-1007`; real friction randomized at
      `base_refiner_env_cfg.py:274-281`.

- [ ] **Slip**: decide the intended semantics of the GROSS latch. `retained_gross`
      (`patch_slip.py:263-264`) holds a patch at GROSS for as long as *any* incipient
      evidence persists — including a static geometric `utilization >= 0.60` — and it is
      cleared in exactly one place, **line 135**, inside `reset()`.

### G2. 诊断 — 先测再改，不需要 GPU

- [ ] Per-channel variance and mass/friction mutual information **during the hold phase**
      on the saved leakage traces. Confirms or kills the sensing findings in minutes
      instead of a retrain.
- [ ] Report the `strict_sugar_hold_success` / `strict_sugar_eligible` numbers that
      already exist in every `summary.json` alongside the physical-outcome ones. **Free.**
      The view is installed at `evaluate_online_patch_mass_bcppo.py:495`, reset stubbed at
      **547**, frame validity skipped at **729**, and the flag is always passed by
      `run_plan15_frozen_seed.sh:63`.
- [ ] Identify which seed-151014 profile fails `eligible_post_jump_window` (the 59 vs 60);
      needs `episodes[i].jump_frame == null` from the summaries on the runtime host.
- [ ] Check `first_termination_terms` for `trajectory_complete` before the window closes —
      if motion 45 is shorter than 450 frames the reference is silently zero-padded under
      `--physical-outcome-view`, which would void both reference-error columns.

### G3. 实验设计

- [ ] Close the train/eval motion split: either train on motion 45 or evaluate on 0–3.
      As it stands the experiment cannot separate "tactile does not help" from "the tactile
      encoder did not generalise across motions". **Where:**
      `carry_box_online_patch_tactile_mass_env_cfg.py:259` sets `start_init_env_ratio = 1.0`;
      the evaluator overrides at `evaluate_online_patch_mass_bcppo.py:479-487`, default 45 at
      **line 41**.
- [ ] Recalibrate the slip thresholds on the **anatomical** geometry — they were fitted on
      a flat R15 capsule where the dominant misalignment term does not exist.
- [ ] Restate the CI honestly: percentile two-level bootstrap over 3 seed clusters, no
      BCa, 180 uncorrected intervals per run. **Where:**
      `compare_online_patch_mass_sweeps.py:119` (plain `np.percentile`), **196** (the method
      string it writes), **line 21** `METRICS` x 5 factors x 3 pairs. **Free.**
- [ ] Note in any write-up that the 20 profiles per run are near-replicates.

### G4. 训练循环（低成本，与重跑一起做）

- [ ] Decide whether the distillation loss should be masked by the handoff mask. Today it
      is not, so the nominal-mass Refiner regresses the student inside the post-jump window
      it has never seen (teacher trained on 0.5–2.0× only).
- [ ] `configure_tactile_actor_finetune` overrides the parent freeze **without calling
      `super()`** and installs no gradient mask — either call it or rename the method. The
      504 warm-started Tracker columns currently train from update 0.
- [ ] Persist `alg.last_training_mask_report` in formal runs. A logged `surrogate = 0.0`
      currently means "no post-handoff transitions this update", indistinguishable from
      convergence.
- [ ] Either honour `BCPPOCfg.learning_rate` or remove it — the warm start overwrites every
      param group with the Tracker checkpoint's LR.
- [ ] Add a terminal `LayerNorm` to the pre-LN patch encoder, or otherwise address that
      only ~3 % of the 128-D embedding's norm varies with the tactile input.
- [ ] Bind a `patch_channel_scales.json` to the channel definitions that produced it —
      nothing does today, and the scales are baked into every checkpoint.

### G5. 不需要动

Branch matching、mass/inertia event 与 readback、Z 的 gradient isolation、510→504 warm
start 与 `2e-6` audit、整个 dimension contract、eligibility gate 的 branch/factor
invariance —— 这些都通过了审计，重跑时不要改。
