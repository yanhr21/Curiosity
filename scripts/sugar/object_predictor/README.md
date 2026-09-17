2026-09-16：按用户最新要求暂停实验推进；当前实验与自动队列已停止，GPU保留。正在由三个subagent并行进行物理/观测、训练/泛化、模型/监督目标的根因调查；未经用户明确恢复不重启实验。 [根因调查与下一步判别建议](../../../experiments/object_predictor_v1/research_review_20260916/ROOT_CAUSE_REVIEW.md)：已有相同允许输入却不同质量标签的确定反例；可辨识离地子集仍存在5.02%输入支持量对52.58%模型误差的差距；物理采集、训练目标和当前已知box回归范围分别审查。仅调查，不代表恢复实验。

2026-09-16 22:17：用户要求的[6008完整搬运rollout视频](../../../experiments/object_predictor_v1/prospective_carry_v1/selected_rollout_6008/index.html)已完成：48秒正常速度、480帧全部解码，独立subagent查看首末帧及实际MP4确认；这是成功搬运展示选例，质量/姿态误差仍保留。全16连续视频仍在原pipeline渲染，完整数值[报告](../../../experiments/object_predictor_v1/prospective_carry_v1/REPORT.md)仍不通过可靠性验收。新增实质进展：隔离官方torch2.4.1/cu121/PyTorch3D0.7.8安装与CPU导入完成，48/48官方网格和48/48原始点云标签处理全部通过。完整官方原生输入0/5触摸对照已准备，实际CPU loader检查48物体/240组通过（8组无有效接触保留），GPU前向/梯度仍等待原串行队列，0新学习。

官方训练后端入口 `qualify_active3d_training_backend.py` 已准备并完成 CPU 输入/坐标检查：直接导入完整官方 t_p 模型和原 Chamfer，固定 TRAIN 8 时刻、30000 点 × 3 重复、系数 9000，保存真实梯度，0 次优化；其 GPU 执行已排在现有流程之后，尚无执行结果。[具体范围与命令](../../../experiments/object_predictor_v1/research_review_20260916/ACTIVE3D_TRAINING_BACKEND.md)。

形状采集入口：`shape_fixture_assets.py` 读取原始官方完整网格；`fixture_touch_scene.py` 通过原 Newton 动态手和 D6 驱动产生真实接触；`collect_fixture_touch.py` 执行固定四次资格采集；`report_fixture_touch.py` 从保存观测构建单个 25 顶点接触片并回放完整网格。当前只排队采集资格测试，未开始48物体语料采集或形状模型训练。

2026-09-16：先验＋接触＋已知地面固定8帧TRAIN融合完成并停止追加实验汇报；位置2.19→1.88cm、旋转25.43→14.67°、网格3.08→2.10cm，全部预测不穿地，实际8图已检查。仍非泛化、质量改善或任意形状成果。[完整实际渲染与报告](../../../experiments/object_predictor_v1/research_review_20260916/prior_contact_floor_fusion_v1/index.html)。

2026-09-16 08:48：按继续指令启动原48配置的新条件采集资格验证（连续掌侧＋响应增益），复用同条件3条TRAIN，保留原4/8几何组门槛。新增5001完整通过，5002采集中；输入路径三例6窗口44项检查通过。尚无完整数据资格或新模型成绩，不自动启动训练。[当前协议与进展](../../../experiments/object_predictor_v1/response_surface_dataset_v1/README.md)。

2026-09-16 13:11：当前三组各2000步对照及全部8例实际网格预测视频已完成，3840帧解码通过。接触面输入部分改善，可靠搬运预测仍未通过验收；TEST仅1/8成功搬起，离地重量监督未改善质量。已按用户要求停止追加实验。[96秒视频与播放页](../../../experiments/object_predictor_v1/training_response_surface_v1/index.html) · [完整报告与原因分析](../../../experiments/object_predictor_v1/training_response_surface_v1/NUMERICAL_ANALYSIS.md)。


2026-09-16 08:40：测得力响应约束增益的固定三例全部通过原物理／稳定性门槛：5024由未抬升变为离地19.06cm，5000保持成功20.21cm，5007载荷稳定性通过／离地15.00cm（高度较旧参考低，且该例同时改变传感覆盖）。24测试、7200步因果回读通过，720帧实际网格视频及7张图检查完成。无Utonia前向或训练，原32例数据资格与模型负结果仍保留。[视频与完整报告](../../../experiments/object_predictor_v1/response_gain_diagnostic_v1/index.html)。

2026-09-16 08:18：连续掌侧理想传感选项（默认关闭）完成固定两例物理／480帧渲染对照。5024仍失败，5000保持成功／离地20.24cm；同一新5000轨迹末段201时刻的剪切估重误差原27区域13.06%→连续掌侧2.16%，仅单例传感信号结果，无Utonia训练或泛化成绩。18测试、256旧数据积分检查及4800保存帧回读通过，两个运行错误已记录；实际执行4801控制步。[视频与完整报告](../../../experiments/object_predictor_v1/continuous_palmar_diagnostic_v1/index.html)。

2026-09-16 07:58：原32例漏测位置核查完成：50944个未分配面帧全在掌侧区域边界外，分配规则重放差异0；成功搬运末段99.88%的未覆盖标量载荷接近两个掌根区域。尚未扩大传感器覆盖或训练。[实际网格定位与完整报告](../../../experiments/object_predictor_v1/uncovered_skin_localization_v1/index.html)。

2026-09-16 07:50：承重缺口来源分解完成。原成功搬运10例，21.37%剪切支持缺口中约18.38个百分点来自未覆盖接触面，法向2.41、投影0.57；全表面验证参考质量误差3.58%，不属于模型观测或预测改进。全32例保留，无新训练。[实际网格对照与完整报告](../../../experiments/object_predictor_v1/force_deficit_decomposition_v1/index.html)。

2026-09-16 07:43：两例弱接触加速诊断与480帧实际网格并排视频完成，整体失败：5024仍未搬起，5000保持通过／离地20.16cm。修改默认关闭，原训练覆盖仍2/8（需4/8），无新模型训练。[渲染与完整报告](../../../experiments/object_predictor_v1/acquisition_schedule_diagnostic_v1/index.html)。可靠搬运估计仍未完成。

2026-09-16 07:22：32例已有触觉力平衡诊断完成，无新训练。相同10例成功搬运/末段80时刻，仅剪切误差21.37%，加CAD法向载荷63.07%，接触面拟合法向24.04%；CAD几何法向不能直接当有效受力方向。[真实网格力箭头对照](../../../experiments/object_predictor_v1/mechanics_observability_v1/index.html) · [完整报告](../../../experiments/object_predictor_v1/mechanics_observability_v1/REPORT.md)。可靠搬运predictor仍未完成。

2026-09-16 07:11：原32例完整 Utonia 冻结搬运评估与实际预测渲染完成。即使只看10例物理验收通过的离地保持阶段，含力质量误差仍62.47%，几何41.93%，力置零30.16%；可靠搬运预测仍失败，无新训练。四条视频、64张全案例对照和原因分析：[渲染播放页](../../../experiments/object_predictor_v1/frozen_surface_carry_v1/index.html) · [完整报告](../../../experiments/object_predictor_v1/frozen_surface_carry_v1/REPORT.md)。

2026-09-16 06:53：原32条搬运轨迹的完整 Utonia 冻结评估已完成17例，剩余继续运行；四张实际物体／预测物体预览已核查。5000成功抬升约20cm，但末帧质量0.454kg被含力模型估为1.041kg，预测问题仍在。全量汇总与四条视频将在评估结束后自动生成，无新训练。 [成功搬运时的真实与预测对照](../../../experiments/object_predictor_v1/frozen_surface_carry_v1/renders/episode_5000/frame_2381.png)。

2026-09-16 05:35: Saved34-trace contact layout and officialhandSDF inversion complete. Whole-hand inversion fails5000left; pad-local inversion passes allfour fixed50frame comparisons/200handframes,5014 errors5.41/5.59->.97/1.11deg,5000 .445/1.153->.278/.573deg. Current equal-kh/localunitSDF assumption;5000left~89.76%area usable. All5finalactualmesh/layout images inspected. [Report](../../../experiments/object_predictor_v1/sustained_grip_diagnostic_v1/contact_layout/REPORT.md) / [renders](../../../experiments/object_predictor_v1/sustained_grip_diagnostic_v1/contact_layout/index.html). No controller/model integration,newphysics oroptimization. Full32validation pending; carrying predictor and prior mass failures unresolved.

2026-09-16: 物体 SDF128→256 数值误差降低，但5011仍未搬起、5000保持通过；[实际网格方向图、并排视频与完整报告](../../../experiments/object_predictor_v1/sdf_resolution_diagnostic_v1/index.html)。无新训练。

2026-09-16 06:03: 固定两例对齐停止诊断完成，5005仍失败、5000保持成功，无新训练；[真实网格并排视频与完整报告](../../../experiments/object_predictor_v1/alignment_deadband_diagnostic_v1/index.html)。

2026-09-16 05:49: 全 32 TRAIN 接触方向校正验证完成：3200 手帧均有输出，但 13/64 案例／手退化超过 0.5°；未接控制器或训练。实际失败网格渲染与完整报告：[报告](../../../experiments/object_predictor_v1/sustained_grip_diagnostic_v1/contact_layout/ALLTRAIN_REPORT.md)。

2026-09-16 continuation: [full matched grip transfer + surface-feedback report](../../../experiments/object_predictor_v1/grip_transfer_v1/REPORT.md), [24-second true-mesh tactile-plane rendering](../../../experiments/object_predictor_v1/grip_motion_dataset_v1/index.html). Three bounded surface-feedback diagnostics complete: physical lift improves and lower gain suppresses force oscillation, but target tracking remains below declared acceptance. No new Utonia training; reliable carrying predictor remains unproven. Next: preserve observed surface geometry through the model adapter and address partial force observability. Historical stop entries below were superseded by explicit continuation.

2026-09-16 rendered-video addendum: [play actual carry + all-six support comparison](../../../experiments/object_predictor_v1/rendered_carry_v1/index.html), [full rendering report](../../../experiments/object_predictor_v1/rendered_carry_v1/RENDER_REPORT.md). Frozen current model on saved native carrying trace fails transfer: mean center11.185cm/rotation65.221deg/size21.330%/mass66.502%. Complete meshes/actual body poses and truthful estimates rendered, all14 verification checks pass; zero training/new physics. Rendering request is complete; stop here.

# Object state from hand geometry and touch

Latest result (2026-09-15): the first useful matched mixed-surface result is complete. Support mass error is4.49%, additional12-mass error7.27%; force-zero degrades to37.57%/36.41%. See the [full Chinese report and figures](../../../experiments/object_predictor_v1/FIRST_USEFUL_RESULT_REPORT.md). The user requested stopping at this result; no further experiments are queued. Broader shape/interaction and fresh native validation remain unfinished. Earlier negative evidence below is retained.

Active user request, 2026-09-15: train a pretrained object predictor from different
combinations of hand positions, contact locations and forces; estimate pose, size
and mass, and try SUGAR/IsaacLab and Newton. This is independent supervised
perception work, not a restart of archived Plan-15 policy training.

## Model choice

Survey checked official sources on 2026-09-15:

| Candidate | Fit and limitation |
| --- | --- |
| [Utonia, ICML 2026](https://github.com/Pointcept/Utonia) | Cross-domain pretrained Point Transformer V3; released full [checkpoint](https://huggingface.co/Pointcept/Utonia). Supports geometry without RGB/normals. Selected for the force/contact point-cloud experiment; transfer to tactile histories is a hypothesis. |
| [Concerto, NeurIPS 2025](https://github.com/Pointcept/Concerto), [Sonata, CVPR 2025](https://github.com/facebookresearch/sonata) | Earlier related point-cloud pretraining; useful references, no claim of a measured local ranking. |
| [AnyTouch 2, ICLR 2026](https://github.com/GeWu-Lab/AnyTouch2), [Sparsh](https://github.com/facebookresearch/sparsh) | Optical tactile encoders. Newton has no matching optical images; AnyTouch 2 README additionally specifies checkpoint access by form. |
| [NeuralFeels](https://github.com/facebookresearch/neuralfeels), [TouchSDF](https://touchsdf.github.io/) | Strong geometric reconstruction references; sensor/image compatibility and joint moving-object pose/shape estimation require separate adaptation. Not pretrained mass predictors. |
| [TacGraph, RA-L 2026](https://tacgraph.github.io/) | Explicit kinematic/force constraints, factor graph rather than a drop-in pretrained regressor. |

Use complete official Utonia code/weights, no replacement Transformer. Local input
affine adaptation and supervised output heads are task glue. Preserve metric scale;
never normalize each cloud to unit size when size is a target. Zero RGB is missing
color, not a force image. Non-Flash attention is an officially supported runtime
path and does not remove layers/parameters. Code is Apache-2.0; released weights
are CC-BY-NC-4.0. Source checkout and binaries live under ignored
`experiments/object_predictor_v1/`.

## Data and evaluation contract

- Newton collection uses the existing SUGAR G1 URDF, scanned CarryBox mesh,
  hydroelastic hand surface and SolverMuJoCo. Reference joint drive is only the
  collector behavior; object state is integrated by physics, never replayed as a
  label. This fixed-base reference-driven collection is not a free-standing policy
  or a proof of successful demo following.
- Vary scale, mass, initial object position/orientation and motion independently.
  Save failed/no-contact sequences as well as successful contacts. First CarryBox
  experiments are within-family estimation; additional geometry/interaction
  coverage and IsaacLab validation remain part of the overall objective.
- Keep causal hand/sensor geometry and local force observations separate from
  labels. No object velocity, true friction coefficient/utilization, object ID,
  reference object pose, future frames, or simulated true slip in model inputs.
- Contact centroids/pressure here are idealized simulated tactile observations,
  not proof of a calibrated hardware sensor. Tangential traction is distributed
  from resolved patch force, not independently measured per triangle.
- Compare independently trained geometry, geometry+contact, and
  geometry+contact+force histories with equal data, initialization and budgets.
  Include test-time force shuffling, train-only constant-target baselines and a
  direct geometry baseline. Group splits by entire episodes and held-out physical
  configurations/motions, never random adjacent frames.
- Targets: object geometric center relative to current hand frame, rotation,
  metric dimensions, mass. Report position cm, rotation degrees (including shape
  symmetries where relevant), dimension relative error, mass relative error and
  stratification by observed contact/support. Report inaccessible mass/no-contact
  ambiguity instead of hiding it by filtering the test set.
- Full pretrained checkpoint load, finite full-model backward/update, saved model
  reload and predictions are required. Loss decrease alone is not a successful
  predictor. Inspect physical contact and prediction curves before concluding.

## Current execution

Full released Utonia loaded: 137,253,744 original parameters, 137,542,639 including
sensor affine and readout. Actual H200 qualification R1 passes all15 checks: exact
released tensors, full backward through every stage, sensor/head gradients, repeated
eval3.49e-9. Official GridPooling children retain shuffle_orders=True even when the
root config is False; the adapter explicitly sets that existing runtime option in
all modules. The initial failed qualification remains, without architecture changes.

The first three full-model arms each completed 600 updates with full Adam states.
Batch-one results on 234 Newton test windows (six entirely held-out episodes):

| Input | Center cm | Raw rotation deg | Size relative | Mass relative |
| --- | ---: | ---: | ---: | ---: |
| Hand geometry | 23.05 | 105.18 | 40.18% | 8.56% |
| + contact positions/flags | 24.57 | 104.38 | 53.36% | 40.27% |
| + forces and contact area | 20.08 | 107.55 | 64.10% | 10.83% |
| TRAIN-only constant target | 26.58 | 106.03 | 6.63% | 20.81% |

These are negative precursor results, not policy-ready perception. In particular,
low mass error in a geometry-only arm is not evidence of physical mass inference.
Setting force to zero while preserving contact locations, flags and area improves
the original force arm's center error to 12.68 cm; it has not learned beneficial
force use. A separate non-neural contact-centroid diagnostic, with offsets fitted
only on TRAIN and conditioned on sensor contact, obtains 4.98 cm on contact windows
(equal-episode 7.09 cm). It is a baseline, never a replacement pretrained model.

All first-round endpoints and six backend evaluations are saved under
`experiments/object_predictor_v1/{training,evaluation_batch1}`. Contact and force
training prediction replay exceeded the unchanged 1e-5 tolerance. Full saved model
and Adam states were nevertheless exact; a local CSR mean/max readout gave repeated
output difference zero. Original scatter failures remain failures. Comparison-only
legacy protocol field handling was fixed without repeating training or inference.

The second matched round (`training_headlr1e5`) also completed all three 600-update
arms, with readout LR 1e-5 and deterministic CSR pooling. Every complete prediction
replay is exact. Batch-one Newton errors are 23.19/15.96/17.64 cm center,
33.72/28.27/31.72% size, and 20.62/24.58/25.85% mass. Force-zero center is 14.88 cm.
Both actual streaming audits pass within 1.19e-7 in physical outputs. Full known-mesh
surface distances are 13.27/9.22/10.12 cm on Newton and 12.96/8.27/12.16 cm on the
historical native trace. Contact geometry helps, but this remains an inaccurate
precursor with no overall force benefit.

A frozen full-model audit on the same eight TRAIN batches (eight stochastic repeats)
found size loss 0.07128 in raw stochastic TRAIN versus 0.55913 in eval, ratio 7.844.
Removing only existing stochastic regularizers from TRAIN reproduces eval exactly;
all tensors are unchanged. This meets the prospective runtime-correction criterion
in `STOCHASTIC_DEPTH_ADAPTATION_PROTOCOL.json`. The third matched full-model round
(`training_no_stochastic`) completed all three 600-update arms: same data, release initialization, seed,
CSR and 600 updates per arm, only disabling existing DropPath/Dropout during task
adaptation. All 15 actual full-model qualification checks passed first. No architecture
or learned parameter was removed. Batch-one Newton center errors are 24.32/15.39/15.50 cm, size errors
8.16/10.01/9.88%, and mass errors 27.33/28.71/28.11%. The size mismatch was reduced;
overall force benefit remains absent. Streaming and physical mesh metrics completed.
Frozen actual training mode now equals inference exactly; size loss is 0.08693 on
the same diagnostic TRAIN batches, versus the prior inference loss 0.55913.

`compare_adaptations.py` confirmed every training batch, budget and held-out label.
The original strict bitwise-initial comparison remains 61/63: contact/force initial
predictions differ by at most 7.33e-6, with all labels exact. A separately recorded
comparison passes 63/63 under the existing 1e-5 prediction-reload tolerance. It does
not reclassify the original bitwise failure or establish exact initial predictions.

The 36-episode corpus has 7,200 frames / 2,463 contact frames, with 24/6/6 whole
configurations in TRAIN/VAL/TEST. Every failure and no-contact interval remains.
Qualification with 0.35/0.7 kg has 194/195 contact frames out of 200, net support /
weight 1.000118/.999723 and summed normal / weight 1.006895/1.006302. Earlier
unstable fixture traces are kept outside the training corpus. Changing scale and
pose still produces failures; the formal corpus is not uniformly successful support.

Four fresh native IsaacLab startup attempts failed Vulkan. The last initialized all
54 force fields before Kit's render queue failed, even under PXR / isolated portable
root. An existing 80-frame native CarryBox trace provides a separate saved-data test.
Its historical (shearX,shearY,normalZ) was the TOTAL local force, not today's separated
normal/friction channels. Explicit legacy conversion reconstructs and separates the
saved vector; recovered normal agrees with 20 * penetration within 4.76e-7 N. SDF
normals used inside that conversion are excluded from network inputs. The first
incorrect signed-channel conversion is rejected and retained. This historical
penalty sensor is not calibrated to Newton resolved forces. Original native center
errors are 24.19/29.43/43.42 cm; mass errors 62.50/198.15/151.91%. No new native
rollout, native fine-tuning or cross-backend benefit has been demonstrated.

`shape_metric.py` measures bidirectional nearest-vertex distances on every vertex
of the full known scanned outer mesh transformed by predicted pose and dimensions.
This complements raw canonical angles; it neither replaces the raw failure metrics
nor establishes arbitrary shape completion. `predict.ObjectStateEstimator` exposes
an eight-frame causal sensor-only interface with warmup/reset/timestamp checks;
its runtime audit must pass before relying on streaming equivalence. Model-only
latency of the first endpoints was about 70–76 ms on H200, excluding sensor capture.

## Controlled mass calibration

The mixed corpus contains many released objects and only one held-out episode with
33 windows meeting the observation-only conservative support condition (both hands
in contact over eight frames, load CV < 0.1, hand translational acceleration < 0.1 g).
In those windows, raw normal sum / gravity gives 2.76% mass error; the third full
force model still gives 13.96%. This is a fixture-specific diagnostic, not an excuse
to remove failures from the primary results.

A new independently declared calibration corpus keeps the complete object geometry,
initial pose and hand motion fixed while drawing 36 masses in [0.25, 1.2] kg with
seed 310119. Geometry seed 310017, whole-episode split 24/6/6. All 7,200 physical
frames are saved, 7,016 have contact, and all 36 rest support checks pass; net support /
weight ranges 0.99976–1.00019. Actual object motion and the complete coverage/mass
plot were inspected. Object poses remain physically integrated, never replayed labels.

`training_mass_calibration` completed all three full-model 600-update arms,
full Adam, batch-one evaluation and streaming. Held-out mass errors are 32.65%,
32.19%, and 12.03% for geometry/contact/force; force-zero preserving contact/area
worsens to 50.01%. The model learned force information, but fails the declared
10% accuracy target (3/4 criteria). Direct normal sum/g on observation-qualified
support windows reaches 1.50%, versus learned force 11.56%. Mixed-geometry transfer
is negative: learned force mass error40.62%, center21.56cm.

A separate VAL-only output calibration fits one log-mass bias per arm (force scale
1.151142), with no neural updates or slope/search. Fresh physical evaluation completed:
12 unseen masses, seed310120, IDs2000..2011, all TEST, unchanged geometryseed310017.
All2400frames retained,2340contact, all12physical support checks pass;468 causal windows.

| Fresh mass error | Geometry | + contact | + force/area | Force zero, area retained |
| --- | ---: | ---: | ---: | ---: |
| Raw full model | 49.91% | 49.39% | 12.66% | 42.81% |
| VAL-only mass scale | 68.51% | 68.45% | 11.56% | 37.38% |

Both fail the declared10% criterion (3/4checks), despite reproducing force dependence.
The0.2515kg episode has persistent49.53%calibrated error; TRAINmin0.2749kg and
VALmin0.5644kg expose a coverage limitation. No test filtering/refitting: all12mass
curves and the failure episode were inspected. On409observation-qualified windows,
normal sum/g gives1.546% versus calibratednetwork9.717%; primaryall-window11.56%
remains negative. Independent saved-array readback44/44; actual200framestreaming
matches39savedwindows within1.124e-7, including reset and duplicate-timestamp checks.

`experiments/object_predictor_v1/MASS_ESTIMATOR_RUNTIME.json` records the unchanged
full endpoint and optional `mass_log_bias` for `ObjectStateEstimator`; it explicitly
marks failed accuracy and limited scope. `FRESH_MASS_CALIBRATION_READBACK.json`,
`FRESH_MASS_RANGE_AUDIT.json`, and `FRESH_MASS_TIME_CURVES.png` retain the evidence.
This establishes controlled force-dependent mass learning, not general pose, shape,
material or policy-ready perception. Fresh native IsaacLab remains renderer-blocked.

A separate voxel audit found >1% normal-load loss in only three of 2,463 mixed
contact frames (one TRAIN, two VAL, none TEST). Averaging extensive force/area in
rare merged voxels is a recorded limitation, not the main explanation of current
mass errors. Original source data and all negative experiments remain intact.

## Multi-face acquisition and long history

A full-mesh free-object probing fixture now explores opposing X sides, opposing Y
sides, and the top. Fixed nominal waypoints plus the previous measured two-hand
loads drive the hands; object pose/size/mass never enter the controller. The object
rests on a real ground plane, so this is geometry acquisition, not mass observability.

First1500control qualification failed (no X contact, asymmetric top contact,
260.98N peak), preserved in `multiface_probe_qualification_r1`. Gentle5400control
nominal qualification passed:375/318/54simultaneous-contact frames, peaks3.64/3.49/
9.47N. All2976recorded contact points were checked against the complete15626vertex
mesh; maximum nearest-vertex distance4.52mm, with distinct X/Y/Z surface normals.
Full hand-mesh ground clearance remains >=48mm. All three physical geometry panels
were inspected; these are actual recorded geometry, not a learned reconstruction.

Two anisotropic qualifications completed: A failed (top simultaneous contact1frame,
peak44.72N); B passed. Saved controller replay matches every5400issued hand poses
within2.98e-8m. A reaches its outward stroke limit while the object extends beyond
that nominal boundary; the top forces alternate every sample. Those failures remain.
A single declared corrective variant widens standoffs, allows additional retreat,
and lowers force-servo gain from.002 to.00005m/s/N. Corrective A completed7500controls
and passed:480/982/832simultaneous-contact frames, peaks2.15/3.23/3.14N; no contact
toggling in the fixed final4s approach windows. Nominal and B also completed: all three corrective cases pass,22500actual controls,
all nine physical/geometry phase panels and controller replays inspected. B peaks
4.58/2.99/13.42N; nominal3.38/6.00/3.15N. All fixed final4s windows have zero contact
sample toggling. This admits geometry acquisition, not a learned prediction result.

`multiface_dataset_v1` is now collecting36new physical configurations,24TRAIN/6VAL/
6TEST, whole-configuration split fixed before collection. Seed310330; independent
stratification of mass[.25,1.2]kg and XYZ scale[.85,1.15], TRAIN includes range endpoints;
geometry seeds310430..310465, IDs3000..3035. Each7500controls, total270000maximum.
All physical failures stay in their original split, with full traces/plots/controller
replay. Primary geometry window140..148s was fixed before collection; all-window and
all-configuration results remain required. No neural training is launched by this
collector. Ground-supported probing cannot establish mass inference; the full goal
still requires useful pose/size estimation and verified mass sensing.

The original8frame window spans only.14s and loses previous probe directions.
Optional `--history 32 --history-policy episode_uniform_recent --time-scale-s 108`
samples30clock-spaced past frames plus the latest two, with no force/phase/label
selection. In the qualified trace it retains102s and all three contact directions.
CPU data/runtime selection291/291 and actual full-model forward/backward15/15 pass.
All137253744pretrained parameters stay; only the existing linear temporal readout
widens, total138407503parameters. Frozen repeats are exact; no optimizer updates.
The original8frame endpoint streaming replay remains within1.124e-7. New history
flags propagate through training, checkpoint restoration, evaluation and readback.
This qualifies an input path, not a new trained32frame estimator or accuracy claim.

## Collision compatibility findings

The Newton legacy palmar helper uses left +Y/right -Y; current SUGAR native
`anatomical_whole_hand_tacsl_g1._surface_frame()` uses left -Y/right +Y. The new
support collector explicitly uses the latter. This discrepancy is not silently
applied to old saved traces.

The scanned box has two closed components (positive outer volume0.035058m³,
negative inner volume-0.031168m³). SUGAR's `SMALLBOX_SDF_CFG` explicitly sets
`solid_outer_shell_only=True`; `geometry.load_sugar_outer_box()` follows the same
unique-positive-component rule. Legacy Newton raw-mesh collection included both.
The adapter retains every outer vertex/face, with no hull collider or remeshing.
Correcting that compatibility issue alone has not produced stable support. CAD
hull facets are used only to choose hand orientation for the acquisition fixture;
all physical hand contacts still use the complete original hand meshes.

## Entry points

Use the retained compute step for all GPU work. Full model dependencies are the
existing `sugar_py311_isaacsim510` environment plus the isolated Utonia checkout and
packages in `experiments/object_predictor_v1/{vendor/Utonia,deps}`. Do not run GPU
training or inference on the login node.

- `collect_probe_dataset.py`, `probe_scene.py`: full-mesh multi-face acquisition;
  source hashes and whole-configuration split are fixed before physical collection.
- `audit_probe.py`, `audit_probe_controller.py`: contact geometry and exact issued
  controller-pose readback from saved measured loads; no physics repetition.
- `audit_history.py`: causal historical coverage and data/runtime selection on real traces.
- `collect_support.py`: real Newton support-fixture acquisition with independent
  object parameters and whole-episode splits. Exact meshes and free object physics.
- `data.py`: shared causal sensor-only encoding and separate object-state labels.
- `train.py`: full-model supervised adaptation, full Adam checkpoints and reload.
- `evaluate_endpoint.py`: saved full endpoint, deployment batch one, optional
  existing second-backend dataset and force-zero control that retains contact area.
- `compare.py`, `shape_metric.py`: saved-array physical-unit comparisons and full
  known-mesh surface metrics, with plots.
- `observation_baselines.py`: explicitly non-neural diagnostic baselines fitted on
  TRAIN only; no replacement model or new physical labels.
- `audit_streaming.py`: execute the real streaming API over every sensor frame of
  the first held-out episode and compare all matching saved evaluation windows.
- `calibrate_mass.py`, `audit_mass_calibration.py`: VAL-only mass offset, separate
  raw/calibrated outputs and independent fresh-TEST saved-array readback.
- `audit_training_mode.py`: fixed TRAIN batches, eight stochastic repetitions,
  deterministic runtime controls and full-state identity, with zero optimizer steps.

The Python interface is `predict.ObjectStateEstimator(endpoint, base_checkpoint, mass_log_bias=0.)`.
Call `reset()` between episodes and `update(frame)` with exactly the documented
sensor fields in `data.OBSERVATION_KEYS`. The first seven updates return `None`.
Outputs include hand-relative/world center, world rotation, metric dimensions,
mass and observed-contact flag. That flag is an observation, not calibrated
confidence. Current checkpoints remain experimental and inaccurate; loading a
checkpoint successfully does not establish readiness for a controller.

## Mixed geometry and weight protocol (2026-09-15 17:32)

The 36-case multi-face corpus is collecting in retained job 297320. The first
three cases include two physical failures; all remain in the original split.
The fixed 32-frame history covers both hands in all three directions in 8/8
primary windows for case 3000, but 0/8 for cases 3001 and 3002. This is a
measured information limitation, not a learned-model result.

`mixed_geometry_mass_v1/PROTOCOL.json` prospectively combines all 36 probes
with the existing 36 support fixtures. Whole-configuration splits are preserved;
each batch has two samples from each acquisition, with 32 causal frames and a
fixed 150-second age scale. Three full pretrained 138,407,503-parameter arms
receive 2,000 updates each, with the existing loss and optimizer settings.
The primary geometric window is 140–148 seconds; all windows and physical
failures remain reported. Primary mass results use the support fixtures.
Ground-supported pressing does not directly identify weight.

Preparation checks: actual-data sampler/input CPU 35/35, independent saved
physical metric CPU 21/21. The mixed full-model GPU qualification and training
have **not run**. The prepared serial script is
`experiments/object_predictor_v1/train_mixed_geometry_mass_prepared.sh`; it
requires the complete corpus and agent inspection records before qualification
and training. No new learned 32-frame endpoint exists yet.

## Normal-channel meaning and candidate CAD query (2026-09-15 17:45)

The stored Newton `hand_normals_w` channel is the local palmar ±Y axis rotated
by the hand pose. It is **not** a local curved-surface normal. Saved readback
on 36 support and four probe episodes confirms this exactly (41/41 checks).
This corrects a description; existing observations and learned endpoints are unchanged.

A separate candidate in `hand_surface_normals.py` queries the complete original
hand triangles at anatomical sites and observed contact centroids, interpolating
vertex normals on the closest triangle. It uses only known hand CAD and sensor
observations, never object shape/pose labels. Five actual TRAIN traces pass 15
basic checks and 15 contact-window/rigid-transform checks; full-history versus
cached normals match exactly, and transformed normals differ by at most 5.62e-8.
These are nearest hand-surface normals, not measured force directions or object
normals. The candidate is not yet wired into the predictor or the declared
mixed training protocol, and no neural improvement is claimed.

Fresh IsaacLab remains unqualified after four preserved Vulkan failures. Local
source inspection found that its no-render mode already bypasses ordinary
render calls, while core startup still updates Kit. See
`ISAAC_HEADLESS_SOURCE_AUDIT.json` and the official
[Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
No fifth native launch or change to the active Newton collection was made.

## CAD-normal integration and revised preparation (2026-09-15 17:54)

`normal_policy=hand_surface` now connects known-hand CAD normals to dataset
caching, online encoding, checkpoint restoration and evaluation. The default
`stored` path keeps existing endpoints' inputs. A 242-check CPU integration
passed on five saved TRAIN traces: cached and online features match exactly,
only normal columns change, and geometry-only features ignore tactile values.
This is input qualification; full-model GPU checks have not run.

`mixed_geometry_mass_surface_v1` supersedes the unexecuted mixed preparation.
All 27 previous protocol fields, including the 6,000-update total budget, remain
identical; normal policy and hand/query source signatures are added. Use
`train_mixed_geometry_mass_surface_prepared.sh` after the cohort and inspection
complete. The former launcher explicitly refuses to start. No new trained
32-frame endpoint exists yet.

All three predictors share force-feedback acquisition trajectories. Thus the
geometry-only predictor can see contact indirectly through the measured hand
motion. This is an ablation of explicit predictor sensor channels, not a test
of acquisition without touch. See `ATTRIBUTION_LIMITS.json` in the new preparation.

## Full-mesh evaluation preparation (2026-09-15 18:10)

The prepared surface-input pipeline now includes the existing full 15,626-vertex
known-asset geometry metric and `report_shape_by_acquisition.py`: all-window
and fixed primary-window grouping, plus one predetermined 3D overlay per TEST
episode. Probe figures use the first sampled point in 140–148 seconds; support
figures use the first sampled frame. No example is selected by prediction error.
This transforms a known complete mesh by predicted pose/size; it does not
reconstruct unseen shapes and does not replace the raw state criteria.

Geometry reporting requires the existing Newton Python3.12 environment because
the official full-mesh loader imports Newton. The initial Python3.11 import
failure is preserved. Qualification on prior real old8-frame outputs passes
24 alignment and 13 independent exact metric/cloud checks. No new model was
run for this qualification.

## Saved contact-loss diagnosis (2026-09-15 19:01)

`audit_probe_contact_loss.py` describes terminal contact loss in the first 17
saved probe episodes, without running the controller, physics or a model.
All six physically failing cases have one or more terminal loss intervals of
10.38–17.76 seconds while the first-touch flag remains latched. During these
intervals the hands move inward at about 0.05 mm/s, versus the source rule's
6 mm/s before first touch; observed inward travel is only 0.518–0.887 mm.
Saved pose displacement agrees with the existing feedback rule within 1.33e-8 m.
The analysis pins input and controller source hashes in
`CONTACT_LOSS_FIRST_SEVENTEEN.json`; CPU child 2821449 exited 0 at 11:01:00 UTC.

This establishes slow continued approach after lost contact in these traces.
It does not establish that a faster recovery would restore contact or improve
prediction. Current physics, fixed corpus, splits, model and training protocol
remain unchanged. Finish the declared cohort and evaluation before drawing a
predictor conclusion; retain these acquisition failures in the results.

## Scope of zero tactile readings (2026-09-15 19:25)

The physical collider is the full hand, while the saved 54 input pads cover
fixed palmar regions. `ContactField` leaves uncovered faces at pad -1, and
both observation and probe feedback exclude them. Thus zero recorded pad
load does not prove zero full-hand contact. `observation()` computes total
resolved hand load and unassigned-face counts, but this probe collector
checks only overflow and does not retain those diagnostics per frame.
See `SENSOR_SCOPE_SOURCE_AUDIT.json` for the hashed source evidence.

The earlier contact-loss diagnosis refers to loss of the recorded pad signal.
Its measured hand-speed finding remains valid; physical separation was not
established. Case3020 has a rise in body-origin height with weak pad readings;
object rotation and actual ground clearance still need saved-geometry analysis.
No new force truth, controller change or repeated physics was introduced.

Saved full-mesh follow-up (`ground_clearance_3020`, CPU2856150 exit0 at
11:29:11 UTC) resolves the height question: all 7,500 saved poses and all
15,626 object vertices were transformed. In phase0 the lowest vertex stays
within -0.769 to -0.101 mm of the ground, while body-origin height increases
18.72 mm and rotation from the 1-second pose reaches 7.85 degrees. Twenty
independent full-XYZ transformation checks match the height calculation exactly.
Both the full chronology and fixed44-second 3D view were inspected and hashed.
This supports grounded tilt, without evidence of suspension. Missing total
hand forces and ground reactions remain unavailable; no physics was repeated.

## Complete TRAIN history coverage (2026-09-15 19:46)

`HISTORY_COVERAGE_TRAIN24.json` now covers all 24 TRAIN probes and 192 fixed
primary windows. Both hands have readings in all three directions in 112/192
windows; at least one hand per direction is represented in 166/192. Among the
14 acquisition-qualified episodes, these counts are 111/112 and 112/112; among
the ten failed episodes, 1/80 and 54/80. Each-hand coverage can come from different
timestamps and does not imply simultaneous bilateral contact. The first12
records exactly match the prior audit. CPU2875861 exited 0 at 11:44:53 UTC.
All failed cases and fixed windows remain included; these are sensor-history
coverage measurements, not predictor performance. Full36 coverage remains pending.

### External interruption during the 36-case corpus

Slurm allocation 297320 was externally cancelled at 2026-09-15 20:01:04
Asia/Shanghai; step 0 ended with signal 9 at 20:01:34. The first 26 complete
cases (3000–3025), their saved traces and controller audits are hash-preserved.
Case 3026 stopped after the last logged zero-indexed frame 7050; no partial
trace or simulation checkpoint was saved, so its exact executed count is
unknown (7051–7500 controls). Its original directory and launch log are archived
under `multiface_dataset_v1/interruptions/job297320_episode3026/` with an
`INCIDENT.json` and snapshots of the original partial reports.

One replacement allocation, 297460, was granted after the old queue was empty.
The guarded recovery launcher checks every completed trace/record/audit and
all five pinned physics sources before continuing the original collector.
It skips the 26 completed cases and executes the unchanged case 3026 plus
3027–3035, with a remaining budget of 75000 controls. The interrupted attempt
has a separately declared allowance of at most 7500 controls: the final cohort
still has 270000 saved controls, while total execution may reach 277500.
This is recovery from an external kill, not an independent replicate, a change
of validation/test configuration, or evidence of successful training. The
original data/model protocol, failed-case denominators and success criteria
are unchanged. No automatic repeated recovery is authorized by this launcher.

Recovery case 3026 completed all 7500 frames and passed the original recorded-pad
coverage/peak criteria (534/465/1041 bilateral frames; peak 2.7684 N); both
physical plots and its saved controller replay were inspected. Comparison of all
142 original logged snapshots through frame 7050 is saved in
`RECOVERY_LOGGED_PREFIX_COMPARISON.json`: only the initial sample is bit exact,
phase clocks agree throughout, maximum position-component difference is
0.05737 mm and load difference 0.03886 N. This quantifies process-to-process
physical variation at the available sampled times; it does not establish
all-frame trajectory equivalence. The externally interrupted attempt remains
outside the completed cohort as separately accounted execution.

### Completed multi-face corpus and mixed training launch

The 36-configuration corpus finished on 2026-09-15 at 13:08:41 UTC with
exit code 0: 270000 saved Newton controls, original splits 24 TRAIN/6 VAL/6 TEST.
Recorded-pad coverage/peak qualifications pass 14/24 TRAIN, 4/6 VAL and 4/6 TEST;
all 14 failures stay in their original partitions. All 72 physical/3D plots have
hash-bound agent inspections in `multiface_dataset_v1/AGENT_INSPECTION.json`.
All five physics source hashes, all 36 controller readbacks, and the 26 complete
traces predating the external interruption were verified unchanged. The
interrupted 3026 attempt remains separately accounted; total executed controls
are between 277051 and 277500, not 270000.

The prepared `train_mixed_geometry_mass_surface_prepared.sh` pipeline launched
once at 13:10:08 UTC on retained allocation 297460/server05/step0, PGID 2991445.
Its data/history checks and full-model GPU qualification precede all optimizer
updates. It then executes the previously declared three 2000-update arms and
all saved-endpoint/evaluation/streaming readbacks. This launch is not a completed
training result or evidence that the predictor meets the state-error criteria.

The first mixed launch exited before GPU qualification because the legacy support
report has no `complete` key. This incident is preserved in
`mixed_surface_support_schema_incident`. Source-specific endpoint checks now
require the full 36-record cohort, per-support frame/balance checks and exact
metadata-to-endpoint agreement. The mixed 72-record view and all 223 sampling
checks then passed; the old H8 GPU streaming replay again passed at maximum
output difference 1.123823e-7.

The next full-model qualification stopped before optimizer updates because its
blanket gradient check also required a gradient for the official 54-element
`backbone.embedding.mask_token`. A separate full three-mode diagnostic found
this was the only missing gradient and found no nonfinite gradients. Official
`Embedding.forward` only uses this parameter when a `mask` input is provided;
the supervised predictor uses the unmasked path. The failed report and exact
named diagnosis are preserved in `mixed_surface_gradient_incident`.

The corrected qualification explicitly requires that sole unused parameter,
its original value, and finite gradients for every used parameter. All
138407503 model parameters remain bound to AdamW. Training now records the
complete name/optimizer-ID mapping; endpoint checks expect moments for all
138407449 used elements, no fabricated moment state for the unused token, and
an unchanged token hash. No architecture, loss, optimizer behavior or training
budget changed. The resume pipeline reuses hashed completed data/streaming
stages and repeats the corrected GPU qualification before three 2000-update
arms. Full history coverage is 175/288 windows for each hand across all three
directions and 254/288 for any hand; each-hand readings can occur at different
times. All gaps and failures remain in the data.

The corrected full GPU qualification passed all 42 checks. The active token-r2
pipeline (PGID 3003510, retained job 297460) launched at 13:18:14 UTC and began
the geometry arm; actual optimizer logs reached update 70 at 13:21:52 UTC.
The active continuation script is
`train_mixed_geometry_mass_surface_resume_token.sh`; it reuses hash-verified
completed stages and preserves both earlier training-before-update incidents.
No learned H32 endpoint or predictor-accuracy conclusion exists yet.

At update 500 of the geometry arm, a read-only CPU checkpoint audit passed all
11 checks for full named bindings, active Adam moments/clocks and unchanged
unused token. Its first launcher used login-local `/tmp` and exited before
reading anything; the preserved path incident was fixed by moving the identical
script into the shared experiment directory. No training was repeated. The
saved progress report stratifies VAL500: probe primary position 2.1257 cm,
rotation 11.5988 degrees and size 8.0531%; support mass error 26.8654%.
These are intermediate validation measurements, not TEST acceptance or a
checkpoint-selection decision. Actual geometry training reached update 1000
on the same live pipeline; all final-arm evaluations remain pending.

The geometry arm completed its 2000 updates, TEST evaluation and exact endpoint
reload (maximum output difference zero); its actual sampling audit passed all
225 checks. Mixed TEST equal-episode errors are 2.6681 cm position, 5.1955 degrees
rotation, 5.4761% size and 36.8563% mass. These are baseline mixed-domain metrics;
acquisition-specific final reporting and the contact/force comparisons remain
pending. The same pipeline has entered the geometry+contact arm, whose full
initial-model hash exactly matches the geometry arm. No intermediate checkpoint
was selected and no extra optimizer budget was introduced.
