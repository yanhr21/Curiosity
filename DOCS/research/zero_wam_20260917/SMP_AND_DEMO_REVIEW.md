# SMP、Zero-WAM 与 SUGAR 接口独立研究

2026-09-17。本轮只读论文、官方源码和已保存结果，没有模型前向、训练、物理、渲染或大文件下载。遵从用户最新暂停：旧 predictor 修复和自动实验不恢复。本文是接口与方法审查，不是新增实验成功报告。

结论：总体分工可以成立，但目前不能把三套检查点直接串接。Zero-WAM 的任务条件/末端动作、SUGAR 的全身参考与关节执行、SMP 的状态轨迹训练奖励是三个不同接口。最先要固定动作、坐标、时间和观测来源合同，然后用明确标注的 oracle 隔离接口，再研究触觉估计替换。SMP 能约束运动分布，不能替代示范目标、实际接触稳定性或成功判定。

## 1. SMP 在这里确指什么

SMP 是 **Score-Matching Motion Priors**。官方论文用预训练运动扩散模型提供策略训练奖励；官方实现是 [MimicKit](https://github.com/xbpeng/MimicKit)。本机 `MimicKit/` 的版本为 `2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69`。官方类名 `TinyMDMModel` 是作者模型，不是项目自造的小模型替代。论文包含 G1 真机运动技能，因此不能说 SMP 不支持 G1；但这不证明本项目的人类示范、触觉物体估计和 G1 搬运已经接通。[SMP 论文](https://arxiv.org/html/2512.03028v3)

实际原代码调用链是：

`已执行 rollout 的短时状态窗口 → prior.normalize → ESM_SDS_loss → 按噪声层归一化 → exp(-scale × mean_loss) → 与 task reward 加权 → PPO`。

它在 `torch.no_grad()` 下计算奖励，并不通过 prior 反传动作，也没有在每个控制步执行一个动作候选规划器。`SMPAgent._compute_rewards` 读取 rollout buffer 的 `disc_obs`，第174行明确混合 `task_r` 和 `smp_r`；第191–205行是 frozen score→reward。详见 [官方 SMPAgent](../../../MimicKit/mimickit/learning/smp_agent.py:162) 和 [作者 TinyMDM 的 ESM 实现](../../../MimicKit/mimickit/learning/tinymdm/tinymdm_model.py:233)。

因此需要区分：

- **训练用途**：奖励偏好类似先验训练运动的实际轨迹；任务奖励仍定义目标是否达成。训练好的执行策略不一定需要部署时运行 SMP。
- **在线候选评分**：可以另行研究，但需要可预测的真实身体/物体状态、评分校准和闭环验证；不是当前官方 SMPAgent 已提供的功能。
- **成功概率**：指数变换后的 reward 不是校准概率。项目 scorer 还会在 rollout 边界更新 `DiffNormalizer`，冻结 prior 权重并不等于跨时刻 reward 尺度固定。
- **直接正则化**：若对 Zero-WAM 或动作输出直接加可微 SMP 正则，这是新的适配方法。当前可忠实复用的是 RL reward 路径。

论文的 ObjectCarry 附录还专门处理运动风格与真实接触任务的冲突，采用 task-dependent style clipping；这支持把任务完成和接触约束单列，而不是假设“先验分越高任务越好”。其阈值不应未经验证直接搬到 SUGAR。[SMP §8.3、附录 D.3](https://arxiv.org/html/2512.03028v3)

## 2. 当前项目 SMP 的真实输入和特权边界

项目已有官方模型 glue：[official_smp_scorer.py](../../../SUGAR/source/sugar_rl/sugar_rl/utils/official_smp_scorer.py:62)、[sugar_smp_feature_window.py](../../../SUGAR/source/sugar_rl/sugar_rl/utils/sugar_smp_feature_window.py:62)、[固定 schema](../../../scripts/sugar/smp/sugar_g1_box_schema.py:118)。

|接口|实际约定|不能偷换的含义|
|---|---|---|
|历史|10帧，50Hz；重置时重复当前帧填充|不是32帧触觉窗口；填充帧不是10次独立观测|
|身体201维|原 `compute_disc_obs`，torso_link 为 anchor，29关节、选定身体/末端；`global_obs=False, root_height_obs=True, dof_vel_obs=False`|不是任意手点云或参考动作latent；关节顺序是Isaac导入序，不是SDK序|
|物体15维|相对 torso 的位置3、旋转切向/法向6、线速度3、角速度3，使用当前 torso heading 变换|没有触觉、质量、材质、任意形状输入|
|来源|`obj.root_pos_w/root_quat_w/root_lin_vel_w/root_ang_vel_w`|当前是模拟器真实物体状态，不能称触觉 belief|
|reward|官方 ESM 噪声层22/15/8，归一化后 `exp(-6 × mean)`；rollout后更新normalizer|不是分类softmax，也不是成功概率|

一般 scorer 构造器要求其指定的200k formal-prior协议；旧 selected/taskwide/conditional 50k 检查点使用各自明确的 adapter/protocol，不能把这些文件随意塞入该构造器、取消 admission 后称接口已通。[scorer admission](../../../SUGAR/source/sugar_rl/sugar_rl/utils/official_smp_scorer.py:80)

SMP 训练奖励使用模拟器特权状态是可明确声明的训练设定。更容易遗漏的是，**当前 SUGAR actor 自身也吃 object GT**：`TrackerCfg.obj_pos_b/obj_ori_b` → `observations.obj_pos_b/obj_ori_b` → `MotionCommand.obj_pos_w/obj_quat_w` → `obj.data.root_*`。所以只替换 Zero-WAM 上层 belief、底层 Tracker 保留原观察，并不构成“实际只靠估计”的实验。[actor 配置](../../../SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/inference/base_inference_env_cfg.py:225)、[坐标变换](../../../SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/observations.py:290)、[真实字段来源](../../../SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py:1712)

后续 oracle 与 estimated 两臂必须列清所有 actor 输入，GT 可以留在训练 reward/critic/评价通路；是否替换 actor 的对象位姿和速度、未知时怎样处理、估计延迟如何进入历史，都要明确。

## 3. Zero-WAM → SUGAR 有三层动作差异

本次官方源码为 `third_party/Zero-WAM-official`，commit `08e2c4ae41e2b63573a299825cebe6753481407c`。Zero-WAM 用人类视频作为任务条件，联合建模未来视频/动作；其 IFP 辅助目标鼓励模型利用示范。该论文与发布代码本身没有提供本项目 G1 动作桥接或 SMP 策略训练组合。[Zero-WAM 论文](https://arxiv.org/abs/2608.26103)、[官方仓库](https://github.com/robbyant-research/Zero-WAM)

|模块|实际输出/输入|接入缺口|
|---|---|---|
|官方 RoboTwin action|30维容器，16有效值：两手各xyz3+xyzw四元数4+gripper1；有效channel0..13、28..29|没有G1腿、腰、全身平衡与橡胶手接触模式|
|SUGAR Generator|36维参考：关节参考29+anchor线速度3+角速度3+contact1；生成后插值|不是16维双臂EEF，也不是29维已执行动作|
|SUGAR Tracker|接受 `generated_command` 与身体/物体观测，输出29维 JointPositionAction|没有现成的EEF→全身参考逆映射|
|Newton 当前搬运夹具|完整双手CAD由kinematic目标驱动，物体真实动力学|没有G1全身、关节限位、脚底支撑与平衡；不能代替全身执行资格|

官方相对动作语义尤其需要单独写合同：

`p_rel = p_target - p_initial`（world-axis 差）；

`q_rel = q_initial^{-1} q_target`（xyzw，固定四元数半球）；

`gripper = target_gripper`。

初态是整个训练 segment 的第一帧，不是每一步上一帧。位置没有左乘初态旋转，所以也不能把整组值直接当标准局部 `T_initial^{-1} T_target`。训练使用q01/q99归一化并clip到[-2,2]；评估按初始EEF恢复绝对目标，再以 `action_type='ee'` 执行。[动作预处理](../../../third_party/Zero-WAM-official/wan_va/dataset/robotwin_action.py:16)、[channel与统计](../../../third_party/Zero-WAM-official/wan_va/configs/va_robotwin_cfg.py:71)、[恢复与执行](../../../third_party/Zero-WAM-official/evaluation/robotwin/eval_policy_client_openpi.py:505)

另一个容易混淆的16是 `action_per_frame=16`，这是时间采样比，不能用它证明有效动作维数。原发布配置还有三相机布局和12fps ICL；需要独立定义与 SUGAR 控制时钟的对应。[发布配置](../../../third_party/Zero-WAM-official/wan_va/configs/va_robotwin_cfg.py:42)

SUGAR 的36维含义见 [wrapper._parse_action](../../../SUGAR/source/sugar_il/sugar_il/wrapper/sugar_il_wrapper.py:250)。虽然配置有 `future_frames=8`，当前 `MotionCommand.generated_command` 实际硬设 `future_frames=1`，每次从插值buffer取一行36维；不能仅凭配置把 Tracker 输入描述成8行。[实际读取](../../../SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py:1488)、[29维动作配置](../../../SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/inference/base_inference_env_cfg.py:200)

未来 bridge 至少要明确：world/torso/object坐标，腕原点或掌面TCP，左/右手及四元数序，rad/metre/速度单位，参考关节与实际归一化action的区别，重采样/延迟，固定橡胶手如何表达gripper开合，脚/腰/躯干与接触可行性。不能用30→29截断/补零、或者让一个未验证小MLP替代全身控制来回避这些问题。Newton夹具可以先检查双手坐标和物体目标，但应与 SUGAR/G1 真正全身闭环分开验收。[Newton kinematic 完整手](../../../scripts/sugar/object_predictor/support_scene.py:40)

## 4. 旧工作哪些真的能复用

这里仅读已经保存的结果，没有重跑；不同实验的passed定义不能互换。

|已有证据|可复用|尚未成立|
|---|---|---|
|实际 SUGAR corpus：100 CarryBox+99 KickBox，8 shards；160 TRAIN/20 VAL/19 TEST，139300实际transitions|动作落地、接触事件、对象运动、示范/时间对齐的基础语料与拆分|仅2任务，同embodiment；不是74.2K人机配对、基础模型预训练规模或官方G1 adapter|
|selected-demo 官方TinyMDM两prior各50k；单clip身份判别通过，semantic extension失败|官方模型加载/ESM、schema、负例与泛化检查|跨任务/跨同类demo语义先验成功|
|conditional-taskwide官方 CondTinyStableMotionDiTModel，50k，160 TRAIN motions；独立motion分类19/19|共享checkpoint/normalizer、完整官方条件prior路径|参考运动分类准确不等于真实恢复状态或任务奖励可靠|
|同一conditional prior的真实恢复状态审计：baseline仅0/20 profile偏好正确Kick，正确window偏好约0.290%|非常有价值的参考分布→执行分布负例|不能把motion类别score当实际任务进度/成功概率|
|correct-Kick vs wrong-Carry prior匹配在线训练，各64updates|确实调用官方score作为训练reward、条件会改变行为|两臂safe kick均16/20、fall均3/20；`correct_condition_has_physical_advantage=false`，没有此条件下SMP物理收益|
|较早生成参考真实rollout三臂均321/400即终止，handoff后163控制/臂；严格评价失败|真实命令路由、因果history、失败轨迹、完整Generator/Tracker接口|这是早期基线，不能当成后来实验的最终成绩|
|较新 `actual_state_expert01512`：demo original/repeat均318总控制、alternate完整400；zero original/repeat231、alternate355|保留实际局部改进：alternate由321到400；905条真实生成控制对应actor输出逐条重放准确|original反而少3控制；alternate末期box-reference偏差约23cm；单分支全程不等于整体成功，不是独立泛化或SMP收益|

来源：[实际 corpus 审计](../../../experiments/demo_following/zero_wam_official_v1/action_corpus_v1_audit/RESULT.json)、[旧 Zero-WAM admission](../../../experiments/demo_following/zero_wam_official_v1/training_admission_v4/ZERO_WAM_TRAINING_ADMISSION.json)、[selected cross-score](../../../experiments/demo_following/selected_demo_smp_v1/cross_score/RESULT.json)、[conditional prior](../../../experiments/demo_following/conditional_taskwide_smp_v1/prior/result.json)、[19-motion 分类](../../../experiments/demo_following/conditional_taskwide_smp_v1/motion_disjoint_score/RESULT.json)、[真实恢复 score](../../../experiments/demo_following/conditional_taskwide_smp_v1/recovery_score/RESULT.json)、[匹配物理实验](../../../experiments/demo_following/conditional_smp_recovery_prefix41_v1/PAIR_RESULT.json)、[实际生成参考失败回读](../../../experiments/demo_following/demo_future_smp_v1/measured32_generated_command_rollout158_r1/RESULT_READBACK.md)。

较新生成参考结果见 [actual-state expert 完整读回](../../../experiments/demo_following/demo_future_smp_v1/matched_generator_actual_state_expert01512/RESULT_READBACK.md)：共同158控制prefix之后，demo两种不同分支分别执行160和242个生成控制；repeat是确定性重放，不是独立seed。仍有已知reference-position辅助（原报告标注known48）、已知TRAIN96/90与phase158条件。表中保留早期321作基线，最新318/400并不被它覆盖。

旧 `zero_wam_official_v1` 文件夹名字不构成官方模型已跑过的证据。其旧 admission 明确 `official_zero_wam_release_ready=false`、`official_29dof_adapter_ready=false`、`training_allowed=false`；这反映当时发布阻塞。本次新下载解决的是官方资产可获得性，**不自动解决G1 adapter或重新宣告旧gate通过**。

Newton 的完整CAD、真实连续掌面流、坐标/力单位读回及失败数据也可以继续作为传感器/物理基础；旧已知box状态回归、全表面teacher输入AE并不证明任意形状触觉重建。最终视频和负例应保留，不再把控制、感知、先验、任务全混成一个“更像示范”的分数。

## 5. 对总体草案的必要修订

1. **在 A/B 之间加明确动作桥接合同，B拆两项。** B1是Newton双手oracle坐标/接触目标测试；B2才是SUGAR/G1全身执行。第一阶段不必等完整触觉形状模型，但也不能用双手夹具通过替代B2。
2. **actor观察来源必须与上层belief一起列出。** 现Tracker物体GT输入不能漏掉。理想状态实验可保留并标oracle；估计替换应覆盖所有实际actor对象状态通道，再比较GT/视觉/视觉+触觉及未知回退。
3. **SMP默认按官方no-grad训练reward描述。** 若写“正则”或“在线筛选”，标成另一个待验证算法；不把Frozen prior/条件分类准确度当校准成功率。先保留任务目标/接触/失败monitor，再做同初态同预算有/无prior闭环比较。
4. **公开资产兼容不等于G1动作兼容。** 优先完成官方RoboTwin原任务检查，再选一个有真实执行数据的G1任务做桥接；双臂gripper语义、全身36参考以及29动作要分别建表，固定原官方模型而非自制降级替代。
5. **对照不仅换demo，还应打破时钟/类别捷径。** 同初态同对象给同任务不同终点示范，以及正确/错误/空demo；记录真实目标关系、接触事件、对象轨迹与完整失败分母。现有两类先验更像短时motion分布约束，不能代替这种目标敏感性验证。

建议未来最小组合：Zero-WAM保留官方demo/video/action路径，提出有时间与坐标定义的短时目标；合格的全身执行接口将其变成实际29关节动作；实测身体/对象窗口只在训练时给官方SMP提供运动奖励；真实观测monitor独立判定目标、接触、掉落和unknown。新增3D belief条件、SMP奖励以及估计替换应分阶段验证。这些是后续协议建议，本轮没有恢复任何实验。
