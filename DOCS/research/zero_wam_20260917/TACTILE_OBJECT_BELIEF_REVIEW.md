# 触觉与手几何到物体信念：官方方法及本项目适配审查

2026-09-17。研究与源码阅读；没有新训练、推理、物理、渲染或队列。旧 predictor／AE 修复准备保持暂停。本报告区分官方已有能力、当前项目已有证据和未实现建议；权重链接可见不等于本轮下载或加载成功。

## 推荐主线

**先建立局部接触与持续位姿约束，再让完整官方形状先验对未观测区域提出条件补全假设；快速触觉反应独立保留。** 当前稀疏力读数不宜直接承担“唯一完整形状＋精确质量＋材质”的联合回归。3D 的作用是明确触点在哪里、哪些位姿/形状与观测相容、下一次触摸能排除什么，而不是把所有触觉先压成一个完整网格才能控制。

具体优先级是：以 **NeuralFeels 的官方持续 shape/pose 后端**为最接近移动物体的复用起点，先检验实测局部几何能否在运动历史中一致配准。其前端换成 SUGAR 观测接口属于项目适配，须单独验证；不能宣称原 checkpoint 直接支持我们的压力数字。完整 shape prior 作为后续条件假设层，采用一个通过原生资格的官方实现；不同时拼接 Active3D、TouchSDF 和 Michelangelo，也不让补全结果覆盖实测面。TouchSDF 是“已配准局部面→先验补全”的直接参照，已有 Michelangelo 保留为先验上界研究，二者当前均不是已验证的 SUGAR 触觉学生。

这是一条“观测约束优先、先验补全其次”的路线。若仅有27区域载荷/剪切而没有可靠接触几何，第一项缺件就是局部接触位置/面积/法向的传感标定或学习前端；不能把区域压力图渲染成假 DIGIT 图像绕过这个缺件。

## 1. 当前到底观测到了什么

|量|可直接使用的证据|不可自动升级成的结论|
|---|---|---|
|手位姿、关节、手 CAD|本体感觉与已知机器人几何；注明传感器到手的标定|物体位姿或真实接触法向|
|接触位置、面积、压力、剪切|仅在该传感器确实输出且已标定的支持域内成立|完整物体面、材质、任意方向净受力|
|已触及的局部面|当前时刻实测点、插值来源、拟合误差；曲面只局部拟合|未触及面的几何和全局拓扑|
|无接触|该次探测在阈值/覆盖/时间条件下未检测接触|探针朝向整条射线或手外大体积为空|
|物体位姿|多次几何约束、视觉或运动先验下的估计|稀疏对称接触能唯一给出6DoF|
|质量/摩擦/刚度|满足相应动力学和激励条件后的推断|由静态夹紧载荷或形状直接确定|

本地 `scripts/sugar/object_predictor/surface_data.py:18–71` 的输入是手坐标接触质心、面积、压力、剪切；位置经实际 hand pose 变换。它比只有27个区域力数的传感器信息更强。连续剪切场还带有模拟器构造假设：`sugar_newton/tactile/field.py` 按接触压力分配 patch 摩擦合力并作切平面投影，不能称每个三角面独立测得剪切。

同文件 `:99–159` 把32帧都放到当前左手坐标，积分后再编码：载荷 `log1p(N)`、剪切 `asinh(N/2)`、面积 `m²×1e4`，2mm体素中手身份可能混合。`observed_force_summary.py:64–101` 已能无参数反解8项力摘要及2项支撑投影。**这些量是编码观测的函数；复现它们不是学会物体信息。** 其中法向合力用拟合/CAD法向近似，不能当 solver 真 wrench。

原不可辨识反例仍有用：TRAIN5000–5003、frame31 的允许历史完全相同且观测无接触，质量却分别约0.454/0.915kg。这只能输出先验或 UNKNOWN；它不解释全部有接触阶段错误。详见[已有根因报告](../../../experiments/object_predictor_v1/research_review_20260916/ROOT_CAUSE_REVIEW.md)。

## 2. 官方方法、真实权重与传感器差异

|方法|官方输入→输出及监督|代码/权重可用性|对当前项目的判断|
|---|---|---|---|
|NeuralFeels|RGB-D、DIGIT 图像和运动学；局部深度进入在线 SDF 与位姿图优化。已知 CAD tracking 和未知形状 SLAM 是不同模式|[官方代码](https://github.com/facebookresearch/neuralfeels)，MIT；[作者权重](https://huggingface.co/suddhu/tactile_transformer/tree/main)实际列出 `dpt_real.p`、`dpt_sim.p`，各约311MB，MIT。它们是**触觉图像→深度前端**，不是完整物体 posterior 权重|最贴近移动物体的持续几何后端。SUGAR 接触点需独立 observation adapter；本轮未核完初始化/标定所需字段，不能声称即插即用|
|TouchSDF|视觉触觉的局部面预测＋DeepSDF 形状先验，多触摸局部点云拟合 latent；局部面与SDF分别监督|[官方代码](https://github.com/maurock/TouchSDF)、[项目](https://touchsdf.github.io/)。实际官方树有 `results/runs_sdf/24_03_190521/weights.pt` 6,326,307B，`runs_touch/30_05_1633/weights.pt` 1,290,134B；本轮只核元数据，未加载。未发现根 LICENSE；网站 CC-BY-SA 不等于模型/代码许可|适合研究“接触约束怎样条件化完整先验”；TacTip marker/depth 前端与压力/剪切不同。不是动态未知位姿已解决的现成系统；许可、坐标初始化及本域重建仍需资格|
|Active3D|局部 touch chart 的 xyz/mask，多次探测→全局多patch网格；训练以完整形状点云重建和主动触摸为目标|[官方代码](https://github.com/facebookresearch/Active-3D-Vision-and-Touch)，MIT；[官方权重包](https://dl.fbaipublicfiles.com/ABC/ActivePretrained.tar)由原 `download_models.sh` 指定；本项目已有完整官方模型/权重|输入不是压力或质量。已有失败结果和完整Engine等价核验保留；不能据当前失败判定所有触觉重建无效，也不应继续默认其网格就是可靠belief|
|Michelangelo|完整表面点及normal→固定query latent→occupancy logits；另有image/text条件生成。它的shape AE不是触觉网络|[官方代码](https://github.com/NeuralCarver/Michelangelo)，GPL-3.0；[官方权重目录](https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints)，本地已真实下载/完整加载过|可作一个正式3D先验/teacher候选；256×64 latent与occupancy语义明确。完整表面输入的原生资格与触觉学生资格必须分开|
|T-Rex|视觉/语言、F6历史、当前力和deformation→多速率动作；正式VQ-VAE及未来视觉辅助目标|[官方代码](https://github.com/ZhuoyangLiu2005/T-Rex)，MIT；[作者发布模型](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6)，页面MIT。发布路径内嵌tokenizer；本轮未下载验证其完整payload|保留快速触觉通路的直接参考；不提供shape/mass belief解码器。SUGAR 的20列不等于10指F6＋形变图，更不能拿8项摘要填成虚构六轴指端力矩|

权重“可复用”的三种状态不能混写：作者给出链接、实际文件已取得、完整结构加载并原生任务通过。NeuralFeels/TouchSDF/T-Rex 本轮达到的是官方发布内容核查；Active3D/Michelangelo 的已有本地实验也只证明其实际测试范围。

### 与路线选择相关的细节

- **NeuralFeels 的强项是持续融合，非预训练全物体幻想。** 官方项目明确区分在线联合map/pose与已知CAD tracking；它使用 RGB-D/触觉深度与 Theseus 位姿优化。官方 README 的实验频率为1–5Hz，测试3090/4090，因此不能承担快接触反馈。论文/项目没有替我们提供校准好的任意shape、pose、mass联合分布。[官方方法说明](https://suddhu.github.io/neural-feels/)、[运行说明](https://github.com/facebookresearch/neuralfeels#run-neuralfeels)。
- **TouchSDF 的接入位置应是局部面之后。** 官方流程先把真实marker图像变成模拟深度，再预测局部点云，最后拟合预训练DeepSDF；不能把它当“牛顿数字→网格”权重。README的20次触摸/5000优化迭代示例约需3090上5分钟，是离线补全参考。[官方流程](https://touchsdf.github.io/)、[脚本](https://github.com/maurock/TouchSDF/blob/master/scripts/pipeline_tactile_deepsdf.py)、[使用说明](https://github.com/maurock/TouchSDF#usage)。
- **Active3D 的mask有观测语义，不是成功标签。** 本地 `pterotactyl/utility/data_loaders.py` 与 `data_making.py` 把未提供、无触碰探测和有效触碰区别编码；不能将控制未收敛/没生成chart直接改成free-space。原全局GCN使用xyz＋mask；force、材料和质量不在其接口内。局部chart插值成功也不等于传感器恢复形状成功。[原模型](https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/reconstruction/vision/model.py)。
- **Michelangelo 输出是occupancy，不是标定SDF距离。** 本地 `michelangelo/models/tsal/sal_perceiver.py:277–306` 的 `query_geometry` 返回logits；原正式模型保留。最新完整表面832步结果4/4几何门槛、3/4完整数值通过，bolt IoU约.877<.90；这不是触觉补全或泛化。仍不能把由GT全表面得到的latent作为实时belief输入。[实际结果](../../../experiments/object_predictor_v1/overfit_repair_v1/michelangelo_persistent_replay_v1/OWNER_REPORT.md)。
- **T-Rex 并未证明必须先重建3D。** 其论文报告直接加触觉的π0.5基线下降，提出时序tokenizer、触觉中期训练及快慢专家。结论支持检查训练/融合接口，不支持“触觉天然有害”或“只有形状辅助任务才有效”。[原论文§4–5](https://arxiv.org/html/2606.17055v1)。

T-Rex配置要以实际发布checkpoint为准：主README的内嵌入口接 `[B,window,10,6]`，带归一化buffer；旧 `tactile_vqvae/README.md`描述的per-hand/1024码本不能直接当发布配置。主README也说明兼容加载可忽略不匹配层；项目复用时须另记录所有缺失/初始化权重，不能把部分加载当完整发布模型。[正式内嵌接口](https://github.com/ZhuoyangLiu2005/T-Rex#tactile-vq-vae)。

## 3. 扩展检索中值得保留、但不并入主线的方法

**Sparsh** 提供视觉触觉自监督表征和官方权重，可用于以后真正获得DIGIT/GelSight图像的情形；当前标量压力/剪切不在其输入域。它也不直接解决完整物体shape/mass。[官方仓库及权重入口](https://github.com/facebookresearch/sparsh)、[官方模型库](https://huggingface.co/facebook/sparsh)。

**Touch2Shape** 用触觉条件3D扩散与探索策略，更直接体现多种可能形状的生成先验；本轮核到正式CVPR论文，但没有确认作者官方代码、可下载checkpoint与许可的完整组合，因此不列为立即可执行替代。[CVPR2025原论文](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Touch2Shape_Touch-Conditioned_3D_Diffusion_for_Shape_Exploration_and_Reconstruction_CVPR_2025_paper.html)。

**SPOT** 的物体相对SE(3)轨迹适合示范接口；其代码使用FoundationPose做视觉物体跟踪，不能拿“得到物体轨迹”省略未知形状/遮挡下的感知难题。代码/数据采用NVIDIA Source Code License，不能与MIT等同。本轮未确认可直接复现我们任务的发布policy权重。[官方仓库](https://github.com/NVlabs/object_centric_diffusion)。**DITTO** 是示范轨迹变换的官方参考；本轮不把它或SPOT作为同时要训练的新policy。[官方DITTO](https://github.com/robot-learning-freiburg/DITTO)。

## 4. 质量与摩擦：条件辨识，不从形状猜一个真值

令 `F_hand→object` 是全部手接触施加到物体的合力，物体COM加速度为 `a`、重力为 `g`。只有其余外力已观测或可排除时，才有 `F_hand→object = m(a−g)`。准静态离地托举可近似用竖直支撑量/g；动态阶段不能忽略a。已知并测得桌面作用力时也可辨识，故条件应是“无**未观测**外力”，不是机械地要求任何环境接触都禁止估计。

形状及接触位置帮助解释力/力矩和支撑状态，但不能决定密度、内部配重或质量。只有面法向压力且方向未知时，必须先解决力矢量方向/覆盖；normal CAD近似＋剪切的确定性汇总也不能跳过这一步。无接触、地面支撑未知、明显滑移导致运动估计失真、丢测，都允许质量UNKNOWN。

静摩擦条件 `||F_t|| ≤ μF_n` 通常只给μ的下界；在局部库仑模型、力已标定、检测到临界滑移并有适当激励时，才可能估计摩擦。刚度需要接触形变/位移和受力关系，还要区分手自身柔顺性、接触几何及物体柔顺性。它们不是同一“材质类别”标签。

[Sundaralingam与Hermans的原论文](https://arxiv.org/abs/2003.13165)使用多指触觉、运动与因子图做惯性推断，摩擦辨识主动逼近滑移；说明相应物理任务有正式方法，但本轮未核到其可直接使用的完整官方代码/权重，不拟自制替代网络。[Fazeli等原论文](https://dspace.mit.edu/entities/publication/d6014ff4-98df-4877-93dd-5f7159cef9f1)进一步讨论已知外力缺失时惯性/接触参数耦合。上述动力学关系和本项目可观测性判断是据物理条件作出的分析，不是这些论文对SUGAR的已验证结论。

## 5. 最小belief接口：观测、推断、未知三层

建议把草案的扁平字典改为带来源的结构，仍只作为接口提案：

```text
track_id                    # 感知跟踪编号；不是TRAIN资产ID/GT物体类别
observed:
  contacts[{sensor_id, timestamp, frame, point/support, area,
            force_or_traction, normal_kind, validity, calibration_id}]
  robot_kinematics, gravity, measured_visibility
estimated:
  object_frame_hypotheses, relative_motion, contact_mode
  conditional_shape_hypotheses, prior_id, observed_support_mask
  mass_or_interval, friction_bound, validity_conditions
unknown:
  uncovered_surface, unresolved_pose_axes/symmetry, unidentifiable_parameters
quality:
  observation_residuals, coverage, age, uncertainty_kind, calibration_status
```

`normal_kind`必须区别实测、局部拟合、CAD fallback；`force_or_traction`注明受力对象、N/Pa以及参考系。shape prior可生成多个候选，但候选多样性、优化残差和后验概率不是同一个量。未校准时写“未校准假设/区间”，不可仅给latent加一列confidence就宣称可靠概率。

跨时刻融合必须使用 `p_object = T_object_world(t) p_world(t)` 或联合估计；不能把移动物体的世界触点直接叠加。未知形状和未知位姿存在耦合，平面/对称接触留下的不可观测方向必须保留。对称物体的位姿分布应在等价类上解释，而不是被迫选一个GT轴顺序。

未接触传感器可约束已知实体探针实际占据/扫过的局部空间，前提是标定、覆盖、检测阈值和运动误差支持；零载荷本身不能推出无限射线自由空间。先验补全也不能成为后续自监督的“实测真值”，否则会把早期错误反复强化。

## 6. 与Zero-WAM、demo-following、SMP的连接

Zero-WAM保留官方视频/动作/示范条件能力。belief先作为**独立观测记录和任务关系接口**，低置信度时允许只用可靠局部接触事件；何时、在哪层投影进原模型是未来适配实验，当前不改架构。完整demo可以作为任务条件，执行belief只能用当前和过去；生成未来视频不能回写成观测。官方注意力中action不直接读ICL的事实要在新增tokens时重新核验，不能简单拼接。[本地官方核查](OFFICIAL_RELEASE_REVIEW.md)。

示范中优先表示物体相对目标/支撑面的变化以及接触、抬起、释放事件。换demo应改变目标/阶段，而不是让episode编号或绝对秒数决定行为。快速触觉支路使用最新接触丢失、滑移和载荷变化；慢速shape/pose更新不能阻塞它。SMP负责身体运动先验，不提供物体感知或任务成功概率；其动作/身体表示由独立SMP审查确定。

以后恢复实验时，最小可证伪顺序是：

1. 原官方Zero-WAM基线与明确oracle物体状态诊断，先验证同一初态不同demo能导致不同正确目标；oracle标签只用于该诊断。
2. 固定一条带实际滑移/物体运动的轨迹，比较世界直接堆点与合法物体坐标配准；按已观测支持域误差、位姿等价类误差和未知覆盖评价。不得用GT pose构造真实估计输入。
3. 同一局部观测，比较不补全和**一个**完整官方prior条件补全；已观测面必须一致，未观测区域单独评估。完整表面AE通过不能代替此项。
4. 同一执行器/任务中比较vision与vision＋touch，保留快触觉反应；再单独验证可辨识质量和主动探测。既报误差也报拒答/UNKNOWN覆盖，避免通过少报困难情况抬高精度。

这里均为后续建议；本轮未执行。

## 7. 对RESEARCH_PLAN.md必须修改或补强的地方

1. **第1节 `object_id` 改为 `track_id`，扁平belief分观测/推断/未知。** 资产ID、形状GT、真实物体frame不能混进实际输入；单独标oracle诊断。
2. **`pose_distribution/mass_distribution/slip_probability`注明资格。** 官方参考并不自动产出校准联合后验；初版可保存hypotheses、UNKNOWN、残差及区间，只有实际校准后再称概率。
3. **第2节加当前传感器真实信息边界。** 现有仿真接触面比稀疏数字更强；normal拟合/CAD、剪切分配、体素混手与力方向必须标注。已有力摘要一致不是物体学习成功。
4. **第2节增加无接触≠任意free-space、旧触点必须随物体配准。** 后一点草案已有，应成为输入资格的硬约束。canonical origin/scale不得来自GT。
5. **第3节保留快支路，收窄T-Rex解释。** 原论文支持规范接入/时序/中期训练，不支持所有控制必须经3D；原VQ子README不是发布checkpoint配置。缺六轴/形变通道须如实列缺件。
6. **第8节C把可辨识局部目标作为首个感知闭环。** 不要重新默认shape、pose、mass、material一并单点回归。先看局部约束是否减少pose/shape歧义，再谈整物体补全和质量；原生prior与SUGAR条件补全设两个gate。
7. **质量条件写成无未观测外力且运动足够可辨，摩擦写成界/条件估计。** 控制器load达到某目标不等于真实质量信息已具备；图像材质类别也不等于μ或刚度。

草案关于两类“想象”、快慢通路、未来观测隔离、oracle接口诊断、对称性及失败分母的方向合理。最重要的收窄是：**当前真实可用的是接触与几何约束；完整形状、质量和材质是有条件、可拒答的推断。** 没有官方发布模型可以直接替我们跨过传感域和可观测性缺口。

核查边界：本轮读取官方网页/论文、现有本地源码及少量GitHub文件树元数据；未新增大权重下载。TouchSDF权重元数据来自官方 `master` 树；其payload/许可证待后续单独核实。旧静态全局AE准备文件仅保留，不作READY、不启动。
