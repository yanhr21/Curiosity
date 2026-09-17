# Agent 大脑与触觉 belief、Zero-WAM、身体执行的接口审查

2026-09-17。独立只读研究；本轮未启动训练、推理、仿真、旧 AE 资格或控制实验。本文中的接口、监控器和验证均是设计建议，不是已有实现或通过结果。根方案见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)，官方发布状态见 [OFFICIAL_RELEASE_REVIEW.md](OFFICIAL_RELEASE_REVIEW.md)。

## 结论

Agent 有用的角色是把示范目标、当前证据和已验证技能组织成可中断的执行过程，并在失败后改变策略。它不能用语言推理补出尚不可观测的物体状态，也不能使一个失败的夹持控制器自动变得可执行。

现方案最需要补齐的是三份合同：**什么是当前事实；一个技能承诺执行什么；什么证据允许宣布完成。** 在这些合同成立以前，加入更多 world model、先验或 agent 都可能把同一个底层失败换成更流畅的解释。当前已知 box 状态回归、完整表面 AE 和单独控制测试，均不能替代示范条件下的真实任务闭环。

## 1. 从主要官方工作实际能借什么

|来源|已核实的方法与代码边界|本项目可以借鉴的部分|
|---|---|---|
|[SayCan 项目](https://say-can.github.io/) / [论文](https://arxiv.org/abs/2204.01691) / [官方代码目录](https://github.com/google-research/google-research/tree/master/saycan)|结合语言相关性和已有技能的 value/affordance 进行选择。官方 README 明确公开实现是桌面模拟版本，不能当作完整真实厨房系统。|先列真实可执行的技能，再排序。学到的 value 是可行性估计，不是碰撞、接触或全身平衡的硬证书。|
|[Inner Monologue 项目](https://innermonologue.github.io/) / [正式论文](https://proceedings.mlr.press/v205/huang23c/huang23c.pdf)|将成功检测、场景描述、主动问答反馈给冻结 LLM。论文限制节承认部分场景描述来自人或脚本 oracle，检测错误和低层技能会限制性能。项目页未提供可核实的官方代码入口，本次不声称复现实现。|反馈必须来自环境证据。主动提问、失败恢复值得借鉴，但不能把自动描述器或 agent 自述默认当真。|
|[ReKep 论文](https://arxiv.org/html/2409.01652v1) / [官方代码](https://github.com/huangwl18/ReKep)|以关键点关系表示阶段目标/路径约束，约束失效时回退。局部预测使用抓持关键点随末端刚体运动的假设，视觉跟踪再纠偏。官方 demo 直接使用模拟关键点、mask、SDF，未发布真实感知管线。|关系约束和阶段回退可复用为设计；滑动接触不能直接套刚性绑定。该 demo 不是 G1 或触觉闭环的现成实现。|

SayCan 的[官方 notebook](https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb)还提供了更具体的复用边界：`affordance_scoring` 因没有 RL value function，用检测到的 pick/place 对象是否存在给出 0/1 分数，`done()` 使用固定 0.2；示例先生成计划列表，再依次运行 CLIPort。它不能直接提供我们所需的抓持稳定概率或每阶段失败恢复。

额外代码核查：[ReKep main.py](https://github.com/huangwl18/ReKep/blob/main/main.py) 在每轮更新观测、检查约束、重算路径并执行有限动作；最后阶段动作队列耗尽后执行抓放操作并返回。这是程序的阶段结束逻辑，不能作为我们搬运成功的独立证据。其默认机器人与 IK 也不是 G1。

## 2. 世界状态应分成四种类型

|类型|允许存放的内容|禁止的混用|
|---|---|---|
|`Observed`|带时间戳的实际 RGB-D、触觉读数、本体状态；已知手几何/标定；明确实际采样的力或位置|不能把命令位置当实际位置；压力积分、PCA 法向、姿态估计需标计算来源，不能统称原始传感器真值|
|`Estimated`|由过去和当前观测推断的物体身份、pose/shape belief、滑移、质量区间；附有效性、覆盖与不确定性|无接触或遮挡不等于物体不存在；未知质量不填“合理默认真值”；局部接触面不等于完整物体表面|
|`Predicted`|绑定某个 belief snapshot、示范和动作假设的未来视频、动作或状态分支|未来成功画面不写入当前观测；不同分支不能混合为一条真实历史；概率未校准时不能假定置信度可信|
|`Oracle`|模拟物体 GT、未来真实轨迹、评估标签|只用于明确标记的训练/评价/隔离诊断，不能进入部署 belief、探测选择或成功判断|

建议每个事实保留 `value / frame_id / units / observed_at / valid_until / provenance / validity`。有效性至少有 `KNOWN、UNKNOWN、INVALID`；未知与失败是不同状态。对象身份需跨遮挡保持关联，关联失败应报 unknown，不能悄悄换一个物体继续统计。

触觉点先由实际手位姿进入世界坐标，再通过当前物体 pose belief 关联到物体局部；旧触点只在运动配准可信时累计。质量要求有效受力条件；不同接触/外部支撑假设应分支记录。无需先得到完整 shape 或精确 mass 才能执行所有低风险技能：只查询当下决策所需的量。

## 3. Typed skill 合同：不仅是一个自然语言名字

下面是拟议类型，单位和动作语义必须由注册表版本固定；不给未验证的机器人随意指定数值限幅。

```text
SkillRequest {
  request_id, skill_id, skill_version,
  embodiment_id, capability_profile_id,
  belief_snapshot_id, demo_id, demo_stage_id,
  target: {object_ref, reference_frame, relation, optional_SE3},
  action_schema: {pose_or_delta, position_unit, quaternion_order, timebase},
  preconditions: [typed_predicate + required_value + evidence_freshness],
  invariants: [contact / clearance / reachability / body_constraint profiles],
  expected_effects, success_monitor_id,
  deadline, retry_budget, interrupt_policy
}
SkillResult {
  request_id, status: RUNNING | SUCCEEDED | FAILED | UNKNOWN | PREEMPTED,
  last_executed_command_id, observation_snapshot_id,
  predicates_with_evidence, failure_code, remaining_budget
}
```

注册表只暴露已有并验证过的技能。LLM 提出新名字不等于系统自动获得新能力。执行器验证 schema、坐标、对象关联、证据新鲜度和身体可行性；拒绝应有可解释原因，例如 `STALE_OBSERVATION、CONTACT_UNVERIFIED、UNREACHABLE、FLOOR_BLOCKED、SLIP、DEADLINE`。完整双臂/全身动作必须共享一次一致性检查，不能两个手分别认为可行后直接相加。

根方案的 `probe_then_carry` 以 bilateral_contact 为前置条件，若它还负责首次接触获取，就无法从无接触开始。建议阶段拆为 `acquire_contact → optional_probe → establish_grasp → transfer → place_and_release`，各自有局部前置条件。复合技能仍可保留，但必须显式包含 acquisition，而不是从假设抓稳的状态起算。

Agent 只改变目标/技能/探测计划。局部控制器保留快速力反馈、滑移反应和全身约束；即使 agent 或视频生成阻塞，也应可中断并进入当前接触模式允许的安全状态。不能一律把所有异常处理成“原地 hold”：有些接触失稳需要受约束卸载或撤回，这也必须属于已验证技能。

## 4. Zero-WAM 与 SMP 的具体责任边界

本次读取官方 commit `08e2c4ae41e2b63573a299825cebe6753481407c` 的 `wan_va/wan_va_server.py`、`wan_va/mcp.py` 和 README。公开 `infer(obs)` 支持 reset、观测缓存更新、生成动作 chunk；ICL 路径从噪声生成视频和动作，MCP helper 用真实未来帧及有效 mask 构造训练目标。参见 [官方 server](https://github.com/robbyant-research/Zero-WAM/blob/08e2c4ae41e2b63573a299825cebe6753481407c/wan_va/wan_va_server.py) 和 [MCP helper](https://github.com/robbyant-research/Zero-WAM/blob/08e2c4ae41e2b63573a299825cebe6753481407c/wan_va/mcp.py)。

**关键修订：联合生成未来视频和动作，不自动等于支持任意指定动作的反事实物理预测。** 已读公开接口没有直接提供“固定候选动作 A/B，然后分别预测后果”的通用调用合同。可以称示范条件下的联合未来/动作提案；若要用于多候选 MPC 排名，须另核对动作条件注入、采样分布与实际后果校准，不能直接凭漂亮视频打分。

执行短前缀、刷新真实观测、取消过期后缀是拟议闭环策略；具体执行长度/缓存刷新必须遵守官方 chunk 协议并测时延，不能宣称当前已经低延迟可用。每个返回 chunk 绑定 observation snapshot、demo stage、动作 schema 和有效时间。观测更新、目标变化、接触突变时，旧结果可被判过期；想象缓存与真实观测缓存要分清。

30D 容器/16 个有效双末端通道不是 G1 全身动作。适配器须说明绝对/相对位姿、相对哪个时刻、四元数顺序、夹爪语义，以及身体执行器的可达域。SMP 在本方案里是官方运动先验对应的身体策略训练组件；不能代替任务成功判据，也不能把分数当成对某条接触轨迹的安全承诺。SMP 具体特征/身体兼容性以并行官方核查为准，本审查不重复声称其已集成。

## 5. 闭环事件、主动探测和失败恢复

建议的因果顺序是：

1. 同步当前观测，更新 belief 和各谓词有效性；保存实际执行历史。
2. Agent 选择下一技能或信息请求；执行器检查合同。UNKNOWN 前置条件不能默认放行。
3. Zero-WAM 或已有技能提出短时动作，身体执行器检查约束并执行可中断前缀。
4. 快速监控器处理接触丢失、滑移、过载和碰撞风险；这些事件不等待 LLM 完成推理。
5. 新真实证据验证 expected effects；偏差持续或关键约束失效时，回退到仍可成立的阶段，重新探测、重抓或终止。
6. 独立成功监控器确认目标完成，才返回 SUCCEEDED；耗尽动作队列、手到位、模型说 done 均不足。

事件需有持续条件/滞回和证据时间，避免每个噪声采样都触发昂贵重规划。相同失败码且证据未改变时，不重复无限尝试；每次恢复必须改变可检查的条件，例如视角、接触配置或可达目标。

主动探测应由“答案是否会改变可执行决策”触发：只需判断是否滑移时，不强制辨识精确质量；遮挡导致对象身份不确定时先改变视角；摩擦未知时可提出已验证的小幅载荷探测，但不能把未知摩擦范围内的危险搬运当探测。信息增益若无校准模型，只能是启发式，不能报精确期望值或保证更优。

## 6. 成功判据与学习/推理分界

搬运目标至少分开检查：正确对象关联；抓持/支撑成立；实际物体而非仅手达到目标关系；释放后目标支撑稳定；过程没有违反约束。只凭“接触消失”无法区分成功释放与掉落；只凭手上升不能证明物体被抬起。缺少可靠支撑或姿态观测时返回 UNKNOWN，并在总任务分母中保留，不删掉困难案例。

|环节|推理时可做|必须另列为训练或资格工作|
|---|---|---|
|Agent|读技能注册表、当前证据、示范图，选技能/探测/恢复|新技能学习、affordance 校准；无需为了初始接口先微调 LLM|
|Belief|用已发布/已验证估计器融合当前和历史观测|传感器适配、shape/pose/物理属性监督；标签只在训练侧|
|Zero-WAM|保持官方模型与条件合同生成候选|object/contact tokens、G1 动作适配、未来物体辅助目标属于新的学习扩展|
|身体执行|已有技能/策略/控制器跟踪目标，处理快反馈|SMP 对应训练与身体映射，以及新技能能力资格|
|成功监控|计算实际证据与目标关系|阈值/检测器需事先定义或校准，不能看到失败后改成功定义|

未来 Agent 增量验证应固定同一低层技能、同一观测/成功监控和总时间预算，对比脚本与 agent；所有对象/任务/扰动和失败保留。分别报最终成功、恢复成功、UNKNOWN、错误成功声明、额外动作/耗时/重规划次数。恢复率的分母是发生可定义失败的所有任务，不能只统计成功重试。感知精度提高、阶段动作完成或视频看起来更合理都不是任务成功替代指标。

## 7. 对 RESEARCH_PLAN.md 的优先修订

1. 第 1/5 节把官方 Zero-WAM 描述为“示范条件联合未来/动作提案”；任意动作反事实预测另列待验证能力。
2. 第 7 节补上述 typed 合同、三值谓词、实际执行回执与独立成功 monitor，并拆 acquisition 与既有接触前置条件。
3. 第 7 节加入 Inner Monologue 的环境反馈依据，同时明确其 oracle/成功检测限制；保留 SayCan 可行性估计与硬约束的区别。
4. 第 7 节补 stale chunk 取消、快反应优先级、事件持续条件及失败预算；agent 不负责逐步 torque。
5. 第 8 节 B 的 oracle 应只替换当前估计状态；不能顺便给未来成功轨迹或 GT 最优动作，否则不能隔离身体接口问题。
6. 第 8 节 D 对比 oracle/视觉/触觉时，应固定或明确适配各输入分布；同一个只见过 oracle 的策略直接吃有偏 belief，结果混入分布迁移，不能全归因传感器信息。
7. 第 8 节 F 固定技能和成功检测后才比较 agent。暂不要求新大模型训练；先检查合同能否支撑现有技能的真实失败恢复。
8. 第 9 节加错误成功声明与 UNKNOWN 覆盖；未来视频显示预测起点/目标时刻/分支 ID，与随后真实结果对齐，避免将重采样后的视频误当原预测。

这些修订是研究建议，不是进一步实验授权。本轮仅写报告，旧失败、未执行的控制/AE 准备和当前下载状态均保持原样。
