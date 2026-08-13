# TODO 15: Online Whole-Hand Patch Tactile for Sudden-Mass Adaptation

## A. 文档与冻结合同

- [x] 将 Plan/TODO 15 设为唯一活动队列，并把 Plan/TODO 14 移入各自 `legacy/`。
- [x] 在 `AGENTS.md` 顶部写明 online、54-patch、causal slip、leakage audit 和
  serious matched-training 规则。
- [x] 固定 nominal mass `0.3023375869 kg` 与 `1.5x/3x/6x/10x` sweep。
- [x] 固定 actor/teacher/critic 边界：`504-D` deployable actor 禁止 measured
  object state；官方 `890-D` Refiner 只能作为 training-only teacher/critic。
- [x] 固定 policy 的空间单元为每手 27 patches；taxel 只留在官方 sensor 内部
  和 sensor debug，不得作为 actor 单元。

## B. IsaacLab live mass jump

- [x] 在完整 G1 CarryBox env 中加入 runtime mass/inertia change event。
- [x] 代码路径保证 jump 不改变 object geometry、material、RGB、pose、reference、sensor
  history 或上一动作。
- [x] 实现每个 jump 后的实际 mass/inertia readback 和 event timestamp 评价字段。
- [x] 实现从 frame 0 连续运行、稳定抬升 10 帧后随机等待 10--50 帧再 jump；
  scheduler 不读取 TacSL，双手连续接触 10 帧由 live trace 独立检查。
- [x] 为 `1.0x` no-jump 条件加入 matched placebo event clock；不写 PhysX mass/
  inertia，并单独记录 `mass_changed=false`。
- [ ] 在真实 IsaacLab physics step 中确认上述各项；两块不同 H200 上的原始
  collector、force-only 和曾成功的 rendering 路径均在 scene creation 前遇到
  相同 `VK_ERROR_DEVICE_LOST`，不能误记成 Plan-15 sensor/mass/slip 失败。
- [ ] 完成 no-jump、`1.5x/3x/6x/10x` 的无学习物理可恢复性 sweep。

## C. 54-patch online observation

- [x] 从官方 bilateral 27-patch `VisuoTactileSensor` 当前帧在线读取 raw tensor。
- [x] 在 GPU 上逐 patch 归并 `contact`、`normal_load_n`、`mean_pressure_pa`、
  `shear_x_n`、`shear_y_n` 和 `friction_utilization`。
- [x] 固定 `[B,4,2,27,9]` contract、anatomical order、单位和符号；公共归一化
  尺度须等 live sweep 后冻结。
- [x] 实现从 live mass-sweep trace 统一拟合 9-channel 公共尺度的工具；真实 scale
  JSON 仍必须等 live sweep，禁止先填猜测值。
- [x] 保证 actor observation 中不存在 20x25 taxel 维度、普通 ContactSensor、
  `hands_contact_label` 或 object-state proxy。
- [x] 实现 exact-zero no-sensor-read observation，保证 zero encoder output 也为零。
- [ ] 用 synchronized patch visualization 检查压力/剪切变化与世界接触同钟对应。

## D. IsaacLab patch slip callable

- [x] 实现 batch-stateful `PatchSlipDetector.update(...)` 与 reset mask。
- [x] 输入仅限当前/历史 patch contact、pressure、signed shear、friction utilization
  和 timestamp。
- [x] 输出每个 patch 的 `NO_CONTACT/STICK/INCIPIENT/GROSS`、`slip_score`、
  `incipient_slip` 和 `gross_slip`。
- [x] 将 callable 接入 IsaacLab observation term；真实 GPU runtime 逐步读取仍待
  Kit/Vulkan 恢复后确认。
- [x] 实现独立 evaluation-only oracle：逐 patch 最大 active-taxel simulator
  tangential speed，并输出 precision、recall 和 onset delay；oracle 与 detector/
  actor/mass scheduler 隔离。
- [ ] 在 controlled stick-to-slide 与 CarryBox jump/slip 中评价 precision、recall、
  false positives 和 detection delay；relative velocity 仅作标签。
- [ ] 确认没有 offline replay、future frame、mass/jump flag 或 object motion 输入。

## E. 质量信息泄漏审计

- [x] 实现串行 paired sweep 入口：每个预定 seed 先采集 nominal `1.0x` 动作，
  再把同一动作逐帧重放到 `1.5x/3x/6x/10x`，完成后运行 leakage analyzer 与
  live scale fitter；真实 trace 尚待 IsaacLab runtime 恢复。
- [x] 冻结 leakage seeds `150814/150815/150816`、training seeds
  `151014/151015/151016` 与 frozen-evaluation seeds
  `152014/152015/152016`。
- [ ] 先记录 nominal controller action sequence，再开环重放同一 sequence 采集
  paired no-jump 与四倍率 jump，避免 teacher object-state action 泄漏。
- [ ] 分别导出 object-state、proprio-only、patch-tactile、patch-tactile+slip 信号组。
- [ ] 在 jump 前 0.5 s/后 1.0 s 窗口报告原始变化、mass-factor linear-probe
  balanced accuracy 和 change-onset latency。
- [ ] 明确证明 deployed actor 没有 `obj_lin_vel_b`、mass、jump flag、RGB 或
  simulator contact velocity。
- [ ] 根据审计结果把最终问题标为“触觉独有感知”或“在 proprioception 上的
  增量帮助”；不允许预先选择前者。
- [ ] 若 live patch load 对质量变化无响应，先修 sensor/aggregation/physics，
  不开始训练。

## F. Serious matched training implementation

- [x] 实现 shared anatomical patch-token encoder：9->128 projection、hand/patch/
  time embedding、3-layer 4-head Transformer、128-D pooled output。
- [x] 接入已有 SUGAR `512/256/128` actor、29-D action 和官方 510-D Tracker warm
  start；H200 structural preflight 已确认 zero-patch action error `1.31e-6`、encoder
  gradient 非零。
- [x] 将新 policy class 注册到 frozen Refiner teacher 和 repository-native BCPPO
  runner；启动器在 scale JSON 缺失时拒绝训练。
- [x] 定义三个共享 policy/runner 配置的 process-local 分支：`Z`、`P`、`PS`，
  并分别提供 one-update preflight 与 512-update formal task。
- [x] 在配置层保持 critic、teacher、optimizer、reward、physics、mass sampling
  和 512-update budget 一致；live runner 实例化仍待 Kit/Vulkan 恢复。
- [ ] task reward 只评价物理持稳/跌落/机器人稳定；不把 mass ID 或 jump flag
  作为 actor 答案。
- [ ] 冻结 3 个 paired formal seeds 和未参与训练的 frozen-evaluation profiles。

## G. 串行训练

- [ ] `Z`：one-update preflight，确认不读取 sensor、patch/slip exact zero；随后
  完成 512 updates。
- [ ] `P`：one-update preflight，确认 live patch signal 和 encoder gradient；随后
  完成匹配 512 updates。
- [ ] `PS`：one-update preflight，确认 live patch 与 callable slip 同时进入；随后
  完成匹配 512 updates。
- [ ] 每个分支完成后保留 GPU allocation；停止/失败时只终止记录的 child PGID。
- [ ] 不在三个分支之间修改架构、reward、seed、mass 分布或训练预算。

## H. Frozen evaluation 与报告

- [ ] 对 no-jump 和每个 mass factor 分别比较 hold success、drop/fall、height loss、
  orientation、recovery latency 和 safe-lower outcome。
- [ ] 比较 `P-Z`、`PS-P` 和主要的 `PS-Z` paired 95% confidence intervals。
- [ ] 检查 action divergence 只发生在 jump 后 live observation 更新之后。
- [ ] 报告 nominal no-jump 是否因无条件强握/降姿而退化。
- [ ] 为每个正式 factor 制作 `Z/P/PS` 同钟 H.264：完整 G1 世界画面、左右 27
  patch contact/pressure/shear/slip、jump overlay 和物理结果。
- [ ] 视频主图只使用 patch 单元；taxel detail 只能进入独立 debug 视频。
- [ ] 只有 frozen physical behavior 改善才能写“触觉帮助训练”；gradient/loss/
  action difference 只能写“触觉被使用”。
