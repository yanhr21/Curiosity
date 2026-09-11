# 当前科学状态

记录日期：2026-09-11。最新用户目标：拉取 GitHub sugar 新代码、区分 bug 修复与数据／GPU
适配，并继续 overfit 直至指标和实际可视化正常。已拉取 4bcf8582；旧结果均保留。

当前使用原始 PhysX 199 轨迹／160-20-19 split、原八个 TRAIN 案例、完整 10.681B 模型，
不使用新提交中的运动学栅格替代数据。单 H200 291647/server31/step0 已在 tmux 保留。
初始 23 项 CPU 回归及当前代码的官方完整 30 层数值检查通过；后续 CPU 回归为 67 项。
本轮 32 步拟合、动作采样及动态视觉检查仍未通过，详见下方实测结果。
首个新 endpoint 为 overfit_resampled_noise_20260911，32 次噪声／时间重采样更新后自动
评估并渲染所有八例。此 endpoint 不等于目标完成，失败后继续依据证据分析和 overfit，
不自动启动 formal 或 physics。
详见[代码对比与完成条件](../experiments/demo_following/paper_zero_wam_v1/bugfix_audit_20260911/CODE_COMPARISON.json)。

9 月 11 日 12:49 初始八例／八噪声评估已完成：匹配条件的 video/action/IFP loss 为
`0.1872653 / 1.4499187 / 0.6574702`。teacher-action 隔离及生成视频到动作的连接通过，
正确 demo 的损失优势未通过。这是**更新前基线**，不能称为本轮 overfit 结果。
14:32 最新完整包级 CPU 回归 56 项通过，包括独立动作诊断的 JSON 配置 tuple 恢复与
动作时序检查；未修改正在运行的训练代码、模型或配置。这不是整个模型正确性或 overfit
成功的证明。
12:57 已在同一计算 shell 的当前前台任务之后排入
`run_paper_zero_wam_bugfix_endpoint_held.sh`：终点完成后导出全部 64 张冻结 VAE 对照帧；
先保存逐关节和时序动作回读；若采样动作未优于零／均值或时序基线，再做同一完整
checkpoint 的真实未来条件诊断。读取使用已有的仅导入重试入口，不重试训练或推理。
该诊断于 17:33 自动启动，launcher PID/PGID `3545495`，日志前缀为
`logs/held_291647_bugfix_endpoint`，不要重复排队。动作回读和 64 张 VAE 对照已完成。
16:53 在 server31 核实首段 32 次真实更新已全部完成，索引连续为 0--31，warmup 后
LR 为 `1e-5`。32 次更新均有限且 GPU/CPU AdamW 主参数回读一致；八个样本记录始终
相同，共使用 256 个不同噪声种子。前两次检查的 64 个 cross-attention 模块均有非零
梯度。22/32 步触发既定梯度裁剪；第 32 次范数 `1.09558`，未触发裁剪。平均每步约
`450.82 s`，主要耗时仍为 CPU 梯度搬运／累加。第 32 次单次 video/action/IFP loss 为
`0.0137297 / 1.0364978 / 0.1997930`；由于每步噪声／时间不同，不把它们与平均初始
探针直接相除来宣称通过。同一进程进入终点评估，尚未得到新画面或完整科学结论。

17:12 新 `FITTING_RESULT.json` 已发布：本段拟合和整体 prompt 判据仍未通过。
两组八次噪声采样的终点／初始比值如下（CarryBox、KickBox 分别检查也未整体通过）：

| 32 步新终点／初始 loss | 视频 | 动作 | IFP |
|---|---:|---:|---:|
| 匹配噪声采样组 | 0.627887 | 0.781634 | 0.333485 |
| 独立噪声采样组 | 0.055699 | 0.733700 | 0.433932 |

IFP 比值达到既定 0.5 阈值，动作在两组均未达到；视频只在独立采样组达到。视频结果
对采样组仍敏感，八次平均并不消除采样方差；不能只引用更有利的一组。完整 checkpoint
正在保存，实际动作与逐帧画面尚待渲染／诊断，因此还不能定位唯一根因或宣称生成正常。

17:16 模型与优化器保存完成，`OVERFIT_RESULT.json` 已发布。prompt 细分结果：视频在
两项任务的 wrong-task／reversed／same-task-alternate 比较中均更偏好正确示范，正差值
从更新前 2/6 项变为 6/6 项（`0.000873–0.001990`）；IFP 仍为 3/6 项，三个负差值
为约 `-3.20e-6` 至 `-1.11e-5`，因此整体 prompt 判据失败。teacher-action 隔离和
生成未来到动作的连接均通过。它们说明条件路径在工作，不证明动作重建或语义跟随成功。

17:40 全部八例的 64 张生成帧及 64 张 VAE 对照已检查：大块伪影和机器人／箱体消失
明显改善，但 slot01／02／06／07 仍有明确转身、起身、伸腿推箱或姿态时序偏差。
冻结 VAE 对照均接近真值；新导出中的 24 张已看过对照经直接 RGB 像素相等检查确认，
另 40 张逐帧补看，未以少量代表帧代替全序列。八例动作 MSE 为 `1.016–1.109`，全部
差于零动作基线 `0.203–0.295`；动作时序误差为零变化基线的 `151.4–533.5` 倍。
逐帧证据见[本轮视觉检查](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911/VISUAL_INSPECTION.json)。

只读 RGB 分析显示前景仅占 `2.22%–4.00%`：八例整图误差均优于保留最后已观测帧的
因果静止基线，但前景时序误差均差于零变化基线（`1.16–2.19` 倍）。这不是新增的
训练目标或降低后的通过标准，而是防止整图 MSE 掩盖动态偏差的诊断。
见[RGB 时序回读](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911/RGB_TEMPORAL_READBACK.json)。

18:00 真实未来条件诊断已正常退出，八例均严格复现已保存采样动作。换成真实未来后，
仍为 `0/8` 动作优于零值基线，单例 MSE 变化的绝对值至多 `0.000744`、相对变化至多
`0.0684%`。这八例不支持将当前动作失败主要归因于生成视频；下一步检查动作专家自身
的去噪拟合。当前没有第 33 次更新，也没有 formal／physics 启动。
见[完整逆动力学诊断](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911/INVERSE_DYNAMICS_PROBE.json)。

17:50 已把同一完整 checkpoint 的 `--action-flow-only` 诊断排在当前前台进程之后，
由同一保留 H200 和 pipeline lock 串行运行，日志前缀为
`logs/held_291647_bugfix_action_flow`，不要重复排队。它要求八例真实未来诊断完成且
严格回放成立，然后在 `t=0.02/0.1/0.25/0.5/0.75/0.9/1.0` 各用两组固定噪声，
共 112 次完整主干前向，读取动作速度误差及配对噪声响应，不训练或改变采样结果。
速度目标使用训练时原始精度减法；噪声／真值分量投影只是统计诊断，不是替代模型。
17:52 最新 67 项 CPU 回归通过。该 GPU 诊断已于 18:00:35 自动启动，PGID `1308711`、
Python PID `1308770`；18:01 正在加载完整 checkpoint，结果尚未发布。

15:09 续训入口已准备但**未启动**：`train_single_gpu --resume-overfit-endpoint <parent>`
要求父终点完成评估、八例渲染、64 张 VAE 对照及必要的逆动力学诊断，再精确恢复 FP32
模型和 CPU AdamW，每段只追加 32 次更新，写入独立 `..._step64` 等目录。保留原始初始
基线，同时报告新段相对父终点的变化。首段没有热加载这些入口修改，续训仍未启动。
15:58 最新 64 项 CPU 回归通过，包含权重／状态文件恢复及恢复后下一步的
逐元素一致性测试；完整 10.681B checkpoint 的真实续训验证仍待父终点完成。
可选 CPU 搬运缓冲区复用仅通过算术／所有权测试和 CPU 小规模性能对比，尚未验证 GPU
搬运提速，未启用到训练；证据记录在现有代码对比 JSON 中。

旧修复版八例的逐关节回读再次证实动作失败：动作 MSE `0.648–0.752`，逐关节常数均值
oracle 的 MSE 仅 `0.0464–0.0904`；相邻帧动作变化误差为零变化基线的 `89.5–352.6` 倍。
这不是新实验结果；详见本地 `bugfix_audit_20260911/HISTORICAL_ACTION_RECONSTRUCTION.json`。

已验证的对比边界：旧梯度裁剪实际触发 24/32 步，而非“从未触发”；真实 attention mask
阻断了 action-target 到视频 token 的所有路径，因此替换这些 token 不能解释视频条件修复。
学习率／初始化／IFP detach 是待实测的优化或方法变更，不能直接等同于已经定位的 bug。
论文的 IFP 目标是监督主干表示，detach 会改变这条训练路径；保留为明确标注的变体诊断，
不宣称忠实复现。[Zero-WAM §3.3](https://arxiv.org/html/2608.26103v2#S3.SS3)
本轮启动时的 `DIAGNOSTIC_CONTRACT.json` 把 `objective_changes` 错标为 false；以实际
`ifp_trunk_gradient=false` 配置和上述方法说明为准。原始记录保留，后续记录的标签已修正。

以下是 9 月 10 日已结束的历史结果，不是当前新实验的结论。

## 修复版 overfit：完成但未通过

本地论文重建保留 10,680,751,069 个参数：独立的 30 层／3072 宽视频和动作专家，
以及四个完整宽度 IFP heads。这不是 Zero-WAM 作者的官方实现。

修复了继承的空文本投影／cross-attention、61 帧全跨度示范支持及反转支持一致性，
并接回官方 RoPE／FlashAttention。199 条新示范缓存完成检查，机器人目标与动作保持不变。
官方完整视频路径与 IFP 数值对照通过；这不等于整个模型或训练目标已被证明正确。

相同 8 个训练案例、固定噪声和时间、batch 8，恰好完成 32 次真实更新：

| 终点／初始 loss | 视频 | 动作 | IFP |
|---|---:|---:|---:|
| 固定噪声／时间 | 1.666156 | 0.222756 | 0.162461 |
| 独立噪声／时间 | 3.097468 | 0.581112 | 0.854718 |

整体拟合和示范区分判据均失败。全部 8 个实际 TRAIN 案例已检查首／中／末帧：
生成画面块状伪影明显，机器人和箱体结构破坏或缺失；8 个冻结 VAE 对照接近真值。
这不是 held-out 展示，也不是闭环物理 rollout。

- [完整结果与原因分析](../experiments/demo_following/paper_zero_wam_v1/overfit_repaired_fixed_noise_20260910/OUTCOME_ANALYSIS.json)
- [执行完成核对](../experiments/demo_following/paper_zero_wam_v1/overfit_repaired_fixed_noise_20260910/COMPLETION_AUDIT.json)
- [全部 TRAIN 视频与帧](../experiments/demo_following/paper_zero_wam_v1/overfit_repaired_fixed_noise_20260910/training_videos)
- [逐案例视觉检查](../experiments/demo_following/paper_zero_wam_v1/overfit_repaired_fixed_noise_20260910/VISUAL_INSPECTION.json)
- [实现修复审计](../experiments/demo_following/paper_zero_wam_v1/REPAIR_IMPLEMENTATION_AUDIT_20260910.json)

## 原因边界与未执行建议

不能把失败简单归因于训练规模或模型太小，也没有找到唯一根因。固定时间／噪声拟合
不等于生成式 overfit；动作训练使用真实未来，部署依赖生成未来，存在误差传播风险。
修复后视频分支损失反而上升，需要先隔离视频、逆动力学与 IFP 的训练路径。

本轮训练计算约 7.1 小时，修复版 attention 适配器有严重效率退化，尚未通过性能剖析
定位唯一瓶颈。建议依次验证官方完整视频路径、真实未来到动作、再到联合生成式 overfit；
这些是尚未执行的方法建议，不是自动训练队列。

## 既有 endpoint 与执行边界

- [step-700 完整分析](../experiments/demo_following/paper_zero_wam_v1/formal/STEP700_OUTCOME_ANALYSIS.json)：
  390 组／1,950 次评分及 8 段预测视频完成，未通过示范跟随。
- [修复前固定噪声诊断](../experiments/demo_following/paper_zero_wam_v1/overfit_debug_fixed_noise_20260909/OUTCOME_ANALYSIS.json)：
  保留原始失败证据，不覆盖。
- 没有第 33 步、全量第 701 步、物理 rollout 或额外参数／种子扫描。
- 训练和渲染进程已退出。整理时单 H200 288297 / server23 仍在 tmux
  `curiosity_pzw_overfit_h200_20260909` 保留；它是带日期的观测，不保证将来仍存活。
- HOST 为负结果，BPP 与 tactile 为未激活历史路线；归档旧计划不会自动重启它们。

旧详细方法、预算、计划与 TODO 见[归档说明](archive.md)。
