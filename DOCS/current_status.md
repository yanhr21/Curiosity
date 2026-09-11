# 当前科学状态

记录日期：2026-09-11。最新用户目标：拉取 GitHub sugar 新代码、区分 bug 修复与数据／GPU
适配，并继续 overfit 直至指标和实际可视化正常。已拉取 4bcf8582；旧结果均保留。

当前使用原始 PhysX 199 轨迹／160-20-19 split、原八个 TRAIN 案例、完整 10.681B 模型，
不使用新提交中的运动学栅格替代数据。单 H200 291647/server31/step0 已在 tmux 保留。
23 项 CPU 回归及当前代码的官方完整 30 层数值检查通过；优化变更的实际效果尚待实验。
首个新 endpoint 为 overfit_resampled_noise_20260911，32 次噪声／时间重采样更新后自动
评估并渲染所有八例。此 endpoint 不等于目标完成，失败后继续依据证据分析和 overfit，
不自动启动 formal 或 physics。
详见[代码对比与完成条件](../experiments/demo_following/paper_zero_wam_v1/bugfix_audit_20260911/CODE_COMPARISON.json)。

9 月 11 日 12:49 初始八例／八噪声评估已完成：匹配条件的 video/action/IFP loss 为
`0.1872653 / 1.4499187 / 0.6574702`。teacher-action 隔离及生成视频到动作的连接通过，
正确 demo 的损失优势未通过。这是**更新前基线**，不能称为本轮 overfit 结果。
完整包级 CPU 回归 54 项通过；后续独立动作诊断另修复了 JSON 配置的 tuple 恢复并通过
往返测试，未修改正在运行的训练代码、模型或配置。
12:57 已在同一计算 shell 的当前前台任务之后排入
`run_paper_zero_wam_bugfix_endpoint_held.sh`：终点完成后导出全部 64 张冻结 VAE 对照帧；
先保存逐关节和时序动作回读；若采样动作未优于零／均值或时序基线，再做同一完整
checkpoint 的真实未来条件诊断。读取使用已有的仅导入重试入口，不重试训练或推理。
该诊断尚未开始，日志前缀为 `logs/held_291647_bugfix_endpoint`，不要重复排队。
13:52 在 server31 核实已完成 8 次真实更新，warmup LR 从 `1.25e-6` 升至 `1e-5`。
八次更新均有限且 GPU/CPU AdamW 主参数回读一致；前两次检查的 64 个 cross-attention
模块均有非零梯度。八步均触发既定梯度裁剪：第 7 次范数 `82.8513`，第 8 次回落至
`7.4131`，尚无持续发散证据。平均每步约 `446.6 s`，主要耗时仍为 CPU 梯度搬运／累加。
第 8 次单次 video/action/IFP loss 为 `0.0297365 / 1.2501502 / 0.3724995`；由于每步
噪声／时间不同，不把它们与平均初始探针直接相除来宣称通过。终点评估和新画面尚未完成。

旧修复版八例的逐关节回读再次证实动作失败：动作 MSE `0.648–0.752`，逐关节常数均值
oracle 的 MSE 仅 `0.0464–0.0904`；相邻帧动作变化误差为零变化基线的 `89.5–352.6` 倍。
这不是新实验结果；详见本地 `bugfix_audit_20260911/HISTORICAL_ACTION_RECONSTRUCTION.json`。

已验证的对比边界：旧梯度裁剪实际触发 24/32 步，而非“从未触发”；真实 attention mask
阻断了 action-target 到视频 token 的所有路径，因此替换这些 token 不能解释视频条件修复。
学习率／初始化／IFP detach 是待实测的优化或方法变更，不能直接等同于已经定位的 bug。
论文的 IFP 目标是监督主干表示，detach 会改变这条训练路径；保留为明确标注的变体诊断，
不宣称忠实复现。[Zero-WAM §3.3](https://arxiv.org/html/2608.26103v2#S3.SS3)

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
