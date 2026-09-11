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
