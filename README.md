# Curiosity

本仓库研究两个问题：示范反馈能否真正改变并约束 humanoid 操作策略，以及在线整手触觉
能否在视觉不变、接触 binary 不变但物体动力学改变时提供额外适应能力。仿真统一使用
IsaacLab/PhysX；Newton 只作为 asset 来源，不作为执行后端。

## 当前结论

### Demo following：信号被使用，语义遵循尚未成立

当前有效实验固定同一个 CarryBox45 official Refiner teacher、相同初始化、物理、seeds、
优化器、reward weights 和 64-update budget，只改变 internal reward 读取的 selected demo：

- `correct`：CarryBox45；
- `unrelated`：KickBox21，任务仍然是 CarryBox。

teacher-only zero-residual gate 在 20/20 profiles 中实现双手接触并抬升至少 5 cm，平均
最大抬升 0.7128 m。冻结评估得到 correct `16/20` success、unrelated `18/20` success，
两臂各 `2/20` physical root-height fall。checkpoint 和 residual action 均发生可测差异，
因此 selected-demo reward 确实进入优化并改变策略；但单个训练 seed 下 correct 没有优于
unrelated，所以不能声称策略理解或遵循了 demo 语义。

当前 internal reward predictor 是约 11.9M 参数的 causal future-mismatch predictor，预测
body、box position、box 6D rotation 和 box velocity 的未来偏差。held-out normalized MAE
为 0.1873，constant baseline 为 0.3609，mean Spearman 为 0.7718；permuted/zero demo 会
显著退化。它读取了指定 demo，但用同一个 predictor 给训练和结果打分仍有循环验证风险。

官方 MimicKit TinyMDM 目前只是 generic motion prior。两个 official single-clip prior 能
完美识别各自训练 clip，但 CarryBox96/KickBox22 的独立同任务扩展没有通过。因此没有把
任意 Transformer hidden state 冒充 SMP latent，也没有授权 selected-demo SMP policy
integration。

### 在线整手触觉：实现保留，收益结论冻结

每只手有 27 个物理解剖 patch：掌心 `4 x 3`，拇指、食指、中指、无名指和小指各有
proximal/middle/distal 三段。policy unit 是 patch；TacSL/R15 taxel 只作为每个 patch 内部
的物理采样与审计后端。每个在线 patch record 包含 contact、normal load、mean pressure、
signed local-XY shear 和 friction utilization；PS 额外使用 causal batch-stateful slip
callable。

2026-08-20 审计发现并修复了接触负奖励、dead contact sensor、缺失 hold reward、
normal/shear 混合、摩擦不一致、slip reset 丢失、训练/评测 motion 不匹配、评测关闭终止
以及统计多重比较问题。旧 Z/P/PS 数字全部撤回。

修正后的 tactile-only diagnostic 在 model1100 的 20-profile 评估为 `14/20` physical hold、
`6/20` strict success、0 drop、0 physical fall 和 10 reference deviations。它证明在线触觉
可以进入 actor 并改变参数，但没有证明触觉改善重量突变后的物理行为。Plan 15 因此冻结，
不得继续盲训或把历史结果称为触觉增益。

独立 frozen-Refiner friction sweep 表明 6x、`mu=1.5` 可满足 5 cm hold；10x 在
`mu=0.5/1.0/1.5/2.0` 下全部掉落。这是物理可行性结论，不是触觉策略收益。

## 明确贡献

本仓库新增并验证的贡献是：

1. 将官方 SUGAR CarryBox 的 Refiner、Tracker、Generator 输入输出和 state-based policy
   边界整理为可复现基线；
2. 在 IsaacLab/PhysX 中为完整 G1 建立双手 54 个在线物理解剖 TacSL patch，并提供同钟
   world/双手 27-patch 可视化；
3. 建立不读取物体速度、质量、jump flag 或未来帧的 causal `PatchSlipDetector.update`；
4. 建立 teacher handoff 后在线改变 mass/inertia 的 matched Z/P/PS 协议及本体感受泄漏
   审计；
5. 实现 serious causal future-mismatch internal reward predictor，并通过 held-out demo
   identity 检查；
6. 建立 fixed-teacher、只改变 selected-demo reward 的因果实验，排除 teacher replacement
   混杂；
7. 对 official single-clip TinyMDM 做 exact-identity 与 independent semantic-extension
   分离测试，得到“记住 clip 但尚未形成可靠语义空间”的负结果。

官方 SUGAR、IsaacLab TacSL 和 MimicKit TinyMDM 本身不是本仓库的原创方法；本仓库的
贡献是忠实接入、实验协议、在线传感扩展、因果隔离与失效审计。

## 最短入口

完整环境、资产、命令、输出合同和结果核验见
[可复现性与证据记录](DOCS/reproducibility.md)。常用入口如下。

```bash
cd /public/home/yanhongru/Curiosity
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python

# 无仿真：检查当前 same-teacher 配置和下一条训练命令
$PYTHON_BIN scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design same_teacher_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --dry-run

# GPU compute node：复核现有冻结评估和完整视频
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 same_teacher_reward_only

# GPU compute node：复现完整 G1 CarryBox 与双手 27-patch 在线触觉视频
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox
```

当前关键视频：

- [correct CarryBox demo 与实际行为](experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161581/videos_update0064/01_correct_demo_and_actual_behavior.mp4)；
- [unrelated KickBox demo 与实际行为](experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161581/videos_update0064/02_unrelated_kickbox_demo_and_actual_behavior.mp4)；
- [6x、mu=1.5 的 G1/CarryBox 与双手 27-patch](experiments/online_patch_tactile_mass_adaptation/visualizations/official_refiner_mu1p5_6x_friction_hold_single_env/official_refiner_mu1p5_6x_world_bilateral27.mp4)。

## 活动目录

```text
README.md                         当前结论、贡献和最短入口
DOCS/reproducibility.md           单一完整复现与证据记录
PLAN/README.md                    当前 demo-following 决策与下一实验
PLAN/15_.../plan.md               冻结的触觉/质量突变协议
TODO/README.md                    当前执行队列
TODO/15_.../todo.md               冻结的 Plan 15 历史清单
scripts/sugar/demo_following/     当前 same-teacher 训练、评估和视频入口
scripts/sugar/native_tactile/     在线整手触觉、泄漏和物理可行性入口
experiments/                      本地最小证据包，不提交
legacy/                           失败、混杂、重复和过期内容，不提交
```

`experiments/`、checkpoint、trace、dataset、视频和 runtime log 均为本地证据，不进入
Git。当前实验目录索引见 [experiments README](experiments/README.md)。

## 下一步

当前不自动启动新训练。下一项有效工作应先使用现有 rollout 建立独立于 predictor 的行为
级 demo-adherence 指标，包括接近方向、双手接触拓扑、脚部踢击、绕箱路径、箱子位移和
抬升。随后才用至少三个 matched training seeds 重复 64-update same-teacher 实验。只有
这些结果稳定后，才考虑以相同 schedule 降低 teacher authority 或重新设计语义 predictor。
