# TODO 15: Online Whole-Hand Patch Tactile for Sudden-Mass Adaptation

## A. 固定合同

- [x] 唯一 backend 固定为 IsaacLab/PhysX；Newton simulator、RGB、demo、ICM 和软体
  训练退出当前队列。
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
- [ ] PS seed `151016`：从 official Tracker 启动，严格训练到 `model_2999.pt` 后停止。
- [ ] 审查 PS-151016 checkpoint finiteness、handoff、mass readback、80-frame window、
  action continuity 和同步 54-patch video。
- [ ] 显式运行 `151016->152016` 五质量 × 20 profiles frozen evaluation。
- [ ] 运行 exact three-seed Z/P/PS paired comparison；不得增加单一分支 profiles。

当前结果：

- Z holds=`59,59,52,1,0`，drops=`0,0,2,58,59`；
- P holds=`59,59,49,0,0`，drops=`0,0,8,59,59`，尚未证明 tactile benefit；
- PS 前两 seed 合计 holds=`39,39,26,0,0`，drops=`0,0,10,39,39`；第三 seed 未完成，
  不得提前给出最终结论。

## D. Endpoint 评审清单

每个新 endpoint 必须逐项完成后才进入下一 seed：

- [ ] checkpoint iteration=2999，model/optimizer tensors finite；
- [ ] live Refiner 在同一 PhysX episode 完成抬箱和 no-reset handoff；
- [ ] requested/readback mass 与 inertia event 一致；
- [ ] nonnominal profiles 在 event 前 10 帧有 bilateral patch contact；
- [ ] 450-frame trace 完整覆盖 handoff、event 和 80-frame outcome；
- [ ] applied action 与 saved policy action 对齐，无 gross handoff/jump discontinuity；
- [ ] H.264 完整解码，画面包含 G1/CarryBox、左右各 27 patches 和 evaluator-only event
  标注；
- [ ] camera video 只描述自己的 camera rollout，不冒充 camera-free formal trace。

PS-151015 已完成以上清单；本节下一次用于 PS-151016。

## E. 高摩擦 6x/10x

- [ ] 仅在 Z/P/PS 正式比较完成后启动。
- [ ] 分别测试 static/dynamic friction
  `0.5/0.5、1.0/1.0、1.5/1.5、2.0/2.0` × `6x/10x`。
- [ ] 每条保存 PhysX material readback、mass readback、post-jump height/contact 和 outcome。
- [ ] 不把高摩擦结果混入原 Z/P/PS comparison。
- [ ] 若 `mu<=2` 未让 6x 持稳，测试 stronger grip/lower posture；必要时用相同 serious
  SUGAR policy 做固定条件 overfit。
- [ ] 至少获得一个可重复的 6x 完整持箱结果，并生成同步 G1/CarryBox/54-patch H.264。

## F. 交付

- [x] experiments 只保留正式 endpoint、冻结评估、关键 sensing 和人眼证据；中间
  checkpoint、旧 runtime、失败/重复实验已移入根 `legacy/`。
- [x] 旧 Plan-13、旧 bundle renderer 和 Newton simulator adapter 已从活动代码移出。
- [ ] 完成 PS-151016 与 friction 结果后更新 README、Plan、TODO、AGENTS。
- [ ] 最终只 commit/push 源码、测试和文档；不提交 checkpoint、trace、视频或日志。
