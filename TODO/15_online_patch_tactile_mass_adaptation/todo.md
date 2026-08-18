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
- [x] PS seed `151016`：从 official Tracker 启动，严格训练到 `model_2999.pt` 后停止。
- [x] 审查 PS-151016 checkpoint finiteness、handoff、mass readback、80-frame window、
  action continuity 和同步 54-patch video。
- [x] 显式运行 `151016->152016` 五质量 × 20 profiles frozen evaluation。
- [x] 运行 exact three-seed Z/P/PS paired comparison；每分支恰好 300 profiles。

当前结果：

- Z holds=`59,59,52,1,0`，drops=`0,0,2,58,59`；
- P holds=`59,59,49,0,0`，drops=`0,0,8,59,59`，尚未证明 tactile benefit；
- PS 三 seed holds=`59,58,33,0,0`，drops=`0,1,22,59,59`；3x PS-P hold 差值
  `-0.2712`，95% CI=`[-0.4655,-0.0667]`，当前 PS 显著劣于 P。

## D. Endpoint 评审清单

每个新 endpoint 必须逐项完成后才进入下一 seed：

- [x] checkpoint iteration=2999，model/optimizer tensors finite；
- [x] live Refiner 在同一 PhysX episode 完成抬箱和 no-reset handoff；
- [x] requested/readback mass 与 inertia event 一致；
- [x] nonnominal profiles 在 event 前 10 帧有 bilateral patch contact；
- [x] 450-frame trace 完整覆盖 handoff、event 和 80-frame outcome；
- [x] applied action 与 saved policy action 对齐，无 gross handoff/jump discontinuity；
- [x] H.264 完整解码，画面包含 G1/CarryBox、左右各 27 patches 和 evaluator-only event
  标注；
- [x] camera video 只描述自己的 camera rollout，不冒充 camera-free formal trace。

PS-151015 与 PS-151016 均完成以上清单。

## E. 高摩擦 6x/10x

- [x] 仅在 Z/P/PS 正式比较完成后启动。
- [x] 分别测试 static/dynamic friction
  `0.5/0.5、1.0/1.0、1.5/1.5、2.0/2.0` × `6x/10x`。
- [x] 每条保存 PhysX material readback、mass readback、post-jump height/contact 和 outcome。
- [x] 不把高摩擦结果混入原 Z/P/PS comparison。
- [x] `6x, mu=1.5` 已满足 hold，因此 stronger-grip/lower-posture/overfit 条件分支未触发。
- [x] 为 `6x, mu=1.5` 成功条件生成并审查同步 G1/CarryBox/54-patch H.264；该
  camera-enabled rollout 高度损失 `0.02552 m`、hold=true、drop=false。

Camera-free 6x height loss 按 `mu=0.5/1.0/1.5/2.0` 为
`0.5589/0.5429/0.02636/0.06596 m`；只有 `mu=1.5` hold。10x 四个条件全部 drop。

## F. 交付

- [x] experiments 只保留正式 endpoint、冻结评估、关键 sensing 和人眼证据；中间
  checkpoint、旧 runtime、失败/重复实验已移入根 `legacy/`。
- [x] 旧 Plan-13、旧 bundle renderer 和 Newton simulator adapter 已从活动代码移出。
- [x] 完成 friction 结果后更新 README、Plan、TODO、AGENTS。
- [x] 最终 commit/push 只包含源码、测试和文档；checkpoint、trace、视频和日志保持
  在 ignored `experiments/`。
