# 本地实验与依赖

当前科学分支已完成负结果，详见[当前状态](../DOCS/current_status.md)。

- [paper_zero_wam_v1](demo_following/paper_zero_wam_v1/)：当前完整证据，包括原 overfit、
  step-700、修复前／后诊断、缓存、checkpoint 和全部可视化；均未覆盖。
- [zero_wam_official_v1](demo_following/zero_wam_official_v1/)：官方 Wan 源码／权重及共享
  SUGAR 示范、动作、RGB、manifest 等当前代码依赖。
- `demo_following/host_official_v1`、`bpp_official_v1`：冻结证据与接口／数据依赖，
  不是活动训练队列。
- 其余保留实验及 `runtime_assets/`、`sugar_reproduction/`、触觉相关目录：
  现有代码仍引用的模型、数据、官方资产或冻结对照；保留原路径以免破坏复现。
- 133 个过时 demo-following 目录及旧 runtime/allocation 记录已归档至
  [统一 legacy](../legacy/repository_cleanup_20260910/ARCHIVE.md)。

整理不触碰当前 H200 会话、最新结果、训练 tensor 或模型参数。本目录所有实验产物为
本地忽略文件，不提交、不推送。完整迁移映射见[归档说明](../DOCS/archive.md)。
