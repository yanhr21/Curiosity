# Demo-following 代码入口

当前科学状态与边界见 [DOCS/current_status.md](../../../DOCS/current_status.md)。
修复版 32-step overfit 和 step-700 均已完成负结果；保留实现用于审计，不继续训练。

- [paper_zero_wam/](paper_zero_wam/)：完整宽度本地论文重建、数据适配、训练、评估和渲染。
- [config.py](paper_zero_wam/config.py)：共享配置与官方 source/checkpoint/data 路径。
- [test_repaired_conditioning.py](paper_zero_wam/test_repaired_conditioning.py)：
  修复版接口回归测试；测试通过不是整体方法成功。
- [results.py](paper_zero_wam/results.py)：生成结果页写入实验根目录 `results.md`，
  不再重建已归档的 PLAN 目录；当前结论以终点 JSON 和 DOCS 为准。
- `config/` 与其他 `audit_*`、HOST／BPP 适配代码：冻结接口及历史方法依赖。
  现有 launcher 保留供审计，不构成新的执行指令。

旧方法详细说明已完整迁入
[legacy README](../../../legacy/repository_cleanup_20260910/scripts/sugar/demo_following/README.md)；
旧 PLAN／TODO 见[归档入口](../../../DOCS/archive.md)。
没有改变模型、loss、训练预算或科学判据；没有用占位模型替换官方架构。
