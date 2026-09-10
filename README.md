# Curiosity

研究 humanoid 示范跟随（demo following）与在线触觉适应。仿真后端为 IsaacLab/PhysX；
当前代码保留完整模型与官方组件，不以简化模型冒充论文实现。

## 当前结果

2026-09-10：修复版 Zero-WAM 思路的本地论文重建完成 32 步 overfit、终点评估和全部
8 个 TRAIN 案例渲染，**仍未通过，科学分支已停止**。视频生成存在明显块状伪影和结构损坏；
冻结 VAE 重建对照正常。此前 step-700 全量评估也未通过，不追加训练或物理 rollout。

- [最新结果、原因与证据](DOCS/current_status.md)
- [代码入口](scripts/sugar/demo_following/README.md)
- [实验与保留依赖](experiments/README.md)
- [归档说明与恢复路径](DOCS/archive.md)
- [执行约束](AGENTS.md)

## 工作区结构

- `scripts/`、`tests/`：实现、适配器与测试。
- `SUGAR/`、`IsaacLab/`、`MimicKit/`、`third_party/`、`external/`：官方组件与依赖。
- `sugar_newton/`：保留的历史代码／资产依赖，不是当前仿真后端。
- `DOCS/`：当前状态和归档入口，不再维护重复 PLAN／TODO 队列。
- `experiments/`：最新科学证据及仍被代码引用的冻结数据、权重和历史依赖。
- `legacy/`：统一的本地历史归档；旧计划、TODO、项目页和失效实验运行记录均在此。

2026-09-10 整理仅移动和重建文档入口，没有删除成果、改动模型／训练目标、启动训练或释放 GPU。
实验、权重、日志及 `legacy/` 均为本地忽略文件，不随 Git 提交。
