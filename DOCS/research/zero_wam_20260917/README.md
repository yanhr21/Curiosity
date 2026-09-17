# Zero-WAM、触觉物体信念与示范跟随研究

2026-09-17研究快照；2026-09-18归档到Git。用户要求暂停predictor训练，本轮仅做官方下载、源码与文献研究、三个subagent独立审查。

- [完整方案与架构图](RESEARCH_PLAN.md)
- [官方发布与动作接口核查](OFFICIAL_RELEASE_REVIEW.md)
- [触觉、形状与物性研究](TACTILE_OBJECT_BELIEF_REVIEW.md)
- [SMP、示范与SUGAR接口审查](SMP_AND_DEMO_REVIEW.md)
- [Agent技能、反馈与成功判定审查](AGENT_REASONING_REVIEW.md)

官方代码：https://github.com/robbyant-research/Zero-WAM ，固定commit `08e2c4ae41e2b63573a299825cebe6753481407c`。本地checkout为 `third_party/Zero-WAM-official`，不将嵌套Git仓库和模型权重提交到本仓库。

研究交付时，两套官方模型共74,129,483,077字节，仍在下载，尚未宣称完成。动态状态和校验记录保存在本地 `experiments/zero_wam_release_20260917/`；此文档快照不代表最新下载进度。报告中指向experiments的实验原始证据链接仅在保有本地数据的工作区可用。

本目录完整保留研究结论和官方来源；方案不代表已实现、已训练或验证通过。
