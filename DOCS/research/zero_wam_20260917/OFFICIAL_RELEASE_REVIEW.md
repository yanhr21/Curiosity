# Zero-WAM 官方发布核查

核查日期：2026-09-17。此次仅下载与研究，predictor 训练及准备中的控制/形状修复均暂停。

代码已完整克隆至 `third_party/Zero-WAM-official`；官方 remote 为 https://github.com/robbyant-research/Zero-WAM ，commit `08e2c4ae41e2b63573a299825cebe6753481407c`，提交日期 2026-09-15，提交标题 Initial public release。工作树未修改。旧本地 paper reconstruction 和全部失败结果保留，不能冒充这次发布的官方实现。

|官方模型|固定 revision|文件数|Hub 精确总字节|
|---|---|---:|---:|
|robbyant-research/zero-wam-pretrain|7040c4195df216c900334ef62d5fdcf05c0601aa|23|37,064,741,866|
|robbyant-research/zero-wam-posttrain-robotwin|07ee865f175d9474a5653e9147698390e47b6143|30|37,064,741,211|

两套完整目录包括 transformer、text encoder、VAE、tokenizer、配置与索引，合计74,129,483,077字节（约74.13 GB十进制）。权重目标目录为本研究目录的 `checkpoints/`。**仅当 `DOWNLOAD_COMPLETE.json` 存在且complete=true时，才表示全部下载和校验完成。** 每文件大小与官方Hub记录核对，LFS文件核官方SHA256，普通Git文件核blob SHA1，并另记本地SHA256。

原始 API 清单保存在两份 `*.remote.json`。下载器先尝试普通HTTP，实测通道很慢；保留初始日志，已切换官方HF Xet下载，下载状态以运行日志和verified文件为准，不以空目录或配置文件到达计为权重完成。没有安装训练依赖或启动模型。

预训练目录与后训练transformer文件正在并行下载。`download_supervisor.py`仅维护下载、复用经过校验的相同文件，并在下载进程退出但未完成时最多恢复3次；不会启动训练、推理或GPU任务。动态状态见 `DOWNLOAD_STATUS.json`，全量成功标志见 `DOWNLOAD_COMPLETE.json`。两套模型有约14.20 GB相同文件，可用硬链接复用，不改动官方文件内容。

## 读到的真实接口

- 官方 `INSTALL.md` 测试环境是 Python3.10 / PyTorch2.9.0 / CUDA12.6；RoboTwin单独环境，固定revision `2eeec322`。本轮不更改现有Newton/Isaac环境。
- `wan_va/configs/va_robotwin_cfg.py`：三相机机器人观测，原始动作容器30维，RoboTwin使用其中16通道；ICL示范单独的视频分辨率/帧率。`eval_policy_client_openpi.py` 将双手相对初始末端位置/四元数和gripper解释为可执行末端动作。不能把它当作G1关节/力矩控制器。
- `wan_va/modules/icl_model.py`：示范内部双向读取；机器人流遵守原时间分块掩码；目标video流可以读取ICL，action不直接读取ICL，而经联合video/action表征获得任务信息。新增object tokens必须重新审注意力因果关系，不能随意拼接后声称等价。
- `wan_va/configs/zerowam_train_config.py` / `wan_va/mcp.py`：官方启用四个未来chunk辅助模块，固定层[3,11,19,29]、权重[.5,.25,.15,.1]、stride2；未来越界位置有valid mask。论文称IFP，源码使用MCP命名；这里MCP不是agent的Model Context Protocol。
- 官方单任务推理server支持1GPU，训练脚本默认8GPU；这不是我们已测得的显存/时延数据。
- 官方发行训练数据只是论文训练数据的一部分，按公开数据重训不会严格得到发布检查点；不能承诺仅下载数据后复现全部论文数字。
- 官方RoboTwin评估对两个任务提供调整后的成功判据（包括stamp seal容差1→3cm）。未来复现应同时记录原始判据与作者判据，不能把不同门槛下的数字混用。

## 直接证据

- [官方代码与下载说明](https://github.com/robbyant-research/Zero-WAM)
- [论文](https://arxiv.org/abs/2608.26103)
- [预训练模型](https://huggingface.co/robbyant-research/zero-wam-pretrain)
- [RoboTwin模型](https://huggingface.co/robbyant-research/zero-wam-posttrain-robotwin)
- [官方安装与评估说明](https://github.com/robbyant-research/Zero-WAM/blob/main/INSTALL.md)

后续研究方案只以本次官方代码为起点；目前不申请GPU、不跑推理或训练。
