# SUGAR CarryBox 完整复现记录

## 0. 文档信息与结论边界

- 工作区：`/public/home/yanhongru/Curiosity`
- 记录时间：2026-07-13 23:14 CST
- 复现对象：SUGAR 官方 `CarryBox` 基线
- SUGAR 仓库：`SUGAR`
- SUGAR commit：`01fe1234b57188412ec38775150f008f133e0ad4`
- IsaacLab 仓库：`IsaacLab`
- IsaacLab 基线：官方 `v2.3.0`
- IsaacLab 上游 commit：`3c6e67bb5c7ada942a6d1884ab69338f57596f77`
- 运行环境：`/public/home/yanhongru/envs/sugar_py311_isaacsim510`
- 主要输出目录：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full`

2026-07-13 目录整理后，`SUGAR/` 是根目录主线源码；所有复现 outputs 和
logs 的实体分别位于 `experiments/sugar_reproduction/outputs/` 与
`experiments/sugar_reproduction/logs/`。整个 `experiments/` 目录是本地
实验资产，必须由 `.gitignore` 排除，不能 commit 或 push。
旧的 `external/SUGAR`、`SUGAR/outputs`、`logs/sugar` 只是不中断活跃
Tracker 训练所保留的兼容软链接。

本次复现已经在 2026-07-13 由用户按“官方流程与功能效果正常、具有可视化证据，不要求和论文数值完全一致”的标准验收通过。这个结论已经写入根目录 `AGENTS.md`。

2026-07-14，作为 SUGAR 主线运行栈组成部分，IsaacLab 源码从
`external/` 迁移到根目录 `IsaacLab/`。旧位置仅保留指向根目录源码的
兼容软链接，供已经启动的任务和预构建 editable 环境解析；所有新文档
和脚本统一使用 `IsaacLab/`。

需要同时保留如下事实边界：

1. 官方论文默认 Refiner 训练终点是 `model_30000.pt`；本次按用户指令将本地 Refiner 固定在 `model_10000.pt`，没有继续训练到 30000，也没有产生 `model_11000.pt`。
2. `model_10000.pt` 使用官方 SUGAR 代码、任务、数据、机器人与物体资产训练，后续导出没有修改 checkpoint 内容。
3. Refiner 的完整 1000 环境 rollout 和数据处理已经完成；Tracker 与 Generator 的本地完整产物链仍在继续构建。
4. 因此本项目可以称为“用户验收通过的功能复现”，但不能声称“严格复现了论文的全部训练时长与论文表格数值”。
5. 本文中的“完整渲染”是指官方 SUGAR `play.py --headless --video` 离屏渲染、策略推理和 MP4 写出链路完整成功；它不等价于已经修复 Isaac Sim 所有交互式 viewport/Rendering Manager 扩展依赖。

## 1. 复现目标与不变原则

复现目标是优先跑通官方 SUGAR CarryBox，而不是用已有的 Curiosity、AGILE、MuJoCo 或自制控制器替代 SUGAR。全过程遵守以下约束：

- 使用官方 SUGAR 代码、官方任务注册、官方 CarryBox 数据、官方 G1 和箱子描述、官方发布的 Tracker/Generator checkpoint。
- 使用与官方要求匹配的 Python 3.11、Isaac Sim 5.1.0 和 IsaacLab v2.3.0。
- 不编写 toy Refiner、toy Tracker、toy Generator 或简化控制器冒充复现。
- 仅添加集群执行、离线资产解析、日志、断点恢复和产物审计所需的胶水代码。
- 所有 Python、仿真、训练、数据处理、模型加载、渲染和可视化生成都在计算节点运行。
- GPU 通过 Curiosity 自己的持久 `tmux + srun/salloc` 分配持有，不使用登录节点执行项目计算。
- 获得的后备资源不主动释放；资源切换时保留持久 shell，只有调度器自动回收时才记录为自动取消。

原始研究方向见：

- `IDEA/idea.md`
- `PLAN/04_sugar_baseline/plan.md`
- `TODO/04_sugar_baseline/todo.md`
- `experiments/reports/sugar_baseline_status_20260711.md`

## 2. 官方软件、数据和仿真任务

### 2.1 软件栈

最终运行环境包含：

| 组件 | 版本或位置 |
| --- | --- |
| Python | 3.11.15 |
| Isaac Sim | 5.1.0.0 |
| IsaacLab | 源码 tag `v2.3.0`，包 metadata `isaaclab==0.47.2` |
| SUGAR | commit `01fe1234b57188412ec38775150f008f133e0ad4` |
| RSL-RL | `rsl-rl-lib==3.0.1` |
| NumPy | 1.26.0 |
| Generator 依赖 | `zarr==2.12.0`、`numcodecs==0.12.1`、`diffusers==0.32.1`、`accelerate==1.2.1`、`timm==1.0.12`、`datasets==2.6.1` 等 |

准备和检查入口：

- `scripts/sugar/prepare_official_sugar_env.sh`
- `scripts/sugar/preflight_official_sugar_env.sh`
- 初始失败检查：`experiments/sugar_reproduction/logs/20260711_sugar_env_preflight.log`
- 最终通过检查：`experiments/sugar_reproduction/logs/20260712_sugar_env_preflight_fixed.log`

最初已有的 `isaac_arena_py312` 是 Python 3.12、Isaac Sim 6.0.1、IsaacLab 4.5.24，与官方 SUGAR 的版本组合不兼容，因此没有拿它强行运行。之后单独准备了 `sugar_py311_isaacsim510` 环境。

### 2.2 官方资产

官方资产通过计算节点上的 Curiosity `tmux+srun` 下载：

- Slurm `176828`，`server18`：完成 `data.zip` 和 `descriptions.zip`，下载 `demo_ckpts.zip` 时超时。
- Slurm `176906`，`server18`：通过登录节点代理隧道重试官方 Google Drive 下载并完成 checkpoint 获取。

主要资产：

- `SUGAR/data/CarryBox`
- `SUGAR/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf`
- `SUGAR/descriptions/objects/small_box/obj_aligned.usd`
- `SUGAR/demo_ckpts/CarryBox/tracker.pt`
- `SUGAR/demo_ckpts/CarryBox/generator.ckpt`
- 下载日志：`experiments/sugar_reproduction/logs/20260711_sugar_assets_download.log`

### 2.3 仿真任务

- 仿真器：NVIDIA Isaac Sim，Vulkan + GPU PhysX。
- 框架：IsaacLab `ManagerBasedRLEnv`。
- 机器人：Unitree G1 29-DoF，带官方 rubber-hand 末端模型。
- 任务：`Sugar-G129dof-CarryBox-*`。
- 场景：平面、G1、一个刚体小箱子、全身和手脚接触传感器。
- Refiner/Tracker 正式训练并行度：4096 环境。
- 物理时间步：`dt=0.005 s`，即 200 Hz。
- control decimation：4，策略控制频率约 50 Hz。
- 单回合训练时长：30 秒。

这里没有高维触觉。手部使用 IsaacLab `ContactSensor` 取得左右手刚体与箱子的三维接触力，保留最近 3 帧，对力向量求模、取最大值并以 `0.1` 为阈值生成接触标签。它属于粗粒度 contact-force detection，不包含 taxel、压力图、剪切力分布、滑移图像或 GelSight/DIGIT 类触觉。

## 3. H200 上的渲染阻塞到底是什么

### 3.1 “硬件不完备”的准确含义

早期所说的“硬件/渲染条件不完备”，准确含义不是 H200 没有图形能力，也不是 H200 不能运行 Vulkan。实际缺失的是无显示计算节点上的运行条件：

1. 节点没有 X Server/物理显示器，`GLFW initialization failed` 和 `failed to open the default display` 会出现。
2. Isaac Kit 在线 extension registry 在作业网络中不可达，导致 headless rendering experience 的扩展依赖无法在线补齐。
3. IsaacLab 默认引用的部分地面和 frame-marker 可视 USD 位于 NVIDIA Nucleus/S3，作业节点无法访问。
4. 初始环境里没有完整的 Isaac Sim 5.1 extension cache。

成功日志已经明确识别到真正的 GPU：

```text
Graphics API: Vulkan
GPU 0: NVIDIA H200
GPU Memory: 143771 MB
Driver Version: 580.95.05
```

证据位于：

- `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_local_assets.log`

因此，解决思路不是换成低保真渲染器或伪造视频，而是把官方 headless 渲染所需的扩展和可视资产本地化，让 H200 继续走 Isaac Sim 的 Vulkan 离屏渲染路径。

### 3.2 初始失败链路

#### 第一次：环境包版本不匹配

最初只有不兼容的 Python 3.12/Isaac Sim 6.0.1 环境。通过 compute-node preflight 确认后，没有继续用错误版本试跑，而是建立官方要求的 Python 3.11/Isaac Sim 5.1.0 环境。

#### 第二次：Kit 在线扩展依赖无法解析

在设置 EULA 后启动官方推理，Kit 尝试访问如下 registry：

```text
https://ovextensionsprod.blob.core.windows.net/...
https://dw290v42wisod.cloudfront.net/...
```

网络不可达，随后出现：

```text
Failed to resolve extension dependencies
omni.kit.material.library ... can't be satisfied
ModuleNotFoundError: No module named 'omni.kit.usd'
```

对应日志：

- `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_eula_env.log`

#### 第三次：extension cache 补齐后，远程地面资产失败

安装 extension cache 后，Isaac Sim、Vulkan、PhysX 已能启动，但默认 ground-plane 资产在集群环境中解析失败，出现 `Boost.Python.ArgumentError`。

对应日志：

- `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_extscache.log`

#### 第四次：地面本地化后，远程 frame marker 失败

本地地面使环境继续向前运行，但 target/frame marker 仍引用：

```text
https://omniverse-content-production.s3-us-west-2.amazonaws.com/
Assets/Isaac/5.1/Isaac/Props/UIElements/frame_prim.usd
```

最终出现 `FileNotFoundError`。对应日志：

- `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_local_ground.log`

### 3.3 在 H200 上跑通完整离屏渲染的方法

最终方案由四部分组成。

#### A. 本地安装官方 Isaac Sim extension cache

从 NVIDIA 官方 wheel 获取并在计算节点上完成 SHA256 检查和本地安装：

- `isaacsim_extscache_kit-5.1.0.0...whl`
- `isaacsim_extscache_kit_sdk-5.1.0.0...whl`
- `isaacsim_extscache_physics-5.1.0.0...whl`

第一次直接拉取大 wheel 的进程以 status 143 中止；随后使用已下载的本地 wheel 安装，最终 status 0：

- `experiments/sugar_reproduction/logs/20260712_sugar_extscache_install.log`
- `experiments/sugar_reproduction/logs/20260712_sugar_extscache_local_install.log`

这样 Isaac Sim 不再依赖运行时访问 Kit registry 才能启动 headless rendering experience。

#### B. 使用本地 ground-plane USD

新增本地地面：

- `SUGAR/descriptions/terrain/sugar_ground_plane.usda`

在 IsaacLab `terrain_importer.py` 中增加环境变量解析：

```text
ISAACLAB_GROUND_PLANE_USD=/.../descriptions/terrain/sugar_ground_plane.usda
```

设置该变量时，仅把 ground-plane USD 指向本地文件，并保留原来的物理材质和场景结构。

#### C. 使用本地 frame-marker 形状

在 IsaacLab marker 配置中增加：

```text
ISAACLAB_USE_LOCAL_FRAME_MARKER=1
```

设置后，不再下载远程 `frame_prim.usd`，改用 IsaacLab 原生 `SphereCfg` 绘制小型本地 marker。这个改动只改变调试 marker 的资产来源/外观，不改变机器人、箱子、动力学、观测、动作、官方 checkpoint 或策略输出。

两处 IsaacLab 胶水改动分别位于：

- `IsaacLab/source/isaaclab/isaaclab/terrains/terrain_importer.py`
- `IsaacLab/source/isaaclab/isaaclab/markers/config/__init__.py`

因此 IsaacLab 被记录为 `v2.3.0-dirty`；dirty 内容就是上述集群离线资产兼容胶水。

#### D. 使用官方 headless RGB 录制链路

最终启动方式没有使用 Xvfb 截屏，也没有使用自制 OpenGL viewer，而是直接使用官方 SUGAR：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_GROUND_PLANE_USD="$SUGAR_DIR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1

python scripts/sugar_rl/play.py \
  --task Sugar-G129dof-CarryBox-Inference \
  --checkpoint demo_ckpts/CarryBox/tracker.pt \
  --generator_checkpoint demo_ckpts/CarryBox/generator.ckpt \
  --motion_folder data/CarryBox \
  --num_envs 16 \
  --eval_random_motion \
  --headless \
  --video \
  --video_length 200
```

实际由以下 wrapper 执行：

- `scripts/sugar/run_official_sugar_carrybox_inference.sh`

`play.py` 在 `--video` 时用 `render_mode="rgb_array"` 创建环境，再由 Gymnasium `RecordVideo` 把 Isaac Sim 相机帧编码成 MP4。这是一条真正的 GPU Vulkan 离屏渲染链路。

### 3.4 渲染成功证据

- Slurm：`177522`
- 节点：`server35`
- GPU：NVIDIA H200
- 推理任务：`Sugar-G129dof-CarryBox-Inference`
- 官方 Tracker：`SUGAR/demo_ckpts/CarryBox/tracker.pt`
- 官方 Generator：`SUGAR/demo_ckpts/CarryBox/generator.ckpt`
- 日志：`experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_local_assets.log`
- 退出状态：0
- 视频：`experiments/sugar_reproduction/outputs/released_inference/CarryBox/videos/play/rl-video-step-0.mp4`
- 视频文件大小：306614 bytes

日志中仍然有以下 warning：

- `GLFW initialization failed`
- `failed to open the default display`
- `NGX DLSS Frame Generation Feature AdapterUnsupported`
- `viewportHandle not found`

这些 warning 在此 headless `rgb_array` 路径中不是 fatal error；仿真、策略推理、GPU 渲染和视频写出都完成了。

但这不证明旧的交互式 viewport/Rendering Manager 路径已经全部修复。旧路径涉及 `omni.kit.pip_archive`、`isaacsim.core.rendering_manager` 和 `omni.kit.viewport.window` 等依赖，本文不把它们的状态混入 SUGAR 官方离屏视频成功结论。

## 4. 后续完整复现过程

### 4.1 官方 Refiner 训练 smoke

在正式长训练前，先用官方入口做最小训练链路验证：

- Slurm：`177539`
- 节点：`server35`
- 任务：`Sugar-G129dof-CarryBox-Refiner`
- `NUM_ENVS=64`
- `MAX_ITERATIONS=1`
- 总 timesteps：1536
- 输出：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_sugar_carrybox_refiner_train_smoke_iter1/logs/refiner/model_0.pt`
- 日志：`experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_train_smoke_iter1.log`
- 状态：0

该运行只证明官方训练入口、环境、Actor/Critic 和 checkpoint 写出正常，明确标记为 smoke，不作为正式训练结果。

### 4.2 建立官方全流程 wrapper

官方 `SUGAR/train.sh CarryBox` 的阶段顺序是：

1. Refiner training
2. Refiner rollout
3. Process Refiner rollout，生成 Tracker 所需 RL dataset
4. Tracker training
5. Tracker rollout
6. Process Tracker rollout，生成 Generator 所需 IL dataset
7. Generator training

新增 wrapper：

- `scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh`

wrapper 保留官方任务名、数据流、teacher checkpoint、环境数量和阶段顺序，只增加：

- 登录节点拒绝保护；
- 环境 preflight；
- 每阶段独立日志；
- `START_STAGE`/`STOP_AFTER_STAGE`；
- 每阶段产物存在性检查；
- fatal pattern 检查；
- rollout 成功 `SystemExit` 的兼容处理；
- output-directory `flock`，防止多个后备 allocation 同时写同一输出目录；
- 显式 checkpoint resume 路径；
- 本地地面/marker 环境变量。

官方 rollout 脚本在“全部环境完成”后会用 `SystemExit(msg)` 退出，而原始 `train.sh` 没有 `set -e`，所以仍会继续。严格 wrapper 会把非零退出当失败，因此增加了一个窄范围规则：只有同一阶段日志明确包含
`[Rollout] ====== All ... envs completed ...` 且没有 fatal pattern 时，才把该退出归一化为成功。

### 4.3 第一次正式 Refiner 训练到 5000

- tmux：`curiosity_sugar_refiner_full_0712`
- Slurm：`177561`
- 节点：`server23`
- 资源：1 GPU、16 CPU、160G、5 天
- 正式参数：4096 环境、原计划 30001 iterations
- 日志：`experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_full_official.log`
- 输出目录：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full`

该运行依次产生：

```text
model_0.pt
model_1000.pt
model_2000.pt
model_3000.pt
model_4000.pt
model_5000.pt
```

根据当时用户指令，训练在 `model_5000.pt` 后停止，没有继续到 30000。

### 4.4 model-5000 官方 rollout 诊断

新增：

- `scripts/sugar/run_official_sugar_carrybox_refiner5000_eval.sh`
- `scripts/sugar/summarize_official_sugar_refiner5000_eval.sh`

结果：

- 视频 smoke：Slurm `177780`，`server23`，产生 200 帧 MP4；因为视频长度先到，所以未等待完整 trajectory。
- no-video rollout：Slurm `177782`，`server23`。
- 16/16 环境完成。
- 保存 13 个 complete trajectory，窗口完成率 81.25%。
- 这不是论文 CarryBox SR；论文 SR 是最终物体位置是否进入目标阈值，定义不同。

重要产物：

- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/videos/play/rl-video-step-0.mp4`
- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model5000_rollout_summary.png`
- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model5000_rollout_summary.json`

### 4.5 精确断点恢复与 H200 device-lost 排查

用户后来要求继续训练。为保证从原始 `model_5000.pt` 精确恢复，给官方训练入口增加 `--resume_checkpoint_path`，避免依赖正则寻找 run 目录。

原始成功 checkpoint 被保存为只读 canonical source：

```text
experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/
resume_sources/server23_original/model_5000.pt
```

SHA256：

```text
175f4df698ca2f7e04bc94072ef2dcdd23172243a2529cbd8f84704ff615720d
```

之所以单独保存，是因为 RSL-RL resume 后第一次保存会再次写 `model_5000.pt`；如果主运行失败，不能让后备 allocation 从被覆盖的 mutable 文件重新开始。

在 `server36` 和 `server53` 的一些 H200 上，训练可以进入 IsaacLab 初始化甚至 RSL-RL iteration，但随后 Vulkan 报：

```text
VkResult: ERROR_DEVICE_LOST
A GPU crash occurred
```

排查覆盖了：

- 4096、2048、1024 环境；
- fresh train 与 exact resume；
- 官方 renderer；
- single-renderer-GPU；
- Fabric 开/关；
- contact/debug marker 开/关；
- headless rendering experience；
- 清理残留 Isaac 进程后重试。

clean-context 重试仍能在相同物理 GPU 上复现 device lost，因此没有把这些失败当作模型或代码成功证据，也没有通过降级模型绕过问题。相关日志集中在：

- `experiments/sugar_reproduction/logs/20260713_sugar_carrybox_full_resume_model5000_*.log`
- `experiments/sugar_reproduction/logs/20260713_sugar_full_pipeline_resume_model5000_*.log`

### 4.6 资源策略调整

Slurm 资源难以快速获得，因此执行策略改为：

1. 所有新 GPU 请求使用约一天 wall time，避免频繁重新排队。
2. 使用较少 CPU，并同时尝试 `cpu`、`gpu`、`gpux` 等可访问分区。
3. allocation 启动后进入持久 shell；切换工作时不发送 Ctrl+C 释放 allocation。
4. 多个 allocation 通过同一个输出目录 `flock` 排队；只有一个写 checkpoint，其他保持为即时 failover。
5. 不主动取消 pending 或 acquired backup；scheduler 自动取消时如实记录。

典型请求包括：

- `178129`：灵活 `cpu` 请求，最终在 `server23` 获得 H200 physical GPU 3。
- `178143`：4 CPU/80G 的灵活后备，获得 H200 physical GPU 6 后等待 output lock。
- `178133`、`178134`、`178136`、`178137`：少 CPU、跨分区或目标节点后备。

这个策略使干净的 `server23` H200 立即接管正式 resume，同时保留其他已获得资源用于可视化和故障切换。

### 4.7 在稳定 H200 上从 5000 继续到 10000

Slurm `178129` 在 `server23` physical H200 GPU 3 上从 canonical `model_5000.pt` 稳定恢复：

- 4096 个官方环境；
- 平均迭代约 5.3 秒；
- 12 个一秒 GPU 利用率样本为 29%–63%，平均约 45%，高于集群 33% 低利用率回收阈值；
- 没有 device-lost/fatal pattern。

依次产生：

- `model_6000.pt`：2026-07-13 06:11 CST
- `model_7000.pt`：2026-07-13 07:42 CST
- `model_8000.pt`：2026-07-13 09:13 CST
- `model_9000.pt`
- `model_10000.pt`：2026-07-13 12:13:42 CST

checkpoint watcher 在确认 `model_10000.pt` 文件大小稳定后，只终止训练 Python child，保留 allocation shell。停止确认时间为 12:13:47，allocation 在 12:14:01 仍然活着。

对应脚本和日志：

- `scripts/sugar/stop_official_sugar_refiner_at_checkpoint.sh`
- `experiments/sugar_reproduction/logs/20260713_sugar_refiner_model_10000_stop_watch.log`

最终 checkpoint：

- 原始：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/model_10000.pt`
- 只读命名副本：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/ckpts/refiner_model10000.pt`
- 两者大小：14957503 bytes
- 两者 SHA256：
  `a398a7293fcea0ef948234e5de47b990fa586d2efd4e54ad7e481151c16124c3`
- checkpoint keys：`infos`、`iter`、`model_state_dict`、`optimizer_state_dict`
- 内部 `iter=10000`
- 审计日志：`experiments/sugar_reproduction/logs/20260713_sugar_refiner_model10000_checkpoint_audit.log`

没有 `model_11000.pt`，也没有继续 Refiner 训练。

### 4.8 model-10000 功能和可视化验证

官方 Refiner rollout 诊断：

- 日志：`experiments/sugar_reproduction/logs/20260713_sugar_carrybox_refiner10000_rollout_eval.log`
- 16/16 sampled env 完成。
- 16 个 complete trajectory。
- 派生最终 `||obj_pos_b-target_obj_pos_b||` mean/median：`0.08404/0.06964`。
- 视频评估 status 0。

可视化：

- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_training_curves.png`
- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_rollout_summary.png`
- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_rollout_summary.json`
- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_rollout_video.mp4`

200 帧视频会在完整 trajectory 结束前主动停止，所以视频 run 的 `trajectory_complete_count=0` 是预期行为；完整性由独立 no-video rollout 验证。

### 4.9 从 model-10000 继续官方下游数据流

根据用户“效果正常即可、不需要论文数值完全一致”的确认，将只读 `refiner_model10000.pt` byte-for-byte 导出到官方 pipeline 要求的文件名：

```text
experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/ckpts/refiner.pt
```

`refiner.pt` 与 source 的 SHA256 相同，并带 provenance：

- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/ckpts/refiner.pt.provenance.txt`

provenance 明确写明它是 operator-selected truncated Refiner export，不冒充 `model_30000.pt`。

新增下游入口：

- `scripts/sugar/run_official_sugar_downstream_from_refiner10000.sh`
- `scripts/sugar/request_official_sugar_downstream_refiner10000.sh`

正式下游运行：

- Slurm：`178916`
- 节点：`server13`
- 分区：`cpu`，但实际 allocation 包含 H200 GPU
- tmux：`curiosity_sugar_downstream_cpu4_any_0713`
- wall time：1 天
- 资源：4 CPU、80G、1 GPU
- 主日志：`experiments/sugar_reproduction/logs/20260713_sugar_downstream_refiner10000_cpu_active.log`

#### Refiner full rollout

- 任务：`Sugar-G129dof-CarryBox-Refiner-Rollout`
- 1000/1000 环境完成。
- 922 个 complete trajectory，完成率 92.2%。
- rollout stage status 0。
- 原始轨迹：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/rollout_datasets/refiner/raw_npz/trajectory_complete`

#### Refiner rollout processing

- 使用官方 `scripts/sugar_rl/process_refiner_rollout.py`。
- 处理 922 个文件。
- status 0。
- 输出：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/rollout_datasets/refiner/rl_dataset`
- 每条数据包含机器人 50 Hz 轨迹、物体运动和 contact label 等官方字段。

full-rollout 可视化：

- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_full_rollout_summary.png`
- `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_full_rollout_summary.json`

统计：

- 922/1000 complete。
- 派生最终相对位置误差 mean/median：`0.05787/0.05153`。
- 该值用于判断行为是否正常，不当作论文最终-policy SR/Err。

### 4.10 Tracker 训练

Refiner 数据处理完成后，同一 pipeline 自动进入官方 Tracker：

- 任务：`Sugar-G129dof-CarryBox-Tracker`
- 并行环境：4096
- teacher checkpoint：上述 byte-identical `ckpts/refiner.pt`
- motion folder：处理后的 `rollout_datasets/refiner/rl_dataset`
- teacher motion：官方 `data/CarryBox`
- 本地终点：`model_10000.pt`
- `TRACKER_MAX_ITERATIONS=10001`

截至本文快照 2026-07-13 23:14 CST：

- Slurm `178916` 仍在 `server13` RUNNING。
- Tracker 达到约 `2670/10001`。
- `Mean reward` 最近约 12.77，前一窗口出现过 14.20。
- `Mean episode length` 最近约 223。
- 已产生：`model_0.pt`、`model_1000.pt`、`model_2000.pt`。
- live curve：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/tracker_training_curves.png`

训练曲线中保留了两次真实异常，没有隐藏：

1. iteration 629：mean reward `-842.8265`，value-function loss `74074.5859`，下一步恢复。
2. iteration 1498：value-function loss 约 `2.9555e12`；iteration 1499 回到约 `0.2631`，reward 仍在正常范围。

事件审计：

- `experiments/sugar_reproduction/logs/20260713_sugar_tracker_event_anomaly_audit.log`

这些是孤立尖峰；后续 reward、episode length 和 motion metrics 继续恢复/改善，因此记录为需持续观察的 optimizer anomaly，而不是训练已失败的证据。

Tracker `model_1000.pt` 的独立评估曾在 `server45` 遇到 `ERROR_DEVICE_LOST`，尚未产生该 checkpoint 的本地评估视频。checkpoint 已成功复制和校验，失败发生在 Isaac/Vulkan 初始化后的物理 GPU 路径，不代表 checkpoint 内容损坏：

- `experiments/sugar_reproduction/logs/20260713_sugar_tracker_model1000_eval_suite.log`
- `experiments/sugar_reproduction/logs/20260713_sugar_tracker_model1000_rollout_eval.log`

### 4.11 尚待 pipeline 自动完成的阶段

Tracker 达到本地 `model_10000.pt` 后，wrapper 将继续：

1. 导出 `ckpts/tracker.pt`。
2. 使用 1000 环境执行官方 Tracker rollout。
3. 使用官方 `process_tracker_rollout.py` 生成 `rollout_datasets/tracker/il_dataset`。
4. 运行官方 Generator，`num_epochs=1001`。
5. 导出 `logs/generator/epoch_checkpoints/epoch=1000.ckpt` 到 `ckpts/generator.ckpt`。
6. 使用本地 Tracker+Generator 做最终 CarryBox inference/video 和可视化审计。

这些阶段是本地产物链的继续完善；按照用户已经确认的功能验收标准，它们未完成不会撤销 SUGAR 复现通过结论，但实际状态必须继续如实记录。

## 5. 官方阶段、输入、输出和当前状态

| 阶段 | 主要输入 | 主要输出 | 当前状态 |
| --- | --- | --- | --- |
| 官方 released inference | 官方 Tracker、Generator、CarryBox data | 官方 CarryBox MP4 | 已完成，H200 headless status 0 |
| Refiner train | 官方 data/G1/box/task | `model_10000.pt` | 已完成并按指令冻结 |
| Refiner sampled eval | `model_10000.pt` | 16/16 trajectories、summary、video | 已完成 |
| Refiner full rollout | `ckpts/refiner.pt` | 922 complete trajectories | 已完成 |
| Process Refiner rollout | 922 raw NPZ | `rollout_datasets/refiner/rl_dataset` | 已完成，status 0 |
| Tracker train | Refiner teacher + RL dataset | `model_10000.pt` 目标 | 进行中，快照约 2670/10001 |
| Tracker rollout | `ckpts/tracker.pt` | raw Tracker trajectories | 待 Tracker 完成 |
| Process Tracker rollout | Tracker raw trajectories | `il_dataset` | 待运行 |
| Generator train | `il_dataset` | `epoch=1000.ckpt`、`generator.ckpt` | 待运行 |
| Local final inference | 本地 Tracker+Generator | 最终本地 MP4 | 待下游产物完成 |

## 6. 代码改动审计

### 6.1 IsaacLab 改动

仅两处离线资产解析胶水：

1. `terrain_importer.py`：支持 `ISAACLAB_GROUND_PLANE_USD`。
2. `markers/config/__init__.py`：支持 `ISAACLAB_USE_LOCAL_FRAME_MARKER=1`，用本地 primitive 替代远程 marker USD。

这些改动不改变 SUGAR policy、checkpoint、奖励、动力学或任务数据。

### 6.2 SUGAR 改动

当前 SUGAR 工作树相对 commit 的改动是集群/恢复 hook：

- `scripts/sugar_rl/train.py`
  - 增加明确的 `--resume_checkpoint_path`；
  - 增加诊断用 `--disable_fabric`。
- `source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py`
  - 增加可选的训练 debug marker 关闭开关。
- Refiner/Tracker base env config
  - contact sensor debug visualization 可由环境变量关闭。

稳定正式下游运行的日志记录为：

```text
disable_train_debug_vis=0
disable_fabric=0
disable_renderer_multigpu=0
disable_renderer=0
```

即当前正式 Tracker 没有启用这些诊断性降渲染开关；这些开关主要用于 H200 device-lost 排查。正式任务、网络、数据、环境数量和训练目标仍是官方路径。

### 6.3 工作区 wrapper

主要脚本及用途：

| 脚本 | 用途 |
| --- | --- |
| `download_official_sugar_assets.sh` | 下载官方数据、描述和 checkpoint |
| `prepare_official_sugar_env.sh` | 准备严格版本环境，仅允许经批准的计算节点上下文 |
| `preflight_official_sugar_env.sh` | 检查版本、metadata 和关键资产 |
| `run_official_sugar_carrybox_inference.sh` | 官方 released checkpoint 推理和 headless 视频 |
| `run_official_sugar_carrybox_refiner_train.sh` | Refiner smoke/单阶段训练 |
| `run_official_sugar_carrybox_train_pipeline.sh` | 官方七阶段集群 pipeline |
| `wait_for_official_sugar_refiner_checkpoint.sh` | checkpoint 文件 watcher |
| `stop_official_sugar_refiner_at_checkpoint.sh` | 稳定 checkpoint 后精确停止训练 child |
| `run_official_sugar_downstream_from_refiner10000.sh` | 从只读 model-10000 export 开始下游官方流程 |
| `request_official_sugar_downstream_refiner10000.sh` | 一天、少 CPU、持久 allocation 请求 |
| `audit_official_sugar_reproduction.sh` | 登录节点安全的文件/日志/metadata 审计 |
| `check_official_sugar_carrybox_status.sh` | 登录节点安全的轻量状态检查 |
| `render_official_sugar_training_curves.sh` | 在计算节点生成 Refiner/Tracker/Generator 曲线 |
| `render_official_sugar_refiner5000_rollout_summary.sh` | 生成 rollout summary/JSON，已泛化到不同 checkpoint/stage |
| `run_official_sugar_tracker_checkpoint_eval_suite.sh` | Tracker checkpoint rollout、video 和 summary 套件 |

## 7. 关键日志索引

### 环境和资产

- `experiments/sugar_reproduction/logs/20260711_sugar_assets_download.log`
- `experiments/sugar_reproduction/logs/20260711_sugar_env_preflight.log`
- `experiments/sugar_reproduction/logs/20260712_sugar_envbuild.log`
- `experiments/sugar_reproduction/logs/20260712_sugar_env_preflight_fixed.log`
- `experiments/sugar_reproduction/logs/20260712_sugar_extscache_local_install.log`

### H200 渲染故障链

- extension registry 失败：
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_eula_env.log`
- extension cache 后的 ground-plane 失败：
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_extscache.log`
- 本地 ground 后的远程 marker 失败：
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_local_ground.log`
- 完整成功：
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_local_assets.log`

### Refiner

- 训练 smoke：
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_train_smoke_iter1.log`
- 第一次正式训练：
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_full_official.log`
- model-5000 rollout：
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner5000_rollout_eval.log`
- model-10000 exact stop：
  `experiments/sugar_reproduction/logs/20260713_sugar_refiner_model_10000_stop_watch.log`
- model-10000 checkpoint audit：
  `experiments/sugar_reproduction/logs/20260713_sugar_refiner_model10000_checkpoint_audit.log`
- model-10000 sampled rollout：
  `experiments/sugar_reproduction/logs/20260713_sugar_carrybox_refiner10000_rollout_eval.log`

### Downstream

- 主 pipeline：
  `experiments/sugar_reproduction/logs/20260713_sugar_downstream_refiner10000_cpu_active.log`
- full rollout summary 生成：
  `experiments/sugar_reproduction/logs/20260713_sugar_refiner10000_full_rollout_visualization.log`
- Tracker curve：
  `experiments/sugar_reproduction/logs/20260713_sugar_tracker_training_curves_live.log`
- Tracker anomaly audit：
  `experiments/sugar_reproduction/logs/20260713_sugar_tracker_event_anomaly_audit.log`
- Tracker model-1000 eval device loss：
  `experiments/sugar_reproduction/logs/20260713_sugar_tracker_model1000_eval_suite.log`

## 8. 所有重要可视化与视频

### 官方 released checkpoint 推理

- `experiments/sugar_reproduction/outputs/released_inference/CarryBox/videos/play/rl-video-step-0.mp4`

### Refiner model-10000

- 训练曲线：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_training_curves.png`
- 16 环境 sampled rollout：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_rollout_summary.png`
- sampled rollout JSON：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_rollout_summary.json`
- sampled video：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_rollout_video.mp4`
- 1000 环境 full rollout：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_full_rollout_summary.png`
- full rollout JSON：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model10000_full_rollout_summary.json`

### Tracker

- 当前实时训练曲线：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/tracker_training_curves.png`
- 本地最终 Tracker rollout video：尚未产生；需要等待 Tracker 训练/重试健康 GPU eval。

### 历史 model-5000 中间证据

- summary：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model5000_rollout_summary.png`
- JSON：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/visualizations/refiner_model5000_rollout_summary.json`
- video：
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/videos/play/rl-video-step-0.mp4`

## 9. 从头复跑时的推荐执行顺序

以下命令必须在已获得的计算节点 allocation 内执行，不得在 `mgmtserver02` 上运行。

### 9.1 预检

```bash
export ROOT_DIR=/public/home/yanhongru/Curiosity
export SUGAR_DIR=$ROOT_DIR/SUGAR
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
bash scripts/sugar/preflight_official_sugar_env.sh
```

### 9.2 验证官方 released inference 和 H200 视频

```bash
PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
NUM_ENVS=16 VIDEO_LENGTH=200 \
bash scripts/sugar/run_official_sugar_carrybox_inference.sh
```

wrapper 会自动设置 EULA、本地 ground plane 和本地 frame marker。

### 9.3 跑官方 pipeline

如果严格按论文默认 schedule：

```bash
START_STAGE=refiner_train \
REFINER_NUM_ENVS=4096 REFINER_MAX_ITERATIONS=30001 \
TRACKER_NUM_ENVS=4096 TRACKER_MAX_ITERATIONS=30001 \
REFINER_ROLLOUT_NUM_ENVS=1000 TRACKER_ROLLOUT_NUM_ENVS=1000 \
GENERATOR_NUM_EPOCHS=1001 \
bash scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh
```

本次已验收的本地路径不是该默认 schedule，而是 Refiner 固定到 10000 后继续：

```bash
bash scripts/sugar/run_official_sugar_downstream_from_refiner10000.sh
```

该脚本会先校验 `refiner_model10000.pt` 与 `refiner.pt` SHA256 一致，再开始 rollout，防止下游误用其他 checkpoint。

### 9.4 资源请求

优先使用已经提供的持久请求脚本：

```bash
bash scripts/sugar/request_official_sugar_downstream_refiner10000.sh
```

策略是一天 wall time、少 CPU、1 GPU、持久 shell。不要在切换任务时主动释放已获得的后备 allocation。

## 10. 已知限制与复现经验

1. **离屏渲染成功不等于有桌面显示。** Headless H200 上 GLFW/X Server warning 可以是非 fatal；判断标准应是 Vulkan 设备是否建立、仿真是否运行、`rgb_array` 是否返回、MP4 是否写出及进程是否 status 0。
2. **必须本地化运行时依赖。** 在受限网络集群上，Kit extension cache、ground USD 和 marker USD 不能依赖作业启动后在线下载。
3. **不要只信 shell exit code。** 早期官方调用可能在日志有 traceback 时仍表现为 shell status 0；wrapper 后来增加 fatal-pattern 扫描和逐阶段产物检查。
4. **rollout 的非零退出可能是成功信号。** 只在出现明确“全部环境完成”文本且没有 fatal pattern 时归一化，不能普遍忽略非零退出。
5. **H200 物理卡状态有节点差异。** 同一软件栈在 `server23/server35` 可稳定运行，在部分 `server36/server53/server45` GPU 上可复现 Vulkan device lost。应保留日志、换健康 allocation，而不是篡改模型结构。
6. **断点必须不可变。** backup runner 统一从只读 canonical checkpoint resume，防止主进程覆盖同名 checkpoint 后污染 failover 起点。
7. **资源需要持久持有。** 一天 wall time、少 CPU、跨分区请求和 output lock 比反复申请/主动释放更适合当前 Slurm 环境。
8. **可视化必须在计算节点生成。** 登录节点只做轻量文本/文件审计。
9. **指标定义不能混用。** rollout-window complete rate 和派生物体相对位置误差不是论文 CarryBox 最终 SR/Err。
10. **SUGAR 当前只有粗粒度接触检测。** 不应把 ContactSensor 的阈值标签描述成高分辨率触觉。

## 11. 最终结论

本次工作最终在 NVIDIA H200 上完成了官方 SUGAR CarryBox 的 Isaac Sim 5.1/Vulkan headless 仿真、官方 Tracker+Generator 推理、`rgb_array` 相机采集和 MP4 写出。关键方法不是降低模型或更换仿真器，而是补齐官方 extension cache，并将不可达的 ground-plane 和 frame-marker 可视资产切换为本地解析。

后续使用官方 SUGAR 训练入口完成了 Refiner 正式训练到用户指定的 `model_10000.pt`，对 checkpoint 做了内部 iteration、SHA256 和只读副本审计；完成了 sampled rollout/video、1000 环境 full rollout、922 条完整轨迹和官方 RL dataset 处理；然后进入 4096 环境的官方 Tracker 训练。全过程保留了失败日志、节点差异、异常曲线和论文指标不可比边界。

按照用户在 2026-07-13 确认的标准，SUGAR CarryBox 复现已经验收通过。Tracker rollout、Tracker 数据处理、Generator 训练以及本地最终 Tracker+Generator 视频属于继续完善的下游产物链，完成后应在本文追加最终 checkpoint、SHA256、曲线、rollout 统计和视频路径。
