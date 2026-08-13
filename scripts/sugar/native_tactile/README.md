# IsaacLab native whole-hand tactile

当前入口只运行 IsaacLab/PhysX，不启动训练。每只 G1 手固定使用 27 个物理解剖 TacSL patch：掌心 `4 x 3`，五指各 proximal/middle/distal 三段。每个 patch 使用官方 `VisuoTactileSensor` 和 `GELSIGHT_R15_CFG` 字段。

## 一条命令复现 CarryBox

在保留的 GPU Slurm shell 中运行：

```bash
bash scripts/sugar/native_tactile/run_plain_carrybox_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_plain_carrybox
```

该入口依次执行：

1. `collect_sugar_whole_hand_carrybox.py`：完整 G1 和自由 CarryBox 的同钟采集；
2. `render_sugar_whole_hand_carrybox.py`：世界画面和双手 27-patch 图；
3. FFmpeg 全文件解码检查。

输出目录中最重要的文件是 `whole_hand_trace.npz`、`summary.json` 和
`videos/plain_carrybox_world_bilateral_taxels.mp4`。

## 其他当前入口

整掌贴合刚体，可改变真实质量和摩擦：

```bash
bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_05kg 0.5

bash scripts/sugar/native_tactile/run_palm_grip_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_palm_2kg 2.0
```

官方 PickBottle Tracker：

```bash
bash scripts/sugar/native_tactile/run_pickbottle_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/reproduce_pickbottle 12 319
```

完整历史 CarryBox 五视频包仍可由 `run_complete_carrybox_visualization.sh` 生成；当前最快人眼检查优先使用上面的单视频入口。

## Trace contract

采集器直接保存官方 sensor tensor，不从物体状态或刚体 contact label 生成触觉：

- taxel 世界位置与 `xyzw` 姿态；
- penetration；
- signed local-Z normal force；
- signed local-XY shear；
- force sequence、timestamp、dt；
- optical 可用时保存独立 RGB/depth 时钟与数据。

物体位姿、PhysX 接触力、相对速度和成败标签只用于诊断，不属于可部署触觉输入。渲染器按解剖顺序显示左右手全部 patch；未接触区域必须保持空白，不能插值成接触。

## 当前边界

- 普通平面 CarryBox 当前主要由指端承载；它证明整套 sensor 在真实 G1 搬箱轨迹中工作，但不证明整掌受力。
- 掌面覆盖由单独的掌形贴合自由刚体证明，不能冒充普通箱子托底动作。
- TacSL 与 PhysX 支撑力的绝对尺度仍待标定；空间和时间对应正确不等于绝对力已经校准。
- 所有结果是模拟触觉，不是硬件 GelSight 或 sim-to-real 结果。

实验输出必须写入忽略的 `experiments/`。不要提交 trace、视频、checkpoint 或 PPT；不要为常规结果增加哈希清单。
