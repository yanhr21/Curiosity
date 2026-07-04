# Reference Clone Inventory: Digit / MuJoCo / mc_rtc Loco-Manipulation

Date: 2026-07-02.

This inventory records reference code and model-data cloned after checking the
Digit box loco-manipulation and BHR10 hybrid RL-WBC papers. These sources are
for reading and design reference only. They have not been built, imported,
executed, rendered, trained, or evaluated on the login node.

## Strict Status

- I did not find an official public code release for `Sim-to-Real Learning for
  Humanoid Box Loco-Manipulation`.
- I did not find a public complete code/data release for the BHR10
  `Robust Visuomotor Control for Humanoid Loco-Manipulation Using Hybrid RL`
  paper.
- The cloned repositories below are nearby references: Digit/MuJoCo model
  assets, OSU/Agility MuJoCo lineage code, and the mc_rtc / mc_mujoco /
  LocomanipController stack related to the BHR10 paper's control lineage.
- No real robot dataset, real video dataset, SUGAR processed data, checkpoint,
  or T-Rex-like asset was downloaded.
- `external/mujoco_menagerie` shallow clone was attempted but checkout did not
  complete in a reasonable time. The incomplete directory was removed. If
  needed later, use sparse checkout for a specific robot model.

## Cloned Reference Sources

### Digit-V3_SDF_Model

- Local path: `external/Digit-V3_SDF_Model`
- Commit: `0471d5e`
- URL: https://github.com/yu-fz/Digit-V3_SDF_Model
- Type: robot model data.
- Why it matters: the README states the model parameters are derived from an
  official Digit V3 MJCF MuJoCo model from Agility Robotics, with permission
  to repackage and redistribute. This is useful for understanding Digit
  geometry, collision meshes, and model conversion issues.
- Limitation: this is an SDF/Drake-oriented model package, not the Digit box
  loco-manipulation training environment.

### Digit-MuJoCo-ROS2

- Local path: `external/Digit-MuJoCo-ROS2`
- Commit: `1a263e0`
- URL: https://github.com/MindSpaceInc/Digit-MuJoCo-ROS2
- Type: Digit MuJoCo XML + ROS2 simulation wrapper.
- Why it matters: the README states it contains `description/model/xml/digit.xml`
  for Digit simulation in MuJoCo and a ROS2 message handler for joint state,
  IMU, odom, and actuator command topics.
- Limitation: this is not the Digit box loco-manipulation paper's official
  environment; it is a useful model/wrapper reference.

### cassie-mujoco-sim

- Local path: `external/cassie-mujoco-sim`
- Commit: `188d3d7`
- URL: https://github.com/osudrl/cassie-mujoco-sim
- Type: OSU/Agility Cassie MuJoCo simulation library.
- Why it matters: this is from the same OSU Dynamic Robotics lineage as many
  Digit/Cassie sim-to-real RL works. It is useful for seeing how Agility-style
  MuJoCo wrappers, UDP interfaces, perturbation hooks, contact-force
  visualization, and Python/C interfaces were structured.
- Limitation: Cassie is not Digit and this is not a box-carrying environment.

### mc_rtc

- Local path: `external/mc_rtc`
- Commit: `b5b39c7`
- URL: https://github.com/jrl-umi3218/mc_rtc
- Type: whole-body controller framework.
- Why it matters: BHR10 explicitly uses mc_rtc as the WBC framework that
  converts task-space targets into joint commands. It is central for
  understanding the FSM + WBC layer.
- Limitation: it is a general control framework, not a learning environment or
  BHR10 paper release.

### mc_mujoco

- Local path: `external/mc_mujoco`
- Commit: `2fc073d`
- URL: https://github.com/rohanpsingh/mc_mujoco
- Type: MuJoCo interface for mc_rtc.
- Why it matters: the BHR10 paper builds on prior FSM + DRL + WBC work whose
  training was in MuJoCo and controlled through mc_rtc-like FSM controllers.
  `mc_mujoco` shows the concrete interface between MuJoCo and mc_rtc, plus
  object loading and force interaction hooks.
- Limitation: it is not BHR10 load-carrying code. It is infrastructure.

### LocomanipController

- Local path: `external/LocomanipController`
- Commit: `ee25576`
- URL: https://github.com/isri-aist/LocomanipController
- Type: open humanoid loco-manipulation controller.
- Why it matters: its README states it is a mc_rtc controller using
  Choreonoid/JVRC1 that accepts loco-manipulation object trajectory/velocity
  commands and considers manipulation forces in balance control. This is the
  closest open controller reference for the kind of WBC force-aware
  loco-manipulation the BHR10 paper cites as related infrastructure.
- Limitation: it is a model-based controller for JVRC1/Choreonoid, not
  BHR10, not RL training code, and not unknown-load video-conditioned carrying.

### BaselineWalkingController

- Local path: `external/BaselineWalkingController`
- Commit: `9cba7e2`
- URL: https://github.com/isri-aist/BaselineWalkingController
- Type: mc_rtc humanoid walking controller.
- Why it matters: LocomanipController is an extension of this walking
  controller. Its baseline walking architecture helps separate locomotion/WBC
  concerns from object-carrying policy learning.
- Limitation: walking baseline only; not box carrying.

## Not Cloned Or Deferred

### Digit Box Loco-Manipulation Official Code

Status: not found.

The paper itself says policies are trained in MuJoCo, but I did not find an
official repository for the actual five-skill Digit box loco-manipulation
training system.

Paper:

- https://arxiv.org/html/2310.03191v1

### BHR10 Hybrid RL-WBC Complete Code

Status: not found.

The paper clearly describes mc_rtc, WBC, FSM, D4PG, depth input, and simulated
training/evaluation, but I did not find a full release of the BHR10 simulation,
robot model, task FSMs, policy training code, or data.

Paper:

- https://www.mdpi.com/2313-7673/10/7/469

### mujoco_menagerie

Status: deferred.

The repository is useful as a general MuJoCo model library, but a full shallow
clone checkout was slow and not required for the immediate question. Use sparse
checkout later for a specific robot directory.

Repository:

- https://github.com/google-deepmind/mujoco_menagerie

## Practical Use For This Project

These references do not change the primary execution path:

1. Use Isaac Lab-Arena first for an executable G1 box loco-manipulation task.
2. Use the Digit/MuJoCo references to understand what a MuJoCo-based Digit
   reproduction would require.
3. Use mc_rtc / mc_mujoco / LocomanipController to understand FSM + WBC +
   task-space target architectures.
4. Do not claim any cloned reference is a complete solution to unknown-load,
   morphology-aware, non-retargeting video-guided carrying.

