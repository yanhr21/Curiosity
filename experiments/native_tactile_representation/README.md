# Native tactile representation: active result

The only active Plan-12 result is `whole_hand_carrybox_v3/`. One collector and
one renderer are reused without changing the tactile representation for:

- a successful bilateral grasp and sustained CarryBox lift;
- a bilateral grasp followed by physical release and drop; and
- a failed closure with left-thumb/index contact, no right-hand contact, and
  no meaningful lift.

Every case reads the same 54 physical IsaacLab `VisuoTactileSensor` objects:
27 patches on each hand, raw `20 x 25` signed local-Z and signed-XY taxels per
patch, and the two center-palm R15 RGB/depth streams. Sensor arrays are never
edited to create success or failure.

Open `whole_hand_carrybox_v3/README.md` for the three physical outcomes and
`whole_hand_carrybox_v3/REPRODUCE.md` for the single end-to-end reproduction
entry point. Replaced experiments, completed launch logs, and old audit files
live only in the single
`/public/home/yanhongru/Curiosity_archive` tree. No hash or SHA-256 workflow is
part of the active visualization task. The retained Slurm allocation remains
alive for user review and requested corrections.
