# Policy And Photometric Reference Audit

Date: 2026-07-01

Scope: lightweight source and README audit only. No dependency installation,
model loading, training, evaluation, dataset conversion, simulation, rendering,
or official sanity run was executed on the login node.

## Reactive Diffusion Policy

- Official repository: `https://github.com/xiaoxiaoxh/reactive_diffusion_policy`
- Local path: `external/reactive_diffusion_policy`
- Local commit: `824c5e8de1fd1811106907a04b5f0186e0138c0b`
- Role: secondary serious visual-tactile policy reference for later baseline
  and ablation design.

Observed source paths:

- `reactive_diffusion_policy/policy/diffusion_unet_image_policy.py`
- `reactive_diffusion_policy/policy/latent_diffusion_unet_image_policy.py`
- `reactive_diffusion_policy/dataset/real_image_tactile_dataset.py`
- `reactive_diffusion_policy/scripts/extract_gelsight_marker_motion.py`
- `reactive_diffusion_policy/scripts/extract_mctac_marker_motion.py`
- `data/PCA_Transform_GelSight/`
- `train_dp.sh`
- `train_rdp.sh`

Observed design:

- Slow-fast visual-tactile diffusion policy for contact-rich manipulation.
- Real robot data format includes camera images, tactile images, marker
  offsets, tactile marker embeddings, robot force, TCP pose/velocity/wrench,
  actions, and timestamps.
- README links official HuggingFace datasets and checkpoints.
- Reported control/data frequencies include 24 FPS tactile recording and
  12/24 FPS policy settings.

Project use:

- Future baseline/reference for serious visual-tactile policy training.
- Useful for tactile marker embedding, PCA marker representation, slow-fast
  policy structure, and real rollout comparison requirements.

Must not claim:

- not a Newton-native grasping infant checkpoint;
- not a Gate 00D/00E/00F completion path;
- not current curiosity success;
- not usable in this project until official environment/checkpoint sanity and
  data schema compatibility are proven.

## ImplicitRDP

- Official repository: `https://github.com/Chen-Wendi/ImplicitRDP`
- Local path: `external/ImplicitRDP`
- Local commit: `4c90646df17787e31c88838106c4a0323ddefb4a`
- Role: secondary newer visual-force diffusion policy reference, linked from
  the RDP README.

Observed source paths:

- `ImplicitRDP/policy/diffusion_transformer_image_policy.py`
- `ImplicitRDP/policy/diffusion_unet_image_policy.py`
- `ImplicitRDP/model/force/rnn.py`
- `ImplicitRDP/model/vision/transformer_obs_encoder.py`
- `ImplicitRDP/config/task/real_flip_image_wrench_implicitrdp_10fps.yaml`
- `train_implicitrdp.sh`
- `train_dpt.sh`
- `train_rdp.sh`

Observed design:

- End-to-end visual-force diffusion policy with structural slow-fast learning.
- Real data format is force/wrench-centered rather than dense tactile-marker
  centered.
- README links official HuggingFace datasets and checkpoints.
- Requires ROS2 Humble and real robot stack; not a simulator-only drop-in.

Project use:

- Future visual-force policy baseline/reference after Phase 00 base tactile
  schema and Gate 00F are resolved.
- Useful as an ablation reference for visual+force vs visual+tactile when dense
  tactile semantics are still being validated.

Must not claim:

- not a dense tactile semantic validator;
- not a Newton-native base grasp checkpoint;
- not current curiosity training evidence.

## Tactile Diffusion

- Official repository: `https://github.com/carolinahiguera/Tactile-Diffusion`
- Local path: `external/Tactile-Diffusion`
- Local commit: `16868fb96d19d93dc5837600c26b48415632e4f6`
- Role: secondary photometric tactile-image generation reference.

Observed source paths:

- `tacto_diffusion/model/diffusion_model.py`
- `tacto_diffusion/train_ycb.py`
- `tacto_diffusion/test_ycb.py`
- `tacto_diffusion/train_braille.py`
- `ycb_slide_sim/touch_simulator.py`
- `ycb_slide_sim/render/digit_render.py`
- `braille_clf/train.py`

Observed design:

- Diffusion model maps simulated contact depth toward realistic vision-based
  tactile images.
- Uses YCB-Slide / TACTO / DIGIT style data and Braille classification
  evaluation.
- Useful for photometric gap thinking, but not a mechanics simulator.

Project use:

- Future reference for candidate gel/marker photometric realism and sim-to-real
  tactile image augmentation.
- Optional comparison path after mandatory UniVTAC/TaCauchy semantic gates.

Must not claim:

- not Gate 00F replacement;
- not dense mechanics validation;
- not a grasp/lift/hold controller.

## Action Conditioned Tactile Prediction

- Official remote checked:
  `https://github.com/imanlab/action_conditioned_tactile_prediction`
- Remote HEAD:
  `085d2ab82d2e0574f39a359dd2c445b8f7f7a3b3`
- Local status: clone attempt failed with
  `fetch-pack: unexpected disconnect while reading sideband packet`.

Current project status:

- remote-available but not source-audited locally;
- record as acquisition blocker, not as local evidence.

## Integration Decision

Reactive Diffusion Policy, ImplicitRDP, and Tactile Diffusion expand the future
comparison set, but they do not change the active gates:

1. Gate 00D/00E still require Newton/Taccel reference-video-aligned dense
   tactile/base evidence.
2. Gate 00F still requires UniVTAC/TaCauchy official semantic validation or an
   accepted faithful blocker.
3. Gate 00G may later use RDP/ImplicitRDP as serious policy baselines and
   Tactile Diffusion as a photometric tactile reference.
4. No checkpoint or official model from these repositories may be claimed as
   working in Curiosity until official download, environment, and schema sanity
   are run inside an approved workflow.
