from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PaperZeroWAMConfig:
    """One explicit configuration shared by overfit and formal training.

    The dimensions are the values reported in Zero-WAM v2.  The overfit run is
    allowed to shorten the data schedule, but is not allowed to shrink this
    architecture.
    """

    manifest: str = "experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/ICL_MANIFEST.jsonl"
    wan_source: str = (
        "experiments/demo_following/zero_wam_official_v1/wan_base_runtime_v1/"
        "Wan2.2-42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
    )
    wan_checkpoint: str = (
        "experiments/demo_following/zero_wam_official_v1/wan_base_runtime_v1/"
        "Wan2.2-TI2V-5B"
    )
    latent_cache: str = "experiments/demo_following/paper_zero_wam_v1/latents"
    output_root: str = "experiments/demo_following/paper_zero_wam_v1"

    hidden_dim: int = 3072
    ffn_dim: int = 14336
    num_heads: int = 24
    head_dim: int = 128
    num_layers: int = 30
    visual_channels: int = 48
    patch_size: tuple[int, int, int] = (1, 2, 2)
    action_dim: int = 29
    action_hidden_dim: int = 3072
    action_per_rgb_interval: int = 5
    rgb_intervals_per_latent: int = 4
    latent_transitions_per_trajectory: int = 35

    human_rope_height_offset: int = 32
    ifp_heads: int = 4
    ifp_stride: int = 2
    ifp_weights: tuple[float, float, float, float] = (0.5, 0.25, 0.15, 0.15)
    ifp_fusion_layers: tuple[int, int, int, int] = (7, 14, 21, 29)
    lambda_video: float = 1.0
    lambda_action: float = 1.0
    lambda_ifp: float = 1.0
    loss_reduction: str = "global_target_element_mean"

    epochs: int = 15
    world_size: int = 8
    samples_per_rank: int = 1
    gradient_accumulation_steps: int = 1
    optimizer_steps_per_epoch: int = 280
    optimizer_steps: int = 4200
    peak_learning_rate: float = 1.0e-4
    minimum_learning_rate: float = 1.0e-6
    warmup_steps: int = 200
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    gradient_clip_norm: float = 2.0
    schedule_seed: int = 291700
    noise_seed: int = 291701
    prompt_dropout_probability: float = 0.1
    precision: str = "bfloat16"
    expected_parameter_count: int = 10_658_724_829
    repaired_conditioning: bool = False
    neutral_text_cache: str = ""
    hash_checks: bool = False

    # --- Optimization repair (2026-09-10) -------------------------------
    # The 32-step fixed-noise overfit ended with a video ratio of 1.666: the
    # pretrained Wan trunk got worse, not better.  Four independent defects
    # are corrected here.  All are optimization-side; the architecture,
    # factorization, IFP geometry and losses are unchanged.
    #
    # 1. overfit ran at a constant peak LR with no warmup.  With beta2=0.95
    #    the second-moment estimate is meaningless for the first ~20 steps,
    #    so every tensor moved by ~lr per step.  Over 32 coherent steps that
    #    is ~18% of the weight scale of a d=3072 Wan linear.
    # 2. weight decay hit norms, biases and the pretrained AdaLN modulation
    #    table, i.e. the conditioning path itself.
    # 3. the four training-only IFP heads back-propagate into the trunk at
    #    layers 7/14/21/29 with total weight 1.05 -- equal to the video term
    #    -- through a randomly initialized fusion MLP.
    # 4. the deployed video sampler was conditioned on all-zero action
    #    tokens at t=1, a distribution training never produced.
    overfit_learning_rate: float = 1.0e-5
    overfit_warmup_steps: int = 8
    decay_only_matrix_parameters: bool = True
    ifp_trunk_gradient: bool = False
    ifp_warmup_steps: int = 0
    inactive_action_token_noise: bool = True
    # A zero-initialized action output projection cannot reach the required
    # output scale inside a 32-step diagnostic (short by ~169x at lr=1e-5),
    # so the head returns ~zero velocity and the sampler returns its initial
    # noise.  Scaled init lets the short run measure learning instead of growth.
    zero_init_action_head: bool = False

    # Paper v2 reports the fixed ICL chunk size and CFG scales.  It does not
    # repeat the lower-level flow scheduler values; use the released
    # LingBot-VA causal-VA implementation that the paper explicitly follows.
    inference_chunk_size: int = 2
    video_guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    flow_integrator: str = "fixed_step_euler"
    video_inference_steps: int = 25
    action_inference_steps: int = 50
    video_snr_shift: float = 5.0
    action_snr_shift: float = 1.0

    def resolved(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.hidden_dim != 3072 or self.action_hidden_dim != 3072:
            raise ValueError("Zero-WAM v2 requires 3072-wide video and action experts")
        if self.num_layers != 30 or self.num_heads != 24 or self.head_dim != 128:
            raise ValueError("Wan-2.2-TI2V-5B depth/head geometry changed")
        if self.ifp_heads != 4 or self.ifp_stride != 2:
            raise ValueError("Zero-WAM IFP geometry changed")
        if self.ifp_weights != (0.5, 0.25, 0.15, 0.15):
            raise ValueError("Zero-WAM IFP loss weights changed")
        if self.loss_reduction != "global_target_element_mean":
            raise ValueError("variable-length losses must use global target-element means")
        if self.world_size * self.samples_per_rank != 8:
            raise ValueError("formal global packed-sample batch must remain 8")
        if self.epochs * self.optimizer_steps_per_epoch != self.optimizer_steps:
            raise ValueError("epoch/optimizer-step arithmetic is inconsistent")
        if not 0.0 < self.minimum_learning_rate < self.peak_learning_rate:
            raise ValueError("cosine floor must be positive and below the peak learning rate")
        if self.warmup_steps <= 0 or self.warmup_steps >= self.optimizer_steps:
            raise ValueError("formal warmup must be positive and shorter than the run")
        if not 0.0 < self.overfit_learning_rate <= self.peak_learning_rate:
            raise ValueError("overfit learning rate must be positive and not exceed the peak")
        if not 0 <= self.overfit_warmup_steps < 32:
            raise ValueError("overfit warmup must fit inside the 32-step budget")
        if self.ifp_warmup_steps < 0:
            raise ValueError("IFP warmup must be nonnegative")
        if not 0.0 < self.adam_beta1 < self.adam_beta2 < 1.0:
            raise ValueError("AdamW beta geometry is invalid")
        if self.adam_epsilon <= 0.0:
            raise ValueError("AdamW epsilon must be positive")
        expected = 10_658_724_829 + (22_026_240 if self.repaired_conditioning else 0)
        if self.expected_parameter_count != expected:
            raise ValueError("full paper reconstruction parameter contract changed")
        if self.repaired_conditioning and not self.neutral_text_cache:
            raise ValueError("repaired conditioning requires official empty-text UMT5 context")
        if self.hash_checks is not False:
            raise ValueError("paper Zero-WAM execution forbids file hash/SHA admission")
        if self.action_per_rgb_interval * self.rgb_intervals_per_latent != 20:
            raise ValueError("SUGAR causal clock must retain twenty actions per Wan latent")
        if self.inference_chunk_size != 2:
            raise ValueError("Zero-WAM v2 fixes inference chunk size to two")
        if self.video_guidance_scale != 5.0 or self.action_guidance_scale != 1.0:
            raise ValueError("Zero-WAM v2 ICL/action CFG scales changed")
        if self.flow_integrator != "fixed_step_euler":
            raise ValueError("frozen local flow integrator changed")
        if self.video_inference_steps != 25 or self.action_inference_steps != 50:
            raise ValueError("released causal-VA flow integration step counts changed")
        if self.video_snr_shift != 5.0 or self.action_snr_shift != 1.0:
            raise ValueError("released causal-VA SNR shifts changed")


def repaired_overfit_config() -> PaperZeroWAMConfig:
    """User-requested correction; legacy checkpoints/config remain readable."""
    return replace(PaperZeroWAMConfig(), repaired_conditioning=True,
        expected_parameter_count=10_680_751_069,
        latent_cache="experiments/demo_following/paper_zero_wam_v1/latents_prompt61_repaired",
        neutral_text_cache="experiments/demo_following/paper_zero_wam_v1/latents_prompt61_repaired/EMPTY_TEXT.pt")
