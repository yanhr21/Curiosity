# Curiosity

This workspace is for the Newton tactile-curiosity project:

> Newton-native closed-loop curiosity adaptation for contact-rich object affordance learning.

The active goal is to build a Newton-based embodied manipulation environment where a robot starts from a basic grasping prior and improves through closed-loop curiosity: prediction errors over object motion, contact, force/contact proxies, tactile-marker evidence, and safety signals should help it adapt to changing object mass, friction, fill level, compliance, and slip.

T-Rex remains a reference model and future bridge, but it is no longer the immediate gate for Newton-native curiosity progress. Strict T-Rex schema promotion is allowed only when synchronized bimanual state/action, accepted cameras, calibrated F6, and dense tactile deformation streams are genuinely available.

## Directory layout

- `IDEA/idea.md`: active high-level research idea and design rationale.
- `PLAN/`: active staged roadmap and milestones, split by phase.
- `TODO/`: active staged task tracking, split by phase.
- `PLAN/legacy/` and `TODO/legacy/`: historical pre-pivot planning records.
- `legacy/`: archived pre-pivot experiment and log records.
- `docs/`: design constraints and implementation notes.
- `external/`: official repositories only, such as Newton, T-Rex, Isaac Lab, and Taccel if needed.
- `src/newton_tactile_curiosity/`: future project code, split by responsibility.
- `data/`: local data staging; large generated datasets should not be committed.
- `checkpoints/`: local checkpoint staging; official checkpoints should be documented with source URLs.
- `experiments/`: experiment outputs, visuals, configs, and reports.
- `runs/` and `logs/`: runtime and job logs.

## Non-negotiable constraints

- Do not implement a toy T-Rex, toy VQ-VAE, toy Transformer, or toy world model and present it as faithful progress.
- Use official T-Rex code, released checkpoints, and embedded tactile VQ-VAE where T-Rex work is explicitly in scope.
- Do not rename Newton/Taccel provenance fields into official T-Rex schema fields unless the strict data contract is genuinely satisfied.
- The login node is for lightweight file/repo/job operations only.
- Real training must run inside `tmux` or an equivalent persistent session and use at least one GPU for at least one hour unless explicitly labeled as a smoke test.
- Do not use one-shot `sbatch` experiments unless the user explicitly approves a different workflow.
- Maintain GPU utilization above 30% during real training; if it stays below 30% for more than 3 hours, fix or release the allocation.
- Run and record a sanity check before each experiment attempt.
