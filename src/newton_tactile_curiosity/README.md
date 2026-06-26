# Source Layout

This directory is reserved for future implementation after official dependency audit.

- `envs/`: Newton / Isaac Lab environment wrappers.
- `adapters/`: Newton-to-T-Rex tactile schema adapter.
- `curiosity/`: reward computation and learning-progress bookkeeping.
- `policies/`: policy wrappers around official models and RL libraries.
- `data_schemas/`: observation/action/dataset schemas.
- `evaluation/`: metrics and ablation runners.

Do not add toy T-Rex, toy VQ-VAE, toy Transformer, or toy world-model implementations here.
