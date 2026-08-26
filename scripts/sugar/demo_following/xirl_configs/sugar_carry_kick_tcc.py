"""Official XIRL/TCC configuration adapter for clean SUGAR reference videos."""

from __future__ import annotations

import os

from configs.xmagical.pretraining.tcc import get_config as get_official_tcc_config


def get_config():
    """Keep the released XIRL architecture/loss and point it at SUGAR data."""
    config = get_official_tcc_config()
    config.root_dir = os.environ["SUGAR_XIRL_RUN_ROOT"]
    config.data.root = os.environ["SUGAR_XIRL_DATA_ROOT"]
    config.seed = int(os.environ.get("SUGAR_XIRL_SEED", "271402"))
    config.data.pretraining_video_sampler = "same_class"
    config.data.pretrain_action_class = ("CarryBox", "KickBox")
    config.data.downstream_action_class = ("CarryBox", "KickBox")
    config.data.max_vids_per_class = int(
        os.environ.get("SUGAR_XIRL_MAX_VIDEOS_PER_CLASS", "-1")
    )
    config.optim.train_max_iters = int(
        os.environ.get("SUGAR_XIRL_TRAIN_ITERS", "4000")
    )
    return config
