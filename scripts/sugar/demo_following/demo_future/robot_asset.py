"""Load an unchanged official URDF-converter output without rerunning the importer."""
from pathlib import Path


def use_converted_robot_usd(robot_cfg, usd_path: Path):
    """Preserve all common spawn, rigid-body, collision and articulation settings."""
    import isaaclab.sim as sim_utils
    usd_path=usd_path.resolve()
    if not usd_path.is_file():raise FileNotFoundError(usd_path)
    original=robot_cfg.spawn
    replacement=sim_utils.UsdFileCfg(usd_path=str(usd_path))
    copied=[]
    for key in vars(replacement):
        if key not in ('func','usd_path') and hasattr(original,key):
            setattr(replacement,key,getattr(original,key));copied.append(key)
    robot_cfg.spawn=replacement
    return dict(usd_path=str(usd_path),original_spawn_type=type(original).__name__,
                loaded_spawn_type=type(replacement).__name__,copied_common_fields=copied,
                geometry_or_physics_asset_edited=False,
                scope='Reuse the exact official converter USD; runtime importer is omitted, complete robot/actuators and common spawn settings retained. Physical equivalence still needs actual readback.')
