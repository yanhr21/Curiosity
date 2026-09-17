"""Compare the unchanged official Engine.validate against eight saved step-0 outputs.

This is a zero-update inference qualification, not another overfit run. It uses
the original mesh loader's item/collate methods, original CUDA graph functions,
full Deformation, and original Engine.validate/Chamfer. A forward hook only
copies actual outputs. No shim, replacement forward, .cuda patch, or smoothing.

Explicit filesystem boundary: Engine.load's pretrained locator is hardcoded to
an absent vendor/pretrained directory. Setup therefore loads the real released
checkpoint by its recorded path with strict=True, then calls unchanged validate.
This does NOT qualify that upstream downloader/locator or reproduce its old
software environment. The current official checkout/backend is the comparison.

--check imports the real Engine and checks the original CPU loader and saved
inputs without constructing a model, invoking CUDA, or creating output files.
Actual execution requires the retained allocation and parent's shared lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'experiments/object_predictor_v1'
EXPECTED = tuple(f'{ident}_repeat{repeat}'
                 for ident in ('18704', '11898', '15737', '13266')
                 for repeat in (0, 1))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(source):
    """CPU only: import official Engine and build its real fixed data loader."""
    import numpy as np
    import torch
    from .qualify_active3d_training_backend import official_imports

    protocol = json.loads((source/'PROTOCOL.json').read_text())
    rows = protocol['cases']
    require(tuple(row['name'] for row in rows) == EXPECTED, 'Require all eight native cases in fixed order')
    vendor = EXPERIMENT/'vendor/Active-3D-Vision-and-Touch'
    utils, model = official_imports(vendor)
    from pterotactyl.reconstruction.vision.train import Engine
    from pterotactyl.utility import data_loaders

    require(digest(vendor/'pterotactyl/reconstruction/vision/model.py') == protocol['official_model_sha256'],
            'Official model source differs from saved run')
    require(digest(vendor/'pterotactyl/utility/utils.py') == protocol['official_utils_sha256'],
            'Official graph/loss source differs from saved run')
    checkpoint = Path(protocol['checkpoint'])
    require(checkpoint.resolve() == (EXPERIMENT/'checkpoints/active3d_official/t_p').resolve(),
            'Require the original released t_p endpoint')
    require(digest(checkpoint/'model') == protocol['checkpoint_sha256']
            and digest(checkpoint/'config.json') == protocol['checkpoint_config_sha256'],
            'Released checkpoint/config changed')
    config = json.loads((checkpoint/'config.json').read_text())
    require(config['finger'] and config['use_touch'] and not config['use_img']
            and config['num_grasps'] == 5 and config['number_points'] == 30000,
            'Require complete native five-grasp t_p configuration')
    args = SimpleNamespace(**config)
    args.eval, args.val_grasps, args.batch_size = True, 5, 8
    args.visualize, args.pretrained = False, False
    native = Path(protocol['dataset'])/'selected48/object_data'
    data_loaders.TOUCH_LOCATION = str(native/'touch_charts')+'/'
    data_loaders.POINT_CLOUD_LOCATION = str(native/'point_cloud_info')+'/'
    data_loaders.OBJ_LOCATION = str(native/'object_info')+'/'
    # Bypass only discovery across the full dataset: retain the original item,
    # selector, point reader, image convention, touch reader and collate methods.
    dataset = object.__new__(data_loaders.mesh_loader_vision)
    dataset.args = args
    dataset.object_names = [(row['object_id'], row['grasp_seed']) for row in rows]
    dataset.get_instance = dataset.get_validation_instance
    saved = []
    for index, row in enumerate(rows):
        obj, grasps = dataset.get_validation_instance(index)
        require(obj == row['object_id'] and grasps == row['grasps'], 'Original action selector differs')
        with np.load(source/'step_0000'/f"{row['name']}.npz", allow_pickle=False) as archive:
            saved.append({key: archive[key] for key in archive.files})
        observed = dataset[index]
        require(np.array_equal(observed['touch_charts'].numpy(), saved[-1]['input_touch_charts']),
                'Original loader input differs: '+row['name'])
        # The official loader shuffles labels; only their ordering changes.
        require(observed['gt_points'].shape == (30000, 3), 'Original point label dimensions changed')
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA')
    return protocol, rows, utils, model, Engine, args, dataset, saved, checkpoint


def run(source, output):
    import numpy as np
    import torch
    from .retained_execution import require_active_resource

    resource = require_active_resource()
    prepared = prepare(source)
    protocol, rows, utils, model, Engine, args, dataset, saved, checkpoint = prepared
    require(torch.cuda.is_available(), 'Allocated CUDA is required for original Engine validation')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    utils.set_seeds(protocol['seed'])
    output.mkdir(parents=True, exist_ok=False)
    # Both directories are contained in the new diagnostic directory; no vendor
    # file, original training result, or released checkpoint is modified.
    args.exp_type, args.exp_id = str(output/'original_engine'), 'equivalence'
    engine = Engine(args)
    engine.mesh_info, engine.initial_mesh = utils.load_mesh_vision(
        args, str(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/vision_charts.obj'))
    engine.initial_mesh = engine.initial_mesh.cuda()
    engine.n_vision_charts = engine.initial_mesh.shape[0]
    engine.encoder = model.Deformation(engine.mesh_info, engine.initial_mesh, args).cuda()
    released = torch.load(checkpoint/'model', map_location='cpu', weights_only=True)
    engine.encoder.load_state_dict(released, strict=True)
    require(all(torch.equal(value.cpu(), released[name]) for name, value in engine.encoder.state_dict().items()),
            'Original Engine checkpoint readback mismatch')
    engine.encoder.requires_grad_(False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,
                                         num_workers=0, collate_fn=dataset.collate)
    captured = []

    def capture(module, inputs, outputs):
        captured.append((outputs[0].detach().cpu().numpy().copy(),
                         outputs[1].detach().cpu().numpy().copy()))

    hook = engine.encoder.register_forward_hook(capture)
    try:
        # args.eval=True and visualize=False mean this original method does not
        # use writer or launch a renderer. Its actual Chamfer also executes.
        with torch.no_grad():
            engine.validate(loader, writer=None)
    finally:
        hook.remove()
    require(len(captured) == 1 and captured[0][0].shape == (8, 1949, 3),
            'Expected one full eight-input original Engine forward')
    vertices, masks = captured[0]
    with np.load(source/'GRAPH.npz', allow_pickle=False) as graph:
        graph_checks = {key: np.array_equal(value.detach().cpu().numpy(), graph[key])
                        for key, value in engine.mesh_info.items()}
        graph_checks['initial_mesh'] = np.array_equal(engine.initial_mesh.cpu().numpy(), graph['initial_mesh'])
    records = []
    for index, (row, reference) in enumerate(zip(rows, saved)):
        delta = float(np.max(np.abs(vertices[index]-reference['observed_vertices_canonical'])))
        record = dict(name=row['name'], vertices_max_abs=delta,
                      vertices_bit_exact=np.array_equal(vertices[index], reference['observed_vertices_canonical']),
                      masks_exact=np.array_equal(masks[index], reference['observed_masks']))
        record['passed'] = bool(np.isfinite(delta) and delta <= 1e-7 and record['masks_exact'])
        records.append(record)
    state_exact = all(torch.equal(value.cpu(), released[name]) for name, value in engine.encoder.state_dict().items())
    np.savez_compressed(output/'original_engine_outputs.npz', vertices=vertices, masks=masks)
    result = dict(complete=True, passed=all(graph_checks.values()) and state_exact
                  and all(row['passed'] for row in records), cases=records, graph_checks=graph_checks,
                  weights_unchanged=state_exact, model_forwards=1, input_cases=8,
                  backwards=0, optimizer_updates=0, physics_controls=0,
                  official_validation_cd_x9000=float(engine.current_loss),
                  retained_resource=resource, compared_source=str(source),
                  released_checkpoint_sha256=digest(checkpoint/'model'),
                  official_engine_source_sha256=digest(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/reconstruction/vision/train.py'),
                  path_boundary='Original Engine.validate and dataset item/collate; setup strict-loads real released checkpoint path instead of absent hardcoded pretrained directory',
                  scope='Current official backend inference equivalence, not checkpoint quality, old dependency equivalence, shape accuracy or SUGAR qualification')
    (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    if args.check:
        prepare(source)
        print(json.dumps(dict(cpu_check_passed=True, input_cases=8,
                              model_forwards=0, cuda_initialized=False, files_written=0)))
    else:
        if args.output is None:
            parser.error('--output NEW_DIRECTORY is required for original Engine inference')
        raise SystemExit(0 if run(source, args.output.resolve())['passed'] else 2)


if __name__ == '__main__':
    main()
