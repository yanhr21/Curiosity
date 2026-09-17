"""Four fixed ABC meshes through the complete released Michelangelo shape VAE.

This is a full-surface autoencoder representation qualification, NOT tactile
prediction, Utonia training, pose/mass estimation, or generalization. Ground-truth
mesh surface points and normals are deliberately the AE input at this stage.
No diffusion/CLIP is needed by the official shape encoder/decoder computational
path. Instantiate the original AlignedShapeLatentPerceiver from its released
YAML and strictly load EVERY model.shape_model.* parameter; never substitute a
decoder. Other checkpoint keys are listed explicitly, not silently ignored.

The public uniform transform is x_vae = 6.138*x_ABC, mapping the existing centered
1/3.1-span fixture into [-.99,.99]. No fitted per-object transform or output
alignment. The model predicts occupancy logits, not metric signed distances.
The unmodified official extract_geometry uses depth7/chunk10000. Its current
grid_size denominator (rather than grid_size-1) is retained and recorded.

The fixed diagnostic gates below are engineering requirements, not published
Michelangelo claims. Passing numerical checks is insufficient: actual raw mesh
renders and independent visual inspection remain mandatory before this AE can
provide a shape auxiliary target. No smoothing/welding/mesh repair is permitted.

CPU preparation: python -m scripts.sugar.object_predictor.qualify_michelangelo_ae --check
GPU execution: --output PATH, only on root's current allocation with fd9 lock.
Never launch this GPU path from the preparing/reviewing subagent.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT/'experiments/object_predictor_v1'
VENDOR = EXPERIMENT/'vendor/Michelangelo'
DATASET = EXPERIMENT/'datasets/active3d_official'
CHECKPOINT = EXPERIMENT/'checkpoints/michelangelo_official/shapevae-256.ckpt'
CHECKPOINT_BYTES = 3934164973
CHECKPOINT_SHA = '0391b81c36240e8f766fedf4265df599884193a5ef65354525074b9a00887454'
CONFIG = Path('configs/aligned_shape_latents/shapevae-256.yaml')
OBJECT_IDS = ('18704', '11898', '15737', '13266')
ABC_TO_VAE = 6.138
INPUT_POINTS = 4096
EVAL_POINTS = 16384
SEED = 1729
DISTANCE = .01  # Original ABC canonical units; physical fixture multiplier .93.
FACE_PROBES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                        [1/3, 1/3, 1/3]], dtype=np.float64)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_cases(dataset):
    selection = json.loads((dataset/'selected48/SELECTION_MANIFEST.json').read_text())
    if tuple(selection['selected']['recon_train'][:4]) != OBJECT_IDS:
        raise ValueError('The fixed first four TRAIN IDs must not change')
    cases = []
    for object_id in OBJECT_IDS:
        prefix = dataset/'selected48/object_data/object_info'/object_id
        vp, fp = Path(str(prefix)+'_verts.npy'), Path(str(prefix)+'_faces.npy')
        vertices, faces = np.load(vp, allow_pickle=False), np.load(fp, allow_pickle=False)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Invalid complete original mesh: '+object_id)
        if not np.allclose(vertices.min(0)+vertices.max(0), 0, atol=2e-6, rtol=0):
            raise ValueError('Public fixed origin does not center this asset')
        if abs(float(np.ptp(vertices, axis=0).max())-1/3.1) > 2e-6:
            raise ValueError('Unexpected official canonical scale')
        cases.append(dict(object_id=object_id, vertices=vertices, faces=faces,
                          source_vertices=str(vp), source_faces=str(fp),
                          source_vertices_sha256=digest(vp), source_faces_sha256=digest(fp)))
    return cases


def sample_surface(vertices, faces, count, seed):
    """Standard area-uniform triangle sampling; no remesh or vertex filtering."""
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if not np.isfinite(twice_area).all() or twice_area.sum() <= 0:
        raise ValueError('Mesh has no finite positive surface area')
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(faces), count, p=twice_area/twice_area.sum())
    uv = rng.random((count, 2)); root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1-root, root*(1-uv[:, 1]), root*uv[:, 1]))
    points = np.einsum('ni,nij->nj', bary, triangles[face_ids])
    normals = cross[face_ids]/twice_area[face_ids, None]
    return points, normals, face_ids, bary


def numerical_gate(metrics):
    required = ('cd_x9000', 'fscore', 'area_ratio', 'maximum_vertex_distance',
                'maximum_face_probe_distance', 'occupancy_iou')
    if not all(k in metrics and np.isfinite(metrics[k]) for k in required):
        return dict(finite=False, passed=False)
    checks = dict(finite=True, cd=metrics['cd_x9000'] <= .45,
                  fscore=metrics['fscore'] >= .95,
                  area=.7 <= metrics['area_ratio'] <= 1.5,
                  vertices=metrics['maximum_vertex_distance'] <= DISTANCE,
                  face_probes=metrics['maximum_face_probe_distance'] <= DISTANCE,
                  occupancy=metrics['occupancy_iou'] >= .90)
    return dict(checks, passed=all(checks.values()))


def dependency_check():
    names = ('torch', 'einops', 'pytorch_lightning', 'torchmetrics',
             'skimage', 'yaml', 'scipy', 'trimesh', 'rtree', 'tqdm', 'cv2', 'omegaconf')
    return {name: importlib.util.find_spec(name) is not None for name in names}


def load_official_model(checkpoint, device):
    import torch
    import yaml
    config = yaml.safe_load((VENDOR/CONFIG).read_text())['model']['params']['shape_module_cfg']
    if config['target'] != 'michelangelo.models.tsal.sal_perceiver.AlignedShapeLatentPerceiver':
        raise ValueError('Unexpected official shape architecture')
    sys.path.insert(0, str(VENDOR))
    module_name, class_name = config['target'].rsplit('.', 1)
    original = getattr(importlib.import_module(module_name), class_name)
    if Path(importlib.import_module(module_name).__file__).resolve() != (VENDOR/'michelangelo/models/tsal/sal_perceiver.py').resolve():
        raise ValueError('Shape implementation did not originate in official vendor')
    model = original(device=None, dtype=None, **config['params'])
    # This trusted checkpoint is the official byte/SHA-verified publication.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    prefix = 'model.shape_model.'
    state = saved['state_dict']
    shape_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    excluded = sorted(k for k in state if not k.startswith(prefix))
    if not shape_state or any(not (k.startswith('model.clip_model.')
                                  or k == 'model.shape_projection'
                                  or k.startswith('loss.')) for k in excluded):
        raise ValueError('Unexpected non-shape checkpoint keys; do not silently ignore')
    model.load_state_dict(shape_state, strict=True)
    for key, value in model.state_dict().items():
        if not torch.equal(value, shape_state[key]):
            raise AssertionError('Official shape parameter differs after loading: '+key)
    receipt = dict(shape_tensor_count=len(shape_state),
                   shape_parameter_count=sum(p.numel() for p in model.parameters()),
                   excluded_nonshape_keys=excluded, strict_shape_load=True,
                   official_config=config)
    del saved, shape_state, state
    model.eval().requires_grad_(False)
    return model.to(device), receipt


def closest_distance(mesh, points):
    import trimesh
    distances = []
    for offset in range(0, len(points), 2048):
        _, distance, _ = trimesh.proximity.closest_point(mesh, points[offset:offset+2048])
        distances.append(distance)
    return np.concatenate(distances)


def evaluate_mesh(truth_vertices, truth_faces, vertices, faces, seed):
    import trimesh
    truth = trimesh.Trimesh(truth_vertices, truth_faces, process=False)
    prediction = trimesh.Trimesh(vertices, faces, process=False)
    target, _, _, _ = sample_surface(truth_vertices, truth_faces, EVAL_POINTS, seed)
    sampled, _, _, _ = sample_surface(vertices, faces, EVAL_POINTS, seed+1)
    p2t, t2p = closest_distance(truth, sampled), closest_distance(prediction, target)
    precision, recall = (p2t <= DISTANCE).mean(), (t2p <= DISTANCE).mean()
    vertex_distance = closest_distance(truth, vertices)
    # Equal face IDs, including small wrong triangles; not area weighted.
    probe = np.einsum('pi,fij->fpj', FACE_PROBES, vertices[faces])
    probe_distance = closest_distance(truth, probe.reshape(-1, 3)).reshape(len(faces), -1)
    return dict(cd_x9000=float(9000*(np.square(p2t).mean()+np.square(t2p).mean())),
                fscore=float(2*precision*recall/(precision+recall)) if precision+recall else 0.,
                precision=float(precision), recall=float(recall),
                area_ratio=float(prediction.area/truth.area),
                maximum_vertex_distance=float(vertex_distance.max()),
                maximum_face_probe_distance=float(probe_distance.max()),
                bad_face_fraction=float((probe_distance.max(1) > DISTANCE).mean()),
                mesh_watertight=bool(prediction.is_watertight),
                mesh_winding_consistent=bool(prediction.is_winding_consistent),
                mesh_vertices=len(vertices), mesh_faces=len(faces),
                target_mesh_watertight=bool(truth.is_watertight))


def run(args, cases):
    from .retained_execution import require_active_resource
    resource = require_active_resource()
    import torch
    import trimesh
    if args.checkpoint.stat().st_size != CHECKPOINT_BYTES or digest(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Official checkpoint length/SHA not verified')
    if not all(dependency_check().values()):
        raise RuntimeError('Missing real dependencies; no package shims allowed')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(scope='full_surface_native_ae_only', object_ids=list(OBJECT_IDS),
                    shape_model='Complete official AlignedShapeLatentPerceiver',
                    checkpoint_sha256=CHECKPOINT_SHA, input_points=INPUT_POINTS,
                    teacher_inputs='Complete true mesh XYZ+face normals, not tactile observations',
                    abc_to_vae_scale=ABC_TO_VAE, posterior='deterministic mode, not random sample',
                    extract_geometry=dict(implementation='unmodified official', octree_depth=7,
                                          num_chunks=10000, bounds=[-1.25]*3+[1.25]*3,
                                          known_coordinate_limitation='Original grid_size denominator retained'),
                    gates=dict(cd_x9000_max=.45, fscore_min=.95, distance_abc=DISTANCE,
                               area_ratio=[.7, 1.5], maximum_vertex_distance=DISTANCE,
                               maximum_seven_face_probe_distance=DISTANCE, occupancy_iou_min=.90),
                    visual_inspection_required=True, model_updates=0, physics_controls=0,
                    sources={str(p.relative_to(VENDOR)): digest(p) for p in VENDOR.rglob('*.py')},
                    config_sha256=digest(VENDOR/CONFIG), resource=resource)
    (args.output/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n')
    torch.manual_seed(SEED)
    model, loading = load_official_model(args.checkpoint, torch.device('cuda'))
    from michelangelo.models.tsal.inference_utils import extract_geometry
    records = []
    with torch.no_grad():
        for index, case in enumerate(cases):
            oid = case['object_id']; record = {k: v for k, v in case.items() if k not in ('vertices', 'faces')}
            try:
                truth_v, truth_f = case['vertices'], case['faces']
                pc, normals, sampled_faces, bary = sample_surface(truth_v, truth_f, INPUT_POINTS, SEED+index)
                surface = torch.from_numpy((pc*ABC_TO_VAE).astype(np.float32))[None].cuda()
                normal_t = torch.from_numpy(normals.astype(np.float32))[None].cuda()
                embed, z, posterior = model.encode(surface, normal_t, sample_posterior=False)
                _, repeat, _ = model.encode(surface, normal_t, sample_posterior=False)
                if z.shape != (1, 256, 64) or not torch.isfinite(z).all() or not torch.equal(z, repeat):
                    raise AssertionError('Shape latent must be finite deterministic [1,256,64]')
                latents = model.decode(z)
                geometry = lambda queries: model.query_geometry(queries, latents)
                meshes, valid = extract_geometry(geometry, surface.device, batch_size=1,
                                                octree_depth=7, num_chunks=10000)
                if not bool(valid[0]):
                    raise RuntimeError('Released decoder produced no extractable surface')
                predicted_vae, predicted_faces = meshes[0]
                prediction = predicted_vae/ABC_TO_VAE
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite extracted shape')
                # Preserve actual model outputs even if later evaluation fails.
                # A failed occupancy check must not hide an existing raw mesh.
                arrays = dict(
                    truth_vertices_canonical=truth_v, truth_faces=truth_f,
                    input_surface_xyz_vae=surface[0].cpu().numpy(), input_normals=normals,
                    input_face_indices=sampled_faces, input_barycentric=bary,
                    shape_latent=z[0].cpu().numpy(), aligned_shape_embed=embed[0].cpu().numpy(),
                    posterior_mean=posterior.mean[0].cpu().numpy(),
                    posterior_logvar=posterior.logvar[0].cpu().numpy(),
                    reconstruction_vertices_canonical=prediction,
                    reconstruction_faces=predicted_faces,
                    reconstruction_vertices_vae=predicted_vae,
                    abc_to_vae_scale=np.float64(ABC_TO_VAE))
                np.savez_compressed(args.output/f'{oid}.npz', **arrays)
                record.update(output_file=f'{oid}.npz', raw_reconstruction_saved=True)
                # Fixed independent volume queries plus near-surface queries;
                # only GT labels, never fed back to encoding or reconstruction.
                rng = np.random.default_rng(SEED+100+index)
                near, _, _, _ = sample_surface(truth_v, truth_f, EVAL_POINTS, SEED+200+index)
                queries_vae = np.concatenate((rng.uniform(-1.1, 1.1, (EVAL_POINTS, 3)),
                                              near*ABC_TO_VAE+rng.normal(0, .02, near.shape)))
                truth_mesh = trimesh.Trimesh(truth_v, truth_f, process=False)
                if not truth_mesh.is_watertight or not truth_mesh.is_winding_consistent:
                    raise ValueError('Ambiguous truth occupancy; preserve failure, do not repair mesh')
                occupancy = truth_mesh.contains(queries_vae/ABC_TO_VAE)
                logits = []
                for offset in range(0, len(queries_vae), 10000):
                    q = torch.from_numpy(queries_vae[offset:offset+10000].astype(np.float32))[None].cuda()
                    logits.append(geometry(q)[0].cpu().numpy())
                logits = np.concatenate(logits); predicted_occupancy = logits >= 0
                union = np.logical_or(occupancy, predicted_occupancy).sum()
                iou = float(np.logical_and(occupancy, predicted_occupancy).sum()/union) if union else 0.
                arrays.update(occupancy_queries_vae=queries_vae,
                              occupancy_labels=occupancy, occupancy_logits=logits)
                np.savez_compressed(args.output/f'{oid}.npz', **arrays)
                metrics = evaluate_mesh(truth_v, truth_f, prediction, predicted_faces, SEED+300+index)
                metrics['occupancy_iou'] = iou
                record.update(complete=True, output_file=f'{oid}.npz', metrics=metrics,
                              checks=numerical_gate(metrics), repeat_latent_max_error=0.)
            except Exception as error:
                record.update(complete=False, checks=dict(passed=False),
                              error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            records.append(record)
            (args.output/'PROGRESS.json').write_text(json.dumps(records, indent=2)+'\n')
            print('MICHELANGELO_NATIVE_AE', oid, record.get('metrics'), record['checks'], flush=True)
    result = dict(complete=len(records) == 4, scope=protocol['scope'], cases=records,
                  numerical_passed=all(r['checks']['passed'] for r in records),
                  representation_qualified=False, visual_inspection='pending',
                  checkpoint_sha256=CHECKPOINT_SHA, loading=loading,
                  model_updates=0, physics_controls=0, tactile_model_forwards=0)
    (args.output/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['numerical_passed'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    cases = read_cases(args.dataset)
    if args.check:
        dependencies = dependency_check()
        print(json.dumps(dict(object_ids=[r['object_id'] for r in cases], dependencies=dependencies,
                              checkpoint_present=args.checkpoint.exists(), vendor_present=VENDOR.is_dir(),
                              model_forwards=0, physics_controls=0), indent=2))
        return 0 if all(dependencies.values()) and args.checkpoint.exists() else 2
    if args.output is None:
        parser.error('--output required for the separately scheduled actual qualification')
    return run(args, cases)


if __name__ == '__main__':
    raise SystemExit(main())
