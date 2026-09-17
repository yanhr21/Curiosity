"""Full official ABC meshes in a public, fixed shape-pretraining fixture.

Public geometry convention: x_world = 0.93 * x_canonical + [0, 0, 0.75],
identity rotation. Official canonical arrays have maximum extent 1 / 3.1,
so every object has maximum physical extent 0.30 m. No per-object rescaling,
mesh repair, hull, internal-shell removal, face filtering or OBJ reload occurs.

Only physics, supervision and rendering may consume the returned GT geometry.
The predictor may use the declared fixture frame/scale to transform observations,
but may not receive mesh vertices/faces, asset identity, or GT-derived features.
This known fixture is for static shape pretraining; it does not estimate unknown
object pose, size, mass or material. Loading assets creates no physics or model.

API:
    fixed_selection() -> tuple of (object_id, split)
    load_fixture_asset(prepared_root, object_id) -> FixtureAsset
    official_action_directions() -> (float32 [50, 3], JSON provenance)

Loading requires actual completed prepare_selected.py mesh-stage receipts and
matching original _verts.npy/_faces.npy bytes. Missing/failed IDs raise explicitly;
callers must preserve those failures and never replace them with another object.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import stat

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'experiments/object_predictor_v1'
DEFAULT_MANIFEST = EXPERIMENT / 'research_review_20260916/abc_data_feasibility_v1/CANDIDATE48_MANIFEST.json'
OFFICIAL_UTILS = EXPERIMENT / 'vendor/Active-3D-Vision-and-Touch/pterotactyl/utility/utils.py'
MANIFEST_SHA256 = '1d2e52858698ace7d7991cdea9502d93c40c59683f434c5f14ffadb9b97f7dc3'
CANONICAL_NORMALIZATION_SCALE = 3.1
METERS_PER_CANONICAL_UNIT = 0.93
FIXTURE_CENTER_M = (0., 0., 0.75)
FIXTURE_QUATERNION_XYZW = (0., 0., 0., 1.)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(root, relative):
    path = root
    for index, part in enumerate(Path(relative).parts):
        path = path / part
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) if index == len(Path(relative).parts) - 1 else stat.S_ISDIR(mode)):
            raise ValueError(f'Expected regular file/real directory, not link: {path}')
    return path


def _readonly(array):
    array = np.asarray(array)
    array.setflags(write=False)
    return array


def fixed_selection(manifest_path=DEFAULT_MANIFEST):
    """The frozen 32/8/8 IDs in source order; no assets are read or substituted."""
    manifest_path = Path(manifest_path)
    if _digest(manifest_path) != MANIFEST_SHA256:
        raise ValueError('Fixed 48-ID manifest SHA256 mismatch')
    selected = json.loads(manifest_path.read_text())['selected']
    if set(selected) != {'recon_train', 'valid', 'test'}:
        raise ValueError('Unexpected fixed split names')
    rows = []
    for split, expected in (('recon_train', 32), ('valid', 8), ('test', 8)):
        ids = selected[split]
        if len(ids) != expected or any(not isinstance(i, str) or not i.isdecimal() for i in ids):
            raise ValueError(f'Invalid fixed IDs for {split}')
        rows.extend((ident, split) for ident in ids)
    if len({ident for ident, _ in rows}) != 48:
        raise ValueError('Expected 48 unique fixed IDs')
    return tuple(rows)


def mesh_statistics(vertices, faces):
    """Validate readability/indexing and describe topology without changing it.

Open/nonmanifold/duplicate/degenerate triangles are reported, not removed.
Signed volume is only an algebraic statistic, not a watertightness certificate
or a mass/inertia estimate. No test here certifies hydroelastic SDF suitability.
"""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
            or not np.issubdtype(vertices.dtype, np.floating) or not np.isfinite(vertices).all()):
        raise ValueError('Expected finite floating vertices [N>=3,3]')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Expected nonempty integer triangle indices within vertex bounds')
    low, high = vertices.min(0), vertices.max(0)
    if np.max(high - low) <= 0:
        raise ValueError('Mesh has zero extent')
    triangles = vertices[faces].astype(np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    signed_volume = np.einsum('ij,ij->i', triangles[:, 0],
                              np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.
    return dict(vertices=len(vertices), faces=len(faces), vertex_dtype=str(vertices.dtype),
                index_dtype=str(faces.dtype), bounds=[low.tolist(), high.tolist()],
                extent=(high - low).tolist(), area=float(twice_area.sum() / 2.),
                algebraic_signed_volume=float(signed_volume), zero_area_faces=int((twice_area == 0).sum()),
                duplicate_unordered_faces=int(len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))),
                unique_edges=len(counts), boundary_edges=int((counts == 1).sum()),
                edges_with_more_than_two_faces=int((counts > 2).sum()),
                referenced_vertices=len(np.unique(faces)),
                hydroelastic_suitability_verified=False, mesh_modified=False)


@dataclass(frozen=True)
class FixtureAsset:
    object_id: str
    split: str
    canonical_vertices: np.ndarray
    physics_vertices_m: np.ndarray
    faces: np.ndarray
    center_m: np.ndarray
    quaternion_xyzw: np.ndarray
    canonical_to_world: np.ndarray
    world_to_canonical: np.ndarray
    metadata: dict

    @property
    def world_vertices_m(self):
        return self.canonical_points_to_world(self.canonical_vertices)

    def canonical_points_to_world(self, points):
        """Apply only the publicly declared fixture transform, no GT alignment."""
        return np.asarray(points, dtype=np.float64) * METERS_PER_CANONICAL_UNIT + self.center_m

    def world_points_to_canonical(self, points):
        """Also usable for measured contact positions using the known fixture."""
        return (np.asarray(points, dtype=np.float64) - self.center_m) / METERS_PER_CANONICAL_UNIT


def load_fixture_asset(prepared_root, object_id, manifest_path=DEFAULT_MANIFEST):
    """Load one actual prepared fixed-ID mesh; failure never selects another ID.

physics_vertices_m are in the body's local frame. Place that body at center_m
with quaternion_xyzw; do not also translate these local vertices. faces retain
the original array's order, winding, duplicate entries and all internal shells.
"""
    selection = fixed_selection(manifest_path)
    object_id = str(object_id)
    splits = dict(selection)
    if object_id not in splits:
        raise ValueError(f'Object {object_id} is not in the fixed 48-ID manifest')
    root = Path(prepared_root).absolute()
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError(f'Prepared root must be a real directory: {root}')
    config_path = _regular(root, 'CONFIG.json')
    result_path = _regular(root, 'mesh.RESULT.json')
    config, result = json.loads(config_path.read_text()), json.loads(result_path.read_text())
    if (config.get('manifest_sha256') != MANIFEST_SHA256
            or config.get('scale') != CANONICAL_NORMALIZATION_SCALE
            or result.get('stage') != 'mesh' or result.get('finished_all_48') is not True
            or result.get('official_sources_unchanged') is not True):
        raise ValueError('Require completed source-bound official mesh preparation')
    rows = result.get('objects', [])
    if [(row['object_id'], row['split']) for row in rows] != list(selection):
        raise ValueError('Prepared results differ from fixed 48 IDs/order')
    row = next(row for row in rows if row['object_id'] == object_id)
    if row['status'] != 'pass':
        raise ValueError(f'Prepared mesh failed for fixed ID {object_id}: {row.get("error")}')
    arrays, source_records = {}, {}
    for kind, suffix in (('prepared_vertices', 'verts'), ('prepared_faces', 'faces')):
        path = _regular(root, f'object_data/object_info/{object_id}_{suffix}.npy')
        record = row['artifacts'][kind]
        actual = dict(bytes=path.stat().st_size, sha256=_digest(path))
        if actual != {key: record[key] for key in ('bytes', 'sha256')}:
            raise ValueError(f'Prepared source array changed: {path}')
        arrays[kind] = np.load(path, allow_pickle=False)
        source_records[kind] = dict(path=str(path), preparation_origin=record.get('origin'), **actual)
    vertices, faces = arrays['prepared_vertices'], arrays['prepared_faces']
    stats = mesh_statistics(vertices, faces)
    bounds = np.asarray(stats['bounds'])
    if (not np.isclose(np.max(bounds[1] - bounds[0]), 1 / CANONICAL_NORMALIZATION_SCALE, atol=1e-6, rtol=0)
            or not np.allclose(bounds.sum(0), 0., atol=1e-6, rtol=0)):
        raise ValueError('Mesh does not satisfy official centered scale=3.1 coordinates; no rescaling fallback')
    physical = vertices.astype(np.float64) * METERS_PER_CANONICAL_UNIT
    center = np.asarray(FIXTURE_CENTER_M)
    transform = np.eye(4)
    transform[:3, :3] *= METERS_PER_CANONICAL_UNIT
    transform[:3, 3] = center
    inverse = np.eye(4)
    inverse[:3, :3] /= METERS_PER_CANONICAL_UNIT
    inverse[:3, 3] = -center / METERS_PER_CANONICAL_UNIT
    metadata = dict(object_id=object_id, split=splits[object_id], manifest_path=str(Path(manifest_path).absolute()),
                    manifest_sha256=MANIFEST_SHA256, source_arrays=source_records,
                    preparation_config_sha256=_digest(config_path), preparation_result_sha256=_digest(result_path),
                    preparation_inputs=row.get('inputs', {}),
                    official_preparation_sources=config.get('official_sources', {}),
                    canonical_mesh_statistics=stats, meters_per_canonical_unit=METERS_PER_CANONICAL_UNIT,
                    expected_max_extent_m=.30, actual_max_extent_m=float(np.ptp(physical, axis=0).max()),
                    fixture_center_m=center.tolist(), fixture_quaternion_xyzw=list(FIXTURE_QUATERNION_XYZW),
                    canonical_to_world=transform.tolist(), world_to_canonical=inverse.tolist(),
                    fixture_frame_is_public=True, geometry_uses=['physics', 'labels', 'rendering'],
                    gt_geometry_or_asset_id_permitted_as_predictor_input=False,
                    unknown_pose_size_mass_material_estimation=False, mesh_repaired=False,
                    topology_and_internal_shells_preserved=True)
    return FixtureAsset(object_id, splits[object_id], _readonly(vertices), _readonly(physical),
                        _readonly(faces), _readonly(center), _readonly(np.asarray(FIXTURE_QUATERNION_XYZW)),
                        _readonly(transform), _readonly(inverse), metadata)


def official_action_directions():
    """Original 50 action directions/order, with the official grasping minus sign.

These are center-outward ray directions, not surface normals. Bind the unchanged
get_circle class AST with CPU Torch, NumPy and math, without importing utils or
the grasping module. No CUDA, grasp hull, object mesh, or physics is invoked.
"""
    import torch

    source = OFFICIAL_UTILS.read_text()
    definitions = [node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name == 'get_circle']
    if len(definitions) != 1:
        raise ValueError('Official get_circle class interface changed')
    node = definitions[0]
    namespace = dict(np=np, math=math, torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(OFFICIAL_UTILS), 'exec'), namespace)
    circle = namespace['get_circle'](50)
    if circle.points.device.type != 'cpu':
        raise RuntimeError('Official action directions must be created on CPU')
    directions = -circle.points.data.numpy()
    if directions.shape != (50, 3) or not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-7, rtol=0):
        raise ValueError('Unexpected official 50-unit-direction output')
    provenance = dict(source=str(OFFICIAL_UTILS), source_sha256=_digest(OFFICIAL_UTILS),
                      class_line=node.lineno, class_source_sha256=hashlib.sha256(
                          ast.get_source_segment(source, node).encode()).hexdigest(),
                      expression='-get_circle(50).points.data.numpy()',
                      sign_reference='pterotactyl/simulator/physics/grasping.py:Agnostic_Grasp.__init__',
                      direction_semantics='center-outward action ray; not an inferred surface normal',
                      action_order_changed=False, cuda_called=False)
    return _readonly(directions), provenance
