"""Inference glue for the complete released Active 3D Vision and Touch GCN.

The network module and six mesh/graph functions execute directly from the
official checkout. Only the OBJ reader is adapted: these two public templates
need ordered vertices and triangle indices, not PyTorch3D's training losses or
renderer. This does not implement the official training engine or optical
touch-to-depth network. Numeric touch charts must already satisfy its contract.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


GRAPH_FUNCTIONS = (
    'load_mesh_vision', 'load_mesh_touch', 'normalize_adj',
    'calc_adj', 'adj_fuse_touch', 'adj_init',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_template_obj(path):
    """Read the shipped triangle templates without welding/reordering vertices.

    Implements only the vertex/index part consumed by official load_mesh_touch.
    Unsupported geometry fails rather than being silently triangulated. UVs and
    normals are unused by this interface, but their seams' duplicate vertices
    MUST remain: official graph construction connects byte-identical positions.
    """
    vertices, faces = [], []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.partition('#')[0].split()
        if not fields:
            continue
        if fields[0] == 'v':
            if len(fields) != 4:
                raise ValueError(f'{path}:{number}: expected xyz vertex')
            vertices.append([float(value) for value in fields[1:]])
        elif fields[0] == 'f':
            if len(fields) != 4:
                raise ValueError(f'{path}:{number}: expected triangle')
            indices = [int(value.split('/')[0]) for value in fields[1:]]
            if min(indices) <= 0:
                raise ValueError(f'{path}:{number}: only positive OBJ indices supported')
            faces.append([index - 1 for index in indices])
        elif fields[0] not in ('vt', 'vn', 'g', 'o', 's', 'mtllib', 'usemtl'):
            raise ValueError(f'{path}:{number}: unsupported OBJ record {fields[0]}')
    xyz = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Invalid template vertices')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.max() >= len(xyz):
        raise ValueError('Invalid template faces')
    return torch.from_numpy(xyz), SimpleNamespace(verts_idx=torch.from_numpy(triangles)), None


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_graph_functions(repo):
    """Compile unchanged official definitions, excluding unrelated loss imports.

    Active3D is MIT licensed; copyright/license remain in its source checkout.
    There are no substitute graph operations, CUDA patches, or fake modules.
    """
    repo = Path(repo).resolve()
    path = repo / 'pterotactyl/utility/utils.py'
    source = path.read_text()
    definitions = {node.name: node for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name in GRAPH_FUNCTIONS}
    if set(definitions) != set(GRAPH_FUNCTIONS):
        raise ValueError('Official mesh/graph API changed')
    objects = load_source_module('_active3d_official_objects', repo / 'pterotactyl/objects/__init__.py')
    namespace = dict(torch=torch, os=os, objects=objects, load_obj=read_template_obj)
    tree = ast.Module(body=[definitions[name] for name in GRAPH_FUNCTIONS], type_ignores=[])
    exec(compile(tree, str(path), 'exec'), namespace)
    hashes = {name: hashlib.sha256(ast.get_source_segment(source, definitions[name]).encode()).hexdigest()
              for name in GRAPH_FUNCTIONS}
    return SimpleNamespace(**{name: namespace[name] for name in GRAPH_FUNCTIONS}), hashes


class OfficialActive3D:
    """Full frozen official Deformation model and graph on one CUDA device."""

    def __init__(self, repo, checkpoint_dir, device='cuda:0'):
        self.repo = Path(repo).resolve()
        checkpoint_dir = Path(checkpoint_dir).resolve()
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Official GPU graph qualification requires an allocated CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if not os.environ.get('SLURM_STEP_ID'):
            raise RuntimeError('Run on the retained explicit compute step, not the login node')
        config_path = checkpoint_dir / 'config.json'
        config = json.loads(config_path.read_text())
        self.args = SimpleNamespace(**config)
        if config['use_img'] or not config['use_touch'] or config['num_grasps'] != 5:
            raise ValueError('This adapter qualifies the released five-observation touch-only models')
        self.touch_vertices = 25 * config['num_grasps'] * (1 if config['finger'] else 4)
        self.module = load_source_module('_active3d_official_vision_model',
                                        self.repo / 'pterotactyl/reconstruction/vision/model.py')
        functions, function_hashes = official_graph_functions(self.repo)
        vision_path = self.repo / 'pterotactyl/objects/vision_charts.obj'
        touch_path = self.repo / 'pterotactyl/objects/touch_chart.obj'
        vertices, face_info, _ = read_template_obj(vision_path)
        touch, touch_faces, _ = read_template_obj(touch_path)
        if vertices.shape != (1824, 3) or face_info.verts_idx.shape != (2304, 3):
            raise ValueError('Unexpected official full shape template')
        if touch.shape != (25, 3) or touch_faces.verts_idx.shape != (32, 3):
            raise ValueError('Unexpected official touch template')
        self.vision_vertex_count = len(vertices)
        self.vision_faces = face_info.verts_idx.numpy().copy()
        # Official functions allocate with .cuda(). Both plain graph dictionary
        # and model parameters must be created on this same current device.
        with torch.cuda.device(self.device):
            self.adj_info, self.initial_mesh = functions.load_mesh_vision(self.args, str(vision_path))
            self.model = self.module.Deformation(self.adj_info, self.initial_mesh, self.args).cuda()
            weights = torch.load(checkpoint_dir / 'model', map_location='cpu', weights_only=True)
            self.model.load_state_dict(weights, strict=True)
            self.model.eval().requires_grad_(False)
        if not all(torch.equal(tensor.cpu(), weights[name]) for name, tensor in self.model.state_dict().items()):
            raise ValueError('Full official checkpoint readback mismatch')
        n = self.vision_vertex_count + self.touch_vertices
        if self.adj_info['adj'].shape != (n, n):
            raise ValueError('Full graph dimension mismatch')
        if self.adj_info['faces'].shape != (2304 + self.touch_vertices // 25 * 32, 3):
            raise ValueError('Full graph face count mismatch')
        for key in ('adj', 'origional'):
            adjacency = self.adj_info[key]
            if not torch.isfinite(adjacency).all() or not torch.allclose(
                    adjacency.sum(1), torch.ones(adjacency.shape[0], device=self.device), atol=2e-6, rtol=0):
                raise ValueError(f'Invalid official normalized graph: {key}')
        if any(value.device != self.device for value in self.adj_info.values()) or self.initial_mesh.device != self.device:
            raise ValueError('Official graph tensors on wrong CUDA device')
        self.provenance = dict(
            checkpoint_sha256=digest(checkpoint_dir / 'model'), config_sha256=digest(config_path),
            config=config, parameters=sum(p.numel() for p in self.model.parameters()),
            state_tensors=len(weights), strict_load=True, full_loaded_state_exact=True,
            model_source_sha256=digest(self.repo / 'pterotactyl/reconstruction/vision/model.py'),
            utils_source_sha256=digest(self.repo / 'pterotactyl/utility/utils.py'),
            unchanged_official_function_sha256=function_hashes,
            vision_template_sha256=digest(vision_path), touch_template_sha256=digest(touch_path),
            graph_vertices=n, graph_faces=len(self.adj_info['faces']),
            graph_device=str(self.device), model_parameter_updates=0,
            scope='Full official frozen touch-only graph and model; OBJ parsing adapter only. '
                  'Not the optical sensing simulator, training losses or unknown-shape accuracy qualification.')

    @torch.inference_mode()
    def predict(self, touch_info):
        """Run original prepare_mesh and all three official deformation passes.

        Input [B, N, 4] stores already normalized xyz and official masks 0/1/2.
        This interface cannot itself verify whether chart topology, normalization
        or near-field mask1 are physically justified; the sensor adapter must.
        """
        info = torch.as_tensor(touch_info, dtype=torch.float32, device=self.device)
        if info.ndim != 3 or info.shape[1:] != (self.touch_vertices, 4) or not len(info):
            raise ValueError(f'Expected [B, {self.touch_vertices}, 4] chart input')
        if not torch.isfinite(info).all() or not torch.isin(info[..., 3], info.new_tensor([0, 1, 2])).all():
            raise ValueError('Chart coordinates or mask invalid')
        with torch.cuda.device(self.device):
            image_batch = info.new_zeros((len(info), 1))  # Official touch-only batch-size convention.
            charts = self.module.prepare_mesh(dict(img=image_batch, touch_charts=info), self.initial_mesh, self.args)
            vertices, mask = self.model(image_batch, charts)
        if not torch.isfinite(vertices).all():
            raise ValueError('Nonfinite official prediction')
        if not torch.equal(vertices[:, self.vision_vertex_count:], info[..., :3]):
            raise ValueError('Official forward altered fixed touch vertices')
        return vertices, mask
