"""Explicit geometry-loss repair for the complete official Active3D model.

This is loss/qualification glue, not a replacement network or a mesh repair.
GT mesh area and point labels are supervision only. The unchanged original
30000-point/three-repeat Chamfer stays in the objective. Only the 1824 global
vertices/2304 faces receive these extra terms; fixed touch charts are excluded.

All coordinates are official canonical units, NOT meters. For each label mesh
with area A and F predicted faces, ell=sqrt(4*A/(sqrt(3)*F)) is the nominal
equilateral-triangle edge length. Edge loss on V/ell is dimensionless squared
length; uniform Laplacian loss on V/ell is dimensionless LENGTH (not squared).
Normal loss is divided by 1-cos(30deg). Their sum and the excess-area penalty
are weighted .045, 10% of the pre-existing .45 Chamfer gate. These are fixed
engineering presets, not published Active3D defaults or tuned optimal weights.

Uniform vertex/face-index losses and the worst 5% face loss use the same 9000
canonical squared-distance multiplier as the original loss. Face probes are
three vertices, three edge midpoints and the centroid. They do not certify a
continuous-surface Hausdorff bound. Chart seams are NOT welded or constrained.
Reference: https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
"""
from __future__ import annotations

import math


DISTANCE_LIMIT = .01
AREA_RATIO_MIN = .7
AREA_RATIO_MAX = 1.5
DISTANCE_WEIGHT = 9000.
REGULARIZER_WEIGHT = .045
TAIL_FRACTION = .05
NORMAL_REFERENCE_DEGREES = 30.
PROBES = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
          (.5, .5, 0.), (.5, 0., .5), (0., .5, .5), (1/3, 1/3, 1/3))


def configuration():
    return dict(
        name='official_active3d_geometry_loss_repair_v1',
        original_loss_unchanged='30000 points x 3 repeats x 9000, all global+touch faces',
        added_loss_scope='Global prediction only; GT meshes/points are supervision, never model input',
        distance_weight=DISTANCE_WEIGHT, regularizer_weight=REGULARIZER_WEIGHT,
        uniform_vertex_squared_distance_weight=1., uniform_face_squared_distance_weight=1.,
        worst_face_squared_distance_weight=1., worst_face_fraction=TAIL_FRACTION,
        face_probes_barycentric=[list(p) for p in PROBES],
        nominal_edge='sqrt(4 * GT_area / (sqrt(3) * global_face_count))',
        edge='Official mesh_edge_loss(V/ell), target_length=0',
        laplacian='Official mesh_laplacian_smoothing(V/ell, method=uniform); length, not squared',
        normal='Official mesh_normal_consistency / (1-cos(30deg))',
        excess_area='relu(predicted_global_area / raw_GT_area - 1.5)^2',
        preset_basis='Regularizer unit costs 10% of original CD gate .45; geometry uses original 9000 distance scale. Fixed engineering presets, no sweep.',
        gate=dict(minimum_area_ratio=AREA_RATIO_MIN, maximum_area_ratio=AREA_RATIO_MAX,
                  maximum_vertex_label_distance=DISTANCE_LIMIT,
                  maximum_seven_probe_label_distance=DISTANCE_LIMIT,
                  every_fixed_observed_case=True, old_numeric_gate_also_required=True,
                  actual_render_and_independent_visual_inspection_required=True),
        seam='Diagnostic only; no welding, hard equality, or closed-sphere topology constraint',
        limitations=['Finite probes do not certify Hausdorff distance or absence of self-intersections.',
                     'Numerical repair gate does not substitute for actual mesh render and visual inspection.',
                     'Native-only qualification cannot substitute for the separate actual SUGAR input gate.'])


def geometry_case_gate(metrics):
    keys = ('area_ratio', 'maximum_vertex_label_distance', 'maximum_face_probe_label_distance',
            'bad_face_fraction', 'predicted_global_area', 'truth_area')
    finite = all(math.isfinite(float(metrics[k])) and float(metrics[k]) >= 0 for k in keys)
    finite = finite and metrics['truth_area'] > 0 and metrics['bad_face_fraction'] <= 1
    return dict(
        finite_geometry_metrics=finite,
        area_ratio_bounded=AREA_RATIO_MIN <= metrics['area_ratio'] <= AREA_RATIO_MAX,
        all_vertices_near_label=metrics['maximum_vertex_label_distance'] <= DISTANCE_LIMIT,
        all_seven_face_probes_near_label=metrics['maximum_face_probe_label_distance'] <= DISTANCE_LIMIT,
        no_bad_probed_face=metrics['bad_face_fraction'] == 0.,
    )


def formal_repair_gate(training, rendering=None, inspection=None):
    """A rendered artifact and explicit all-eight-case visual review are required.

    Inspection schema: {checkpoint_sha256, cases:[{name, passed, images:[three
    actual still filenames]}]}.
Every expected view00/view32/view64 image must be explicitly listed per case.
This helper never manufactures a review or changes a saved result.
"""
    rendering, inspection = rendering or {}, inspection or {}
    expected = {f'{ident}_repeat{r}' for ident in ('18704', '11898', '15737', '13266') for r in (0, 1)}
    reviews = inspection.get('cases', [])
    names = [row.get('name') for row in reviews]
    reviewed = len(reviews) == 8 and set(names) == expected
    for row in reviews:
        name = row.get('name')
        reviewed = reviewed and row.get('passed') is True and len(row.get('images', [])) == 3 and set(row.get('images', [])) == {
            f'{name}_view{view:02d}.png' for view in (0, 32, 64)}
    reviewed = reviewed and bool(training.get('checkpoint_sha256')) and (
        inspection.get('checkpoint_sha256') == training.get('checkpoint_sha256'))
    render_names = [row.get('name') for row in rendering.get('records', [])]
    checks = dict(
        original_numeric_gate=training.get('native_numeric_gate_passed') is True,
        geometry_gate=training.get('geometry_gate_passed') is True,
        complete_fixed_training=(training.get('complete') is True and training.get('fixed_input_cases') == 8
                                 and training.get('optimizer_updates') == 1000
                                 and training.get('checkpoint_reload', {}).get('passed') is True),
        actual_complete_render=(rendering.get('complete') is True and rendering.get('actual_mesh_stills') == 24
                                and rendering.get('frames') == 768 and rendering.get('all_frames_decoded') is True
                                and len(render_names) == 8 and set(render_names) == expected
                                and bool(training.get('checkpoint_sha256'))
                                and rendering.get('checkpoint_sha256') == training.get('checkpoint_sha256')),
        independent_all_case_visual_inspection=bool(reviewed),
    )
    return dict(passed=all(checks.values()), checks=checks)


class GeometryRepairLoss:
    """Use only official PyTorch3D operators plus explicit loss reductions."""

    def __init__(self, target_points, truth_vertices, truth_faces, global_faces):
        import torch
        from pytorch3d.structures import Meshes

        if target_points.ndim != 3 or target_points.shape[-1] != 3:
            raise ValueError('Expected batch of original XYZ point labels')
        if len(truth_vertices) != len(target_points) or len(truth_faces) != len(target_points):
            raise ValueError('One complete GT mesh per unchanged input is required')
        self.target = target_points.detach()
        self.faces = global_faces.detach().to(device=target_points.device, dtype=torch.long)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or len(self.faces) == 0:
            raise ValueError('Expected original global triangle indices')
        with torch.no_grad():
            meshes = Meshes(
                verts=[torch.as_tensor(v, device=target_points.device, dtype=target_points.dtype).detach()
                       for v in truth_vertices],
                faces=[torch.as_tensor(f, device=target_points.device, dtype=torch.long).detach()
                       for f in truth_faces])
            packed = meshes.faces_areas_packed()
            self.truth_area = torch.stack([part.sum() for part in packed.split([len(f) for f in truth_faces])])
            if not bool(torch.isfinite(self.truth_area).all()) or not bool((self.truth_area > 0).all()):
                raise ValueError('Original GT surfaces require finite positive area')
            self.nominal_edge = torch.sqrt(4*self.truth_area/(math.sqrt(3)*len(self.faces)))
        self.probes = target_points.new_tensor(PROBES)
        self.tail_faces = max(1, math.ceil(TAIL_FRACTION*len(self.faces)))

    def geometry(self, vertices):
        """No RNG and no area weighting in any point/face distance reduction."""
        import torch
        from pytorch3d.ops import knn_points
        from pytorch3d.structures import Meshes

        if vertices.ndim != 3 or len(vertices) != len(self.target) or vertices.shape[-1] != 3:
            raise ValueError('Expected unchanged global vertex batch')
        triangles = vertices[:, self.faces]
        probes = torch.einsum('qk,bfkd->bfqd', self.probes, triangles)
        vertex_d2 = knn_points(vertices, self.target, K=1).dists[..., 0]
        probe_d2 = knn_points(probes.reshape(len(vertices), -1, 3), self.target, K=1).dists[..., 0]
        probe_d2 = probe_d2.reshape(len(vertices), len(self.faces), len(PROBES))
        face_max_d2 = probe_d2.max(dim=2).values
        meshes = Meshes(verts=list(vertices), faces=[self.faces]*len(vertices))
        area = meshes.faces_areas_packed().reshape(len(vertices), len(self.faces)).sum(dim=1)
        return dict(
            uniform_vertex_d2=vertex_d2.mean(dim=1),
            uniform_face_probe_d2=probe_d2.mean(dim=(1, 2)),
            worst_face_d2=face_max_d2.topk(self.tail_faces, dim=1).values.mean(dim=1),
            maximum_vertex_label_distance=vertex_d2.max(dim=1).values.sqrt(),
            maximum_face_probe_label_distance=face_max_d2.max(dim=1).values.sqrt(),
            bad_face_fraction=(face_max_d2 > DISTANCE_LIMIT**2).to(vertices.dtype).mean(dim=1),
            predicted_global_area=area, truth_area=self.truth_area,
            area_ratio=area/self.truth_area, nominal_edge=self.nominal_edge,
        )

    def augment(self, original_loss, vertices):
        import torch
        from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing
        from pytorch3d.structures import Meshes

        values = self.geometry(vertices)
        scaled = vertices/self.nominal_edge[:, None, None]
        meshes = Meshes(verts=list(scaled), faces=[self.faces]*len(vertices))
        edge = mesh_edge_loss(meshes)
        normal = mesh_normal_consistency(meshes)/(1-math.cos(math.radians(NORMAL_REFERENCE_DEGREES)))
        laplacian = mesh_laplacian_smoothing(meshes, method='uniform')
        area = torch.relu(values['area_ratio']-AREA_RATIO_MAX).square().mean()
        geometric = DISTANCE_WEIGHT*(values['uniform_vertex_d2']+values['uniform_face_probe_d2']
                                     +values['worst_face_d2']).mean()
        regularizer = REGULARIZER_WEIGHT*(edge+normal+laplacian+area)
        extra = geometric+regularizer
        if not bool(torch.isfinite(extra)):
            raise FloatingPointError('Nonfinite geometry repair objective')
        record = dict(extra_geometry_loss=float(extra.detach()),
                      uniform_vertex_d2=float(values['uniform_vertex_d2'].mean().detach()),
                      uniform_face_probe_d2=float(values['uniform_face_probe_d2'].mean().detach()),
                      worst_face_d2=float(values['worst_face_d2'].mean().detach()),
                      normalized_official_edge=float(edge.detach()), normalized_official_normal=float(normal.detach()),
                      normalized_official_uniform_laplacian=float(laplacian.detach()), excess_area=float(area.detach()))
        return original_loss+extra, record

    def evaluate(self, vertices):
        import torch
        with torch.no_grad():
            values = self.geometry(vertices)
        return [{key: float(value[i]) for key, value in values.items()} for i in range(len(vertices))]
