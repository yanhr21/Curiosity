"""Project fusion adapter: empirical neural prior + closest-surface residuals.

Uses the complete official triangle query and TorchLie SO(3) operators. This is
not the original CHSEL loss, a calibrated posterior, or a learned replacement.
"""
import numpy as np
import torch
import torchlie.functional as lie
from scipy.spatial.transform import Rotation
from chsel.costs import RegistrationCost


class PriorContactCost(RegistrationCost):
    def __init__(self, factory, points, area, hand, original_center, original_rotation, scales,
                 floor_vertices=None, hand_pose_w=None):
        self.factory = factory
        self.dtype, self.device = points.dtype, points.device
        self.points = points
        self.center = torch.as_tensor(original_center, dtype=self.dtype, device=self.device)
        self.rotation = torch.as_tensor(original_rotation, dtype=torch.float64, device=self.device)
        self.pose_scale = torch.as_tensor(scales['pose_scales_m_rad'], dtype=self.dtype, device=self.device)
        self.contact_scale = torch.as_tensor(scales['contact_scales_m'], dtype=self.dtype, device=self.device)
        self.weights = []
        for h in (0, 1):
            w = torch.as_tensor(np.where(hand == h, area, 0), dtype=self.dtype, device=self.device)
            self.weights.append(w / w.sum() if w.sum() > 0 else w)
        assert torch.isfinite(self.pose_scale).all() and (self.pose_scale > 0).all()
        assert torch.isfinite(self.contact_scale).all() and (self.contact_scale > 0).all()
        self.floor_vertices = None
        if floor_vertices is not None:
            assert hand_pose_w is not None and len(floor_vertices) == 15626
            self.floor_vertices = torch.as_tensor(floor_vertices, dtype=torch.float64, device=self.device)
            self.world_up_in_hand = torch.as_tensor(Rotation.from_quat(hand_pose_w[3:]).as_matrix()[2],
                                                    dtype=torch.float64, device=self.device)
            self.hand_world_z = float(hand_pose_w[2])
            self.floor_sigma_m = .001

    def minimum_world_z(self, R, T):
        Q = R.transpose(-1, -2)
        center = -(Q @ T[..., None])[..., 0]
        up = self.world_up_in_hand.to(dtype=R.dtype)
        up_in_object = (R @ up[:, None])[..., 0]
        vertex_z = self.floor_vertices.to(dtype=R.dtype) @ up_in_object.T
        return vertex_z.min(dim=0).values + (center * up).sum(-1) + self.hand_world_z

    def terms(self, R, T):
        # R,T map current left-hand coordinates into the object frame.
        q = self.points @ R.transpose(-1, -2) + T[:, None, :]
        closest = self.factory.object_frame_closest_point(q.detach()).closest
        squared_distance = (q - closest.detach()).square().sum(-1)
        surface = sum((squared_distance * w).sum(-1) / sigma.square()
                      for w, sigma in zip(self.weights, self.contact_scale))
        Q = R.transpose(-1, -2)
        c = -(Q @ T[..., None])[..., 0]
        displacement = (c - self.center) / self.pose_scale[:3]
        rotvec = lie.SO3.log(Q @ self.rotation.to(dtype=Q.dtype).T)
        rotation_residual = rotvec / self.pose_scale[3:]
        terms = dict(contact=.5 * surface, position_prior=.5 * displacement.square().sum(-1),
                     rotation_prior=.5 * rotation_residual.square().sum(-1))
        if self.floor_vertices is not None:
            # Known environment plane from SupportScene, no object/contact GT mask.
            terms['floor'] = .5 * (torch.relu(-self.minimum_world_z(R,T)) / self.floor_sigma_m).square()
        return terms

    def __call__(self, R, T, s, other_info=None):
        return sum(self.terms(R, T).values())


def qualify_rotation(device):
    """Official TorchLie short-arc agreement and finite zero-angle derivatives."""
    axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
    angles = np.array([0., 1e-9, 1e-4, .3, np.pi-1e-4, np.pi+1e-4, 3.9])
    vectors = angles[:, None]*axis
    x = torch.tensor(vectors, dtype=torch.float64, device=device, requires_grad=True)
    matrices = lie.SO3.exp(x)
    logs = lie.SO3.log(matrices)
    expected = Rotation.from_matrix(matrices.detach().cpu().numpy()).as_rotvec()
    maximum = float(np.max(abs(logs.detach().cpu().numpy()-expected)))
    gradient, = torch.autograd.grad(.5*logs.square().sum(), x)
    assert maximum < 1e-6 and torch.isfinite(gradient).all()
    assert float(gradient[0].abs().max()) < 1e-10
    # Check the full log Jacobian, not only its zero-weighted square.
    jac = torch.autograd.functional.jacobian(lambda v: lie.SO3.log(lie.SO3.exp(v[None]))[0],
                                             torch.zeros(3, dtype=torch.float64, device=device))
    assert torch.allclose(jac, torch.eye(3, dtype=jac.dtype, device=device), atol=1e-8)
    return dict(short_arc_max_difference_rad=maximum, zero_gradient_max=float(gradient[0].abs().max()),
                zero_log_exp_jacobian_max_difference=float((jac-torch.eye(3,device=device)).abs().max()))


def qualify_cost(cost):
    """Six physical pose-direction finite differences with fresh mesh queries."""
    # Use float64 pose variables; official triangle queries still use float32.
    Q0, c0 = cost.rotation.double(), cost.center.double()
    def value(x):
        Q = lie.SO3.exp(x[None, 3:]) @ Q0
        R = Q.transpose(-1, -2)
        T = -(R @ (c0+x[:3])[None, :, None])[..., 0]
        # The operator preserves float64 query coordinates until the official query.
        original_points = cost.points
        cost.points = original_points.double()
        try:
            terms = cost.terms(R, T)
        finally:
            cost.points = original_points
        return sum(terms.values())
    checks = []
    for values in ([0.]*6, [.003,-.004,.005,.03,-.02,.01]):
        x = torch.tensor(values, device=cost.device, dtype=torch.float64, requires_grad=True)
        y = value(x)
        g, = torch.autograd.grad(y.sum(), x)
        assert torch.isfinite(g).all()
        numeric = []
        # 0.1mm / 0.0001rad exceeds float32 triangle-query quantization.
        for j in range(6):
            d = torch.zeros_like(x); d[j] = 1e-4
            numeric.append(float((value(x.detach()+d)-value(x.detach()-d))/(2e-4)))
        fd = torch.tensor(numeric,device=cost.device,dtype=g.dtype)
        scaled_error = float((abs(g-fd)/(1+abs(g)+abs(fd))).max())
        assert scaled_error < .03, (g,fd,scaled_error)
        checks.append(dict(pose_perturbation_m_rad=values, autograd=g.tolist(), finite_difference=numeric,
                           max_scaled_difference=scaled_error))
    return checks


def main(args):
    import inspect
    import json
    import os
    from pathlib import Path
    import pytorch_volumetric as pv
    import torchlie
    from .diagnose_hand_exclusion import sdf_for
    from .qualify_chsel_backend import read, digest
    from .shape_metric import rotation
    assert os.environ.get('SLURM_STEP_ID') and torch.cuda.is_available()
    torch.set_num_threads(4); np.random.seed(20260916)
    src=Path(args.inputs); output=Path(args.output)
    assert not output.exists()
    scales=json.loads(Path(args.scales).read_text())
    mesh=read(Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz'))
    v=mesh['vertices'].astype(np.float64); dims=np.ptp(v,axis=0); v-=(v.max(0)+v.min(0))/2
    rotation_checks=qualify_rotation('cuda'); checks=[]
    for case in json.loads((src/'RESULT.json').read_text())['cases']:
        data=read(src/case['input_file']); p=data['prediction']
        sdf=sdf_for(v*(np.exp(p[9:12])/dims),mesh['faces'])
        cost=PriorContactCost(sdf.obj_factory,
            torch.as_tensor(data['surface_points_current_left_hand_m'],device='cuda',dtype=torch.float32),
            data['surface_area_m2'],data['surface_hand'],p[:3],rotation(p[3:9]),scales,
            floor_vertices=v*(np.exp(p[9:12])/dims) if args.floor else None,
            hand_pose_w=data['hand_pose_w'][0] if args.floor else None)
        row=dict(episode=case['episode'],frame=case['frame'],checks=qualify_cost(cost))
        checks.append(row); print('FUSION_GRADIENT_CASE',json.dumps(row),flush=True)
    output.write_text(json.dumps(dict(complete=True,rotation=rotation_checks,cases=checks,
        scales_sha256=digest(Path(args.scales)), adapter_sha256=digest(Path(__file__)),
        official_mesh_query_source_sha256=digest(Path(inspect.getfile(pv.MeshObjectFactory))),
        torchlie_file=inspect.getfile(torchlie),floor=args.floor,
        model_forwards=0,registration_calls=0,physics_steps=0),indent=2))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True)
    ap.add_argument('--scales',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--floor',action='store_true')
    main(ap.parse_args())
