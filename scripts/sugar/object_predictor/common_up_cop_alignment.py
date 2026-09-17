"""Independent initial tangent-frame adapter; unchanged qualified CoP feedback."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .qualified_cop_response import QualifiedCoPResponseController, TELEMETRY
from .surface_grip_scene import SurfaceGripScene

INTERVENTION='common_up_cop_alignment_v1'
DESCRIPTION='Only initial CAD tangent roll aligns both hands projected CAD+X with world+Z; original normal/support centers/yaw and qualified CoP feedback unchanged.'


def common_up_frames(frames):
    result=[]
    for frame, normal in zip(frames,([1.,0.,0.],[-1.,0.,0.]),strict=True):
        normal=np.asarray(normal);up=np.array([0.,0.,1.])
        original_turn=Rotation.align_vectors(normal[None],up[None])[0]
        tangent=original_turn.apply([1.,0.,0.])
        angle=np.arctan2(normal @ np.cross(tangent,up),tangent @ up)
        # Rotate about original approach normal, preserving support triangle.
        new_base=Rotation.from_rotvec([0.,0.,angle])*frame[0]
        result.append((new_base,frame[1].copy()))
    return result


class CommonUpCoPController(QualifiedCoPResponseController):
    intervention_name=INTERVENTION

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.frames=common_up_frames(self.frames)
        # Rotation about local support Z leaves the original CAD normal exact.
        # Keep existing support_normals instead of introducing roundoff changes.


def initialize_common_up_scene(scene):
    """Before the first physical step, replace only the two hand joint poses."""
    import newton
    if scene.frame!=0 or scene.time!=0.:raise ValueError('Initial hand frame must be installed before stepping')
    poses,_,scene.probe_record=deepcopy(scene.probe).command(0.,np.zeros(2),.02)
    q=scene.state_0.joint_q.numpy()
    q_before=q.copy();qd_before=scene.state_0.joint_qd.numpy().copy()
    for start,pose in zip(scene.q_starts,poses,strict=True):q[start:start+7]=pose
    scene.state_0.joint_q.assign(q)
    newton.eval_fk(scene.model,scene.state_0.joint_q,scene.state_0.joint_qd,scene.state_0)
    untouched=np.ones(len(q),bool)
    for start in scene.q_starts:untouched[start:start+7]=False
    if not np.array_equal(scene.state_0.joint_q.numpy()[untouched],q_before[untouched]):raise ValueError('Non-hand joint state changed')
    if not np.array_equal(scene.state_0.joint_qd.numpy(),qd_before):raise ValueError('Initial velocities changed')
    return poses


class CommonUpSurfaceGripScene(SurfaceGripScene):
    def __init__(self,*args,**kwargs):
        if kwargs.get('controller_type') is not CommonUpCoPController:
            raise ValueError('Explicit common-up controller required')
        super().__init__(*args,**kwargs)
        poses=initialize_common_up_scene(self)
        from .qualify_common_up_geometry import qualify_actual_scene
        self.initial_geometry_receipt=qualify_actual_scene(self,poses)


def register_intervention():
    from .controller_interventions import INTERVENTIONS,INTERVENTION_DEPENDENCIES
    spec=('common_up_cop_alignment','CommonUpCoPController',DESCRIPTION,TELEMETRY)
    if INTERVENTION in INTERVENTIONS and INTERVENTIONS[INTERVENTION]!=spec:
        raise ValueError('Conflicting local registration')
    INTERVENTIONS[INTERVENTION]=spec
    INTERVENTION_DEPENDENCIES[INTERVENTION]=('qualified_cop_response','prelift_cop_alignment',
        'freeze_postlift_alignment','fixed_path_surface_grip','alignment_rollback','response_gain')
