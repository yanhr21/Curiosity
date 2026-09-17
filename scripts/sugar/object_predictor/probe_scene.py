"""Multi-face acquisition using complete SUGAR meshes and actual force feedback.

The free object rests on a real ground plane. This is a sensing fixture, not a
whole-body policy. The controller sees clock and hand loads only; waypoints use a
fixed nominal workspace, never the randomized object's pose/scale or simulator IDs.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import newton
from .support_scene import SupportScene
from .geometry import palmar_support_frame,load_sugar_outer_box
from sugar_newton.hand.patches import load_hand_mesh


class ProbeController:
    segment_seconds=10.

    def __init__(self,gentle=False,stable_servo=False):
        self.gentle=gentle
        self.stable_servo=stable_servo
        self.segment_seconds=50. if stable_servo else (36. if gentle else 10.)
        self.frames=[palmar_support_frame(load_hand_mesh(side),sign)
                     for side,sign in [('left',-1.),('right',1.)]]
        self.phase=-1
        self.distance=np.zeros(2)
        self.touched=np.zeros(2,bool)
        self.last_positions=None
        self.last_rotations=None

    def configuration(self,phase):
        if phase==0:
            centers=np.array([[-.50,0.,.20],[.50,0.,.20]])
            normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
        elif phase==1:
            centers=np.array([[0.,-.50,.20],[0.,.50,.20]])
            normals=np.array([[0.,1.,0.],[0.,-1.,0.]])
        else:
            centers=np.array([[-.08,0.,.60],[.08,0.,.60]])
            normals=np.array([[0.,0.,-1.],[0.,0.,-1.]])
        if self.gentle:
            if phase==0: centers=np.array([[-.25,0.,.14],[.25,0.,.14]])
            elif phase==1: centers=np.array([[0.,-.30,.14],[0.,.30,.14]])
            else: centers=np.array([[-.08,0.,.38],[.08,0.,.38]])
        if self.stable_servo:
            if phase==0: centers=np.array([[-.35,0.,.14],[.35,0.,.14]])
            elif phase==1: centers=np.array([[0.,-.40,.14],[0.,.40,.14]])
            else: centers=np.array([[-.08,0.,.45],[.08,0.,.45]])
        rotations=[]
        for (base,_),normal in zip(self.frames,normals,strict=True):
            turn=Rotation.align_vectors(normal[None],np.array([[0.,0.,1.]]))[0]
            rotations.append(turn*base)
        return centers,normals,rotations

    def command(self,time,loads,dt):
        phase=min(int(time/self.segment_seconds),2)
        local=time-phase*self.segment_seconds
        centers,normals,rotations=self.configuration(phase)
        high=centers.copy();high[:,2]=.85
        if phase!=self.phase:
            self.previous_high=(self.configuration(max(phase-1,0))[0]).copy()
            self.previous_high[:,2]=.85
            self.previous_rotations=self.configuration(max(phase-1,0))[2]
            self.phase=phase;self.distance[:]=0.;self.touched[:]=False
        # High lateral transfer + rotation, lower outside the nominal object,
        # approach under force feedback, retreat outward, then return overhead.
        if local<1.:
            f=local
            points=(1-f)*self.previous_high+f*high
            rotations=[Slerp([0.,1.],Rotation.from_quat([a.as_quat(),b.as_quat()]))([f])[0]
                       for a,b in zip(self.previous_rotations,rotations,strict=True)]
        elif local<2.:
            f=local-1.;points=(1-f)*high+f*centers
        elif local<self.segment_seconds-2.:
            # Latch first actual load; no true contact normal or object state.
            self.touched |= loads>=.5
            if self.gentle:
                gain=.00005 if self.stable_servo else .002
                speed=np.where(self.touched,np.clip((1.-loads)*gain,-.02,.002),.006)
                limits=(-.12,.40) if self.stable_servo else (0.,.23)
                self.distance=np.clip(self.distance+speed*dt,*limits)
            else:
                self.distance+=np.where(self.touched,0.,.06*dt)
                self.distance=np.minimum(self.distance,.34)
            points=centers+normals*self.distance[:,None]
        elif local<self.segment_seconds-1.:
            points=centers+normals*self.distance[:,None]*(self.segment_seconds-1.-local)
        else:
            f=min(local-(self.segment_seconds-1.),1.);points=(1-f)*centers+f*high
        poses=np.stack([np.r_[point-R.apply(frame[1]),R.as_quat()]
                        for point,R,frame in zip(points,rotations,self.frames,strict=True)])
        if self.last_positions is None:
            velocity=np.zeros((2,6))
        else:
            linear=(poses[:,:3]-self.last_positions)/dt
            angular=np.stack([(a*b.inv()).as_rotvec()/dt for a,b in zip(rotations,self.last_rotations,strict=True)])
            velocity=np.concatenate((linear,angular),axis=1)
        self.last_positions=poses[:,:3].copy();self.last_rotations=rotations
        return poses,velocity,dict(phase=phase,phase_time=local,touched=self.touched.copy(),distance=self.distance.copy())


class ProbeScene(SupportScene):
    def __init__(self,gentle=False,canonical_init=False,stable_servo=False,**kwargs):
        super().__init__(**kwargs)
        self.probe=ProbeController(gentle=gentle,stable_servo=stable_servo)
        q=self.state_0.joint_q.numpy()
        poses,_,self.probe_record=self.probe.command(0.,np.zeros(2),.02)
        for start,pose in zip(self.q_starts,poses,strict=True): q[start:start+7]=pose
        obj_joint=next(j for j,b in enumerate(self.model.joint_child.numpy()) if b==self.box_body)
        start=int(self.model.joint_q_start.numpy()[obj_joint])
        obj=q[start:start+7]
        if canonical_init:
            # Fixed intrinsic-axis convention across anisotropic scales. This
            # sets an actual initial state, never a controller observation.
            base,_=load_sugar_outer_box()
            _,basis=np.linalg.eigh(np.cov(base.T))
            if np.linalg.det(basis)<0: basis[:,0]*=-1
            basis=basis[:,[1,2,0]]
            yaw=np.random.default_rng(kwargs.get('seed',0)).uniform(-.25,.25)
            R=Rotation.from_euler('z',yaw)*Rotation.from_matrix(basis.T)
            com=self.model.body_com.numpy()[self.box_body]
            world_com=Rotation.from_quat(obj[3:]).apply(com)+obj[:3]
            obj[:2]=world_com[:2]-R.apply(com)[:2];obj[3:]=R.as_quat()
        low=(Rotation.from_quat(obj[3:]).apply(self.box_verts)+obj[:3])[:,2].min()
        q[start+2]+=.003-low
        self.state_0.joint_q.assign(q)
        newton.eval_fk(self.model,self.state_0.joint_q,self.state_0.joint_qd,self.state_0)
        self.probe.last_positions=None
        self.frame=0

    def step(self,dt=.02,substeps=8):
        field=self.field.to_numpy()
        loads=np.array([(field['area'][(field['pad']//27)==side]*field['pressure'][(field['pad']//27)==side]).sum() for side in range(2)]) if self.frame else np.zeros(2)
        oldposes=self.state_0.body_q.numpy()[self.patch_frame].copy()
        target,velocity,self.probe_record=self.probe.command((self.frame+1)*dt,loads,dt)
        for i in range(substeps):
            self.time+=dt/substeps;f=(i+1)/substeps
            q=self.state_0.joint_q.numpy();qd=self.state_0.joint_qd.numpy()
            for side,(qs,qds) in enumerate(zip(self.q_starts,self.qd_starts,strict=True)):
                R=Slerp([0.,1.],Rotation.from_quat([oldposes[side,3:],target[side,3:]]))([f])[0]
                q[qs:qs+7]=np.r_[(1-f)*oldposes[side,:3]+f*target[side,:3],R.as_quat()]
                qd[qds:qds+6]=velocity[side]
            self.state_0.joint_q.assign(q);self.state_0.joint_qd.assign(qd)
            newton.eval_fk(self.model,self.state_0.joint_q,self.state_0.joint_qd,self.state_0,body_flag_filter=newton.BodyFlags.KINEMATIC)
            self.state_0.clear_forces();self.pipeline.collide(self.state_0,self.contacts)
            self.solver.step(self.state_0,self.state_1,self.control,self.contacts,dt/substeps)
            self.state_0,self.state_1=self.state_1,self.state_0
        self.solver.update_contacts(self.contacts,self.state_0)
        surface=self.pipeline.hydroelastic_sdf.get_contact_surface()
        self.tactile.update(self.state_0,self.contacts,contact_surface=surface)
        self.field.update(self.state_0,surface)
        self.frame+=1
