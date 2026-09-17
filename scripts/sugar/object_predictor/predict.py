"""Causal runtime interface for a trained full-Utonia object predictor."""
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .data import OBSERVATION_KEYS,encode_observations,collate,history_indices
from .model import ObjectPredictor,rotation_matrix


class ObjectStateEstimator:
    def __init__(self,endpoint,base_checkpoint,device='cuda',mass_log_bias=0.):
        payload=torch.load(Path(endpoint),map_location='cpu',weights_only=False)
        protocol=payload['protocol']
        self.history=int(protocol['history'])
        self.mode=protocol['mode']
        self.mass_log_bias=float(mass_log_bias)
        self.history_policy=protocol.get('history_policy','contiguous')
        self.time_scale_s=protocol.get('time_scale_s',1.)
        self.normal_policy=protocol.get('normal_policy','stored')
        if self.normal_policy=='hand_surface':
            from .hand_surface_normals import verify_geometry_signature
            verify_geometry_signature(protocol.get('normal_geometry_signature'))
        self.frames=deque(maxlen=self.history if self.history_policy=='contiguous' else None)
        self.device=torch.device(device)
        self.model=ObjectPredictor(base_checkpoint,self.history,protocol.get('deterministic_pooling',False)).to(self.device)
        self.model.load_state_dict(payload['model'],strict=True)
        self.model.eval()

    def reset(self):
        self.frames.clear()

    def _history_observations(self):
        selected=history_indices(len(self.frames)-1,self.history,self.history_policy)
        return {k:np.stack([self.frames[int(i)][k] for i in selected]) for k in OBSERVATION_KEYS}

    @torch.inference_mode()
    def update(self,frame):
        missing=set(OBSERVATION_KEYS)-set(frame)
        if missing:
            raise ValueError(f'Missing sensor observations: {sorted(missing)}')
        record={k:np.asarray(frame[k]).copy() for k in OBSERVATION_KEYS}
        if not all(np.isfinite(v).all() for v in record.values()):
            raise ValueError('Nonfinite sensor input')
        if self.frames and float(record['timestamp_s']) <= float(self.frames[-1]['timestamp_s']):
            raise ValueError('Timestamps must increase; reset the estimator between episodes')
        self.frames.append(record)
        if len(self.frames)<self.history:
            return None
        obs=self._history_observations()
        row=encode_observations(obs,self.mode,time_scale_s=self.time_scale_s,normal_policy=self.normal_policy)
        # Collation metadata is discarded by the network and carries no labels.
        row.update(target=np.zeros(13,np.float32),episode=0,frame=0,
                   contact=bool((obs['normal_load_n']>1e-3).any()))
        batch={k:v.to(self.device) for k,v in collate([row]).items()}
        pred=self.model(batch)[0]
        Rh=Rotation.from_quat(record['hand_pose_w'][0,3:]).as_matrix()
        relative_R=rotation_matrix(pred[3:9]).cpu().numpy()
        center_relative=pred[:3].cpu().numpy()
        return dict(center_relative_left_hand_m=center_relative,
                    center_w_m=Rh@center_relative+record['hand_pose_w'][0,:3],
                    object_rotation_w_xyzw=Rotation.from_matrix(Rh@relative_R).as_quat(),
                    dimensions_m=pred[9:12].exp().cpu().numpy(),
                    mass_kg=float((pred[12]+self.mass_log_bias).exp()),
                    contact_observed=row['contact'],timestamp_s=float(record['timestamp_s']))
