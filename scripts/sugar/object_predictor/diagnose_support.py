"""Bounded physical contact audit; no predictor or optimizer, no fitted labels."""
import json
import os
from pathlib import Path
import numpy as np


def main(output):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Compute step required')
    import warp as wp
    from .support_scene import SupportScene
    wp.init()
    out=Path(output);out.mkdir(exist_ok=False)
    scene=SupportScene(mass=.35,seed=310017)
    for frame in range(51):
        scene.step()
        if frame not in (0,10,25,40,50):
            continue
        c=scene.contacts
        n=int(c.rigid_contact_count.numpy()[0])
        arrays={name:getattr(c,name).numpy()[:n] for name in (
            'force','rigid_contact_shape0','rigid_contact_shape1',
            'rigid_contact_point0','rigid_contact_point1','rigid_contact_normal')}
        arrays.update(body_q=scene.state_0.body_q.numpy(),body_qd=scene.state_0.body_qd.numpy(),
                      body_com=scene.model.body_com.numpy(),body_mass=scene.model.body_mass.numpy(),
                      shape_body=scene.model.shape_body.numpy(),object_vertices=scene.box_verts,
                      patch_shapes=np.asarray(scene.patch_shapes),object_shape=np.asarray(scene.box_shape),
                      tactile_normal=scene.tactile.normal_vec.numpy(),
                      tactile_friction=scene.tactile.friction_vec.numpy(),
                      tactile_normal_load=scene.tactile.normal_load.numpy())
        arrays.update({'surface_'+k:v for k,v in scene.field.to_numpy().items()})
        np.savez_compressed(out/f'frame_{frame:03d}.npz',**arrays)
        print(json.dumps(dict(frame=frame,contacts=n,normal_load=arrays['tactile_normal_load'].tolist(),
                              normal_vec=arrays['tactile_normal'].tolist(),friction_vec=arrays['tactile_friction'].tolist())),flush=True)


if __name__=='__main__':
    import sys
    main(sys.argv[1])
