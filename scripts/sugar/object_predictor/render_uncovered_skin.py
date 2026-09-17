"""Saved contact locations on full hand meshes plus exact footprint projections."""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
import pyrender
import trimesh

from sugar_newton.hand.patches import patch_footprints
from .render_device import configure_retained_egl
from .render_object_state import World, load, true_state, canvas, text, mat, PW, PH, material

COLORS = [(0, 183, 176), (237, 49, 136), (239, 158, 36)]


def main():
    if not os.environ.get('SLURM_STEP_ID'): raise RuntimeError('Use retained compute step')
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); args = ap.parse_args()
    root = Path(args.output); p = json.loads((root/'PROTOCOL.json').read_text())
    configure_retained_egl(); world = World(); renderer = pyrender.OffscreenRenderer(PW, PH)
    markers = [pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1, radius=.00035),
               material=material(c)) for c in COLORS]
    records = []
    try:
        for ep in p['visual_episodes']:
            a = load(Path(p['source'])/'cases'/f'episode_{ep}'/f'episode_{ep}.npz')
            d = load(root/f'points_{ep}.npz'); frame = p['visual_frame']; poses = a['hand_pose_w'][frame]
            rot = Rotation.from_quat(poses[:, 3:]); positions = np.empty_like(d['position_hand_frame_m'])
            for h in (0, 1):
                ix = d['hand']==h
                positions[ix] = rot[h].apply(d['position_hand_frame_m'][ix])+poses[h, :3]
            c, q, dims, mass = true_state(a, frame)
            im = canvas('掌侧触觉漏测位置：原始区域边界之外的接触',
                        f'案例 {ep} | {a["timestamp_s"][frame]:.2f} 秒 | 青色：已覆盖，粉色：掌侧未覆盖，橙色：另一半空间 | 验证诊断',
                        ['实际双手与物体', '左手接触近景／隐藏物体', '右手接触近景／隐藏物体'])
            for panel in range(3):
                world.set(poses, c, q, dims, a['contact_position_w'][frame], np.zeros(54))
                if panel == 0:
                    world.camera_at(c+[.7, -1.2, .5], c)
                    selected = np.ones(len(positions), bool)
                else:
                    h = panel-1; selected = d['hand']==h
                    local = d['position_hand_frame_m'][selected]
                    target_local = (local.min(0)+local.max(0))*.5
                    target = rot[h].apply(target_local)+poses[h, :3]
                    span = max(.016, float(np.ptp(local, axis=0).max()))
                    offset = np.array([.15, -1. if h==0 else 1., .15])*max(.05, span*2.2)
                    eye = target+rot[h].apply(offset); eye[2] = max(.04, eye[2])
                    world.camera_at(eye, target)
                    world.scene.set_pose(world.object_node, mat([0, 0, -20]))
                    for name, node, _ in world.robot:
                        if name.startswith('right' if h==0 else 'left'): world.scene.set_pose(node, mat([0, 0, -20]))
                nodes = [world.scene.add(markers[int(cat)], pose=mat(pos)) for pos, cat in zip(positions[selected], d['category'][selected])]
                rgb, _ = renderer.render(world.scene, flags=0)
                if rgb.std()<5: raise RuntimeError('Constant RGB rendering')
                im.paste(Image.fromarray(rgb), (8+panel*532, 144))
                for node in nodes: world.scene.remove_node(node)
                draw = ImageDraw.Draw(im); x=24+panel*532
                if panel==0:
                    text(draw, (x, 755), f'实际离地 {a["validation_full_mesh_min_z_m"][frame]*100:.2f} cm', 25)
                    text(draw, (x, 798), f'真实质量 {mass:.3f} kg', 25)
                else:
                    counts = [int(((d['category']==k)&selected).sum()) for k in range(3)]
                    text(draw, (x, 755), f'覆盖／掌侧缺口／另一侧：{counts[0]}/{counts[1]}/{counts[2]} 面', 22)
                    ix = selected & (d['category']==1)
                    if ix.any(): text(draw, (x, 798), f'缺口点距区域边缘 {d["distance_m"][ix].min()*1000:.2f}–{d["distance_m"][ix].max()*1000:.2f} mm', 22)
                    text(draw, (x, 840), '距离是手坐标XZ投影，不是皮肤测地距离', 18)
            text(ImageDraw.Draw(im), (24, 925), '所有接触点来自同一保存帧，未移动；显示球半径0.35mm，不代表接触面积。未覆盖点仅作验证，没有送入模型。', 18)
            image=f'contact_locations_{ep}.png'; im.save(root/image)
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            for h, ax in enumerate(axes):
                for fp in patch_footprints():
                    x, z, hx, hz, theta = fp
                    corners = np.array([[-hx,-hz],[hx,-hz],[hx,hz],[-hx,hz]])
                    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                    ax.add_patch(Polygon((corners@rotation.T+[x,z])*1000, fill=False, edgecolor='gray', linewidth=.6))
                for cat, name in enumerate(('assigned','palmar outside footprints','opposite halfspace')):
                    ix = (d['hand']==h)&(d['category']==cat)
                    ax.scatter(d['position_hand_frame_m'][ix,0]*1000,d['position_hand_frame_m'][ix,2]*1000,
                               c=[np.array(COLORS[cat])/255],s=8,label=name)
                ax.set_aspect('equal');ax.set_xlim(-8,135);ax.set_ylim(-48,75)
                ax.set_xlabel('Hand local X (mm)');ax.set_ylabel('Hand local Z (mm)');ax.set_title(f'{ep} / {"left" if h==0 else "right"} / 47.64s')
            axes[0].legend(fontsize=8);fig.suptitle('Exact original27 footprints and saved contact face centroids; no sensor expansion')
            fig.tight_layout(); projection=f'footprints_{ep}.png';fig.savefig(root/projection,dpi=150);plt.close(fig)
            records.append(dict(episode=ep,frame=frame,mesh_image=image,footprint_image=projection,faces=len(positions)))
    finally: renderer.delete()
    (root/'RENDER.json').write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')
    print('UNCOVERED_SKIN_RENDER_COMPLETE',flush=True)


if __name__ == '__main__': main()
