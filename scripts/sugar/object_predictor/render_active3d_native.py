"""Render all 48 fixed native-data objects and their actual frozen predictions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def render(root):
    if not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Retained explicit compute step required")
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    from PIL import Image, ImageDraw
    import pyrender
    import trimesh
    from .render_device import configure_retained_egl
    from .render_object_state import canvas, text, material, look, PW, PH, BG, TEAL, AMBER

    result = json.loads((root / "RESULT.json").read_text())
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    if not result["complete"] or len(result["cases"]) != 240:
        raise RuntimeError("All 48 objects and all five repetitions must finish before rendering")
    rows = [row for row in result["cases"] if row["repeat"] == 0]
    if len(rows) != 48 or len({r["object_id"] for r in rows}) != 48:
        raise RuntimeError("Render every fixed object, with no outcome selection")
    native = Path(protocol["dataset"]) / "selected48/object_data/object_info"
    out = root / "renders"
    out.mkdir(exist_ok=False)
    device = configure_retained_egl()
    renderer = pyrender.OffscreenRenderer(PW, PH)
    images = []
    try:
        for row in rows:
            obj = row["object_id"]
            truth_v = np.load(native / f"{obj}_verts.npy", allow_pickle=False)
            truth_f = np.load(native / f"{obj}_faces.npy", allow_pickle=False)
            with np.load(root / row["output_file"], allow_pickle=False) as saved:
                pred = {key: saved[key] for key in saved.files}
            im = canvas("官方 Active3D · 原生接触条件下的形状重建",
                        f'{row["split"]} / ABC {obj} | 固定 repeat 0 | 同一完整冻结模型 | 原始 canonical 坐标',
                        ["真实完整网格", "零次触摸尝试", "五次触摸尝试"])
            for panel, arm in enumerate((None, "attempts0", "attempts5")):
                vertices = truth_v if arm is None else pred[arm + "_vertices"][:1824]
                faces = truth_f if arm is None else pred["global_faces"]
                if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
                    raise RuntimeError(f"Invalid saved mesh {obj}/{arm}")
                scene = pyrender.Scene(bg_color=[*np.asarray(BG) / 255, 1], ambient_light=[.45, .45, .45])
                shader = material(TEAL if arm is None else AMBER)
                shader.doubleSided = True
                scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, faces, process=False),
                          material=shader, smooth=False))
                target = np.zeros(3)
                for eye, intensity in (([.6, -.8, .9], 2.2), ([-.6, .5, .5], 1.2)):
                    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                              pose=look(np.asarray(eye), target))
                scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(43), znear=.01, zfar=10.),
                          pose=look(np.array([.58, -.78, .52]), target))
                rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
                if rgb.std() < 5 or not np.isfinite(depth).all() or np.count_nonzero(depth) < 100:
                    raise RuntimeError(f"Empty or invalid actual mesh render {obj}/{arm}")
                im.paste(Image.fromarray(rgb), (8 + panel * 532, 144))
                draw = ImageDraw.Draw(im)
                x = 24 + panel * 532
                if arm is None:
                    text(draw, (x, 752), f'完整资产 {len(truth_v)} 顶点 / {len(truth_f)} 面', 22)
                    text(draw, (x, 797), "原生 30000 点标签监督／评价", 22)
                    text(draw, (x, 842), "未配准、未按预测重新缩放", 22)
                else:
                    score = row["metrics"][arm]
                    text(draw, (x, 752), f'官方全表面 CD×9000：{score["official_all_faces"]:.3f}', 22)
                    text(draw, (x, 797), f'全局表面辅助：{score["auxiliary_global_only"]:.3f}', 22)
                    text(draw, (x, 842), f'触摸顶点 mask0/1/2：{"/".join(str(score["mask_vertex_counts"][str(m)]) for m in (0,1,2))}', 19)
                text(draw, (x, 879), "显示完整真值" if arm is None else "显示全局预测表面：1824 顶点 / 2304 面", 18)
            text(ImageDraw.Draw(im), (24, 920), "原生静态形状对照，非 SUGAR 压力输入或搬运结果。五次尝试不保证五次接触；全表面主指标另含固定触摸 chart。", 16)
            name = f'{row["split"]}_{obj}.png'
            im.save(out / name)
            images.append(name)
            print("NATIVE_REFERENCE_RENDER", len(images), name, flush=True)
    finally:
        renderer.delete()
    (out / "RESULT.json").write_text(json.dumps(dict(complete=True, actual_mesh_stills=48,
        images=images, selection="repeat0 all48, fixed before scores", renderer=device), indent=2) + "\n")
    html = ['<!doctype html><meta charset="utf-8"><title>官方原生形状重建</title>',
            '<style>body{max-width:1600px;margin:24px auto;background:#f2f6f9;color:#172e40;font:18px/1.6 system-ui}img{width:100%}</style>',
            '<h1>固定48物体：零次／五次触摸尝试</h1>',
            '<p>全部48物体的固定第0组动作；完整官方冻结模型。原生静态数据，非SUGAR触觉或搬运估计。</p>',
            '<p><a href="REPORT.md">五组动作完整统计</a> · <a href="RESULT.json">全部数值</a></p>']
    for name in images:
        html.append(f'<h2>{name[:-4]}</h2><img loading="lazy" src="renders/{name}">')
    (root / "index.html").write_text("\n".join(html) + "\n")
    report = root / "REPORT.md"
    report.write_text(report.read_text().replace("实际48张网格渲染尚待独立render阶段。",
                      "全部48张实际网格对照渲染已生成，固定每对象repeat0；尚不代表人工逐张查看。[打开完整画廊](index.html)。"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    render(parser.parse_args().root)
