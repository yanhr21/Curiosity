import numpy as np
import trimesh
from pxr import Usd, UsdGeom

D = ('/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew'
     '/robot_baby/Curiosity/SUGAR/descriptions/')
for side in ("left", "right"):
    m = trimesh.load(D + f"robots/g1/meshes/{side}_rubber_hand.STL", force="mesh")
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    print(f"{side}_rubber_hand: verts {len(v)} faces {len(f)} watertight={m.is_watertight} "
          f"volume={m.volume:.6f} m^3")
    print(f"   bounds x {v[:,0].min():.4f}..{v[:,0].max():.4f}  "
          f"y {v[:,1].min():.4f}..{v[:,1].max():.4f}  z {v[:,2].min():.4f}..{v[:,2].max():.4f}")

st = Usd.Stage.Open(D + "objects/small_box/obj_aligned.usd")
for p in st.TraverseAll():
    if not p.IsA(UsdGeom.Mesh):
        continue
    g = UsdGeom.Mesh(p)
    pts = g.GetPointsAttr().Get()
    if not pts:
        continue
    v = np.array([[q[0], q[1], q[2]] for q in pts], dtype=float)
    counts = np.array(g.GetFaceVertexCountsAttr().Get())
    idx = np.array(g.GetFaceVertexIndicesAttr().Get())
    tris, o = [], 0
    for c in counts:
        fv = idx[o:o + c]; o += c
        for k in range(1, c - 1):
            tris.append([fv[0], fv[k], fv[k + 1]])
    tris = np.array(tris)
    mb = trimesh.Trimesh(vertices=v, faces=tris, process=False)
    lo, hi = v.min(0), v.max(0)
    print(f"small_box: verts {len(v)} tris {len(tris)} watertight={mb.is_watertight} "
          f"volume={mb.volume:.5f} bbox={np.prod(hi-lo):.5f} fill={mb.volume/np.prod(hi-lo):.3f}")
    print(f"   half {(hi-lo)/2}  centre {(hi+lo)/2}")
    break
