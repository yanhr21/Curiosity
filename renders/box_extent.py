from pxr import Usd, UsdGeom
import numpy as np
import pickle

D = ('/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew'
     '/robot_baby/Curiosity/SUGAR/')
clip = pickle.load(open(D + "data/CarryBox/data_000/obj_motion_global_50hz.pkl", "rb"))
R = np.asarray(clip["obj_rot"][0], dtype=float)
z0 = float(clip["obj_trans"][0][2])
print(f"clip data_000: obj z at rest {z0:.4f} m, hands close to 0.280 m apart\n")
print(f"{'object':<10} {'size (mm)':<26} {'centre offset (mm)':<22} {'support':<9} {'z0-support'}")
for obj in ("small_box", "big_box", "bottle", "chair"):
    for name in ("obj_aligned.usd", "Props/instanceable_meshes.usd"):
        st = Usd.Stage.Open(D + f"descriptions/objects/{obj}/{name}")
        got = False
        for p in st.TraverseAll():
            if not p.IsA(UsdGeom.Mesh):
                continue
            pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not pts:
                continue
            v = np.array([[q[0], q[1], q[2]] for q in pts], dtype=float)
            size = (v.max(0) - v.min(0)) * 1e3
            centre = ((v.max(0) + v.min(0)) / 2) * 1e3
            support = -float((v @ R.T)[:, 2].min())
            print(f"{obj:<10} {str(size.round(1)):<26} {str(centre.round(1)):<22} "
                  f"{support:8.4f}  {z0 - support:+.4f}")
            got = True
            break
        if got:
            break
