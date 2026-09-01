# Newton-native RL for CarryBox

> **Corrections, 2026-08-31.** Three silent-truncation-class bugs were found and fixed, and
> they invalidate the numbers in the older sections below. Everything here was measured on a
> dedicated 8xA100 node; timings come from serial runs, because eight concurrent rollouts
> contend for CPU on the NumPy observation and add a ~20 % spread to fps.
>
> 1. **The box weighed 4.39 kg instead of 0.5 kg** (8.78x). `ShapeConfig.density` defaults
>    to 1000 kg/m^3 and `add_shape_mesh` *adds* the shape's mass to the body, so
>    `add_body(mass=0.5)` plus a default-density mesh silently added 3.89 dm^3 of water. The
>    robot's own links were unaffected, because the URDF importer overwrites the accumulated
>    mass with the URDF inertial -- which is why the symptom looked like a contact problem
>    localised to the wrists. Fixed by deriving density from the asset mass and the mesh
>    volume, which also scales the inertia consistently, as Isaac does. Verify with
>    `python -m sugar_newton.validation.check_masses`.
> 2. **`njmax` was 2048 while the playback path used 16384.** It caps constraint *rows*, and
>    an elliptic cone costs several rows per contact, so it has to scale with `nconmax`. The
>    2048 came from overflow messages logged while `nconmax` was still truncating at 1024 --
>    sizing one limit from measurements taken while another was clipping.
> 3. **Hydroelastic buffers were dropping contacts at low SDF resolution.** The iso buffer is
>    sized as `buffer_mult_iso * total_num_tiles`, so it shrinks with `sdf_resolution` while
>    the grip's demand does not: at resolution 32 the grasp asked for 1280 L1 subblocks
>    against 960 and lost the rest, 1596 times in one rollout. `buffer_mult_iso=4` clears it.
>
> With the mass fixed, `tracker.pt` transfers far better than previously reported --
> two thirds of the reference lift rather than one third, consistently across three clips:
>
>     clip      lift    reference   fraction
>     data_000  0.430   0.628       68 %
>     data_001  0.458   0.692       66 %
>     data_005  0.460   0.643       72 %
>
> 4. **The triangle-pair buffer was overflowing too**, and this one does not have a fix by
>    sizing. See "Throughput is the open problem" below.
>
> The retracted claim: wrist saturation was never a Newton-vs-Isaac contact discrepancy. The
> wrists were at their limit because they were holding 8.8x the intended mass.

## Throughput is the open problem, and the cause is now identified

Measured with `python -m sugar_newton.rl.bench_env`, which reports peak contacts against the
limits alongside the timing, because a fast row that was silently dropping contacts is not a
result. `env-steps/s` is aggregate across worlds.

    collision   envs   step_ms   env-steps/s   peak contacts
    mesh           1      91.0          11.0             495
    mesh           4     365.9          10.9           2 687
    mesh          16   1 155.0          13.9           6 742
    mesh          64   4 697.2          13.6           4 056   <- drops contacts
    hydro          1     529.7           1.9           2 892
    hydro          4   2 226.4           1.8          11 630
    hydro         16   6 065.0           2.6          11 354
    hydro         64  14 104.8           4.5           5 007

### It is collision detection, and that was measured rather than assumed

`--profile` wraps the env's own `pipeline.collide` and `solver.step` in synchronising timers,
so the phases are literally the calls the step makes:

    envs   collide   solve   obs+reward+python
       1    75.6 %  18.6 %              5.8 %
      16    94.5 %   4.5 %              1.1 %

Read the proportions, not the absolute instrumented times: the per-phase `wp.synchronize`
serialises work the GPU would overlap, which inflated the 1-world step from 61 ms to 146 ms.
At 16 worlds instrumented (989 ms) and real (1105 ms) agree closely, so that split is solid.

This rules out the alternative explanations directly:

* **Not rigging.** Model construction is 3-7 s (the larger figure is first-run kernel
  compilation) and happens once in `__init__`, outside every timed window.
* **Not rendering.** There is no viewer in this path at all. Rendering is expensive where it
  exists -- the 481-frame playback goes from 100 s to 310 s with `--render` -- but training
  never pays it.
* **Not the observation.** Obs, reward and Python bookkeeping are 5.8 % at one world and
  1.1 % at sixteen, because they are Torch ops on the GPU over a batch.
* **Not launch overhead.** If the step were bound by kernel launches, CUDA graph capture
  would have collapsed it; it bought ~10 %.
* **Not even the solver.** Solve is 4.5 % at 16 worlds. Cutting `iterations` from 100 to 10
  is consistent with this -- it barely moves the clock.

Collide's share *grows* with world count, 76 % to 94.5 %, which is precisely why batching
does not amortise anything: the dominant term scales linearly with the number of worlds.

Two things follow, and neither is what was assumed before:

* **Batching does not help.** Aggregate throughput is flat at ~11-14 env-steps/s from 1 to 16
  worlds. Adding worlds adds proportional cost, so the usual vectorisation argument does not
  apply to this scene as configured.
* **The raw triangle mesh is the ceiling, not the solver.** The box collides as a ~100k
  triangle mesh. Candidate triangle pairs scale with world count and reached 5.58e6 at 256
  worlds, but deterministic contact packing indexes contacts with 20 bits, so 2**20 is a hard
  cap: request more and the pipeline raises, request less and it drops contacts. Measured
  demand is ~19k pairs per world, so **the mesh path is correct up to ~55 worlds and wrong
  above it**, and the env now warns instead of failing. The 64-world row above runs, but its
  overflow log shows 1.06-1.31e6 against the 1.05e6 cap, so its physics is not trustworthy.
* **Hydroelastic is not the escape.** It handles the concavity properly and its cost does not
  depend on triangle count, but measured here it is 3-6x slower than the mesh path, not
  faster. It buys fidelity, not speed.

### The fix: decimate the box collider. 15x throughput, 1.6 mm of geometry given up

The box ships as 100k triangles of median 7.8 mm^2 tiling 1.2264 m^2 -- within 0.4 % of the
exact surface area of an open carton with its bounding box, and enclosing a 3.2 mm wall. So
the triangle budget is tessellation density, not shape, and quadric decimation gives it back
almost for free (`validation/check_decimation.py`, symmetric surface deviation):

    target   mean      p99       max        vs 5 mm margin / 3.2 mm wall
     5000    0.037 mm  0.205 mm  1.046 mm   safe
     2000    0.076 mm  0.428 mm  1.634 mm   safe, default
     1000    0.135 mm  0.865 mm  4.353 mm   approaches both
      200    0.624 mm  2.808 mm  6.077 mm   exceeds the margin

`--box-tris 2000` is a 50x triangle reduction whose worst case moves the surface by half the
wall thickness and a third of the contact margin. It is decimation, not hulling and not
decomposition: the mesh stays non-convex and the carton stays open, so the concavity the
grasp depends on is untouched.

What it buys, and what it costs:

    envs   box_tris   step_ms   env-steps/s   collide %
      16    100 000   1 052.9          15.2      94.9
      16      5 000     186.4          85.8      73.6
      16      2 000     153.1         104.5      61.8
      64      2 000     435.3         147.0
     256      2 000   1 369.2         187.0
     512      2 000   2 472.9         207.0

    playback (481 frames)   fps    lift     tracking
    100k triangles          4.8    0.4287   8.18 deg
    5000                   19.2    0.4131   8.10 deg
    2000                   21.5    0.4255   8.28 deg

Three things to note. Aggregate throughput **starts scaling with world count again**
(147 -> 207 from 64 to 512 worlds) because collide is no longer 95 % of the step. The
**2**20 ceiling stops binding**, so 512 worlds runs clean where 64 previously dropped
contacts. And the lift moves by -0.7 % against a run-to-run spread of 0.07 m, with joint
tracking unchanged, so the accuracy was not spent to buy the speed.

At 207 env-steps/s, SUGAR's ~5.8M env-steps is about 8 hours rather than about 5 days.

### The closed loop: policy -> physics -> render -> visual+tactile -> policy

Every other fps number in this file is physics-only. `validation/bench_loop.py` measures the
whole loop and attributes it, fencing each stage with `wp.synchronize()` so the split is real,
then re-timing unfenced for the total a training loop would actually see. Single world, A100,
`box_tris=2000 hand_tris=5000`, started in the carry (frame 180):

    resolution   policy  physics  render  readout  tactile   full ms   FULL fps   render fps
    no render      0.77    49.75       -        -     0.29     51.04       19.6            -
    320x240        0.76    49.86    4.56     0.71     0.38     56.17       17.8        189.8
    640x480        0.77    49.84    4.57     0.94     0.38     56.77       17.6        181.5
    1280x720       0.77    50.70    4.66     1.32     0.38     57.56       17.4        167.2
    1920x1080      0.80    49.66    4.62     2.06     0.38     57.45       17.4        149.7
    2560x1440      0.80    50.22    4.73     3.01     0.38     58.39       17.1        129.2
    3840x2160      0.76    49.26    4.55     5.73     0.38     60.87       16.4         97.3

**Resolution is nearly free.** The draw cost is flat at 4.55-4.73 ms across a 108x range in
pixel count, because this scene is draw-call and geometry bound, not fill-rate bound. Only
`get_frame()` scales with pixels (0.71 -> 5.73 ms), and sublinearly at that. Going from QVGA to
4K costs the full loop 8 %. So pick the resolution the policy needs and stop worrying about it.

**Physics is the wall, at 81-89 % of the loop.** Rendering adds 5-10 ms to a ~50 ms physics
step. The two sensing paths are nearly free: the tactile reduction (`PatchTactile`, all Warp
kernels) is 0.38 ms and the frame readout is a GPU->GPU PBO copy, so neither forces a host
round trip. Anything spent optimising this loop should go into the solver, not the renderer --
and note physics cost tracks contact count, so the same scene runs 30.0 ms during the approach
and 40-50 ms once the grasp closes.

`validation/make_loop_video.py` renders the scene beside the live tactile map for both hands,
flat on the palm, with all three rates burned in (`_out/loop_render_tactile.mp4`). The
compositing is deliberately outside the timers, since drawing a matplotlib panel per frame
costs more than the simulation does.

> **Open issue: the absolute contact-force scale is not trustworthy yet.** Summed contact-force
> magnitude on one hand swings between ~40 N and ~4000 N across frames of the same carry, and a
> per-hand *net* of 1242 N appeared in one probe -- non-physical for a 0.5 kg box. `PatchTactile`
> agrees with a hand-rolled sum over `contacts.force` to the digit (6770.41 N both), so the
> reducer is not the problem; the suspects are the force units after a multi-substep step and
> transient penetration spikes. All the decimation comparisons in this file are *relative*, run
> through one identical path with an exact A/A control, so they stand -- but do not quote these
> newtons as physical grip force until this is chased down.

### What decimation costs a tactile channel: net load survives, per-taxel pattern does not

Surface deviation is a geometric proxy, so the contact-level question was measured directly
(`validation/compare_contacts.py`). Comparing two *rollouts* would only measure chaos -- this
scene has a 0.07 m spread at identical parameters -- so instead one reference rollout is
recorded and every state is replayed into both colliders, making any difference attributable
to the collider alone. Probing two identical colliders is the control, and it agrees to 0.1 %,
so the numbers below are signal:

    vs original 100k   net load/hand   sum|f|   patch centroid   per-contact corr
    20 000 tris          1.4-2.0 %     8-15 %      1.6-2.0 mm       0.65-0.97
     2 000 tris          2.8-3.2 %    18-19 %      4.9-10.1 mm      0.48-0.80

**Most reported "contacts" are not touches.** Newton emits every shape pair inside the 5 mm
margin, and 95.8 % of those carry *exactly* zero force. Per hand per frame there are ~78-87
margin candidates but only **3.6-4.5 load-bearing contacts**. Two consequences worth carrying
around:

* Any contact *count* is a proximity statistic, not a contact-mechanics one. `compare_contacts`
  now prints `d_cand` and `d_load` separately for this reason.
* Patch centroid and spread must be **force-weighted**, or ~85 zero-force candidates dilute
  the answer. Doing it properly roughly doubles the measured shift (hands at 5000: 0.97 mm
  unweighted vs 1.92 mm weighted on the right, 2.15 vs 2.95 mm on the left).

The force-based metrics -- net load, `sum|f|`, correlation -- were never affected, since a
zero-force entry contributes nothing to them.

Read the rest as two different answers, not one bad one:

* **Net wrench is preserved.** Each hand puts 30.9 N of squeeze on the box, and that survives
  decimation to within 3 %. It is pinned by the physics -- the grasp has to hold the same
  0.5 kg -- which is why the lift and the tracking do not move.
* **The distribution of that wrench over contacts is not preserved.** A 6-DOF load spread
  across ~80 contacts per hand is underdetermined, so which triangles carry it is set by the
  tessellation and the solver's regularisation rather than by the shape. The giveaway that
  this is conditioning and not accuracy loss: per-contact correlation is *not monotone* in
  triangle count (0.65 at 20k, 0.80 at 2k on the left hand).

So decimation is safe for dynamics and for patch-integrated tactile readings, and unsafe for
per-taxel pressure patterns. Hence the split default: `box_tris=2000` in the RL env, where
throughput is the constraint, and the full 100k mesh in `validation/`, which is the fidelity
reference. Anyone training on per-taxel tactile input should raise `box_tris` and re-run this
comparison rather than inherit the RL default.

### The hands are the bigger collider, and decimating them adds 32 %

`check_geometry` reports the hands at 5.9-6.4k triangles, but that is the precomputed convex
hull in `HAND_HULLS`, used only by the `--hull-hands` ablation. The URDF actually collides
them as **45 748 and 43 852 triangles** -- 89.6k together, comparable to the box's 100k. On
top of `box_tris=2000`, `--hand-tris 2000` gives:

    envs   hand_tris   step_ms   env-steps/s   collide %
       1     45k/44k      46.7          21.4      19.4
       1       2 000      44.2          22.6      16.3
      16     45k/44k     136.0         117.6      63.6
      16       5 000     119.4         134.0
      16       2 000     117.0         136.7      59.6
      64     45k/44k     491.8         130.1      86.4
      64       5 000     345.9         185.0
      64       2 000     373.1         171.5      79.6

Note the 5000 and 2000 rows are tied within run-to-run variation (5000 even measures faster at
64 worlds). Almost all of the gain is already banked by the first 9x reduction, so pushing on
to 2000 buys no throughput while costing 5x the contact error -- see the fidelity sweep below.

The gain grows with world count (+5 % at 1 world, +32 % at 64) because it is a broad-phase
effect: hand triangles multiply against box triangles, so it compounds with the box reduction
instead of adding to it. Deviation is 0.13 mm mean / 0.89 mm max at 2000, roughly six times
tighter than the box at the same budget because the hand is a much smaller object; below ~300
triangles the max deviation crosses the 5 mm margin and enclosed volume drifts over 13 %.

Playback fidelity holds, which matters more here than for the box because this is the
grasping surface itself:

    box_tris=2000, playback   fps    lift      tracking      wrist sat (carry)
    hands 45k/44k            21.4   0.4067 m   0.1408 rad        12.0 %
    hands 10000              23.6   0.4029 m   0.1400 rad        12.0 %
    hands 5000               24.7   0.4080 m   0.1407 rad        12.9 %
    hands 2000               25.3   0.4018 m   0.1408 rad        11.7 %
    hands 1000               25.6   0.4148 m   0.1408 rad        13.3 %

Single-world fps is flat from 5000 down (24.7 vs 25.3), the same tie the 64-world numbers
show, and 5000 lands closest to the full-mesh lift of the four.

Lift moves within the 0.07 m run-to-run spread and mean joint tracking is identical to four
decimals, so `--hand-tris 2000` is free *for the dynamics*.

It is not free for contact fidelity, and 2000 turns out to be past a cliff. Sweeping the
budget against the contact probe (`--vary hand`):

    hand_tris (each)   net load/hand   sum|f|    patch centroid   per-contact corr
    10 000               1.1-1.3 %    2.3-2.6 %    0.8-0.9 mm        0.95
     5 000               1.8-2.2 %    3.8-4.6 %    1.0-2.1 mm        0.92-0.94
     2 000              10.1-10.7 %   12-14 %      1.7-2.7 mm        0.57-0.59   <-- cliff

**5000 per hand is the setting to use**, not 2000: a 9x reduction (89.6k -> 10k) for 2 % net
load error, and better tactile fidelity than the box gets at its own default (box 2000 is
3.1 % net at 0.37-0.75 correlation). Unlike the box, the hand degrades *monotonically* in
triangle count, so it can be traded off predictably instead of reshuffling.

Two things are worth extracting from this. First, surface deviation is not the right
predictor: at 2000 the hands have **half the box's surface deviation and three times its
net-load error**. The reason is visible in `_out/hand_tactile_*.png` -- this grasp is a
fingertip pinch and only 8-14 % of the hand is ever touched, so the hand *is* the contact
patch and its local curvature sets contact area directly, whereas the gripped part of the box
is a flat panel whose curvature retessellation does not change. A mesh-averaged deviation is
dominated by the untouched palm and cannot see this.

Second, the safe budget has to be set by curvature in the contact region, not by a global
triangle count. That is why the box's 2000 does not transfer: it was tuned to a five-panel
carton. Genuinely concave objects should be swept with this probe rather than inheriting it.

`--hand-tris` is off by default in both the RL env and validation, since it has not yet been
through a training run. When you turn it on, use **5000**: it costs nothing in throughput
against 2000 and gives 5x lower contact error, so 2000 is dominated and there is no reason to
prefer it.

### Visualising both, `validation/plot_hand_tactile.py`

Everything is drawn on the flat palm layout in `validation/hand_atlas.py`: the real G1
collider (`meshes/{left,right}_rubber_hand.STL` out of
`g1_29dof_rev_1_0_with_rubber_hand.urdf`, the fixed five-finger rubber hand, 45748 and 43852
triangles) projected orthographically down its own palm normal, fingertips up, digits
labelled. The projection is free of information loss here because a box grasp puts **100.0 %
of its force on one side of the hand**, both hands.

Three independent tests agree on which side that is, and they matter because a closed mesh
does not announce which of its faces are "the palm":

| test | left hand | right hand |
| --- | --- | --- |
| concave side (gap to convex hull, thumb excluded) | -y, 13.0 mm vs 1.2 mm | +y, 12.9 mm vs 1.2 mm |
| direction all four fingers curl (tip deflection off a straight base fit) | -y, 34-42 mm | +y, 32-42 mm |
| side the grasp load lands on | -y, 100.0 % | +y, 100.0 % |

With fingers at local +x and pinky-to-thumb at local +z, that gives `f x p == -n` for the
right hand and `+n` for the left, the correct chirality for each -- the two STLs are a proper
mirrored pair and are not swapped.

    python -m sugar_newton.validation.compare_contacts --vary hand --variants 0 0 2000 \
        --dump sugar_newton/_out/tactile_hand.npz
    python -m sugar_newton.validation.plot_hand_tactile --hand both \
        --dump sugar_newton/_out/tactile_hand.npz

`hand_shape_*.png` puts the colliders, cross-section outlines through the fingers and palm, a
deviation heat map and a deviation CDF against the 5 mm margin on one sheet. The
cross-sections are the panel worth looking at: at 2000 triangles the finger outlines are
coincident with the original at plotting scale, which is why the geometric check passes.

`hand_tactile_flat.png` is the tactile map for both hands -- accumulated contact force over
the carry, splatted with a 4 mm kernel onto a common canvas mesh so the two colliders can be
differenced (contact points are reported in the hand's body frame, `contacts.py:210`, so this
is well defined even though the two meshes differ). It shows the same digits loaded in both
cases with the same rank order, but 16.5 % (left) and 18.0 % (right) of the total force
redistributed between them, peaking at 22-30 N on a single fingertip. That is the honest
summary of hand decimation: the grasp is qualitatively the same, the per-fingertip force is
not.

Splatting only ever uses load-bearing contacts. About 94 % of what Newton reports as a
contact carries exactly zero force, and those proximity candidates sit up to 100 mm off the
surface; the load-bearing ones all sit 2.1-2.6 mm off it, half the 5 mm margin.

For the animated version, the dump needs per-contact frame indices, so it is a separate
artifact from the integrated one:

    python -m sugar_newton.validation.compare_contacts --vary hand --variants 0 5000 \
        --window 190 360 --dump sugar_newton/_out/tactile_frames.npz
    python -m sugar_newton.validation.plot_hand_tactile --hand both --only video \
        --stride 2 --fps 12 --dump sugar_newton/_out/tactile_frames.npz

That writes `hand_tactile_flat.mp4`, both hands with the original collider beside the
5000-triangle one, and you can watch the load walk across the fingertips as the box settles
while each pair of panels stays locked together. Two gotchas: the container has no ffmpeg for
matplotlib's writer so it falls back to GIF, and the video's colour scale is per-frame force
(peak 2.8 N) whereas the static figure's is integrated over the whole carry (peak ~150 N) --
they are not comparable numbers.

### The grasp was floating on a 4.5 mm air gap. `margin` is now 0

Inspecting the exported frame (`validation/export_frame.py`, writes world-space PLYs plus a
per-digit audit) showed the loaded fingertips **not touching the box**: signed distance from
the nearest fingertip vertex to the box surface was **+4.0 to +4.7 mm** while they carried
28-45 N. The cause was `default_shape_cfg.margin = 0.005`, and the name is misleading enough
to be worth spelling out.

Newton splits the two things PhysX calls `restOffset` and `contactOffset` into two fields, and
we had been using the wrong one:

* **`margin`** (`ShapeConfig.margin`, library default **0.0**) is an *outward offset of the
  surface*. The separation the solver constrains is

      sep = dot(n, p1_world - p0_world) - (margin0 + margin1)      # sim/contacts.py:65

  so a pair reaches `sep = 0` while the drawn surfaces are still `margin0 + margin1` apart,
  and rests there. This is `restOffset`.
* **`gap`** (`ShapeConfig.gap`, default `builder.rigid_gap = 0.1`) is the detection distance:
  broad phase expands AABBs by `margin + gap`. This is `contactOffset`.

So earliness of detection never depended on `margin` at all -- there is already a 10 cm
broad-phase band -- and the 5 mm was pure geometric inflation. It also has no mass
side-effect to trade away: `margin` only enters `compute_inertia_shape` for hollow shapes
(`is_solid=False`), and ours are solid.

It was never chosen for this scene either. `git log -S` puts it in `5dc90dc`, the Allegro
tactile scene, from where it was carried into the G1 carry unexamined.

Measured on `data_000`, decimated colliders, one clip each:

| collider margin | fingertip gap to box | solver `sep` | load-bearing contacts | box lift | playback |
| --- | --- | --- | --- | --- | --- |
| 5 mm (was default) | +4.0 to +4.7 mm | -0.2 to -1.0 mm | 4-5 | 0.408 m | 24.5 fps |
| 1 mm | +0.5 to +0.9 mm | -0.2 to -1.1 mm | 4-8 | 0.439 m | 25.1 fps |
| **0 (now default)** | **-0.0 to +0.9 mm** | -0.04 to -4.3 mm | 27 | **0.433 m** | 25.2 fps |

Reference lift is 0.628 m, so this moves the reproduction from 65 % to 69 % of it. Zero and
1 mm are indistinguishable on lift; 0 is preferred because the collider is then exactly the
asset surface with nothing to tune. Throughput is unchanged (a 64-env `CarryBoxEnv` smoke
test came out 85 vs 102 env-steps/s the other way round, i.e. inside run-to-run noise), and
neither setting diverged over 150 steps.

    python -m sugar_newton.validation.export_frame --frame 359 --margin 0.005   # the old behaviour

Two things this does *not* fix. It leaves most of the lift shortfall in place, which is
consistent with the diagnosis further up -- the observation carries no absolute reference
position, so position error is uncorrectable by construction. And contact forces roughly
double (54-89 N peaks against 28-45 N) because the solver now sees real penetration rather
than margin-offset penetration, which feeds directly into the still-open question of why the
absolute force scale moves by two orders of magnitude between probes.

The non-loaded digits are genuinely clear, so "only two digits carry the box" was correct
rather than a sensing miss: middle 6-12 mm, ring 8-24 mm, pinky 12-37 mm. Eyeballing the
whole hand against the box in `scene.ply` reads as ~2 cm because most of the hand is that far
away; only the loaded fingertips are anywhere near the surface.

**Open, and it follows directly from this:** the decimation analysis above uses "the 5 mm
contact margin" as its geometric error budget -- that is where `box_tris=2000` and
`hand_tris=5000` come from, and the figures from `plot_hand_tactile.py` still draw a 5 mm
line. With `margin = 0` that budget has lost its stated justification and needs re-deriving
from something physical instead (the 3.2 mm carton wall, or a target contact-position error).
The *conclusions* are probably safe, because `compare_contacts.py` measured contact accuracy
directly rather than inferring it from surface deviation, but the tolerance those scripts
print and plot is now a leftover rather than a derived number.

Convex decomposition of the box is *not* the next lever -- coacd returns
52 hulls at threshold 0.1 and 112 at 0.05, because it is fighting tessellation noise rather
than real concavity, and it costs 30 s to build. Single-world playback speed, for reference,
is 4.8 fps eager and 5.3 fps with CUDA graph capture (`validation/g1_carrybox_policy.py
--graph`); capture is worth ~10 % here because disabling conditional graph nodes, which
driver 12.2 forces, also disables the solver's early exit.

Retrain (or fine-tune) SUGAR's tracker against Newton's contact model, because
`validation/g1_carrybox_policy.py` shows the official `tracker.pt` only partially
transfers: it reaches about two thirds of the reference lift height.

    # smoke test
    python -m sugar_newton.rl.train_bcppo --num-envs 16 --max-iterations 3 \
        --clips data_000 --logger tensorboard

    # training, logging to wandb
    python -m sugar_newton.rl.train_bcppo --num-envs 512 --max-iterations 30001 \
        --wandb-project sugar_newton --run-name carrybox_bcppo

Run inside the container, via the dev node:

    sbatch slurm/devnode.sbatch                        # from the repo root
    bash slurm/devrun.sh "source env/activate.sh && python -m sugar_newton.rl.train_bcppo ..."

See `SETUP.md` for the conda env these commands assume.

## The algorithm is SUGAR's, imported not reimplemented

`train_bcppo.py` imports `BCPPO` from `SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py`
and runs it inside `rsl_rl`'s own `OnPolicyRunner`, with the hyperparameters transcribed
from `BCPPORunnerCfg`. `BCPPO` is registered by the same mechanism SUGAR uses
(`setattr(builtins, "BCPPO", ...)`, `scripts/sugar_rl/train.py:147-150`), because the
runner resolves the algorithm with `eval(alg_cfg["class_name"])`.

The only local code in the training loop is `vec_env.py`, which presents the Newton
environment as an `rsl_rl.env.VecEnv` with the three observation groups the config asks
for:

    policy   510-D   validated against Isaac's recorded actions to RMSE 0.088
    critic   890-D   obs_890.py
    teacher  890-D   obs_890.py -- what the frozen refiner is asked to imitate

BCPPO's curriculum, for reference:

    stage 1   step < 500          loss = distill                    (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value    (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

The teacher checkpoint is required, not optional: `BCPPO.__init__` asserts on a missing
one, and stages 1-2 have no loss without it. Default path is the recovered
`refiner_model10000.pt` (see TODO 16).

An earlier version of this directory carried a hand-written PPO (`ppo.py`, `train.py`).
It has been deleted. It was stage 3 with the distillation dropped, which is not the
algorithm SUGAR trains the tracker with.

## Logging

`rsl_rl` has native wandb support, so nothing here writes to wandb directly: the runner
config sets `logger: wandb` and `wandb_project`, and the run name is the log directory's
basename. Credentials follow the convention used elsewhere in this workspace --
`WANDB_API_KEY` from the environment, else `~/.netrc` for `api.wandb.ai` -- and
`train_bcppo.py` checks for one before building the environment, so a run cannot get
several minutes in with logging silently off. Pass `--logger tensorboard` to opt out.

## STATUS: BCPPO trains on Newton and logs to wandb

First working run, 8 worlds, 3 iterations, `data_000`
(https://wandb.ai/nvr-amri/sugar_newton/runs/acs79r73)::

    iter 0   reward  9.48   ep_len 10.75   noise std 0.51   diverged 0
    iter 1   reward  7.54   ep_len 11.19                    diverged 0
    iter 2   reward  6.23   ep_len 11.21                    diverged 0

Per-term at iteration 0: anchor_pos 0.869, anchor_ori 0.795, body_pos 0.920,
obj_pos 0.853, obj_ori 0.946, obj_ang_vel 0.934, joint_pos 0.100.
Three iterations is far too short to read a trend from; what it establishes is that the
port runs, stays stable and logs.

### The BC curriculum is the stability mechanism

Against the hand-written PPO that used to live here (deleted), same environment, from
scratch::

                        hand-written PPO        SUGAR's BCPPO
    mean reward         -15.8 -> -48948         +9.48
    divergences         47 -> 138 -> 234        0
    joint_acc term      ~2e11                   883929

That last row settles an earlier false alarm. 883929 sits inside the range measured from
Isaac's own rollouts (mean 25.9k, worst step 744k), so `joint_acc` was never
mis-specified -- the flailing policy was the entire problem, and stages 1-2 remove it.

### Contact limits were wrong, and that invalidated everything measured before

`njmax`/`nconmax` are per world, and they must be sized for the WORST case. Leaving them
`None` lets Newton size from the initial near-static pose, which gives `nconmax=1024`;
real motion generates up to **6524 contacts per world**, and MJWarp silently drops
everything above the limit (489 `exceeded MJWarp limit` messages in one short benchmark,
144 `nefc overflow`). So the physics was wrong wherever contact matters most -- exactly
during the grip -- and every number measured before this was measured on it. Now 8192 /
2048 per world, with headroom over the measured peaks. Overflow count: 0.

### Speed, measured against correct physics

`env-steps/s` below is AGGREGATE across worlds (num_envs / step). The per-world rate is
`1000 / step_ms` -- at 8 worlds that is 1.45 policy steps/s, not 11.6.

    envs   step_ms   obs+rew   contacts   env-steps/s (aggregate)
      1     133.8m     10.6m       1677      7.5
      2     231.9m     10.7m       2437      8.6
      4     258.8m     10.8m       3011     15.5
      8     686.8m     10.7m       4422     11.6
     16    1654.3m     10.9m       6082      9.7

Trustworthy:

* `obs+rew` is ~11 ms and flat in world count -- about 1.6% of a step at 8 worlds. The
  observation and reward code is not the bottleneck.
* ~8-15 env-steps/s overall, and it does not improve past 4 worlds. Step spreads are
  91-619 ms, so treat these as approximate.

NOT trustworthy, and not to be quoted: the `collide`/`solve` split. The consistency check
`(collide + solve + obs) / step` comes out at 1.13x-3.16x, and standalone `collide`
exceeds the whole step that contains it -- most likely GPU work that overlaps inside
`step` but serialises when timed in a tight loop. The decomposition is wrong; only the
end-to-end `step_ms` and the pure-torch `obs+rew` column mean anything.

At ~12 env-steps/s, SUGAR's 30k iterations x 24 steps x 8 envs is about 5.8M env-steps,
i.e. **~5.5 days**. SUGAR trains in Isaac with thousands of environments. One to two
orders of magnitude are missing and more worlds alone will not supply them.

### Where the time goes, by elimination

Measured at 8 worlds. This is physics only -- there is no viewer anywhere in the training
loop or the benchmark; video rendering happens once per `--video-interval` and is separate.

* `obs+rew`: ~11 ms of a ~690 ms step -> **1.6%**. Not the bottleneck.
* solver iterations: dropping `iterations`/`ls_iterations` from 100/50 to 10/5 takes the
  step from 687.5 ms to 602.2 ms -> **~12%** for a tenfold reduction. Not the bottleneck.
* remainder -> **~85% is collision detection**, over a 45 748-triangle hand and a
  100 000-triangle box, generating 1 677-6 082 contacts per world.

### Which contact path is active -- check this before trusting any of the above

`use_mujoco_contacts=False` (carrybox_env.py), so Newton's `CollisionPipeline` does the
collision and MuJoCo only keeps a stable geom id -- see the comment at
`solver_mujoco.py:5131`. This matters because in the *other* regime
(`use_mujoco_contacts=True`, the solver default) MuJoCo compiles every mesh through its own
convex-hull path capped at `Mesh.MAX_HULL_VERTICES = 64`, so the source triangle counts
would be irrelevant and the argument below would invert. Hydroelastic is also NOT enabled
here; when it is turned on for tactile, `sdf_max_resolution` becomes the collision lever
instead.

### The lever: the asset's own collision approximation

`descriptions/objects/small_box/Props/instanceable_meshes.usd` authors
`physics:approximation = 'convexDecomposition'` on the box mesh, and Isaac honours it.
This environment collides the raw 100 000-triangle mesh instead.

Note what the box actually is, because it matters: it is an **open carton**, not a solid
block. Mesh volume 3 889.9 cm^3 against a convex hull of 35 502.8 cm^3 -- a single hull
would fill it in and add 813% material, which would be a real change to the physics and is
exactly what the no-hulling rule is about. Convex *decomposition* is a different thing: it
preserves the concavity with a set of convex parts, and it is the setting the asset itself
declares. Honouring it is arguably more faithful than what is done here now.

The hand is concave too (hull adds 135%), so the same distinction applies there.

**Newton already implements this and we bypass it.** `import_usd.py` maps
`physics:approximation` onto a remeshing method (`convexdecomposition` -> `coacd`, which is
installed). `load_box_mesh` here reads USD points by hand and hands `add_shape_mesh` a raw
`newton.Mesh`, so the asset's declared approximation never fires. The fix is Newton's own
hook -- `ModelBuilder.approximate_meshes(method="coacd", ...)` (`builder.py:6831`) or
loading through `import_usd` -- not a hand-rolled decomposition.

Part of the gap is not a defect:  Isaac is fast partly *because* its URDF importer hulls
every collider, and this project does not hull interacting geometry. Accurate contact
costs more. An untested idea worth trying: keep exact meshes for the hands and the box,
simplify colliders on links that never touch the box.

## Evaluation videos

`--video-interval N` (default 100) renders a deterministic rollout and logs it to wandb as
`video/rollout`, alongside `video/box_lift` and `video/box_lift_reference` so the clip has
a number attached. `--video-interval 0` disables it.

A separate one-world environment is used: the training worlds are replicated at zero
spacing and sit on top of each other, so rendering the training model shows every robot
superimposed. Actions are the policy mean rather than a sample, and the clip and start
frame are fixed, so successive videos differ because the policy changed. Rendering needs
hardware EGL -- without it the viewer silently falls back to software rasterisation and each
frame costs seconds; the recorder warns once if it detects this. `slurm/devnode.sbatch`
arranges it for you (`setup_container.sh` writes the missing NVIDIA glvnd ICD), and
`slurm/render_env_egl.sh` does it explicitly outside that path. A render failure is caught
and logged, never allowed to end a training run.

## The algorithm is SUGAR's, imported not reimplemented

`train_bcppo.py` imports `BCPPO` from `SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py`
and runs it inside `rsl_rl`'s own `OnPolicyRunner`, with the hyperparameters transcribed
from `BCPPORunnerCfg`. `BCPPO` is registered by the same mechanism SUGAR uses
(`setattr(builtins, "BCPPO", ...)`, `scripts/sugar_rl/train.py:147-150`), because the
runner resolves the algorithm with `eval(alg_cfg["class_name"])`.

The only local code in the training loop is `vec_env.py`, which presents the Newton
environment as an `rsl_rl.env.VecEnv` with the three observation groups the config asks
for:

    policy   510-D   validated against Isaac's recorded actions to RMSE 0.088
    critic   890-D   obs_890.py
    teacher  890-D   obs_890.py -- what the frozen refiner is asked to imitate

BCPPO's curriculum, for reference:

    stage 1   step < 500          loss = distill                    (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value    (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

The teacher checkpoint is required, not optional: `BCPPO.__init__` asserts on a missing
one, and stages 1-2 have no loss without it. Default path is the recovered
`refiner_model10000.pt` (see TODO 16).

An earlier version of this directory carried a hand-written PPO (`ppo.py`, `train.py`).
It has been deleted. It was stage 3 with the distillation dropped, which is not the
algorithm SUGAR trains the tracker with.

## Logging

`rsl_rl` has native wandb support, so nothing here writes to wandb directly: the runner
config sets `logger: wandb` and `wandb_project`, and the run name is the log directory's
basename. Credentials follow the convention used elsewhere in this workspace --
`WANDB_API_KEY` from the environment, else `~/.netrc` for `api.wandb.ai` -- and
`train_bcppo.py` checks for one before building the environment, so a run cannot get
several minutes in with logging silently off. Pass `--logger tensorboard` to opt out.

## STATUS: BCPPO trains on Newton and logs to wandb

First working run, 8 worlds, 3 iterations, `data_000`
(https://wandb.ai/nvr-amri/sugar_newton/runs/acs79r73)::

    iter 0   reward  9.48   ep_len 10.75   noise std 0.51   diverged 0
    iter 1   reward  7.54   ep_len 11.19                    diverged 0
    iter 2   reward  6.23   ep_len 11.21                    diverged 0

Per-term at iteration 0: anchor_pos 0.869, anchor_ori 0.795, body_pos 0.920,
obj_pos 0.853, obj_ori 0.946, obj_ang_vel 0.934, joint_pos 0.100.
Three iterations is far too short to read a trend from; what it establishes is that the
port runs, stays stable and logs.

### The BC curriculum is the stability mechanism

Against the hand-written PPO that used to live here (deleted), same environment, from
scratch::

                        hand-written PPO        SUGAR's BCPPO
    mean reward         -15.8 -> -48948         +9.48
    divergences         47 -> 138 -> 234        0
    joint_acc term      ~2e11                   883929

That last row settles an earlier false alarm. 883929 sits inside the range measured from
Isaac's own rollouts (mean 25.9k, worst step 744k), so `joint_acc` was never
mis-specified -- the flailing policy was the entire problem, and stages 1-2 remove it.

### Open: throughput

~3.6 env-steps/s at 8 worlds (collection 43-53 s for 192 timesteps). Algorithm-independent
-- the same figure appeared under the hand-written PPO -- so it is a property of the
environment. `njmax`/`nconmax` were being scaled by `num_envs` when they are per world;
that was wrong and is fixed, and fixing it changed nothing. A per-component breakdown
across 1/2/4/8/16 worlds is the next measurement; the hypothesis to test is that the
broad phase is N-by-N across all shapes in all worlds rather than per world, which would
make cost quadratic in world count.

## The algorithm is SUGAR's, imported not reimplemented

`train_bcppo.py` imports `BCPPO` from `SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py`
and runs it inside `rsl_rl`'s own `OnPolicyRunner`, with the hyperparameters transcribed
from `BCPPORunnerCfg`. `BCPPO` is registered by the same mechanism SUGAR uses
(`setattr(builtins, "BCPPO", ...)`, `scripts/sugar_rl/train.py:147-150`), because the
runner resolves the algorithm with `eval(alg_cfg["class_name"])`.

The only local code in the training loop is `vec_env.py`, which presents the Newton
environment as an `rsl_rl.env.VecEnv` with the three observation groups the config asks
for:

    policy   510-D   validated against Isaac's recorded actions to RMSE 0.088
    critic   890-D   obs_890.py
    teacher  890-D   obs_890.py -- what the frozen refiner is asked to imitate

BCPPO's curriculum, for reference:

    stage 1   step < 500          loss = distill                    (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value    (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

The teacher checkpoint is required, not optional: `BCPPO.__init__` asserts on a missing
one, and stages 1-2 have no loss without it. Default path is the recovered
`refiner_model10000.pt` (see TODO 16).

An earlier version of this directory carried a hand-written PPO (`ppo.py`, `train.py`).
It has been deleted. It was stage 3 with the distillation dropped, which is not the
algorithm SUGAR trains the tracker with.

## Logging

`rsl_rl` has native wandb support, so nothing here writes to wandb directly: the runner
config sets `logger: wandb` and `wandb_project`, and the run name is the log directory's
basename. Credentials follow the convention used elsewhere in this workspace --
`WANDB_API_KEY` from the environment, else `~/.netrc` for `api.wandb.ai` -- and
`train_bcppo.py` checks for one before building the environment, so a run cannot get
several minutes in with logging silently off. Pass `--logger tensorboard` to opt out.

## STATUS: environment runs; BCPPO port is UNTESTED

The numbers below are from the deleted hand-written PPO, from scratch with no teacher.
They are kept because the throughput figure is a property of the environment, not the
algorithm, and it is still unexplained. The reward collapse should not recur under BCPPO,
whose stages 1-2 exist precisely to stop the policy flailing -- but that is a prediction,
not a measurement.

Measured, 16 worlds, from scratch, 3 iterations, hand-written PPO (deleted):

    it 1  return    -15.8  ep_len 4.6  diverged  47   4 env-steps/s
    it 2  return  -1242.9  ep_len 2.8  diverged 138   4 env-steps/s
    it 3  return -48947.7  ep_len 1.8  diverged 234   3 env-steps/s

Two open problems, neither solved:

1. **Throughput.** 4 env-steps/s across 16 worlds is ~20x *worse* per world than the
   single-world validation scene (5.2 steps/s). `njmax`/`nconmax` are per world
   (`solver_mujoco.py:3183`) and an earlier version scaled `nconmax` by `num_envs`,
   allocating 128k contacts per world; that was wrong and is fixed, but fixing it changed
   the number not at all, so the cause is still unknown. Note episodes are terminating
   after ~2 steps here, so `reset()` runs almost every step -- that is the next thing to
   rule out, not a diagnosis.
2. **Reward explodes from scratch.** Diverged envs are detected and zeroed, so this comes
   from envs that are finite but violent.

   The `joint_acc` term was the suspect and has been **ruled out by measurement**. Taking
   Isaac's own recorded `joint_vel` from `isaac/rollouts_isaac` and differencing it exactly
   the way this env does gives `joint_acc_l2` of mean 25.9k and worst-step 744k, i.e. a
   reward contribution of -0.0065 mean and -0.19 at worst. SUGAR's -2.5e-7 weight is
   correctly calibrated for this quantity. Reaching a return of -4.9e4 needs
   `joint_acc_l2` around 2e11, about 8e6x Isaac's mean -- roughly 3000x Isaac's joint
   velocities. That is a flailing policy, not a mis-specified reward term.

   Which points at the missing BC curriculum: stages 1-2 exist precisely so the policy is
   never allowed to flail (below).

## The algorithm is NOT SUGAR's

SUGAR's tracker uses **BCPPO** (`BCPPORunnerCfg` -> `utils/rsl_rl_bcppo.py`), a three-stage
curriculum around the frozen refiner as teacher:

    stage 1   step < 500        loss = distill                       (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value     (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

Supervising signals in SUGAR's tracker training:

| signal | where it enters |
|---|---|
| RL reward, the 17 terms below | surrogate loss, stage 3 only |
| teacher action distribution, KL(teacher \|\| student) over the 29-D Gaussians | stages 1-2 dominant, fades in stage 3 |
| value targets (returns) | critic, warmed up alone in stage 2 |
| entropy bonus 0.005 | stage 3, scaled by alpha |
| adaptive-KL LR control, desired_kl 0.01 | fixed during stage 1 |

and three observation groups, not one: policy 510-D, critic 890-D privileged, teacher
890-D. Symmetry and RND exist in the class but neither config enables them for CarryBox.

**What is implemented here is stage 3 without the distillation term** -- plain PPO. The
teacher checkpoint is recovered (TODO 16) but the 890-D teacher group is not built, so
BCPPO is the next piece of work, and is likely a precondition for training stability
rather than an enhancement.

## What is faithful to SUGAR, and what is not

Faithful, transcribed rather than re-derived:

- the 510-D observation, validated offline against Isaac's own recorded actions to
  RMSE 0.088 by `validation/verify_tracker_obs.py`
- actuator gains, armature, effort limits and the `0.25 * effort / stiffness` action scale
- reward weights and stds (`BaseRewardsCfg`) and the `exp(-error/std^2)` term shapes
- termination thresholds (`BaseTerminationsCfg`)
- PPO hyperparameters, identical between `BasePPORunnerCfg` and `BCPPORunnerCfg` except
  `init_noise_std`, which is 0.5 for the tracker (an earlier version used the inference
  task's 1.0)

Deliberate gaps, all recorded in the code:

- **Four reward terms are missing**: `feet_slide`, `feet_air_time`, `undesired_contacts`,
  `hoi_contact`. All four need per-body contact forces, which the env does not surface
  yet. See `rewards.OMITTED`.
- **The critic is not privileged.** SUGAR's critic takes an 890-D observation built from
  future reference frames and teacher terms; this one takes the same 510-D actor
  observation and trains from scratch, so `--warm-start` loads the actor only.
- **Timeout bootstrapping.** GAE cuts the value target at every `done`, timeouts included,
  which biases truncated episodes low.

## Cost of accurate contact

Worlds are replicated with `spacing=(0,0,0)` on Newton's own advice -- separated worlds
are numerically worse, and worlds do not collide. Note that Newton's `replicate` docstring
recommends `approximate_meshes()` before replication so one simplified mesh is shared;
that is convex approximation of interacting geometry, which this project does not do, so
model construction scales with the real 45k-triangle hand and 50k-triangle box. Expect
build time, not step time, to dominate at high `--num-envs`.
