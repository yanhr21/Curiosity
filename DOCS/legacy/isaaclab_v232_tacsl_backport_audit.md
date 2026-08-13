# IsaacLab v2.3.2 TacSL Backport Audit

Audit date: 2026-07-14.

## Provenance

- Accepted SUGAR control stack: IsaacLab v2.3.0 commit
  `3c6e67bb5c7ada942a6d1884ab69338f57596f77`.
- TacSL source: official IsaacLab v2.3.2 commit
  `37ddf626871758333d6ed89cf64ad702aef127d0`.
- Local provenance marker:
  `IsaacLab/source/isaaclab_contrib/CURIOSITY_UPSTREAM_COMMIT`.
- A recursive comparison against a clean v2.3.2 checkout reports the complete
  `source/isaaclab_contrib` tree as identical except for the provenance marker
  and `visuotactile_sensor.py` changes enumerated below.
- `isaaclab_assets/sensors/gelsight.py` is byte-for-byte identical to the
  official v2.3.2 file.

## Core v2.3.0 integration glue

The accepted v2.3.0 core is not replaced wholesale. A clean full v2.3.2
checkout is retained under the ignored experiment runtime tree for comparison;
attempting to use its entire core directly exposed SUGAR Manager API
incompatibilities. Three bounded integration pieces are therefore carried into
the accepted core:

1. `markers/config/__init__.py` defines the official
   `VISUO_TACTILE_SENSOR_MARKER_CFG` used by the contrib sensor.
2. `scene/interactive_scene.py` recognizes `VisuoTactileSensorCfg` and resolves
   `{ENV_REGEX_NS}` in the camera and contact-object paths. The import is
   optional, so the frozen no-contrib SUGAR control still imports normally.
3. `sensors/camera/tiled_camera.py` replaces only the removed legacy
   `XFormPrim` lookup/pose-read dependency with the corresponding direct-USD
   prim matching and pose-read behavior from v2.3.2. Renderer creation,
   Replicator products/annotators, camera intrinsics, and the official TAXIM
   render path are unchanged.

The official GelSight sensor config is exported from
`isaaclab_assets.sensors`. No alternative local sensor interface is introduced.

## Audited changes inside the official sensor

Only `visuotactile_sensor.py` differs from the pinned upstream contribution:

1. **Merged/instanced mesh lookup fallback.** The original visual-mesh lookup
   remains first. If Isaac Sim 5.1 exposes an official referenced elastomer as
   an instance proxy without a visual-only child, traversal reads that same
   mesh. It raises if no mesh exists; it never fabricates a plane or taxel
   array. The final R15 branch uses the official referenced elastomer, while
   this fallback remains auditable compatibility glue.
2. **Geometry-derived tactile frame.** Upstream hard-codes local Z as the
   normal. The official R15 mesh is sampled on its X-Z plane with outward
   normal on local ±Y. The local tactile basis is therefore derived from the
   same `slim_axis`, in-plane axes, and dome direction already computed by the
   official ray caster. SDF sampling and penalty equations are unchanged.
3. **Pressure/shear channel separation.** The upstream code projects total
   force onto one sensor Z axis. A footprint crossing a CarryBox edge then
   produces negative projected pressure and leaks normal load into shear. The
   local output exposes the already-computed official scalar penalty term
   `fc_norm = normal_contact_stiffness * penetration_depth` as the non-negative
   taxel normal force, and projects only the official Coulomb-limited
   tangential force into the signed two-axis shear channel. Stiffness,
   friction, relative velocity, SDF gradients, taxel layout, and contact mask
   are unchanged.
4. **Nominal-render initialization guard.** Manager observation-shape discovery
   can request the force buffer before the recorder has captured the required
   no-load TAXIM render. The sensor updates its official force/SDF fields for
   that query but defers the camera update until `_nominal_tactile` has been
   established by the official `get_initial_render` path. Subsequent
   RGB/depth/deformation updates use the upstream camera code unchanged.

## Assets and claim boundary

- The dual-hand geometry references official R15 USD subprims; no sensor mesh
  is reconstructed.
- Official Isaac 5.1 TAXIM `gelsight_r15_data/bg.jpg` and `polycalib.npz` are
  stored under the ignored experiment asset cache. `real_bg.npy` is optional
  in the upstream config and absent (HTTP 404) from that official directory.
- Left and right camera streams are validated in sequential one-camera
  processes because two simultaneous v2.3.0 `TiledCamera`/Replicator instances
  conflict during initialization in the mixed-version runtime. Both force
  sensors remain active and recorded in each process; this is a runtime
  isolation boundary, not a simplified sensor or a policy result.
- Until physical GelSight load, footprint, shear/slip, latency/noise, and image
  calibration is complete, every result is labeled `high-fidelity simulated
  tactile`.
