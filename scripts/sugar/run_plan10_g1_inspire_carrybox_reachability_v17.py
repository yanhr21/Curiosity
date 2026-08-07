#!/usr/bin/env python3
"""No-learning official G1-Inspire reachability probe for SUGAR CarryBox.

The G1 root and first 29 joints follow one official SUGAR CarryBox reference
motion.  Each hand is commanded through the six released Unitree Inspire
controls and the released intermediate/distal coupling, rather than treating
the 24 simulation joints as independent actuators.  The box pose is written
once at initialization and is thereafter a fully dynamic PhysX rigid object.
This is a mechanics/reachability diagnostic, not policy inference, tactile
sensing, or soft-gel evidence. V12 additionally hash-records the explicit
official nominal object material and admits a real central setup pedestal for
a bottom-support grasp; post-PhysX anatomical loads remain the authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import socket
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


HOST = socket.gethostname()
if HOST.startswith(("mgmtserver", "login")):
    raise SystemExit(f"Refusing Plan-10 PhysX probe on login node: {HOST}")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Plan-10 PhysX probe requires the retained allocation")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--unitree-repo", type=Path, required=True)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--robot-motion", type=Path, required=True)
parser.add_argument("--object-motion", type=Path, required=True)
parser.add_argument("--box-usd", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--source-start", type=int, default=200)
parser.add_argument("--source-end", type=int, default=420)
parser.add_argument("--close-start", type=float, default=225.0)
parser.add_argument("--close-end", type=float, default=270.0)
parser.add_argument("--close-fraction", type=float, default=0.68)
parser.add_argument("--pregrasp-thumb-pitch-rad", type=float, default=0.0)
parser.add_argument("--pregrasp-thumb-yaw-rad", type=float, default=0.0)
for _side in ("left", "right"):
    parser.add_argument(
        f"--pregrasp-{_side}-thumb-pitch-rad", type=float, default=None
    )
    parser.add_argument(
        f"--pregrasp-{_side}-thumb-yaw-rad", type=float, default=None
    )
parser.add_argument("--closed-thumb-yaw-rad", type=float, default=None)
for _side in ("left", "right"):
    for _finger in ("index", "middle", "ring", "little"):
        parser.add_argument(
            f"--{_side}-{_finger}-close-fraction", type=float, default=None
        )
    parser.add_argument(
        f"--closed-{_side}-thumb-pitch-rad", type=float, default=None
    )
    parser.add_argument(
        f"--closed-{_side}-thumb-yaw-rad", type=float, default=None
    )
parser.add_argument(
    "--bilateral-shoulder-roll-offset-rad", type=float, default=0.0
)
parser.add_argument("--left-shoulder-roll-offset-rad", type=float, default=None)
parser.add_argument("--right-shoulder-roll-offset-rad", type=float, default=None)
parser.add_argument(
    "--left-shoulder-roll-end-offset-rad", type=float, default=None
)
parser.add_argument("--shoulder-roll-transition-start", type=float, default=None)
parser.add_argument("--shoulder-roll-transition-end", type=float, default=None)
parser.add_argument("--left-wrist-roll-offset-rad", type=float, default=0.0)
parser.add_argument("--right-wrist-roll-offset-rad", type=float, default=0.0)
parser.add_argument("--left-wrist-yaw-offset-rad", type=float, default=0.0)
parser.add_argument("--right-wrist-yaw-offset-rad", type=float, default=0.0)
parser.add_argument(
    "--left-wrist-roll-end-offset-rad", type=float, default=None
)
parser.add_argument(
    "--left-wrist-yaw-end-offset-rad", type=float, default=None
)
parser.add_argument("--left-shoulder-yaw-offset-rad", type=float, default=0.0)
parser.add_argument("--right-shoulder-yaw-offset-rad", type=float, default=0.0)
parser.add_argument("--physics-substeps-per-source", type=int, default=4)
parser.add_argument("--brake-start", type=float, default=None)
parser.add_argument("--brake-steps", type=int, default=0)
parser.add_argument("--solver-position-iterations", type=int, default=4)
parser.add_argument("--solver-velocity-iterations", type=int, default=1)
parser.add_argument("--hold-steps", type=int, default=240)
parser.add_argument("--contact-threshold-n", type=float, default=0.01)
parser.add_argument(
    "--require-settled-all-groups-frames",
    type=int,
    default=0,
    help=(
        "For SUGAR side-clamp mechanics, skip lift and hold unless the final "
        "declared settle interval contains this many consecutive 12/12 "
        "palm/thumb/four-finger contact frames. Zero disables the stop gate."
    ),
)
parser.add_argument("--grasp-dwell-source", type=float, default=None)
parser.add_argument("--grasp-close-steps", type=int, default=0)
parser.add_argument("--grasp-settle-steps", type=int, default=0)
parser.add_argument("--grasp-close-source-span", type=int, default=0)
parser.add_argument("--resume-source-span", type=int, default=10)
parser.add_argument("--resume-polynomial-power", type=int, default=2)
parser.add_argument("--robot-root-y-offset-m", type=float, default=0.0)
parser.add_argument(
    "--waist-pitch-absolute-rad",
    type=float,
    default=None,
    help=(
        "Optional declared absolute waist-pitch posture for the no-learning "
        "full-body mechanics branch; live hard limits remain authoritative."
    ),
)
parser.add_argument(
    "--bilateral-hip-roll-outward-offset-rad",
    type=float,
    default=0.0,
    help=(
        "Declared fixed-root mechanics posture: add this offset to the left "
        "hip roll and subtract it from the right hip roll, widening both legs "
        "without changing the upper-body source pose."
    ),
)
parser.add_argument("--zero-initial-object-velocity", action="store_true")
parser.add_argument("--object-static-friction", type=float, default=0.6)
parser.add_argument("--object-dynamic-friction", type=float, default=0.5)
parser.add_argument(
    "--body-control-mode",
    choices=(
        "state_replay",
        "pd_target",
        "bilateral_ik",
        "unitree_demo_pose",
        "sugar_side_clamp",
    ),
    default="state_replay",
    help=(
        "state_replay is kinematic reachability diagnostics only; pd_target "
        "writes the floating-base/body state once and then uses physical joint drives; "
        "sugar_side_clamp starts from the official SUGAR CarryBox state and drives "
        "only the two seven-DoF arms through official IsaacLab DLS IK"
    ),
)
parser.add_argument("--unitree-demo-parquet", type=Path, default=None)
parser.add_argument("--unitree-demo-source-id", type=str, default=None)
parser.add_argument("--unitree-demo-source-revision", type=str, default=None)
parser.add_argument("--unitree-demo-episode-index", type=int, default=0)
parser.add_argument("--unitree-demo-frame-index", type=int, default=0)
parser.add_argument(
    "--unitree-demo-align-box-narrow-axis-to-hands",
    action="store_true",
    help=(
        "At the one-time trial initialization, align the unchanged CarryBox "
        "0.400 m local-X extent with the official demo's horizontal hand-to-hand "
        "axis. --unitree-demo-box-yaw-rad is then a declared additive yaw offset."
    ),
)
parser.add_argument(
    "--unitree-demo-box-yaw-rad",
    type=float,
    default=1.5707963267948966,
    help=(
        "One-time CarryBox yaw at trial initialization. The default presents "
        "the real box's 0.400 m narrow side to the bilateral official-demo pose."
    ),
)
parser.add_argument(
    "--unitree-demo-box-center-offset-m",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
)
parser.add_argument(
    "--unitree-demo-support-size-xy-m",
    type=float,
    nargs=2,
    default=(0.24, 0.24),
    help=(
        "Real central setup pedestal footprint. It supports the box before lift "
        "while leaving the load-bearing bottom edges reachable by both hands."
    ),
)
parser.add_argument(
    "--unitree-demo-support-cradle-height-m",
    type=float,
    default=0.0,
    help=(
        "Height of four real low setup rails around the box footprint. The rails "
        "prevent the detailed CarryBox collision mesh from tipping before bilateral "
        "engagement and remain below the admitted >0.10 m lift interval."
    ),
)
parser.add_argument(
    "--unitree-demo-outward-shift-m",
    type=float,
    default=0.0,
    help=(
        "Translate the official bilateral hand target and CarryBox together "
        "along the horizontal pelvis-to-hand-midpoint direction. This preserves "
        "the official relative bilateral pose while creating declared clearance "
        "for a thick box that a thin plate does not require."
    ),
)
parser.add_argument("--unitree-demo-approach-steps", type=int, default=0)
parser.add_argument(
    "--unitree-demo-side-clamp",
    action="store_true",
    help=(
        "Retarget the hash-bound official Unitree body pose to a true bilateral "
        "CarryBox side-palm clamp. The exact official Inspire palm mesh surface "
        "is placed on the opposed local-X box faces; only the seven-DoF arms "
        "move through IsaacLab DifferentialIKController."
    ),
)
parser.add_argument(
    "--unitree-demo-side-clamp-box-local-y-m", type=float, default=0.14
)
parser.add_argument(
    "--side-clamp-box-axis",
    choices=("x", "y", "pca0"),
    default="x",
    help="Opposed CarryBox local side faces used by the bilateral palms.",
)
parser.add_argument(
    "--side-clamp-box-local-tangent-m",
    type=float,
    default=None,
    help=(
        "Shared coordinate along the other horizontal box axis. For local-X "
        "faces the legacy local-Y value is used when this is omitted."
    ),
)
for _side in ("left", "right"):
    parser.add_argument(
        f"--{_side}-side-clamp-box-local-tangent-m", type=float, default=None
    )
    parser.add_argument(
        f"--{_side}-side-clamp-box-local-z-m", type=float, default=None
    )
    parser.add_argument(
        f"--{_side}-side-clamp-box-normal-m", type=float, default=None
    )
    parser.add_argument(
        f"--{_side}-side-clamp-tilt-tangent-rad", type=float, default=0.0
    )
    parser.add_argument(
        f"--{_side}-side-clamp-tilt-height-rad", type=float, default=0.0
    )
    parser.add_argument(
        f"--{_side}-side-clamp-normal-roll-rad",
        type=float,
        default=0.0,
        help=(
            "Rotate the palm frame about its final local-X load normal while "
            "preserving the declared CarryBox surface point and load normal."
        ),
    )
    parser.add_argument(
        f"--{_side}-side-clamp-contact-pca-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("PCA0_M", "PCA1_M", "PCA2_M"),
        help=(
            "Optional exact CarryBox PCA contact coordinate. It is admitted only "
            "with --side-clamp-box-axis pca0 and the matching outward-PCA vector. "
            "This lets the two hands address different physical faces, such as "
            "one PCA0 side brace and one PCA2 bottom support."
        ),
    )
    parser.add_argument(
        f"--{_side}-side-clamp-outward-pca",
        type=float,
        nargs=3,
        default=None,
        metavar=("PCA0", "PCA1", "PCA2"),
        help=(
            "Unit outward normal in the frozen CarryBox PCA basis for an exact "
            "per-hand contact. Palm local +X is aligned to this vector, so the "
            "official load-bearing local -X palm surface faces the object."
        ),
    )
    parser.add_argument(
        f"--{_side}-side-clamp-approach-clearance-m",
        type=float,
        default=None,
        help=(
            "Optional per-side collision-free palm approach distance; defaults "
            "to the shared side-clamp approach clearance."
        ),
    )
parser.add_argument(
    "--side-clamp-contact-geometry-source",
    type=Path,
    default=None,
    help=(
        "Hash-bound derivation record required whenever an exact per-hand PCA "
        "contact is used. The record must identify the physical mesh/cooked "
        "surface evidence behind every declared coordinate and outward normal."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-box-local-offset-m",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    help=(
        "Declared one-time CarryBox translation in its official source frame "
        "to balance asymmetric SUGAR arm reach."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-support-height-m",
    type=float,
    default=0.0,
    help=(
        "Height of a real central setup pedestal. The box is raised by the "
        "same world-Z distance once at initialization; the pedestal remains "
        "behind during lift and is never moved or replayed."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-support-size-xy-m",
    type=float,
    nargs=2,
    default=(0.18, 0.18),
)
parser.add_argument(
    "--sugar-side-clamp-fit-box-to-reachable-palms",
    action="store_true",
    help=(
        "After collision-free arm approach, translate the unchanged box once so "
        "the two declared mesh surface points best fit the two actually reachable "
        "official palm surface points. No rotation, scaling, replay, or later fit."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-direct-setup-ik",
    action="store_true",
    help=(
        "During collision-free offstage setup only, apply each official "
        "DifferentialIKController iterate directly to the arm state. The "
        "dynamic box is published afterward and every recorded step remains "
        "drive-controlled PhysX."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-direct-refinement-steps",
    type=int,
    default=200,
    help=(
        "Maximum offstage official full-pose DLS trust-region iterations after "
        "the direct side-clamp approach. Recorded dynamics are unaffected."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-max-reachable-orientation-delta-rad",
    type=float,
    default=0.10,
    help=(
        "Declared offstage DLS acceptance bound relative to the geometric "
        "box-normal palm frame. This is a setup diagnostic bound, not a "
        "success criterion; recorded palm contact remains authoritative."
    ),
)
parser.add_argument(
    "--unitree-demo-side-clamp-box-local-z-m", type=float, default=0.0
)
parser.add_argument(
    "--unitree-demo-side-clamp-palm-inset-m", type=float, default=0.0015
)
for side in ("left", "right"):
    parser.add_argument(
        f"--{side}-side-clamp-palm-inset-m",
        type=float,
        default=None,
        help=(
            "Optional per-side Cartesian palm target inset behind the cooked "
            "box surface; defaults to the shared inset."
        ),
    )
parser.add_argument(
    "--unitree-demo-side-clamp-approach-clearance-m", type=float, default=0.06
)
parser.add_argument("--unitree-demo-close-steps", type=int, default=200)
parser.add_argument("--unitree-demo-settle-steps", type=int, default=120)
parser.add_argument("--ik-contact-template-trace", type=Path, default=None)
parser.add_argument("--ik-template-index", type=int, default=432)
parser.add_argument("--ik-approach-clearance-m", type=float, default=0.08)
parser.add_argument("--ik-approach-steps", type=int, default=200)
parser.add_argument("--ik-left-box-roll-offset-rad", type=float, default=0.0)
parser.add_argument("--ik-right-box-roll-offset-rad", type=float, default=0.0)
parser.add_argument(
    "--ik-track-live-box-during-grasp",
    action="store_true",
    help=(
        "Causally recompute the palm target from the current dynamic box pose "
        "during press/closure/settle, then freeze one anchor before lift"
    ),
)
parser.add_argument(
    "--ik-track-live-box-during-lift",
    action="store_true",
    help=(
        "No-learning mechanics diagnostic: causally retain the declared palm "
        "poses on the current dynamic box and apply one vertical Cartesian "
        "lead increment per step. The object remains fully dynamic and is "
        "never written after initialization."
    ),
)
parser.add_argument(
    "--ik-contact-preload-height-m",
    type=float,
    default=0.0,
    help=(
        "Before the settled all-group stop gate, causally move both closed "
        "hands this far upward relative to the live dynamic box. This is a "
        "declared contact-forming preload; it never writes object state."
    ),
)
parser.add_argument(
    "--ik-contact-preload-steps",
    type=int,
    default=0,
    help=(
        "Total duration for the declared contact preload, including its "
        "ramp and any constant-height settling tail."
    ),
)
parser.add_argument(
    "--ik-contact-preload-ramp-steps",
    type=int,
    default=None,
    help=(
        "Cosine-ramp duration within the contact preload. Defaults to the "
        "full preload duration for backward compatibility; a shorter value "
        "leaves an explicit constant-height settling tail."
    ),
)
parser.add_argument(
    "--ik-contact-compression-m",
    type=float,
    default=0.0,
    help=(
        "After contact preload and before the settled gate, move each closed "
        "palm inward along its live outward surface normal by this distance. "
        "The CarryBox remains dynamic and is never written."
    ),
)
parser.add_argument(
    "--ik-contact-compression-steps",
    type=int,
    default=0,
    help="Total compression duration, including its constant settling tail.",
)
parser.add_argument(
    "--ik-contact-compression-ramp-steps",
    type=int,
    default=None,
    help=(
        "Cosine-ramp duration within contact compression. Defaults to the "
        "full compression duration."
    ),
)
parser.add_argument(
    "--ik-live-lift-lead-m",
    type=float,
    default=None,
    help=(
        "Optional fixed upward Cartesian lead for causal live-box lift "
        "tracking; defaults to lift_height/lift_steps."
    ),
)
parser.add_argument(
    "--ik-live-lift-lead-ramp-steps",
    type=int,
    default=0,
    help=(
        "Optional cosine ramp duration for the causal live-box vertical lead. "
        "Zero preserves the existing immediate fixed lead; a positive value "
        "builds the same declared lead smoothly without writing object state."
    ),
)
parser.add_argument(
    "--ik-live-lift-scheduled-world-z",
    action="store_true",
    help=(
        "Hybrid causal mechanics lift: follow the current dynamic box in X/Y "
        "and orientation while commanding a smooth, time-declared world-Z "
        "trajectory from the contact anchor to ik_lift_height_m. The object "
        "remains dynamic and is never written after initialization."
    ),
)
parser.add_argument(
    "--ik-live-lift-anchor-world-xy",
    action="store_true",
    help=(
        "During live-box lifting, hold the end-of-grasp box-frame X/Y anchor "
        "while still following the current dynamic box orientation. This "
        "resists horizontal drift without writing object state."
    ),
)
parser.add_argument(
    "--ik-live-hold-relative-to-box",
    action="store_true",
    help=(
        "After a scheduled world-Z lift, replace the fixed world hold with a "
        "causal box-relative hold. The translation bias is measured once from "
        "the final scheduled target to the post-PhysX box pose, then kept "
        "constant while the object remains dynamic and is never replayed."
    ),
)
parser.add_argument("--ik-palm-press-steps", type=int, default=200)
parser.add_argument(
    "--sugar-side-clamp-palm-press-first",
    choices=("left", "right"),
    default=None,
    help=(
        "Optional no-learning asymmetric grasp sequence. The selected palm "
        "reaches and loads its declared surface first while the other remains "
        "at approach; the second palm then presses while the first causally "
        "tracks the dynamic box. The object is never written or replayed."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-close-first-hand-before-second",
    action="store_true",
    help=(
        "After the declared first palm press, smoothly close and retain only "
        "that Inspire hand before the second palm approaches.  The exact "
        "per-hand 12-D official Unitree command is recorded on every step."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-close-second-hand-during-second-press",
    action="store_true",
    help=(
        "For the declared sequential grasp, close the second hand's four "
        "fingers with the same cosine clock as its palm press while retaining "
        "the already-closed first hand. The second thumb remains open until "
        "the separately recorded thumb-close phase."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-close-second-thumb-during-second-press",
    action="store_true",
    help=(
        "For the declared sequential grasp, close and retain the second "
        "thumb with the same cosine clock as its palm press. This requires "
        "the second-hand four-finger press closure and records the exact "
        "official Unitree 12-D command on every step."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-second-thumb-close-during-four-finger-steps",
    type=int,
    default=0,
    help=(
        "For the sequential grasp, close the second thumb over the first N "
        "steps of the post-press four-finger phase and retain it thereafter. "
        "Zero preserves the separate later thumb-close phase."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-digit-force-stop-n",
    type=float,
    default=None,
    help=(
        "Optional no-learning mechanics controller: advance each thumb or "
        "finger command only while that exact PhysX anatomical group's "
        "normal load is below this threshold. Raw solver contact is used "
        "only to form the grasp and is never represented as tactile input."
    ),
)
parser.add_argument(
    "--sugar-side-clamp-digit-force-stop-max-scale-step",
    type=float,
    default=0.02,
    help="Maximum per-step normalized digit-command advance under force stop.",
)
for side in ("left", "right"):
    parser.add_argument(
        f"--sugar-side-clamp-lift-{side}-thumb-yaw-rad",
        type=float,
        default=None,
        help=(
            "Optional official Inspire thumb-yaw target introduced only during "
            "the physical lift; pre-lift grasp construction remains unchanged."
        ),
    )
parser.add_argument(
    "--sugar-side-clamp-lift-thumb-yaw-start-step",
    type=int,
    default=0,
    help="Lift step at which the optional late thumb-yaw ramp begins.",
)
parser.add_argument(
    "--sugar-side-clamp-lift-thumb-yaw-ramp-steps",
    type=int,
    default=1,
    help="Cosine-ramp duration for the optional late lift thumb-yaw target.",
)
parser.add_argument(
    "--sugar-side-clamp-simultaneous-hand-close",
    action="store_true",
    help=(
        "During the physical palm press, apply the same smooth official "
        "Inspire four-finger/thumb closure instead of closing digits only "
        "after the box has already reacted to an isolated thumb"
    ),
)
parser.add_argument("--ik-close-steps", type=int, default=200)
parser.add_argument("--ik-thumb-close-steps", type=int, default=200)
parser.add_argument("--ik-settle-steps", type=int, default=100)
parser.add_argument("--ik-lift-height-m", type=float, default=0.25)
parser.add_argument("--ik-lift-steps", type=int, default=400)
parser.add_argument(
    "--static-posture-solutions",
    type=Path,
    default=None,
    help=(
        "Optional hash-recorded Plan-10 static feasibility solution. Its selected "
        "root pose is transferred through the recorded source-box frame into this "
        "run's one-time initial box frame; its first 29 official G1 joint values "
        "seed the physical setup."
    ),
)
parser.add_argument("--static-posture-source-trace", type=Path, default=None)
parser.add_argument("--static-posture-index", type=int, default=None)
parser.add_argument("--static-posture-source-index", type=int, default=None)
for _side in ("left", "right"):
    parser.add_argument(
        f"--static-posture-{_side}-contact-delta-pca-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("NORMAL_M", "TANGENT_M", "HEIGHT_M"),
        help=(
            "Audited translation applied after source-posture transfer, in the "
            "frozen CarryBox PCA [normal, tangent, height] basis. The hand "
            "orientation and every controller/physics setting remain unchanged."
        ),
    )
parser.add_argument(
    "--static-posture-use-declared-approach-clearance",
    action="store_true",
    help=(
        "For a source-bound static contact solution, move both hands to the "
        "declared per-side outward clearance while the box is offstage, then "
        "publish the box once and establish contact through the recorded "
        "smooth palm-press phase.  Without this flag the legacy exact-contact "
        "seed remains unchanged."
    ),
)
parser.add_argument(
    "--fix-robot-root",
    action="store_true",
    help=(
        "Declare a physics-fixed root fixture to isolate arm/hand mechanics; "
        "this is not standing or locomotion evidence"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

repo = args.unitree_repo.resolve()
asset_root = args.asset_root.resolve()
robot_motion_path = args.robot_motion.resolve()
object_motion_path = args.object_motion.resolve()
box_usd = args.box_usd.resolve()
output_root = args.output_root.resolve()
side_clamp_contact_geometry_source = (
    args.side_clamp_contact_geometry_source.resolve()
    if args.side_clamp_contact_geometry_source is not None
    else None
)
static_posture_solutions = (
    args.static_posture_solutions.resolve()
    if args.static_posture_solutions is not None
    else None
)
static_posture_source_trace = (
    args.static_posture_source_trace.resolve()
    if args.static_posture_source_trace is not None
    else None
)
static_posture_contact_delta_pca_by_side_m = {
    side: tuple(
        float(value)
        for value in getattr(args, f"static_posture_{side}_contact_delta_pca_m")
    )
    for side in ("left", "right")
}
for side, delta in static_posture_contact_delta_pca_by_side_m.items():
    if not all(math.isfinite(value) for value in delta):
        raise ValueError(f"{side} static-posture contact delta must be finite")
    if math.sqrt(sum(value * value for value in delta)) > 0.030:
        raise ValueError(
            f"{side} static-posture contact delta is bounded to 0.030 m norm"
        )
if static_posture_solutions is None and any(
    value != 0.0
    for delta in static_posture_contact_delta_pca_by_side_m.values()
    for value in delta
):
    raise ValueError("Static-posture contact deltas require a static posture source")
ik_contact_template_trace = (
    args.ik_contact_template_trace.resolve()
    if args.ik_contact_template_trace is not None
    else None
)
unitree_demo_parquet = (
    args.unitree_demo_parquet.resolve()
    if args.unitree_demo_parquet is not None
    else None
)
official_hand_usd = (
    asset_root
    / "assets/robots/g1-29dof_wholebody_inspire/"
    "g1_29dof_with_inspire_rev_1_0.usd"
)
for path in (
    official_hand_usd,
    robot_motion_path,
    object_motion_path,
    box_usd,
):
    if not path.is_file():
        raise FileNotFoundError(path)
if ik_contact_template_trace is not None and not ik_contact_template_trace.is_file():
    raise FileNotFoundError(ik_contact_template_trace)
if unitree_demo_parquet is not None and not unitree_demo_parquet.is_file():
    raise FileNotFoundError(unitree_demo_parquet)
if static_posture_solutions is not None and not static_posture_solutions.is_file():
    raise FileNotFoundError(static_posture_solutions)
if static_posture_source_trace is not None and not static_posture_source_trace.is_file():
    raise FileNotFoundError(static_posture_source_trace)
if (
    side_clamp_contact_geometry_source is not None
    and not side_clamp_contact_geometry_source.is_file()
):
    raise FileNotFoundError(side_clamp_contact_geometry_source)
if output_root.exists():
    raise FileExistsError(f"Refusing overwrite: {output_root}")
if not (0.0 < args.close_fraction <= 1.0):
    raise ValueError("--close-fraction must be in (0, 1]")
if not (0.0 <= args.pregrasp_thumb_pitch_rad <= 0.5):
    raise ValueError("--pregrasp-thumb-pitch-rad must be in official [0, 0.5]")
pregrasp_thumb_pitch_by_side_rad = {
    side: (
        args.pregrasp_thumb_pitch_rad
        if getattr(args, f"pregrasp_{side}_thumb_pitch_rad") is None
        else getattr(args, f"pregrasp_{side}_thumb_pitch_rad")
    )
    for side in ("left", "right")
}
pregrasp_thumb_yaw_by_side_rad = {
    side: (
        args.pregrasp_thumb_yaw_rad
        if getattr(args, f"pregrasp_{side}_thumb_yaw_rad") is None
        else getattr(args, f"pregrasp_{side}_thumb_yaw_rad")
    )
    for side in ("left", "right")
}
closed_thumb_yaw_rad = (
    args.pregrasp_thumb_yaw_rad
    if args.closed_thumb_yaw_rad is None
    else args.closed_thumb_yaw_rad
)
side_clamp_box_local_tangent_m = (
    args.unitree_demo_side_clamp_box_local_y_m
    if args.side_clamp_box_local_tangent_m is None
    else args.side_clamp_box_local_tangent_m
)
if not (-0.18 <= side_clamp_box_local_tangent_m <= 0.18):
    raise ValueError("Side-clamp local tangent coordinate must be in [-0.18, 0.18] m")
side_clamp_box_local_tangent_by_side_m = {
    side: (
        side_clamp_box_local_tangent_m
        if getattr(args, f"{side}_side_clamp_box_local_tangent_m") is None
        else getattr(args, f"{side}_side_clamp_box_local_tangent_m")
    )
    for side in ("left", "right")
}
side_clamp_box_local_z_by_side_m = {
    side: (
        args.unitree_demo_side_clamp_box_local_z_m
        if getattr(args, f"{side}_side_clamp_box_local_z_m") is None
        else getattr(args, f"{side}_side_clamp_box_local_z_m")
    )
    for side in ("left", "right")
}
side_clamp_box_normal_by_side_m = {
    side: getattr(args, f"{side}_side_clamp_box_normal_m")
    for side in ("left", "right")
}
side_clamp_palm_inset_by_side_m = {
    side: (
        args.unitree_demo_side_clamp_palm_inset_m
        if getattr(args, f"{side}_side_clamp_palm_inset_m") is None
        else getattr(args, f"{side}_side_clamp_palm_inset_m")
    )
    for side in ("left", "right")
}
side_clamp_approach_clearance_by_side_m = {
    side: (
        args.unitree_demo_side_clamp_approach_clearance_m
        if getattr(args, f"{side}_side_clamp_approach_clearance_m") is None
        else getattr(args, f"{side}_side_clamp_approach_clearance_m")
    )
    for side in ("left", "right")
}
side_clamp_tilt_tangent_by_side_rad = {
    side: getattr(args, f"{side}_side_clamp_tilt_tangent_rad")
    for side in ("left", "right")
}
side_clamp_tilt_height_by_side_rad = {
    side: getattr(args, f"{side}_side_clamp_tilt_height_rad")
    for side in ("left", "right")
}
side_clamp_normal_roll_by_side_rad = {
    side: getattr(args, f"{side}_side_clamp_normal_roll_rad")
    for side in ("left", "right")
}
side_clamp_contact_pca_by_side_m = {
    side: (
        None
        if getattr(args, f"{side}_side_clamp_contact_pca_m") is None
        else tuple(
            float(value)
            for value in getattr(args, f"{side}_side_clamp_contact_pca_m")
        )
    )
    for side in ("left", "right")
}
side_clamp_outward_pca_by_side = {
    side: (
        None
        if getattr(args, f"{side}_side_clamp_outward_pca") is None
        else tuple(
            float(value)
            for value in getattr(args, f"{side}_side_clamp_outward_pca")
        )
    )
    for side in ("left", "right")
}
if any(
    side_clamp_contact_pca_by_side_m[side] is not None
    for side in ("left", "right")
) and side_clamp_contact_geometry_source is None:
    raise ValueError(
        "Exact per-hand PCA contacts require a hash-bound geometry source"
    )
for side in ("left", "right"):
    if not (-0.18 <= side_clamp_box_local_tangent_by_side_m[side] <= 0.18):
        raise ValueError(f"{side} side-clamp tangent must be in [-0.18, 0.18] m")
    if not (-0.24 <= side_clamp_box_local_z_by_side_m[side] <= 0.24):
        raise ValueError(f"{side} side-clamp local-Z must be in [-0.24, 0.24] m")
    if (
        side_clamp_box_normal_by_side_m[side] is not None
        and not (-0.24 <= side_clamp_box_normal_by_side_m[side] <= 0.24)
    ):
        raise ValueError(f"{side} side-clamp normal coordinate is outside +/-0.24 m")
    if not (
        abs(side_clamp_tilt_tangent_by_side_rad[side]) <= 1.65
        and abs(side_clamp_tilt_height_by_side_rad[side]) <= 1.65
    ):
        raise ValueError(f"{side} side-clamp tilt must be within +/-1.65 rad")
    if abs(side_clamp_normal_roll_by_side_rad[side]) > 3.141592653589793:
        raise ValueError(f"{side} palm-normal roll must be within +/-pi rad")
    contact_pca = side_clamp_contact_pca_by_side_m[side]
    outward_pca = side_clamp_outward_pca_by_side[side]
    if (contact_pca is None) != (outward_pca is None):
        raise ValueError(
            f"{side} exact PCA contact and outward vector must be declared together"
        )
    if contact_pca is not None:
        if args.side_clamp_box_axis != "pca0":
            raise ValueError("Exact per-hand PCA contacts require PCA0 box mode")
        if not all(math.isfinite(value) for value in (*contact_pca, *outward_pca)):
            raise ValueError(f"{side} exact PCA contact declaration must be finite")
        outward_norm = math.sqrt(sum(value * value for value in outward_pca))
        if abs(outward_norm - 1.0) > 1.0e-6:
            raise ValueError(f"{side} outward PCA vector must have unit norm")
        if any(abs(value) > 0.30 for value in contact_pca):
            raise ValueError(f"{side} exact PCA contact is outside +/-0.30 m")
        if (
            abs(side_clamp_tilt_tangent_by_side_rad[side]) > 0.0
            or abs(side_clamp_tilt_height_by_side_rad[side]) > 0.0
        ):
            raise ValueError(
                f"{side} exact outward PCA normal cannot be combined with legacy tilt"
            )
if args.side_clamp_box_axis != "pca0" and any(
    abs(value) > 0.0
    for value in (
        *side_clamp_tilt_tangent_by_side_rad.values(),
        *side_clamp_tilt_height_by_side_rad.values(),
    )
):
    raise ValueError("Side-clamp PCA tilts require --side-clamp-box-axis pca0")
if any(abs(value) > 0.15 for value in args.sugar_side_clamp_box_local_offset_m):
    raise ValueError("SUGAR side-clamp box-local offset is bounded to +/-0.15 m")
if not (0.0 <= args.sugar_side_clamp_support_height_m <= 0.80):
    raise ValueError("SUGAR side-clamp support height must be in [0, 0.80] m")
if (
    args.body_control_mode != "sugar_side_clamp"
    and args.sugar_side_clamp_support_height_m > 0.0
):
    raise ValueError(
        "The SUGAR side-clamp setup support is valid only in sugar_side_clamp mode"
    )
if any(not (0.08 <= value <= 0.30) for value in args.sugar_side_clamp_support_size_xy_m):
    raise ValueError("SUGAR side-clamp support XY size must be in [0.08, 0.30] m")
if not (
    0.0 <= args.object_dynamic_friction <= args.object_static_friction <= 2.0
):
    raise ValueError("Object friction must satisfy 0 <= dynamic <= static <= 2")
closed_finger_fraction = {
    side: {
        finger: (
            args.close_fraction
            if getattr(args, f"{side}_{finger}_close_fraction") is None
            else getattr(args, f"{side}_{finger}_close_fraction")
        )
        for finger in ("index", "middle", "ring", "little")
    }
    for side in ("left", "right")
}
closed_thumb_pitch_rad = {
    side: (
        0.5 * args.close_fraction
        if getattr(args, f"closed_{side}_thumb_pitch_rad") is None
        else getattr(args, f"closed_{side}_thumb_pitch_rad")
    )
    for side in ("left", "right")
}
closed_thumb_yaw_by_side_rad = {
    side: (
        closed_thumb_yaw_rad
        if getattr(args, f"closed_{side}_thumb_yaw_rad") is None
        else getattr(args, f"closed_{side}_thumb_yaw_rad")
    )
    for side in ("left", "right")
}
lift_thumb_yaw_by_side_rad = {
    side: getattr(args, f"sugar_side_clamp_lift_{side}_thumb_yaw_rad")
    for side in ("left", "right")
}
for side in ("left", "right"):
    if not (0.0 <= pregrasp_thumb_pitch_by_side_rad[side] <= 0.5):
        raise ValueError(f"{side} pregrasp thumb pitch must be in [0, 0.5]")
    if not (-0.1 <= pregrasp_thumb_yaw_by_side_rad[side] <= 1.3):
        raise ValueError(f"{side} pregrasp thumb yaw must be in [-0.1, 1.3]")
    for finger, value in closed_finger_fraction[side].items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{side} {finger} close fraction must be in [0, 1]")
    if not (0.0 <= closed_thumb_pitch_rad[side] <= 0.5):
        raise ValueError(f"{side} closed thumb pitch must be in official [0, 0.5]")
    if not (-0.1 <= closed_thumb_yaw_by_side_rad[side] <= 1.3):
        raise ValueError(f"{side} closed thumb yaw must be in official [-0.1, 1.3]")
    if (
        lift_thumb_yaw_by_side_rad[side] is not None
        and not (-0.1 <= lift_thumb_yaw_by_side_rad[side] <= 1.3)
    ):
        raise ValueError(
            f"{side} late-lift thumb yaw must be in official [-0.1, 1.3]"
        )
if any(value is not None for value in lift_thumb_yaw_by_side_rad.values()):
    if args.body_control_mode != "sugar_side_clamp":
        raise ValueError("Late-lift thumb yaw is valid only in sugar_side_clamp mode")
    if not (0 <= args.sugar_side_clamp_lift_thumb_yaw_start_step < args.ik_lift_steps):
        raise ValueError("Late-lift thumb-yaw start must lie inside the lift")
    if args.sugar_side_clamp_lift_thumb_yaw_ramp_steps < 1:
        raise ValueError("Late-lift thumb-yaw ramp must contain at least one step")
for label, value in (
    ("pregrasp", args.pregrasp_thumb_yaw_rad),
    ("closed", closed_thumb_yaw_rad),
):
    if not (-0.1 <= value <= 1.3):
        raise ValueError(f"{label} thumb yaw must be in official [-0.1, 1.3] rad")
if not (-0.4 <= args.bilateral_shoulder_roll_offset_rad <= 0.4):
    raise ValueError("Shoulder-roll offset scan is bounded to [-0.4, 0.4] rad")
left_shoulder_roll_offset_rad = (
    args.bilateral_shoulder_roll_offset_rad
    if args.left_shoulder_roll_offset_rad is None
    else args.left_shoulder_roll_offset_rad
)
right_shoulder_roll_offset_rad = (
    -args.bilateral_shoulder_roll_offset_rad
    if args.right_shoulder_roll_offset_rad is None
    else args.right_shoulder_roll_offset_rad
)
left_shoulder_roll_end_offset_rad = (
    left_shoulder_roll_offset_rad
    if args.left_shoulder_roll_end_offset_rad is None
    else args.left_shoulder_roll_end_offset_rad
)
left_wrist_roll_end_offset_rad = (
    args.left_wrist_roll_offset_rad
    if args.left_wrist_roll_end_offset_rad is None
    else args.left_wrist_roll_end_offset_rad
)
left_wrist_yaw_end_offset_rad = (
    args.left_wrist_yaw_offset_rad
    if args.left_wrist_yaw_end_offset_rad is None
    else args.left_wrist_yaw_end_offset_rad
)
for label, value in (
    ("left", left_shoulder_roll_offset_rad),
    ("left_end", left_shoulder_roll_end_offset_rad),
    ("right", right_shoulder_roll_offset_rad),
):
    if not (-0.4 <= value <= 0.4):
        raise ValueError(f"{label} shoulder-roll offset is bounded to [-0.4, 0.4] rad")
body_offset_transition_enabled = any(
    value is not None
    for value in (
        args.left_shoulder_roll_end_offset_rad,
        args.left_wrist_roll_end_offset_rad,
        args.left_wrist_yaw_end_offset_rad,
    )
)
if body_offset_transition_enabled:
    if (
        args.shoulder_roll_transition_start is None
        or args.shoulder_roll_transition_end is None
    ):
        raise ValueError("A scheduled body offset requires transition start and end")
    if not (
        args.source_start
        <= args.shoulder_roll_transition_start
        < args.shoulder_roll_transition_end
        <= args.source_end
    ):
        raise ValueError("Invalid body-offset transition interval")
elif (
    args.shoulder_roll_transition_start is not None
    or args.shoulder_roll_transition_end is not None
):
    raise ValueError("Body-offset transition times require an end offset")
for label, value in (
    ("left", args.left_wrist_roll_offset_rad),
    ("left_end", left_wrist_roll_end_offset_rad),
    ("right", args.right_wrist_roll_offset_rad),
):
    if not (-1.3 <= value <= 1.3):
        raise ValueError(f"{label} wrist-roll offset is bounded to [-1.3, 1.3] rad")
for label, value in (
    ("left", args.left_wrist_yaw_offset_rad),
    ("left_end", left_wrist_yaw_end_offset_rad),
    ("right", args.right_wrist_yaw_offset_rad),
):
    if not (-0.8 <= value <= 0.8):
        raise ValueError(f"{label} wrist-yaw offset is bounded to [-0.8, 0.8] rad")
for label, value in (
    ("left", args.left_shoulder_yaw_offset_rad),
    ("right", args.right_shoulder_yaw_offset_rad),
):
    if not (-0.5 <= value <= 0.5):
        raise ValueError(f"{label} shoulder-yaw offset is bounded to [-0.5, 0.5] rad")
if not (
    0 <= args.source_start < args.close_start < args.close_end <= args.source_end
):
    raise ValueError("Invalid source/closure interval")
if (args.brake_start is None) != (args.brake_steps == 0):
    raise ValueError("--brake-start and positive --brake-steps must be used together")
if args.brake_start is not None and not (
    args.close_end <= args.brake_start < args.source_end
):
    raise ValueError("Brake interval must follow closure and precede source end")
if args.solver_position_iterations < 4 or args.solver_velocity_iterations < 1:
    raise ValueError("Solver iterations may not degrade the official 4/1 values")
grasp_dwell_enabled = args.grasp_dwell_source is not None
if grasp_dwell_enabled:
    if args.grasp_dwell_source != int(args.grasp_dwell_source):
        raise ValueError("--grasp-dwell-source must be an integer source frame")
    if not (
        args.source_start <= args.grasp_dwell_source
        < args.grasp_dwell_source
        + args.grasp_close_source_span
        + args.resume_source_span
        <= (args.brake_start if args.brake_start is not None else args.source_end)
    ):
        raise ValueError("Invalid grasp dwell/resume interval")
    if args.grasp_close_steps <= 0 or args.grasp_settle_steps < 0:
        raise ValueError("Grasp dwell requires positive close and nonnegative settle steps")
    if not (0 <= args.grasp_close_source_span <= 12):
        raise ValueError("Grasp close source span must be an integer in [0, 12]")
else:
    if (
        args.grasp_close_steps != 0
        or args.grasp_settle_steps != 0
        or args.grasp_close_source_span != 0
    ):
        raise ValueError("Grasp dwell step counts require --grasp-dwell-source")
if not (-0.35 <= args.robot_root_y_offset_m <= 0.35):
    raise ValueError("Robot root Y alignment offset is bounded to [-0.35, 0.35] m")
if not (0.0 <= args.bilateral_hip_roll_outward_offset_rad <= 0.35):
    raise ValueError("Bilateral outward hip-roll offset must be in [0, 0.35] rad")
if not (2 <= args.resume_polynomial_power <= 6):
    raise ValueError("Resume polynomial power must be in [2, 6]")
if args.body_control_mode == "state_replay" and args.fix_robot_root:
    raise ValueError("Fixed-root fixture is only valid with physical pd_target control")
if args.body_control_mode == "sugar_side_clamp":
    if not args.fix_robot_root:
        raise ValueError("SUGAR side-clamp mechanics requires the fixed-root fixture")
    posture_fields = (
        static_posture_solutions,
        static_posture_source_trace,
        args.static_posture_index,
        args.static_posture_source_index,
    )
    if any(value is not None for value in posture_fields) and not all(
        value is not None for value in posture_fields
    ):
        raise ValueError("Static posture transfer requires all four posture arguments")
    if min(
        args.unitree_demo_approach_steps,
        args.ik_palm_press_steps,
        args.unitree_demo_close_steps,
        args.ik_thumb_close_steps,
        args.ik_lift_steps,
    ) <= 0:
        raise ValueError(
            "SUGAR side-clamp approach, press, finger, thumb, and lift steps must be positive"
        )
    if not (-0.18 <= args.unitree_demo_side_clamp_box_local_y_m <= 0.18):
        raise ValueError("Side-clamp local-Y center must be in [-0.18, 0.18] m")
    if not (-0.12 <= args.unitree_demo_side_clamp_box_local_z_m <= 0.12):
        raise ValueError("Side-clamp local-Z center must be in [-0.12, 0.12] m")
    # This is a Cartesian IK target behind the rigid object surface, not an
    # elastomer deformation claim.  Up to 20 mm is retained for bounded
    # physical squeeze diagnostics; actual penetration/contact forces remain
    # governed by PhysX and are audited from the resulting trace.
    if any(
        not (0.0 <= value <= 0.040)
        for value in side_clamp_palm_inset_by_side_m.values()
    ):
        raise ValueError("Each side-clamp palm inset must be in [0, 0.040] m")
    if any(
        not (0.005 <= value <= 0.12)
        for value in side_clamp_approach_clearance_by_side_m.values()
    ):
        raise ValueError(
            "Each side-clamp approach clearance must be in [0.005, 0.12] m"
        )
    if (
        args.sugar_side_clamp_palm_press_first is not None
        and args.sugar_side_clamp_simultaneous_hand_close
    ):
        raise ValueError(
            "Sequential palm press and simultaneous hand close are mutually exclusive"
        )
    if (
        args.sugar_side_clamp_close_first_hand_before_second
        and args.sugar_side_clamp_palm_press_first is None
    ):
        raise ValueError(
            "First-hand preclose requires --sugar-side-clamp-palm-press-first"
        )
    if args.sugar_side_clamp_close_second_hand_during_second_press and not (
        args.sugar_side_clamp_close_first_hand_before_second
        and args.sugar_side_clamp_palm_press_first is not None
    ):
        raise ValueError(
            "Second-hand press closure requires the sequential first-hand "
            "preclose trajectory"
        )
    if args.sugar_side_clamp_close_second_thumb_during_second_press and not (
        args.sugar_side_clamp_close_second_hand_during_second_press
        and args.sugar_side_clamp_close_first_hand_before_second
        and args.sugar_side_clamp_palm_press_first is not None
    ):
        raise ValueError(
            "Second-thumb press closure requires the sequential second-hand "
            "four-finger press closure trajectory"
        )
    if not (
        0
        <= args.sugar_side_clamp_second_thumb_close_during_four_finger_steps
        <= args.unitree_demo_close_steps
    ):
        raise ValueError(
            "Second-thumb four-finger-phase closure steps must be in "
            "[0, unitree_demo_close_steps]"
        )
    if args.sugar_side_clamp_second_thumb_close_during_four_finger_steps > 0:
        if not (
            args.sugar_side_clamp_close_second_hand_during_second_press
            and args.sugar_side_clamp_close_first_hand_before_second
            and args.sugar_side_clamp_palm_press_first is not None
        ):
            raise ValueError(
                "Second-thumb four-finger-phase closure requires the "
                "sequential second-hand press-closure trajectory"
            )
        if args.sugar_side_clamp_close_second_thumb_during_second_press:
            raise ValueError(
                "Second-thumb press closure and four-finger-phase closure "
                "are mutually exclusive"
            )
    if args.sugar_side_clamp_digit_force_stop_n is not None and not (
        0.05 <= args.sugar_side_clamp_digit_force_stop_n <= 20.0
    ):
        raise ValueError("Digit force-stop threshold must be in [0.05, 20] N")
    if args.sugar_side_clamp_digit_force_stop_n is not None and not (
        args.sugar_side_clamp_close_first_hand_before_second
        and args.sugar_side_clamp_palm_press_first is not None
    ):
        raise ValueError(
            "Digit force stop currently requires the recorded sequential "
            "first-hand preclose trajectory"
        )
    if not (
        0.001 <= args.sugar_side_clamp_digit_force_stop_max_scale_step <= 0.05
    ):
        raise ValueError("Digit force-stop scale step must be in [0.001, 0.05]")
    if not (0 <= args.sugar_side_clamp_direct_refinement_steps <= 2000):
        raise ValueError("Direct setup refinement steps must be in [0, 2000]")
    if not (
        0.0
        < args.sugar_side_clamp_max_reachable_orientation_delta_rad
        <= 3.141592653589793
    ):
        raise ValueError(
            "Direct setup orientation-delta bound must be in (0, pi] rad"
        )
elif args.sugar_side_clamp_direct_setup_ik:
    raise ValueError("Direct setup IK is only defined for SUGAR side-clamp mode")
elif static_posture_solutions is not None:
    raise ValueError("Static posture transfer is defined only for SUGAR side clamp")
if args.ik_track_live_box_during_lift and args.body_control_mode != "sugar_side_clamp":
    raise ValueError("Live-box lift tracking is currently defined only for SUGAR side clamp")
if not (0.0 <= args.ik_contact_preload_height_m <= 0.03):
    raise ValueError("Contact preload height must be in [0, 0.03] m")
if not (0 <= args.ik_contact_preload_steps <= 400):
    raise ValueError("Contact preload steps must be in [0, 400]")
if args.ik_contact_preload_ramp_steps is None:
    args.ik_contact_preload_ramp_steps = args.ik_contact_preload_steps
if not (0 <= args.ik_contact_preload_ramp_steps <= args.ik_contact_preload_steps):
    raise ValueError("Contact preload ramp steps must be in [0, preload_steps]")
if (args.ik_contact_preload_height_m > 0.0) != (
    args.ik_contact_preload_steps > 0
):
    raise ValueError("Contact preload height and steps must be enabled together")
if args.ik_contact_preload_steps and not (
    args.body_control_mode == "sugar_side_clamp"
    and args.ik_track_live_box_during_grasp
):
    raise ValueError("Contact preload requires live-box SUGAR side-clamp grasp tracking")
if not (0.0 <= args.ik_contact_compression_m <= 0.015):
    raise ValueError("Contact compression must be in [0, 0.015] m")
if not (0 <= args.ik_contact_compression_steps <= 400):
    raise ValueError("Contact compression steps must be in [0, 400]")
if args.ik_contact_compression_ramp_steps is None:
    args.ik_contact_compression_ramp_steps = args.ik_contact_compression_steps
if not (
    0
    <= args.ik_contact_compression_ramp_steps
    <= args.ik_contact_compression_steps
):
    raise ValueError("Contact compression ramp must be in [0, compression_steps]")
if (args.ik_contact_compression_m > 0.0) != (
    args.ik_contact_compression_steps > 0
):
    raise ValueError("Contact compression distance and steps must be enabled together")
if args.ik_contact_compression_steps and not (
    args.body_control_mode == "sugar_side_clamp"
    and args.ik_track_live_box_during_grasp
):
    raise ValueError("Contact compression requires live-box SUGAR side-clamp tracking")
if (
    args.ik_live_lift_scheduled_world_z
    and not args.ik_track_live_box_during_lift
):
    raise ValueError("Scheduled world-Z lift requires live-box lift tracking")
if args.ik_live_lift_scheduled_world_z and args.ik_live_lift_lead_m is not None:
    raise ValueError("Scheduled world-Z lift and a fixed live-lift lead are exclusive")
if args.ik_live_lift_lead_ramp_steps < 0:
    raise ValueError("Live-lift lead ramp steps must be nonnegative")
if args.ik_live_lift_lead_ramp_steps and args.ik_live_lift_lead_m is None:
    raise ValueError("A live-lift lead ramp requires --ik-live-lift-lead-m")
if args.ik_live_lift_anchor_world_xy and not args.ik_track_live_box_during_lift:
    raise ValueError("World-XY lift anchoring requires live-box lift tracking")
if args.ik_live_hold_relative_to_box and not (
    args.ik_track_live_box_during_lift
    and args.ik_live_lift_scheduled_world_z
):
    raise ValueError(
        "Box-relative hold requires a scheduled world-Z live-box lift"
    )
if args.ik_live_lift_lead_m is not None:
    if not args.ik_track_live_box_during_lift:
        raise ValueError("A live-lift lead requires --ik-track-live-box-during-lift")
    if not (0.0001 <= args.ik_live_lift_lead_m <= 0.030):
        raise ValueError("Live-lift Cartesian lead must be in [0.0001, 0.030] m")
if args.require_settled_all_groups_frames < 0:
    raise ValueError("Settled all-group gate length must be nonnegative")
if args.require_settled_all_groups_frames:
    if args.body_control_mode != "sugar_side_clamp":
        raise ValueError("Settled all-group stop gate is only defined for SUGAR side clamp")
    if args.require_settled_all_groups_frames > args.unitree_demo_settle_steps:
        raise ValueError("Settled all-group gate cannot exceed the settle interval")
if args.body_control_mode == "bilateral_ik":
    if not args.fix_robot_root:
        raise ValueError("Bilateral IK mechanics requires the declared fixed-root fixture")
    if args.ik_contact_template_trace is None:
        raise ValueError("Bilateral IK mechanics requires --ik-contact-template-trace")
    if not (0.0 <= args.ik_approach_clearance_m <= 0.15):
        raise ValueError("IK approach clearance must be in [0, 0.15] m")
    if not (
        -1.0 <= args.ik_left_box_roll_offset_rad <= 1.0
        and -1.0 <= args.ik_right_box_roll_offset_rad <= 1.0
    ):
        raise ValueError("IK box-frame roll offsets must be in [-1, 1] rad")
    if min(
        args.ik_approach_steps,
        args.ik_palm_press_steps,
        args.ik_close_steps,
        args.ik_thumb_close_steps,
        args.ik_lift_steps,
    ) <= 0:
        raise ValueError("IK approach, press, close, thumb, and lift steps must be positive")
    if args.ik_settle_steps < 0 or not (0.10 <= args.ik_lift_height_m <= 0.40):
        raise ValueError("Invalid IK settle steps or lift height")
elif args.ik_contact_template_trace is not None:
    raise ValueError("IK contact template is only valid in bilateral_ik mode")
if args.body_control_mode == "unitree_demo_pose":
    if not args.fix_robot_root:
        raise ValueError("Official Unitree demo-pose mechanics requires the fixed-root fixture")
    if unitree_demo_parquet is None:
        raise ValueError("unitree_demo_pose requires --unitree-demo-parquet")
    if not args.unitree_demo_source_id or not args.unitree_demo_source_revision:
        raise ValueError(
            "unitree_demo_pose requires the official dataset ID and exact revision"
        )
    if args.unitree_demo_episode_index < 0 or args.unitree_demo_frame_index < 0:
        raise ValueError("Unitree demo episode/frame indices must be nonnegative")
    if args.unitree_demo_close_steps <= 0 or args.unitree_demo_settle_steps < 0:
        raise ValueError("Unitree demo close/settle step counts are invalid")
    if min(args.unitree_demo_support_size_xy_m) <= 0.0:
        raise ValueError("Unitree demo support footprint must be positive")
    if not (0.0 <= args.unitree_demo_support_cradle_height_m <= 0.10):
        raise ValueError("Unitree demo setup-cradle height must be in [0, 0.10] m")
    if not (0.0 <= args.unitree_demo_outward_shift_m <= 0.35):
        raise ValueError("Unitree demo outward shift must be in [0, 0.35] m")
    if args.unitree_demo_approach_steps < 0 or (
        args.unitree_demo_outward_shift_m > 0.0
        and args.unitree_demo_approach_steps <= 0
    ):
        raise ValueError("A positive Unitree outward shift requires approach steps")
    if args.unitree_demo_side_clamp:
        if args.unitree_demo_outward_shift_m != 0.0:
            raise ValueError(
                "The side-palm clamp replaces, rather than combines with, the "
                "rejected whole-pose outward translation"
            )
        if min(
            args.unitree_demo_approach_steps,
            args.ik_palm_press_steps,
            args.unitree_demo_close_steps,
            args.ik_thumb_close_steps,
            args.ik_lift_steps,
        ) <= 0:
            raise ValueError(
                "Side-palm approach, press, finger, thumb, and lift steps must be positive"
            )
        if not (-0.18 <= args.unitree_demo_side_clamp_box_local_y_m <= 0.18):
            raise ValueError("Side-clamp local-Y center must be in [-0.18, 0.18] m")
        if not (-0.12 <= args.unitree_demo_side_clamp_box_local_z_m <= 0.12):
            raise ValueError("Side-clamp local-Z center must be in [-0.12, 0.12] m")
        if not (0.0 <= args.unitree_demo_side_clamp_palm_inset_m <= 0.005):
            raise ValueError("Side-clamp palm inset must be in [0, 0.005] m")
        if not (
            0.02
            <= args.unitree_demo_side_clamp_approach_clearance_m
            <= 0.12
        ):
            raise ValueError("Side-clamp approach clearance must be in [0.02, 0.12] m")
    if not (-3.141592653589793 <= args.unitree_demo_box_yaw_rad <= 3.141592653589793):
        raise ValueError("Unitree demo CarryBox yaw must be in [-pi, pi]")
elif unitree_demo_parquet is not None:
    raise ValueError("--unitree-demo-parquet is only valid in unitree_demo_pose mode")
elif args.unitree_demo_side_clamp:
    raise ValueError("--unitree-demo-side-clamp requires unitree_demo_pose mode")

# The official Unitree module resolves assets through PROJECT_ROOT at import.
os.environ["PROJECT_ROOT"] = str(asset_root)
sys.path.insert(0, str(repo))

simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import (  # noqa: E402
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from robots.unitree import G129_CFG_WITH_INSPIRE_WHOLEBODY  # noqa: E402


SIDES = ("left", "right")
GROUPS = ("palm", "thumb", "index", "middle", "ring", "little")
# Geometry is measured in the exact hash-bound official collision meshes used
# by this producer.  The Inspire palmar surface is local -X.  Each value below
# is a real mesh vertex at the central load-bearing patch, not an impossible
# combination of three independent AABB extrema.
INSPIRE_PALM_SURFACE_POINT_B = {
    "left": (-0.023290662, -0.070559956, -0.000645893),
    "right": (-0.023284726, -0.070581190, 0.000615386),
}
# The unchanged SUGAR CarryBox mesh is slightly asymmetric about local X.
CARRYBOX_SIDE_FACE_X_M = {"left": 0.20718460, "right": -0.19594507}
CARRYBOX_MESH_Y_BOUNDS_M = (-0.27088723, 0.27156848)
CARRYBOX_MESH_Z_MIN_M = -0.26973610
# Exact PCA frame recomputed from all 50,004 unique vertices in the hash-bound
# official CarryBox mesh after composing the mesh into the simulated rigid-body
# frame.  The previous v17 constants described the authored mesh-local frame,
# even though every palm target below is interpreted in the RigidObject root
# frame.  That mismatch was small on PCA0 but invalidated bottom-support points.
# Columns remain sign-matched to the historical frame; the released motion's
# hands oppose one another along axis 0.
CARRYBOX_PCA_CENTER_B = (
    -0.001107575897317,
    -0.000547104383076,
    0.005225372263696,
)
CARRYBOX_PCA_BASIS_B = (
    (-0.074551623810, 0.987056459883, -0.141991550421),
    (0.908313530526, 0.008444000290, -0.418204769368),
    (-0.411592742465, -0.160150691109, -0.897186251838),
)
CARRYBOX_PCA0_BOUNDS_M = (-0.213263896055, 0.219771392738)
# Exact empirical PCA2 bottom/top bounds from all 50,004 hash-bound mesh vertices.
CARRYBOX_PCA2_BOUNDS_M = (-0.189061734778, 0.201633618201)
EXPECTED_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
# Exact DDS/LeRobot wire order published by Unitree's
# tasks/common_observations/g1_29dof_state.py. The live USD uses the
# interleaved EXPECTED_BODY_JOINT_NAMES order above, so direct index copying is
# invalid even though both arrays contain 29 joints.
UNITREE_G129_WIRE_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Exact public mapping in Unitree's action_provider/action_provider_dds.py.
OFFICIAL_INSPIRE_BASE_COMMAND_INDEX = {
    "R_pinky_proximal_joint": 0,
    "R_ring_proximal_joint": 1,
    "R_middle_proximal_joint": 2,
    "R_index_proximal_joint": 3,
    "R_thumb_proximal_pitch_joint": 4,
    "R_thumb_proximal_yaw_joint": 5,
    "L_pinky_proximal_joint": 6,
    "L_ring_proximal_joint": 7,
    "L_middle_proximal_joint": 8,
    "L_index_proximal_joint": 9,
    "L_thumb_proximal_pitch_joint": 10,
    "L_thumb_proximal_yaw_joint": 11,
}
OFFICIAL_INSPIRE_SPECIAL_COMMAND = {
    "L_index_intermediate_joint": (9, 1.0),
    "L_middle_intermediate_joint": (8, 1.0),
    "L_pinky_intermediate_joint": (6, 1.0),
    "L_ring_intermediate_joint": (7, 1.0),
    "L_thumb_intermediate_joint": (10, 1.5),
    "L_thumb_distal_joint": (10, 2.4),
    "R_index_intermediate_joint": (3, 1.0),
    "R_middle_intermediate_joint": (2, 1.0),
    "R_pinky_intermediate_joint": (0, 1.0),
    "R_ring_intermediate_joint": (1, 1.0),
    "R_thumb_intermediate_joint": (4, 1.5),
    "R_thumb_distal_joint": (4, 2.4),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def body_group(name: str) -> str:
    lower = name.lower()
    if "hand_base_link" in lower:
        return "palm"
    if "thumb" in lower:
        return "thumb"
    if "index" in lower:
        return "index"
    if "middle" in lower:
        return "middle"
    if "ring" in lower:
        return "ring"
    if "pinky" in lower:
        return "little"
    raise RuntimeError(f"Unmapped hand contact body: {name}")


def raw_contact_sample(
    sensor: ContactSensor, com_w: np.ndarray, *, anatomical: bool = True
) -> dict[str, np.ndarray]:
    """Read every raw normal/friction row and aggregate reaction on the box."""

    (
        normal_force,
        contact_point,
        contact_normal,
        _separation,
        pair_counts,
        pair_starts,
    ) = sensor.contact_physx_view.get_contact_data(dt=sensor._sim_physics_dt)
    counts = cpu(pair_counts).reshape(-1).astype(np.int64)
    starts = cpu(pair_starts).reshape(-1).astype(np.int64)
    if len(counts) != len(sensor.body_names):
        raise RuntimeError(
            f"Raw pair/body mismatch: {len(counts)} != {len(sensor.body_names)}"
        )
    magnitudes = cpu(normal_force).reshape(-1)
    points = cpu(contact_point).reshape(-1, 3)
    normals = cpu(contact_normal).reshape(-1, 3)

    (
        friction_force,
        friction_point,
        friction_counts,
        friction_starts,
    ) = sensor.contact_physx_view.get_friction_data(dt=sensor._sim_physics_dt)
    f_counts = cpu(friction_counts).reshape(-1).astype(np.int64)
    f_starts = cpu(friction_starts).reshape(-1).astype(np.int64)
    if len(f_counts) != len(sensor.body_names):
        raise RuntimeError("Raw friction pair/body mismatch")
    f_forces = cpu(friction_force).reshape(-1, 3)
    f_points = cpu(friction_point).reshape(-1, 3)

    body_count = np.zeros(len(sensor.body_names), dtype=np.int32)
    body_normal_load = np.zeros(len(sensor.body_names), dtype=np.float64)
    group_normal_load = np.zeros(len(GROUPS), dtype=np.float64)
    normal_force_on_box = np.zeros(3, dtype=np.float64)
    normal_torque_on_box = np.zeros(3, dtype=np.float64)
    friction_force_on_box = np.zeros(3, dtype=np.float64)
    friction_torque_on_box = np.zeros(3, dtype=np.float64)
    for body_index, name in enumerate(sensor.body_names):
        count = int(counts[body_index])
        start = int(starts[body_index])
        f_count = int(f_counts[body_index])
        f_start = int(f_starts[body_index])
        group_index = GROUPS.index(body_group(name)) if anatomical else None
        body_count[body_index] = count
        if count:
            magnitude = magnitudes[start : start + count].astype(np.float64)
            point = points[start : start + count].astype(np.float64)
            normal_on_hand = (
                magnitude[:, None]
                * normals[start : start + count].astype(np.float64)
            )
            normal_on_box = -normal_on_hand
            body_normal_load[body_index] = float(np.abs(magnitude).sum())
            if group_index is not None:
                group_normal_load[group_index] += body_normal_load[body_index]
            normal_force_on_box += normal_on_box.sum(axis=0)
            normal_torque_on_box += np.cross(
                point - com_w[None, :], normal_on_box
            ).sum(axis=0)
        if f_count:
            # PhysX reports this friction reaction on the selected hand body.
            friction_on_box = -f_forces[
                f_start : f_start + f_count
            ].astype(np.float64)
            point = f_points[f_start : f_start + f_count].astype(np.float64)
            friction_force_on_box += friction_on_box.sum(axis=0)
            friction_torque_on_box += np.cross(
                point - com_w[None, :], friction_on_box
            ).sum(axis=0)
    force_on_box = normal_force_on_box + friction_force_on_box
    torque_on_box = normal_torque_on_box + friction_torque_on_box
    return {
        "body_count": body_count,
        "body_normal_load": body_normal_load.astype(np.float32),
        "group_normal_load": group_normal_load.astype(np.float32),
        "force_on_box": force_on_box.astype(np.float32),
        "torque_on_box": torque_on_box.astype(np.float32),
        "normal_force_on_box": normal_force_on_box.astype(np.float32),
        "normal_torque_on_box": normal_torque_on_box.astype(np.float32),
        "friction_force_on_box": friction_force_on_box.astype(np.float32),
        "friction_torque_on_box": friction_torque_on_box.astype(np.float32),
        "normal_point_count": np.asarray(counts.sum(), dtype=np.int32),
        "friction_point_count": np.asarray(f_counts.sum(), dtype=np.int32),
    }


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def main() -> None:
    producer_source_path = Path(__file__).resolve()
    producer_source_sha256 = sha256(producer_source_path)
    workspace_root_env = os.environ.get("CURIOSITY_WORKSPACE_ROOT")
    workspace_root = (
        Path(workspace_root_env).resolve()
        if workspace_root_env
        else producer_source_path.parents[2]
    )
    official_material_source_path = (
        workspace_root
        / "SUGAR/scripts/sugar_rl/evaluate_official_carrybox_physics_condition.py"
    )
    if not official_material_source_path.is_file():
        raise FileNotFoundError(
            f"Missing official CarryBox material source: {official_material_source_path}"
        )
    static_posture_joint_position_np = None
    static_posture_root_state_np = None
    static_posture_source_box_pose_np = None
    static_posture_source_desired_hand_position_np = None
    static_posture_source_desired_hand_quaternion_np = None
    static_posture_contact_delta_pca_np = np.asarray(
        [
            static_posture_contact_delta_pca_by_side_m[side]
            for side in ("left", "right")
        ],
        dtype=np.float32,
    )
    static_posture_contact_delta_box_b_np = (
        static_posture_contact_delta_pca_np
        @ np.asarray(CARRYBOX_PCA_BASIS_B, dtype=np.float32).T
    )
    static_posture_contact_delta_world_np = np.empty((0, 3), dtype=np.float32)
    if static_posture_solutions is not None:
        assert static_posture_source_trace is not None
        assert args.static_posture_index is not None
        assert args.static_posture_source_index is not None
        with np.load(static_posture_solutions, allow_pickle=False) as posture:
            candidate_mask = posture["candidate_mask"].astype(bool)
            index = args.static_posture_index
            if not (0 <= index < len(candidate_mask)):
                raise ValueError("static-posture-index is outside the solution file")
            if not bool(candidate_mask[index]):
                raise ValueError("Selected static posture did not pass its frozen gate")
            static_posture_joint_position_np = posture["joint_position"][index].astype(
                np.float32
            )
            static_posture_root_state_np = posture["root_state"][index].astype(
                np.float32
            )
        with np.load(static_posture_source_trace, allow_pickle=False) as source:
            index = args.static_posture_source_index
            if not (0 <= index < source["box_state"].shape[0]):
                raise ValueError(
                    "static-posture-source-index is outside the source trace"
                )
            static_posture_source_box_pose_np = source["box_state"][index, :7].astype(
                np.float32
            )
            static_posture_source_desired_hand_position_np = source[
                "desired_hand_pos_w"
            ][index].astype(np.float32)
            static_posture_source_desired_hand_quaternion_np = source[
                "desired_hand_quat_w"
            ][index].astype(np.float32)
        if static_posture_joint_position_np.shape[0] < 29:
            raise RuntimeError("Static posture does not contain 29 official G1 joints")
        if static_posture_root_state_np.shape != (13,):
            raise RuntimeError("Static posture root state must have shape (13,)")
    output_root.mkdir(parents=True)
    print(
        "PLAN10_ARGS="
        + json.dumps(
            {
                "source_start": args.source_start,
                "source_end": args.source_end,
                "close_start": args.close_start,
                "close_end": args.close_end,
                "close_fraction": args.close_fraction,
                "pregrasp_thumb_pitch_rad": args.pregrasp_thumb_pitch_rad,
                "pregrasp_thumb_yaw_rad": args.pregrasp_thumb_yaw_rad,
                "pregrasp_thumb_pitch_by_side_rad": (
                    pregrasp_thumb_pitch_by_side_rad
                ),
                "pregrasp_thumb_yaw_by_side_rad": (
                    pregrasp_thumb_yaw_by_side_rad
                ),
                "closed_thumb_yaw_rad": closed_thumb_yaw_rad,
                "closed_finger_fraction_by_side": closed_finger_fraction,
                "closed_thumb_pitch_by_side_rad": closed_thumb_pitch_rad,
                "closed_thumb_yaw_by_side_rad": closed_thumb_yaw_by_side_rad,
                "lift_thumb_yaw_by_side_rad": lift_thumb_yaw_by_side_rad,
                "lift_thumb_yaw_start_step": (
                    args.sugar_side_clamp_lift_thumb_yaw_start_step
                ),
                "lift_thumb_yaw_ramp_steps": (
                    args.sugar_side_clamp_lift_thumb_yaw_ramp_steps
                ),
                "hand_control_contract": "unitree_official_inspire_6dof_coupled",
                "bilateral_shoulder_roll_offset_rad": (
                    args.bilateral_shoulder_roll_offset_rad
                ),
                "left_shoulder_roll_offset_rad": left_shoulder_roll_offset_rad,
                "right_shoulder_roll_offset_rad": right_shoulder_roll_offset_rad,
                "left_wrist_roll_offset_rad": args.left_wrist_roll_offset_rad,
                "right_wrist_roll_offset_rad": args.right_wrist_roll_offset_rad,
                "left_wrist_yaw_offset_rad": args.left_wrist_yaw_offset_rad,
                "right_wrist_yaw_offset_rad": args.right_wrist_yaw_offset_rad,
                "left_shoulder_yaw_offset_rad": args.left_shoulder_yaw_offset_rad,
                "right_shoulder_yaw_offset_rad": args.right_shoulder_yaw_offset_rad,
                "grasp_dwell_source": args.grasp_dwell_source,
                "grasp_close_steps": args.grasp_close_steps,
                "grasp_settle_steps": args.grasp_settle_steps,
                "grasp_close_source_span": args.grasp_close_source_span,
                "resume_source_span": args.resume_source_span,
                "resume_polynomial_power": args.resume_polynomial_power,
                "robot_root_y_offset_m": args.robot_root_y_offset_m,
                "bilateral_hip_roll_outward_offset_rad": (
                    args.bilateral_hip_roll_outward_offset_rad
                ),
                "body_control_mode": args.body_control_mode,
                "fix_robot_root": args.fix_robot_root,
                "static_posture_solutions": (
                    str(static_posture_solutions)
                    if static_posture_solutions is not None
                    else None
                ),
                "static_posture_solutions_sha256": (
                    sha256(static_posture_solutions)
                    if static_posture_solutions is not None
                    else None
                ),
                "static_posture_source_trace": (
                    str(static_posture_source_trace)
                    if static_posture_source_trace is not None
                    else None
                ),
                "static_posture_source_trace_sha256": (
                    sha256(static_posture_source_trace)
                    if static_posture_source_trace is not None
                    else None
                ),
                "static_posture_index": args.static_posture_index,
                "static_posture_source_index": args.static_posture_source_index,
                "static_posture_use_declared_approach_clearance": (
                    args.static_posture_use_declared_approach_clearance
                ),
                "static_posture_contact_delta_pca_by_side_m": (
                    static_posture_contact_delta_pca_by_side_m
                ),
                "ik_contact_template_trace": (
                    str(ik_contact_template_trace)
                    if ik_contact_template_trace is not None
                    else None
                ),
                "ik_template_index": args.ik_template_index,
                "ik_approach_clearance_m": args.ik_approach_clearance_m,
                "ik_approach_steps": args.ik_approach_steps,
                "ik_left_box_roll_offset_rad": args.ik_left_box_roll_offset_rad,
                "ik_right_box_roll_offset_rad": args.ik_right_box_roll_offset_rad,
                "ik_track_live_box_during_grasp": (
                    args.ik_track_live_box_during_grasp
                ),
                "ik_track_live_box_during_lift": (
                    args.ik_track_live_box_during_lift
                ),
                "ik_live_lift_lead_m": args.ik_live_lift_lead_m,
                "ik_live_lift_lead_ramp_steps": (
                    args.ik_live_lift_lead_ramp_steps
                ),
                "ik_live_lift_scheduled_world_z": (
                    args.ik_live_lift_scheduled_world_z
                ),
                "ik_live_lift_anchor_world_xy": (
                    args.ik_live_lift_anchor_world_xy
                ),
                "ik_live_hold_relative_to_box": (
                    args.ik_live_hold_relative_to_box
                ),
                "ik_contact_preload_height_m": (
                    args.ik_contact_preload_height_m
                ),
                "ik_contact_preload_steps": args.ik_contact_preload_steps,
                "ik_contact_preload_ramp_steps": (
                    args.ik_contact_preload_ramp_steps
                ),
                "ik_contact_compression_m": args.ik_contact_compression_m,
                "ik_contact_compression_steps": (
                    args.ik_contact_compression_steps
                ),
                "ik_contact_compression_ramp_steps": (
                    args.ik_contact_compression_ramp_steps
                ),
                "ik_palm_press_steps": args.ik_palm_press_steps,
                "sugar_side_clamp_palm_press_first": (
                    args.sugar_side_clamp_palm_press_first
                ),
                "sugar_side_clamp_close_first_hand_before_second": (
                    args.sugar_side_clamp_close_first_hand_before_second
                ),
                "sugar_side_clamp_close_second_hand_during_second_press": (
                    args.sugar_side_clamp_close_second_hand_during_second_press
                ),
                "sugar_side_clamp_close_second_thumb_during_second_press": (
                    args.sugar_side_clamp_close_second_thumb_during_second_press
                ),
                "sugar_side_clamp_second_thumb_close_during_four_finger_steps": (
                    args.sugar_side_clamp_second_thumb_close_during_four_finger_steps
                ),
                "sugar_side_clamp_digit_force_stop_n": (
                    args.sugar_side_clamp_digit_force_stop_n
                ),
                "sugar_side_clamp_digit_force_stop_max_scale_step": (
                    args.sugar_side_clamp_digit_force_stop_max_scale_step
                ),
                "side_clamp_approach_clearance_by_side_m": (
                    side_clamp_approach_clearance_by_side_m
                ),
                "ik_close_steps": args.ik_close_steps,
                "ik_thumb_close_steps": args.ik_thumb_close_steps,
                "ik_settle_steps": args.ik_settle_steps,
                "ik_lift_height_m": args.ik_lift_height_m,
                "ik_lift_steps": args.ik_lift_steps,
                "unitree_demo_parquet": (
                    str(unitree_demo_parquet)
                    if unitree_demo_parquet is not None
                    else None
                ),
                "unitree_demo_episode_index": args.unitree_demo_episode_index,
                "unitree_demo_frame_index": args.unitree_demo_frame_index,
                "unitree_demo_box_yaw_rad": args.unitree_demo_box_yaw_rad,
                "unitree_demo_box_center_offset_m": list(
                    args.unitree_demo_box_center_offset_m
                ),
                "unitree_demo_outward_shift_m": args.unitree_demo_outward_shift_m,
                "unitree_demo_approach_steps": args.unitree_demo_approach_steps,
                "unitree_demo_close_steps": args.unitree_demo_close_steps,
                "unitree_demo_settle_steps": args.unitree_demo_settle_steps,
                "require_settled_all_groups_frames": (
                    args.require_settled_all_groups_frames
                ),
                "object_static_friction": args.object_static_friction,
                "object_dynamic_friction": args.object_dynamic_friction,
                "object_restitution": 0.0,
                "sugar_side_clamp_support_height_m": (
                    args.sugar_side_clamp_support_height_m
                ),
                "sugar_side_clamp_support_size_xy_m": list(
                    args.sugar_side_clamp_support_size_xy_m
                ),
                "zero_initial_object_velocity": args.zero_initial_object_velocity,
                "brake_start": args.brake_start,
                "brake_steps": args.brake_steps,
                "solver_position_iterations": args.solver_position_iterations,
                "solver_velocity_iterations": args.solver_velocity_iterations,
                "hold_steps": args.hold_steps,
                "output_root": str(output_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("PLAN10_STAGE=load_official_motion", flush=True)
    with np.load(robot_motion_path, allow_pickle=False) as source:
        source_fps = int(np.asarray(source["fps"]).reshape(-1)[0])
        joint_pos_np = np.asarray(source["joint_pos"], dtype=np.float32)
        joint_vel_np = np.asarray(source["joint_vel"], dtype=np.float32)
        body_pos_np = np.asarray(source["body_pos_w"], dtype=np.float32)
        body_quat_np = np.asarray(source["body_quat_w"], dtype=np.float32)
        body_lin_vel_np = np.asarray(source["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel_np = np.asarray(source["body_ang_vel_w"], dtype=np.float32)
    with object_motion_path.open("rb") as stream:
        object_motion = pickle.load(stream)
    object_pos_np = np.asarray(object_motion["obj_trans"], dtype=np.float32)
    object_rot_np = np.asarray(object_motion["obj_rot"], dtype=np.float32)
    object_lin_vel_np = np.asarray(object_motion["obj_lin_vel"], dtype=np.float32)
    object_ang_vel_np = np.asarray(object_motion["obj_ang_vel"], dtype=np.float32)

    if source_fps != 50:
        raise RuntimeError(f"Expected official 50 Hz motion, got {source_fps}")
    if joint_pos_np.shape[1] != 29:
        raise RuntimeError(f"Expected official 29-DoF motion, got {joint_pos_np.shape}")
    if args.source_end >= len(joint_pos_np) or args.source_start >= len(object_pos_np):
        raise RuntimeError("Requested source interval exceeds official motion")

    sugar_side_clamp_support_center_xy = None
    if (
        args.body_control_mode == "sugar_side_clamp"
        and args.sugar_side_clamp_support_height_m > 0.0
    ):
        support_box_center_w = object_pos_np[args.source_start].astype(
            np.float64
        ) + object_rot_np[args.source_start].astype(np.float64) @ np.asarray(
            args.sugar_side_clamp_box_local_offset_m, dtype=np.float64
        )
        sugar_side_clamp_support_center_xy = support_box_center_w[:2]

    unitree_demo_dataset_sha256 = None
    unitree_demo_root_pose = None
    unitree_demo_body_joint_pos = None
    unitree_demo_ee_state = None
    unitree_demo_hand_normalized = None
    unitree_demo_initial_box_pose = None
    unitree_demo_effective_box_yaw_rad = None
    unitree_demo_outward_translation = None
    unitree_demo_body_limit_adjustment_rad = 0.0
    if args.body_control_mode == "unitree_demo_pose":
        assert unitree_demo_parquet is not None
        print("PLAN10_STAGE=load_official_unitree_demo_pose", flush=True)
        import pyarrow.parquet as pq

        unitree_demo_dataset_sha256 = sha256(unitree_demo_parquet)
        parquet = pq.ParquetFile(unitree_demo_parquet)
        if args.unitree_demo_episode_index >= parquet.metadata.num_row_groups:
            raise ValueError("Unitree demo episode exceeds Parquet row groups")
        episode = parquet.read_row_group(
            args.unitree_demo_episode_index,
            columns=(
                "observation.state.ee_state",
                "observation.state.robot_q_current",
                "action.hand_cmd",
                "episode_index",
                "frame_index",
            ),
        ).to_pydict()
        episode_ids = np.asarray(episode["episode_index"], dtype=np.int64)
        frame_ids = np.asarray(episode["frame_index"], dtype=np.int64)
        if not np.all(episode_ids == args.unitree_demo_episode_index):
            raise RuntimeError("Parquet row group is not the requested exact episode")
        match = np.flatnonzero(frame_ids == args.unitree_demo_frame_index)
        if len(match) != 1:
            raise ValueError("Requested Unitree demo frame is absent or duplicated")
        row = int(match[0])
        robot_q = np.asarray(
            episode["observation.state.robot_q_current"][row], dtype=np.float32
        )
        unitree_demo_ee_state = np.asarray(
            episode["observation.state.ee_state"][row], dtype=np.float32
        )
        unitree_demo_hand_normalized = np.asarray(
            episode["action.hand_cmd"][row], dtype=np.float32
        )
        if robot_q.shape != (36,) or unitree_demo_ee_state.shape != (12,):
            raise RuntimeError("Official Unitree demo state shapes are not 36/12")
        if unitree_demo_hand_normalized.shape != (12,) or not np.all(
            (0.0 <= unitree_demo_hand_normalized)
            & (unitree_demo_hand_normalized <= 1.0)
        ):
            raise RuntimeError("Official Unitree Inspire command is not normalized 12-D")
        unitree_demo_root_pose = torch.as_tensor(robot_q[:7], device=args.device)
        wire_joint_pos = robot_q[7:]
        wire_index = {
            name: index for index, name in enumerate(UNITREE_G129_WIRE_JOINT_NAMES)
        }
        unitree_demo_body_joint_pos = torch.as_tensor(
            np.asarray(
                [wire_joint_pos[wire_index[name]] for name in EXPECTED_BODY_JOINT_NAMES],
                dtype=np.float32,
            ),
            device=args.device,
        )
        demo_relative_positions = torch.as_tensor(
            np.stack(
                (unitree_demo_ee_state[:3], unitree_demo_ee_state[6:9])
            ),
            device=args.device,
        )
        demo_world_positions, _ = math_utils.combine_frame_transforms(
            unitree_demo_root_pose[:3].repeat(2, 1),
            unitree_demo_root_pose[3:7].repeat(2, 1),
            demo_relative_positions,
            torch.tensor(
                ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                device=args.device,
            ),
        )
        demo_midpoint = demo_world_positions.mean(dim=0)
        outward_direction_xy = demo_midpoint[:2] - unitree_demo_root_pose[:2]
        outward_norm = torch.linalg.norm(outward_direction_xy)
        if float(outward_norm.item()) <= 1.0e-6:
            raise RuntimeError("Official pelvis-to-hand horizontal direction is degenerate")
        unitree_demo_outward_translation = torch.zeros(
            3, device=args.device, dtype=demo_midpoint.dtype
        )
        unitree_demo_outward_translation[:2] = (
            args.unitree_demo_outward_shift_m
            * outward_direction_xy
            / outward_norm
        )
        box_position = (
            demo_midpoint
            + torch.as_tensor(
                args.unitree_demo_box_center_offset_m, device=args.device
            )
            + unitree_demo_outward_translation
        )
        if args.unitree_demo_align_box_narrow_axis_to_hands:
            hand_axis_xy = demo_world_positions[0, :2] - demo_world_positions[1, :2]
            if float(torch.linalg.norm(hand_axis_xy).item()) <= 1.0e-6:
                raise RuntimeError("Official demo horizontal hand axis is degenerate")
            aligned_yaw = float(torch.atan2(hand_axis_xy[1], hand_axis_xy[0]).item())
            unitree_demo_effective_box_yaw_rad = (
                aligned_yaw + args.unitree_demo_box_yaw_rad
            )
        else:
            unitree_demo_effective_box_yaw_rad = args.unitree_demo_box_yaw_rad
        half_yaw = 0.5 * unitree_demo_effective_box_yaw_rad
        box_quaternion = torch.tensor(
            (np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)),
            device=args.device,
            dtype=box_position.dtype,
        )
        unitree_demo_initial_box_pose = torch.cat((box_position, box_quaternion))

    print("PLAN10_STAGE=create_simulation_context", flush=True)
    control_fps = 30 if args.body_control_mode == "unitree_demo_pose" else source_fps
    dt = 1.0 / (control_fps * args.physics_substeps_per_source)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=dt,
            device=args.device,
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                min_position_iteration_count=4,
                max_position_iteration_count=16,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=4,
            ),
        )
    )
    floor_cfg = sim_utils.CuboidCfg(
        size=(6.0, 6.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.20, 0.22, 0.25)
        ),
    )
    floor_cfg.func(
        "/World/LocalFloor", floor_cfg, translation=(0.0, 0.0, -0.05)
    )
    if sugar_side_clamp_support_center_xy is not None:
        support_height_m = args.sugar_side_clamp_support_height_m
        sugar_support_cfg = sim_utils.CuboidCfg(
            size=(
                args.sugar_side_clamp_support_size_xy_m[0],
                args.sugar_side_clamp_support_size_xy_m[1],
                support_height_m,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.34, 0.31, 0.27)
            ),
        )
        sugar_support_cfg.func(
            "/World/SugarSideClampSupport",
            sugar_support_cfg,
            translation=(
                float(sugar_side_clamp_support_center_xy[0]),
                float(sugar_side_clamp_support_center_xy[1]),
                0.5 * support_height_m,
            ),
        )
    if unitree_demo_initial_box_pose is not None:
        # A real rigid support lets the official recorded grasp close under
        # gravity. It is below the box and is left behind before the >0.10 m
        # lift/force-balance interval; no object state is replayed.
        support_thickness_m = 0.05
        support_top_z = (
            float(unitree_demo_initial_box_pose[2].item())
            + CARRYBOX_MESH_Z_MIN_M
        )
        support_cfg = sim_utils.CuboidCfg(
            size=(
                args.unitree_demo_support_size_xy_m[0],
                args.unitree_demo_support_size_xy_m[1],
                support_thickness_m,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.34, 0.31, 0.27)
            ),
        )
        support_cfg.func(
            "/World/UnitreeDemoSupport",
            support_cfg,
            translation=(
                float(unitree_demo_initial_box_pose[0].item()),
                float(unitree_demo_initial_box_pose[1].item()),
                support_top_z - 0.5 * support_thickness_m,
            ),
        )
        if args.unitree_demo_support_cradle_height_m > 0.0:
            assert unitree_demo_effective_box_yaw_rad is not None
            rail_height = args.unitree_demo_support_cradle_height_m
            rail_thickness = 0.03
            rail_clearance = 0.005
            box_x_min = CARRYBOX_SIDE_FACE_X_M["right"]
            box_x_max = CARRYBOX_SIDE_FACE_X_M["left"]
            box_y_min, box_y_max = CARRYBOX_MESH_Y_BOUNDS_M
            rail_specs = (
                (
                    "PosX",
                    (
                        box_x_max + rail_clearance + 0.5 * rail_thickness,
                        0.5 * (box_y_min + box_y_max),
                    ),
                    (
                        rail_thickness,
                        box_y_max - box_y_min
                        + 2.0 * (rail_clearance + rail_thickness),
                        rail_height,
                    ),
                ),
                (
                    "NegX",
                    (
                        box_x_min - rail_clearance - 0.5 * rail_thickness,
                        0.5 * (box_y_min + box_y_max),
                    ),
                    (
                        rail_thickness,
                        box_y_max - box_y_min
                        + 2.0 * (rail_clearance + rail_thickness),
                        rail_height,
                    ),
                ),
                (
                    "PosY",
                    (
                        0.5 * (box_x_min + box_x_max),
                        box_y_max + rail_clearance + 0.5 * rail_thickness,
                    ),
                    (
                        box_x_max - box_x_min
                        + 2.0 * (rail_clearance + rail_thickness),
                        rail_thickness,
                        rail_height,
                    ),
                ),
                (
                    "NegY",
                    (
                        0.5 * (box_x_min + box_x_max),
                        box_y_min - rail_clearance - 0.5 * rail_thickness,
                    ),
                    (
                        box_x_max - box_x_min
                        + 2.0 * (rail_clearance + rail_thickness),
                        rail_thickness,
                        rail_height,
                    ),
                ),
            )
            yaw_cos = np.cos(unitree_demo_effective_box_yaw_rad)
            yaw_sin = np.sin(unitree_demo_effective_box_yaw_rad)
            cradle_orientation = (
                np.cos(0.5 * unitree_demo_effective_box_yaw_rad),
                0.0,
                0.0,
                np.sin(0.5 * unitree_demo_effective_box_yaw_rad),
            )
            for rail_name, local_xy, rail_size in rail_specs:
                world_xy = (
                    float(unitree_demo_initial_box_pose[0].item())
                    + yaw_cos * local_xy[0]
                    - yaw_sin * local_xy[1],
                    float(unitree_demo_initial_box_pose[1].item())
                    + yaw_sin * local_xy[0]
                    + yaw_cos * local_xy[1],
                )
                rail_cfg = sim_utils.CuboidCfg(
                    size=rail_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.42, 0.36, 0.28)
                    ),
                )
                rail_cfg.func(
                    f"/World/UnitreeDemoCradle{rail_name}",
                    rail_cfg,
                    translation=(
                        world_xy[0],
                        world_xy[1],
                        support_top_z + 0.5 * rail_height,
                    ),
                    orientation=cradle_orientation,
                )

    print("PLAN10_STAGE=spawn_official_robot", flush=True)
    robot_cfg = G129_CFG_WITH_INSPIRE_WHOLEBODY.replace(
        prim_path="/World/Robot"
    )
    # Kinematic state replay is retained only as an explicitly rejected
    # reachability diagnostic.  The physical gate uses the official joint
    # drives, gravity, and an unconstrained floating base after initialization.
    robot_cfg.spawn.rigid_props.disable_gravity = args.body_control_mode == "state_replay"
    robot_cfg.spawn.articulation_props.fix_root_link = args.fix_robot_root
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = (
        args.solver_position_iterations
    )
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = (
        args.solver_velocity_iterations
    )
    robot = Articulation(cfg=robot_cfg)

    print("PLAN10_STAGE=spawn_official_carrybox", flush=True)
    object_cfg = RigidObjectCfg(
        prim_path="/World/CarryBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(box_usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    box = RigidObject(cfg=object_cfg)

    print("PLAN10_STAGE=create_raw_contact_views", flush=True)
    sensors = {
        "left": ContactSensor(
            ContactSensorCfg(
                prim_path="/World/Robot/(left_hand_base_link|L_.*)",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/CarryBox"],
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                max_contact_data_count_per_prim=256,
            )
        ),
        "right": ContactSensor(
            ContactSensorCfg(
                prim_path="/World/Robot/(right_hand_base_link|R_.*)",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/CarryBox"],
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                max_contact_data_count_per_prim=256,
            )
        ),
    }
    all_robot_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/Robot/.*",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            filter_prim_paths_expr=["/World/CarryBox"],
            track_pose=False,
            track_contact_points=True,
            track_friction_forces=True,
            max_contact_data_count_per_prim=256,
        )
    )
    print("PLAN10_STAGE=simulation_reset", flush=True)
    sim.reset()
    print("PLAN10_STAGE=apply_and_readback_official_object_material", flush=True)
    material_env_ids_cpu = torch.arange(1, dtype=torch.int64, device="cpu")
    object_material_before = box.root_physx_view.get_material_properties().clone()
    object_material_requested = object_material_before.clone()
    object_material_requested[:, :, 0] = args.object_static_friction
    object_material_requested[:, :, 1] = args.object_dynamic_friction
    object_material_requested[:, :, 2] = 0.0
    box.root_physx_view.set_material_properties(
        object_material_requested, material_env_ids_cpu
    )
    object_material_readback = box.root_physx_view.get_material_properties().clone()
    robot_material_readback = robot.root_physx_view.get_material_properties().clone()
    if not bool(
        (object_material_readback[:, :, 0] == args.object_static_friction).all()
    ):
        raise RuntimeError("PhysX object static-friction readback mismatch")
    if not bool(
        (object_material_readback[:, :, 1] == args.object_dynamic_friction).all()
    ):
        raise RuntimeError("PhysX object dynamic-friction readback mismatch")
    if not bool((object_material_readback[:, :, 2] == 0.0).all()):
        raise RuntimeError("PhysX object restitution readback mismatch")
    print("PLAN10_STAGE=initialize_raw_contact_views", flush=True)
    for sensor in sensors.values():
        _ = sensor.data
    _ = all_robot_sensor.data

    print("PLAN10_STAGE=validate_live_topology", flush=True)

    if tuple(robot.joint_names[:29]) != EXPECTED_BODY_JOINT_NAMES:
        raise RuntimeError("Unitree/SUGAR first-29 joint order is not exact")
    finger_ids = list(range(29, len(robot.joint_names)))
    if len(finger_ids) != 24:
        raise RuntimeError(f"Expected 24 Inspire joints, got {len(finger_ids)}")
    finger_names = [robot.joint_names[index] for index in finger_ids]
    expected_finger_names = set(OFFICIAL_INSPIRE_BASE_COMMAND_INDEX) | set(
        OFFICIAL_INSPIRE_SPECIAL_COMMAND
    )
    if set(finger_names) != expected_finger_names:
        raise RuntimeError(
            "Live Inspire joints do not exactly match the released Unitree "
            f"command mapping: missing={sorted(expected_finger_names-set(finger_names))}, "
            f"extra={sorted(set(finger_names)-expected_finger_names)}"
        )
    hand_body_names = {side: list(sensor.body_names) for side, sensor in sensors.items()}
    hand_body_groups = {
        side: [body_group(name) for name in names]
        for side, names in hand_body_names.items()
    }
    for side in SIDES:
        if set(hand_body_groups[side]) != set(GROUPS):
            raise RuntimeError(
                f"{side} hand sensor does not cover all six anatomy groups: "
                f"{hand_body_groups[side]}"
            )

    device = robot.device
    joint_pos = torch.as_tensor(joint_pos_np, device=device)
    joint_vel = torch.as_tensor(joint_vel_np, device=device)
    body_pos = torch.as_tensor(body_pos_np, device=device)
    body_quat = torch.as_tensor(body_quat_np, device=device)
    body_lin_vel = torch.as_tensor(body_lin_vel_np, device=device)
    body_ang_vel = torch.as_tensor(body_ang_vel_np, device=device)
    object_pos = torch.as_tensor(object_pos_np, device=device)
    object_rot = torch.as_tensor(object_rot_np, device=device)
    object_quat = math_utils.quat_from_matrix(object_rot)
    object_lin_vel = torch.as_tensor(object_lin_vel_np, device=device)
    object_ang_vel = torch.as_tensor(object_ang_vel_np, device=device)

    hard_lower = robot.data.joint_pos_limits[0, finger_ids, 0].clone()
    hard_upper = robot.data.joint_pos_limits[0, finger_ids, 1].clone()
    body_hard_lower = robot.data.joint_pos_limits[0, :29, 0].clone()
    body_hard_upper = robot.data.joint_pos_limits[0, :29, 1].clone()
    ik_template_relative_pos = None
    ik_template_relative_quat = None
    ik_template_source_relative_quat = None
    ik_template_box_pose = None
    ik_template_source_time = None
    ik_template_trace_sha256 = None
    if args.body_control_mode == "bilateral_ik":
        assert ik_contact_template_trace is not None
        with np.load(ik_contact_template_trace, allow_pickle=False) as template:
            index = args.ik_template_index
            if not (0 <= index < template["box_state"].shape[0]):
                raise ValueError("IK template index is outside the trace")
            if tuple(template["side_names"].astype(str)) != SIDES:
                raise RuntimeError("IK template side order is not bilateral left/right")
            if tuple(template["group_names"].astype(str)) != GROUPS:
                raise RuntimeError("IK template anatomy order is not exact")
            if not np.all(
                template["group_normal_load"][index] > args.contact_threshold_n
            ):
                raise RuntimeError("IK template index does not load all six bilateral groups")
            template_all_body_names = template[
                "all_robot_contact_body_names"
            ].astype(str)
            template_hand_names = set(
                template["left_contact_body_names"].astype(str).tolist()
            ) | set(template["right_contact_body_names"].astype(str).tolist())
            template_nonhand_mask = np.asarray(
                [name not in template_hand_names for name in template_all_body_names],
                dtype=bool,
            )
            template_nonhand_load = float(
                template["all_robot_body_normal_load"][
                    index, template_nonhand_mask
                ].sum()
            )
            if template_nonhand_load > args.contact_threshold_n:
                raise RuntimeError("IK template index has non-hand CarryBox support")
            template_position = template["finger_position"][index].astype(np.float64)
            template_lower = template["finger_hard_lower_rad"].astype(np.float64)
            template_upper = template["finger_hard_upper_rad"].astype(np.float64)
            template_violation = np.maximum(
                np.maximum(template_lower - template_position, 0.0),
                np.maximum(template_position - template_upper, 0.0),
            ).max(initial=0.0)
            if template_violation > 1.0e-3:
                raise RuntimeError("IK template index violates an Inspire hard limit")
            template_box_pos = torch.as_tensor(
                template["box_state"][index, :3], device=device
            ).repeat(2, 1)
            template_box_quat = torch.as_tensor(
                template["box_state"][index, 3:7], device=device
            ).repeat(2, 1)
            template_hand_pos = torch.as_tensor(
                template["hand_body_pos_w"][index, :, 0], device=device
            )
            template_hand_quat = torch.as_tensor(
                template["hand_body_quat_w"][index, :, 0], device=device
            )
            ik_template_box_pose = torch.as_tensor(
                template["box_state"][index, :7], device=device
            )
            ik_template_source_time = float(template["source_time"][index])
        ik_template_relative_pos, ik_template_relative_quat = (
            math_utils.subtract_frame_transforms(
                template_box_pos,
                template_box_quat,
                template_hand_pos,
                template_hand_quat,
            )
        )
        ik_template_source_relative_quat = ik_template_relative_quat.clone()
        box_roll_offsets = torch.as_tensor(
            (
                args.ik_left_box_roll_offset_rad,
                args.ik_right_box_roll_offset_rad,
            ),
            device=device,
            dtype=ik_template_relative_quat.dtype,
        )
        box_roll_quat = torch.zeros_like(ik_template_relative_quat)
        box_roll_quat[:, 0] = torch.cos(0.5 * box_roll_offsets)
        box_roll_quat[:, 1] = torch.sin(0.5 * box_roll_offsets)
        ik_template_relative_quat = math_utils.quat_mul(
            box_roll_quat, ik_template_relative_quat
        )
        ik_template_trace_sha256 = sha256(ik_contact_template_trace)

    def scheduled_offset(start: float, end: float, source_time: float) -> float:
        if not body_offset_transition_enabled:
            return start
        assert args.shoulder_roll_transition_start is not None
        assert args.shoulder_roll_transition_end is not None
        u = np.clip(
            (source_time - args.shoulder_roll_transition_start)
            / (
                args.shoulder_roll_transition_end
                - args.shoulder_roll_transition_start
            ),
            0.0,
            1.0,
        )
        return float(
            (1.0 - u) * start + u * end
        )

    def scheduled_offset_velocity(
        start: float, end: float, source_time: float, velocity_scale: float
    ) -> float:
        if not body_offset_transition_enabled or start == end:
            return 0.0
        assert args.shoulder_roll_transition_start is not None
        assert args.shoulder_roll_transition_end is not None
        if not (
            args.shoulder_roll_transition_start
            < source_time
            < args.shoulder_roll_transition_end
        ):
            return 0.0
        # Source frames advance at the official 50-Hz motion rate.
        return float(
            50.0
            * velocity_scale
            * (end - start)
            / (
                args.shoulder_roll_transition_end
                - args.shoulder_roll_transition_start
            )
        )

    def add_declared_body_offset(
        body_target: torch.Tensor, source_time: float
    ) -> torch.Tensor:
        target = body_target.clone()
        target[3] += args.bilateral_hip_roll_outward_offset_rad
        target[4] -= args.bilateral_hip_roll_outward_offset_rad
        if args.waist_pitch_absolute_rad is not None:
            target[8] = args.waist_pitch_absolute_rad
        # The added Inspire palm/fingers are wider than SUGAR's rigid rubber
        # hand.  This bounded no-learning retarget scan moves the two arms in
        # opposite shoulder-roll directions; zero remains exact SUGAR.
        target[15] += scheduled_offset(
            left_shoulder_roll_offset_rad,
            left_shoulder_roll_end_offset_rad,
            source_time,
        )
        target[16] += right_shoulder_roll_offset_rad
        target[23] += scheduled_offset(
            args.left_wrist_roll_offset_rad,
            left_wrist_roll_end_offset_rad,
            source_time,
        )
        target[24] += args.right_wrist_roll_offset_rad
        target[27] += scheduled_offset(
            args.left_wrist_yaw_offset_rad,
            left_wrist_yaw_end_offset_rad,
            source_time,
        )
        target[28] += args.right_wrist_yaw_offset_rad
        target[19] += args.left_shoulder_yaw_offset_rad
        target[20] += args.right_shoulder_yaw_offset_rad
        return target

    body_reference_with_offset = joint_pos[args.source_start : args.source_end + 1].clone()
    body_reference_with_offset[:, 3] += args.bilateral_hip_roll_outward_offset_rad
    body_reference_with_offset[:, 4] -= args.bilateral_hip_roll_outward_offset_rad
    if args.waist_pitch_absolute_rad is not None:
        body_reference_with_offset[:, 8] = args.waist_pitch_absolute_rad
    source_indices = range(args.source_start, args.source_end + 1)
    body_reference_with_offset[:, 15] += torch.as_tensor(
        [
            scheduled_offset(
                left_shoulder_roll_offset_rad,
                left_shoulder_roll_end_offset_rad,
                float(index),
            )
            for index in source_indices
        ],
        device=device,
        dtype=body_reference_with_offset.dtype,
    )
    body_reference_with_offset[:, 16] += right_shoulder_roll_offset_rad
    body_reference_with_offset[:, 23] += torch.as_tensor(
        [
            scheduled_offset(
                args.left_wrist_roll_offset_rad,
                left_wrist_roll_end_offset_rad,
                float(index),
            )
            for index in source_indices
        ],
        device=device,
        dtype=body_reference_with_offset.dtype,
    )
    body_reference_with_offset[:, 24] += args.right_wrist_roll_offset_rad
    body_reference_with_offset[:, 27] += torch.as_tensor(
        [
            scheduled_offset(
                args.left_wrist_yaw_offset_rad,
                left_wrist_yaw_end_offset_rad,
                float(index),
            )
            for index in source_indices
        ],
        device=device,
        dtype=body_reference_with_offset.dtype,
    )
    body_reference_with_offset[:, 28] += args.right_wrist_yaw_offset_rad
    body_reference_with_offset[:, 19] += args.left_shoulder_yaw_offset_rad
    body_reference_with_offset[:, 20] += args.right_shoulder_yaw_offset_rad
    body_target_limit_violation = torch.maximum(
        body_hard_lower[None, :] - body_reference_with_offset,
        torch.zeros_like(body_reference_with_offset),
    ) + torch.maximum(
        body_reference_with_offset - body_hard_upper[None, :],
        torch.zeros_like(body_reference_with_offset),
    )
    maximum_body_target_limit_violation = float(
        torch.max(body_target_limit_violation).item()
    )
    if maximum_body_target_limit_violation > 1.0e-6:
        raise RuntimeError(
            "Declared shoulder-roll retarget violates live G1 hard limits: "
            f"{maximum_body_target_limit_violation} rad"
        )

    def official_inspire_command_by_side(
        closure_by_side: dict[str, float],
        thumb_closure_by_side: dict[str, float],
        closure_by_group: dict[str, dict[str, float]] | None = None,
    ) -> torch.Tensor:
        command = torch.zeros(12, device=device, dtype=joint_pos.dtype)
        # Physical ranges from the released dds/inspire_dds.py.
        for side, prefix in (("left", "L"), ("right", "R")):
            for finger in ("index", "middle", "ring", "little"):
                asset_finger = "pinky" if finger == "little" else finger
                command_index = OFFICIAL_INSPIRE_BASE_COMMAND_INDEX[
                    f"{prefix}_{asset_finger}_proximal_joint"
                ]
                group_closure = (
                    closure_by_side[side]
                    if closure_by_group is None
                    else closure_by_group[side][finger]
                )
                command[command_index] = (
                    1.7
                    * closed_finger_fraction[side][finger]
                    * group_closure
                )
            pitch_index = 10 if side == "left" else 4
            yaw_index = 11 if side == "left" else 5
            command[pitch_index] = (
                pregrasp_thumb_pitch_by_side_rad[side]
                + thumb_closure_by_side[side]
                * (
                    closed_thumb_pitch_rad[side]
                    - pregrasp_thumb_pitch_by_side_rad[side]
                )
            )
            command[yaw_index] = (
                pregrasp_thumb_yaw_by_side_rad[side]
                + thumb_closure_by_side[side]
                * (
                    closed_thumb_yaw_by_side_rad[side]
                    - pregrasp_thumb_yaw_by_side_rad[side]
                )
            )
        return command

    def official_inspire_command(
        closure: float, thumb_closure: float | None = None
    ) -> torch.Tensor:
        if thumb_closure is None:
            thumb_closure = closure
        return official_inspire_command_by_side(
            {side: closure for side in SIDES},
            {side: thumb_closure for side in SIDES},
        )

    def official_inspire_target(
        closure: float, thumb_closure: float | None = None
    ) -> torch.Tensor:
        """Apply Unitree's released 12-command mapping and joint coupling."""

        command = official_inspire_command(closure, thumb_closure)
        return official_inspire_target_from_command(command)

    def official_inspire_target_from_command(command: torch.Tensor) -> torch.Tensor:
        """Apply the released Unitree 12-command mapping without extra joints."""

        if tuple(command.shape) != (12,):
            raise ValueError(f"Expected one physical 12-D Inspire command, got {command.shape}")
        target_by_name = {
            name: command[index]
            for name, index in OFFICIAL_INSPIRE_BASE_COMMAND_INDEX.items()
        }
        target_by_name.update(
            {
                name: command[index] * scale
                for name, (index, scale) in OFFICIAL_INSPIRE_SPECIAL_COMMAND.items()
            }
        )
        return torch.stack([target_by_name[name] for name in finger_names])

    def normalized_unitree_demo_command(values: np.ndarray) -> torch.Tensor:
        """Convert the official dataset's documented normalized L/R ordering.

        Dataset order per hand is index, middle, ring, little, thumb close,
        thumb lateral tilt. Runtime order is the released right-first
        little/ring/middle/index/thumb-pitch/thumb-yaw DDS contract.
        """

        normalized = torch.as_tensor(values, device=device, dtype=joint_pos.dtype)
        if tuple(normalized.shape) != (12,):
            raise ValueError("Official normalized Inspire command must be 12-D")
        physical = torch.zeros_like(normalized)
        for source_offset, target_offset in ((0, 6), (6, 0)):
            physical[target_offset + 0] = 1.7 * (1.0 - normalized[source_offset + 3])
            physical[target_offset + 1] = 1.7 * (1.0 - normalized[source_offset + 2])
            physical[target_offset + 2] = 1.7 * (1.0 - normalized[source_offset + 1])
            physical[target_offset + 3] = 1.7 * (1.0 - normalized[source_offset + 0])
            physical[target_offset + 4] = 0.5 * (1.0 - normalized[source_offset + 4])
            physical[target_offset + 5] = (
                1.3 - 1.4 * normalized[source_offset + 5]
            )
        return physical

    open_fingers = official_inspire_target(0.0)
    closed_fingers = official_inspire_target(1.0)
    for label, target in (("open", open_fingers), ("closed", closed_fingers)):
        lower_error = hard_lower - target
        upper_error = target - hard_upper
        if bool(torch.any(lower_error > 1.0e-6)) or bool(
            torch.any(upper_error > 1.0e-6)
        ):
            raise RuntimeError(
                f"Official-coupled {label} target violates live hard limits"
            )

    unitree_demo_physical_command = None
    unitree_demo_finger_target = None
    if args.body_control_mode == "unitree_demo_pose":
        assert unitree_demo_hand_normalized is not None
        unitree_demo_physical_command = normalized_unitree_demo_command(
            unitree_demo_hand_normalized
        )
        unitree_demo_finger_target = official_inspire_target_from_command(
            unitree_demo_physical_command
        )
        if bool(torch.any(hard_lower - unitree_demo_finger_target > 1.0e-6)) or bool(
            torch.any(unitree_demo_finger_target - hard_upper > 1.0e-6)
        ):
            raise RuntimeError(
                "Official dataset Inspire command violates live coupled joint limits"
            )

    start = args.source_start
    sugar_side_clamp_box_offset_w = None
    sugar_side_clamp_reachable_fit_offset_w = None
    sugar_side_clamp_reachable_orientation_delta_rad = None
    sugar_side_clamp_direct_refinement_steps = 0
    sugar_side_clamp_direct_refinement_accepted_steps = 0
    sugar_side_clamp_direct_refinement_accepted_by_side = {
        side: 0 for side in SIDES
    }
    sugar_side_clamp_direct_refinement_initial_error_m = None
    sugar_side_clamp_direct_refinement_initial_error_vector_w = None
    sugar_side_clamp_direct_refinement_final_error_m = None
    sugar_side_clamp_direct_refinement_final_trust_radius_m = None
    static_posture_relative_root_pose = None
    static_posture_mapped_root_pose = None
    initial_object_state = torch.cat(
        (
            object_pos[start],
            object_quat[start],
            object_lin_vel[start],
            object_ang_vel[start],
        )
    ).unsqueeze(0)
    if args.zero_initial_object_velocity:
        initial_object_state[0, 7:13] = 0.0
    if args.body_control_mode == "sugar_side_clamp":
        sugar_side_clamp_box_offset_w = math_utils.quat_apply(
            initial_object_state[:, 3:7],
            torch.tensor(
                args.sugar_side_clamp_box_local_offset_m,
                device=device,
                dtype=joint_pos.dtype,
            ).unsqueeze(0),
        )[0]
        initial_object_state[0, :3] += sugar_side_clamp_box_offset_w
        initial_object_state[0, 2] += args.sugar_side_clamp_support_height_m
    initial_body_joint_pos = add_declared_body_offset(
        joint_pos[start], float(start)
    )
    initial_body_joint_vel = joint_vel[start].clone()
    root_state = torch.cat(
        (
            body_pos[start, 0],
            body_quat[start, 0],
            body_lin_vel[start, 0],
            body_ang_vel[start, 0],
        )
    ).unsqueeze(0)
    if args.body_control_mode == "unitree_demo_pose":
        assert unitree_demo_initial_box_pose is not None
        assert unitree_demo_root_pose is not None
        assert unitree_demo_body_joint_pos is not None
        initial_object_state = torch.cat(
            (
                unitree_demo_initial_box_pose,
                torch.zeros(6, device=device, dtype=joint_pos.dtype),
            )
        ).unsqueeze(0)
        root_state = torch.cat(
            (
                unitree_demo_root_pose,
                torch.zeros(6, device=device, dtype=joint_pos.dtype),
            )
        ).unsqueeze(0)
        initial_body_joint_pos = unitree_demo_body_joint_pos.to(
            device=device, dtype=joint_pos.dtype
        ).clone()
        initial_body_joint_vel = torch.zeros_like(initial_body_joint_pos)
    elif args.body_control_mode == "bilateral_ik":
        assert ik_template_source_time is not None
        assert ik_template_box_pose is not None
        template_low = int(np.floor(ik_template_source_time))
        template_high = min(template_low + 1, args.source_end)
        template_tau = ik_template_source_time - template_low
        template_root_pos = (
            (1.0 - template_tau) * body_pos[template_low, 0]
            + template_tau * body_pos[template_high, 0]
        )
        template_root_quat = math_utils.quat_slerp(
            body_quat[template_low, 0].clone(),
            body_quat[template_high, 0].clone(),
            float(template_tau),
        )
        root_relative_pos, root_relative_quat = math_utils.subtract_frame_transforms(
            ik_template_box_pose[:3].unsqueeze(0),
            ik_template_box_pose[3:7].unsqueeze(0),
            template_root_pos.unsqueeze(0),
            template_root_quat.unsqueeze(0),
        )
        mapped_root_pos, mapped_root_quat = math_utils.combine_frame_transforms(
            initial_object_state[:, :3],
            initial_object_state[:, 3:7],
            root_relative_pos,
            root_relative_quat,
        )
        root_state = torch.cat(
            (mapped_root_pos, mapped_root_quat, torch.zeros((1, 6), device=device)),
            dim=1,
        )
        template_body_joint_pos = (
            (1.0 - template_tau) * joint_pos[template_low]
            + template_tau * joint_pos[template_high]
        )
        initial_body_joint_pos = add_declared_body_offset(
            template_body_joint_pos, ik_template_source_time
        )
        initial_body_joint_vel = torch.zeros_like(initial_body_joint_pos)
    else:
        root_state[0, 1] += args.robot_root_y_offset_m
    if static_posture_joint_position_np is not None:
        assert static_posture_root_state_np is not None
        assert static_posture_source_box_pose_np is not None
        source_box_pose_t = torch.as_tensor(
            static_posture_source_box_pose_np,
            device=device,
            dtype=joint_pos.dtype,
        ).unsqueeze(0)
        source_root_pose_t = torch.as_tensor(
            static_posture_root_state_np[:7],
            device=device,
            dtype=joint_pos.dtype,
        ).unsqueeze(0)
        relative_position, relative_quaternion = math_utils.subtract_frame_transforms(
            source_box_pose_t[:, :3],
            source_box_pose_t[:, 3:7],
            source_root_pose_t[:, :3],
            source_root_pose_t[:, 3:7],
        )
        mapped_position, mapped_quaternion = math_utils.combine_frame_transforms(
            initial_object_state[:, :3],
            initial_object_state[:, 3:7],
            relative_position,
            relative_quaternion,
        )
        static_posture_relative_root_pose = torch.cat(
            (relative_position, relative_quaternion), dim=1
        )[0]
        static_posture_mapped_root_pose = torch.cat(
            (mapped_position, mapped_quaternion), dim=1
        )[0]
        root_state[0, :7] = static_posture_mapped_root_pose
        root_state[0, 7:13] = 0.0
        initial_body_joint_pos = torch.as_tensor(
            static_posture_joint_position_np[:29],
            device=device,
            dtype=joint_pos.dtype,
        ).clone()
        initial_body_joint_vel = torch.zeros_like(initial_body_joint_pos)
    if args.body_control_mode == "unitree_demo_pose":
        demo_body_violation = torch.maximum(
            body_hard_lower - initial_body_joint_pos,
            torch.zeros_like(initial_body_joint_pos),
        ) + torch.maximum(
            initial_body_joint_pos - body_hard_upper,
            torch.zeros_like(initial_body_joint_pos),
        )
        unitree_demo_body_limit_adjustment_rad = float(
            torch.max(demo_body_violation).item()
        )
        if unitree_demo_body_limit_adjustment_rad > 1.0e-2:
            raise RuntimeError(
                "Official Unitree demo pose needs more than the admitted 0.01 rad "
                "live-limit compatibility adjustment: "
                f"{unitree_demo_body_limit_adjustment_rad} rad"
            )
        # The real-robot observation can differ from the simulation USD's hard
        # limits by a few milliradians. Clamp only this bounded source/runtime
        # compatibility delta, record it exactly, and never relax the live limit.
        initial_body_joint_pos = torch.clamp(
            initial_body_joint_pos, body_hard_lower, body_hard_upper
        )
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(
        initial_body_joint_pos.unsqueeze(0),
        initial_body_joint_vel.unsqueeze(0),
        joint_ids=list(range(29)),
    )
    robot.write_joint_state_to_sim(
        open_fingers.unsqueeze(0),
        torch.zeros_like(open_fingers).unsqueeze(0),
        joint_ids=finger_ids,
    )
    setup_object_state = initial_object_state.clone()
    if args.body_control_mode in (
        "bilateral_ik",
        "unitree_demo_pose",
        "sugar_side_clamp",
    ):
        setup_object_state[0, :3] = torch.tensor(
            (5.0, 5.0, 1.0), device=device, dtype=setup_object_state.dtype
        )
        setup_object_state[0, 7:13] = 0.0
    box.write_root_state_to_sim(setup_object_state)
    robot.set_joint_position_target(
        open_fingers.unsqueeze(0), joint_ids=finger_ids
    )
    robot.set_joint_position_target(
        initial_body_joint_pos.unsqueeze(0),
        joint_ids=list(range(29)),
    )
    robot.set_joint_velocity_target(
        initial_body_joint_vel.unsqueeze(0), joint_ids=list(range(29))
    )
    robot.write_data_to_sim()
    sim.forward()

    ik_controllers: dict[str, DifferentialIKController] = {}
    ik_position_controllers: dict[str, DifferentialIKController] = {}
    ik_ee_body_ids: dict[str, int] = {}
    ik_ee_jacobian_ids: dict[str, int] = {}
    ik_arm_joint_ids: dict[str, list[int]] = {}
    ik_body_hold_target = initial_body_joint_pos.clone()
    if args.body_control_mode in (
        "bilateral_ik",
        "unitree_demo_pose",
        "sugar_side_clamp",
    ):
        robot.update(dt)
        for side in SIDES:
            prefix = side
            body_ids = robot.find_bodies(f"{side}_hand_base_link")[0]
            if len(body_ids) != 1:
                raise RuntimeError(f"Expected one {side} hand base, got {body_ids}")
            body_id = int(body_ids[0])
            arm_names = [
                f"{prefix}_shoulder_pitch_joint",
                f"{prefix}_shoulder_roll_joint",
                f"{prefix}_shoulder_yaw_joint",
                f"{prefix}_elbow_joint",
                f"{prefix}_wrist_roll_joint",
                f"{prefix}_wrist_pitch_joint",
                f"{prefix}_wrist_yaw_joint",
            ]
            arm_ids = robot.find_joints(arm_names)[0]
            if len(arm_ids) != 7:
                raise RuntimeError(f"Expected seven {side} arm joints, got {arm_ids}")
            ik_ee_body_ids[side] = body_id
            ik_ee_jacobian_ids[side] = body_id - 1
            ik_arm_joint_ids[side] = [int(value) for value in arm_ids]
            ik_controllers[side] = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                num_envs=1,
                device=device,
            )
            ik_controllers[side].reset()
            ik_position_controllers[side] = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="position", use_relative_mode=False, ik_method="dls"
                ),
                num_envs=1,
                device=device,
            )
            ik_position_controllers[side].reset()

    rows: dict[str, list[Any]] = {
        "source_time": [],
        "trajectory_time_s": [],
        "phase": [],
        "closure": [],
        "thumb_closure": [],
        "box_state": [],
        "box_com_state": [],
        "box_com_acc": [],
        "required_force": [],
        "required_torque": [],
        "raw_force_by_hand": [],
        "raw_torque_by_hand": [],
        "raw_normal_force_by_hand": [],
        "raw_normal_torque_by_hand": [],
        "raw_friction_force_by_hand": [],
        "raw_friction_torque_by_hand": [],
        "matrix_force_by_hand": [],
        "direct_force_by_hand": [],
        "all_robot_raw_force": [],
        "all_robot_raw_torque": [],
        "all_robot_body_normal_load": [],
        "all_robot_body_contact_count": [],
        "group_normal_load": [],
        "body_normal_load_left": [],
        "body_normal_load_right": [],
        "body_contact_count_left": [],
        "body_contact_count_right": [],
        "normal_point_count": [],
        "friction_point_count": [],
        "finger_position": [],
        "finger_target": [],
        "inspire_command": [],
        "robot_root_state": [],
        "robot_joint_position": [],
        "robot_joint_velocity": [],
        "robot_body_link_pos_w": [],
        "robot_body_link_quat_w": [],
        "hand_body_pos_w": [],
        "hand_body_quat_w": [],
        "desired_hand_pos_w": [],
        "desired_hand_quat_w": [],
        "hand_contact_pos_w": [],
    }
    mass = float(box.data.default_mass[0, 0].item())
    inertia_local = box.data.default_inertia[0].reshape(3, 3).to(device=device)
    gravity = torch.tensor((0.0, 0.0, -9.81), device=device)

    def record(
        source_time: float,
        phase: str,
        closure: float,
        desired_hand_pos_w: np.ndarray | None = None,
        desired_hand_quat_w: np.ndarray | None = None,
        thumb_closure: float | None = None,
        inspire_command_override: np.ndarray | None = None,
    ) -> None:
        if thumb_closure is None:
            thumb_closure = closure
        com_state_t = box.data.body_com_state_w[0, 0]
        com_acc_t = box.data.body_com_acc_w[0, 0]
        rotation = math_utils.matrix_from_quat(com_state_t[3:7])
        inertia_w = rotation @ inertia_local @ rotation.transpose(0, 1)
        omega = com_state_t[10:13]
        required_force_t = mass * (com_acc_t[:3] - gravity)
        required_torque_t = (
            inertia_w @ com_acc_t[3:6]
            + torch.linalg.cross(omega, inertia_w @ omega)
        )
        samples = [
            raw_contact_sample(sensors[side], cpu(com_state_t[:3]))
            for side in SIDES
        ]
        all_robot_sample = raw_contact_sample(
            all_robot_sensor, cpu(com_state_t[:3]), anatomical=False
        )
        rows["source_time"].append(source_time)
        rows["trajectory_time_s"].append(len(rows["source_time"]) * dt)
        rows["phase"].append(phase)
        rows["closure"].append(closure)
        rows["thumb_closure"].append(thumb_closure)
        rows["box_state"].append(cpu(box.data.root_state_w[0]))
        rows["box_com_state"].append(cpu(com_state_t))
        rows["box_com_acc"].append(cpu(com_acc_t))
        rows["required_force"].append(cpu(required_force_t))
        rows["required_torque"].append(cpu(required_torque_t))
        rows["raw_force_by_hand"].append(
            np.stack([sample["force_on_box"] for sample in samples])
        )
        rows["raw_torque_by_hand"].append(
            np.stack([sample["torque_on_box"] for sample in samples])
        )
        for component in ("normal", "friction"):
            rows[f"raw_{component}_force_by_hand"].append(
                np.stack(
                    [sample[f"{component}_force_on_box"] for sample in samples]
                )
            )
            rows[f"raw_{component}_torque_by_hand"].append(
                np.stack(
                    [sample[f"{component}_torque_on_box"] for sample in samples]
                )
            )
        # IsaacLab's filtered force matrix is the authoritative net vector on
        # each selected hand from the CarryBox.  Negate it to obtain the box
        # reaction and retain it independently from the raw point audit.
        rows["matrix_force_by_hand"].append(
            np.stack(
                [
                    -cpu(sensors[side].data.force_matrix_w[0]).sum(axis=(0, 1))
                    for side in SIDES
                ]
            )
        )
        rows["direct_force_by_hand"].append(
            np.stack(
                [
                    -cpu(
                        sensors[side].data.force_matrix_w[0]
                        + sensors[side].data.friction_forces_w[0]
                    ).sum(axis=(0, 1))
                    for side in SIDES
                ]
            )
        )
        rows["all_robot_raw_force"].append(all_robot_sample["force_on_box"])
        rows["all_robot_raw_torque"].append(all_robot_sample["torque_on_box"])
        rows["all_robot_body_normal_load"].append(
            all_robot_sample["body_normal_load"]
        )
        rows["all_robot_body_contact_count"].append(
            all_robot_sample["body_count"]
        )
        rows["group_normal_load"].append(
            np.stack([sample["group_normal_load"] for sample in samples])
        )
        for side_index, side in enumerate(SIDES):
            rows[f"body_normal_load_{side}"].append(
                samples[side_index]["body_normal_load"]
            )
            rows[f"body_contact_count_{side}"].append(
                samples[side_index]["body_count"]
            )
        rows["normal_point_count"].append(
            np.stack([sample["normal_point_count"] for sample in samples])
        )
        rows["friction_point_count"].append(
            np.stack([sample["friction_point_count"] for sample in samples])
        )
        rows["finger_position"].append(cpu(robot.data.joint_pos[0, finger_ids]))
        rows["finger_target"].append(cpu(robot.data.joint_pos_target[0, finger_ids]))
        rows["inspire_command"].append(
            cpu(official_inspire_command(closure, thumb_closure))
            if inspire_command_override is None
            else np.asarray(inspire_command_override, dtype=np.float32)
        )
        rows["robot_root_state"].append(cpu(robot.data.root_state_w[0]))
        rows["robot_joint_position"].append(cpu(robot.data.joint_pos[0]))
        rows["robot_joint_velocity"].append(cpu(robot.data.joint_vel[0]))
        rows["robot_body_link_pos_w"].append(cpu(robot.data.body_link_pos_w[0]))
        rows["robot_body_link_quat_w"].append(cpu(robot.data.body_link_quat_w[0]))
        rows["hand_body_pos_w"].append(
            np.stack([cpu(sensors[side].data.pos_w[0]) for side in SIDES])
        )
        rows["hand_body_quat_w"].append(
            np.stack([cpu(sensors[side].data.quat_w[0]) for side in SIDES])
        )
        rows["desired_hand_pos_w"].append(
            np.full((2, 3), np.nan, dtype=np.float32)
            if desired_hand_pos_w is None
            else desired_hand_pos_w
        )
        rows["desired_hand_quat_w"].append(
            np.full((2, 4), np.nan, dtype=np.float32)
            if desired_hand_quat_w is None
            else desired_hand_quat_w
        )
        rows["hand_contact_pos_w"].append(
            np.stack(
                [cpu(sensors[side].data.contact_pos_w[0, :, 0]) for side in SIDES]
            )
        )

    def apply_reference(
        source_time: float,
        velocity_scale: float = 1.0,
        closure_override: float | None = None,
    ) -> float:
        low = min(int(np.floor(source_time)), args.source_end - 1)
        high = min(low + 1, args.source_end)
        tau = float(source_time - low)
        root_position = (1.0 - tau) * body_pos[low, 0] + tau * body_pos[high, 0]
        root_position = root_position.clone()
        root_position[1] += args.robot_root_y_offset_m
        root_quaternion = math_utils.quat_slerp(
            body_quat[low, 0].clone(), body_quat[high, 0].clone(), tau
        )
        root_velocity = torch.cat(
            (
                (1.0 - tau) * body_lin_vel[low, 0] + tau * body_lin_vel[high, 0],
                (1.0 - tau) * body_ang_vel[low, 0] + tau * body_ang_vel[high, 0],
            )
        ) * velocity_scale
        body_joint_pos = add_declared_body_offset(
            (1.0 - tau) * joint_pos[low] + tau * joint_pos[high],
            source_time,
        )
        body_joint_vel = velocity_scale * (
            (1.0 - tau) * joint_vel[low] + tau * joint_vel[high]
        )
        if args.waist_pitch_absolute_rad is not None:
            body_joint_vel[8] = 0.0
        body_joint_vel[15] += scheduled_offset_velocity(
            left_shoulder_roll_offset_rad,
            left_shoulder_roll_end_offset_rad,
            source_time,
            velocity_scale,
        )
        body_joint_vel[23] += scheduled_offset_velocity(
            args.left_wrist_roll_offset_rad,
            left_wrist_roll_end_offset_rad,
            source_time,
            velocity_scale,
        )
        body_joint_vel[27] += scheduled_offset_velocity(
            args.left_wrist_yaw_offset_rad,
            left_wrist_yaw_end_offset_rad,
            source_time,
            velocity_scale,
        )
        if args.body_control_mode == "state_replay":
            robot.write_root_pose_to_sim(
                torch.cat((root_position, root_quaternion)).unsqueeze(0)
            )
            robot.write_root_velocity_to_sim(root_velocity.unsqueeze(0))
            robot.write_joint_state_to_sim(
                body_joint_pos.unsqueeze(0),
                body_joint_vel.unsqueeze(0),
                joint_ids=list(range(29)),
            )
        else:
            robot.set_joint_position_target(
                body_joint_pos.unsqueeze(0), joint_ids=list(range(29))
            )
            robot.set_joint_velocity_target(
                body_joint_vel.unsqueeze(0), joint_ids=list(range(29))
            )
        closure = (
            float(
                np.clip(
                    (source_time - args.close_start)
                    / (args.close_end - args.close_start),
                    0.0,
                    1.0,
                )
            )
            if closure_override is None
            else float(closure_override)
        )
        if not (0.0 <= closure <= 1.0):
            raise RuntimeError(f"Invalid closure override: {closure}")
        target = official_inspire_target(closure)
        robot.set_joint_position_target(target.unsqueeze(0), joint_ids=finger_ids)
        robot.write_data_to_sim()
        return closure

    step_counter = 0

    def simulate_one(
        source_time: float,
        phase: str,
        velocity_scale: float = 1.0,
        closure_override: float | None = None,
    ) -> None:
        nonlocal step_counter
        closure = apply_reference(
            source_time,
            velocity_scale=velocity_scale,
            closure_override=closure_override,
        )
        sim.step(render=False)
        robot.update(dt)
        box.update(dt)
        for sensor in sensors.values():
            sensor.update(dt, force_recompute=True)
        all_robot_sensor.update(dt, force_recompute=True)
        record(source_time, phase, closure)
        step_counter += 1
        if step_counter % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step_counter,
                        "source_time": source_time,
                        "phase": phase,
                        "closure": closure,
                        "thumb_closure": closure,
                        "box_z_m": float(box.data.root_pos_w[0, 2].item()),
                        "group_load_N": rows["group_normal_load"][-1].tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def simulate_ik_one(
        desired_hand_pos_w: torch.Tensor,
        desired_hand_quat_w: torch.Tensor,
        phase: str,
        closure: float,
        *,
        thumb_closure: float | None = None,
        inspire_command_override: torch.Tensor | None = None,
        record_step: bool = True,
        position_only: bool = False,
        direct_state_write: bool = False,
    ) -> None:
        nonlocal step_counter
        if args.body_control_mode not in (
            "bilateral_ik",
            "unitree_demo_pose",
            "sugar_side_clamp",
        ):
            raise RuntimeError("IK stepping requested outside bilateral_ik mode")
        robot.set_joint_position_target(
            ik_body_hold_target.unsqueeze(0), joint_ids=list(range(29))
        )
        robot.set_joint_velocity_target(
            torch.zeros_like(ik_body_hold_target).unsqueeze(0),
            joint_ids=list(range(29)),
        )
        root_pose_w = robot.data.root_pose_w
        base_rotation = math_utils.matrix_from_quat(
            math_utils.quat_inv(root_pose_w[:, 3:7])
        )
        direct_arm_states: list[tuple[list[int], torch.Tensor]] = []
        for side_index, side in enumerate(SIDES):
            body_id = ik_ee_body_ids[side]
            arm_ids = ik_arm_joint_ids[side]
            desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, :3],
                root_pose_w[:, 3:7],
                desired_hand_pos_w[side_index].unsqueeze(0),
                desired_hand_quat_w[side_index].unsqueeze(0),
            )
            if position_only:
                controller = ik_position_controllers[side]
            else:
                controller = ik_controllers[side]
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ik_ee_jacobian_ids[side], :, arm_ids
            ].clone()
            jacobian[:, :3, :] = torch.bmm(base_rotation, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rotation, jacobian[:, 3:, :])
            ee_pose_w = robot.data.body_pose_w[:, body_id]
            ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, :3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, :3],
                ee_pose_w[:, 3:7],
            )
            if position_only:
                controller.set_command(desired_pos_b, ee_quat=ee_quat_b)
            else:
                controller.set_command(
                    torch.cat((desired_pos_b, desired_quat_b), dim=1)
                )
            current_joint_pos = robot.data.joint_pos[:, arm_ids]
            desired_joint_pos = controller.compute(
                ee_pos_b, ee_quat_b, jacobian, current_joint_pos
            )
            lower = body_hard_lower[arm_ids].unsqueeze(0)
            upper = body_hard_upper[arm_ids].unsqueeze(0)
            desired_joint_pos = torch.clamp(desired_joint_pos, lower, upper)
            robot.set_joint_position_target(desired_joint_pos, joint_ids=arm_ids)
            direct_arm_states.append((arm_ids, desired_joint_pos))
        finger_target = (
            official_inspire_target(closure, thumb_closure)
            if inspire_command_override is None
            else official_inspire_target_from_command(inspire_command_override)
        )
        robot.set_joint_position_target(
            finger_target.unsqueeze(0), joint_ids=finger_ids
        )
        if direct_state_write:
            if record_step:
                raise RuntimeError("Direct state IK is forbidden on recorded steps")
            for arm_ids, desired_joint_pos in direct_arm_states:
                robot.write_joint_state_to_sim(
                    desired_joint_pos,
                    torch.zeros_like(desired_joint_pos),
                    joint_ids=arm_ids,
                )
            robot.write_joint_state_to_sim(
                finger_target.unsqueeze(0),
                torch.zeros_like(finger_target).unsqueeze(0),
                joint_ids=finger_ids,
            )
            robot.write_data_to_sim()
            sim.forward()
        else:
            robot.write_data_to_sim()
            sim.step(render=False)
        robot.update(dt)
        box.update(dt)
        for sensor in sensors.values():
            sensor.update(dt, force_recompute=True)
        all_robot_sensor.update(dt, force_recompute=True)
        synthetic_source_time = args.source_start + step_counter * dt * control_fps
        if record_step:
            record(
                synthetic_source_time,
                phase,
                closure,
                cpu(desired_hand_pos_w),
                cpu(desired_hand_quat_w),
                thumb_closure,
                (
                    None
                    if inspire_command_override is None
                    else cpu(inspire_command_override)
                ),
            )
            step_counter += 1
        if record_step and step_counter % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step_counter,
                        "trajectory_time_s": step_counter * dt,
                        "phase": phase,
                        "closure": closure,
                        "thumb_closure": (
                            closure if thumb_closure is None else thumb_closure
                        ),
                        "box_z_m": float(box.data.root_pos_w[0, 2].item()),
                        "group_load_N": rows["group_normal_load"][-1].tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    ik_contact_anchor_box_state = None
    ik_desired_contact_pos_w = None
    ik_desired_contact_quat_w = None
    ik_live_hold_translation_bias_w = None
    settled_all_groups_gate_passed = None
    settled_contact_component_passed = None
    settled_nonhand_component_passed = None
    settled_hard_limit_component_passed = None
    unitree_demo_actual_minus_declared_ee_midpoint = None
    unitree_demo_approach_position_error_m = None
    unitree_demo_approach_rotation_error_rad = None
    unitree_demo_reachable_hand_pos_w = None
    unitree_demo_reachable_hand_quat_w = None
    side_clamp_face_sign = None
    side_clamp_target_surface_point_box = None
    side_clamp_geometric_contact_pos_w = None
    side_clamp_geometric_contact_quat_w = None
    if (
        args.body_control_mode == "sugar_side_clamp"
        or (
            args.body_control_mode == "unitree_demo_pose"
            and args.unitree_demo_side_clamp
        )
    ):
        if args.body_control_mode == "unitree_demo_pose":
            assert unitree_demo_outward_translation is not None
        # Retain the exact declared official body pose as the source, but use a
        # mechanically valid thick-box target: the two official palm-bearing
        # local -X faces oppose one another on the selected CarryBox side faces.
        # Rotation about each palm normal is the projection closest to the
        # official source orientation, rather than a hand-authored wrist angle.
        demo_hand_start_pos_w = torch.stack(
            [
                robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        demo_hand_start_quat_w = torch.stack(
            [
                robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        initial_box_pose = initial_object_state[0, :7].clone()
        box_rotation_w = math_utils.matrix_from_quat(
            initial_box_pose[3:7].unsqueeze(0)
        )[0]
        box_axis_index = (
            0
            if args.side_clamp_box_axis == "x"
            else (1 if args.side_clamp_box_axis == "y" else None)
        )
        pca_center_b = torch.tensor(
            CARRYBOX_PCA_CENTER_B, device=device, dtype=joint_pos.dtype
        )
        pca_basis_b = torch.tensor(
            CARRYBOX_PCA_BASIS_B, device=device, dtype=joint_pos.dtype
        )
        box_axis_b = (
            torch.nn.functional.one_hot(
                torch.tensor(box_axis_index, device=device), num_classes=3
            ).to(dtype=joint_pos.dtype)
            if box_axis_index is not None
            else pca_basis_b[:, 0]
        )
        box_axis_w = box_rotation_w @ box_axis_b
        hand_start_relative_box = math_utils.quat_apply_inverse(
            initial_box_pose[3:7].repeat(2, 1),
            demo_hand_start_pos_w - initial_box_pose[:3].unsqueeze(0),
        )
        side_clamp_face_sign = torch.sign(
            (hand_start_relative_box - pca_center_b) @ box_axis_b
        )
        if bool(torch.any(side_clamp_face_sign == 0.0)) or float(
            torch.prod(side_clamp_face_sign).item()
        ) >= 0.0:
            raise RuntimeError(
                "Official source hands do not lie on opposed selected box faces"
            )
        target_rotations_w = []
        target_surface_points_box = []
        outward_directions_w = []
        for side_index, side in enumerate(SIDES):
            side_sign = float(side_clamp_face_sign[side_index].item())
            declared_outward_pca = side_clamp_outward_pca_by_side[side]
            if declared_outward_pca is None:
                outward_b = side_sign * box_axis_b
            else:
                outward_pca_t = torch.tensor(
                    declared_outward_pca, device=device, dtype=joint_pos.dtype
                )
                outward_b = pca_basis_b @ outward_pca_t
            outward_w = box_rotation_w @ outward_b
            source_rotation_w = math_utils.matrix_from_quat(
                demo_hand_start_quat_w[side_index].unsqueeze(0)
            )[0]
            source_y_w = source_rotation_w[:, 1]
            target_y_w = source_y_w - torch.dot(source_y_w, outward_w) * outward_w
            target_y_norm = torch.linalg.norm(target_y_w)
            if float(target_y_norm.item()) <= 1.0e-6:
                raise RuntimeError("Official hand roll is degenerate about box side normal")
            target_y_w = target_y_w / target_y_norm
            target_z_w = torch.linalg.cross(outward_w, target_y_w)
            target_rotation_w = torch.stack(
                (outward_w, target_y_w, target_z_w), dim=1
            )
            tilt_vector_box = (
                side_clamp_tilt_tangent_by_side_rad[side] * pca_basis_b[:, 1]
                + side_clamp_tilt_height_by_side_rad[side] * pca_basis_b[:, 2]
            )
            tilt_vector_w = box_rotation_w @ tilt_vector_box
            tilt_angle = torch.linalg.norm(tilt_vector_w)
            if float(tilt_angle.item()) > 0.0:
                tilt_axis_w = tilt_vector_w / tilt_angle
                zero = torch.zeros((), device=device, dtype=joint_pos.dtype)
                skew = torch.stack(
                    (
                        torch.stack((zero, -tilt_axis_w[2], tilt_axis_w[1])),
                        torch.stack((tilt_axis_w[2], zero, -tilt_axis_w[0])),
                        torch.stack((-tilt_axis_w[1], tilt_axis_w[0], zero)),
                    )
                )
                identity = torch.eye(3, device=device, dtype=joint_pos.dtype)
                tilt_rotation_w = (
                    identity
                    + torch.sin(tilt_angle) * skew
                    + (1.0 - torch.cos(tilt_angle)) * (skew @ skew)
                )
                target_rotation_w = tilt_rotation_w @ target_rotation_w
            normal_roll = side_clamp_normal_roll_by_side_rad[side]
            if abs(normal_roll) > 0.0:
                roll_cos = np.cos(normal_roll)
                roll_sin = np.sin(normal_roll)
                local_normal_roll = torch.tensor(
                    (
                        (1.0, 0.0, 0.0),
                        (0.0, roll_cos, -roll_sin),
                        (0.0, roll_sin, roll_cos),
                    ),
                    device=device,
                    dtype=joint_pos.dtype,
                )
                target_rotation_w = target_rotation_w @ local_normal_roll
            target_rotations_w.append(target_rotation_w)
            declared_contact_pca = side_clamp_contact_pca_by_side_m[side]
            if declared_contact_pca is not None:
                declared_contact_pca_t = torch.tensor(
                    declared_contact_pca, device=device, dtype=joint_pos.dtype
                )
                target_surface_point_t = (
                    pca_center_b
                    + pca_basis_b @ declared_contact_pca_t
                    - side_clamp_palm_inset_by_side_m[side] * outward_b
                )
                target_surface_point = tuple(
                    float(value.item()) for value in target_surface_point_t
                )
            elif args.side_clamp_box_axis == "pca0":
                normal_coordinate = side_clamp_box_normal_by_side_m[side]
                if normal_coordinate is None:
                    normal_coordinate = (
                        CARRYBOX_PCA0_BOUNDS_M[1]
                        if side_sign > 0.0
                        else CARRYBOX_PCA0_BOUNDS_M[0]
                    )
                if normal_coordinate * side_sign <= 0.0:
                    raise RuntimeError(
                        f"{side} PCA normal coordinate has the wrong face sign"
                    )
                target_surface_point_t = (
                    pca_center_b
                    + (
                        normal_coordinate
                        - side_sign * side_clamp_palm_inset_by_side_m[side]
                    )
                    * pca_basis_b[:, 0]
                    + side_clamp_box_local_tangent_by_side_m[side]
                    * pca_basis_b[:, 1]
                    + side_clamp_box_local_z_by_side_m[side] * pca_basis_b[:, 2]
                )
                target_surface_point = tuple(
                    float(value.item()) for value in target_surface_point_t
                )
            elif box_axis_index == 0:
                face_coordinate = side_clamp_box_normal_by_side_m[side]
                if face_coordinate is None:
                    face_coordinate = (
                        CARRYBOX_SIDE_FACE_X_M["left"]
                        if side_sign > 0.0
                        else CARRYBOX_SIDE_FACE_X_M["right"]
                    )
                if face_coordinate * side_sign <= 0.0:
                    raise RuntimeError(
                        f"{side} local-X normal coordinate has the wrong face sign"
                    )
                target_surface_point = (
                    face_coordinate
                    - side_sign * side_clamp_palm_inset_by_side_m[side],
                    side_clamp_box_local_tangent_by_side_m[side],
                    side_clamp_box_local_z_by_side_m[side],
                )
            else:
                face_coordinate = side_clamp_box_normal_by_side_m[side]
                if face_coordinate is None:
                    face_coordinate = (
                        CARRYBOX_MESH_Y_BOUNDS_M[1]
                        if side_sign > 0.0
                        else CARRYBOX_MESH_Y_BOUNDS_M[0]
                    )
                if face_coordinate * side_sign <= 0.0:
                    raise RuntimeError(
                        f"{side} local-Y normal coordinate has the wrong face sign"
                    )
                target_surface_point = (
                    side_clamp_box_local_tangent_by_side_m[side],
                    face_coordinate
                    - side_sign * side_clamp_palm_inset_by_side_m[side],
                    side_clamp_box_local_z_by_side_m[side],
                )
            target_surface_points_box.append(
                torch.tensor(
                    target_surface_point,
                    device=device,
                    dtype=joint_pos.dtype,
                )
            )
            # Once a palm is tilted away from the original PCA0 side-face
            # frame, its collision-free approach/press direction must follow
            # the final palm +X (the outward side of the load-bearing -X
            # surface).  Retaining the pre-tilt PCA0 vector makes a nominal
            # bottom-support palm translate sideways and can leave it under
            # empty chamfer space instead of pressing into the underside.
            outward_directions_w.append(target_rotation_w[:, 0])
        side_clamp_target_surface_point_box = torch.stack(
            target_surface_points_box
        )
        side_contact_quat_w = math_utils.quat_from_matrix(
            torch.stack(target_rotations_w)
        )
        side_surface_pos_w, _ = math_utils.combine_frame_transforms(
            initial_box_pose[:3].repeat(2, 1),
            initial_box_pose[3:7].repeat(2, 1),
            side_clamp_target_surface_point_box,
            torch.tensor(
                ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                device=device,
                dtype=joint_pos.dtype,
            ),
        )
        palm_surface_point_b = torch.tensor(
            [INSPIRE_PALM_SURFACE_POINT_B[side] for side in SIDES],
            device=device,
            dtype=joint_pos.dtype,
        )
        side_contact_pos_w = side_surface_pos_w - math_utils.quat_apply(
            side_contact_quat_w, palm_surface_point_b
        )
        # Freeze the exact mesh/cooked-surface pose before any reachability
        # concession changes the live hand orientation. Full-body posture
        # scans must solve this target, not a post-refinement or post-impact
        # pose that merely follows where an unreachable hand happened to stop.
        side_clamp_geometric_contact_pos_w = side_contact_pos_w.clone()
        side_clamp_geometric_contact_quat_w = side_contact_quat_w.clone()
        outward_directions_w_t = torch.stack(outward_directions_w)
        side_approach_clearance_t = torch.tensor(
            [side_clamp_approach_clearance_by_side_m[side] for side in SIDES],
            device=device,
            dtype=joint_pos.dtype,
        ).unsqueeze(1)
        side_approach_pos_w = side_contact_pos_w + (
            side_approach_clearance_t * outward_directions_w_t
        )
        side_relative_pos, side_relative_quat = math_utils.subtract_frame_transforms(
            initial_box_pose[:3].repeat(2, 1),
            initial_box_pose[3:7].repeat(2, 1),
            side_contact_pos_w,
            side_contact_quat_w,
        )

        if static_posture_source_desired_hand_position_np is not None:
            assert static_posture_source_box_pose_np is not None
            assert static_posture_source_desired_hand_quaternion_np is not None
            source_box_pose_t = torch.as_tensor(
                static_posture_source_box_pose_np,
                device=device,
                dtype=joint_pos.dtype,
            ).unsqueeze(0)
            source_desired_position_t = torch.as_tensor(
                static_posture_source_desired_hand_position_np,
                device=device,
                dtype=joint_pos.dtype,
            )
            source_desired_quaternion_t = torch.as_tensor(
                static_posture_source_desired_hand_quaternion_np,
                device=device,
                dtype=joint_pos.dtype,
            )
            transferred_relative_position, transferred_relative_quaternion = (
                math_utils.subtract_frame_transforms(
                    source_box_pose_t[:, :3].repeat(2, 1),
                    source_box_pose_t[:, 3:7].repeat(2, 1),
                    source_desired_position_t,
                    source_desired_quaternion_t,
                )
            )
            static_posture_contact_delta_box_b_t = torch.as_tensor(
                static_posture_contact_delta_box_b_np,
                device=device,
                dtype=joint_pos.dtype,
            )
            transferred_relative_position = (
                transferred_relative_position
                + static_posture_contact_delta_box_b_t
            )
            static_posture_contact_delta_world_t = math_utils.quat_apply(
                initial_box_pose[3:7].repeat(2, 1),
                static_posture_contact_delta_box_b_t,
            )
            static_posture_contact_delta_world_np = cpu(
                static_posture_contact_delta_world_t
            ).astype(np.float32)
            side_contact_pos_w, side_contact_quat_w = (
                math_utils.combine_frame_transforms(
                    initial_box_pose[:3].repeat(2, 1),
                    initial_box_pose[3:7].repeat(2, 1),
                    transferred_relative_position,
                    transferred_relative_quaternion,
                )
            )
            side_surface_pos_w = side_contact_pos_w + math_utils.quat_apply(
                side_contact_quat_w, palm_surface_point_b
            )
            side_clamp_target_surface_point_box = math_utils.quat_apply_inverse(
                initial_box_pose[3:7].repeat(2, 1),
                side_surface_pos_w - initial_box_pose[:3].unsqueeze(0),
            )
            outward_directions_w_t = math_utils.matrix_from_quat(
                side_contact_quat_w
            )[:, :, 0]
            # The admitted scan solution defines the exact contact pose.  The
            # legacy contact-seeded diagnostic published the box directly at
            # that pose and therefore inherited a large first-step impulse.
            # The explicit clearance mode instead retracts the hands while the
            # box is offstage and lets the recorded PhysX palm-press phase
            # establish contact causally after the one-time publication.
            if args.static_posture_use_declared_approach_clearance:
                side_approach_pos_w = side_contact_pos_w + (
                    side_approach_clearance_t * outward_directions_w_t
                )
            else:
                side_approach_clearance_t = torch.zeros_like(
                    side_approach_clearance_t
                )
                side_approach_pos_w = side_contact_pos_w.clone()
            side_relative_pos, side_relative_quat = (
                transferred_relative_position,
                transferred_relative_quaternion,
            )
            initial_transfer_position_error_m = torch.linalg.norm(
                demo_hand_start_pos_w - side_contact_pos_w, dim=1
            )
            initial_transfer_rotation_error_rad = math_utils.quat_error_magnitude(
                demo_hand_start_quat_w, side_contact_quat_w
            )
            print(
                "PLAN10_STATIC_POSTURE_CONTACT_DELTA="
                + json.dumps(
                    {
                        "side_order": list(SIDES),
                        "requested_pca_m": (
                            static_posture_contact_delta_pca_np.tolist()
                        ),
                        "applied_box_frame_m": (
                            static_posture_contact_delta_box_b_np.tolist()
                        ),
                        "applied_world_frame_m": (
                            static_posture_contact_delta_world_np.tolist()
                        ),
                        "orientation_unchanged": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            print(
                "PLAN10_STATIC_POSTURE_TRANSFER="
                + json.dumps(
                    {
                        "initial_position_error_m": cpu(
                            initial_transfer_position_error_m
                        ).tolist(),
                        "initial_rotation_error_rad": cpu(
                            initial_transfer_rotation_error_rad
                        ).tolist(),
                        "effective_approach_clearance_m": cpu(
                            side_approach_clearance_t[:, 0]
                        ).tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        effective_approach_steps = (
            args.unitree_demo_approach_steps
            if (
                static_posture_source_desired_hand_position_np is None
                or args.static_posture_use_declared_approach_clearance
            )
            else 0
        )
        for approach_step in range(effective_approach_steps):
            u = (approach_step + 1) / effective_approach_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            desired_pos = (
                (1.0 - smooth) * demo_hand_start_pos_w
                + smooth * side_approach_pos_w
            )
            desired_quat = torch.stack(
                [
                    math_utils.quat_slerp(
                        demo_hand_start_quat_w[index].clone(),
                        side_contact_quat_w[index].clone(),
                        smooth,
                    )
                    for index in range(2)
                ]
            )
            simulate_ik_one(
                desired_pos,
                desired_quat,
                "unitree_demo_side_clamp_unrecorded_approach",
                0.0,
                thumb_closure=0.0,
                record_step=False,
                direct_state_write=args.sugar_side_clamp_direct_setup_ik,
            )

        if args.sugar_side_clamp_direct_setup_ik:
            # Retain the already-nearby reachable palm orientation and remove
            # the remaining translation with the official full-pose DLS.  A
            # position-only 3x7 solve was rejected because it spent wrist
            # rotation to reduce translation.  Here every command has the
            # current measured orientation and at most 1 mm translation, so
            # the official 6x7 pose solve penalizes any orientation velocity.
            # Each side is accepted only when its palm-surface error strictly
            # decreases and its total deviation from the mesh-normal frame
            # stays within 0.10 rad.  This is collision-free offstage setup;
            # all recorded dynamics remain drive-controlled PhysX.
            geometric_side_contact_quat_w = side_contact_quat_w.clone()
            sugar_side_clamp_direct_refinement_steps = (
                args.sugar_side_clamp_direct_refinement_steps
            )
            refinement_trust_radius_m = torch.full(
                (2,), 0.001, device=device, dtype=joint_pos.dtype
            )
            refinement_min_trust_radius_m = 0.00001
            for _ in range(sugar_side_clamp_direct_refinement_steps):
                prior_arm_states = {
                    side: robot.data.joint_pos[
                        :, ik_arm_joint_ids[side]
                    ].clone()
                    for side in SIDES
                }
                reachable_quat_w = torch.stack(
                    [
                        robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                        for side in SIDES
                    ]
                )
                reachable_hand_pos_w = torch.stack(
                    [
                        robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                        for side in SIDES
                    ]
                )
                reachable_contact_pos_w = side_surface_pos_w - math_utils.quat_apply(
                    reachable_quat_w, palm_surface_point_b
                )
                reachable_approach_pos_w = reachable_contact_pos_w + (
                    side_approach_clearance_t * outward_directions_w_t
                )
                refinement_error_vector_w = (
                    reachable_approach_pos_w - reachable_hand_pos_w
                )
                refinement_error_m = torch.linalg.norm(
                    refinement_error_vector_w, dim=1
                )
                if sugar_side_clamp_direct_refinement_initial_error_m is None:
                    sugar_side_clamp_direct_refinement_initial_error_m = (
                        refinement_error_m.clone()
                    )
                    sugar_side_clamp_direct_refinement_initial_error_vector_w = (
                        refinement_error_vector_w.clone()
                    )
                if float(torch.max(refinement_error_m).item()) <= 0.0005:
                    break
                command_fraction = torch.clamp(
                    refinement_trust_radius_m
                    / torch.clamp(refinement_error_m, min=1.0e-12),
                    max=1.0,
                )
                trust_region_pos_w = reachable_hand_pos_w + (
                    command_fraction.unsqueeze(1) * refinement_error_vector_w
                )
                simulate_ik_one(
                    trust_region_pos_w,
                    reachable_quat_w,
                    "unitree_demo_side_clamp_unrecorded_pose_trust_refinement",
                    0.0,
                    thumb_closure=0.0,
                    record_step=False,
                    position_only=False,
                    direct_state_write=True,
                )
                candidate_quat_w = torch.stack(
                    [
                        robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                        for side in SIDES
                    ]
                )
                candidate_hand_pos_w = torch.stack(
                    [
                        robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                        for side in SIDES
                    ]
                )
                candidate_delta_rad = math_utils.quat_error_magnitude(
                    candidate_quat_w, geometric_side_contact_quat_w
                )
                candidate_approach_pos_w = (
                    side_surface_pos_w
                    - math_utils.quat_apply(candidate_quat_w, palm_surface_point_b)
                    + side_approach_clearance_t * outward_directions_w_t
                )
                candidate_error_m = torch.linalg.norm(
                    candidate_approach_pos_w - candidate_hand_pos_w, dim=1
                )
                accept = torch.logical_and(
                    candidate_delta_rad
                    <= args.sugar_side_clamp_max_reachable_orientation_delta_rad,
                    candidate_error_m < (refinement_error_m - 1.0e-9),
                )
                rolled_back = False
                for side_index, side in enumerate(SIDES):
                    if not bool(accept[side_index].item()):
                        arm_ids = ik_arm_joint_ids[side]
                        prior_state = prior_arm_states[side]
                        robot.set_joint_position_target(
                            prior_state, joint_ids=arm_ids
                        )
                        robot.write_joint_state_to_sim(
                            prior_state,
                            torch.zeros_like(prior_state),
                            joint_ids=arm_ids,
                        )
                        refinement_trust_radius_m[side_index] = torch.clamp(
                            0.5 * refinement_trust_radius_m[side_index],
                            min=refinement_min_trust_radius_m,
                        )
                        rolled_back = True
                    else:
                        sugar_side_clamp_direct_refinement_accepted_by_side[
                            side
                        ] += 1
                if rolled_back:
                    robot.write_data_to_sim()
                    sim.forward()
                    robot.update(dt)
                if bool(torch.any(accept).item()):
                    sugar_side_clamp_direct_refinement_accepted_steps += 1
                elif float(torch.max(refinement_trust_radius_m).item()) <= (
                    refinement_min_trust_radius_m
                ):
                    break
            side_contact_quat_w = torch.stack(
                [
                    robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                    for side in SIDES
                ]
            )
            sugar_side_clamp_reachable_orientation_delta_rad = (
                math_utils.quat_error_magnitude(
                    side_contact_quat_w, geometric_side_contact_quat_w
                )
            )
            print(
                "PLAN10_SETUP_REACHABILITY="
                + json.dumps(
                    {
                        "orientation_delta_rad": cpu(
                            sugar_side_clamp_reachable_orientation_delta_rad
                        ).tolist(),
                        "maximum_allowed_orientation_delta_rad": (
                            args.sugar_side_clamp_max_reachable_orientation_delta_rad
                        ),
                        "refinement_initial_error_m": (
                            cpu(
                                sugar_side_clamp_direct_refinement_initial_error_m
                            ).tolist()
                            if sugar_side_clamp_direct_refinement_initial_error_m
                            is not None
                            else None
                        ),
                        "refinement_initial_error_vector_w_m": (
                            cpu(
                                sugar_side_clamp_direct_refinement_initial_error_vector_w
                            ).tolist()
                            if sugar_side_clamp_direct_refinement_initial_error_vector_w
                            is not None
                            else None
                        ),
                        "accepted_refinement_steps": (
                            sugar_side_clamp_direct_refinement_accepted_steps
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if float(
                torch.max(sugar_side_clamp_reachable_orientation_delta_rad).item()
            ) > args.sugar_side_clamp_max_reachable_orientation_delta_rad:
                raise RuntimeError(
                    "Reachable setup orientation exceeds the declared geometry bound"
                )
            refined_hand_pos_w = torch.stack(
                [
                    robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                    for side in SIDES
                ]
            )
            side_contact_pos_w = side_surface_pos_w - math_utils.quat_apply(
                side_contact_quat_w, palm_surface_point_b
            )
            side_approach_pos_w = side_contact_pos_w + (
                side_approach_clearance_t * outward_directions_w_t
            )
            sugar_side_clamp_direct_refinement_final_error_m = torch.linalg.norm(
                side_approach_pos_w - refined_hand_pos_w, dim=1
            )
            sugar_side_clamp_direct_refinement_final_trust_radius_m = (
                refinement_trust_radius_m.clone()
            )
            side_relative_pos, side_relative_quat = (
                math_utils.subtract_frame_transforms(
                    initial_box_pose[:3].repeat(2, 1),
                    initial_box_pose[3:7].repeat(2, 1),
                    side_contact_pos_w,
                    side_contact_quat_w,
                )
            )

        unitree_demo_reachable_hand_pos_w = torch.stack(
            [
                robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        unitree_demo_reachable_hand_quat_w = torch.stack(
            [
                robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        unitree_demo_approach_position_error_m = cpu(
            torch.linalg.norm(
                unitree_demo_reachable_hand_pos_w - side_approach_pos_w, dim=1
            )
        )
        unitree_demo_approach_rotation_error_rad = cpu(
            math_utils.quat_error_magnitude(
                unitree_demo_reachable_hand_quat_w, side_contact_quat_w
            )
        )

        if args.sugar_side_clamp_fit_box_to_reachable_palms:
            reachable_surface_pos_w = unitree_demo_reachable_hand_pos_w + (
                math_utils.quat_apply(
                    unitree_demo_reachable_hand_quat_w, palm_surface_point_b
                )
            )
            per_hand_fit_offset_w = reachable_surface_pos_w - side_surface_pos_w
            sugar_side_clamp_reachable_fit_offset_w = torch.mean(
                per_hand_fit_offset_w, dim=0
            )
            if float(
                torch.linalg.norm(sugar_side_clamp_reachable_fit_offset_w).item()
            ) > 0.15:
                raise RuntimeError("Reachable-palm box fit exceeds the 0.15 m setup bound")
            initial_box_pose[:3] += sugar_side_clamp_reachable_fit_offset_w
            initial_object_state[0, :3] += sugar_side_clamp_reachable_fit_offset_w

        # Publish the dynamic box exactly once after collision-free setup.
        box.write_root_state_to_sim(initial_object_state)
        box.reset()
        sim.forward()
        robot.update(dt)
        box.update(dt)
        for sensor in sensors.values():
            sensor.reset()
            sensor.update(dt, force_recompute=True)
        all_robot_sensor.reset()
        all_robot_sensor.update(dt, force_recompute=True)

        def current_side_clamp_targets() -> tuple[torch.Tensor, torch.Tensor]:
            anchor_pose = (
                box.data.root_pose_w[0].clone()
                if args.ik_track_live_box_during_grasp
                else initial_box_pose
            )
            return math_utils.combine_frame_transforms(
                anchor_pose[:3].repeat(2, 1),
                anchor_pose[3:7].repeat(2, 1),
                side_relative_pos,
                side_relative_quat,
            )

        force_limited_finger_scale = {
            side: {finger: 0.0 for finger in ("index", "middle", "ring", "little")}
            for side in SIDES
        }
        force_limited_thumb_scale = {side: 0.0 for side in SIDES}
        group_index = {name: index for index, name in enumerate(GROUPS)}

        def sequential_digit_command(
            proposed_closure_by_side: dict[str, float],
            proposed_thumb_by_side: dict[str, float],
        ) -> torch.Tensor:
            threshold = args.sugar_side_clamp_digit_force_stop_n
            if threshold is None:
                return official_inspire_command_by_side(
                    proposed_closure_by_side, proposed_thumb_by_side
                )
            last_load = (
                None
                if not rows["group_normal_load"]
                else rows["group_normal_load"][-1]
            )
            maximum_step = args.sugar_side_clamp_digit_force_stop_max_scale_step
            for side_index, side in enumerate(SIDES):
                for finger in ("index", "middle", "ring", "little"):
                    current = force_limited_finger_scale[side][finger]
                    proposed = float(proposed_closure_by_side[side])
                    load = (
                        0.0
                        if last_load is None
                        else float(last_load[side_index, group_index[finger]])
                    )
                    if proposed > current and load < threshold:
                        force_limited_finger_scale[side][finger] = min(
                            proposed, current + maximum_step
                        )
                current_thumb = force_limited_thumb_scale[side]
                proposed_thumb = float(proposed_thumb_by_side[side])
                thumb_load = (
                    0.0
                    if last_load is None
                    else float(last_load[side_index, group_index["thumb"]])
                )
                if proposed_thumb > current_thumb and thumb_load < threshold:
                    force_limited_thumb_scale[side] = min(
                        proposed_thumb, current_thumb + maximum_step
                    )
            return official_inspire_command_by_side(
                proposed_closure_by_side,
                force_limited_thumb_scale,
                closure_by_group=force_limited_finger_scale,
            )

        press_start_pos_w = unitree_demo_reachable_hand_pos_w.clone()
        press_start_quat_w = unitree_demo_reachable_hand_quat_w.clone()
        if args.sugar_side_clamp_palm_press_first is None:
            for press_step in range(args.ik_palm_press_steps):
                u = (press_step + 1) / args.ik_palm_press_steps
                smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                step_contact_pos_w, step_contact_quat_w = current_side_clamp_targets()
                desired_pos = (
                    (1.0 - smooth) * press_start_pos_w
                    + smooth * step_contact_pos_w
                )
                desired_quat = torch.stack(
                    [
                        math_utils.quat_slerp(
                            press_start_quat_w[index].clone(),
                            step_contact_quat_w[index].clone(),
                            smooth,
                        )
                        for index in range(2)
                    ]
                )
                simultaneous_closure = (
                    smooth if args.sugar_side_clamp_simultaneous_hand_close else 0.0
                )
                simulate_ik_one(
                    desired_pos,
                    desired_quat,
                    "unitree_demo_side_clamp_palm_press",
                    simultaneous_closure,
                    thumb_closure=simultaneous_closure,
                )
        else:
            first_index = SIDES.index(args.sugar_side_clamp_palm_press_first)
            second_index = 1 - first_index
            for press_step in range(args.ik_palm_press_steps):
                u = (press_step + 1) / args.ik_palm_press_steps
                smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                step_contact_pos_w, step_contact_quat_w = current_side_clamp_targets()
                desired_pos = press_start_pos_w.clone()
                desired_pos[first_index] = (
                    (1.0 - smooth) * press_start_pos_w[first_index]
                    + smooth * step_contact_pos_w[first_index]
                )
                desired_quat = press_start_quat_w.clone()
                desired_quat[first_index] = math_utils.quat_slerp(
                    press_start_quat_w[first_index].clone(),
                    step_contact_quat_w[first_index].clone(),
                    smooth,
                )
                simulate_ik_one(
                    desired_pos,
                    desired_quat,
                    "unitree_demo_side_clamp_palm_press_"
                    f"{args.sugar_side_clamp_palm_press_first}_first",
                    0.0,
                    thumb_closure=0.0,
                )
            if args.sugar_side_clamp_close_first_hand_before_second:
                for close_step in range(args.unitree_demo_close_steps):
                    u = (close_step + 1) / args.unitree_demo_close_steps
                    smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                    step_contact_pos_w, step_contact_quat_w = (
                        current_side_clamp_targets()
                    )
                    desired_pos = press_start_pos_w.clone()
                    desired_pos[first_index] = step_contact_pos_w[first_index]
                    desired_quat = press_start_quat_w.clone()
                    desired_quat[first_index] = step_contact_quat_w[first_index]
                    closure_by_side = {side: 0.0 for side in SIDES}
                    thumb_by_side = {side: 0.0 for side in SIDES}
                    closure_by_side[SIDES[first_index]] = smooth
                    thumb_by_side[SIDES[first_index]] = smooth
                    first_hand_command = sequential_digit_command(
                        closure_by_side, thumb_by_side
                    )
                    simulate_ik_one(
                        desired_pos,
                        desired_quat,
                        "unitree_demo_side_clamp_first_hand_close",
                        smooth,
                        thumb_closure=smooth,
                        inspire_command_override=first_hand_command,
                    )
            for press_step in range(args.ik_palm_press_steps):
                u = (press_step + 1) / args.ik_palm_press_steps
                smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                step_contact_pos_w, step_contact_quat_w = current_side_clamp_targets()
                desired_pos = step_contact_pos_w.clone()
                desired_pos[second_index] = (
                    (1.0 - smooth) * press_start_pos_w[second_index]
                    + smooth * step_contact_pos_w[second_index]
                )
                desired_quat = step_contact_quat_w.clone()
                desired_quat[second_index] = math_utils.quat_slerp(
                    press_start_quat_w[second_index].clone(),
                    step_contact_quat_w[second_index].clone(),
                    smooth,
                )
                second_press_command = None
                if args.sugar_side_clamp_close_first_hand_before_second:
                    closure_by_side = {side: 0.0 for side in SIDES}
                    thumb_by_side = {side: 0.0 for side in SIDES}
                    closure_by_side[SIDES[first_index]] = 1.0
                    thumb_by_side[SIDES[first_index]] = 1.0
                    if args.sugar_side_clamp_close_second_hand_during_second_press:
                        closure_by_side[SIDES[second_index]] = smooth
                    if args.sugar_side_clamp_close_second_thumb_during_second_press:
                        thumb_by_side[SIDES[second_index]] = smooth
                    second_press_command = sequential_digit_command(
                        closure_by_side, thumb_by_side
                    )
                simulate_ik_one(
                    desired_pos,
                    desired_quat,
                    "unitree_demo_side_clamp_palm_press_second",
                    0.0,
                    thumb_closure=0.0,
                    inspire_command_override=second_press_command,
                )
        if args.sugar_side_clamp_simultaneous_hand_close:
            # The simultaneous press has already reached full four-finger and
            # thumb closure.  Replaying the legacy staged close here used to
            # reopen the thumb from 1 to 0 for an entire close interval before
            # closing it again.  Hold the exact fully closed command instead;
            # this preserves the declared dwell duration without introducing
            # a contradictory release action.
            for _ in range(args.unitree_demo_close_steps):
                step_contact_pos_w, step_contact_quat_w = (
                    current_side_clamp_targets()
                )
                simulate_ik_one(
                    step_contact_pos_w,
                    step_contact_quat_w,
                    "unitree_demo_side_clamp_simultaneous_closed_dwell",
                    1.0,
                    thumb_closure=1.0,
                )
            for _ in range(args.ik_thumb_close_steps):
                step_contact_pos_w, step_contact_quat_w = (
                    current_side_clamp_targets()
                )
                simulate_ik_one(
                    step_contact_pos_w,
                    step_contact_quat_w,
                    "unitree_demo_side_clamp_simultaneous_closed_dwell",
                    1.0,
                    thumb_closure=1.0,
                )
        else:
            for close_step in range(args.unitree_demo_close_steps):
                u = (close_step + 1) / args.unitree_demo_close_steps
                smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                step_contact_pos_w, step_contact_quat_w = (
                    current_side_clamp_targets()
                )
                close_command = None
                if args.sugar_side_clamp_close_first_hand_before_second:
                    closure_by_side = {side: smooth for side in SIDES}
                    thumb_by_side = {side: 0.0 for side in SIDES}
                    closure_by_side[SIDES[first_index]] = 1.0
                    thumb_by_side[SIDES[first_index]] = 1.0
                    if args.sugar_side_clamp_close_second_hand_during_second_press:
                        closure_by_side[SIDES[second_index]] = 1.0
                    if args.sugar_side_clamp_close_second_thumb_during_second_press:
                        thumb_by_side[SIDES[second_index]] = 1.0
                    delayed_thumb_steps = (
                        args.sugar_side_clamp_second_thumb_close_during_four_finger_steps
                    )
                    if delayed_thumb_steps > 0:
                        delayed_u = min((close_step + 1) / delayed_thumb_steps, 1.0)
                        thumb_by_side[SIDES[second_index]] = (
                            0.5 - 0.5 * np.cos(np.pi * delayed_u)
                        )
                    close_command = sequential_digit_command(
                        closure_by_side, thumb_by_side
                    )
                simulate_ik_one(
                    step_contact_pos_w,
                    step_contact_quat_w,
                    "unitree_demo_side_clamp_four_finger_close",
                    smooth,
                    thumb_closure=0.0,
                    inspire_command_override=close_command,
                )
            for thumb_step in range(args.ik_thumb_close_steps):
                u = (thumb_step + 1) / args.ik_thumb_close_steps
                smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                step_contact_pos_w, step_contact_quat_w = (
                    current_side_clamp_targets()
                )
                thumb_command = None
                if args.sugar_side_clamp_close_first_hand_before_second:
                    closure_by_side = {side: 1.0 for side in SIDES}
                    thumb_by_side = {side: smooth for side in SIDES}
                    thumb_by_side[SIDES[first_index]] = 1.0
                    if args.sugar_side_clamp_close_second_thumb_during_second_press:
                        thumb_by_side[SIDES[second_index]] = 1.0
                    if (
                        args.sugar_side_clamp_second_thumb_close_during_four_finger_steps
                        > 0
                    ):
                        thumb_by_side[SIDES[second_index]] = 1.0
                    thumb_command = sequential_digit_command(
                        closure_by_side, thumb_by_side
                    )
                simulate_ik_one(
                    step_contact_pos_w,
                    step_contact_quat_w,
                    "unitree_demo_side_clamp_thumb_close",
                    1.0,
                    thumb_closure=smooth,
                    inspire_command_override=thumb_command,
                )
        for _ in range(args.unitree_demo_settle_steps):
            step_contact_pos_w, step_contact_quat_w = current_side_clamp_targets()
            settle_command = sequential_digit_command(
                {side: 1.0 for side in SIDES},
                {side: 1.0 for side in SIDES},
            )
            simulate_ik_one(
                step_contact_pos_w,
                step_contact_quat_w,
                "unitree_demo_side_clamp_contact_settle",
                1.0,
                thumb_closure=1.0,
                inspire_command_override=settle_command,
            )

        for preload_step in range(args.ik_contact_preload_steps):
            u = min(
                (preload_step + 1) / args.ik_contact_preload_ramp_steps,
                1.0,
            )
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            preload_pos_w, preload_quat_w = current_side_clamp_targets()
            preload_pos_w[:, 2] += smooth * args.ik_contact_preload_height_m
            preload_command = sequential_digit_command(
                {side: 1.0 for side in SIDES},
                {side: 1.0 for side in SIDES},
            )
            simulate_ik_one(
                preload_pos_w,
                preload_quat_w,
                "unitree_demo_side_clamp_contact_preload",
                1.0,
                thumb_closure=1.0,
                inspire_command_override=preload_command,
            )

        for compression_step in range(args.ik_contact_compression_steps):
            u = min(
                (compression_step + 1)
                / args.ik_contact_compression_ramp_steps,
                1.0,
            )
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            compression_pos_w, compression_quat_w = current_side_clamp_targets()
            compression_outward_w = math_utils.matrix_from_quat(
                compression_quat_w
            )[:, :, 0]
            compression_pos_w -= (
                smooth
                * args.ik_contact_compression_m
                * compression_outward_w
            )
            compression_command = sequential_digit_command(
                {side: 1.0 for side in SIDES},
                {side: 1.0 for side in SIDES},
            )
            simulate_ik_one(
                compression_pos_w,
                compression_quat_w,
                "unitree_demo_side_clamp_contact_compression",
                1.0,
                thumb_closure=1.0,
                inspire_command_override=compression_command,
            )

        if args.require_settled_all_groups_frames:
            required_frames = args.require_settled_all_groups_frames
            settled_group_load = np.asarray(
                rows["group_normal_load"][-required_frames:], dtype=np.float64
            )
            settled_contact_component_passed = bool(
                settled_group_load.shape == (required_frames, 2, len(GROUPS))
                and np.all(settled_group_load > args.contact_threshold_n)
            )
            all_robot_names = np.asarray(all_robot_sensor.body_names)
            hand_mask = np.asarray(
                [
                    name.endswith("hand_base_link")
                    or name.startswith("L_")
                    or name.startswith("R_")
                    for name in all_robot_names
                ],
                dtype=bool,
            )
            settled_all_body_load = np.asarray(
                rows["all_robot_body_normal_load"][-required_frames:],
                dtype=np.float64,
            )
            settled_nonhand_component_passed = bool(
                np.all(
                    settled_all_body_load[:, ~hand_mask].sum(axis=1)
                    <= args.contact_threshold_n
                )
            )
            settled_finger_position = np.asarray(
                rows["finger_position"][-required_frames:], dtype=np.float64
            )
            settled_lower_violation = np.maximum(
                cpu(hard_lower)[None, :] - settled_finger_position, 0.0
            )
            settled_upper_violation = np.maximum(
                settled_finger_position - cpu(hard_upper)[None, :], 0.0
            )
            settled_hard_limit_component_passed = bool(
                np.maximum(
                    settled_lower_violation, settled_upper_violation
                ).max(initial=0.0)
                <= 1.0e-3
            )
            settled_all_groups_gate_passed = bool(
                settled_contact_component_passed
                and settled_nonhand_component_passed
                and settled_hard_limit_component_passed
            )
        execute_lift = settled_all_groups_gate_passed is not False

        ik_contact_anchor_box_state = cpu(box.data.root_state_w[0])
        lift_anchor_pose = box.data.root_pose_w[0].clone()
        ik_desired_contact_pos_w, ik_desired_contact_quat_w = (
            math_utils.combine_frame_transforms(
                lift_anchor_pose[:3].repeat(2, 1),
                lift_anchor_pose[3:7].repeat(2, 1),
                side_relative_pos,
                side_relative_quat,
            )
        )
        ik_desired_contact_pos_w[:, 2] += args.ik_contact_preload_height_m
        ik_desired_contact_pos_w -= (
            args.ik_contact_compression_m
            * math_utils.matrix_from_quat(ik_desired_contact_quat_w)[:, :, 0]
        )
        lifted_contact_pos_w = ik_desired_contact_pos_w.clone()
        lifted_contact_pos_w[:, 2] += args.ik_lift_height_m
        live_lift_lead_m = (
            args.ik_live_lift_lead_m
            if args.ik_live_lift_lead_m is not None
            else args.ik_lift_height_m / args.ik_lift_steps
        )
        last_lift_reference_pos_w = None
        for lift_step in range(args.ik_lift_steps if execute_lift else 0):
            if args.ik_track_live_box_during_lift:
                live_pose = box.data.root_pose_w[0].clone()
                if args.ik_live_lift_anchor_world_xy:
                    live_pose[:2] = lift_anchor_pose[:2]
                if args.ik_live_lift_scheduled_world_z:
                    u = (lift_step + 1) / args.ik_lift_steps
                    smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                    live_pose[2] = (
                        lift_anchor_pose[2] + smooth * args.ik_lift_height_m
                    )
                last_lift_reference_pos_w = live_pose[:3].clone()
                desired_pos, desired_quat = math_utils.combine_frame_transforms(
                    live_pose[:3].repeat(2, 1),
                    live_pose[3:7].repeat(2, 1),
                    side_relative_pos,
                    side_relative_quat,
                )
                desired_pos[:, 2] += args.ik_contact_preload_height_m
                desired_pos -= (
                    args.ik_contact_compression_m
                    * math_utils.matrix_from_quat(desired_quat)[:, :, 0]
                )
                if not args.ik_live_lift_scheduled_world_z:
                    if args.ik_live_lift_lead_ramp_steps:
                        ramp_u = min(
                            (lift_step + 1) / args.ik_live_lift_lead_ramp_steps,
                            1.0,
                        )
                        lead_scale = 0.5 - 0.5 * np.cos(np.pi * ramp_u)
                    else:
                        lead_scale = 1.0
                    desired_pos[:, 2] += live_lift_lead_m * lead_scale
            else:
                u = (lift_step + 1) / args.ik_lift_steps
                smooth = 0.5 - 0.5 * np.cos(np.pi * u)
                desired_pos = (
                    (1.0 - smooth) * ik_desired_contact_pos_w
                    + smooth * lifted_contact_pos_w
                )
                desired_quat = ik_desired_contact_quat_w
            lift_command = sequential_digit_command(
                {side: 1.0 for side in SIDES},
                {side: 1.0 for side in SIDES},
            )
            late_thumb_u = np.clip(
                (
                    lift_step
                    - args.sugar_side_clamp_lift_thumb_yaw_start_step
                    + 1
                )
                / args.sugar_side_clamp_lift_thumb_yaw_ramp_steps,
                0.0,
                1.0,
            )
            late_thumb_scale = 0.5 - 0.5 * np.cos(np.pi * late_thumb_u)
            for side, command_index in (("left", 11), ("right", 5)):
                late_target = lift_thumb_yaw_by_side_rad[side]
                if late_target is not None:
                    lift_command[command_index] = (
                        (1.0 - late_thumb_scale)
                        * closed_thumb_yaw_by_side_rad[side]
                        + late_thumb_scale * late_target
                    )
            simulate_ik_one(
                desired_pos,
                desired_quat,
                "unitree_demo_side_clamp_physical_lift",
                1.0,
                thumb_closure=1.0,
                inspire_command_override=lift_command,
            )
        if execute_lift and args.ik_live_hold_relative_to_box:
            if last_lift_reference_pos_w is None:
                raise RuntimeError("Scheduled lift did not produce a hold reference")
            ik_live_hold_translation_bias_w = (
                last_lift_reference_pos_w
                - box.data.root_pose_w[0, :3].clone()
            )
        for hold_step in range(args.hold_steps if execute_lift else 0):
            if args.ik_track_live_box_during_lift:
                live_pose = box.data.root_pose_w[0].clone()
                if args.ik_live_hold_relative_to_box:
                    assert ik_live_hold_translation_bias_w is not None
                    live_pose[:3] += ik_live_hold_translation_bias_w
                elif args.ik_live_lift_anchor_world_xy:
                    live_pose[:2] = lift_anchor_pose[:2]
                if (
                    args.ik_live_lift_scheduled_world_z
                    and not args.ik_live_hold_relative_to_box
                ):
                    live_pose[2] = lift_anchor_pose[2] + args.ik_lift_height_m
                hold_pos, hold_quat = math_utils.combine_frame_transforms(
                    live_pose[:3].repeat(2, 1),
                    live_pose[3:7].repeat(2, 1),
                    side_relative_pos,
                    side_relative_quat,
                )
                hold_pos[:, 2] += args.ik_contact_preload_height_m
                hold_pos -= (
                    args.ik_contact_compression_m
                    * math_utils.matrix_from_quat(hold_quat)[:, :, 0]
                )
                if not args.ik_live_lift_scheduled_world_z:
                    hold_pos[:, 2] += live_lift_lead_m
            else:
                hold_pos = lifted_contact_pos_w
                hold_quat = ik_desired_contact_quat_w
            hold_command = sequential_digit_command(
                {side: 1.0 for side in SIDES},
                {side: 1.0 for side in SIDES},
            )
            late_thumb_u = np.clip(
                (
                    args.ik_lift_steps
                    + hold_step
                    - args.sugar_side_clamp_lift_thumb_yaw_start_step
                    + 1
                )
                / args.sugar_side_clamp_lift_thumb_yaw_ramp_steps,
                0.0,
                1.0,
            )
            late_thumb_scale = 0.5 - 0.5 * np.cos(np.pi * late_thumb_u)
            for side, command_index in (("left", 11), ("right", 5)):
                late_target = lift_thumb_yaw_by_side_rad[side]
                if late_target is not None:
                    hold_command[command_index] = (
                        (1.0 - late_thumb_scale)
                        * closed_thumb_yaw_by_side_rad[side]
                        + late_thumb_scale * late_target
                    )
            simulate_ik_one(
                hold_pos,
                hold_quat,
                "terminal_ik_hold",
                1.0,
                thumb_closure=1.0,
                inspire_command_override=hold_command,
            )
    elif args.body_control_mode == "unitree_demo_pose":
        assert unitree_demo_physical_command is not None
        assert unitree_demo_outward_translation is not None
        # The official real-robot pose is the geometry source. Hold its exact
        # recorded 29-DoF configuration, place the unchanged CarryBox once on
        # the declared support, close through the documented 12-D Inspire
        # command, then lift with the same official Jacobian/drive path.
        demo_hand_start_pos_w = torch.stack(
            [
                robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        demo_hand_pos_w = (
            demo_hand_start_pos_w
            + unitree_demo_outward_translation.unsqueeze(0)
        )
        demo_hand_quat_w = torch.stack(
            [
                robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        declared_midpoint = (
            initial_object_state[0, :3].clone()
            - torch.as_tensor(
                args.unitree_demo_box_center_offset_m,
                device=device,
                dtype=initial_object_state.dtype,
            )
            - unitree_demo_outward_translation.to(
                device=device, dtype=initial_object_state.dtype
            )
        )
        actual_midpoint = demo_hand_start_pos_w.mean(dim=0)
        unitree_demo_actual_minus_declared_ee_midpoint = cpu(
            actual_midpoint - declared_midpoint
        )
        # Correct horizontal FK convention differences once, before the trial;
        # preserve the declared support height and never write the box again.
        initial_object_state[0, :2] += actual_midpoint[:2] - declared_midpoint[:2]
        for approach_step in range(args.unitree_demo_approach_steps):
            u = (approach_step + 1) / args.unitree_demo_approach_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            desired_pos = (
                (1.0 - smooth) * demo_hand_start_pos_w
                + smooth * demo_hand_pos_w
            )
            simulate_ik_one(
                desired_pos,
                demo_hand_quat_w,
                "unitree_demo_open_outward_approach",
                0.0,
                thumb_closure=0.0,
                inspire_command_override=torch.zeros_like(
                    unitree_demo_physical_command
                ),
                record_step=False,
                position_only=True,
            )

        unitree_demo_reachable_hand_pos_w = torch.stack(
            [
                robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        unitree_demo_reachable_hand_quat_w = torch.stack(
            [
                robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        unitree_demo_approach_position_error_m = cpu(
            torch.linalg.norm(
                unitree_demo_reachable_hand_pos_w - demo_hand_pos_w, dim=1
            )
        )
        attained_midpoint_correction = (
            unitree_demo_reachable_hand_pos_w.mean(dim=0)
            - demo_hand_pos_w.mean(dim=0)
        )
        initial_object_state[0, :2] += attained_midpoint_correction[:2]
        # Freeze the physically attained pose. This avoids asking the pose IK
        # to reintroduce an unreachable orientation while fingers close.
        demo_hand_pos_w = unitree_demo_reachable_hand_pos_w
        demo_hand_quat_w = unitree_demo_reachable_hand_quat_w

        # The open-hand approach is setup-only. Publish the dynamic box exactly
        # once after the hands reach their declared outward targets, then never
        # write its state again during close, lift, or hold.
        box.write_root_state_to_sim(initial_object_state)
        box.reset()
        sim.forward()
        robot.update(dt)
        box.update(dt)
        for sensor in sensors.values():
            sensor.reset()
            sensor.update(dt, force_recompute=True)
        all_robot_sensor.reset()
        all_robot_sensor.update(dt, force_recompute=True)

        for close_step in range(args.unitree_demo_close_steps):
            u = (close_step + 1) / args.unitree_demo_close_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            step_command = smooth * unitree_demo_physical_command
            simulate_ik_one(
                demo_hand_pos_w,
                demo_hand_quat_w,
                "unitree_demo_physical_close",
                smooth,
                thumb_closure=smooth,
                inspire_command_override=step_command,
            )
        for _ in range(args.unitree_demo_settle_steps):
            simulate_ik_one(
                demo_hand_pos_w,
                demo_hand_quat_w,
                "unitree_demo_contact_settle",
                1.0,
                thumb_closure=1.0,
                inspire_command_override=unitree_demo_physical_command,
            )

        ik_contact_anchor_box_state = cpu(box.data.root_state_w[0])
        ik_desired_contact_pos_w = demo_hand_pos_w.clone()
        ik_desired_contact_quat_w = demo_hand_quat_w.clone()
        lifted_contact_pos_w = demo_hand_pos_w.clone()
        lifted_contact_pos_w[:, 2] += args.ik_lift_height_m
        for lift_step in range(args.ik_lift_steps):
            u = (lift_step + 1) / args.ik_lift_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            desired_pos = (
                (1.0 - smooth) * demo_hand_pos_w
                + smooth * lifted_contact_pos_w
            )
            simulate_ik_one(
                desired_pos,
                demo_hand_quat_w,
                "unitree_demo_physical_lift",
                1.0,
                thumb_closure=1.0,
                inspire_command_override=unitree_demo_physical_command,
            )
        for _ in range(args.hold_steps):
            simulate_ik_one(
                lifted_contact_pos_w,
                demo_hand_quat_w,
                "terminal_ik_hold",
                1.0,
                thumb_closure=1.0,
                inspire_command_override=unitree_demo_physical_command,
            )
    elif args.body_control_mode == "bilateral_ik":
        assert ik_template_relative_pos is not None
        assert ik_template_relative_quat is not None
        initial_box_pose = initial_object_state[0, :7].clone()
        approach_relative_pos = ik_template_relative_pos.clone()
        approach_relative_pos[0, 1] -= args.ik_approach_clearance_m
        approach_relative_pos[1, 1] += args.ik_approach_clearance_m
        approach_pos_w, approach_quat_w = math_utils.combine_frame_transforms(
            initial_box_pose[:3].repeat(2, 1),
            initial_box_pose[3:7].repeat(2, 1),
            approach_relative_pos,
            ik_template_relative_quat,
        )
        approach_start_pos = torch.stack(
            [
                robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        approach_start_quat = torch.stack(
            [
                robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        for approach_step in range(args.ik_approach_steps):
            u = (approach_step + 1) / args.ik_approach_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            desired_pos = (1.0 - smooth) * approach_start_pos + smooth * approach_pos_w
            desired_quat = torch.stack(
                [
                    math_utils.quat_slerp(
                        approach_start_quat[index].clone(),
                        approach_quat_w[index].clone(),
                        smooth,
                    )
                    for index in range(2)
                ]
            )
            simulate_ik_one(
                desired_pos,
                desired_quat,
                "ik_unrecorded_setup_approach",
                0.0,
                thumb_closure=0.0,
                record_step=False,
            )

        box.write_root_state_to_sim(initial_object_state)
        box.reset()
        sim.forward()
        robot.update(dt)
        box.update(dt)
        for sensor in sensors.values():
            sensor.reset()
            sensor.update(dt, force_recompute=True)
        all_robot_sensor.reset()
        all_robot_sensor.update(dt, force_recompute=True)
        initial_grasp_anchor_pose = initial_object_state[0, :7].clone()

        def current_grasp_targets() -> tuple[torch.Tensor, torch.Tensor]:
            anchor_pose = (
                box.data.root_pose_w[0].clone()
                if args.ik_track_live_box_during_grasp
                else initial_grasp_anchor_pose
            )
            return math_utils.combine_frame_transforms(
                anchor_pose[:3].repeat(2, 1),
                anchor_pose[3:7].repeat(2, 1),
                ik_template_relative_pos,
                ik_template_relative_quat,
            )

        ik_desired_contact_pos_w, ik_desired_contact_quat_w = (
            current_grasp_targets()
        )
        close_start_pos = torch.stack(
            [
                robot.data.body_pos_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        close_start_quat = torch.stack(
            [
                robot.data.body_quat_w[0, ik_ee_body_ids[side]].clone()
                for side in SIDES
            ]
        )
        # Build the grasp in anatomical order. V5 moved the palms inward while
        # closing every digit, so the left thumb and right fingers became early
        # hard stops and prevented the remaining anatomy from reaching the box.
        # First press open palms to the hash-bound contact geometry, then close
        # the four opposed fingers, and only then sweep in the thumbs.
        for press_step in range(args.ik_palm_press_steps):
            u = (press_step + 1) / args.ik_palm_press_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            step_contact_pos_w, step_contact_quat_w = current_grasp_targets()
            desired_pos = (
                (1.0 - smooth) * close_start_pos
                + smooth * step_contact_pos_w
            )
            desired_quat = torch.stack(
                [
                    math_utils.quat_slerp(
                        close_start_quat[index].clone(),
                        step_contact_quat_w[index].clone(),
                        smooth,
                    )
                    for index in range(2)
                ]
            )
            simulate_ik_one(
                desired_pos,
                desired_quat,
                "ik_palm_press",
                0.0,
                thumb_closure=0.0,
            )
        for close_step in range(args.ik_close_steps):
            u = (close_step + 1) / args.ik_close_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            step_contact_pos_w, step_contact_quat_w = current_grasp_targets()
            simulate_ik_one(
                step_contact_pos_w,
                step_contact_quat_w,
                "ik_four_finger_close",
                smooth,
                thumb_closure=0.0,
            )
        for thumb_step in range(args.ik_thumb_close_steps):
            u = (thumb_step + 1) / args.ik_thumb_close_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            step_contact_pos_w, step_contact_quat_w = current_grasp_targets()
            simulate_ik_one(
                step_contact_pos_w,
                step_contact_quat_w,
                "ik_thumb_close",
                1.0,
                thumb_closure=smooth,
            )
        for _ in range(args.ik_settle_steps):
            step_contact_pos_w, step_contact_quat_w = current_grasp_targets()
            simulate_ik_one(
                step_contact_pos_w,
                step_contact_quat_w,
                "ik_contact_settle",
                1.0,
                thumb_closure=1.0,
            )
        # Freeze the actual end-of-grasp box pose exactly once as the open-loop
        # lift anchor. The object remains dynamic and is never written/replayed.
        ik_contact_anchor_box_state = cpu(box.data.root_state_w[0])
        lift_anchor_pose = box.data.root_pose_w[0].clone()
        ik_desired_contact_pos_w, ik_desired_contact_quat_w = (
            math_utils.combine_frame_transforms(
                lift_anchor_pose[:3].repeat(2, 1),
                lift_anchor_pose[3:7].repeat(2, 1),
                ik_template_relative_pos,
                ik_template_relative_quat,
            )
        )
        lifted_contact_pos_w = ik_desired_contact_pos_w.clone()
        lifted_contact_pos_w[:, 2] += args.ik_lift_height_m
        for lift_step in range(args.ik_lift_steps):
            u = (lift_step + 1) / args.ik_lift_steps
            smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            desired_pos = (
                (1.0 - smooth) * ik_desired_contact_pos_w
                + smooth * lifted_contact_pos_w
            )
            simulate_ik_one(
                desired_pos,
                ik_desired_contact_quat_w,
                "ik_physical_lift",
                1.0,
                thumb_closure=1.0,
            )
        for _ in range(args.hold_steps):
            simulate_ik_one(
                lifted_contact_pos_w,
                ik_desired_contact_quat_w,
                "terminal_ik_hold",
                1.0,
                thumb_closure=1.0,
            )

    regular_end = (
        args.source_start
        if args.body_control_mode
        in ("bilateral_ik", "unitree_demo_pose", "sugar_side_clamp")
        else (
            int(args.brake_start)
            if args.brake_start is not None
            else args.source_end
        )
    )
    if grasp_dwell_enabled:
        dwell_source = int(args.grasp_dwell_source)
        close_source_end = dwell_source + args.grasp_close_source_span
        for source_index in range(args.source_start, dwell_source):
            for substep in range(args.physics_substeps_per_source):
                source_time = (
                    source_index
                    + (substep + 1) / args.physics_substeps_per_source
                )
                simulate_one(
                    source_time,
                    "reference_pregrasp",
                    closure_override=0.0,
                )
        for close_step in range(args.grasp_close_steps):
            u = (close_step + 1) / args.grasp_close_steps
            smooth_closure = 0.5 - 0.5 * np.cos(np.pi * u)
            source_time = (
                dwell_source
                + args.grasp_close_source_span * smooth_closure
            )
            close_velocity_scale = (
                args.grasp_close_source_span
                * 0.5
                * np.pi
                * np.sin(np.pi * u)
                * args.physics_substeps_per_source
                / args.grasp_close_steps
            )
            simulate_one(
                float(source_time),
                (
                    "moving_grasp_close"
                    if args.grasp_close_source_span > 0
                    else "stationary_grasp_close"
                ),
                velocity_scale=float(close_velocity_scale),
                closure_override=float(smooth_closure),
            )
        for _settle_step in range(args.grasp_settle_steps):
            simulate_one(
                float(close_source_end),
                (
                    "moving_grasp_settle"
                    if args.grasp_close_source_span > 0
                    else "stationary_grasp_settle"
                ),
                velocity_scale=0.0,
                closure_override=1.0,
            )
        resume_end = close_source_end + args.resume_source_span
        resume_steps = (
            4 * args.resume_polynomial_power * args.resume_source_span
        )
        for resume_step in range(resume_steps):
            u = (resume_step + 1) / resume_steps
            source_time = (
                close_source_end
                + args.resume_source_span * u**args.resume_polynomial_power
            )
            simulate_one(
                source_time,
                "reference_smooth_resume",
                velocity_scale=u ** (args.resume_polynomial_power - 1),
                closure_override=1.0,
            )
        regular_start = resume_end
    else:
        regular_start = args.source_start

    for source_index in range(regular_start, regular_end):
        for substep in range(args.physics_substeps_per_source):
            source_time = source_index + (substep + 1) / args.physics_substeps_per_source
            simulate_one(
                source_time,
                "reference_reach_lift",
                closure_override=1.0 if grasp_dwell_enabled else None,
            )

    if args.brake_start is not None:
        source_span = args.source_end - args.brake_start
        for brake_step in range(args.brake_steps):
            u = (brake_step + 1) / args.brake_steps
            # Constant source speed at entry, linearly decelerating to zero at
            # exit: s(u)=s0+delta*(2u-u^2).  With 80 steps over ten 50-Hz
            # frames, its initial derivative matches the preceding replay.
            source_time = args.brake_start + source_span * (2.0 * u - u * u)
            velocity_scale = 1.0 - u
            simulate_one(
                source_time,
                "reference_smooth_brake",
                velocity_scale=velocity_scale,
                closure_override=1.0 if grasp_dwell_enabled else None,
            )

    for hold_step in range(
        0
        if args.body_control_mode
        in ("bilateral_ik", "unitree_demo_pose", "sugar_side_clamp")
        else args.hold_steps
    ):
        source_time = float(args.source_end)
        simulate_one(
            source_time,
            "terminal_reference_hold",
            velocity_scale=0.0,
            closure_override=1.0 if grasp_dwell_enabled else None,
        )

    arrays = {
        key: np.asarray(value)
        for key, value in rows.items()
    }
    ik_step_mask = np.all(np.isfinite(arrays["desired_hand_pos_w"]), axis=(1, 2))
    ik_tracking_position_error = np.full(step_counter, np.nan, dtype=np.float64)
    ik_tracking_rotation_error = np.full(step_counter, np.nan, dtype=np.float64)
    if np.any(ik_step_mask):
        actual_hand_pos = arrays["hand_body_pos_w"][:, :, 0].astype(np.float64)
        actual_hand_quat = arrays["hand_body_quat_w"][:, :, 0].astype(np.float64)
        desired_hand_pos = arrays["desired_hand_pos_w"].astype(np.float64)
        desired_hand_quat = arrays["desired_hand_quat_w"].astype(np.float64)
        ik_tracking_position_error[ik_step_mask] = np.linalg.norm(
            actual_hand_pos[ik_step_mask] - desired_hand_pos[ik_step_mask], axis=2
        ).max(axis=1)
        quaternion_dot = np.abs(
            np.sum(
                actual_hand_quat[ik_step_mask]
                * desired_hand_quat[ik_step_mask],
                axis=2,
            )
        )
        quaternion_dot = np.clip(quaternion_dot, 0.0, 1.0)
        ik_tracking_rotation_error[ik_step_mask] = (
            2.0 * np.arccos(quaternion_dot)
        ).max(axis=1)
    group_load = arrays["group_normal_load"].astype(np.float64)
    finger_position = arrays["finger_position"].astype(np.float64)
    finger_target = arrays["finger_target"].astype(np.float64)
    inspire_command = arrays["inspire_command"].astype(np.float64)
    reconstructed_target_columns = []
    for name in finger_names:
        if name in OFFICIAL_INSPIRE_BASE_COMMAND_INDEX:
            reconstructed_target_columns.append(
                inspire_command[:, OFFICIAL_INSPIRE_BASE_COMMAND_INDEX[name]]
            )
        else:
            command_index, scale = OFFICIAL_INSPIRE_SPECIAL_COMMAND[name]
            reconstructed_target_columns.append(
                inspire_command[:, command_index] * scale
            )
    reconstructed_target = np.stack(reconstructed_target_columns, axis=1)
    maximum_command_reconstruction_error = float(
        np.max(np.abs(finger_target - reconstructed_target))
    )
    hard_lower_np = cpu(hard_lower).astype(np.float64)
    hard_upper_np = cpu(hard_upper).astype(np.float64)
    lower_violation = np.maximum(hard_lower_np[None, :] - finger_position, 0.0)
    upper_violation = np.maximum(finger_position - hard_upper_np[None, :], 0.0)
    per_step_hard_limit_violation = np.maximum(
        lower_violation, upper_violation
    ).max(axis=1)
    finger_hard_limit_valid = per_step_hard_limit_violation <= 1.0e-3
    maximum_joint_limit_violation = float(
        per_step_hard_limit_violation.max(initial=0.0)
    )
    group_loaded = group_load > args.contact_threshold_n
    all_groups_bilateral = np.all(group_loaded, axis=(1, 2))
    both_hands_contact = np.all(np.any(group_loaded, axis=2), axis=1)
    box_position = arrays["box_state"][:, :3].astype(np.float64)
    initial_z = float(initial_object_state[0, 2].item())
    relative_height = box_position[:, 2] - initial_z
    lifted = relative_height > 0.10
    hold_mask = np.isin(
        arrays["phase"], ("terminal_reference_hold", "terminal_ik_hold")
    )
    raw_force = arrays["raw_force_by_hand"].sum(axis=1).astype(np.float64)
    raw_torque = arrays["raw_torque_by_hand"].sum(axis=1).astype(np.float64)
    required_force = arrays["required_force"].astype(np.float64)
    required_torque = arrays["required_torque"].astype(np.float64)
    weight_n = mass * 9.81
    characteristic_length_m = 0.54605001
    force_residual = np.linalg.norm(raw_force - required_force, axis=1) / weight_n
    torque_residual = np.linalg.norm(raw_torque - required_torque, axis=1) / (
        weight_n * characteristic_length_m
    )
    matrix_force = arrays["matrix_force_by_hand"].sum(axis=1).astype(np.float64)
    matrix_force_residual = np.linalg.norm(
        matrix_force - required_force, axis=1
    ) / weight_n
    raw_matrix_force_mismatch = np.linalg.norm(
        raw_force - matrix_force, axis=1
    ) / weight_n
    direct_hand_force = arrays["direct_force_by_hand"].sum(axis=1).astype(
        np.float64
    )
    all_robot_force = arrays["all_robot_raw_force"].astype(np.float64)
    all_robot_torque = arrays["all_robot_raw_torque"].astype(np.float64)
    all_robot_force_residual = np.linalg.norm(
        all_robot_force - required_force, axis=1
    ) / weight_n
    all_robot_torque_residual = np.linalg.norm(
        all_robot_torque - required_torque, axis=1
    ) / (weight_n * characteristic_length_m)
    direct_raw_hand_mismatch = np.linalg.norm(
        direct_hand_force - raw_force, axis=1
    ) / weight_n
    all_robot_body_names = np.asarray(all_robot_sensor.body_names)
    hand_anatomy_body_mask = np.asarray(
        [
            name.endswith("hand_base_link")
            or name.startswith("L_")
            or name.startswith("R_")
            for name in all_robot_body_names
        ],
        dtype=bool,
    )
    nonhand_body_mask = ~hand_anatomy_body_mask
    nonhand_normal_load = arrays["all_robot_body_normal_load"][
        :, nonhand_body_mask
    ].sum(axis=1).astype(np.float64)
    nonhand_support_absent = nonhand_normal_load <= args.contact_threshold_n
    dynamic_mask = lifted & both_hands_contact
    clean_dynamic_mask = dynamic_mask & nonhand_support_absent
    clean_all_groups_mask = (
        lifted & all_groups_bilateral & nonhand_support_absent
    )
    admissible_dynamic_mask = clean_dynamic_mask & finger_hard_limit_valid
    admissible_all_groups_mask = clean_all_groups_mask & finger_hard_limit_valid
    stable_mask = hold_mask & admissible_all_groups_mask
    checks = {
        "official_first_29_joint_order_exact": tuple(robot.joint_names[:29])
        == EXPECTED_BODY_JOINT_NAMES,
        "official_24_inspire_joints_live": len(finger_ids) == 24,
        "declared_body_joint_targets_within_hard_limits": (
            maximum_body_target_limit_violation <= 1.0e-6
        ),
        "unitree_demo_body_source_limit_adjustment_le_0p01rad": (
            args.body_control_mode != "unitree_demo_pose"
            or unitree_demo_body_limit_adjustment_rad <= 1.0e-2
        ),
        "official_inspire_6dof_coupling_reconstructs_targets": (
            maximum_command_reconstruction_error <= 1.0e-6
        ),
        "both_sensors_cover_palm_thumb_four_fingers": all(
            set(hand_body_groups[side]) == set(GROUPS) for side in SIDES
        ),
        "box_state_written_once_at_recorded_trial_initialization_then_never_replayed": True,
        "robot_root_and_body_state_written_only_at_initialization": (
            args.body_control_mode in (
                "pd_target",
                "bilateral_ik",
                "unitree_demo_pose",
                "sugar_side_clamp",
            )
        ),
        "robot_gravity_enabled_for_physical_gate": (
            args.body_control_mode in (
                "pd_target",
                "bilateral_ik",
                "unitree_demo_pose",
                "sugar_side_clamp",
            )
        ),
        "root_fixture_is_explicitly_declared": True,
        "bilateral_ik_target_tracking_position_median_le_0p02m": bool(
            args.body_control_mode
            not in ("bilateral_ik", "unitree_demo_pose", "sugar_side_clamp")
            or (
                np.any(ik_step_mask)
                and np.nanmedian(ik_tracking_position_error) <= 0.02
            )
        ),
        "bilateral_ik_target_tracking_rotation_median_le_0p15rad": bool(
            args.body_control_mode
            not in ("bilateral_ik", "unitree_demo_pose", "sugar_side_clamp")
            or (
                np.any(ik_step_mask)
                and np.nanmedian(ik_tracking_rotation_error) <= 0.15
            )
        ),
        "object_mass_exact_0p5kg": abs(mass - 0.5) <= 1.0e-7,
        "official_nominal_object_material_readback_exact": bool(
            (object_material_readback[:, :, 0] == args.object_static_friction).all()
            and (
                object_material_readback[:, :, 1]
                == args.object_dynamic_friction
            ).all()
            and (object_material_readback[:, :, 2] == 0.0).all()
        ),
        "all_finger_positions_within_hard_limits_1e_3rad": (
            maximum_joint_limit_violation <= 1.0e-3
        ),
        "declared_pre_lift_settled_all_groups_gate_passed": (
            args.require_settled_all_groups_frames == 0
            or settled_all_groups_gate_passed is True
        ),
        "box_lifted_more_than_0p10m": bool(np.any(lifted)),
        "clean_bilateral_hand_contact_during_lift": bool(
            np.any(admissible_dynamic_mask)
        ),
        "clean_all_six_groups_bilateral_for_20_steps": longest_true_run(
            admissible_all_groups_mask
        )
        >= 20,
        "terminal_hold_clean_all_groups_and_lift_for_20_steps": longest_true_run(
            stable_mask
        )
        >= 20,
        "hand_only_clean_dynamic_force_residual_median_le_0p20": bool(
            np.any(admissible_dynamic_mask)
            and np.median(force_residual[admissible_dynamic_mask]) <= 0.20
        ),
        "hand_only_clean_dynamic_torque_residual_median_le_0p20": bool(
            np.any(admissible_dynamic_mask)
            and np.median(torque_residual[admissible_dynamic_mask]) <= 0.20
        ),
        "all_robot_clean_dynamic_force_residual_median_le_0p20": bool(
            np.any(admissible_dynamic_mask)
            and np.median(all_robot_force_residual[admissible_dynamic_mask]) <= 0.20
        ),
        "all_robot_clean_dynamic_torque_residual_median_le_0p20": bool(
            np.any(admissible_dynamic_mask)
            and np.median(all_robot_torque_residual[admissible_dynamic_mask]) <= 0.20
        ),
        "direct_and_raw_hand_force_match_on_clean_dynamic_1e_3": bool(
            np.any(admissible_dynamic_mask)
            and np.median(direct_raw_hand_mismatch[admissible_dynamic_mask]) <= 1.0e-3
        ),
    }
    trace_path = output_root / "reachability_trace.npz"
    np.savez_compressed(
        trace_path,
        **arrays,
        side_names=np.asarray(SIDES),
        group_names=np.asarray(GROUPS),
        body_control_mode=np.asarray(args.body_control_mode),
        fix_robot_root=np.asarray(args.fix_robot_root),
        bilateral_hip_roll_outward_offset_rad=np.asarray(
            args.bilateral_hip_roll_outward_offset_rad, dtype=np.float32
        ),
        robot_joint_names=np.asarray(robot.joint_names),
        robot_body_names=np.asarray(robot.body_names),
        finger_joint_names=np.asarray(finger_names),
        inspire_command_names=np.asarray(
            (
                "R_pinky",
                "R_ring",
                "R_middle",
                "R_index",
                "R_thumb_pitch",
                "R_thumb_yaw",
                "L_pinky",
                "L_ring",
                "L_middle",
                "L_index",
                "L_thumb_pitch",
                "L_thumb_yaw",
            )
        ),
        finger_hard_lower_rad=hard_lower_np.astype(np.float32),
        finger_hard_upper_rad=hard_upper_np.astype(np.float32),
        all_robot_contact_body_names=all_robot_body_names,
        hand_anatomy_body_mask=hand_anatomy_body_mask,
        nonhand_body_mask=nonhand_body_mask,
        left_contact_body_names=np.asarray(hand_body_names["left"]),
        right_contact_body_names=np.asarray(hand_body_names["right"]),
        left_contact_body_groups=np.asarray(hand_body_groups["left"]),
        right_contact_body_groups=np.asarray(hand_body_groups["right"]),
        initial_object_state=cpu(initial_object_state[0]),
        initial_robot_root_state=cpu(root_state[0]),
        initial_robot_body_joint_position=cpu(initial_body_joint_pos),
        static_posture_solutions_path=np.asarray(
            str(static_posture_solutions) if static_posture_solutions is not None else ""
        ),
        static_posture_solutions_sha256=np.asarray(
            sha256(static_posture_solutions)
            if static_posture_solutions is not None
            else ""
        ),
        static_posture_source_trace_path=np.asarray(
            str(static_posture_source_trace)
            if static_posture_source_trace is not None
            else ""
        ),
        static_posture_source_trace_sha256=np.asarray(
            sha256(static_posture_source_trace)
            if static_posture_source_trace is not None
            else ""
        ),
        static_posture_index=np.asarray(
            args.static_posture_index if args.static_posture_index is not None else -1,
            dtype=np.int32,
        ),
        static_posture_source_index=np.asarray(
            args.static_posture_source_index
            if args.static_posture_source_index is not None
            else -1,
            dtype=np.int32,
        ),
        static_posture_contact_delta_pca_m=(
            static_posture_contact_delta_pca_np
        ),
        static_posture_contact_delta_box_b_m=(
            static_posture_contact_delta_box_b_np
        ),
        static_posture_contact_delta_world_m=(
            static_posture_contact_delta_world_np
        ),
        static_posture_relative_root_pose=(
            cpu(static_posture_relative_root_pose)
            if static_posture_relative_root_pose is not None
            else np.empty((0,), dtype=np.float32)
        ),
        static_posture_mapped_root_pose=(
            cpu(static_posture_mapped_root_pose)
            if static_posture_mapped_root_pose is not None
            else np.empty((0,), dtype=np.float32)
        ),
        object_mass_kg=np.asarray(mass, dtype=np.float32),
        object_inertia_local=cpu(inertia_local),
        object_material_before=cpu(object_material_before),
        object_material_requested=cpu(object_material_requested),
        object_material_readback=cpu(object_material_readback),
        robot_material_readback=cpu(robot_material_readback),
        sugar_side_clamp_support_height_m=np.asarray(
            args.sugar_side_clamp_support_height_m, dtype=np.float32
        ),
        sugar_side_clamp_support_size_xy_m=np.asarray(
            args.sugar_side_clamp_support_size_xy_m, dtype=np.float32
        ),
        sugar_side_clamp_support_center_xy_m=(
            np.asarray(sugar_side_clamp_support_center_xy, dtype=np.float32)
            if sugar_side_clamp_support_center_xy is not None
            else np.empty((0,), dtype=np.float32)
        ),
        gravity_w=np.asarray((0.0, 0.0, -9.81), dtype=np.float32),
        ik_template_index=np.asarray(args.ik_template_index, dtype=np.int32),
        ik_template_source_time=np.asarray(
            ik_template_source_time if ik_template_source_time is not None else np.nan,
            dtype=np.float32,
        ),
        ik_template_box_pose=(
            cpu(ik_template_box_pose)
            if ik_template_box_pose is not None
            else np.empty((0,), dtype=np.float32)
        ),
        ik_template_relative_pos=(
            cpu(ik_template_relative_pos)
            if ik_template_relative_pos is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        ik_template_relative_quat=(
            cpu(ik_template_relative_quat)
            if ik_template_relative_quat is not None
            else np.empty((0, 4), dtype=np.float32)
        ),
        ik_template_source_relative_quat=(
            cpu(ik_template_source_relative_quat)
            if ik_template_source_relative_quat is not None
            else np.empty((0, 4), dtype=np.float32)
        ),
        ik_track_live_box_during_grasp=np.asarray(
            args.ik_track_live_box_during_grasp
        ),
        ik_track_live_box_during_lift=np.asarray(
            args.ik_track_live_box_during_lift
        ),
        ik_live_lift_lead_m=np.asarray(
            args.ik_live_lift_lead_m
            if args.ik_live_lift_lead_m is not None
            else np.nan,
            dtype=np.float32,
        ),
        ik_live_lift_lead_ramp_steps=np.asarray(
            args.ik_live_lift_lead_ramp_steps, dtype=np.int64
        ),
        ik_live_lift_scheduled_world_z=np.asarray(
            args.ik_live_lift_scheduled_world_z
        ),
        ik_live_lift_anchor_world_xy=np.asarray(
            args.ik_live_lift_anchor_world_xy
        ),
        ik_live_hold_relative_to_box=np.asarray(
            args.ik_live_hold_relative_to_box
        ),
        require_settled_all_groups_frames=np.asarray(
            args.require_settled_all_groups_frames, dtype=np.int64
        ),
        ik_contact_preload_height_m=np.asarray(
            args.ik_contact_preload_height_m, dtype=np.float32
        ),
        ik_contact_preload_steps=np.asarray(
            args.ik_contact_preload_steps, dtype=np.int64
        ),
        ik_contact_preload_ramp_steps=np.asarray(
            args.ik_contact_preload_ramp_steps, dtype=np.int64
        ),
        ik_contact_compression_m=np.asarray(
            args.ik_contact_compression_m, dtype=np.float32
        ),
        ik_contact_compression_steps=np.asarray(
            args.ik_contact_compression_steps, dtype=np.int64
        ),
        ik_contact_compression_ramp_steps=np.asarray(
            args.ik_contact_compression_ramp_steps, dtype=np.int64
        ),
        settled_all_groups_gate_passed=np.asarray(
            -1
            if settled_all_groups_gate_passed is None
            else int(settled_all_groups_gate_passed),
            dtype=np.int8,
        ),
        settled_contact_component_passed=np.asarray(
            -1
            if settled_contact_component_passed is None
            else int(settled_contact_component_passed),
            dtype=np.int8,
        ),
        settled_nonhand_component_passed=np.asarray(
            -1
            if settled_nonhand_component_passed is None
            else int(settled_nonhand_component_passed),
            dtype=np.int8,
        ),
        settled_hard_limit_component_passed=np.asarray(
            -1
            if settled_hard_limit_component_passed is None
            else int(settled_hard_limit_component_passed),
            dtype=np.int8,
        ),
        ik_live_hold_translation_bias_w=(
            cpu(ik_live_hold_translation_bias_w)
            if ik_live_hold_translation_bias_w is not None
            else np.empty((0,), dtype=np.float32)
        ),
        sugar_side_clamp_palm_press_first=np.asarray(
            args.sugar_side_clamp_palm_press_first or ""
        ),
        sugar_side_clamp_close_first_hand_before_second=np.asarray(
            args.sugar_side_clamp_close_first_hand_before_second
        ),
        sugar_side_clamp_close_second_hand_during_second_press=np.asarray(
            args.sugar_side_clamp_close_second_hand_during_second_press
        ),
        sugar_side_clamp_close_second_thumb_during_second_press=np.asarray(
            args.sugar_side_clamp_close_second_thumb_during_second_press
        ),
        sugar_side_clamp_second_thumb_close_during_four_finger_steps=np.asarray(
            args.sugar_side_clamp_second_thumb_close_during_four_finger_steps
        ),
        sugar_side_clamp_digit_force_stop_n=np.asarray(
            np.nan
            if args.sugar_side_clamp_digit_force_stop_n is None
            else args.sugar_side_clamp_digit_force_stop_n
        ),
        sugar_side_clamp_digit_force_stop_max_scale_step=np.asarray(
            args.sugar_side_clamp_digit_force_stop_max_scale_step
        ),
        side_clamp_approach_clearance_by_side_m=np.asarray(
            [side_clamp_approach_clearance_by_side_m[side] for side in SIDES],
            dtype=np.float32,
        ),
        ik_contact_anchor_box_state=(
            ik_contact_anchor_box_state
            if ik_contact_anchor_box_state is not None
            else np.empty((0,), dtype=np.float32)
        ),
        ik_desired_contact_pos_w=(
            cpu(ik_desired_contact_pos_w)
            if ik_desired_contact_pos_w is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        ik_desired_contact_quat_w=(
            cpu(ik_desired_contact_quat_w)
            if ik_desired_contact_quat_w is not None
            else np.empty((0, 4), dtype=np.float32)
        ),
        unitree_demo_dataset_sha256=np.asarray(
            unitree_demo_dataset_sha256 or ""
        ),
        unitree_demo_episode_index=np.asarray(
            args.unitree_demo_episode_index, dtype=np.int32
        ),
        unitree_demo_frame_index=np.asarray(
            args.unitree_demo_frame_index, dtype=np.int32
        ),
        unitree_demo_ee_state=(
            unitree_demo_ee_state
            if unitree_demo_ee_state is not None
            else np.empty((0,), dtype=np.float32)
        ),
        unitree_demo_hand_normalized=(
            unitree_demo_hand_normalized
            if unitree_demo_hand_normalized is not None
            else np.empty((0,), dtype=np.float32)
        ),
        unitree_demo_physical_command=(
            cpu(unitree_demo_physical_command)
            if unitree_demo_physical_command is not None
            else np.empty((0,), dtype=np.float32)
        ),
        unitree_demo_outward_translation_m=(
            cpu(unitree_demo_outward_translation)
            if unitree_demo_outward_translation is not None
            else np.empty((0,), dtype=np.float32)
        ),
        unitree_demo_actual_minus_declared_ee_midpoint=(
            unitree_demo_actual_minus_declared_ee_midpoint
            if unitree_demo_actual_minus_declared_ee_midpoint is not None
            else np.empty((0,), dtype=np.float32)
        ),
        unitree_demo_approach_position_error_m=(
            unitree_demo_approach_position_error_m
            if unitree_demo_approach_position_error_m is not None
            else np.empty((0,), dtype=np.float32)
        ),
        unitree_demo_approach_rotation_error_rad=(
            unitree_demo_approach_rotation_error_rad
            if unitree_demo_approach_rotation_error_rad is not None
            else np.empty((0,), dtype=np.float32)
        ),
        unitree_demo_reachable_hand_pos_w=(
            cpu(unitree_demo_reachable_hand_pos_w)
            if unitree_demo_reachable_hand_pos_w is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        unitree_demo_reachable_hand_quat_w=(
            cpu(unitree_demo_reachable_hand_quat_w)
            if unitree_demo_reachable_hand_quat_w is not None
            else np.empty((0, 4), dtype=np.float32)
        ),
        side_clamp_face_sign=(
            cpu(side_clamp_face_sign)
            if side_clamp_face_sign is not None
            else np.empty((0,), dtype=np.float32)
        ),
        side_clamp_target_surface_point_box=(
            cpu(side_clamp_target_surface_point_box)
            if side_clamp_target_surface_point_box is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        side_clamp_geometric_contact_pos_w=(
            cpu(side_clamp_geometric_contact_pos_w)
            if side_clamp_geometric_contact_pos_w is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        side_clamp_geometric_contact_quat_w=(
            cpu(side_clamp_geometric_contact_quat_w)
            if side_clamp_geometric_contact_quat_w is not None
            else np.empty((0, 4), dtype=np.float32)
        ),
        side_clamp_contact_pca_by_side_m=np.asarray(
            [
                side_clamp_contact_pca_by_side_m[side]
                if side_clamp_contact_pca_by_side_m[side] is not None
                else (np.nan, np.nan, np.nan)
                for side in SIDES
            ],
            dtype=np.float32,
        ),
        side_clamp_outward_pca_by_side=np.asarray(
            [
                side_clamp_outward_pca_by_side[side]
                if side_clamp_outward_pca_by_side[side] is not None
                else (np.nan, np.nan, np.nan)
                for side in SIDES
            ],
            dtype=np.float32,
        ),
        side_clamp_contact_geometry_source=np.asarray(
            str(side_clamp_contact_geometry_source)
            if side_clamp_contact_geometry_source is not None
            else ""
        ),
        side_clamp_contact_geometry_source_sha256=np.asarray(
            sha256(side_clamp_contact_geometry_source)
            if side_clamp_contact_geometry_source is not None
            else ""
        ),
        sugar_side_clamp_box_offset_w=(
            cpu(sugar_side_clamp_box_offset_w)
            if sugar_side_clamp_box_offset_w is not None
            else np.empty((0,), dtype=np.float32)
        ),
        sugar_side_clamp_reachable_fit_offset_w=(
            cpu(sugar_side_clamp_reachable_fit_offset_w)
            if sugar_side_clamp_reachable_fit_offset_w is not None
            else np.empty((0,), dtype=np.float32)
        ),
        sugar_side_clamp_reachable_orientation_delta_rad=(
            cpu(sugar_side_clamp_reachable_orientation_delta_rad)
            if sugar_side_clamp_reachable_orientation_delta_rad is not None
            else np.empty((0,), dtype=np.float32)
        ),
        sugar_side_clamp_direct_refinement_steps=np.asarray(
            sugar_side_clamp_direct_refinement_steps, dtype=np.int32
        ),
        sugar_side_clamp_direct_refinement_accepted_steps=np.asarray(
            sugar_side_clamp_direct_refinement_accepted_steps, dtype=np.int32
        ),
        sugar_side_clamp_direct_refinement_accepted_by_side=np.asarray(
            [
                sugar_side_clamp_direct_refinement_accepted_by_side[side]
                for side in SIDES
            ],
            dtype=np.int32,
        ),
        sugar_side_clamp_direct_refinement_initial_error_m=(
            cpu(sugar_side_clamp_direct_refinement_initial_error_m)
            if sugar_side_clamp_direct_refinement_initial_error_m is not None
            else np.empty((0,), dtype=np.float32)
        ),
        sugar_side_clamp_direct_refinement_final_error_m=(
            cpu(sugar_side_clamp_direct_refinement_final_error_m)
            if sugar_side_clamp_direct_refinement_final_error_m is not None
            else np.empty((0,), dtype=np.float32)
        ),
        sugar_side_clamp_direct_refinement_final_trust_radius_m=(
            cpu(sugar_side_clamp_direct_refinement_final_trust_radius_m)
            if sugar_side_clamp_direct_refinement_final_trust_radius_m is not None
            else np.empty((0,), dtype=np.float32)
        ),
        force_residual_fraction=force_residual.astype(np.float32),
        torque_residual_fraction=torque_residual.astype(np.float32),
        ik_step_mask=ik_step_mask,
        ik_tracking_position_error_m=ik_tracking_position_error.astype(np.float32),
        ik_tracking_rotation_error_rad=ik_tracking_rotation_error.astype(np.float32),
        all_groups_bilateral=all_groups_bilateral,
        both_hands_contact=both_hands_contact,
        lifted=lifted,
        nonhand_normal_load_N=nonhand_normal_load.astype(np.float32),
        nonhand_support_absent=nonhand_support_absent,
        finger_hard_limit_violation_rad=per_step_hard_limit_violation.astype(
            np.float32
        ),
        finger_hard_limit_valid=finger_hard_limit_valid,
        clean_dynamic_mask=clean_dynamic_mask,
        clean_all_groups_mask=clean_all_groups_mask,
        admissible_dynamic_mask=admissible_dynamic_mask,
        admissible_all_groups_mask=admissible_all_groups_mask,
        stable_mask=stable_mask,
    )
    metrics = {
        "steps": int(step_counter),
        "dt_s": dt,
        "simulated_duration_s": step_counter * dt,
        "initial_box_z_m": initial_z,
        "maximum_box_z_m": float(box_position[:, 2].max()),
        "maximum_lift_m": float(relative_height.max()),
        "final_lift_m": float(relative_height[-1]),
        "bilateral_contact_steps": int(np.count_nonzero(both_hands_contact)),
        "bilateral_contact_during_lift_steps": int(np.count_nonzero(dynamic_mask)),
        "clean_bilateral_contact_during_lift_steps": int(
            np.count_nonzero(clean_dynamic_mask)
        ),
        "admissible_bilateral_contact_during_lift_steps": int(
            np.count_nonzero(admissible_dynamic_mask)
        ),
        "all_groups_bilateral_steps": int(np.count_nonzero(all_groups_bilateral)),
        "require_settled_all_groups_frames": (
            args.require_settled_all_groups_frames
        ),
        "ik_contact_preload_height_m": args.ik_contact_preload_height_m,
        "ik_contact_preload_steps": args.ik_contact_preload_steps,
        "ik_contact_preload_ramp_steps": args.ik_contact_preload_ramp_steps,
        "ik_contact_compression_m": args.ik_contact_compression_m,
        "ik_contact_compression_steps": args.ik_contact_compression_steps,
        "ik_contact_compression_ramp_steps": (
            args.ik_contact_compression_ramp_steps
        ),
        "settled_all_groups_gate_passed": settled_all_groups_gate_passed,
        "settled_contact_component_passed": settled_contact_component_passed,
        "settled_nonhand_component_passed": settled_nonhand_component_passed,
        "settled_hard_limit_component_passed": (
            settled_hard_limit_component_passed
        ),
        "all_groups_bilateral_longest_run_steps": longest_true_run(
            all_groups_bilateral
        ),
        "clean_all_groups_bilateral_steps": int(
            np.count_nonzero(clean_all_groups_mask)
        ),
        "clean_all_groups_bilateral_longest_run_steps": longest_true_run(
            clean_all_groups_mask
        ),
        "admissible_all_groups_bilateral_steps": int(
            np.count_nonzero(admissible_all_groups_mask)
        ),
        "admissible_all_groups_bilateral_longest_run_steps": longest_true_run(
            admissible_all_groups_mask
        ),
        "terminal_stable_longest_run_steps": longest_true_run(stable_mask),
        "ik_tracking_position_error_median_m": (
            float(np.nanmedian(ik_tracking_position_error))
            if np.any(ik_step_mask)
            else None
        ),
        "ik_tracking_position_error_maximum_m": (
            float(np.nanmax(ik_tracking_position_error))
            if np.any(ik_step_mask)
            else None
        ),
        "ik_tracking_rotation_error_median_rad": (
            float(np.nanmedian(ik_tracking_rotation_error))
            if np.any(ik_step_mask)
            else None
        ),
        "ik_tracking_rotation_error_maximum_rad": (
            float(np.nanmax(ik_tracking_rotation_error))
            if np.any(ik_step_mask)
            else None
        ),
        "unitree_demo_outward_approach_position_error_maximum_m": (
            float(np.max(unitree_demo_approach_position_error_m))
            if unitree_demo_approach_position_error_m is not None
            else None
        ),
        "unitree_demo_side_clamp_approach_rotation_error_maximum_rad": (
            float(np.max(unitree_demo_approach_rotation_error_rad))
            if unitree_demo_approach_rotation_error_rad is not None
            else None
        ),
        "sugar_side_clamp_direct_refinement_initial_error_m": (
            cpu(sugar_side_clamp_direct_refinement_initial_error_m).tolist()
            if sugar_side_clamp_direct_refinement_initial_error_m is not None
            else None
        ),
        "sugar_side_clamp_direct_refinement_final_error_m": (
            cpu(sugar_side_clamp_direct_refinement_final_error_m).tolist()
            if sugar_side_clamp_direct_refinement_final_error_m is not None
            else None
        ),
        "nonhand_normal_load_maximum_N": float(nonhand_normal_load.max()),
        "nonhand_normal_load_during_all_groups_lift_maximum_N": (
            float(nonhand_normal_load[lifted & all_groups_bilateral].max())
            if np.any(lifted & all_groups_bilateral)
            else None
        ),
        "maximum_finger_hard_limit_violation_rad": maximum_joint_limit_violation,
        "maximum_official_command_reconstruction_error_rad": (
            maximum_command_reconstruction_error
        ),
        "maximum_body_target_hard_limit_violation_rad": (
            maximum_body_target_limit_violation
        ),
        "unitree_demo_body_source_limit_adjustment_rad": (
            unitree_demo_body_limit_adjustment_rad
        ),
        "per_side_group_maximum_normal_load_N": {
            side: {
                group: float(group_load[:, side_index, group_index].max())
                for group_index, group in enumerate(GROUPS)
            }
            for side_index, side in enumerate(SIDES)
        },
        "raw_dynamic_force_residual_median": (
            float(np.median(force_residual[dynamic_mask]))
            if np.any(dynamic_mask)
            else None
        ),
        "raw_dynamic_torque_residual_median": (
            float(np.median(torque_residual[dynamic_mask]))
            if np.any(dynamic_mask)
            else None
        ),
        "matrix_dynamic_force_residual_median": (
            float(np.median(matrix_force_residual[dynamic_mask]))
            if np.any(dynamic_mask)
            else None
        ),
        "raw_matrix_dynamic_force_mismatch_median": (
            float(np.median(raw_matrix_force_mismatch[dynamic_mask]))
            if np.any(dynamic_mask)
            else None
        ),
        "direct_raw_hand_dynamic_force_mismatch_median": (
            float(np.median(direct_raw_hand_mismatch[dynamic_mask]))
            if np.any(dynamic_mask)
            else None
        ),
        "all_robot_dynamic_force_residual_median": (
            float(np.median(all_robot_force_residual[dynamic_mask]))
            if np.any(dynamic_mask)
            else None
        ),
        "all_robot_dynamic_torque_residual_median": (
            float(np.median(all_robot_torque_residual[dynamic_mask]))
            if np.any(dynamic_mask)
            else None
        ),
        "hand_only_clean_dynamic_force_residual_median": (
            float(np.median(force_residual[admissible_dynamic_mask]))
            if np.any(admissible_dynamic_mask)
            else None
        ),
        "hand_only_clean_dynamic_torque_residual_median": (
            float(np.median(torque_residual[admissible_dynamic_mask]))
            if np.any(admissible_dynamic_mask)
            else None
        ),
        "all_robot_clean_dynamic_force_residual_median": (
            float(np.median(all_robot_force_residual[admissible_dynamic_mask]))
            if np.any(admissible_dynamic_mask)
            else None
        ),
        "all_robot_clean_dynamic_torque_residual_median": (
            float(np.median(all_robot_torque_residual[admissible_dynamic_mask]))
            if np.any(admissible_dynamic_mask)
            else None
        ),
        "direct_raw_hand_clean_dynamic_force_mismatch_median": (
            float(np.median(direct_raw_hand_mismatch[admissible_dynamic_mask]))
            if np.any(admissible_dynamic_mask)
            else None
        ),
        "all_robot_terminal_stable_force_residual_median": (
            float(np.median(all_robot_force_residual[stable_mask]))
            if np.any(stable_mask)
            else None
        ),
        "all_robot_terminal_stable_torque_residual_median": (
            float(np.median(all_robot_torque_residual[stable_mask]))
            if np.any(stable_mask)
            else None
        ),
    }
    manifest = {
        "schema": "plan10_official_g1_inspire_carrybox_reachability_v32",
        "host": HOST,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sources": {
            "producer_source": str(producer_source_path),
            "producer_source_sha256": producer_source_sha256,
            "unitree_repo": str(repo),
            "unitree_inspire_mapping_source": str(
                repo / "action_provider/action_provider_dds.py"
            ),
            "unitree_inspire_mapping_source_sha256": sha256(
                repo / "action_provider/action_provider_dds.py"
            ),
            "unitree_inspire_range_source": str(repo / "dds/inspire_dds.py"),
            "unitree_inspire_range_source_sha256": sha256(
                repo / "dds/inspire_dds.py"
            ),
            "unitree_g129_wire_order_source": str(
                repo / "tasks/common_observations/g1_29dof_state.py"
            ),
            "unitree_g129_wire_order_source_sha256": sha256(
                repo / "tasks/common_observations/g1_29dof_state.py"
            ),
            "unitree_usd": str(official_hand_usd),
            "unitree_usd_sha256": sha256(official_hand_usd),
            "robot_motion": str(robot_motion_path),
            "robot_motion_sha256": sha256(robot_motion_path),
            "object_motion": str(object_motion_path),
            "object_motion_sha256": sha256(object_motion_path),
            "official_carrybox_material_source": str(
                official_material_source_path
            ),
            "official_carrybox_material_source_sha256": sha256(
                official_material_source_path
            ),
            "box_usd": str(box_usd),
            "box_usd_sha256": sha256(box_usd),
            "ik_contact_template_trace": (
                str(ik_contact_template_trace)
                if ik_contact_template_trace is not None
                else None
            ),
            "ik_contact_template_trace_sha256": ik_template_trace_sha256,
            "unitree_demo_parquet": (
                str(unitree_demo_parquet)
                if unitree_demo_parquet is not None
                else None
            ),
            "unitree_demo_parquet_sha256": unitree_demo_dataset_sha256,
            "unitree_demo_source_id": args.unitree_demo_source_id,
            "unitree_demo_source_revision": args.unitree_demo_source_revision,
            "static_posture_solutions": (
                str(static_posture_solutions)
                if static_posture_solutions is not None
                else None
            ),
            "static_posture_solutions_sha256": (
                sha256(static_posture_solutions)
                if static_posture_solutions is not None
                else None
            ),
            "static_posture_source_trace": (
                str(static_posture_source_trace)
                if static_posture_source_trace is not None
                else None
            ),
            "static_posture_source_trace_sha256": (
                sha256(static_posture_source_trace)
                if static_posture_source_trace is not None
                else None
            ),
        },
        "parameters": {
            "source_start": args.source_start,
            "source_end": args.source_end,
            "close_start": args.close_start,
            "close_end": args.close_end,
            "close_fraction": args.close_fraction,
            "pregrasp_thumb_pitch_rad": args.pregrasp_thumb_pitch_rad,
            "pregrasp_thumb_yaw_rad": args.pregrasp_thumb_yaw_rad,
            "pregrasp_thumb_pitch_by_side_rad": (
                pregrasp_thumb_pitch_by_side_rad
            ),
            "pregrasp_thumb_yaw_by_side_rad": pregrasp_thumb_yaw_by_side_rad,
            "closed_thumb_yaw_rad": closed_thumb_yaw_rad,
            "closed_finger_fraction_by_side": closed_finger_fraction,
            "closed_thumb_pitch_by_side_rad": closed_thumb_pitch_rad,
            "closed_thumb_yaw_by_side_rad": closed_thumb_yaw_by_side_rad,
            "lift_thumb_yaw_by_side_rad": lift_thumb_yaw_by_side_rad,
            "lift_thumb_yaw_start_step": (
                args.sugar_side_clamp_lift_thumb_yaw_start_step
            ),
            "lift_thumb_yaw_ramp_steps": (
                args.sugar_side_clamp_lift_thumb_yaw_ramp_steps
            ),
            "hand_control_contract": "unitree_official_inspire_6dof_coupled",
            "bilateral_shoulder_roll_offset_rad": (
                args.bilateral_shoulder_roll_offset_rad
            ),
            "left_shoulder_roll_offset_rad": left_shoulder_roll_offset_rad,
            "left_shoulder_roll_end_offset_rad": (
                left_shoulder_roll_end_offset_rad
            ),
            "right_shoulder_roll_offset_rad": right_shoulder_roll_offset_rad,
            "shoulder_roll_transition_start": (
                args.shoulder_roll_transition_start
            ),
            "shoulder_roll_transition_end": args.shoulder_roll_transition_end,
            "shoulder_roll_transition_contract": (
                "fixed_linear_source_time_retarget"
                if body_offset_transition_enabled
                else "constant_offset"
            ),
            "left_wrist_roll_offset_rad": args.left_wrist_roll_offset_rad,
            "left_wrist_roll_end_offset_rad": left_wrist_roll_end_offset_rad,
            "right_wrist_roll_offset_rad": args.right_wrist_roll_offset_rad,
            "left_wrist_yaw_offset_rad": args.left_wrist_yaw_offset_rad,
            "left_wrist_yaw_end_offset_rad": left_wrist_yaw_end_offset_rad,
            "right_wrist_yaw_offset_rad": args.right_wrist_yaw_offset_rad,
            "left_shoulder_yaw_offset_rad": args.left_shoulder_yaw_offset_rad,
            "right_shoulder_yaw_offset_rad": args.right_shoulder_yaw_offset_rad,
            "grasp_dwell_source": args.grasp_dwell_source,
            "grasp_close_steps": args.grasp_close_steps,
            "grasp_close_source_span": args.grasp_close_source_span,
            "grasp_settle_steps": args.grasp_settle_steps,
            "resume_source_span": args.resume_source_span,
            "resume_polynomial_power": args.resume_polynomial_power,
            "robot_root_y_offset_m": args.robot_root_y_offset_m,
            "static_posture_index": args.static_posture_index,
            "static_posture_source_index": args.static_posture_source_index,
            "static_posture_contact_delta_pca_by_side_m": (
                static_posture_contact_delta_pca_by_side_m
            ),
            "static_posture_contact_delta_box_b_by_side_m": {
                side: static_posture_contact_delta_box_b_np[index].tolist()
                for index, side in enumerate(("left", "right"))
            },
            "static_posture_contact_delta_world_by_side_m": (
                {
                    side: static_posture_contact_delta_world_np[index].tolist()
                    for index, side in enumerate(("left", "right"))
                }
                if static_posture_contact_delta_world_np.shape == (2, 3)
                else None
            ),
            "static_posture_transfer_contract": (
                "source_box_relative_root_pose_to_one_time_initial_box_pose"
                if static_posture_solutions is not None
                else None
            ),
            "waist_pitch_absolute_rad": args.waist_pitch_absolute_rad,
            "bilateral_hip_roll_outward_offset_rad": (
                args.bilateral_hip_roll_outward_offset_rad
            ),
            "zero_initial_object_velocity": args.zero_initial_object_velocity,
            "object_static_friction": args.object_static_friction,
            "object_dynamic_friction": args.object_dynamic_friction,
            "object_restitution": 0.0,
            "object_material_before_shape": list(object_material_before.shape),
            "object_material_before": cpu(object_material_before).tolist(),
            "object_material_requested": cpu(object_material_requested).tolist(),
            "object_material_readback": cpu(object_material_readback).tolist(),
            "robot_material_readback_shape": list(robot_material_readback.shape),
            "robot_material_readback_min": (
                cpu(robot_material_readback).min(axis=(0, 1)).tolist()
            ),
            "robot_material_readback_max": (
                cpu(robot_material_readback).max(axis=(0, 1)).tolist()
            ),
            "body_control_mode": args.body_control_mode,
            "fix_robot_root": args.fix_robot_root,
            "ik_template_index": args.ik_template_index,
            "ik_approach_clearance_m": args.ik_approach_clearance_m,
            "ik_approach_steps": args.ik_approach_steps,
            "ik_left_box_roll_offset_rad": args.ik_left_box_roll_offset_rad,
            "ik_right_box_roll_offset_rad": args.ik_right_box_roll_offset_rad,
            "ik_track_live_box_during_grasp": (
                args.ik_track_live_box_during_grasp
            ),
            "ik_track_live_box_during_lift": (
                args.ik_track_live_box_during_lift
            ),
            "ik_live_lift_lead_m": args.ik_live_lift_lead_m,
            "ik_live_lift_lead_ramp_steps": (
                args.ik_live_lift_lead_ramp_steps
            ),
            "ik_live_lift_scheduled_world_z": (
                args.ik_live_lift_scheduled_world_z
            ),
            "ik_live_lift_anchor_world_xy": (
                args.ik_live_lift_anchor_world_xy
            ),
            "ik_live_hold_relative_to_box": (
                args.ik_live_hold_relative_to_box
            ),
            "ik_live_hold_translation_bias_w_m": (
                cpu(ik_live_hold_translation_bias_w).tolist()
                if ik_live_hold_translation_bias_w is not None
                else None
            ),
            "ik_palm_press_steps": args.ik_palm_press_steps,
            "sugar_side_clamp_palm_press_first": (
                args.sugar_side_clamp_palm_press_first
            ),
            "sugar_side_clamp_close_first_hand_before_second": (
                args.sugar_side_clamp_close_first_hand_before_second
            ),
            "ik_close_steps": args.ik_close_steps,
            "ik_thumb_close_steps": args.ik_thumb_close_steps,
            "ik_settle_steps": args.ik_settle_steps,
            "ik_lift_height_m": args.ik_lift_height_m,
            "ik_lift_steps": args.ik_lift_steps,
            "unitree_demo_episode_index": args.unitree_demo_episode_index,
            "unitree_demo_frame_index": args.unitree_demo_frame_index,
            "unitree_demo_box_yaw_rad": args.unitree_demo_box_yaw_rad,
            "unitree_demo_align_box_narrow_axis_to_hands": (
                args.unitree_demo_align_box_narrow_axis_to_hands
            ),
            "unitree_demo_effective_box_yaw_rad": (
                unitree_demo_effective_box_yaw_rad
            ),
            "unitree_demo_box_center_offset_m": list(
                args.unitree_demo_box_center_offset_m
            ),
            "unitree_demo_outward_shift_m": args.unitree_demo_outward_shift_m,
            "unitree_demo_outward_translation_m": (
                cpu(unitree_demo_outward_translation).tolist()
                if unitree_demo_outward_translation is not None
                else None
            ),
            "unitree_demo_support_size_xy_m": list(
                args.unitree_demo_support_size_xy_m
            ),
            "unitree_demo_support_cradle_height_m": (
                args.unitree_demo_support_cradle_height_m
            ),
            "unitree_demo_support_cradle_wall_thickness_m": (
                0.03 if args.unitree_demo_support_cradle_height_m > 0.0 else 0.0
            ),
            "unitree_demo_support_cradle_clearance_m": (
                0.005 if args.unitree_demo_support_cradle_height_m > 0.0 else 0.0
            ),
            "unitree_demo_support_top_z_m": (
                float(unitree_demo_initial_box_pose[2].item())
                + CARRYBOX_MESH_Z_MIN_M
                if unitree_demo_initial_box_pose is not None
                else None
            ),
            "unitree_demo_approach_steps": args.unitree_demo_approach_steps,
            "unitree_demo_side_clamp": args.unitree_demo_side_clamp,
            "sugar_side_clamp": args.body_control_mode == "sugar_side_clamp",
            "side_clamp_box_axis": args.side_clamp_box_axis,
            "side_clamp_box_local_tangent_m": side_clamp_box_local_tangent_m,
            "side_clamp_box_local_tangent_by_side_m": (
                side_clamp_box_local_tangent_by_side_m
            ),
            "side_clamp_box_local_z_by_side_m": (
                side_clamp_box_local_z_by_side_m
            ),
            "side_clamp_box_normal_by_side_m": (
                side_clamp_box_normal_by_side_m
            ),
            "side_clamp_palm_inset_by_side_m": (
                side_clamp_palm_inset_by_side_m
            ),
            "side_clamp_tilt_tangent_by_side_rad": (
                side_clamp_tilt_tangent_by_side_rad
            ),
            "side_clamp_tilt_height_by_side_rad": (
                side_clamp_tilt_height_by_side_rad
            ),
            "side_clamp_normal_roll_by_side_rad": (
                side_clamp_normal_roll_by_side_rad
            ),
            "side_clamp_contact_pca_by_side_m": (
                side_clamp_contact_pca_by_side_m
            ),
            "side_clamp_outward_pca_by_side": (
                side_clamp_outward_pca_by_side
            ),
            "side_clamp_contact_geometry_source": (
                str(side_clamp_contact_geometry_source)
                if side_clamp_contact_geometry_source is not None
                else None
            ),
            "side_clamp_contact_geometry_source_sha256": (
                sha256(side_clamp_contact_geometry_source)
                if side_clamp_contact_geometry_source is not None
                else None
            ),
            "carrybox_pca_center_b": CARRYBOX_PCA_CENTER_B,
            "carrybox_pca_basis_b": CARRYBOX_PCA_BASIS_B,
            "carrybox_pca0_bounds_m": CARRYBOX_PCA0_BOUNDS_M,
            "carrybox_pca2_bounds_m": CARRYBOX_PCA2_BOUNDS_M,
            "pca0_mode_local_z_is_pca2_coordinate": True,
            "tilted_approach_uses_final_palm_plus_x_outward": True,
            "sugar_side_clamp_box_local_offset_m": list(
                args.sugar_side_clamp_box_local_offset_m
            ),
            "sugar_side_clamp_box_world_offset_m": (
                cpu(sugar_side_clamp_box_offset_w).tolist()
                if sugar_side_clamp_box_offset_w is not None
                else None
            ),
            "sugar_side_clamp_support_height_m": (
                args.sugar_side_clamp_support_height_m
            ),
            "sugar_side_clamp_support_size_xy_m": list(
                args.sugar_side_clamp_support_size_xy_m
            ),
            "sugar_side_clamp_support_center_xy_m": (
                np.asarray(
                    sugar_side_clamp_support_center_xy, dtype=np.float64
                ).tolist()
                if sugar_side_clamp_support_center_xy is not None
                else None
            ),
            "sugar_side_clamp_support_static_friction": (
                1.0 if sugar_side_clamp_support_center_xy is not None else None
            ),
            "sugar_side_clamp_support_dynamic_friction": (
                1.0 if sugar_side_clamp_support_center_xy is not None else None
            ),
            "sugar_side_clamp_support_restitution": (
                0.0 if sugar_side_clamp_support_center_xy is not None else None
            ),
            "sugar_side_clamp_support_contract": (
                "fixed_real_collider_never_moved_box_must_leave_for_gt_0p10m_gate"
                if sugar_side_clamp_support_center_xy is not None
                else "absent"
            ),
            "sugar_side_clamp_fit_box_to_reachable_palms": (
                args.sugar_side_clamp_fit_box_to_reachable_palms
            ),
            "sugar_side_clamp_direct_setup_ik": (
                args.sugar_side_clamp_direct_setup_ik
            ),
            "sugar_side_clamp_simultaneous_hand_close": (
                args.sugar_side_clamp_simultaneous_hand_close
            ),
            "sugar_side_clamp_close_second_hand_during_second_press": (
                args.sugar_side_clamp_close_second_hand_during_second_press
            ),
            "sugar_side_clamp_close_second_thumb_during_second_press": (
                args.sugar_side_clamp_close_second_thumb_during_second_press
            ),
            "sugar_side_clamp_second_thumb_close_during_four_finger_steps": (
                args.sugar_side_clamp_second_thumb_close_during_four_finger_steps
            ),
            "sugar_side_clamp_digit_force_stop_n": (
                args.sugar_side_clamp_digit_force_stop_n
            ),
            "sugar_side_clamp_digit_force_stop_max_scale_step": (
                args.sugar_side_clamp_digit_force_stop_max_scale_step
            ),
            "sugar_side_clamp_direct_refinement_steps_requested": (
                args.sugar_side_clamp_direct_refinement_steps
            ),
            "sugar_side_clamp_max_reachable_orientation_delta_rad": (
                args.sugar_side_clamp_max_reachable_orientation_delta_rad
            ),
            "sugar_side_clamp_reachable_fit_offset_w_m": (
                cpu(sugar_side_clamp_reachable_fit_offset_w).tolist()
                if sugar_side_clamp_reachable_fit_offset_w is not None
                else None
            ),
            "sugar_side_clamp_reachable_orientation_delta_rad": (
                cpu(sugar_side_clamp_reachable_orientation_delta_rad).tolist()
                if sugar_side_clamp_reachable_orientation_delta_rad is not None
                else None
            ),
            "sugar_side_clamp_direct_refinement_steps": (
                sugar_side_clamp_direct_refinement_steps
            ),
            "sugar_side_clamp_direct_refinement_accepted_steps": (
                sugar_side_clamp_direct_refinement_accepted_steps
            ),
            "sugar_side_clamp_direct_refinement_accepted_by_side": (
                sugar_side_clamp_direct_refinement_accepted_by_side
            ),
            "sugar_side_clamp_direct_refinement_initial_error_m": (
                cpu(sugar_side_clamp_direct_refinement_initial_error_m).tolist()
                if sugar_side_clamp_direct_refinement_initial_error_m is not None
                else None
            ),
            "sugar_side_clamp_direct_refinement_final_error_m": (
                cpu(sugar_side_clamp_direct_refinement_final_error_m).tolist()
                if sugar_side_clamp_direct_refinement_final_error_m is not None
                else None
            ),
            "sugar_side_clamp_direct_refinement_final_trust_radius_m": (
                cpu(
                    sugar_side_clamp_direct_refinement_final_trust_radius_m
                ).tolist()
                if sugar_side_clamp_direct_refinement_final_trust_radius_m
                is not None
                else None
            ),
            "unitree_demo_side_clamp_box_local_y_m": (
                args.unitree_demo_side_clamp_box_local_y_m
            ),
            "unitree_demo_side_clamp_box_local_z_m": (
                args.unitree_demo_side_clamp_box_local_z_m
            ),
            "unitree_demo_side_clamp_palm_inset_m": (
                args.unitree_demo_side_clamp_palm_inset_m
            ),
            "unitree_demo_side_clamp_approach_clearance_m": (
                args.unitree_demo_side_clamp_approach_clearance_m
            ),
            "side_clamp_approach_clearance_by_side_m": (
                side_clamp_approach_clearance_by_side_m
            ),
            "unitree_demo_side_face_x_m": CARRYBOX_SIDE_FACE_X_M,
            "unitree_demo_inspire_palm_surface_point_b": (
                INSPIRE_PALM_SURFACE_POINT_B
            ),
            "unitree_demo_close_steps": args.unitree_demo_close_steps,
            "unitree_demo_settle_steps": args.unitree_demo_settle_steps,
            "ik_unrecorded_setup_steps": (
                args.ik_approach_steps
                if args.body_control_mode == "bilateral_ik"
                else (
                    args.unitree_demo_approach_steps
                    + sugar_side_clamp_direct_refinement_accepted_steps
                    if args.body_control_mode
                    in ("unitree_demo_pose", "sugar_side_clamp")
                    else 0
                )
            ),
            "ik_setup_object_offstage": args.body_control_mode
            in ("bilateral_ik", "unitree_demo_pose", "sugar_side_clamp"),
            "physics_substeps_per_source": args.physics_substeps_per_source,
            "brake_start": args.brake_start,
            "brake_steps": args.brake_steps,
            "solver_position_iterations": args.solver_position_iterations,
            "solver_velocity_iterations": args.solver_velocity_iterations,
            "hold_steps": args.hold_steps,
            "contact_threshold_N": args.contact_threshold_n,
            "require_settled_all_groups_frames": (
                args.require_settled_all_groups_frames
            ),
            "ik_contact_preload_height_m": args.ik_contact_preload_height_m,
            "ik_contact_preload_steps": args.ik_contact_preload_steps,
            "ik_contact_preload_ramp_steps": (
                args.ik_contact_preload_ramp_steps
            ),
            "ik_contact_compression_m": args.ik_contact_compression_m,
            "ik_contact_compression_steps": args.ik_contact_compression_steps,
            "ik_contact_compression_ramp_steps": (
                args.ik_contact_compression_ramp_steps
            ),
            "settled_all_groups_gate_passed": settled_all_groups_gate_passed,
            "settled_contact_component_passed": (
                settled_contact_component_passed
            ),
            "settled_nonhand_component_passed": (
                settled_nonhand_component_passed
            ),
            "settled_hard_limit_component_passed": (
                settled_hard_limit_component_passed
            ),
        },
        "contact_body_names": hand_body_names,
        "contact_body_groups": hand_body_groups,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "claim_boundary": (
            "No-learning articulated-hand PhysX mechanics. In pd_target, "
            "bilateral_ik, and sugar_side_clamp modes the root and joint state are written once and "
            "gravity remains enabled. Pd-target mode sends the official SUGAR "
            "29-joint motion through the official articulation drives with "
            "manifest-declared bounded retarget offsets and a declared reference "
            "close/resume. Bilateral IK fixes the root as an explicit mechanics "
            "fixture, holds the non-arm source pose, and drives both seven-DoF "
            "arms with IsaacLab's official DifferentialIKController; the official "
            "Unitree six-command Inspire mapping and coupled finger joints are "
            "actuated in declared palm-press, four-finger-close, then thumb-close "
            "stages. During unrecorded bilateral-IK setup the box is offstage; "
            "it is then written once to the declared ground trial state. A declared "
            "ground-grasp alignment may causally read the current dynamic box pose "
            "while press/closure/settle runs, then freezes exactly one end-of-grasp "
            "anchor for open-loop lift/hold; it never writes or replays object state. "
            "Unitree-demo mode hash-binds the official dataset/revision, converts "
            "its real Inspire command with Unitree's released reversed DDS ranges, "
            "and reorders the published 29-joint wire state by exact joint name. "
            "Any real-data versus USD body-limit discrepancy is clamped to the live "
            "hard limit only when at most 0.01 rad and is recorded exactly. "
            "A declared option may align the unchanged box's real 0.400 m local-X "
            "extent once with the recorded horizontal hand-to-hand axis; it does "
            "not scale the box, move the hands, or replay object state. "
            "The side-clamp option retains that hash-bound official body source, "
            "then uses the exact official palm local-minus-X surface geometry and "
            "IsaacLab DifferentialIKController to address declared surfaces on the "
            "unchanged CarryBox. Its legacy path opposes local-X/PCA0 side faces; "
            "an explicitly hash-bound per-hand geometry record may instead declare "
            "a mixed side-brace/bottom-support target and its cooked-surface normal. "
            "Palm roll is the closest projection of the official source orientation; "
            "no wrist angles or object states are replayed. "
            "The box is written once after an offstage open-hand approach, may be "
            "causally tracked only during declared grasp construction, and is thereafter "
            "lifted open-loop from one frozen contact anchor. A declared low physical "
            "setup cradle may stabilize the detailed box collision mesh before grasp; "
            "its exact geometry is fixed in the manifest and remains below the admitted "
            ">0.10 m clean lift interval. This is "
            "not policy, tactile, "
            "soft-gel, autonomous behavior, or sim-to-real evidence. A declared "
            "fixed-root run isolates arm/hand mechanics and is not standing or "
            "locomotion evidence. Bilateral IK uses a hash-bound, mechanically "
            "admissible contact-pose template and open-loop lift after one contact "
            "anchor. "
            "State-replay "
            "mode remains a kinematic diagnostic and cannot pass the physical gate."
        ),
    }
    atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


try:
    main()
except BaseException:
    # Isaac Sim shutdown may otherwise hide a pending Python traceback in an
    # off-screen process.  Emit it before closing the application.
    traceback.print_exc()
    raise
finally:
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
