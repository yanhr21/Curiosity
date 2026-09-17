"""Independent data-pairing readback; never equate action differences with following."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "experiments/demo_following/demo_future_smp_v1/refiner_pair_feasibility"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args()
    run=args.run
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    switch = protocol["switch_control_frame"]
    arms = protocol["arms"]
    results = {arm: json.loads((run / arm / "RESULT.json").read_text()) for arm in arms}
    traces = {arm: dict(np.load(run / arm / "TRACE.npz", allow_pickle=False)) for arm in arms}
    startup = {arm: dict(np.load(run / arm / "STARTUP.npz", allow_pickle=False)) for arm in arms}
    checks = {"frozen_teachers": all(r["teacher_frozen"]["passed"] for r in results.values()),
              "all_finite": all(r["all_numeric_finite"] for r in results.values()),
              "all_requested_steps_without_reset": all(r["full_budget_without_reset"] for r in results.values()),
              "original_lifts_ten_consecutive": results["original"]["at_least_ten_consecutive_lifted_frames"],
              "alternate_lifts_ten_consecutive": results["alternate"]["at_least_ten_consecutive_lifted_frames"],
              "actual_alternate_executed": results["alternate"]["actual_alternate_reference_executed"]}
    original = traces["original"]
    comparisons = {}
    fields = ["robot_body_state_before_w", "robot_root_state_before_w", "object_state_before_w",
              "joint_pos_before", "joint_vel_before", "requested_action", "teacher_observation"]
    for arm in ("repeat", "alternate"):
        n = min(switch, len(original["done"]), len(traces[arm]["done"]))
        errors = {key: float(np.max(np.abs(original[key][:n]-traces[arm][key][:n]))) for key in fields}
        equal = {key: bool(np.array_equal(startup['original'][key], startup[arm][key])) for key in startup['original']}
        checks[f"{arm}_startup_equal"] = all(equal.values())
        checks[f"{arm}_common_prefix"] = n == switch and max(errors.values()) <= 1e-5
        world_fields = fields[:5]
        switch_errors = {key: float(np.max(np.abs(original[key][switch]-traces[arm][key][switch])))
                         for key in world_fields} if min(len(original['done']),len(traces[arm]['done'])) > switch else {}
        checks[f"{arm}_same_switch_world_state"] = bool(switch_errors and max(switch_errors.values()) <= 1e-5)
        comparisons[arm] = {"prefix_samples": n, "prefix_max_absolute_error": errors,
                            "switch_world_max_absolute_error":switch_errors, "startup_equal": equal}
    end = min(len(x["done"]) for x in traces.values())
    valid = np.ones(max(0, end-switch), dtype=bool)
    for x in traces.values():
        valid &= ~x['done'][switch:end].reshape(-1)
    metrics = {}
    for arm in ("repeat", "alternate"):
        delta = (traces[arm]['object_state_after_w'][switch:end, 0, :3] - original['object_state_after_w'][switch:end, 0, :3])[valid]
        metrics[f"{arm}_box_position_rms_m"] = float(np.sqrt(np.mean(delta**2))) if len(delta) else None
    signal, noise = metrics['alternate_box_position_rms_m'], metrics['repeat_box_position_rms_m']
    checks['box_difference_exceeds_repeat_noise'] = bool(signal is not None and signal > max(1e-4, 5*noise))
    for arm, x in traces.items():
        valid_arm = ~x['done'].reshape(-1)
        # Reference and live measurement are both captured before the action on
        # the same frame. End-of-step auto-reset data never enters this metric.
        box = x['object_state_before_w'][:,0,:3]
        reference = x['reference_object_pos_w'][:,0]
        contact = np.linalg.norm(x['contact_force_after_w'], axis=-1)[:,0] > .1
        mask = valid_arm & (np.arange(len(valid_arm)) >= switch)
        metrics[arm] = {'post_switch_reference_box_rmse_m': float(np.sqrt(np.mean((box[mask]-reference[mask])**2))) if mask.any() else None,
                        'post_switch_bilateral_hand_contact_fraction': float(contact[mask,:2].all(axis=-1).mean()) if mask.any() else None,
                        'post_switch_valid_frames': int(mask.sum()),
                        'same_state_reference_action_rms': float(np.sqrt(np.mean((x['requested_action']-x['alternate_query_action'])**2)))}
    passed = all(checks.values())
    result = dict(execution_completed=True, data_pair_feasibility_passed=passed, checks=checks,
                  prefix_comparisons=comparisons, metrics=metrics, arm_results=results,
                  claim_boundary="Even a paired-future feasibility pass does not establish selected-demo following. Reference tracking, contact behavior, official SMP feature scores and visual evidence remain required. Single seed and nominal physics are diagnostic only.",
                  next_action="Inspect reference feasibility and exact live-state official SMP features; no predictor/PPO budget follows solely from action or future differences." if passed else "Diagnose failed teacher floor, common-prefix or reference-switch checks before collecting training pairs.")
    (run/'READBACK.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12,7))
    for arm, x in traces.items():
        valid_arm = ~x['done'].reshape(-1)
        time = np.arange(len(valid_arm))*.02
        for j, coordinate in enumerate((0,2)):
            axes[0,j].plot(time[valid_arm],x['object_state_before_w'][valid_arm,0,coordinate],label=arm)
        axes[1,0].plot(time[valid_arm],np.linalg.norm(x['object_state_before_w'][valid_arm,0,:3]-x['reference_object_pos_w'][valid_arm,0],axis=-1),label=arm)
        axes[1,1].plot(time[valid_arm],np.linalg.norm(x['contact_force_after_w'][valid_arm,0,:2],axis=-1).sum(-1),label=arm)
    for ax,title in zip(axes.flat,('Actual box X (m)','Actual box Z (m)','Current reference box error (m)','Actual hand-to-box load sum (N)')):
        ax.set_title(title);ax.set_xlabel('Control time (s)');ax.axvline(switch*.02,color='gray',ls='--');ax.legend()
    fig.suptitle(f'Frozen {protocol.get("controller", "refiner")} PhysX diagnostic: different motion is not yet demonstrated following')
    fig.tight_layout();fig.savefig(run/'PHYSICAL_READBACK.png',dpi=160);plt.close(fig)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
