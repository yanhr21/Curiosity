"""Report all 32 recorded TRAIN cases without filtering acquisition failures."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    root = Path(args.data)
    if args.wait_train:
        owner = dict(line.split('=', 1) for line in (root / 'collection.process').read_text().splitlines())
        while not (root / 'TRAIN_QUALIFICATION.json').exists():
            if (root / 'collection.status').exists():
                raise RuntimeError('Collection terminated before completing TRAIN')
            os.kill(int(owner['child_pid']), 0)
            time.sleep(10)
    # The coordinator finishes the TRAIN records before writing qualification.
    qualification = json.loads((root / 'TRAIN_QUALIFICATION.json').read_text())
    records = [r for r in json.loads((root / 'COLLECTION_RESULT.json').read_text())['records']
               if r['split'] == 'train']
    if len(records) != 32 or len({r['episode'] for r in records}) != 32:
        raise RuntimeError('Expected all 32 distinct TRAIN cases')
    rows = []
    for record in records:
        source = Path(record['source'])
        result = json.loads((source / 'RESULT.json').read_text())
        with np.load(source / f"episode_{record['episode']:04d}.npz") as saved:
            load = saved['normal_load_n'].reshape(-1, 2, 27).sum(2)
            previous = np.vstack([np.zeros((1, 2)), load[:-1]])
            clock = saved['timestamp_s']
            eligible = (clock >= 24) & (clock <= 40) & (saved['validation_controller_lift_start_s'] < 0)
            components = {
                'planes': saved['validation_controller_fit_valid'].all(1),
                'alignment': (saved['validation_controller_alignment_error_deg'] <= 5).all(1),
                'load': (np.abs(previous / result['grip_target_per_hand_n'] - 1) <= .25).all(1),
            }
            fractions = {k: float(v[eligible].mean()) if eligible.any() else None for k, v in components.items()}
            clearance = saved['validation_full_mesh_min_z_m']
            if len(clock) != 2400 or not np.isfinite(load).all() or not np.isfinite(clearance).all():
                raise RuntimeError('Incomplete/nonfinite physical recording')
            rows.append(dict(episode=record['episode'], geometry_group=record['geometry_group'],
                controller_passed=record['controller_passed'], failed_checks=[k for k, v in result['checks'].items() if not v],
                lift_start_s=result['lift_start_s'], true_hold_frames=result['hold_frames'],
                airborne_hold_label_frames=record['airborne_hold_label_frames'],
                prelift_eligible_frames=int(eligible.sum()), readiness_component_fractions=fractions,
                final_recorded_load_n=load[-1].tolist(), peak_recorded_load_n=float(load.max()),
                final_mesh_clearance_m=float(clearance[-1]),
                recorded_full_normal_fraction_during_mass_labels=record['recorded_full_normal_fraction_during_mass_labels']))
    report = dict(train_complete=True, total_cases=32, coverage=qualification,
                  controller_passes=sum(r['controller_passed'] for r in rows),
                  actual_lift_triggers=sum(r['lift_start_s'] >= 0 for r in rows), rows=rows,
                  scope='All fixed TRAIN cases, including failed grasps. No held-out/model/policy result; full-force and object states are validation only.',
                  readiness_scope='Component fractions cover eligible 24..40s before lift; load uses prior recorded measurement. Empty intervals are missing, not zero.',
                  physics_controls_added=0, model_forwards_added=0, optimizer_updates_added=0)
    (root / 'TRAIN_COLLECTION_REPORT.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    x = np.arange(len(rows))
    colors = ['#138f99' if r['controller_passed'] else '#b56739' for r in rows]
    axes[0].bar(x, [r['airborne_hold_label_frames'] for r in rows], color=colors)
    axes[0].axhline(100, color='firebrick', ls='--', label='Per-setting coverage minimum')
    axes[0].set_ylabel('Airborne hold label frames'); axes[0].legend()
    for key, offset, color in [('planes', -.25, '#138f99'), ('alignment', 0, '#637baa'), ('load', .25, '#b56739')]:
        values = [r['readiness_component_fractions'][key] for r in rows]
        axes[1].bar(x + offset, [np.nan if v is None else v for v in values], width=.24, color=color, label=key)
    axes[1].set_ylabel('Pre-lift eligible fraction'); axes[1].set_ylim(0, 1.1); axes[1].legend(ncol=3)
    axes[1].set_title('Separate readiness components; n/a = already lifted at earliest eligible time')
    for index, row in enumerate(rows):
        if not row['prelift_eligible_frames']:
            axes[1].text(index, .05, 'n/a', ha='center', fontsize=7, color='#637080')
    axes[2].bar(x, [100 * r['final_mesh_clearance_m'] for r in rows], color=colors)
    axes[2].axhline(0, color='black', lw=.6); axes[2].set_ylabel('Final full-mesh clearance (cm)')
    for ax in axes:
        for boundary in np.arange(3.5, 31, 4):
            ax.axvline(boundary, color='#cad5df', lw=1)
        ax.grid(axis='y', alpha=.2)
    axes[2].set_xticks(x, [r['episode'] for r in rows], rotation=65)
    fig.suptitle(f"All 32 TRAIN cases: {report['controller_passes']} control passes; "
                 f"{qualification['covered_geometry_groups']}/8 fully covered groups (minimum 4)")
    fig.tight_layout(); fig.savefig(root / 'train_collection_summary.png', dpi=140); plt.close(fig)
    print(json.dumps(dict(complete=True, report=str(root / 'TRAIN_COLLECTION_REPORT.json'),
                         coverage_passed=qualification['passed'], controller_passes=report['controller_passes'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--wait-train', action='store_true')
    main(parser.parse_args())
