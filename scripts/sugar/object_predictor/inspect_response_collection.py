"""Follow the bounded collector: causal readback and actual-mesh previews, no simulation."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .collect_response_corpus import write_json
from .response_gain import ObservedForceGain


FRAMES = (750, 1200, 1500, 2399)


def inspect_case(case, episode):
    with np.load(case / f'episode_{episode}.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    with np.load(case / 'contact_surface.npz') as saved:
        field = {k: saved[k] for k in ('offset', 'pad', 'area_m2', 'normal_pressure_pa')}
    clock = data['timestamp_s']
    gains = data['validation_controller_response_gain_m_per_ns']
    upper = data['validation_controller_response_upper_n_m']
    counts = data['validation_controller_response_samples']
    estimator = ObservedForceGain(.000025)
    previous = np.zeros(2)
    max_gain = max_response = 0.
    exact_counts = True
    for frame in range(len(clock)):
        distance = np.zeros(2) if frame == 0 else data['validation_controller_distance_m'][frame - 1]
        for hand in (0, 1):
            gain, response, count = estimator.update(hand, float(clock[frame]), float(previous[hand]),
                float(distance[hand]), .02, bool(data['validation_controller_fit_valid'][frame, hand]))
            max_gain = max(max_gain, abs(gain - gains[frame, hand]))
            max_response = max(max_response, abs(response - upper[frame, hand]))
            exact_counts &= count == int(counts[frame, hand])
        begin, end = field['offset'][frame:frame + 2]
        sides = field['pad'][begin:end] // 27
        previous = np.array([(field['area_m2'][begin:end][sides == h] *
                              field['normal_pressure_pa'][begin:end][sides == h]).sum() for h in (0, 1)])
    product = float(np.max(gains * .02 * upper))
    replay = dict(frames=len(clock), gain_max_abs=max_gain, response_max_abs=max_response,
        sample_counts_exact=bool(exact_counts), response_gain_dt_max=product,
        passed=bool(len(clock) == 2400 and max_gain == 0 and max_response == 0 and
                    exact_counts and product <= .1 + 1e-12 and gains.max() <= .00015 + 1e-12))
    write_json(case / 'RESPONSE_REPLAY.json', replay)
    if not replay['passed']:
        raise RuntimeError(f'Causal response replay failed: {episode}')
    result = json.loads((case / 'RESULT.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    loads = data['normal_load_n'].reshape(-1, 2, 27).sum(2)
    target = result['grip_target_per_hand_n']
    for hand in (0, 1):
        axes[0].plot(clock, loads[:, hand], label=('left', 'right')[hand])
        axes[2].plot(clock, gains[:, hand] * 1e6, label=('left', 'right')[hand])
    axes[0].axhspan(.75 * target, 1.25 * target, color='gray', alpha=.15)
    axes[0].set_ylabel('Observed normal load (N)'); axes[0].legend()
    axes[1].plot(clock, data['validation_full_mesh_min_z_m'] * 100)
    axes[1].axhline(10, ls=':', color='gray'); axes[1].set_ylabel('Full-mesh clearance (cm)')
    axes[2].set_ylabel('Gain (um / N s)'); axes[2].set_xlabel('Time (s)')
    for ax in axes:
        ax.grid(alpha=.2); ax.axvline(24, ls=':', color='gray'); ax.axvline(40, ls=':', color='firebrick')
    failures = ', '.join(k for k, v in result['checks'].items() if not v)
    fig.suptitle(f'{episode}: continuous palmar + response gain; physical PASS={result["passed"]}\n'
                 f'Failed checks: {failures or "none"}; no model predictions')
    fig.tight_layout(); fig.savefig(case / 'physical_trace.png', dpi=120); plt.close(fig)
    subprocess.run([sys.executable, '-m', 'scripts.sugar.object_predictor.render_tactile_planes',
        '--source', str(case), '--preview', '--preview-frames', *map(str, FRAMES)], check=True)
    return dict(episode=episode, physical_passed=result['passed'], failed_checks=failures,
        final_clearance_cm=float(data['validation_full_mesh_min_z_m'][-1] * 100),
        replay=replay, preview_frames=list(FRAMES), visual_scope='Four fixed actual-mesh frames; local geometric fit only, no neural prediction; human inspection tracked separately.')


def publish(root, records, terminal=None):
    write_json(root / 'INSPECTION_PROGRESS.json', dict(complete=terminal is not None,
        collector_terminal=terminal, records=records, new_physics_steps=0, model_forwards=0,
        scope='Automatic causal replay and nonconstant actual-mesh previews; not a human image-review claim.'))
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>连续掌侧采集核查</title><style>body{max-width:1500px;margin:25px auto;padding:20px;font:18px/1.6 system-ui;background:#f4f7fa;color:#182838}img{max-width:100%}summary{cursor:pointer}</style>',
        '<h1>原配置：连续掌侧与响应增益</h1><p>全部已完成案例按原顺序展示，不删除失败。每例固定15.02、24.02、30.02、48秒真实网格；右侧为触觉局部几何诊断，尚无新模型预测。原物理门槛不变。</p>',
        '<p><a href="README.md">进展与范围</a> · <a href="PROTOCOL.json">协议</a> · <a href="INSPECTION_PROGRESS.json">逐帧回读</a></p>',
        f'<p>已核查 {len(records)} 条。采集状态：{html.escape(str(terminal)) if terminal else "进行中"}。</p>']
    for row in records:
        episode = row['episode']; base = f'cases/episode_{episode}'
        parts.append(f'<details><summary>{episode}：物理通过 {row["physical_passed"]}，最终离地 {row["final_clearance_cm"]:.2f} cm；失败项 {html.escape(row["failed_checks"] or "无")}</summary>')
        parts.append(f'<img loading="lazy" src="{base}/physical_trace.png">')
        for frame in FRAMES:
            parts.append(f'<img loading="lazy" src="{base}/renders/planes_frame_{frame:04d}.png">')
        parts.append('</details>')
    parts.append('</html>')
    (root / 'index.html').write_text('\n'.join(parts))


def main(root):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    owner = dict(line.split('=', 1) for line in (root / 'collection.process').read_text().splitlines())
    records = []; seen = set()
    while True:
        state = json.loads((root / 'COLLECTION_RESULT.json').read_text())
        for record in state['records']:
            episode = record['episode']
            if episode not in seen:
                records.append(inspect_case(Path(record['source']), episode)); seen.add(episode)
                publish(root, records)
                print('INSPECTED_SAVED_CASE', episode, len(records), flush=True)
        terminal_path = root / 'collection.status'
        if terminal_path.exists():
            # Re-read after observing terminal status to avoid missing its last atomic record.
            latest = json.loads((root / 'COLLECTION_RESULT.json').read_text())
            if len(latest['records']) != len(seen):
                continue
            terminal = dict(line.split('=', 1) for line in terminal_path.read_text().splitlines())
            publish(root, records, terminal)
            print('INSPECTION_COMPLETE', len(records), terminal, flush=True)
            return
        os.kill(int(owner['child_pid']), 0)
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--data', type=Path, required=True)
    main(parser.parse_args().data)
