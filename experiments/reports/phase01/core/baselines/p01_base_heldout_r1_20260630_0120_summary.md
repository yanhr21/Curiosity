# Phase 01 Held-Out Baseline Summary

Status: complete baseline evidence; not training and not curiosity success.

- baseline run tag: `p01_base_heldout_r1_20260630_0120`
- MP4 export run tag: `p01_base_mp4_20260630_0132`
- Slurm job: `157902`
- host: `server64`
- GPU: `NVIDIA H200`
- held-out cells: 4
- baseline methods: `no_adaptation`, `scripted_feedback`
- completed metrics: 8/8
- completed MP4 videos: 8/8

## Metrics

| eval tag | method | status | lift height m | hold duration s | max slip m | not dropped | contact loss frames | max accel m/s2 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `p01_base_heldout_r1_20260630_0120_no_adaptation_heldout_box_heavy_low_large_offset` | `no_adaptation` | success | 0.230657 | 27.332916 | 0.008755 | true | 1 | 2.837742 |
| `p01_base_heldout_r1_20260630_0120_no_adaptation_heldout_cup_empty_high_misleading` | `no_adaptation` | success | 0.166397 | 27.132919 | 0.003988 | true | 1 | 0.836516 |
| `p01_base_heldout_r1_20260630_0120_no_adaptation_heldout_cup_full_low_hidden` | `no_adaptation` | success | 0.159533 | 27.099586 | 0.004037 | true | 0 | 1.119794 |
| `p01_base_heldout_r1_20260630_0120_no_adaptation_heldout_cylinder_heavy_low_masked_vision` | `no_adaptation` | success | 0.199326 | 27.182919 | 0.024882 | true | 0 | 5.157203 |
| `p01_base_heldout_r1_20260630_0120_scripted_feedback_heldout_box_heavy_low_large_offset` | `scripted_feedback` | success | 0.229832 | 27.132919 | 0.008188 | true | 1 | 5.288425 |
| `p01_base_heldout_r1_20260630_0120_scripted_feedback_heldout_cup_empty_high_misleading` | `scripted_feedback` | success | 0.166390 | 26.882923 | 0.004115 | true | 1 | 0.618227 |
| `p01_base_heldout_r1_20260630_0120_scripted_feedback_heldout_cup_full_low_hidden` | `scripted_feedback` | success | 0.159994 | 26.832924 | 0.004016 | true | 0 | 2.174650 |
| `p01_base_heldout_r1_20260630_0120_scripted_feedback_heldout_cylinder_heavy_low_masked_vision` | `scripted_feedback` | success | 0.199727 | 26.932922 | 0.016995 | true | 0 | 3.706063 |

## Evidence Paths

- Metrics JSON/CSV:
  `experiments/outputs/phase01/core/baselines/`
- MP4 export summary:
  `experiments/outputs/phase01/core/baselines/p01_base_mp4_20260630_0132_mp4_summary.json`
- MP4 export report:
  `experiments/reports/phase01/core/baselines/p01_base_mp4_20260630_0132_mp4.md`
- Visual videos:
  `experiments/visuals/phase01/core/baselines/<eval_tag>/rollout_video.mp4`

This baseline evidence defines the non-curiosity held-out comparison target.
It does not update a policy and does not count as a positive curiosity result.
