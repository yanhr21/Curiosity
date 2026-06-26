# Phase 05 Contact-Proxy Ablation Report V1

## Scope

This report records the first tactile/contact ablation evidence available for
the Newton lift-hold task. It reuses the validated Phase 03 curiosity replay
output and the Phase 05 Newton contact source manifest.

This is diagnostic replay evidence only:

- no model training;
- no policy update;
- no learned residual adapter;
- no learned world model;
- no T-Rex schema promotion;
- no calibrated tactile F6;
- no dense tactile deformation stream.

## Evidence

- Curiosity replay output:
  `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`
- Curiosity replay report:
  `experiments/reports/2026-06-27_phase03_curiosity_reward_replay_v1.md`
- Newton contact source manifest:
  `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`
- Residual adapter and forward-model target contract:
  `experiments/configs/residual_adapter_forward_model_contract_v1.json`

Source status:

- curiosity replay status: pass;
- replay rollout count: 9;
- held-out cells: `full_low`, `empty_high`;
- tactile source: `newton.contact_proxy_only`;
- contact manifest status: pass;
- contact manifest source runs: 10;
- contact manifest records: 3600;
- contact proxy range: 29 to 63;
- generated T-Rex fields: none;
- schema promotion: blocked.

## Ablation Mapping

The current Phase 05 ablation labels map to existing diagnostic replay outputs
as follows:

| Phase 05 label | Current evidence field | Current limitation |
| --- | --- | --- |
| vision-only | `ablation_object_motion_only_mean` | Object-motion proxy, not a trained RGB-only policy. |
| tactile/contact-only | `ablation_tactile_only_mean` | Newton contact proxy only, not tactile F6. |
| vision+tactile/contact | `ablation_vision_tactile_mean` | Object-motion plus contact proxy reward diagnostic. |
| shuffled tactile/contact | `ablation_shuffled_tactile_mean` | Deterministic contact-proxy shuffle. |
| delayed tactile/contact | `ablation_delayed_tactile_mean` | One-step delayed contact proxy. |

## Aggregate Values

Mean across 9 validated Phase 02 mass/friction replay rollouts:

| Ablation | Mean |
| --- | ---: |
| object-motion-only proxy | 0.0008622763 |
| contact-only | 0.0117951099 |
| tactile/contact-only proxy | 0.0117951099 |
| vision+tactile/contact proxy | 0.0063286931 |
| shuffled tactile/contact proxy | 0.0452058186 |
| delayed tactile/contact proxy | 0.0117951099 |

## Per-Cell Values

| Cell | Held out | Object proxy | Contact proxy | Vision+contact | Shuffled contact | Delayed contact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `empty_medium` | false | 0.0008447798 | 0.0126183844 | 0.0067315821 | 0.0380501393 | 0.0126183844 |
| `half_medium` | false | 0.0008602376 | 0.0115598886 | 0.0062100631 | 0.0442896936 | 0.0115598886 |
| `full_medium` | false | 0.0009072839 | 0.0122841226 | 0.0065957032 | 0.0491922006 | 0.0122841226 |
| `empty_low` | false | 0.0008445678 | 0.0110306407 | 0.0059376042 | 0.0382172702 | 0.0110306407 |
| `half_low` | false | 0.0008568009 | 0.0108077994 | 0.0058323002 | 0.0431476323 | 0.0108077994 |
| `full_low` | true | 0.0009177010 | 0.0129247911 | 0.0069212460 | 0.0474651811 | 0.0129247911 |
| `half_high` | false | 0.0008173565 | 0.0122562674 | 0.0065368119 | 0.0507242340 | 0.0122562674 |
| `full_high` | false | 0.0008702904 | 0.0117548747 | 0.0063125825 | 0.0583565460 | 0.0117548747 |
| `empty_high` | true | 0.0008414687 | 0.0109192201 | 0.0058803444 | 0.0374094708 | 0.0109192201 |

## Interpretation

This report satisfies the current Phase 05 reporting gate for contact-proxy
ablations. It does not prove tactile benefit in a trained policy.

The evidence is still useful because corrupted contact proxy is separable from
the noncorrupted replay signals: shuffled contact has a substantially larger
diagnostic value than delayed or nonshuffled contact on this rollout set. The
next claim must remain narrower: contact-proxy diagnostics are wired and
auditable, not that a residual adapter has learned to use touch.

The next Phase 04 step is still to train a residual controller-parameter
adapter, with this ablation structure preserved for evaluation.
