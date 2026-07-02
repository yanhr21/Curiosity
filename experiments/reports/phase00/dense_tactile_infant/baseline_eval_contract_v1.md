# Baseline And Evaluation Contract

Date: 2026-07-01

Machine-readable record:
`experiments/configs/phase00/dense_tactile_infant/baseline_eval_contract_v1.json`

Status: contract ready, training not started.

## Baselines

Future training must compare against:

- no-adaptation base;
- scripted feedback;
- no-curiosity residual/adaptation;
- closed-loop curiosity;
- vision-only ablation;
- tactile-only masked-vision ablation;
- noisy/delayed/shuffled tactile ablation;
- serious official/reference method or documented blocker.

## Harder Tasks

If base grasp/lift/hold is already easy, the task must get harder before a
curiosity claim:

- lower friction;
- mass or fill mismatch;
- off-center grasp;
- shape change;
- deformable/compliant object;
- fragile force limit;
- held-out material or stiffness.

## Metrics

Required metrics:

- lift;
- hold duration;
- slip;
- drop;
- contact loss;
- object acceleration;
- force/contact cost;
- safety regression;
- strongest-baseline comparison.

Success claim condition:

```text
harder held-out tasks beat strongest baseline without safety regression
```

Anything weaker is incomplete or negative evidence.
