# G1 Low-Carry Control Matrix

This is a diagnostic summary only, not a success claim.

| case | check | fall/drop | first fall/drop | target travel robot/box | final lat robot/box | max tilt robot/box | terminal | final hold | lateral |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| lowcarry700_pass | pass | 0/0 | -/- | 1.994/2.025 | 0.428/0.374 | 0.227/0.242 | scale=0.015, latch=- | final_steps=-, final_stand=-, stand_delay=-, stand_steps=-, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=False, terminal_only=-, start=-, excess=-, steps=0, max=0.000 |
| nolateral900 | fail | 44/25 | 860/880 | 3.423/3.440 | 0.229/0.188 | 0.976/1.275 | scale=0.015, latch=- | final_steps=-, final_stand=-, stand_delay=-, stand_steps=-, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=False, terminal_only=-, start=-, excess=-, steps=0, max=0.000 |
| latched_zero900 | fail | 141/97 | 760/810 | 0.379/0.356 | 0.320/0.507 | 1.678/1.657 | scale=0.000, latch=True | final_steps=-, final_stand=-, stand_delay=-, stand_steps=-, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=False, terminal_only=-, start=-, excess=-, steps=0, max=0.000 |
| latched_micro900 | fail | 90/42 | 810/840 | 1.386/1.344 | 1.376/1.389 | 1.705/1.905 | scale=0.006, latch=True | final_steps=-, final_stand=-, stand_delay=-, stand_steps=-, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=False, terminal_only=-, start=-, excess=-, steps=0, max=0.000 |
| terminal_lateral900 | fail | 288/269 | 620/630 | 0.950/0.959 | -0.596/-0.658 | 2.069/2.172 | scale=0.006, latch=True | final_steps=-, final_stand=-, stand_delay=-, stand_steps=-, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=True, terminal_only=True, start=-, excess=-, steps=519, max=0.006 |
| terminal_lateral_threshold_invalid611 | fail | 0/0 | -/- | 1.241/1.269 | 0.551/0.559 | 0.335/0.328 | scale=0.006, latch=True | final_steps=-, final_stand=-, stand_delay=-, stand_steps=-, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=True, terminal_only=True, start=0.550, excess=False, tilt_gate=999.000/999.000, tilt_supp=0, steps=0, max=0.000 |
| terminal_lateral_threshold_fix900 | fail | 0/0 | -/- | 1.612/1.645 | 1.532/1.673 | 0.594/0.946 | scale=0.006, latch=True | final_steps=-, final_stand=-, stand_delay=-, stand_steps=-, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=True, terminal_only=True, start=0.550, excess=False, tilt_gate=999.000/999.000, tilt_supp=0, steps=289, max=0.003 |
| terminal_lateral_threshold045_tiltgate900 | fail | 210/158 | 690/709 | 1.086/1.034 | 1.477/1.439 | 3.139/3.135 | scale=0.006, latch=True, final=-@-, final_latch=- | final_steps=-, final_stand=-, stand_delay=-, stand_steps=-, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=True, terminal_only=True, start=0.450, excess=False, tilt_gate=0.450/0.450, tilt_supp=245, steps=105, max=0.003 |
| terminal015_final006_2m900 | fail | 54/29 | 846/871 | 3.355/3.349 | 0.494/0.543 | 0.346/0.725 | scale=0.015, latch=True, final=0.006@2.000, final_latch=True | final_steps=205, final_stand=False, stand_delay=0, stand_steps=0, target_window=-, window=-+/--, both_steps=-, both_streak=- | on=False, terminal_only=False, start=0.000, excess=False, tilt_gate=999.000/999.000, tilt_supp=0, steps=0, max=0.000 |
| terminal015_final000_2m900 | missing | - | - | - | - | - | - | - | - |
| terminal015_final000_2m_finalstand900 | missing | - | - | - | - | - | - | - | - |
| terminal_lateral_excess_tiltgate_fallback900 | missing | - | - | - | - | - | - | - | - |
