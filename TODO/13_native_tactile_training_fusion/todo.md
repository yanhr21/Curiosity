# TODO 13: Native Whole-Hand Tactile Training and Fusion

## A. Serious fusion path

- [x] Define the four-frame 54-patch native force/shear tensor without RGB,
  contact proxy, object state, or simulator contact velocity.
- [x] Reuse the existing SUGAR spatial tactile encoder and official actor MLP.
- [x] Define the official-width reference-only actor, privileged critic, and
  frozen official Refiner teacher observation groups.
- [x] Preserve the released Refiner mapping at zero tactile and restrict actor
  training to the serious spatial encoder plus first-layer tactile columns.
- [x] Define matched physical-tactile and exact-zero/no-read arms.
- [x] Add process-local task registration and a retained-allocation launcher.

## B. Live preflight

- [x] Identify and reject random mid-trajectory physical-skin resets using the
  measured `37.46%` versus continuous `0.744%` nonzero-taxel mismatch.
- [x] Run and withdraw the random compressed-student continuous-start route:
  4,464 sampled frames stayed exact zero and never reached contact.
- [x] Confirm that no RGB/depth camera is instantiated or consumed.
- [x] Exclude all 54 elastomer sensor bodies from the original G1 rigid-body
  material randomization while retaining the official robot/object events.
- [x] Record the first live-contact update separately from update zero so its
  physical tactile values, encoder gradients, and parameter change come from
  the same PPO update.
- [x] Feed the complete canonical CarryBox taxel trace through the exact
  serious late-fusion path: causal history/order pass, zero histories remain
  exact zero, and all nonzero histories reach the spatial encoder.
- [x] Backpropagate actual canonical contact frames through the fixed adapter:
  base-column gradient stays exact zero while tactile columns and the spatial
  encoder both receive nonzero gradients.
- [x] Confirm statically that the official-width actor/critic are `890-D`, raw
  tactile is `324000-D`, the embedding is `256-D`, zero tactile recovers the
  released actor and critic exactly, base-column gradients are zero, and
  gradients reach the tactile columns and spatial encoder.
- [x] Reconfirm the same actor/tactile contract inside a live IsaacLab rollout.
- [x] Confirm that tactile becomes nonzero only after continuously reached
  physical contact and then produces encoder gradients.
- [x] Run the fresh official-warm-start exact-zero update and confirm its
  observation is exact zero without calling the sensor-reading function.

## C. Matched training and evaluation

- [x] Train tactile and zero arms serially at identical declared endpoints.
  The update-63 pair is complete; initialization is exact and only the eight
  declared adapter tensors differ at the endpoint.
- [ ] Freeze and evaluate both on common motion, mass, friction, and push
  conditions. Nominal and 1.0 kg/low-friction pairs are complete and mixed;
  push conditions and cross-seed evaluation remain.
- [x] Test one frozen tactile checkpoint with live, exact-zero, and fixed
  anatomical-patch-permuted inputs. Actions remain identical before contact
  and diverge at the first tactile-supported step, proving real modality and
  spatial-layout dependence.
- [x] Render the synchronized camera-enabled live, exact-zero, and fixed
  anatomical-patch-permuted CarryBox cohort. All three H.264 files full-decode
  and use one readable physical-taxel scale; this remains a presentation
  cohort separate from camera-free numerical evaluation.
- [x] Run the predeclared frozen authority curve at tactile-column scales
  `0/0.25/0.5/0.75/1.0`, verify matched initial states and scale-zero
  equivalence, and choose the next serious fusion route from the full curve.
  Lift and tracking error both increase with authority; no nonzero scale
  jointly improves reward, tracking, and lift.
- [x] Implement and structurally audit the declared bounded first-layer
  tactile correction (`0.15` cap) with contact-balanced official-teacher
  distillation, exact zero-tactile recovery, and unchanged serious spatial
  encoder/base SUGAR actor. The CPU audit passes zero equivalence, saturation,
  gradient isolation, and supported-sample reduction; live optimization is a
  separate gate.
- [x] Pass the bounded route's live import/checkpoint and first real-contact
  optimization gates. Contact occurs in updates 9--12 and reaches all eight
  declared adapter tensors.
- [x] Train bounded tactile and matched exact-zero arms serially for the
  predeclared 16-update gate, then freeze and evaluate them. The outcome is
  mixed and contact-state teacher alignment is worse with live tactile, so the
  route is not extended.
- [x] Define and structurally audit the next serious fusion route: retain the
  hidden `0.15` cap, restore official all-sample distillation, and bound each
  normalized tactile action residual at `0.1` relative to the exact official
  zero-tactile action.
- [x] Pass the action-residual route's live contact gate, then run its matched
  exact-zero arm and frozen comparison serially. Common-horizon lift and
  tracking improve, but reward and termination remain worse at one seed.
- [ ] Extend the exact action-residual route to the declared 64-update endpoint
  with a fresh matched zero arm; freeze and compare before any architecture
  change or cross-task claim.
- [ ] Decide whether tactile improves behavior, not merely optimization loss.

## D. RGB fusion after the no-RGB result

- [ ] Compare the existing official ResNet18 RGB actor against tactile-only.
- [ ] Fuse the RGB and tactile embeddings before the unchanged SUGAR actor MLP.
- [ ] Compare RGB, tactile, RGB+tactile, and matched missing-modality controls.
