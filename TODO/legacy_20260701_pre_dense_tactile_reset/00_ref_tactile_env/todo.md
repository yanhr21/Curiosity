# Phase 00 Reference-Video Tactile Environment TODO

## Active Rules

- This is the current active TODO after the 2026-07-01 reference-video reset.
- High-signal active evidence index:
  `experiments/reports/phase00/ref_tactile/active_evidence_index.md`.
- Current requirement-status audit:
  `experiments/reports/phase00/ref_tactile/phase00_requirement_status.md`
  and
  `experiments/configs/phase00/ref_tactile/phase00_requirement_status_v1.json`.
- Previous contact-count Phase 00/01 work is archived under
  `TODO/legacy_20260630_contact_proxy_stopgate/` and is not the active queue.
- Do not run simulation, rendering, model loading, dataset conversion,
  training, or NumPy/PyTorch-heavy checks on the login node.
- Use Curiosity-owned tmux-held Slurm allocations for all compute-side sanity,
  rendering, simulation, dataset conversion, training, and evaluation.
- Keep new artifact paths short and grouped under `phase00/ref_tactile/`.
- Do not commit unless the user explicitly asks.
- Do not start curiosity training until the dense tactile environment and base
  grasp evidence pass.

## Codebase Audit

- [x] Inspect existing local `external/` repositories.
- [x] Check upstream commits for Newton, Taccel, T-Rex, HydroShear, and
      IsaacLabTactile.
- [x] Write source audit report under
      `experiments/reports/phase00/ref_tactile/`.
- [x] Update or clone missing official repositories without overwriting dirty
      local work.
- [x] Record dependency and environment blockers before any compute run.
- [x] Clone and audit official UniVTAC as a visuo-tactile manipulation
      benchmark/reference without running compute on the login node.
- [x] Clone and audit official TaCauchy as an FEM tactile semantic reference
      without running compute on the login node.
- [x] Recheck additional 2026 tactile references and record code availability.
      - Report:
        `experiments/reports/phase00/ref_tactile/latest_reference_code_recheck.md`
      - FreeTacMan cloned at `external/FreeTacMan`, commit
        `9285740a5d33385d3a9cf5ccdb185e3387b547bd`, as a secondary real
        visuo-tactile data/pretraining reference.
      - DiffTactile cloned at `external/DiffTactile`, commit
        `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`, as a secondary
        differentiable tactile simulator/reference.
      - DiffTactile environment note: `requirements.txt` is UTF-16
        little-endian text and must not be blindly installed without
        encoding-aware review.
      - Tacmap and ControlTac remain code-unavailable comparison gaps after web
        and common GitHub remote probes.
- [x] Clone and audit official APPLE and Tactile MNIST as secondary active
      tactile perception / curiosity-design references.
      - APPLE: `external/APPLE`, commit
        `4b1d71fadb786d865d4ee29a184ab408b9605083`.
      - Tactile MNIST: `external/tactile-mnist`, commit
        `9e4e59139e9349ab361a3b9297f4815724ad6387`.
      - Report:
        `experiments/reports/phase00/ref_tactile/curiosity_reference_audit.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/curiosity_reference_matrix_v1.json`.
      - Status: source audit only; no official example or training has run.
        These are Gate 00G design references, not grasping checkpoints and not
        Gate 00F semantic-validation replacements.
- [x] Clone and audit Reactive Diffusion Policy, ImplicitRDP, and Tactile
      Diffusion as secondary policy/photometric references.
      - Reactive Diffusion Policy:
        `external/reactive_diffusion_policy`, commit
        `824c5e8de1fd1811106907a04b5f0186e0138c0b`.
      - ImplicitRDP:
        `external/ImplicitRDP`, commit
        `4c90646df17787e31c88838106c4a0323ddefb4a`.
      - Tactile Diffusion:
        `external/Tactile-Diffusion`, commit
        `16868fb96d19d93dc5837600c26b48415632e4f6`.
      - Report:
        `experiments/reports/phase00/ref_tactile/policy_reference_audit.md`.
      - Status: source audit only; no checkpoint download, model loading,
        training, or official sanity has run. These references are future
        baselines/comparisons, not current base-model success.
      - Action-conditioned tactile prediction remote HEAD
        `085d2ab82d2e0574f39a359dd2c445b8f7f7a3b3` was observed, but local
        clone failed with `fetch-pack: unexpected disconnect while reading
        sideband packet`.
- [x] Record the Phase 00 semantic reference matrix under
      `experiments/configs/phase00/ref_tactile/semantic_validation_reference_matrix_v1.json`.
- [x] Create active evidence index for the current Phase 00 reset under
      `experiments/reports/phase00/ref_tactile/active_evidence_index.md`.
- [x] Create requirement-status audit for the current user requirements under
      `experiments/reports/phase00/ref_tactile/phase00_requirement_status.md`
      and
      `experiments/configs/phase00/ref_tactile/phase00_requirement_status_v1.json`.
      - Classification: status audit only, not training, not base-model
        completion, and not Gate 00D/00E/00F completion.
      - Current result: latest codebase audit is partial positive, Newton main
        steel-spec direct-force candidate is partial positive, 92.6 FPS Panda
        hydro base is partial positive, official semantic validation is
        blocked by missing official UniVTAC/TaCauchy dependency readiness and
        official sanity, and curiosity training remains disallowed.
- [x] Refresh latest remote source HEADs without modifying external repos.
      - Report:
        `experiments/reports/phase00/ref_tactile/latest_source_remote_refresh.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/latest_source_remote_refresh_v1.json`.
      - Current result: Newton upstream `main` is now
        `d58e70266be0db803261f3e46a2f7d923a43db37`, ahead of active evidence
        worktree `external/newton_main` at
        `a217e55fab3d373a08fba374cc5cafc1826cf27f`. Do not call the active
        worktree latest upstream main until a fresh update and compute-side
        sanity pass.
      - Current result: Taccel, UniVTAC, TaCauchy, HydroShear, FreeTacMan,
        DiffTactile, APPLE, Tactile MNIST, Reactive Diffusion Policy,
        ImplicitRDP, and Tactile Diffusion local checkouts match observed
        remote HEADs.
- [x] Recheck latest source truth after Gate 00F env/asset progress without
      running compute or dependency installation on the login node.
      - Report:
        `experiments/reports/phase00/ref_tactile/latest_reference_recheck_20260701_v3.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/latest_reference_recheck_20260701_v3.json`.
      - Current result: Newton upstream `main` is now
        `8c501b47847569fecdda97a9f7f01205c6f7964f`; source-only worktree
        `external/newton_8c501` was added at that commit.
      - Current result: `external/TacEx` is cloned from official
        `https://github.com/DH-Ng/TacEx.git` at
        `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`.
      - Current result: `external/IsaacLabTactile` is cloned from official
        `https://github.com/UM-ARM-Lab/IsaacLabTactile.git` at
        `21bcb476b27ceedccccd63afef6bbd822adc2b2b` using
        `GIT_LFS_SKIP_SMUDGE=1` and blob filtering. `git-lfs` is not on the
        current PATH, so LFS asset completeness and official sanity are not
        verified.
      - URL corrections: use `https://github.com/Taccel-Simulator/Taccel.git`
        for Taccel, `https://github.com/figsama/TaCauchy.git` for TaCauchy,
        and `https://github.com/MMintLab/hydroshear.git` for HydroShear.
        Earlier failed probes of other namespaces are not official source
        evidence.
      - Classification: source readiness only. No official sanity, dependency
        installation, rendering, training, or Gate promotion was run.
- [x] Prepare Newton `8c501...` compute-side sanity handoff without launching
      experiments on the login node.
      - Report:
        `experiments/reports/phase00/ref_tactile/newton_8c501_sanity_handoff.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/newton_8c501_sanity_handoff_v1.json`.
      - Current result: exact tmux-held Slurm commands are recorded for
        runtime benchmark, dense tactile export, reference-video comparison,
        channel audit, and Gate review.
      - Classification: handoff only. No benchmark, tactile export, render,
        Gate review, or training was executed.
- [x] Request a Curiosity tmux-held H200 allocation for Newton `8c501...`
      sanity.
      - Report:
        `experiments/reports/phase00/ref_tactile/newton_8c501_allocation_request.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/newton_8c501_allocation_request_v1.json`.
      - Current result: job `160854`, job name
        `curiosity_p00_8c501_1gpu_1day`, tmux window
        `curiosity_phase00_ref_tactile:alloc_8c501`, initial state
        `PENDING (Priority)`.
      - Classification: allocation request only, not experiment evidence.
- [x] Run Newton `8c501...` runtime benchmark in a Curiosity tmux-held H200
      allocation using the handoff commands.
      - Do not run from the login node.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/newton_8c501_benchmark_status.md`
        and
        `experiments/configs/phase00/ref_tactile/newton_8c501_benchmark_status_v1.json`.
      - Result: job `160854` ran on `server30` with H200. Runs measured
        `80.1 FPS` and `80.8 FPS`; both executed successfully and are
        acceptable around 80 FPS. The old `82 FPS` number is historical
        reference only.
      - Decision: do not block `8c501...` on FPS; proceed to dense tactile
        export when a Curiosity tmux-held Slurm allocation is available.
- [x] Correct the runtime gate so 82 FPS is not treated as a hard blocker.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/runtime_gate_correction_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/runtime_gate_correction_20260701_v1.json`.
      - Result: active policy is around-80-FPS continuation. Do not spend time
        optimizing 80 FPS to 82 FPS before tactile export.
- [x] Run the `8c501...` candidate dense
      tactile export, reference compare, channel audit, and Gate review using
      the handoff sequence.
      - Do not promote `8c501...` above d58 evidence until the full evidence
        chain exists.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/newton_8c501_cont_chain_status.md`
        and
        `experiments/configs/phase00/ref_tactile/newton_8c501_cont_chain_status_v1.json`.
      - Current status: dense tactile export, reference compare, channel audit,
        and Gate review completed on job `160924`. Gate review remains
        `open_not_curiosity_ready` due official reference sanity and semantic
        blockers.
- [x] Update Gate 00F bundle defaults to use the latest 8c501 candidate chain.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_reference_bundle_handoff.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_reference_bundle_handoff_v1.json`,
        and
        `experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh`.
      - Current result: future Gate 00F bundle Gate review defaults to the
        8c501 benchmark/tactile/reference/channel evidence paths.
- [x] Record post-8c501 Gate 00F readiness state.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_post_8c501_readiness_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_post_8c501_readiness_20260701_v1.json`.
      - Current result: Gate 00F remains blocked by dependency-complete
        official runtime readiness, not by Newton candidate evidence.
- [x] Request a Curiosity tmux-held H200 allocation for the `8c501...`
      continuation chain.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/newton_8c501_cont_allocation_request.md`
        and
        `experiments/configs/phase00/ref_tactile/newton_8c501_cont_allocation_request_v1.json`.
      - Current result: job `160924`, tmux window `alloc_8c501_cont`, initial
        state `PENDING (Priority)`.
- [x] Rerun lightweight Gate 00F readiness checks after the `8c501...`
      runtime result and IsaacLabTactile source acquisition.
      - Report:
        `experiments/reports/phase00/ref_tactile/gate00f_readiness_refresh_20260701.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/gate00f_readiness_refresh_20260701_v1.json`.
      - Current result: candidate UniVTAC/TaCauchy env pythons and copied
        assets are present, but `gate00f_ready=false`; effective failed checks
        remain `univtac_official_reference_sanity` and
        `tacauchy_official_reference_sanity`.
      - Current result: IsaacLabTactile source is cloned, but LFS asset
        completeness is not verified because `git-lfs` is unavailable and the
        clone skipped LFS smudge.
- [x] Run lightweight Gate 00F tool lookup without installing dependencies.
      - Report:
        `experiments/reports/phase00/ref_tactile/gate00f_tool_lookup_20260701.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/gate00f_tool_lookup_20260701_v1.json`.
      - Current result: PATH exposes no `git-lfs`, `cmake`, `nvcc`, or
        `nvidia-smi`; project-local lookup found only
        `envs/taccel/cuda-toolkit/bin/nvcc`; no prebuilt Isaac/Lab/TacEx/UIPC
        env directories were found under `envs` at max depth 4.
      - Gate effect: does not clear Gate 00F.
- [x] Run lightweight Gate 00F static source audit without executing
      simulation, rendering, dependency installation, official sanity, model
      loading, data conversion, or training.
      - Report:
        `experiments/reports/phase00/ref_tactile/gate00f_static_source_audit_20260701.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/gate00f_static_source_audit_20260701_v1.json`.
      - Current result: UniVTAC provides the official left/right tactile HDF5
        schema (`rgb_marker`, `marker`, `depth`, `rgb`, `pose`) plus
        ACT/ViTAL manipulation benchmark references, but still needs
        dependency-complete official sanity. TaCauchy provides Cauchy stress,
        normal pressure, tangential traction, optical tactile RGB, marker
        motion, and height/depth semantic references, but still needs
        dependency-complete official sanity. The local IsaacLabTactile clone
        currently looks like generic Isaac Lab/contact-sensor source and is not
        an adequate Gate 00F replacement.
      - Gate effect: does not clear Gate 00F; it refines the exact remaining
        blocker and prevents replacing UniVTAC/TaCauchy with generic contact
        sensor evidence.
- [x] Run lightweight Gate 00F module/env probe without dependency
      installation or Python imports.
      - Report:
        `experiments/reports/phase00/ref_tactile/gate00f_module_env_probe_20260701.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/gate00f_module_env_probe_20260701_v1.json`.
      - Current result: the current login shell has no `module` or `ml`
        command, so module-based lookup for `cmake`, `git-lfs`, CUDA, or Isaac
        is unavailable from this shell. Shallow file-name probing under
        `envs/univtac/conda` and `envs/tacauchy/conda` found no Isaac, TacEx,
        UIPC, cuRobo, or Torch component names.
      - Gate effect: does not clear Gate 00F; it reinforces that the existing
        base env prefixes are not dependency-complete official reference envs.
- [x] Run lightweight Gate 00F container path audit without building images or
      running containers.
      - Report:
        `experiments/reports/phase00/ref_tactile/gate00f_container_path_audit_20260701.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/gate00f_container_path_audit_20260701_v1.json`.
      - Current result: `docker` exists, but `singularity`, `apptainer`,
        `enroot`, and `podman` are not on current PATH. TacEx/TaCauchy provide
        Docker build recipes and IsaacLabTactile provides a Singularity helper,
        but no approved project-local SIF/image tar was found. The discovered
        paths still require image build/setup or placeholder cluster SIF
        configuration.
      - Gate effect: does not clear Gate 00F. A valid container path would
        require an already-built, approved, Curiosity-owned image/SIF on shared
        storage before official sanity can run.
- [x] Refresh latest 2026-07-01 web/source truth for official tactile
      simulation, policy, and representation codebases.
      - Report:
        `experiments/reports/phase00/ref_tactile/latest_20260701_web_codebase_refresh.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/latest_20260701_web_codebase_refresh_v1.json`.
      - New sparse/blobless sources:
        `external/IsaacLab_official` at
        `b4c321024792976150ca55fddb26fa34480d974e`,
        `external/ftp1-policy` at
        `dd7cda66c7e97a170e0435fc6c4428b350cbdcc0`, and
        `external/AnyTouch2` at
        `82c5677d9cf0176d97a1fe04745f63cd02dd6f54`.
      - Current result: official Isaac Lab main TacSL exposes tactile RGB,
        depth, penetration, normal force, and shear force fields plus force
        physics config and a demo entrypoint. It becomes a Gate 00F candidate
        source but still needs official environment/assets and compute-side
        sanity. FTP-1 and AnyTouch2 are future serious policy/representation
        references, not current base-model or Gate-completion evidence.
- [x] Record supplementary photometric/actuator/representation source audit.
      - Report:
        `experiments/reports/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701.md`.
      - Matrix:
        `experiments/configs/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701_v1.json`.
      - Current result: `external/TactSim-IsaacLab` is a secondary
        photometric GelSight/DIGIT-style IsaacLab tactile reference at
        `4f92257177cd0ee18928de720b880505ec7f7638`;
        `external/newton-actuators` is deprecated Newton actuator background
        at `134dacb0912f4b8ce0465ecebf564479f2e62315`; UniT is remote-head-
        only at `52a286520b09708934b25c77aa826360d72c79db`.
      - Gate effect: supplementary source audit only. It does not clear Gate
        00D/00E/00F, does not authorize curiosity training, and does not change
        the current d58 decision.
- [x] Recheck tracked official source freshness without modifying checkouts.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/latest_source_freshness_20260701_v4.md`
        and
        `experiments/configs/phase00/ref_tactile/latest_source_freshness_20260701_v4.json`.
      - Result: tracked official refs for Newton, Taccel, T-Rex, IsaacLab,
        TacEx, TaCauchy, UniVTAC, FTP-1, AnyTouch2, and HydroShear match the
        current records. No stale tracked source was found.
      - Gate effect: source freshness only. Does not run official sanity or
        clear Gate 00F.
- [x] Refresh latest serious policy/checkpoint availability for post-Gate
      curiosity planning.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/latest_policy_checkpoint_refresh_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/latest_policy_checkpoint_refresh_20260701_v1.json`.
      - New clean T-Rex source snapshots:
        `external/T-Rex_43ff` at
        `43ff632259d76f08373c085c53111825060d029b` and
        `external/T-Rex_full_b23` at
        `b23eafe564a1457cd4eacb889aaf6fbf29a29034`.
      - Current result: T-Rex official released checkpoints now give a serious
        future path: `miniFranka/T-Rex_pretrain_mecka22k_epoch1` and
        `miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`. The midtrain
        checkpoint embeds the tactile VQ-VAE and should be the future strongest
        tactile-reactive starting point if a faithful Newton-to-T-Rex data
        contract is built.
      - Gate effect: does not clear Gate 00F, does not download checkpoints,
        does not load models, and does not start curiosity training.
- [x] Define post-Gate 00F policy/checkpoint bridge checklist.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/post_gate00f_policy_bridge_checklist.md`
        and
        `experiments/configs/phase00/ref_tactile/post_gate00f_policy_bridge_checklist_v1.json`.
      - Current result: future T-Rex, FTP-1, AnyTouch2, and Sparsh work now has
        explicit preconditions, schema requirements, ablations, and forbidden
        shortcuts. T-Rex promotion requires real slow/fast RGB, eef-62 or a
        validated adapter, high-frequency hand/finger F6, deformation/tactile
        image alignment, timing metadata, leak-free splits, and normalization
        compatibility.
      - Gate effect: planning only. It does not authorize checkpoint loading or
        training while Gate 00D/00E/00F remain open.
- [x] Extract T-Rex metadata/data-contract gate from official source.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/trex_data_contract.md`
        and
        `experiments/configs/phase00/ref_tactile/trex_data_contract_v1.json`.
      - Validator:
        `src/newton_tactile_curiosity/trex_contract_validate.py`.
      - Current result: future Newton-to-T-Rex conversion must provide
        `observation.images.head`, `observation.images.wrist_right`,
        `observation.images.wrist_left`, `observation.state [62]`,
        `action [16,62]`, `action_abs [62]`,
        `observation.tactile_f6 [10,6]`, ten tactile-deform video streams, and
        q01/q99/mask stats for action/state/tactile_f6.
      - Gate effect: metadata/schema gate only. It confirms current Newton
        Panda evidence is not yet T-Rex-compatible.
- [x] Prepare an official IsaacLab TacSL sanity handoff path.
      - Candidate source:
        `external/IsaacLab_official/scripts/demos/sensors/tacsl_sensor.py`.
      - Required output fields:
        `tactile_rgb_image`, `tactile_depth_image`, `penetration_depth`,
        `tactile_normal_force`, and `tactile_shear_force`.
      - Required config fields:
        `normal_contact_stiffness`, `friction_coefficient`,
        `tangential_stiffness`, `tactile_array_size`, and
        `contact_object_prim_path_expr`.
      - Do not run it from the login node, do not install dependencies on a
        compute node, and do not use it to bypass UniVTAC/TaCauchy official
        sanity.
      - Handoff prepared:
        `experiments/reports/phase00/ref_tactile/isaaclab_tacsl_sanity_handoff.md`
        and
        `experiments/configs/phase00/ref_tactile/isaaclab_tacsl_sanity_handoff_v1.json`.
      - Scripts:
        `experiments/configs/phase00/ref_tactile/run_isaaclab_tacsl_sanity_in_alloc.sh`
        and
        `experiments/configs/phase00/ref_tactile/launch_isaaclab_tacsl_sanity_tmux.sh`.
      - Current status: handoff ready, not run; blocked by missing approved
        dependency-complete IsaacLab/TacSL environment or prebuilt container.
- [x] Wire official IsaacLab TacSL sanity into Gate 00F review logic.
      - Code:
        `src/newton_tactile_curiosity/phase00_gate_review.py`.
      - Launchers:
        `experiments/configs/phase00/ref_tactile/run_phase00_gate_review_in_alloc.sh`
        and
        `experiments/configs/phase00/ref_tactile/launch_phase00_gate_review_tmux.sh`.
      - Current result: future Gate 00F reviews require the semantic matrix to
        include `OfficialIsaacLabTacSL`, require
        `candidate.newton_mjw.penetration_or_compression` in the bridge spec,
        and require an optional-but-hard sanity summary with status
        `pass_official_isaaclab_tacsl_demo_exited_zero` before TacSL can clear
        official reference sanity.
      - Verification: lightweight source compile, `bash -n`, `jq empty`, and
        `git diff --check` passed on the touched files.
- [ ] Run official IsaacLab TacSL sanity only after an approved
      dependency-complete Isaac Lab/TacSL environment or prebuilt Curiosity
      container exists.
      - Must run inside a Curiosity tmux-held Slurm allocation.
      - Must not install dependencies on the compute node.
      - Must feed the resulting summary into Gate 00F via
        `ISAACLAB_TACSL_SANITY_SUMMARY`.
- [x] Record current IsaacLab TacSL env/container blocker refresh.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/isaaclab_tacsl_env_blocker_refresh_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/isaaclab_tacsl_env_blocker_refresh_20260701_v1.json`.
      - Result: running Slurm job `160860` is Reflex-owned
        (`WorkDir=/public/home/yanhongru/ICLR2027/Reflex`) and cannot be
        reused; no `envs/isaaclab_tacsl` prefix and no project-local TacSL/
        Isaac/TacEx prebuilt container archive were found in limited checks.
      - Gate effect: does not clear Gate 00F. It confirms the remaining
        blocker is external to the current active Curiosity workspace state.
- [x] Add unified Gate 00F official reference sanity bundle handoff.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_reference_bundle_handoff.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_reference_bundle_handoff_v1.json`.
      - Scripts:
        `experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh`
        and
        `experiments/configs/phase00/ref_tactile/launch_gate00f_reference_bundle_tmux.sh`.
      - Current result: future Gate 00F sanity can run UniVTAC, TaCauchy,
        IsaacLab TacSL, and then Gate review with fixed summary paths in one
        Curiosity-owned allocation workflow.
      - Safety check:
        `experiments/reports/phase00/ref_tactile/gate00f_bundle_launcher_reflex_refuse_check_20260701.md`
        shows the launcher refused Slurm job `160860` because its workdir is
        `/public/home/yanhongru/ICLR2027/Reflex`.
      - Gate effect: handoff only. It does not run sanity or clear Gate 00F
        without a valid Curiosity allocation and dependency-complete envs or
        explicit blocker summaries.
- [x] Add strict Gate 00F bundle acceptance checker.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_bundle_acceptance_handoff.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_bundle_acceptance_handoff_v1.json`.
      - Validator:
        `src/newton_tactile_curiosity/gate00f_bundle_acceptance.py`.
      - Current result: a future bundle is accepted only if UniVTAC,
        TaCauchy, and IsaacLab TacSL statuses match their required official
        pass values, Gate 00F review status is
        `pass_official_semantic_reference_sanity`, blocker sanity is disabled,
        and Gate review has no failed checks or hard blockers.
      - Gate effect: acceptance guard only. It does not run compute or clear
        Gate 00F without a real passing bundle summary.
- [x] Prepare latest-upstream Newton code path without overwriting active
      evidence.
      - Worktree:
        `external/newton_d58`.
      - Status:
        `experiments/reports/phase00/ref_tactile/newton_d58_worktree_status.md`
        and
        `experiments/configs/phase00/ref_tactile/newton_d58_worktree_status_v1.json`.
      - Current result: detached worktree at
        `d58e70266be0db803261f3e46a2f7d923a43db37`; active evidence worktree
        `external/newton_main` preserved at
        `a217e55fab3d373a08fba374cc5cafc1826cf27f`.
      - Current result: runtime benchmark and candidate tactile export have
        now run on d58; Gate review is still open.
- [x] Request Curiosity tmux-held H200 allocation for Newton d58 sanity.
      - Status:
        `experiments/reports/phase00/ref_tactile/newton_d58_allocation_request.md`
        and
        `experiments/configs/phase00/ref_tactile/newton_d58_allocation_request_v1.json`.
      - Current result: job `160467` was granted on `server02` in
        `curiosity_phase00_ref_tactile:alloc_d58` and reused for benchmark,
        tactile export, and reference-video comparison work.
- [x] Run official Newton hydro benchmark/sanity on `external/newton_d58`
      inside the held H200 allocation after job `160467` is granted.
      - Status:
        `experiments/reports/phase00/ref_tactile/newton_d58_benchmark_status.md`
        and
        `experiments/configs/phase00/ref_tactile/newton_d58_benchmark_status_v1.json`.
      - First run:
        `p00_bench_d58_v1_20260701_070459`, `70.8 FPS`, below the 82 FPS
        target.
      - Hot/longer run:
        `p00_bench_d58_hot_v1_20260701_070611`, `82.7 FPS`, meets the 82 FPS
        target.
      - Classification: runtime sanity passed, not Gate completion.
- [x] Run dense tactile/mechanics export on `external/newton_d58` after
      confirming the existing export scripts remain compatible with d58.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/newton_d58_tactile_export_status.md`,
        `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_summary.json`,
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile.avi`,
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_sheet.jpg`,
        and
        `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_timeseries.npz`.
      - Result: `240` frames, `147` frames with pad-object contacts, max lift
        `0.22254392504692078 m`, max candidate Fn sum `40.08497619628906`,
        max candidate Ft sum `12.025492668151855`, left/right marker-flow
        norms `3.722446918487549` and `3.3947927951812744`, and observed
        steel-candidate material settings `mu=0.3`, `kh=1e12`.
      - Manual visual inspection: sheet is nonblank and shows synchronized
        scene-plus-tactile response during grasp/lift/hold.
      - Classification: candidate direct-force dense tactile evidence only;
        `direct_tactile_claim_allowed=false`, so Gate completion is still
        blocked on reference comparison and tactile semantic validation.
- [x] Consume the d58 tactile export in reference-video comparison and Gate
      review before promoting it over the older active evidence chain.
      - Reference compare:
        `p00_refcmp_d58_marker_v1_20260701_071521`,
        status `pass_reference_comparison_assets`.
      - Channel audit:
        `p00_chan_d58_marker_v1_20260701_071757`,
        status `pass_channel_audit_open_validation`, failed checks `[]`.
      - Gate review:
        `p00_gate_d58_marker_v1_20260701_071843`,
        status `open_not_curiosity_ready`.
      - Passed Gate checks:
        runtime 82 FPS, base grasp/lift final test, steel material, candidate
        Fn/Ft, SensorContact alignment, normal/area proxy overlay, marker-style
        render, reference comparison assets, channel layout audit, semantic
        matrix, bridge spec, and asset reuse plan availability.
      - Failed Gate checks:
        `reference_env_availability`, `reference_asset_availability`,
        `univtac_official_reference_sanity`, and
        `tacauchy_official_reference_sanity`.
      - Classification: d58 is the strongest candidate evidence chain so far,
        but not base/tactile Gate completion and not curiosity readiness.
- [x] Add Gate 00E base evidence audit for the current d58 chain.
      - Evidence:
        `src/newton_tactile_curiosity/gate00e_base_evidence_audit.py`,
        `experiments/reports/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit.md`,
        and
        `experiments/outputs/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit_summary.json`.
      - Result:
        `partial_positive_gate00e_base_candidate_tactile_validation_blocked`.
      - Gate effect: d58 is the current strongest base candidate, but Gate 00E
        remains open because tactile semantics and official reference sanity
        are blocked.
- [x] Add Gate 00D environment evidence audit for the current d58 chain.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701_v1.json`.
      - Result:
        `partial_positive_environment_candidate_reference_semantics_blocked`.
      - Gate effect: d58 has candidate environment/tactile mechanics evidence,
        but contact area is proxy-only and dense penetration/compression
        semantics are not validated.
- [x] Audit the TaCauchy asset blocker after the d58 Gate review.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/envprep/tacauchy_asset_blocker_audit.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/tacauchy_asset_blocker_audit_v1.json`.
      - Result: official `setup_assets.sh` requires `git-lfs`, which is not on
        current PATH. TaCauchy target assets are `1.8M`, lack
        `Sensors/GelSight_Mini/Sensor.usd`, and have `0` tactile test shape
        USD files. UniVTAC bundled TacEx has `410M` of candidate assets,
        including GelSight Mini USDs and `21` test shape USDs.
      - Constraint: do not silently copy the UniVTAC assets into
        `external/TaCauchy`; that is a material official-repo mutation and
        needs explicit approval or a cleaner official `git-lfs` path.
- [x] Audit the UniVTAC/TaCauchy reference environment blocker after the d58
      Gate review.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/envprep/reference_env_blocker_audit.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/reference_env_blocker_audit_v1.json`.
      - Result: target base env pythons for UniVTAC and TaCauchy are now
        present; official dependency installation and official sanity remain
        blocked.
      - Constraint: do not silently run the heavy Isaac/TacEx/UIPC env build on
        the login node, and do not move dependency installation to compute
        nodes.
- [x] Create a Gate 00F decision packet so future work cannot silently mutate
      official repos, use non-Curiosity resources, or start heavy env builds.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/envprep/gate00f_decision_packet.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/gate00f_decision_packet_v1.json`.
      - Result: project-local `nvcc` exists at
        `envs/taccel/cuda-toolkit/bin/nvcc` with CUDA `12.8`, but `git-lfs`,
        executable `cmake`, and target UniVTAC/TaCauchy env pythons remain
        missing.
      - Boundary: asset copy into `external/TaCauchy` and heavy
        Isaac/TacEx/UIPC env construction both require explicit approval.
- [x] Prepare compute-side official sanity launchers for UniVTAC and TaCauchy
      using only prebuilt local environments.
- [x] If no suitable prebuilt Isaac/TacEx/UIPC environment exists, record the
      missing environment as a blocker instead of installing on compute nodes.
      - Evidence:
        `p00_ref_univtac_sanity_v1_20260701_054900` and
        `p00_ref_tacauchy_sanity_v1_20260701_054900` ran inside held Slurm job
        `160450` on `server02`; both matched official commits and both recorded
        `blocked_missing_prebuilt_environment`.
      - Gate evidence:
        `p00_gate_review_v4_20260701_055100` keeps Gate 00F
        `open_official_semantic_validation_blocked`.
- [ ] Prepare approved local shared-filesystem environments for UniVTAC and
      TaCauchy official sanity without installing dependencies on compute
      nodes.
      - Environment plan:
        `experiments/reports/phase00/ref_tactile/reference_environment_plan.md`.
      - Do not run the all-in-one official install scripts blindly; they include
        Isaac installs, TacEx/UIPC builds, asset setup, tests, and data
        collection.
      - Dry-run stage manifest and commands now exist:
        `experiments/configs/phase00/ref_tactile/envprep/reference_env_manifest_v1.json`
        and
        `experiments/configs/phase00/ref_tactile/envprep/prepare_reference_env_stage.sh`.
      - Dry-run stage evidence has been generated for both `univtac` and
        `tacauchy` under
        `experiments/outputs/phase00/ref_tactile/envprep/` and
        `experiments/reports/phase00/ref_tactile/envprep/`. These are command
        plans only, not installed environments and not official sanity passes.
      - Toolchain preflight:
        `experiments/reports/phase00/ref_tactile/envprep/toolchain_preflight.md`.
        Current login environment has Curiosity conda and GCC/G++ 11.4, but
        does not expose `cmake`, `git-lfs`, `nvcc`, or `nvidia-smi`.
      - Repeatable availability preflight:
        `experiments/configs/phase00/ref_tactile/envprep/check_reference_env_availability.sh`.
        Latest output:
        `experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json`
        and
        `experiments/reports/phase00/ref_tactile/envprep/reference_env_availability.md`.
        Current result: UniVTAC and TaCauchy conda Python paths are present;
        venv alternates remain absent; `git-lfs`, `cmake`, and `nvcc` are
        missing on current PATH. Gate 00F remains blocked by official
        UniVTAC/TaCauchy sanity, not by base env file presence.
      - Location audit:
        `experiments/reports/phase00/ref_tactile/envprep/reference_env_location_audit.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/reference_env_location_audit_v1.json`.
        Current result: no approved prebuilt UniVTAC or TaCauchy environment
        was found in project `envs/` or common home conda/env locations; only
        non-target autoresearch venvs were observed outside the project.
        Project-local `envs/taccel/miniforge/bin/conda` exists, but target
        envs have not been created and heavy env construction remains a
        controlled, logged step.
      - Staged environment checklist:
        `experiments/reports/phase00/ref_tactile/envprep/reference_env_stage_checklist.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/reference_env_stage_checklist_v1.json`.
        Current result: UniVTAC and TaCauchy have different official Isaac/Lab
        version requirements and must use separate controlled envs unless a
        future compatibility proof exists. The checklist records exact
        official versions, forbidden all-in-one install shortcuts, required
        assets/toolchain, project-local conda availability, and official
        sanity commands.
      - Stage runner guard:
        `experiments/configs/phase00/ref_tactile/envprep/prepare_reference_env_stage.sh`
        now marks `preflight` as `dry_run_preflight_ready`, `create_env` as
        `dry_run_create_env_ready_not_executed`, and all later stages as
        `blocked_missing_target_env` until the target Python exists.
        Refreshed status JSON files live under
        `experiments/outputs/phase00/ref_tactile/envprep/univtac/` and
        `experiments/outputs/phase00/ref_tactile/envprep/tacauchy/`.
      - Asset availability audit:
        `experiments/reports/phase00/ref_tactile/envprep/reference_asset_availability.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/reference_asset_availability_v1.json`.
        Current result: UniVTAC bundled TacEx has useful GelSight/GF225/shape
        assets present; TaCauchy has partial placeholder assets only and lacks
        required full sensor USD/calibration, valid Franka UIPC assets, and
        tactile test shapes. Gate 00F therefore has both target-env and
        TaCauchy-asset blockers.
      - Candidate local asset reuse plan:
        `experiments/reports/phase00/ref_tactile/envprep/reference_asset_reuse_plan.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/reference_asset_reuse_plan_v1.json`.
        Current status: plan only, not executed. UniVTAC bundled TacEx data is
        about `410M` while TaCauchy data is about `1.8M`; local reuse may avoid
        a Git LFS/network blocker, but it must not be done silently because it
        changes official asset provenance.
      - Asset stage runner:
        `experiments/configs/phase00/ref_tactile/envprep/prepare_reference_asset_stage.sh`
        records guarded dry-run commands for `audit`, `reuse_copy`, and
        `verify`. Current status: `audit` is dry-run ready, `reuse_copy` is
        dry-run not executed, and `verify` is blocked because TaCauchy still
        lacks `Sensors/GelSight_Mini/Sensor.usd`.
      - The tmux launcher
        `experiments/configs/phase00/ref_tactile/launch_tactile_reference_sanity_tmux.sh`
        now refuses to consume a Slurm allocation when the target reference
        Python is missing, unless
        `ALLOW_MISSING_REFERENCE_ENV_BLOCKER_RUN=1` is explicitly set to
        record a compute-side blocker.
      - Gate review now consumes the availability status via
        `REFERENCE_ENV_AVAILABILITY_SUMMARY` and records
        `reference_env_availability` as a Gate 00F check.
      - Gate review now also consumes asset blocker evidence via
        `REFERENCE_ASSET_AVAILABILITY_SUMMARY` and
        `REFERENCE_ASSET_REUSE_PLAN`, adding checks for
        `reference_asset_availability` and
        `reference_asset_reuse_plan_available`. This has only passed syntax
        checks; the Gate review has not been rerun after this code change.

## Environment Spec

- [x] Define dense tactile schema with left/right pad fields, pressure or
      compression maps, `Fn`, `Ft`, shear direction, contact area,
      penetration/compression, material labels, and time-series statistics.
- [x] Define rigid steel/metal first-scene parameters: density, friction,
      stiffness/contact response, object geometry, pad geometry, and target
      diagnostic frame rate.
- [x] Define output layout under:
      `experiments/outputs/phase00/ref_tactile/`,
      `experiments/visuals/phase00/ref_tactile/`,
      `experiments/reports/phase00/ref_tactile/`,
      `logs/newton/phase00/ref_tactile/`.

## Official Sanity

- [x] Prepare tmux-held Slurm launcher for reference tactile environment sanity.
- [x] Run Newton official sanity on compute and record pass/fail.
- [x] Run Taccel official example/sanity on compute and record pass/fail.
- [x] If official sanity fails, document the blocker before custom glue code.

## Reference Diagnostic

- [x] Prepare first rigid-metal diagnostic runner with synchronized visual+tactile
      maps and mechanics plots.
- [ ] Generate first rigid-metal diagnostic with synchronized visual+tactile
      maps and mechanics plots.
      - Current Taccel Panda/Tac-Man-style attempts are partial only:
        videos/arrays were generated, but `max_collision_count=0` and
        `force_nonzero=false`.
      - Current Taccel official peg-style attempts are also partial:
        full 200-step run produced no collision/force/deformation evidence in
        the instrumented path.
      - Next faithful action: use Newton official contact/hydro path as the
        mechanics source, then only merge tactile maps when the source fields
        are nonzero and provenance-labeled.
      - Positive visual base evidence now exists for official Newton Panda
        hydro:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_panda_usd_20260701_023155/panda_hydro.usd`.
        This is not yet a synchronized tactile-map diagnostic.
      - Positive hydro-derived tactile evidence now exists:
        `p00_hydro_tac_avi3_20260701_024826` exported 240 frames,
        left/right tactile maps, NPZ source arrays, AVI videos, and metrics.
        It remains partial because visual scene and tactile maps are not yet
        fused into one synchronized rollout and force/shear fields are not
        complete.
      - Positive synchronized scene+tactile diagnostic now exists:
        `p00_sync_hydro_20260701_025818` exported one synchronized AVI with
        scene schematic, left/right tactile maps, object-z, contact area,
        `hydro_proxy.Fn`, and `hydro_proxy.shear_motion`.
        It remains partial because the scene panel is schematic, not a USD
        render, and `Ft`/pad shear vectors are proxy-only.
      - Stronger base mechanics diagnostic now exists:
        `p00_base_mech_20260701_030544` adds stress proxy, force-weighted
        normal summary, tangential-capacity proxy, force balance, lift/hold/
        drop/slip/safety metrics, and synchronized curve panels. It remains
        partial because `Ft` and pad shear vectors are still proxy-only,
        tactile patches remain sparse, and observed material parameters do not
        match the steel-first spec.
      - Steel-spec material diagnostic now exists:
        `p00_steel_v1_20260701_032709` applies explicit `mu=0.3` and
        `kh=1e12` overrides, verifies observed material arrays, and exports
        synchronized mechanics/tactile video plus source arrays. It remains
        partial because direct `Ft`, pad shear vectors, dense tactile richness,
        and photoreal/USD scene fusion are still missing.
      - Grid-tactile diagnostic now exists:
        `p00_grid_v1_20260701_033556` adds HydroShear-style Gaussian grid
        fields for left/right `Fn`, stress, deformation, shear vectors, and
        shear magnitude. It verifies nonzero grid `Fn` maps and nonzero grid
        shear maps under the steel-spec material override. It remains partial
        because the shear is a contact-center-motion proxy, not direct
        tangential force, and the scene is still schematic.
      - F6 proxy diagnostic now exists:
        `p00_f6_v1_20260701_034033` adds per-pad F6 proxy arrays for normal
        wrench, `Ft_capacity` tangent wrench, and combined wrench. This is a
        T-Rex-aligned bridge shape only; it is not official T-Rex tactile force
        and not direct `SensorContact` force.
      - Direct-force comparison diagnostic now exists:
        `p00_mjc_sensor_v1_20260701_034541` verifies nonzero official
        `SensorContact` force/friction on a separate MuJoCo-contact Panda
        variant. It is useful as a direct-force comparison source, but it does
        not use the Newton hydro collision pipeline and cannot replace the
        active hydro tactile base.
      - Calibrated-view tactile diagnostic now exists:
        `p00_calib_view_v1_20260701_040715` keeps raw grid/F6 arrays and adds
        `*_calibrated_view_*` arrays whose visualization window is derived from
        the rollout's 1%-99% contact local-yz range. This improves tactile
        panel readability but is not a new physical tactile sensor.
      - Latest-main official scene/USD evidence now exists:
        `p00_main_usd_v1_20260701_041900` exported the official Newton main
        Panda hydro cube rollout through `ViewerUSD` from commit
        `a217e55fab3d373a08fba374cc5cafc1826cf27f`. This verifies official
        scene/geometry/rollout export on the same latest-main base, but it is
        not yet rasterized or fused with the calibrated tactile diagnostic.
      - USD raster capability probe:
        `p00_usd_probe_v2_20260701_042430` verifies the USD stage opens
        (`220` prims, `0-239` time codes at `60 Hz`) but current prebuilt
        environment lacks `usdrecord`, `usdview`, `usdcat`, `ffmpeg`,
        `pxr.UsdAppUtils`, and `pxr.UsdImagingGL`. Direct USD rasterization is
        therefore blocked unless an approved prebuilt render path is provided.
      - Real scene-camera probe now exists:
        `p00_scene_cam_v3_20260701_043330` uses official Newton main
        `SensorTiledCamera` on the Panda hydro model to render nonblank
        head/right-wrist/left-wrist scene frames, AVI, and contact sheet.
        This is the active replacement path for the schematic scene panel.
      - Fused scene+tactile diagnostic now exists:
        `p00_fused_cam_v1_20260701_043900` fuses official Newton main
        `SensorTiledCamera` head/right-wrist/left-wrist scene frames with
        calibrated `Fn`, shear-vector, deformation tactile maps and mechanics
        curves in one synchronized diagnostic AVI/contact sheet. This replaces
        the schematic scene panel for active Phase 00 visual evidence.
      - Steel-spec direct-force reference comparison now exists:
        `p00_refcmp_v3_20260701_065300` compares the user reference MP4 against
        `p00_mjw_direct_steel_v1_20260701_060500`. It confirms both videos are
        nonblank and creates a side-by-side sheet. It also records the active
        gaps: gel/marker-style tactile camera rendering, validated marker/
        deformation tracking, contact-normal/contact-area overlays in the same
        direct-force video, and final Gate 00D/00E review.
      - Normal/area overlay diagnostic now exists:
        `p00_mjw_normarea_v1_20260701_071900` adds contact-normal overlays from
        MJWarp `contact.frame` and contact-area proxy overlays from pad-object
        point-contact density to the same direct-force video. The follow-up
        reference comparison `p00_refcmp_normarea_v2_20260701_073000` records
        these as current candidate channels rather than missing channels.
      - Candidate marker-render diagnostic now exists:
        `p00_mjw_marker_v1_20260701_074200` adds a blue gel-like marker/
        deformation panel derived from candidate `Fn`, `Ft`, normal, and
        contact-area-proxy fields. `p00_refcmp_marker_v1_20260701_074900`
        records this as a current candidate channel, while keeping
        photometric/semantic validation open.
      - Strict Gate 00D/00E review now exists:
        `p00_gate_review_v2_20260701_080800` passes all current evidence
        checks but keeps Gate 00D/00E open. It explicitly sets
        `curiosity_training_allowed=false`.
      - Channel semantic layout audit now exists:
        `p00_chan_audit_v1_20260701_082100` audits the reference and candidate
        videos by channel layout. `p00_gate_review_v3_20260701_082600` consumes
        it and marks `channel_semantic_layout_audit` as passed, while keeping
        validated semantic equivalence open.
- [ ] Verify nonblank visual frames and nonzero tactile/mechanics fields.
      - Verified so far: USD visual asset is nonempty; tactile source arrays
        have nonzero hydro face count, nonzero left/right pressure maps, and
        170 active left/right tactile frames.
      - Verified synchronized diagnostic: `sync_scene_tactile.avi` is
        `1180 x 700`, `30 fps`; `sync_scene_tactile_sheet.jpg` manually shows
        scene/tactile/metric panels advancing together.
      - Verified base mechanics diagnostic:
        `p00_base_mech_20260701_030544/sync_scene_tactile.avi` is
        `1180 x 760`, `30 fps`; the sheet is `2360 x 3040` and manually shows
        synchronized scene, tactile, object-z, contact-area, `Fn`, stress,
        tangential-capacity, and shear-motion panels.
      - Verified steel-spec diagnostic:
        `p00_steel_v1_20260701_032709/sync_scene_tactile.avi` and contact sheet
        are nonblank and synchronized; observed material arrays match the
        requested steel-spec candidate override.
      - Verified grid-tactile diagnostic:
        `p00_grid_v1_20260701_033556/sync_scene_tactile.avi` is
        `1180 x 940`, `30 fps`; contact sheet is `2360 x 3760`; manual
        inspection shows synchronized scene, grid `Fn`, shear-vector,
        deformation maps, and mechanics curves. Tactile response is more
        informative than the raw sparse pressure map, but contact remains
        lower-pad concentrated.
      - Verified F6 proxy diagnostic:
        `p00_f6_v1_20260701_034033/sync_scene_tactile.avi` is `1180 x 940`,
        `30 fps`; contact sheet is `2360 x 3760`; source arrays include
        nonzero left/right F6 proxy norms while visual panels remain nonblank
        and synchronized.
      - Verified calibrated-view diagnostic:
        `p00_calib_view_v1_20260701_040715/sync_scene_tactile.avi` is
        `1180 x 940`, `30 fps`; contact sheet is `2360 x 3760`; source arrays
        include raw maps plus calibrated-view maps. Max raw Fn nonzero cell
        ratio was `0.03515625`; max calibrated-view Fn nonzero cell ratio was
        `0.236328125` for both pads.
      - Verified latest-main official USD scene evidence:
        `p00_main_usd_v1_20260701_041900/panda_hydro.usd` exists and is
        `6903124` bytes; the official Newton example exited 0 with no
        traceback under Slurm job `160324` on `server30`.
      - Verified USD capability probe:
        `p00_usd_probe_v2_20260701_042430` opened the USD stage and recorded
        missing raster dependencies. `p00_usd_probe_v1_20260701_042300` is
        invalid for conclusion because it imported `newton.viewer` from the
        older `external/newton` path before `PYTHONPATH` was corrected.
      - Verified real scene-camera probe:
        `p00_scene_cam_v3_20260701_043330/scene_camera.avi` is `768 x 308`,
        `12 fps`; `scene_camera_sheet.jpg` is `2220 x 1160`; manual inspection
        shows nonblank synchronized Panda/hand/object views. Summary reports
        `pixel_std=97.0467646595397` and object lift
        `0.19237708300352097` m.
      - Verified fused scene+tactile diagnostic:
        `p00_fused_cam_v1_20260701_043900/sync_scene_tactile.avi` is
        `1180 x 940`, `30 fps`; contact sheet is `2360 x 3760`; summary
        reports `scene_camera_nonblank=true`,
        `scene_camera_pixel_std=96.05477790619898`, lift success,
        max object lift `0.22351396083831787` m, max `hydro_proxy.Fn`
        `22550.27734375`, and calibrated Fn cell ratio `0.2470703125` for
        both pads.
      - Verified reference comparison:
        `p00_refcmp_v3_20260701_065300` decoded the reference MP4
        (`720` frames, `30 FPS`, `2846x1510`) and candidate steel direct-force
        AVI (`240` frames, `30 FPS`, `1180x820`). Both are nonblank. The
        reference-vs-candidate sheet shows the current candidate is closer than
        proxy-only assets but still lacks reference-level gel/marker tactile
        richness and channel overlays.
      - Verified normal/area overlay diagnostic:
        `p00_mjw_normarea_v1_20260701_071900` passed official final test,
        recorded zero read errors, `147` pad-object contact frames, max
        candidate `Fn` sum `40.0997428894043`, max candidate `Ft` sum
        `12.027881622314453`, left/right contact-area proxy cell ratios
        `0.2900390625` / `0.279296875`, and left/right normal-yz norm maxima
        `9.213287353515625` / `8.88884162902832`.
      - Verified candidate marker-render diagnostic:
        `p00_mjw_marker_v1_20260701_074200` passed official final test,
        recorded zero read errors, `146` pad-object contact frames, max object
        lift `0.2225111573934555` m, max candidate `Fn` sum
        `41.90861511230469`, max candidate `Ft` sum `12.294239044189453`, and
        nonzero left/right marker-flow norms `4.690944671630859` /
        `3.1349213123321533`.
      - Verified strict gate review:
        `p00_gate_review_v2_20260701_080800` passed checks for runtime,
        grasp/lift, steel material, candidate direct `Fn`/`Ft`,
        `SensorContact` alignment, normal/area overlay, candidate marker
        render, and reference-comparison assets. Gate 00D remains
        `open_reference_semantics_blocked`; Gate 00E remains
        `open_tactile_validation_blocked`.
      - Verified channel semantic audit and updated gate review:
        `p00_chan_audit_v1_20260701_082100` passed all layout checks with no
        failed checks. `p00_gate_review_v3_20260701_082600` now has passed
        checks including `channel_semantic_layout_audit`. Remaining hard
        blockers are validated gel/marker photometric semantics, validated
        deformation-marker tracking, validated real contact-area semantics, and
        validated channel-level semantic equivalence beyond layout audit.
      - Remaining: full scene rollout inspection, direct force/shear field
        validation inside the active hydro path, gel/marker-style tactile
        photometric validation beyond the current candidate rendering,
        validated real contact-area semantics beyond the current proxy, and
        final reference-video-level tactile density.
- [ ] Validate candidate tactile semantics against official references.
      - UniVTAC local audit:
        `external/UniVTAC` at
        `05bcd3edb92237107efa40105292a24f1a9fd761` provides the target
        left/right tactile schema with `rgb`, `rgb_marker`, `depth`, `marker`,
        and tactile pose, plus ACT/ViTAL modality baselines.
      - TaCauchy local audit:
        `external/TaCauchy` at
        `c228cfe9050904cd5d71d64f6eb5104768d4cbda` provides the target
        physical semantics for Cauchy stress, normal pressure, tangential
        traction, mesh refinement, force-field visualization, and tactile RGB.
      - Current candidate `marker_flow`, `area_proxy`, and layout audit are
        not enough to close the gate; they must either pass this official
        semantic mapping or be recorded as a faithful blocker.
      - Gate review v4 consumes the reference matrix and official sanity
        blockers. Passed checks now include `semantic_reference_matrix_available`;
        failed checks are `univtac_official_reference_sanity` and
        `tacauchy_official_reference_sanity`.
      - Detailed bridge spec:
        `experiments/configs/phase00/ref_tactile/semantic_bridge_spec_v1.json`
        and
        `experiments/reports/phase00/ref_tactile/semantic_bridge_spec.md`.
        This maps candidate `Fn`, `Ft`, marker flow, area proxy, contact
        normal, and scene RGB to official UniVTAC/TaCauchy target semantics.
      - Gate review code now consumes the bridge spec through
        `--semantic-bridge-spec` and checks `semantic_bridge_spec_available`.
        Future Gate 00F reviews cannot pass from the reference matrix alone.
      - Compute verification:
        `p00_gate_review_v5_20260701_060100` ran in Curiosity Slurm job
        `160454` on `server02`, passed `semantic_bridge_spec_available`, and
        still failed `univtac_official_reference_sanity` and
        `tacauchy_official_reference_sanity`.
- [ ] Export MP4 plus contact sheet and metadata.
      - Current cluster environment has no `ffmpeg`, imageio, cv2, or av
        encoder. Exported AVI plus dense PPM frame sequence instead:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_hydro_tac_avi3_20260701_024826/tactile_maps_enhanced.avi`.
      - Synchronized diagnostic exported AVI plus contact sheet:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_sync_hydro_20260701_025818/sync_scene_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_sync_hydro_20260701_025818/sync_scene_tactile_sheet.jpg`.
      - Base mechanics diagnostic exported AVI plus contact sheet:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_base_mech_20260701_030544/sync_scene_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_base_mech_20260701_030544/sync_scene_tactile_sheet.jpg`.
      - Steel-spec diagnostic exported AVI plus contact sheet:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_steel_v1_20260701_032709/sync_scene_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_steel_v1_20260701_032709/sync_scene_tactile_sheet.jpg`.
      - Grid-tactile diagnostic exported AVI plus contact sheet:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_grid_v1_20260701_033556/sync_scene_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_grid_v1_20260701_033556/sync_scene_tactile_sheet.jpg`.
      - F6 proxy diagnostic exported AVI plus contact sheet:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_f6_v1_20260701_034033/sync_scene_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_f6_v1_20260701_034033/sync_scene_tactile_sheet.jpg`.
      - Calibrated-view diagnostic exported AVI plus contact sheet:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_calib_view_v1_20260701_040715/sync_scene_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_calib_view_v1_20260701_040715/sync_scene_tactile_sheet.jpg`.
      - Latest-main official scene/USD export:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_main_usd_v1_20260701_041900/panda_hydro.usd`.
        Keep rasterized scene+tactile fusion open until a faithful prebuilt USD
        render/viewer path is confirmed inside the held compute allocation.
      - Real scene-camera probe exported AVI, sheet, and PNG frames:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_scene_cam_v3_20260701_043330/scene_camera.avi`,
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_scene_cam_v3_20260701_043330/scene_camera_sheet.jpg`,
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_scene_cam_v3_20260701_043330/frames/`.
      - Fused scene+tactile diagnostic exported AVI plus contact sheet:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_fused_cam_v1_20260701_043900/sync_scene_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_fused_cam_v1_20260701_043900/sync_scene_tactile_sheet.jpg`.
        Source arrays:
        `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_fused_cam_v1_20260701_043900/sync_hydro_timeseries.npz`.
      - Reference comparison exported sheets and metadata:
        `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_vs_candidate_sheet.jpg`,
        `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_sheet.jpg`,
        `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/candidate_sheet.jpg`,
        and
        `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_video_compare_summary.json`.
      - Normal/area overlay diagnostic exported AVI, sheet, source arrays, and
        reference comparison:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile.avi`,
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile_sheet.jpg`,
        `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile_timeseries.npz`,
        and
        `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_normarea_v2_20260701_073000/reference_vs_candidate_sheet.jpg`.
      - Candidate marker-render diagnostic exported AVI, sheet, source arrays,
        and reference comparison:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile.avi`,
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_sheet.jpg`,
        `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_timeseries.npz`,
        and
        `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_marker_v1_20260701_074900/reference_vs_candidate_sheet.jpg`.
      - Gate review exported summary and report:
        `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_review_v2_20260701_080800/phase00_gate_review_summary.json`
        and
        `experiments/reports/phase00/ref_tactile/gate_review/p00_gate_review_v2_20260701_080800/phase00_gate_review.md`.
      - Channel semantic audit and updated gate review exported:
        `experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit_summary.json`,
        `experiments/visuals/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit_sheet.jpg`,
        `experiments/reports/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit.md`,
        `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_review_v3_20260701_082600/phase00_gate_review_summary.json`,
        and
        `experiments/reports/phase00/ref_tactile/gate_review/p00_gate_review_v3_20260701_082600/phase00_gate_review.md`.
      - Keep MP4 export open until an approved prebuilt encoder path exists.
- [ ] Manually inspect the diagnostic and record pass/fail.
      - Manual sheet inspection: enhanced tactile sheet shows persistent
        lower-pad contact patches, but tactile distribution is sparse and not
        reference-video-level rich.
      - Manual synchronized sheet inspection: scene schematic, tactile maps,
        object-z, contact-area, `Fn` proxy, and shear-motion proxy are visibly
        synchronized; still not reference-level because tactile patches are
        sparse and the scene panel is schematic.
      - Manual base mechanics sheet inspection: stress and tangential-capacity
        proxy curves are synchronized with grasp/lift/contact changes; tactile
        patches still remain sparse and lower-edge concentrated.
      - Manual steel-spec sheet inspection: stress and tangential-capacity
        proxy curves respond to grasp/lift; tactile patches still remain sparse
        and lower-edge concentrated.
      - Manual grid-tactile sheet inspection: six tactile panels are nonblank
        during contact and show `Fn`, shear-vector, and deformation changes
        synchronized with lift/contact curves; the evidence is stronger than
        scalar contact or raw pressure alone, but still not direct `Ft` or a
        reference-grade tactile sensor rendering.
      - Manual F6 proxy sheet inspection: visualization remains synchronized
        and nonblank; F6 evidence is in source arrays and summary, not a visual
        success claim.
      - Manual calibrated-view sheet inspection: calibrated Fn/shear/deform
        panels are substantially easier to read and remain synchronized with
        object/contact curves. This is still not direct `Ft`, photoreal scene
        fusion, or validated gel tactile rendering.
      - Latest-main official USD inspection is pending raster/viewer support:
        the USD file is valid nonempty official scene evidence, but without a
        confirmed `usdrecord`/viewer path it is not yet a manually inspected
        frame/video asset.
      - Manual real scene-camera sheet inspection:
        `p00_scene_cam_v3_20260701_043330` shows nonblank Panda, gripper,
        object, and tabletop views from the head/right-wrist/left-wrist
        cameras. It is a real rendered scene-frame path, but not yet fused with
        calibrated tactile/mechanics panels.
      - Manual fused scene+tactile sheet inspection:
        `p00_fused_cam_v1_20260701_043900` shows real three-camera scene
        frames, calibrated `Fn` maps, shear-vector maps, deformation maps, and
        mechanics curves advancing together. This is still not final
        reference-video equivalence because direct `Ft`, direct pad shear
        force, and validated gel/marker tactile rendering remain missing.
      - Manual reference comparison inspection:
        `p00_refcmp_v3_20260701_065300/reference_vs_candidate_sheet.jpg`
        shows the reference video has richer multi-column tactile diagnostics,
        blue tactile heatmaps, vector/line overlays, and several time-series
        curves per panel. The candidate has real Newton scene views, left/right
        candidate `Fn`/`Ft` maps, shear arrows, and object-z/force curves, but
        remains less dense and lacks gel/marker visual tactile channels plus
        direct contact-normal/contact-area overlays.
      - Manual normal/area overlay inspection:
        `p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile_sheet.jpg`
        shows real scene views, `Fn + normal`, `Ft vector`, and
        `contact area proxy + normal` panels for both pads. The panels become
        nonblank during the grasp/lift window and remain synchronized with
        object-z and force curves. `p00_refcmp_normarea_v2_20260701_073000`
        still shows the reference has richer gel/marker-style tactile panels
        and denser mechanics overlays.
      - Manual candidate marker-render inspection:
        `p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_sheet.jpg`
        shows blue gel-like marker panels with a regular marker grid. During
        the contact/lift window, the marker panels develop centered
        deformation/flow synchronized with `Fn`, `Ft`, area proxy, object-z,
        and force curves. `p00_refcmp_marker_v1_20260701_074900` shows this is
        visually closer to the reference, but still not validated
        photometric/deformation marker semantics.

## Base Model / Controller

- [x] Select the base grasp controller/model after official sanity results:
      official Newton Panda hydro prior first unless evidence supports a better
      serious base.
      - Current selected base: official Newton
        `newton.examples.robot.example_robot_panda_hydro`.
      - Positive evidence:
        `p00_newton_panda_hydro_20260701_022557` exited 0 on held H200 Slurm
        job `160324`.
      - Positive visual evidence:
        `p00_panda_usd_20260701_023155` produced a 6.9 MB USD rollout asset
        from the same official hydro example.
      - This is only base-grasp/hydro-mechanics evidence; it is not dense
        tactile success or curiosity success.
- [ ] Make the base controller produce the dense tactile/mechanics schema.
      - Partial: official Newton Panda hydro now exports hydro-derived
        left/right tactile maps and object lift/contact metrics.
      - Missing: `Fn`, `Ft`, shear direction/vector fields, contact area
        calibration, and synchronized scene+tactile video in one rollout.
      - New partial: `p00_sync_hydro_20260701_025818` exports contact area,
        `hydro_proxy.Fn`, and `hydro_proxy.shear_motion` from Newton hydro
        reducer buffers. Direct solver `Ft` and pad-resolved shear vectors are
        still missing.
      - Stronger partial: `p00_base_mech_20260701_030544` exports
        `hydro_proxy.stress`, force-weighted contact normal, `Ft_capacity`
        proxy, force balance, hold/drop/slip/safety metrics, and source arrays.
        Direct solver `Ft`, pad-resolved shear vector fields, steel-calibrated
        material settings, and dense tactile richness are still missing.
      - Direct-force probe `p00_force_probe_20260701_032310` attempted to use
        `Contacts.force` plus `SolverMuJoCo.update_contacts()` on the official
        Panda hydro Newton-contacts path. It failed with CUDA illegal memory
        access and produced no valid force arrays. Treat direct `Ft` as a
        blocker until a faithful official force path is found.
      - MJWarp EFC array audit:
        `p00_mjw_force_audit_v1_20260701_045000` showed nonzero official
        `mjw_data.efc.force` normal/tangent arrays but no pad-object force in
        the first `90` frames. The longer
        `p00_mjw_force_audit_v2_20260701_045700` showed nonzero pad-object EFC
        force for `128` of `240` frames, max pad-object EFC abs sum
        `253.05938720703125`, max pad-object tangent EFC abs sum
        `141.14100646972656`, and zero read errors. This is a candidate
        direct-force path, not final tactile success; it must be validated
        against official `SensorContact` on a compatible MuJoCo-contact scene
        before promotion.
      - Candidate direct-force tactile export:
        `p00_mjw_direct_v1_20260701_052900` maps audited MJWarp EFC
        normal/tangent force into left/right pad-local dense `Fn`/`Ft` maps
        and renders a synchronized `SensorTiledCamera` scene + tactile AVI.
        It passed the official 240-frame Panda hydro final test, had `127`
        pad-object contact frames, max pad-object candidate `Fn` sum
        `48.28089141845703`, max pad-object candidate `Ft` sum
        `48.28089141845703`, max left/right candidate `Fn` maps
        `13.648624420166016` / `11.802962303161621`, and max left/right
        candidate `Fn` nonzero cell ratios `0.3154296875` / `0.31640625`.
        Manual sheet inspection found nonblank scene-camera panels and
        synchronized candidate force heatmaps/arrows after contact begins.
      - Candidate/SensorContact alignment:
        `p00_mjw_align_v1_20260701_055200` validates the candidate MJWarp EFC
        frame mapping against official `SensorContact.force_matrix` and
        `force_matrix_friction` on the compatible MuJoCo-contact Panda variant.
        Best sign is `shape0_negative` for both force and friction; force
        relative RMSE is `3.2491620810680347e-08`, friction relative RMSE is
        `2.0018143688320552e-07`, both mean cosine values are `1.0`, and update
        errors are `0`. This validates the mapping on the compatible scene.
        Still need active hydro steel-spec merge and reference tactile
        comparison before final tactile gate completion.
      - Steel-spec validated-sign direct-force tactile export:
        `p00_mjw_direct_steel_v1_20260701_060500` applies steel-spec material
        override (`mu=0.3`, `kh=1e12`), records `material_notify_status=pass`,
        uses the validated `shape0_negative` force sign, passes the official
        240-frame Panda hydro final test, and exports real scene-camera +
        candidate direct `Fn`/`Ft` maps. It has `146` pad-object contact
        frames, max object lift `0.2225421965122223` m, max pad-object
        candidate `Fn` sum `40.099632263183594`, max pad-object candidate `Ft`
        sum `12.027974128723145`, and read errors `0`.
      - Steel-spec partial: `p00_steel_v1_20260701_032709` verifies the
        requested `mu=0.3`, `kh=1e12` candidate material arrays and exports the
        same synchronized mechanics schema. Direct `Ft` and pad shear vectors
        remain blocked/missing.
      - Grid-tactile partial: `p00_grid_v1_20260701_033556` exports dense
        grid source arrays for `Fn`, stress, deformation, shear-vector, and
        shear-magnitude maps. Direct `Ft` remains blocked, and the shear vector
        is explicitly only a `hydro_proxy` contact-center-motion field.
      - F6 proxy partial: `p00_f6_v1_20260701_034033` exports
        `left_f6_normal_proxy`, `right_f6_normal_proxy`,
        `left_f6_ft_capacity_proxy`, `right_f6_ft_capacity_proxy`,
        `left_f6_combined_proxy`, and `right_f6_combined_proxy`. These arrays
        are only a schema bridge for later T-Rex-style conversion.
      - Direct-force comparison partial:
        `p00_mjc_sensor_v1_20260701_034541` exports official
        `SensorContact.total_force`, `total_force_friction`, `force_matrix`,
        and `force_matrix_friction` on the MuJoCo-contact variant. The active
        hydro base still lacks direct `Ft`.
- [ ] Evaluate grasp/lift/hold with full diagnostic output.
      - Partial positive: max object lift is `0.22351960837841034` m in
        `p00_hydro_tac_avi3_20260701_024826`.
      - Partial positive: max object lift is `0.2235533893108368` m in
        `p00_sync_hydro_20260701_025818`, with max contact area
        `0.0034376755356788635` m^2 and max `hydro_proxy.Fn`
        `2931.3955078125`.
      - Stronger partial positive: `p00_base_mech_20260701_030544` has lift
        success over `0.15` m, first lift frame `169`, `71` hold frames above
        threshold, no detected drop after lift, max object lift
        `0.22364932298660278` m, max `hydro_proxy.Fn`
        `2936.611083984375`, max stress proxy `1489732.5`, max
        tangential-capacity proxy `2936.611083984375`, and max object
        acceleration `1.310336709022522` m/s^2.
      - Official runtime benchmark `p00_hydro_bench_20260701_030813` measured
        `67.5 FPS` on H200 with Newton null viewer, meeting the `60 FPS`
        minimum but not the `82 FPS` target.
      - Hot-cache official runtime benchmark `p00_bench_hot_20260701_034952`
        measured `79.2 FPS` over `30` seconds on H200 with Newton null viewer,
        still below the `82 FPS` target.
      - Longer official runtime benchmark `p00_bench_60_20260701_035208`
        measured `79.1 FPS` over `60` seconds, confirming the current faithful
        official hydro base is still below the `82 FPS` target.
      - Latest-main official runtime benchmark
        `p00_bench_main_20260701_035529` measured `92.6 FPS` on H200 with
        Newton main commit `a217e55fab3d373a08fba374cc5cafc1826cf27f`,
        meeting the `82 FPS` target.
      - Latest-main tactile diagnostic `p00_main_f6_v1_20260701_035926` passed
        with steel-spec material override, grid tactile maps, F6 proxy arrays,
        lift/hold metrics, AVI, and contact sheet.
      - Calibrated-view tactile diagnostic `p00_calib_view_v1_20260701_040715`
        passed on Newton main with steel-spec material override. It preserved
        lift success, `71` hold frames, no detected post-lift drop, and raised
        max Fn nonzero cell ratio from `0.03515625` raw to `0.236328125` in
        the calibrated visualization view.
      - Steel-spec candidate `p00_steel_v1_20260701_032709` has lift success,
        `71` hold frames above threshold, no detected post-lift drop, observed
        `mu=0.30000001192092896`, observed `kh=999999995904.0`, max object
        lift `0.2235182225704193` m, max `hydro_proxy.Fn`
        `22572.54296875`, max stress proxy `6979607.5`, and max
        tangential-capacity proxy `6771.763671875`.
      - Grid-tactile candidate `p00_grid_v1_20260701_033556` has lift success,
        `71` hold frames above threshold, observed steel-spec material arrays,
        max object lift `0.2234652042388916` m, max `hydro_proxy.Fn`
        `22551.130859375`, max left/right grid `Fn` maps
        `3512.72900390625` / `1440.114013671875`, and max left/right grid
        shear-magnitude maps `165.96604919433594` /
        `42.3233757019043`.
      - F6 proxy candidate `p00_f6_v1_20260701_034033` has lift success,
        `71` hold frames above threshold, no detected post-lift drop, max
        object lift `0.2235504388809204` m, max `hydro_proxy.Fn`
        `22694.48046875`, max left/right F6 combined proxy norms
        `1477.2451171875` / `2114.81494140625`.
      - Direct-force comparison candidate `p00_mjc_sensor_v1_20260701_034541`
        has max total force norm `29.69374656677246`, max total friction norm
        `8.532435417175293`, max matrix friction norm `4.5348076820373535`,
        lift success over `0.15` m, and max object lift
        `0.21197126805782318` m.
      - MJWarp direct-force array audit `p00_mjw_force_audit_v2_20260701_045700`
        has max EFC abs sum `500.2020568847656`, max tangent EFC abs sum
        `272.9135437011719`, max pad-object contact count `66`, max
        pad-object EFC abs sum `253.05938720703125`, and max pad-object
        tangent EFC abs sum `141.14100646972656`. This supports building a
        candidate direct MJWarp force exporter, but still does not close the
        dense tactile gate.
      - Candidate direct-force visual export `p00_mjw_direct_v1_20260701_052900`
        produced:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_v1_20260701_052900/candidate_mjw_direct_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_v1_20260701_052900/candidate_mjw_direct_tactile_sheet.jpg`.
      - Candidate/SensorContact alignment
        `p00_mjw_align_v1_20260701_055200` produced:
        `experiments/outputs/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment_timeseries.npz`
        and
        `experiments/outputs/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment_summary.json`.
      - Steel-spec validated-sign direct-force visual export
        `p00_mjw_direct_steel_v1_20260701_060500` produced:
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile.avi`
        and
        `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile_sheet.jpg`.
        Manual inspection confirms nonblank scene views and synchronized
        candidate `Fn`/`Ft` heatmaps/arrows.
      - Reference comparison `p00_refcmp_v3_20260701_065300` produced:
        `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_vs_candidate_sheet.jpg`
        and
        `experiments/reports/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_video_compare.md`.
        It confirms the candidate direct-force video is nonblank and aligned
        enough to compare against the reference, but still below the
        reference's tactile density and gel/marker channel richness.
      - Normal/area overlay export `p00_mjw_normarea_v1_20260701_071900`
        produced direct-force video panels with contact-normal and area-proxy
        overlays. Reference comparison
        `p00_refcmp_normarea_v2_20260701_073000` now lists normal overlay and
        area-proxy overlay as candidate channels. The remaining base-gate
        issue is not absence of overlays; it is validated real area semantics,
        gel/marker tactile rendering, channel-level semantic match, and final
        Gate 00D/00E review.
      - Candidate marker-render export `p00_mjw_marker_v1_20260701_074200`
        produced blue gel-like marker/deformation panels derived from
        direct-force fields. Reference comparison
        `p00_refcmp_marker_v1_20260701_074900` now lists candidate
        gel/marker-style rendering as a current channel. Still need validated
        photometric/deformation marker semantics, real contact-area semantics,
        channel-level semantic match, and final Gate 00D/00E review before
        declaring base model gate complete.
      - Gate review `p00_gate_review_v2_20260701_080800` completed the current
        formal review. It has no failed evidence checks, but it classifies Gate
        00D/00E as open because candidate force-derived tactile visuals do not
        validate photometric marker semantics or real contact-area semantics.
        Next action is to solve one of those semantic validation gaps, not to
        start curiosity training.
      - Channel semantic audit `p00_chan_audit_v1_20260701_082100` and updated
        gate review `p00_gate_review_v3_20260701_082600` narrow the remaining
        channel blocker: channel layout audit exists, but validated
        channel-level semantic equivalence is still missing. Next action should
        target real tactile semantics using official/reference methods such as
        Taccel, UniVTAC, Tacmap, TaCauchy, or ControlTac after official sanity.
- [ ] Declare baselines for later curiosity: no adaptation, scripted tactile
      feedback, no-curiosity learned residual, and serious reference methods or
      documented blockers.
- [x] Add and run a repeatable Gate 00F readiness checker.
      - Script:
        `experiments/configs/phase00/ref_tactile/envprep/check_gate00f_readiness.sh`.
      - Output:
        `experiments/outputs/phase00/ref_tactile/envprep/gate00f_readiness/gate00f_readiness_status.json`
        and
        `experiments/reports/phase00/ref_tactile/envprep/gate00f_readiness.md`.
      - Current result: `gate00f_ready=false`,
        `reason=blocked_official_sanity_or_gate_review_not_passed`.
      - Observed positives: project-local conda exists; project-local CUDA
        12.8 `nvcc` exists; UniVTAC bundled TacEx has `410M` assets,
        `Sensor.usd`, and `21` test shape USDs; UniVTAC and TaCauchy base
        env pythons are now present.
      - Still missing: official UniVTAC/TaCauchy dependency readiness,
        official UniVTAC/TaCauchy sanity, `git-lfs`, executable `cmake`,
        default-PATH `nvcc`, and `nvidia-smi`.
      - No longer the current file-presence blocker after approved asset reuse:
        TaCauchy `Sensor.usd` is present and TaCauchy tactile test shape USD
        count is `21`.
- [x] Execute approved UniVTAC bundled TacEx asset reuse into TaCauchy.
      - Approval: user said `全都允许继续`.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/envprep/approved_asset_reuse_execution.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/approved_asset_reuse_execution_v1.json`.
      - Result: created `273` files, transferred `429244198` bytes, target
        asset tree is now `412M`, TaCauchy `Sensors/GelSight_Mini/Sensor.usd`
        is present, and TaCauchy tactile test shape USD count is `21`.
      - Remaining blocker: official reference sanity has not passed, and a
        fresh Gate review has not consumed the post-copy asset/env
        availability.
- [x] Create approved base reference env prefixes and record evidence.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/envprep/reference_env_create_execution.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/reference_env_create_execution_v1.json`.
      - Result: UniVTAC base env Python exists at
        `/public/home/yanhongru/Curiosity/envs/univtac/conda/bin/python`
        (`Python 3.10.20`, `140M`); TaCauchy base env Python exists at
        `/public/home/yanhongru/Curiosity/envs/tacauchy/conda/bin/python`
        (`Python 3.11.15`, `166M`).
      - Not a claim: this is not official dependency installation, official
        sanity, Gate 00F completion, or curiosity readiness.
- [x] Generate dry-run official dependency stage commands for UniVTAC/TaCauchy.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/envprep/reference_dependency_stage_plan.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/reference_dependency_stage_plan_v1.json`.
      - Result: dry-run command files and reports exist for `install_isaac`,
        `install_isaaclab`, `install_curobo_or_assets`, `install_tacex_core`,
        `build_uipc`, `setup_assets`, and `official_sanity` for both UniVTAC
        and TaCauchy.
      - Not a claim: no official dependency installation or sanity was run.
- [x] Record official dependency install location blocker.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/envprep/reference_dependency_install_blocker.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/reference_dependency_install_blocker_v1.json`.
      - Result: base envs and dry-run commands exist, but official readiness
        requires heavy Isaac/TacEx/UIPC dependency installation or builds.
        Current project rules forbid heavy work on login nodes and dependency
        installation/builds on compute nodes.
      - Needed to continue: approved prebuilt Curiosity reference envs or a
        compliant non-login env-prep workflow.
- [x] Record Gate 00F dependency resolution packet.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_dependency_resolution_packet.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_dependency_resolution_packet_v1.json`.
      - Result: official UniVTAC, TaCauchy/TacEx/UIPC, and IsaacLab TacSL
        dependency requirements are now explicit, along with allowed
        env/container resolution paths and disallowed login-node or
        compute-allocation install/build paths.
      - Next action: locate or prepare dependency-complete reference envs or
        prebuilt containers without violating the cluster safety rules, then
        run Gate 00F bundle plus strict acceptance checker.
- [x] Record refreshed Gate 00F runtime locator probe.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_locator_probe_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_runtime_locator_probe_20260701_v1.json`.
      - Result: shell path exposes `/usr/bin/docker` but no `module`/`ml`,
        `singularity`/`apptainer`/`enroot`, `git-lfs`, `cmake`, `nvcc`, or
        `nvidia-smi`; approved shallow checks found no dependency-complete
        UniVTAC/TaCauchy/IsaacLab TacSL runtime.
      - Gate effect: does not clear Gate 00F.
- [x] Record shared runtime locator and runtime preflight handoff.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_shared_runtime_locator_20260701.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_shared_runtime_locator_20260701_v1.json`,
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_handoff.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_runtime_preflight_handoff_v1.json`,
        and
        `experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`.
      - Result: no existing shared Isaac/TacEx/TaCauchy/UniVTAC/TacSL/UIPC
        runtime or Docker image was found in the lightweight locator. A future
        compute-side preflight now exists to require registry acceptance and
        then check Python executability and module specs before running the
        Gate 00F bundle.
      - Safety check:
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_20260701.md`
        confirms the preflight refuses login-node execution without
        `SLURM_JOB_ID`.
      - Gate effect: does not clear Gate 00F.
- [x] Add Gate 00F runtime registry and current validation.
      - Evidence:
        `experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json`,
        `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py`,
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_handoff.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_runtime_registry_handoff_v1.json`,
        and
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_current_20260701.md`.
      - Result: current validation status is `fail_gate00f_runtime_registry`
        because UniVTAC/TaCauchy are only base Python envs and IsaacLab TacSL
        has no registered runtime.
      - Required order after a real dependency solution: update registry,
        validate registry, run runtime preflight, run Gate 00F reference
        bundle, then run strict bundle acceptance.
- [x] Add controlled Gate 00F runtime registration handoff.
      - Evidence:
        `src/newton_tactile_curiosity/gate00f_runtime_register.py`,
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_registration_handoff.md`,
        and
        `experiments/configs/phase00/ref_tactile/gate00f_runtime_registration_handoff_v1.json`.
      - Result: future dependency-complete Python envs, local Docker image IDs,
        or shared container artifacts can be written into a copied candidate
        registry without ad hoc active-registry edits.
      - Container guard: container registrations must include a matching
        `pass_gate00f_container_provenance` summary.
      - Gate effect: metadata-only. It does not pull/build images, run
        containers, import Isaac/TacSL modules, install dependencies, or clear
        Gate 00F.
- [x] Lock the post-8c501 Gate 00F runtime acceptance path.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff_v1.json`,
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_handoff.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_runtime_preflight_handoff_v1.json`,
        and
        `experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`.
      - Result: future Gate 00F attempts must reuse the latest 8c501
        candidate chain and advance through runtime registration, registry
        validation, runtime preflight, bundle, and strict acceptance. Runtime
        preflight now reads registered `python_env` paths from the accepted
        registry instead of default shell paths. Container module preflight now
        supports registered docker local image IDs and
        singularity/apptainer/sif artifact paths; enroot, sqsh, and tar still
        require explicit runners.
      - Gate effect: handoff/guard only. It does not clear Gate 00F or allow
        curiosity training.
- [x] Record bounded project-local artifact probe for Gate 00F runtimes.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_project_artifact_probe_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_project_artifact_probe_20260701_v1.json`.
      - Result: no `.sif`, `.sqsh`, `.tar`, `.tar.gz`, or `.img` container
        artifact was found under scoped project paths at max depth `5`; no
        `cmake`, `git-lfs`, `singularity`, `apptainer`, or `docker` file was
        found under `envs` at max depth `4`; only
        `envs/taccel/cuda-toolkit/bin/nvcc` was found.
      - Gate effect: blocker evidence only. No runtime is registered.
- [x] Gate the Gate 00F bundle behind runtime preflight.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_bundle_preflight_gate_update_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_bundle_preflight_gate_update_20260701_v1.json`.
      - Result: `run_gate00f_reference_bundle_in_alloc.sh` now runs runtime
        preflight before official reference sanity and exits with
        `fail_gate00f_bundle_runtime_preflight_not_passed` if preflight does
        not pass.
      - Container update: the bundle forwards `RUNTIME_REGISTRY` to runtime
        preflight and to all official sanity sub-scripts. The official sanity
        runners can dispatch registered docker/singularity/apptainer/sif
        runtimes through the shared container helper.
      - Gate effect: does not clear Gate 00F; prevents direct bundle execution
        without registry-gated preflight.
- [x] Add container-aware official sanity dispatchers.
      - Evidence:
        `experiments/configs/phase00/ref_tactile/gate00f_container_runtime_common.sh`,
        `experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh`,
        and
        `experiments/configs/phase00/ref_tactile/run_isaaclab_tacsl_sanity_in_alloc.sh`.
      - Result: UniVTAC, TaCauchy, and IsaacLab TacSL sanity scripts can read
        accepted `RUNTIME_REGISTRY` entries and dispatch registered
        docker/singularity/apptainer/sif runtimes through the shared helper.
      - Failure evidence: UniVTAC/TaCauchy schema probe failures and IsaacLab
        TacSL official demo failures now write blocker summaries instead of
        exiting without evidence; TacSL keeps `--use_tactile_rgb` and records
        runtime/asset failures rather than weakening the command.
      - Gate effect: glue only. A real dependency-complete runtime must still
        be registered, preflighted, and run on compute before Gate 00F can
        pass.
- [x] Recheck login-node refusal after container sanity dispatch support.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_container_dispatch_login_refuse_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_container_dispatch_login_refuse_20260701_v1.json`.
      - Result: UniVTAC/TaCauchy sanity, IsaacLab TacSL sanity, and Gate 00F
        bundle all exit with code `2` before registry/container/module logic
        when `SLURM_JOB_ID` is missing.
      - Gate effect: safety check only.
- [x] Record Gate 00F container acquisition plan.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_container_acquisition_plan_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_container_acquisition_plan_20260701_v1.json`.
      - Result: official Isaac Sim/Isaac Lab container paths exist, with
        `nvcr.io/nvidia/isaac-lab:2.3.2` as a current IsaacLab candidate, but
        TacEx/UniVTAC and TaCauchy still need project image layers over an
        Isaac Lab base image or an existing prebuilt project image.
      - Gate effect: does not clear Gate 00F; defines the faithful container
        acquisition route before registry/preflight/bundle.
- [x] Extend runtime registry for strict container registration.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_container_support_update_20260701.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_runtime_registry_container_support_update_20260701_v1.json`,
        `experiments/reports/phase00/ref_tactile/gate00f_container_runtime_registration_examples.md`,
        and
        `experiments/configs/phase00/ref_tactile/gate00f_container_runtime_registration_examples_v1.json`.
      - Result: `container` targets must include a supported
        `container_runtime` and either local `image_id` or existing shared
        `artifact_path`; remote `image_ref` alone cannot pass registry
        validation. The validators now also reject image IDs that do not look
        like immutable local digests/IDs, image IDs equal to image refs, and
        artifact paths that are missing, directories, or lack `.sif`, `.sqsh`,
        `.tar`, `.tar.gz`, or `.img` suffixes; the runtime registration helper
        applies the same checks before writing a copied registry.
      - Current status: registry still fails because no real container runtime
        is registered.
- [x] Add Gate 00F container provenance contract and negative control.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_container_provenance_contract.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_contract_v1.json`,
        `src/newton_tactile_curiosity/gate00f_container_provenance_validate.py`,
        `experiments/reports/phase00/ref_tactile/gate00f_container_provenance_negative_control_20260701.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_isaaclab_ref_only_20260701_v1.json`,
        and
        `experiments/outputs/phase00/ref_tactile/container_provenance/p00_isaaclab_ref_only_20260701/container_provenance_validation_summary.json`.
      - Result: validator rejects the IsaacLab TacSL remote-image-only packet
        with `fail_gate00f_container_provenance`; a local `image_id` or
        existing `artifact_path` is required before registry registration.
      - Runtime-register connection: future container registry writes now
        require this validator to pass first.
      - Gate effect: guard evidence only. No container was pulled/built/run and
        no runtime is registered.
- [x] Add Gate 00F runtime intake chain.
      - Evidence:
        `src/newton_tactile_curiosity/gate00f_runtime_intake_chain.py`,
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_intake_chain_handoff.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_runtime_intake_chain_handoff_v1.json`,
        and
        `experiments/outputs/phase00/ref_tactile/runtime_intake/p00_isaaclab_ref_only_20260701/runtime_intake_summary.json`.
      - Result: remote-image-only negative control stops at
        `fail_container_provenance` before registry registration and writes no
        candidate registry.
      - Gate effect: metadata-only guard. A future pass would only authorize
        runtime preflight, not Gate 00F completion.
- [x] Add TacSL source compatibility check for the Isaac Lab container
      candidate.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_handoff.md`,
        `experiments/configs/phase00/ref_tactile/gate00f_tacsl_source_compat_handoff_v1.json`,
        `src/newton_tactile_curiosity/gate00f_tacsl_source_compat_validate.py`,
        `experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_current_20260701.md`,
        and
        `experiments/outputs/phase00/ref_tactile/tacsl_source_compat/p00_tacsl_src_compat_20260701/tacsl_source_compat_summary.json`.
      - Result: source check passed for local IsaacLab VERSION `2.3.2`,
        candidate image ref `nvcr.io/nvidia/isaac-lab:2.3.2`, required TacSL
        data fields, demo flags, and imports.
      - Gate effect: source compatibility only. It does not register a
        runtime, run Isaac Sim, import TacSL modules, or clear Gate 00F.
- [x] Refresh official TacSL container/docs and record RGB asset risk.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701_v1.json`.
      - Result: official Isaac Lab docs and NGC catalog support the container
        route, while a public IsaacLab issue and local static check indicate
        `--use_tactile_rgb` can fail if GelSight R15 `bg.jpg` is missing.
      - Gate effect: source/docs risk record only. Do not silently remove
        tactile RGB from the official TacSL sanity path.
- [x] Recheck IsaacLab official upstream freshness.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701_v1.json`.
      - Result: local `external/IsaacLab_official` matches upstream
        `main`/`HEAD` at `b4c321024792976150ca55fddb26fa34480d974e`; visible
        `v3.0.0-beta*` tags are release context, not source staleness for the
        active main checkout.
      - Gate effect: source freshness only. It does not register a runtime or
        clear Gate 00F.
- [x] Recheck UniVTAC, TaCauchy, and TacEx upstream freshness.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701_v1.json`.
      - Result: local UniVTAC, TaCauchy, and TacEx match upstream main.
      - Gate effect: source freshness only. It does not register runtimes, run
        official sanity, or clear Gate 00F.
- [x] Recheck runtime preflight login-node refusal after container support.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_after_container_support_20260701.md`
        and
        `experiments/configs/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_after_container_support_20260701_v1.json`.
      - Result: script exits with code `2` before registry validation,
        container commands, or module imports when `SLURM_JOB_ID` is missing.
      - Gate effect: safety/refuse check only. It does not clear Gate 00F.
- [x] Record approved UniVTAC env create attempts and lock workaround.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/envprep/univtac_env_create_attempts.md`
        and
        `experiments/configs/phase00/ref_tactile/envprep/univtac_env_create_attempts_v1.json`.
      - Result: three local conda create attempts failed with
        `LockError: Failed to acquire lock`; a fourth `--no-lock --solver
        classic` retry succeeded and created the base UniVTAC Python env.
      - Partial artifacts: `envs/conda_pkgs/univtac` (`137M`) and
        `envs/conda_pkgs/univtac_classic` (`170M`) are package caches only,
        not executable reference envs.
      - Next action: do not claim official readiness from the base Python env;
        stage official dependency installation or document blockers, then run
        official sanity on compute.

## Curiosity Readiness

- [ ] Design dense visuo-tactile prediction objective.
      - Must predict dense force/pressure/shear/marker/contact-area/mechanics
        fields, not scalar contact counts.
      - Use APPLE only as a future active-perception sequence/RL reference; do
        not treat it as current Newton training evidence.
- [ ] Design active probing and bounded tactile learning-progress reward.
      - APPLE provides useful future baselines: SAC/CrossQ/PPO, random
        actions, and grid policies.
      - Tactile MNIST provides useful future active-touch task structure:
        tactile-only observations, relative sensor movement, train/test/holdout
        splits, and hidden-object exploration.
- [ ] Design tactile-mask/vision-mask training protocol.
      - Required evaluation modes: vision+tactile, tactile-only masked vision,
        vision-only, and noisy/mismatched tactile.
      - Future curiosity should include random/grid/scripted/no-curiosity
        baselines before any success claim.
- [x] Accept the current official reference runtime blocker for proceeding to
      Newton-only Phase 01 training.
      - Evidence:
        `experiments/reports/phase00/ref_tactile/newton_8c501_cont_chain_status.md`,
        `experiments/configs/phase00/ref_tactile/newton_8c501_cont_chain_status_v1.json`,
        `experiments/reports/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff.md`,
        and the user decision on 2026-07-01 that Newton is sufficient to
        proceed with full training rather than continuing to block on official
        runtime glue.
      - Result: Phase 01 may start as Newton-only dense tactile curiosity
        training.
      - Boundary: this does not mean Gate 00F passed. UniVTAC/TaCauchy/
        IsaacLab TacSL official runtime validation remains pending and must be
        reported as a comparison/validation gap.
- [x] Downgrade Gate 00F priority and pause current work by user request.
      - Decision: Gate 00F is now low-priority final validation/comparison-gap
        work, not a high-priority active experiment and not an active blocker.
      - Current status: blocked by user pause on 2026-07-01. Wait for the
        user's next instruction before starting allocations, training,
        evaluation, or more implementation.
