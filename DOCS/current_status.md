USER PAUSE / RESEARCH 2026-09-17：用户要求先别继续 predictor 训练，改为下载 Zero-WAM 官方代码与权重，广泛调研触觉/位姿→物体想象→SMP+Zero-WAM demo-following→agent reasoning 的完整方案。semantic r1已03:42:42UTC终态exit2，298134在14:51:04本地时间被外部取消，当前squeue为空；无新训练/物理/资格/渲染/follower启动，两个准备中的修复不启动。不申请替代GPU。当前工作与下载记录：experiments/zero_wam_release_20260917/PAUSE_AND_SCOPE.json。此暂停覆盖下方历史继续修复。

USER RESUMED REPAIR：用户“先把所有东西修好 修好了之后告诉我”授权持续有界修复与验证。2026-09-17T10:33+08:00：共享总200mm接触调整固定3例7200控制/57600子步/720真实视频完成，floor3/3、搬运2/3，5014借用5.52mm后仍FAIL，局部接触迁移小于.07mm并非已证根因。完整Michelangelo持久查询覆盖832步及384真实网格视频完成，4/4mesh几何、3/4完整数值PASS，boltIoU.877<.90；不是触觉预测。Utonia原attempt CPUhash FAIL0model已定位137坐标分量最大约12nm跨CPU差异；一次冻结全部1904编码、原464资格及server29加载逐字节EXACT，独立8tests/32source与root79bindings通过。r1 PGID872130在唯一298134/server29/step0于02:31:35Z启动原两臂各20完整更新，CPUguard实际通过，模型结果待出。整体未修好；详见experiments/object_predictor_v1/overfit_repair_v1/REPAIR_PROGRESS.md。

2026-09-16：按用户最新要求暂停实验推进；当前实验与自动队列已停止，GPU保留。正在由三个subagent并行进行物理/观测、训练/泛化、模型/监督目标的根因调查；未经用户明确恢复不重启实验。 [根因调查与下一步判别建议](../experiments/object_predictor_v1/research_review_20260916/ROOT_CAUSE_REVIEW.md)：已有相同允许输入却不同质量标签的确定反例；可辨识离地子集仍存在5.02%输入支持量对52.58%模型误差的差距；物理采集、训练目标和当前已知box回归范围分别审查。仅调查，不代表恢复实验。

2026-09-16 22:17：用户要求的[6008完整搬运rollout视频](../experiments/object_predictor_v1/prospective_carry_v1/selected_rollout_6008/index.html)已完成：48秒正常速度、480帧全部解码，独立subagent查看首末帧及实际MP4确认；这是成功搬运展示选例，质量/姿态误差仍保留。全16连续视频仍在原pipeline渲染，完整数值[报告](../experiments/object_predictor_v1/prospective_carry_v1/REPORT.md)仍不通过可靠性验收。新增实质进展：隔离官方torch2.4.1/cu121/PyTorch3D0.7.8安装与CPU导入完成，48/48官方网格和48/48原始点云标签处理全部通过。完整官方原生输入0/5触摸对照已准备，实际CPU loader检查48物体/240组通过（8组无有效接触保留），GPU前向/梯度仍等待原串行队列，0新学习。

Latest 2026-09-16T18:55:31.842469+08:00: Prospective fresh-config continuous-carry pipeline PREPARED, not yet executed. Fixed16 configurations =4 geometry/trajectory groups x2 masses x2 grip loads,38400 physics controls and768 causal32-history inference clocks; frozen full Utonia endpoint and existing empirical prior/contact/floor scales,0 neural updates. New current-zero-contact fallback keeps exact prior; actual video holds previous WORLD prediction between1Hz simulation clocks, all16 failures retained. Independent static review complete; syntax and synthetic clock/world tests pass, helper agrees with all16 archived qualification arm/case transforms. Original physics uses isaac_arena_py312/Newton, model and CHSEL use separate existing py311 envs. Sole replacement297947 PENDING AssocGrpGRES; live salloc2230112/tmux%1 verified. Explicit srun and full serial collect/infer/fuse/report/render auto-launch queued, no child yet; do not duplicate allocation. User continuation supersedes old stop boundaries; reliable arbitrary-shape/mass/material/demo objective unproven.

Latest 2026-09-16T18:45:44.506550+08:00: USER explicitly resumes: 继续推进 中间不用停下了. This supersedes historical stop/report boundaries below. Preparing prospective fresh-config continuous-carry evaluation with frozen full Utonia and existing empirical prior/contact/floor fusion. Previous allocation 297713 was externally CANCELLED at 15:56:03 CST; squeue empty and tmux returned to login verified. Requesting ONE replacement retained GPU, no new result claim. Full reliable prediction/arbitrary shape/mass/material/demo objective remains unproven. Active artifacts: experiments/object_predictor_v1/prospective_carry_v1.

Latest 2026-09-16T15:44:14.252937+08:00: FIRST POSITIVE EMPIRICAL PRIOR/CONTACT/FLOOR FUSION qualification and ALL8 actual fullmesh still renders COMPLETE. Sole297713/server31/step0 retained. Frozen fullUtonia112forwards on56 separate TRAIN calibration clocks,0newtraining/physics. Full official triangle queries/TorchLie0.1.0/CHSEL local optimizer; explicitly project fusion cost, not originalCHSEL or independentMAP. Fixed8TRAIN final mean center2.1897->1.8782cm, rotation25.431->14.6714deg, mesh3.0840->2.09765cm; improve6/8,8/8,8/8. Size3.93721%/mass9.11485% unchanged. All8 fullmesh minima >=0; onlyknownmesh state, no arbitraryshape/material/demo/generalization claim.

Previous prior-contact-only result center1.7962/rotation18.1366/mesh2.4404 kept with4grounded predictions penetrating4.63–7.34cm; NOT reliable success. Sole additional floorfactor usespublicz0/.001msoftscale, noGTgroundmask/sweep. Final8 actualstills individuallyviewed; physicalfloorvisualalignedz0 (oldfloor−1.75cm), minimumheightlabelsvisible. Qualification/render processes exit0, complete sources/statuses under research_review_20260916/prior_contact_floor_fusion_v1, index/REPORT/IMAGE_INSPECTION/DELIVERY_SUMMARY. Floor finite differencecomponent savedreadback8active8inactive maxscaled1.19e−08. Completecase timing6.64–17.69s inclchecks, notrealtime. Residualfailures:2centerregressions, groundedfloat8.49mm, airbornebottom4.55–7.80cmlow; oldTEST1/8 andforcezero negatives preserved.

STOP/REPORT boundary reached again: FUSION_REPORT_STOP_BOUNDARY.json. Deliver full useful comparison/render and stop additional training/collection/registration sweeps/allocationchanges. Full reliableprediction/arbitraryshape/mass/material/demo objective remains unproven; do NOT mark achieved. No continuousnewvideo claim (actual8stills). Userrequested fresh mhsubagent research completed with source/gradient/coordinate/floor reviews; all earlier negatives/incidents retained.

Latest 2026-09-16T15:17:11.529637+08:00: Official CHSEL local-stage ablation COMPLETE. local_stage_r1 PGID1780177 terminalexit0 at07:13:15UTC; readback/render terminalexit0 at07:14:48UTC. Eight fixed TRAIN clocks, sameB30/seeds/localdefaults, register_single(skip_qd=True), copied first-start trajectory, no model/physics updates. Original costs EXACT vs old fullQD; all30 observation-selected prediction maxdiff5.960464477539063e-08 vs oldQD. Raw original-start center2.1897->9.3517cm (8/8worse), rotation25.431->19.339deg, mesh3.0840->4.5909cm; not success. Complete actual8fullmesh stills + trajectory plot individually viewed, report/index/IMAGE_INSPECTION under research_review_20260916/chsel_local_stage_v1_r1. No newcontinuousvideo. Firstattempt terminalsaveerror and corrected rerun preserved; no source optimizer changes.

Candidate landscape complete: lower-cost candidate minimum mesh displacement4–9cm; FREE6/8pool has no candidate improvingbothcenter/rotation. mhsubagent confirmed originalinitialization used correctly; drift already local, no persistentprior. Next explicit prior/contact fusion needs consistent residual-gradient and calibrated units/uncertainty; official CHSEL abs-forward/squared-gradient mismatch noted. No blindQD/thresholdsweep or claiming complete. perfectlyconstrained clone terminalEOF exit128 (session20770), unbuilt. Sole297713/server31/step0 retained; global reliableprediction/arbitraryshape/demo objective unproven; currentusercontinue remains authorized.

Latest 2026-09-16T14:55:12.876804+08:00: Both official CHSEL eight-TRAIN conditions and ALL16 actual fullmesh still renders COMPLETE and individually inspected. Surfaceonly qualification_r2 PGID1609529 exit0 06:39:54UTC: baselinecenter2.1897cm/rotation25.431deg/mesh3.0840cm versus8.0046/49.320/4.6157, FAIL. Added10mmhandcore FREE officialDoubleDirectCost hand_free PGID1659346 exit0 06:52:07UTC:10.0465cm/55.0444deg/5.4675cm, position/rotation/mesh0of8improve, FAIL. Quality unchanged mass9.1149%/size3.9372%; no learning/physics. Reports/galleries under research_review_20260916/chsel_qualification_v1_r1 and chsel_hand_free_v1. No continuousvideo claim for these16stills.

ActualFREE saved readback PGID1688195 exit0 06:53:04UTC: all8 original+selected official costs EXACT0 difference;273quantizedpoints each, minhanddepth8.39mm/mincontactdistance22.47mm/GTactualgrid overlap0all8. Default objectFREEtolerance18.69–19.92mm allows muchpenetration; wrong15cm selectedpose canstillhave0overlap. Preserve independent mhsubagent findings and allnegativeproof. Next: saved candidate-set observationcost vs priorposition/rotation displacement analysis; then justified prior-preserving fusion/uncertainty or new observation route, NOT blanketFREEweight/tolerance sweep. No newcode beyondofficialadapters; rootdocs/reports updated. Sole297713/server31/step0retained. Currentusercontinue overrides historicalstop; global reliableprediction/arbitraryshape/mass/material/demo objectives unproven.

Latest 2026-09-16T14:48:58.628030+08:00: Resumed official CHSEL study progressed to actual results. Full official backend installed in isolated chsel_py311; original env unchanged. Import Shapely omission and official PK missing axis_angle_to_matrix re-export incidents preserved; adapter binds existing identical official function, no solver edits. qualification_r2 PGID1609529 EXIT0 at06:39:54UTC; 8 fixed TRAIN registrations and all8 full-mesh still renders COMPLETE/individually viewed. Surface-only FAIL: center2.1897->8.0046cm, rawrotation25.431->49.320deg, nearest-vertex3.0840->4.6157cm despite contact1.4209->0.1152cm. Position2/8, mesh3/8, rotation0/8improve. Report/gallery: experiments/object_predictor_v1/research_review_20260916/chsel_qualification_v1_r1/REPORT.md and index.html. No new learning/physics; known mesh only.

Saved hand-interior diagnosis PGID1644116 EXIT0 at06:43:11UTC: both full hand meshes closed/winding/positivevolume and512deep-point independentray checks each pass; no separate selfintersection certificate. 5/10mm erosion bands plus1cm contact exclusion have zero truth overlap inall8TRAIN; 10mmcore273points identifies5/8surface-only chosen overlaps,3/8remainunexcluded. Fixed follow-up hand_free PGID1659346 onsole297713/server31/step0 running officialDoubleDirectCost,3mmFREEvoxels,27310mmcorepoints,defaultFREEweight20/defaultsurface_threshold, same8TRAIN/30initial/100QD/1register/seeds; no sweep. First5000/1206done thennext, fullGTmetrics/renders pending. Preserve oldnegative study; do not claim generalization/arbitraryshape/mass/demo success or completion. Currentusercontinue overrides historicalstopbelow; no release/duplicateallocation.

Latest 2026-09-16T14:15:50.529434+08:00: USER RESUMED after reviewing the completed video: explicitly requests web research, fresh mhsubagent review and continued improvement. This supersedes the historical stop/report boundary below; old results remain unchanged. Active research: experiments/object_predictor_v1/research_review_20260916/RESEARCH_AND_NEXT_STEP.md and RESUMPTION.json. Independent review and saved geometry diagnosis COMPLETE; frozen full-Utonia inputs for 8 predeclared TRAIN windows COMPLETE (16 forwards including exact repeats, 0 updates/physics). Official CHSEL backend dependency download and adapter qualification in progress, no registration/improvement claim yet. Preserve sole297713/server31/step0, no duplicate allocation. Full reliable prediction/arbitrary-shape/demo objective remains unproven.

Latest 2026-09-16T13:12:38.194514+08:00: Current fixed response-surface study and ALL actual prediction renders COMPLETE. Original pipeline PGID312301/coordinator312307/renderer842179 exited normally on sole297713/server31/step0, training.status exit0 at2026-09-16T05:08:54Z; directcompute process absence verified. Exactly6000 updates (three2000), no additional training/collection/sweeps. Full8 TEST/3840frames/40fps/96s video complete and allframes decoded; originalfullmeshes MesaEGL. VIDEO: experiments/object_predictor_v1/training_response_surface_v1/renders/surface_state_comparison.mp4 (43416381bytes,SHA256cf2596ed29d91b2f23cc6617da64f1b14f416de3eb6fe799895dea085c670ab9). index.html, REPORT.md, NUMERICAL_ANALYSIS.md finalized with all8metrics, conditional1case9window caveat, forcezero negatives and observed carrying failures. FINAL_DELIVERY_AUDIT16/16PASS; fullAdam51/51, comparison30/30, visualclock22/22 with EXACT0 primarypredictiondifferences; all12 pinnedsourcehashes unchanged. AGENT_INSPECTION22 individuallyviewed images (2plots+19actualframes+all8poster); finalVALgraph separate. All3840decoded automatically, notallhumanviewed.

Scientific acceptance FAIL remains: TEST physical1/8, primaryonly5045/9windows. Allwindow mass centroid48.44%/surface43.69%/guard57.27%; forcezero34.91/33.74/45.05%. Actual5045 successfullyairborne17.2cm but finaltruth.355kg vscentroid.529kg49.1%/130.5deg andguard.574kg61.8%/50.8deg. Actual5046/5047remain grounded whilehandsraised andpredictedobjectsincorrectlyelevated (centroid19.1/20.1cm,guard16.1/16.6cm positionerrors). Specificlearnedshortcutnotproven. Onlyknownmeshpose/scale, noarbitraryshape/material/native/demo/policyclaim.

USER STOP/REPORT BOUNDARY NOW REACHED: REPORT_STOP_BOUNDARY.json records stopping additional experiments after this completed useful verified comparison and actualrender. Fully report results; do not auto-append newtraining,collection,sweeps or allocation. Global reliableprediction/arbitraryshape/demo objective remains scientifically unproven; do notmark achieved. Priornegatives, failedacquisition denominators, sensorlimits, CPUimportincident/correctedenv andolderresourceincidents preserved. Allocation not released.

Latest 2026-09-16T12:50:01.863547+08:00: Actualpredictionrender first5cases5040..5044 COMPLETE480each, sixth5045started; latest2421/3840frames. Same842179 directcomputeSl143%CPUelapsed29:26 underPGID312301/sole297713/server31/step0/MesaEGLfullmeshes. Fifthfixed24.04/47.94s inspected: groundedtruth.355kg; midcentroid.460kg29.6%/7.2cm/165.8degvsguard.553kg56%/10.1cm/92.2deg, clearorientation/offsetfailure. Finalcentroid.436kg22.9%/1.9cm/147.6degvsguard.490kg38.1%/2.7cm/79.8deg,bothsize5.4%; lowcenter/sizeNOTcorrectpose/carry. RootAGENT_INSPECTION14imagesallhashverified(2plots+12actual), finalVALgraphseparate.

Thisturn verifiedwait+case5inspection only, no model/physics/source/controller/protocol/budgetchanges. Nextinspectactualsuccessful5045lift, then5046/47failures, full3840encoding/decode/fullreportandstop. All3training6000/fullAdam51PASS/batch1comparison30PASS+scientificFAIL/visualclock22PASSbothprimarydifferenceEXACT0/analysisdraft preserved. Original48/TEST1of8and9primarywindows/coverageFAIL/105acquisitionimages/44+39+393qualification/globalreliableprediction-arbitraryshape-demo goalunproven retained,no restart/duplicate/release.

Latest 2026-09-16T12:45:07.854837+08:00: Actualpredictionrender first4cases5040..5043 COMPLETE480each, fifth5044started; latest1938/3840frames. Same842179 directcomputeRl143%CPUelapsed25:00 underPGID312301/sole297713/server31/step0/MesaEGLfullmeshes. Fourthcasefixed24.04/47.94s inspected: groundedtruth.773kg, midcentroid.921kg19.1%/4.8cm/34.4degvsguard.879kg13.6%/2.9cm/11deg; finalcentroid1.109kg43.4%/4.2cm/71.2degtiltedvsguard.821kg6.1%/3.2cm/13.7deg, sizes16.6/16.5%wrong. Lowindividualguardmasserrorpreserved butnoactualcarry/fullstateclaim. RootAGENT_INSPECTION12imagesallhashverified(2plots+10actual), finalVALgraphseparate.

Thisturn verifiedwaits+case4inspection only, no model/physics/source/controller/protocol/budgetchanges. All3training6000/fullAdam51PASS/batch1comparison30PASS+scientificFAIL/visualclock22PASSbothprimarydifferenceEXACT0/source-boundanalysisdraft retained. Remaining4cases/full3840/encoding+decode/allcaseimageinspection/fullreportthenstop, especiallysuccessful5045actualcarryview. Original48/TEST1of8and9primarywindows/coverageFAIL/105acquisitionimages/44+39+393qualification/globalreliableprediction-arbitraryshape-demo objectiveunproven preserved,no restart/duplicate/release.

Latest 2026-09-16T12:39:26.923875+08:00: Actualpredictionrender first3cases5040..5042 COMPLETE480each, fourth5043started; latest1515/3840frames. Same842179 directcomputeSl142%CPUelapsed19:51 underPGID312301/sole297713/server31/step0/MesaEGLfullmeshes. Thirdcasefixed24.04/47.94s inspected: groundedtruth.773kg; midcentroid1.071kg38.5%/6cm/41.3degvsguard.868kg12.3%/4.1cm/14.9deg; finalcentroid1.075kg39%/5.7cm/65.7degtiltedvsguard.851kg10.1%/4.2cm/18.2deg, sizes15.9/16.2%wrong. Keepbothlocalmassimprovementandacquisition/posefailure; nooverallclaim. RootAGENT_INSPECTION10imagesallhashverified(2plots+8actual), finalVALgraphseparate.

Thisturn verifiedwaits andactualcase3inspection only; no model/physics/source/controller/protocol/budgetchanges. All3training6000/fullAdam51PASS/batch1comparison30PASS+scientificFAIL/visualclock22PASSbothprimarydifferenceEXACT0/analysisdraft retained. Remaining5cases/full3840/encoding+decode/othercaseinspection/fullreportthenstop. Original48/TEST1of8and9primarywindows/coverageFAIL/105acquisitionimages/44+39+393qualification/globalreliableprediction-arbitraryshape-demo objectiveunproven retained,no restart/duplicate/release.

Latest 2026-09-16T12:32:33.582882+08:00: Actualpredictionrender first2cases5040/5041 COMPLETE480frameseach, third5042started; latestverified973/3840frames. Same842179 directcomputeSl142%CPUelapsed12:45 underoriginalPGID312301/sole297713/server31/step0/MesaEGLfullmeshes. Secondcasefixed24.04s/47.94s inspected: groundedtruth.391kg, midcentroid.627kg60.3%vsguard.896kg129.1%; finalcentroid.893kg128.3%/49.2degtiltedvsguard.747kg91%/13.8deg, size17.5/16.2%wrong. RootAGENT_INSPECTION8images allhashverified(2plots+4case5040+2case5041), finalVALgraphseparate; othersnotclaimedviewed.

NEW saved-only visualclockreadback CPUexit0 at04:28:10UTC:22/22PASS, bothcomplete3792visual and760primaryclockgrids/all8cases, exacttargets/masks/clocks acrossarms, BOTHmaxprimarypredictiondifferenceEXACT0. VISUAL_CLOCK_READBACK.json binds4npzsourcehashes andpercaseprimarycoverage0/0/0/0/0/9/0/0.0model/physics/updates; NUMERICAL_ANALYSIS andsourcebindingsupdated withthisevidence. Thisprovesfrozenpredictionconsistency, notcompletedvideooraccuracy.

All3training6000/reload0/fullAdam51PASS/batch1comparison30PASS+scientificFAIL/qualifications unchanged. Remaining6rendercases/full3840/encodeddecode/allcaseimageinspection/fullreportthenstop; no source/model/controller/protocol/budget/physicschanges thisturn, no extraexperiments. Original48/TEST1of8coverageFAIL/105acquisitionimages/44+39+393qualification/globalreliableprediction-arbitraryshape-demo goalunproven retained,no restart/duplicate/release.

Latest 2026-09-16T12:26:36.956790+08:00: Actualmeshpredictionrender case5040 COMPLETE480frames, second5041started; latestverified487/3840frames. Same renderer842179 directcomputeSl142%CPUelapsed6:32 underPGID312301/sole297713/server31/step0, existingfullmeshMesaEGLfallback. Firstcaseframes1/301/1201/2396individuallyviewed; rootAGENT_INSPECTION6images(2plots+4actualframes) allhashverified; finalVALgraphseparate. Warmupcorrect2/32history,nofabricatedprediction. At24.04s groundedtruth.391kg versuscentroid.748kg91.1%/guard.849kg116.9%; final47.94s centroid.888kg127%/4.2cm/47.4degvisiblytilted, guard.764kg95.2%/3.1cm/17deg, sizes~17%wrong. Failurevisible, noactualcarryingorqualitysuccessclaim.

All3training6000/reload0/fullAdam51PASS/batch1comparison30PASS+scientificFAIL/bothdense3792clocks andsource-boundanalysisdraft retained. No model/source/controller/protocol/budget/physics changes thisturn; verifiedwait+actualframeinspection. Remaining7cases/full3840/encodedvideo+decode/othercaseinspection/reportthenstop; no newexperiment. Original48/TEST1of8and9primarywindows/coverageFAIL/105acquisitionimages/44+39+393qualification/globalreliableprediction-arbitraryshape-demo objectiveunproven preserved,no restart/duplicate/release.

Latest 2026-09-16T12:21:00.277322+08:00: BOTH densevisualarms COMPLETE3792TESTclocks each/7584total, stride5batch1/0optimizer; renderer842179 startedunderoriginalPGID312301/sole297713/server31/step0. DirectcomputeRl140%CPUelapsed1:21, renderlog configuredexistingMesa softwareEGL fallback(fullmeshesretained), notGPUblankframeclaim. Rendererprechecks matchedallprimaryclocks/targets andpredictiontolerance beforeframes; noactualnumericmaxreadbackclaimed. First65+actualPNGframes generated; full3840/8cases/video stillpending.

Inspected firstactual3panelprediction frame5040/0301 at6.04s: groundedtruth.391kg, centroid.609kg/55.6%error versusguard.876kg/123.9%; handsaligned,cyantruthoutline/orangepredictedgeometry visible, nonsuccess retained. RootAGENT_INSPECTION now3images(2plots+1actualframe), finalVALgraphseparate. All3training6000/fullAdam51PASS/batch1comparison30PASS+scientificFAIL/analysisdraftsourcehashes unchanged. Continueoriginalall8fullrender/encodeddecode/primaryclockverification/fullinspection+reportthenstop; no newmodel/physics/controller/protocol/budget orrepeat. Original48/TEST1of8and9primarywindows/coverageFAIL/105acquisitionimages/44+39+393qualifications/globalreliableprediction-arbitraryshape-demo goalunproven retained,no restart/duplicate/release.

Latest 2026-09-16T12:12:28.168369+08:00: Firstdensevisual centroid_force COMPLETE3792savedTESTpredictionclocks/all8episodes/stride5/batch1/0newupdates. FIRST_VISUAL_READBACK pinsreport; originalprimaryclockequivalence willbeverified byexistingrendererafterbotharms, notyetclaimed. SamepipelinePGID312301/coordinator312307/sole297713/server31/step0 automaticallyenteredsecondvisual surface_force_airborne_mass_guard evaluator796850, directcompute liveverification completed. No actualrenderframesyet.

Prepared NUMERICAL_ANALYSIS.md plushash-bound6-source record: currentallwindow/conditional1case9windowtables, oldsupport4.4896% versusold10physicalPASSairborneprediction62.4729%/zero30.1617%, old27pad10case80latewindowobservationdeficit21.3707%vsfullsurfaceshear3.5797%/solverreference.01889%. Strictlyseparatespriorconditionfromcurrent, physicalacquisition/modelgeneralization/forceidentifiability, demonstratedobservationsvsunprovenmechanism; no arbitraryshape/material/demo claim ornewexperiments. ReportexplicitlymarksactualvideoPENDING.

All3training6000/reload0/fullAdam51PASS/batch1comparison30PASS+scientificFAIL/final15VALcurve+2trainingplotsinspected retained. Remainingseconddensevisual/readbackclocks/all8actualmeshvideoinspection/fullreportthenstop; no newmodel/source/controller/protocol/budget/physicschanges. Original48/TEST1of8coverageFAIL/105acquisitionimages/44+39+393qualification/globalreliableprediction-arbitraryshape-demo objectiveunproven preserved,no restart/duplicate/release.

Latest 2026-09-16T12:05:24.806651+08:00: ALL3 independentbatch1+forcezero COMPLETE. COMPARISON.json integrity30/30PASS, scientificacceptanceFALSE: coverage/center/rotation/mass/improvement/sizenonregression fail. All760TESTwindows8episodes equalepisode centroid5.567003cm/91.573622deg/11.248552%/48.440336%, surface4.885209cm/60.356413deg/11.044969%/43.688621%, guard5.370326cm/56.635758deg/12.267070%/57.271001%. Onlyairborne5045/9windows1of8 centroid4.443974cm/134.921082deg/6.918541%/50.847935%, surface3.779485cm/39.229130deg/7.565470%/37.202641%, guard3.304287cm/59.072014deg/8.616498%/65.293945%. Forcezero reducesall3masserrors; guardairborne65.293945->20.774943%, all57.271001->45.053494%. No reliablecarry-state/guard/touchsuccess. Surface/guardmodel-onlymedianlatency92.582215/92.652474ms. NUMERICAL_READBACK sourcepinsallreports.

Primarycomparison andlearningcurves viewed/hashbound rootAGENT_INSPECTION2images, complete15VALgraph separatelyreviewed. Primarychart conditionalonly1episode9windows; coverageFAILmustaccompanyanypresentation. TrainingguardmasslosslowdoesNOTimplytestbenefit. All3fullmodel2000/6000total/reload0/fullAdam51PASS/importincident preserved.

SamepipelinePGID312301/coordinator312307/sole297713/server31/step0 nowvisualstride5batch1 centroid evaluator754719 directlyverifiedRl90.1%CPUelapsed2:36, originaldeclared extra renderingclocks only/0optimization. Remainingtwoarmsdensevisualreadback/all8actualmeshvideo/frameinspection/fullreportthenstop. No source/controller/protocol/budgetchanges ornewphysics. Original48/TEST1of8coverageFAIL/105acquisitionimages/44+39+393qualifications/globalreliableprediction-arbitraryshape-demo goalunproven retained,no restart/duplicate/release.

Latest 2026-09-16T11:56:22.952149+08:00: Firstcentroid_force independentbatch1 COMPLETE (original2000/0newupdates), medianmodel-onlylatency93.023863ms. All760TESTwindows8episodes center5.567002cm/rotation91.573624deg/size11.248553%/mass48.440334%; forcezero4.803960cm/85.658203deg/10.543624%/34.907556%. Airborneonly5045/9windows1of8:4.443974cm/134.921112deg/6.918542%/50.847948%; forcezero5.843271cm/130.330093deg/4.282378%/2.345916%. Negativeforce-mass effectpersistsbatch1; no overalltouchbenefit/coverageacceptance. FIRST_BATCH1_READBACK pinsreport andtrainingbatchsource/comparison.

SamepipelinePGID312301/coordinator312307/sole297713/server31/step0 automaticallyenteredsecondbatch1 surface evaluator721548 verifiedliveRl76.6%CPUelapsed43s. All3trainingCOMPLETE6000/reload0/fullAdam51PASS/final15VALreportcurveviewed; importincidentandcorrectionretained. No optimization/source/controller/protocol/budgetchange. Remainingsecond+thirdbatch1-forcezero/finalcomparisonplots/visualstride5twoarms/actual8TESTpredictionrenderinspection/fullreportthenstop. Original48/TEST1of8and9primarywindows/coverageFAIL/105acquisitionimages/44+39+393qualification/globalreliableprediction-arbitraryshape-demo objectiveunproven preserved,no restart/duplicate/release.

Latest 2026-09-16T11:54:29.795840+08:00: ALL3 response-condition fullUtonia arms COMPLETE exactly2000each/6000total; all3endpointreload prediction EXACT0. ALL_THREE_TRAINING_ENDPOINT_READBACK fullinitial/all6000actualclocks/matchedbatches+labels5checksPASS. Thirdguard training-batch TEST all760windows8episodes5.378205cm/56.503361deg/12.285093%/57.200754%; forcezero5.260889cm/45.412479deg/11.299095%/45.020092%. Onlyairborne5045/9windows1of8:3.316020cm/59.510654deg/8.755746%/66.700751%, forcezero3.157833cm/51.898460deg/7.103962%/21.937768%. Guardmassworse thanbothothers; originalcoverageFAIL/negativeforceeffect preserved. Thesearetraining-batchTEST, independentbatch1pending.

FinalsavedVALgraph COMPLETE15reports CPUexit0 at03:48:47UTC; all15sourcehasheschecked,8panelsviewed/hashbound invalidation_readback_final/AGENT_INSPECTION. ThirdVAL2000all3.539155cm/49.886005deg/8.765951%/33.700180%, airborne3.215324cm/54.668495deg/8.833397%/22.515543%. FullendpointAdam firstCPU audit554102failedimportModuleNotFoundErrorutonia beforecheckpoint/modelread; FINAL_AUDIT_IMPORT_INCIDENT pinsfailedlogs/status. Correctedexistingdeps/vendor/root PYTHONPATH ONLY, r1PGID713761 launch03:53:03UTC terminalexit0: FULL_ENDPOINT_ADAM_AUDIT51/51PASS fullmodel/latest/Adam/protocol/clocks/bindings/unused54tokenunchanged/138407449moments/actualbatches. All12declaredsource+qualificationhashesstillunchanged. No trainingrepeat/newmodel/physics fromaudit.

SamepipelinePGID312301/coordinator312307/sole297713/server31/step0 now independentbatch1 centroid evaluator706757 verifiedliveRl92.2%CPUelapsed2:20. No remainingoptimization. Remainingall3batch1+forcezero/comparisonplots/visualstride5twoarms/8actualmeshpredictionvideoandinspection/fullreportthenstop. Original48/TEST1of8and9primarywindows/105acquisitionimages/44+39+393qualifications/allnegativehistory/globalreliableprediction-arbitraryshape-demo goalunproven retained; no restart/duplicate/release.

Latest 2026-09-16T11:45:01.804342+08:00: Thirdmass_guard VAL1500 COMPLETE all760windows8episodes3.928063cm/58.400581deg/5.512097%/39.886865%; airborne280windows7of8episodes4.075559cm/67.555901deg/5.414176%/23.171578%. Againstown1000center/sizeimprovebutrotation55.065205->67.555901deg andmass21.209872->23.171578% worsen. Againstsurface1500airborne4.433963cm/61.555508deg/7.165583%/18.940462%, guardcenter/sizebetter,rotation/massworse; mixednotoverallbenefit. ALL_THREE_VAL1500_COMPARISON pinsall3reports/fullinitial/actual1500clocks-batches-labels3checksPASS; noTEST/selection/finalacceptance.

SamePGID312301/thirdtrainer538555 onsole297713/server31/step0 verifiedliveRl98.3%CPUelapsed18:49/update1580. First2armscomplete2000/reload0/negativeTEST-forcezero and9onlyprimaryTESTwindows retained. FinalVAL371845/fullAdam554102 CPUfollowerswaiting; thirdfixed2000/total6000 unchanged. Thisturn verifiedwait+new3wayVALreadback only; no model/source/controller/protocol/threshold/budget/physicschanges. Remainingthird2000/allbatch1-forcezero/finalaudits+plots/8actualpredictionrenders/stop-report. Original48/TEST1of8coverageFAIL/105images/44+39+393qualification/globalreliableprediction-arbitraryshape-demo goalunproven preserved,no restart/duplicate/release.

Latest 2026-09-16T11:39:25.194048+08:00: Thirdmass_guard VAL1000 COMPLETE all760windows8episodes6.628207cm/55.811192deg/8.325130%/32.578802%; airborne280windows7of8episodes6.466242cm/55.065205deg/9.194630%/21.209872%. Againstown500center/masslowerbutrotation/sizehigher; againstsurface1000airborne5.770347cm/58.738213deg/6.771883%/18.270032%, guardonlyrotationlower, center/size/massworse. ALL_THREE_VAL1000_COMPARISON hashesall3reports/fullinitial/actual1000clocks-batches-labels3checksPASS. NoTEST/finalacceptance/selectionclaim; nooverallguardbenefit.

SamePGID312301/thirdtrainer538555 onsole297713/server31/step0 verifiedliveRl97.8%CPUelapsed13:09/update1000. First2armscomplete2000/reload0/negativeTEST-forcezero retained. FinalVAL371845/fullAdam554102 CPUfollowerswaiting; no duplicate orsourcechange. Thisturn verifiedwait+new3wayVALreadback only, originalthird2000/total6000budget unchanged. Remainingthird2000/allbatch1-forcezero/finalaudits+plots/8actualpredictionrenders/stop-report. Original48/TEST1of8/9primarywindows/coverageFAIL/105imagechecks/44+39+393qualification/allnegatives/globalreliableprediction-arbitraryshape-demo objectiveunproven preserved; no restart/release.

Latest 2026-09-16T11:35:01.791529+08:00: Thirdsurface_force_airborne_mass_guard VAL500 COMPLETE: all760windows8episodes7.487449cm/58.619911deg/7.940602%/37.560427%; airborne280windows7of8episodes7.948387cm/48.495110deg/7.245119%/22.484204%. Comparedsameclock surface3.505564cm/52.063419deg/5.039201%/14.632586%, guardonlyrotationlower; center/size/massworse. Versuscentroid7.701492cm/56.696728deg/4.344005%/32.135656%, guardrotation/masslowerbutcenter/sizeworse. ALL_THREE_VAL500_COMPARISON pins3reports/samefullinitial/all3actual500clocks-batches-labels3checksPASS. IntermediateVAL only, noTEST/selection/overallimprovement.

SamePGID312301/thirdtrainer538555 liveRl97.2%CPUelapsed8:50/update580 onsole297713/server31/step0. First2armsCOMPLETE2000/reload0 andnegativeTEST-forcezero retained; thirdcontinuesfixed2000. FinalVAL371845/fullAdam554102 CPUfollowerswaiting; nopendingtaskrerun. Thisturn verifiedsameprocesswaits and3wayVALreadback, no model/source/controller/protocol/threshold/budgetchanges oradditionalphysics. Remainingthird2000/allbatch1-forcezero/finalaudits+plots/8actualpredictionrenders/stop-report. Original48/TEST1of8and9primarywindows/coverageFAIL/105imagechecks/44+39+393qualifications andglobalreliableprediction/arbitraryshape/demo objectiveunproven preserved; no restart/duplicate/release.

Latest 2026-09-16T11:30:32.941825+08:00: Third surface_force_airborne_mass_guard actualtraining started; directcompute trainer538555 Rl94.9%CPUelapsed4:25/update140, actualprefixreadback229. All3 fullinitialSHA/bindings andactualprefixbatches-clocks-labels EXACT; second/thirdprotocol differONLYmass_airborne_guard/output. Zero-labelbatchmasslossEXACT0, positivemasslabelbatchespresent; MATCHED_THIRD_ARM_START7checksPASS, no modelbenefitclaim. SamepipelinePGID312301/sole297713/server31/step0, originalthird2000budget unchanged. Prior2endpoints2000/reload0 andnegativeTEST-forcezero results retained.

Reusedexisting audit_mixed_endpoints.py withoptional --arms (originaldefaultsunchanged), compiled; newCPU-only finish_endpoint_audit.py waitsall3RESULT thenfullmodel/latest/Adam/protocol/exactclocks/namedbindings/unused54tokenunchanged/138407449moments/batches checks. RetainedfollowerPGID554102 launched03:28:38UTC verifiedlivewaiting;0modelforward/physics/optimizer, outputFULL_ENDPOINT_ADAM_AUDIT pending. ExistingfinalVALfollower371845 stillwaitsall15reports. No pinnedmodel/training/evalsource/controller/thresholdchanges oradditionalbudget. Remainingthird2000/allbatch1-forcezero/bothfinalaudits+plots/8actualpredictionrenders/stop-report; original48/TESTcoverageFAIL9primarywindows/105images/44+39+393qualification/globalreliableprediction-arbitraryshape-demo objectiveunproven preserved,no restart/duplicate/release.

Latest 2026-09-16T11:26:52.155985+08:00: Second surface_force COMPLETE2000 andendpointreloadEXACT0. Training-batch TEST all760windows8episodes center4.875691cm/rotation60.409771deg/size11.039481%/mass43.617487%; forcezero5.538803cm/46.547058deg/10.434873%/33.756265%. Actualairborneonly5045/9windows/1of8:3.763268cm/38.933701deg/7.641167%/37.233749%, forcezero4.033576cm/40.007523deg/6.848352%/22.236432%. Surfaceimprovesfirstcentroidposition/rotation/mass butstillpoor, forcezeroagainlowersmasserror; nottouchsuccess. FIRST_TWO_ENDPOINT_READBACK pinsbothreports/protocols/full2000logs,6checksPASS(samefullinitial/actual2000batches-labels/optimizerbindings/bothreload0). All-armfullAdam andindependentbatch1 stillpending; originalcoverageFAIL retained.

SamepipelinePGID312301/coordinator312307 automaticallyenteredthird surface_force_airborne_mass_guard trainer538555 onsole297713/server31/step0, directcompute processverifiedlive; no thirdprotocol/updateclaimyet. SecondVAL2000all3.436764cm/64.057617deg/6.240182%/22.099595%, airborne2.961699cm/81.896812deg/6.129724%/17.298894%; highrotationfailure preserved. FinalVALCPUfollower371845 waitsall15reports. Continuethird2000 thenallbatch1-forcezero/fullmodelAdam/finalVALplots/8actualpredictionrenders/stop-report; no newsource/controller/model/protocol/budget/physics. Original48acquisition/TEST1of8/105actualimageinspection/44+39+393qualification/alloldnegatives andglobalreliableprediction/arbitraryshape/demo objectiveunfinished retained; no duplicate/release/restart.

Latest 2026-09-16T11:18:58.955547+08:00: Surface_force VAL1500 COMPLETE all760windows/8episodes4.171397cm/55.720234deg/6.949394%/22.960933%; airborne280windows/7of8episodes4.433963cm/61.555508deg/7.165583%/18.940462%. Versusown1000center5.770347->4.433963cm improvesbutrotation58.738213->61.555508deg/size6.771883->7.165583%/mass18.270032->18.940462% worsen. Againstcentroid1500airborne4.673003cm/64.972221deg/6.252403%/35.759383%, surfacecenter/rotation/masslower,sizehigher; mixedVALonly, stilllargeposeerror. MATCHED_VAL1500_COMPARISON hashesbothreports/samefullinitial/all1500actualbatch-clock-label3checksPASS. NoTESTsuccess/selectionclaim.

SamePGID312301/secondtrainer423138 liveSl98.1%CPUelapsed18:26/update1540 onsole297713/server31/step0; firstcentroidCOMPLETE2000/negativeTEST-forcezero result retained, thirdarmpending. Thisturn verifiedsameprocesswaits andnewmatchedVALreadback, no source/controller/protocol/threshold/budget changes oradditionalphysics/modelwork. FinalVALfollower371845 waitsall15reports. Remainingsecond2000/third2000/allbatch1-forcezero/allarmfullAdam/finalVALplots/8actualpredictionrenders/stop-report. Allprior48corpus/TEST1of8physicaland9primarywindows/coverageFAIL/105imagechecks/44+39+393qualification preserved; globalreliableprediction/arbitraryshape/demo goalactive/unproven, no restart/duplicate/release.

Latest 2026-09-16T11:14:28.708892+08:00: Surface_force VAL1000 COMPLETE: all760windows/8episodes center5.941474cm/rotation52.957535deg/size7.039046%/mass21.371423%; airborne280windows/7of8 center5.770347cm/58.738213deg/6.771883%/18.270032%. Allfourairborneerrors WORSEN vsown5003.505564cm/52.063419deg/5.039201%/14.632586%; preserve nonmonotonicfailure. Againstmatchedcentroid1000airborne6.404878cm/62.222446deg/4.672036%/31.670022%, surfacecenter/rotation/masslowerbutsizehigher. MATCHED_VAL1000_COMPARISON pinsfourreports/samefullinitial/all1000actualclocks+batches+labels3checksPASS; noTEST/finalacceptance/selectionclaim.

SamepipelinePGID312301/secondtrainer423138 onsole297713/server31/step0 liveRl97.7%CPUelapsed14:03/update1120. Firstcentroidcomplete2000/negativeTEST-forcezero/only9primaryTESTwindows/coverageFAIL retained. Thirdarmnotstarted; finalVALfollower371845 waitsall15reports. Thisturn verifiedsameprocesswaits andnewmatchedVALreadback, no newtrainingbudget/physics/model/controller/protocolchanges. Remainingsecond2000/third2000/allbatch1-forcezero/fullall-armAdam/allVALplots/actual8TESTpredictionrenders andstop-report. Original48/105imagechecks/44+39+393qualification andglobalreliableprediction/arbitraryshape/demo goalunproven preserved; no restart/duplicate/release.

Latest 2026-09-16T11:09:41.809337+08:00: Second surface_force VAL500 COMPLETE; same500actualbatches/clocks/labels andfullinitialmodel EXACT tocentroid, MATCHED_VAL500_COMPARISON.json hashesbothreports/3checksPASS. All760VALwindows8episodes surface4.297007cm/60.415585deg/5.131625%/21.499658% versuscentroid7.931739cm/63.096577deg/4.505380%/29.066205%. Airborne280windows7of8 surface3.505564cm/52.063419deg/5.039201%/14.632586% versuscentroid7.701492cm/56.696728deg/4.344005%/32.135656%: center/rotation/massimprove,sizeworse; rotationstilllarge. IntermediateVALNOTTEST/finalacceptance, noselection. MATCHED_SECOND_ARM_START alsoverifiedfull138407503/bindings/onlyinput+outputprotocoldifferences/actualfirst157exact.

SamePGID312301/coordinator312307/surface trainer423138 onsole297713/server31/step0 directcomputeSl97.1%CPUelapsed9:16/update620; unchangedpreserved2000budget. Firstcentroidcomplete2000/reloadEXACT0 andnegativeTEST/forcezero resultretained. Thirdarmnotstarted; finalVALfollower371845 pending15reports. No model/data/controller/protocol/budgetchanges, extramodelupdates/physics. Remainingsecond2000+third2000/allbatch1-forcezero/fullall-armAdam/finalVALplots/8caseactualpredictionrenders andstop-report. All48acquisition/TEST1of8physical/9onlyprimarywindows/coverageFAIL/105imagechecks/44+39+393qualifications retained; globalreliableprediction/arbitraryshape/demo goalunproven,no duplicate/release/restart.

Latest 2026-09-16T11:01:34.721973+08:00: First response-condition centroid_force COMPLETE2000, all2000actualclocks/matchedfoursettingbatches retained; fullendpointreload prediction EXACT0. Training-batch TEST all760windows/8episodes equalepisode center5.547159cm/rotation91.728523deg/size11.227287%/mass48.320538%; forcezero4.788316cm/85.684372deg/10.527037%/34.899133%. ActualairborneTEST coverage only5045/9windows/1of8:4.450313cm/134.760376deg/6.904071%/50.478083%, forcezero5.890982cm/130.039444deg/4.323005%/2.278381%. Clearnegative force-mass evidence inavailablewindow subset, notsuccess/generalization; original>=10windowsinevery8case gateFAIL, evenonlycoveredcasehas9. Finalbatch1stillpending. FIRST_ENDPOINT_READBACK hashesRESULT/protocol/full2000logs and6checksPASS; fullall-armAdam auditstillrequired.

SamepipelinePGID312301/coordinator312307 automaticallyenteredsecond surface_force trainer423138 onsole297713/server31/step0; directcomputeRl84.9%CPUelapsed1:24, secondPROTOCOL notyetpresent atreadback, noinitialSHA/updateclaimyet. FirstVAL2000all3.529074cm/70.195473deg/6.067020%/26.964039%, airborne3.908787cm/82.766151deg/6.569117%/27.830392% retained. FinalVALcurveCPUfollower371845 continueswaiting. No sourcemodel/data/controller/threshold/budget changes, extraoptimization orphysics. Remainingsecond+third2000/batch1-forcezero/fullendpointAdam/finalallVALplots/actual8TESTpredictionvideo thenstop-report. Original48/TEST1of8physicalPASS/105imagechecks/44+39+393qualification andearliernegatives preserved; globalreliableprediction/arbitraryshape/demo objectiveunproven,no duplicate/release/restart.

Latest 2026-09-16T10:53:53.828521+08:00: Firstcentroid_force VAL1500 complete; all760windows/8episodes4.033689cm/58.843460deg/6.362066%/30.338103%; airborne280windows/7of8episodes4.673003cm/64.972221deg/6.252403%/35.759383%. Against1000 airbornecenter improves6.404878->4.673003cm, rotation62.222446->64.972221deg/size4.672036->6.252403%/mass31.670022->35.759383% worsen. Saved INTERMEDIATE_CENTROID_VAL1500.json hashes500/1000/1500, preservesnonmonotonicfailure; noselection/TEST/conclusion.

SamePGID312301/trainer312312 liveonsole297713/server31/step0: directcomputeRl98.1%CPUelapsed16:33/update1600. Added compiled CPU-only finish_validation_report.py andlaunched retainedfollowerPGID371845 at02:50:10UTC: waitsall15originalVALreports, thenrunsalreadydeclaredsaved-onlyplotter tovalidation_readback_final, 0model/physics/optimizer; terminaliforiginaltrainingfails, no restart. Currentlywaiting; finalimageinspectionpending. No change toany pinnedmodel/training/evalsource/controller/protocol or6000budget. Continuefirst2000/thensecond+thirdarms/allbatch1-forcezero/endpointAdam/actual8TESTmeshpredictionvideo andstop-report. Original48/TEST1of8coverageFAIL/105actualimagechecks/44+39+393qualification/oldpredictionnegatives remain; globalreliableprediction/arbitraryshape/demo goalunproven, no duplicate/release/restart.

Latest 2026-09-16T10:48:56.034859+08:00: Samefirstarm centroid_force fullUtonia training continues PGID312301/trainer312312/sole297713/server31/step0; directcomputeRl97.6%CPUelapsed11:33 at1040, savedactuallog now1125. VAL1000 COMPLETE: all760windows/8episodes equalepisode5.867782cm/54.481930deg/4.838710%/27.023041%; airborne280windows/7of8episodes6.404878cm/62.222446deg/4.672036%/31.670022%. AgainstVAL500 airbornecenter/mass improvebutrotation worsens56.696728->62.222446deg andsizerises4.344005->4.672036%; mixed/nonmonotonic, no endpointselection orTEST/cross-armbenefitclaim. INTERMEDIATE_CENTROID_VAL1000.json pinsbothsavedreports/denominators.

Thisturn verifiedwaitedonsameactualprocess andreadback1000result; no newmodel/physics/budget/sourcechange. Existing0/500VALgraphreview retained; finalall3 fullVALcurves stillrequiredafterreportsready. Rendercode reviewed: all8TEST/fullactualmeshes/sameclock/frozenbatch1estimates/no truthsmoothing, toexecuteexistingpipelineafterthree2000+batch1. Original48acquisition/TEST1of8coverageFAIL/105reviewedimages/all44+39+393qualifications preserved. Remainingfirst2000/twomorearms/fullendpointAdam/forcezero/finalpredictionvideo andstop-report, globalgoalreliableprediction/arbitraryshape/demo unfinished; no restart/release/duplicate.

Latest 2026-09-16T10:45:19.827970+08:00: Response-condition firstfullUtonia centroid_force continues underunchangedPGID312301/trainer312312/sole297713/server31/step0; directcomputeSl96.8%CPUelapsed7:47 atverifiedupdate660, currentcompleteactual log through760. FirstVAL500 saved: all760windows/8episodes equalepisode center7.931739cm/rotation63.096577deg/size4.505380%/mass29.066205%; actualairborne280windows/7of8episodes center7.701492cm/rotation56.696728deg/size4.344005%/mass32.135656%. IntermediateVAL only, stilllargeerrors, noTEST/comparison/selectionclaim.

Added saved-only plot_saved_validation.py (compiled), CPUretainedplot exit0 at02:44:12UTC; source-bound validation_readback_00500 REPORT plus8panelgraph inspected/hashsaved. No extra model/physics/optimization; allinitial/500 points shown, laterarmsabsent. Samehelper must render all3 complete0/500/1000/1500/2000 VAL reports aftersource readiness (uniqueoutput), inadditiontoexistingpipelineplots; no change topinnedmodel/training/evalsource orbudget. Prior48complete/32physicalPASS16FAIL/TEST1of8andzero-labelseven/105actualimageinspection/44+39+393qualification allretained. Continueexactthree2000 anddeclaredbatch1-forcezero/fullmodelAdam/actual8casepredictedmeshvideo thenstop/report. Fullgoalreliablecarryprediction/arbitraryshape/demo unproven; no duplicate/release/restart.

Latest 2026-09-16T10:40:00.840934+08:00: Original48 response-condition collection COMPLETE, both collector/inspection exit0 at02:30:48/02:30:58UTC.115200saved=108000new+7200source-exactTRAINreuse; physicalTRAIN24/32,VAL7/8,TEST1/8 (total32PASS16FAIL). TRAIN5/8groups passesoriginal4required. TEST5045onlyPASS/227airborneholdlabels; other7zero, originalall8primarycoverageFAIL remains, noacceptanceredefinition.5046/47 lifttriggeredbutgrounded/rightobservedload0; actualrenders/traces inspected, rootcauseunproven. All48fullcausalreplaysPASS and105imagesindividuallyviewed/hashverified; otherautomaticpreviews notclaimedviewed. ACQUISITION_FINAL_SUMMARY.json/md pinsoriginal48results/denominators/sources.

Declaredthree2000 fullUtonia study STARTED02:36:50UTC onsole297713/server31/step0 PGID312301/coordinator312307/trainer312312. Directcompute confirmsRl; firstcentroid_force actualupdates through165 verifiedconsecutive andallmatchedfoursetting/sameclock batches. Full137253744released/138407503adapted andall138407503optimizerbindings; initialmodelSHA9deb321928d7bc923833a151dc042c2a5c880ab35dc7dd31c085cadea67f8076. Launchguard verifiedfull48/105imagehashes/pinned44input39fullmodel393fullTRAINchecks; TRAINING_LAUNCH_EVIDENCE andTRAINING_START_READBACK saved. NoVAL500/learnedendpoint oraccuracyclaimyet; surface_force andsurface_force_airborne_mass_guard pending. All3armscontainforce; thisstudyisolatesrepresentation/labelguard, notgeometry-onlytouchbenefit. Continueexactserial6000/batch1-forcezero/fullmodelAdam/8TESTactualpredictionrenders thenstop/report; noextratraining/collection/sweep. Earlierpredictionnegatives/spikes/arbitraryshape/demo limitations preserved; goalactive/unproven,noallocationrestart/duplicate/release.

Latest 2026-09-16T10:19:26+08:00: FirsttwoTEST5040/5041 COMPLETEbothphysicalFAIL/no lift/zeroairborneholdlabels; finalobserved6.0/7.6N and7.3/9.1N, fittedlocalplanes88.7/88.4degapart andspacingestimate correctlyabsent.5041peak38.8722N/righttransient/loadcollapse retained. Geometryobservationdescriptive, notprovenrootcause. NEW TEST_COVERAGE_LIMITATION.json sourcepinsoriginalcriteria/5040result/replay: complete_all8_test_coverage CANNOTPASSthisfixedcorpus because5040haszeroprimarywindows. KeepallfailedTESTcases/denominators, noTESTretune oracceptanceredefinition; alreadydeclaredtrainingcanstillquantifymixed/negativeoutcomes, notwhole-systemsuccess.

Samecollector3944359 onsole297713/server31/step0 continues; directcompute child208253 Rl elapsed43s runningthirdTEST5042. Corpus42/48complete100800saved=93600new+7200reuse,31physicalPASS11FAIL;TRAIN24/32,VAL7/8,TEST0/2. All42causalreplaysPASS;93imagesindividuallyviewed/hashverified through5041 plusTRAINsummary/5014+5022peakpairs; otherautomaticpreviews notclaimedviewed. No newmodel/physicsbeyondfixedcollection/controllerchange/repeat. FullTRAIN393/input44/fullUtonia39 qualificationsunchanged; no trainingqueued. Finishremaining6TEST/full48 thenpreservedthree2000officialmatchedtraining, reportcoveragefailurealongsideallwindow/availableairborne metrics. Priornegativeprediction/spikes/arbitraryshape/demo limitations andgoalactive/unproven retained; no duplicate/release/restart.

Latest 2026-09-16T10:14:12+08:00: Full8VAL acquisition COMPLETE7physicalPASS1FAIL(5034 retained/zeroairborneholdlabels); otherseven1001labels each. Latest5037/38/39 allPASS, final20.1/19.7/19.9cm andstablelate22/14/22N, initialspikes/holddrift retained. VAL_ACQUISITION_SUMMARY pinsoriginal8RESULT hashes; acquisition evidenceonly, no predictor/modelacceptance. Corpus40/48=96000saved(88800new+7200reuse),31physicalPASS9FAIL. TRAIN24/32PASS5/8groupcoverage unchanged.

Samecollector3944359 onsole297713/server31/step0 automaticallyenteredfirstTEST5040, directcompute child186045 Rl elapsed19s; renderer3976006 continues. All40savedcausalreplaysPASS;89imagesindividuallyviewed/hashverified throughallTRAIN+VAL plusTRAINsummary and5014/5022peakpairs, otherautomaticpreviews notclaimedviewed. Thisturn verifiedsameprocesswaits andnewrenderinspection; no newmodel/physicsbeyonddeclaredcollection or source/protocol/controlchanges. FullTRAIN393checks,44input,39officialUtonia checkscomplete unchanged; no trainingqueued. Finishall8TEST/fullcorpus thenpreserved3x2000officialstudy; no sweep/repeats/failurefilter. Prior5034/5014/5022spikes,negativeoldpredictorcarry,arbitraryshape/demo limitations retained; goalactive/unproven, no duplicate/release/restart.

Latest 2026-09-16T10:06:34+08:00: Same response collectorPGID3944359 continues sole297713/server31/step0; directcompute child140874 Rl elapsed36s runningVAL5037. Corpus37/48complete88800saved=81600new+7200reuse,28physicalPASS9FAIL; TRAIN24/32PASS and5/8groupcoverage preserved, currentVAL4/5PASS. New5034VALFAIL/zeroairborneholdlabels/826.514526Npeak, brief~4cmrise thengrounded, finalobserved11.9/13.2N.5035samemasshighergripPASS1001labels/final12.5cm/2.4205cmholdloss; causalmechanismnotestablished, failureskept.5036PASS1001labels/final19.9cm,stablelate14Nloads; initialspike/driftretained.

All37savedcausalreplaysPASS;83imagesindividuallyviewed/hashverified through5036 plusTRAINsummary/5014+5022peakpairs; otherautomaticpreviews notclaimedviewed. No newdiagnosticphysics/model/optimization/sourcechanges. CompletedfullTRAIN393checks/96windows/2000samplerdraws and44input/39officialfullmodel checksunchanged. Remaining3VAL+8TEST beforefull48qualification/preservedthree2000officialUtoniatraining. No trainingqueued orcontrollerretune, no resource restart/duplicate/release. Priornegativepredictorcarry/spikes/partialgeometryscope andgoalreliableprediction/arbitraryshape/demo active/unproven retained.

Latest 2026-09-16T10:00:45+08:00: All32TRAIN response cases COMPLETE,24physicalPASS8FAIL,26lifttriggers,5/8coveredgroups(0/1/4/6/7), original4required gatePASS; allfailures retained. Formal TRAIN_QUALIFICATION andTRAIN_COLLECTION_REPORT saved; summaryplotviewed. FullTRAINCPUreadback4057794/child90166 COMPLETE exit0 01:56:13UTC;393/393cache/loss/samplerchecksPASS across32cases/96histories, cachedvsrawEXACT0,2000matchedbatches/8000exposures, all8groupdraws234/260/283/256/225/247/261/234. Massmaskdoesnotchangesampling;maskedtargetsnograd/unmaskedlossoriginal. No modelupdates. TRAIN_READBACK_SUMMARY.json/md pinreport/sourcehashes; checksproveimplementation/dataeligibility, notaccuracy.

Samecollector3944359 onsole297713/server31/step0 automaticallyenteredVALafterfullTRAINgate; directcompute child111509 Rl elapsed51s runningthirdVAL5034. Saved34/48=81600controls(74400new+7200reuse),26physicalPASS8FAIL. First2VAL5032/5033PASS1001labels each, final12.9/18.0cm; onsetforce dips,5032tilt/heightdrift retained. All34savedcausalreplaysPASS;77imagesindividuallyviewed/hashrecorded through5033 plusTRAINsummary and5014/5022peakpairs. Otherautomaticpreviews notclaimedviewed. Renderer3976006 live; CPUreadbackterminalsuccess/childabsent. Remaining6VAL+8TEST/fullcorpus thenpreservedthree2000officialUtonia study; no trainingqueued, no extrabudget/source/controller/thresholdchange. Prior44input/39fullmodelqualifications retained. Originalnegativepredictorcarry,5014/5022spikes,labelscope/arbitraryshape/demo limitations allpreserved; goalactive/unproven,no duplicate/release/restart.

Latest 2026-09-16T09:46:03+08:00: Originalresponse collectorPGID3944359 continues sole297713/server31/step0; directcompute child47566 Rl elapsed53s running5028. Prefix28/48complete67200saved=60000new+7200reuse,20physicalPASS8FAIL. NEWgroup6allfour settings covered:labels557/1001/1001/1001; groups0/1/4/6 nowfourcovered, meetingnumeric4of8coverage count, BUT full32TRAIN/officialTRAIN_QUALIFICATION stillpending and remainingfourcases retained. No earlygate override ortraininglaunch.5025/26/27 physicalPASS final21.2/20.4/20.7cm; contactfluctuations andholdheightloss retained. Four-groupcoverage isdataeligibility only, notpredictoraccuracy/generalization.

First28savedcausalreplaysPASS;64imagesindividuallyviewed/hashverified through5027 plus5014/5022peakpairs; otherautomaticpreviews notclaimedviewed. Thisturn verifiedwaitedonsamecollector andcompletednewcasephysical/renderinspection; no source/controller/model/budgetchanges. ExistingfullTRAINreadback4057800 waitsall32, renderer3976006 followscollector; existing44input/39fullUtonia/55firstgroupchecks remainPASS, all32samplerpending. ContinueexactremainingTRAIN thenoriginalconditionalVALTEST/fullqualification/preserved3x2000Utonia. Alloriginalfailuredenominators/5014+5022spikes/oldnegativecarryprediction retained; goalreliableprediction/arbitraryshape/demo active/unproven; no restart/duplicate/release.

Latest 2026-09-16T09:39:43+08:00: Continued exact response collection onsole297713/server31/step0/PGID3944359; directcompute child9503 Rl elapsed2:00 running5025. Savedprefix25/48=60000controls(52800new+7200sourceexactreuse),17physicalPASS8FAIL. Group5 complete:5020/21/22 fail(noairborneholdlabels),5023PASS862labels/final18.6cm/holdloss3.1044cm. Reused5024 exactexistingPASS557labels/final19.06cm, norepeatedphysics. Stillonly3coveredgroups0/1/4; groups6/7incomplete, full32TRAIN4of8gatepending. No trainingqueued.

NEW saved-only5022peak diagnosis CPU1906 exit0 01:36:28UTC: maxclearance2.522388cm/zeroairborneholdlabels/sixframes>100N; peakframe1864/time37.30s observed1373.258301/203.061325N agreesfullnormal1373.258179/203.061371N. PeakfullmeshminZ−.713949mm, nextframe−1.51901mm andbothforces0; not anobservationaggregationamplification, contact/groundpenetration chronologyonly, rootcauseunproven. FAILURE_5022_PEAK.json/md pinsinputhashes andnotes0newphysics/model. Posthoc1863/1864actualrenders exit0 01:38:10UTC, bothviewed. No controller/threshold/label/datafilterchanges.

First25causalreplaysPASS,58imagesindividuallyviewed/allhashesverified through5024 plus5014/5022peakpairs; remainingautomaticpreviews notclaimedviewed. Renderer3976006/fullTRAINreadback4057800 continuetheirexistingwaits. Allprior44input/39fullUtonia/55firstgroupqualification preserved, fullTRAINqualificationpending. Prior5014negative andoldfrozenUtoniacarrymass62.47% retained; goodacquisitionstillnotpredictoraccuracy. Goalreliableprediction/arbitraryshape/demo active/unproven; no duplicate/release/restart, continueoriginal32gate thenconditionalVALTEST andqualifiedpreservedthree2000study.

Latest 2026-09-16T09:32:03+08:00: Same response collectorPGID3944359 onsole297713/server31/step0 continues; directcompute child4163790 Rl elapsed1:23 running5021, renderer3976006 andfullTRAINreadback4057800 livewaiting. Prefix21/48 complete50400saved=45600new+4800reuse,15physicalPASS6FAIL. Group4/5016..5019 allfourPASS and1001airborneholdlabels each: thirdcoveredgroup alongside0/1; full32TRAIN/4of8gate remainspending. Actualfinal clearances21.1..22.0cm; localplane spacing42.26..42.27cm/validation3.30..3.38mm isone-dimensional geometrydiagnostic, notneuralreconstruction. Contactonsetspikes/asymmetry andholdheightdecline retained.

New5020FAIL/no lift/zeroairbornelabels: loadpeak7.4099226N thenfalls, finalmean.9021/.6157N, actualrendergrounded/zero fittedplanes/no widthestimate. Allfailedcases kept; no controller/threshold/source changes orrepeats. First21fullcausalreplaysPASS;48imagesindividuallyviewed andallhashesverified, includingnew5016..5020 finalactualmesh+physicalplots. Otherautomaticpreviewframes notclaimedviewed. Currentturn addsactualimageinspection/evidence, no newmodeloptimization/physics beyonddeclaredcollector. Fullinput44/fullUtonia39/firstgroup55 checks retained; completeTRAINsampler reportpending. Continueexactcollector/full32gate/conditionalVALTEST beforepreserved3x2000officialtraining. Prior5014impact/labelscope andoldpredictorcarrymassnegative remain; reliableprediction/arbitraryshape/demo goalactive/unproven, no allocationrelease/duplication.

Latest 2026-09-16T09:23:16+08:00: Samecollector3944359 onsole297713/server31/step0 continues; latestdirectcompute child4112969 Rl running5017. Currentcorpus17/48saved40800controls (36000new/4800reuse),12physicalPASS/5FAIL. Fixedfirst16 groupingcomplete:group0/1covered,group2all4fail,group3threePASSbut5014FAIL/24labels<100, so stillonly2covered amongfirst4groups; fullTRAIN32pending.5013PASS/final16.1cm;5015PASS/final11.5cm withonset~49Nspike retained.

NEW5014failed afterlift, maxclearance8.990997cm,29controls>100N, peakframe1431/time28.64s left3425.241211N versusfullnormalreference3425.239990N/right0, meshminZ−7.737831mm.28.44s still8.8475cm/loads7.75,8.32;28.62s~.1395cm/bothobserved+referencezero. Supports simulatedcontact/landing spike withpenetration, NOTproven rootcause orlearnedphysicalproperty.24briefairbornebilateralphase2 labels retained: labelmask doesNOTguaranteestable/quasistaticbearing. No threshold/label/controlchange orcasefilter. FailureCPUexit0 01:18:13UTC;twoactualposthoc1421/1431 renders exit0 01:20:13UTC. FAILURE_5014_PEAK.md/json preserves chronology/scope.

First16 causalreplayPASS,automaticallcasepreviewfollower3975881 continues.38imagesindividuallyviewed through5015 plus2posthoc5014peakframes; newerautomaticpreviews awaitreview. SamefullTRAINreadback4057794 waitsall32; old44input/39fullofficialUtonia/55firstgroup checks remainPASS, no additionalmodelupdates orphysicsfromdiagnosis. Original4/8gate andconditionalVAL/TEST unchanged; no trainingqueued, goalreliableprediction/arbitraryshape/demo active/unproven.

Latest 2026-09-16T09:13:25+08:00: Prefix13 response cases COMPLETE31200saved=26400new+4800reuse,9physicalPASS4FAIL. NEW5012 originalFAIL nowPASS/lift24s/1001airborneholdlabels/final12.427614cm,bothband100%,holdloss1.575415cm. Lift-onset forceoscillation retained. All13savedcausalreplaysPASS,65automaticimages,30individuallyviewed withscope. Samecollector3944359 directcompute child4066270 Rl running5013 atlastcheck; prior5012 process terminalnormaltransition, no restart. Existingfullmodel39checks alreadyPASS; all32data/samplerreadback4057794/Python4057800 remainswaiting, no additional qualification result claimed. Original4/8TRAINgate/conditionalVALTEST/0trainingupdates preserved; goalactive/unproven.

Latest 2026-09-16T09:12:04+08:00: Same response_surface collectorPGID3944359 continues sole297713/server31/step0; prefix12/48 complete28800saved=24000new+4800reuse,8physicalPASS4FAIL. Group2/5008..5011 allfail/no lift/0masslabels; groups0/1 covered, fullTRAIN gate notyetdecided. New5009final.702962/4.159451N;5010 8.593065/6.208762N;5011 .156412/4.630406N.5009/5011 late observedleftloadcollapse retained,5010final fittedplane disagreement70.3deg; no spacing estimate. Preserved alloriginalconfigs/failures/physicalthresholds, no gainretune. Lastdirectcompute child4054928 running5012, no restart.

Inspection follower3975881 continues; first12 causal replay28800clocks PASS/EXACT,60automaticimages produced,28 individuallyviewed with explicitremainingpreview scope inAGENT_INSPECTION. New qualification codeonly optional --all-train-groups extends existingCPU cache/loss/sampler checks to32originalcases/8groups/96originalwindows/all2000matchedclockdraws; originalfirstgroup path unchanged. Compiled. Bounded CPUreadbackPGID4057794 nowwaits originalcollector fullTRAIN_QUALIFICATION; thenall32physicalreport and fullTRAIN sampler/input readback, evenif coveragefails, no training/physics/model forwards. No checkresult claimed beforeexecution; expectedoutput response_surface_full_training_qualification_v1.

Priornewcondition44input/39fullUtonia/55firstgroupchecks complete,9fullmodel forwards/3backwards/0updates only; notaccuracy. Currentcollection mustfinish32TRAIN thenoriginal4of8gate before16VAL/TEST. No trainingqueued, full48 andfullsamplerpath stillrequired forpreserved3x2000plan. Originalmass/model/arbitraryshape/demo limitations andgoalactive/unproven preserved; no resource release/duplication.

Latest 2026-09-16T09:05:30+08:00: New-condition FULL official Utonia compatibility COMPLETE39/39PASS, full137253744released/138407503adapted parameters and everyofficialtensor exact, all5stage usedgradients finite/nonzero, onlyofficialunused54masktoken hasnone, frozenrepeats EXACT0. GPUqualificationPGID4006991 exit0 at00:59:09UTC, declared9forwards/3backwards/0optimizer; peak3.5232GB. CPUfirst-group cachedcentroid/surface/label/loss/2000draws check55/55PASS, PGID4007003 exit0 at00:59:06UTC. Reports response_surface_input_v1/FULL_MODEL_REPORT.json and response_surface_training_qualification_v1/TRAINING_PATH_REPORT.json; fullTRAIN8group sampler qualification stillpending, noaccuracy/learnedendpoint claim.

Original48collectorPGID3944359 continues sole297713/server31/step0, lastdirectcompute5009 child4023311 Rl. Prefix9/48 complete21600saved=16800new+4800reuse; first8PASS,5008FAIL/no lift/0holdlabels. Groups0 and1 each covered (group1 labels925/1001/812/1001), full32TRAINgate stillpending.5008final loads6.80895/6.33702N; local fittedplane normals34deg apart, no renderedwidth estimate. Its load rose near targetearlythenfell despitebothgains later.00015; failure NOT simply originalfixedlowgain. Retain failedcase andexactprotocol, no retune.5006brief nearzero leftload around28s retained before successfullift.

Inspection followerPGID3975881/Python3976006 continues; first9 full causal replay21600clocks PASS/EXACTgain+response+counts;45automaticimages (4mesh+1trace each),22individuallyviewed withexplicitremainingpreview scope. Physicalrise/hold drift/nonparallelfailedcontacts keptinrenders. No wholebodypolicy orneuralestimate visuals implied. Priorfirstgroup2.18..4.30%shearonly physicaldiagnostic remainsoneTRAINgeometry, notgeneralized.

Continue exactsame collector throughall32TRAIN beforeoriginal4/8gate, thenconditionalVAL/TEST, allfailures retained. No trainingqueued; preserved3x2000plan awaits fullnewcondition corpus/input/sampler qualification. Original27padnegativebaseline, frozenfullmodelmass62.47%failure, arbitraryshape/demo objectiveunproven; goalactive, no duplicate/release/restart.

Latest 2026-09-16T08:56:45+08:00: response_surface_dataset_v1 continues on sole297713/server31/step0 collectorPGID3944359, directcompute child3990141 Rl running5005 atlastcheck. First5/48 complete, allphysicalPASS;12000savedcontrols=9600new+2400reused. Group0 allfour5000..5003 have1001airborneholdlabels each, firstgroup covered but fullTRAIN gate pending. NEW5004 (oldoriginalFAIL) PASS/lift25.52s/925actualairborneholdlabels/bothband100%/peak12.455945N/holdloss2.189997cm. Original/new differs sensor+gain; no isolatedcausal claim.

Saved-case inspection followerPGID3975881/Python3976006 LIVE, waits on originalcollector then exits; no physics/model. First5 causal gain/response/count replay all12000clocks EXACT0, all25plots/actualmesh snapshots produced (4fixedframes+1physicalplot percase). Fourteen images individuallyviewed, explicit remainingpreview scope inAGENT_INSPECTION; currentpage28linksvalid. Local plane extent is diagnostic only; final firstgroup separation29.86..29.88cm/fullmesh-width discrepancy7.47..8.03mm retained,5004 separation26.80cm/discrepancy5.63mm; no shapecompletion/neuralprediction claim.

CPU3984398 exit0 at00:53:12UTC: firstoriginalgeometrygroup fixed44..48s tactile-only verticalshear/9.81 analytic diagnostic complete,4x201validframes. True.4542kg ->.4443/.4657kg at12/24N; true.9145kg ->.8752/.8945kg. Errors2.178/2.536/4.299/2.191% versus SAMEnewtrajectory anatomical27 validation12.966/10.051/8.921/13.109%. FIRST_GROUP_SIGNAL files preserveall4; no calibration/solverforce estimate, omitsnormalcontribution/acceleration; oneTRAINgeometry NOTheldout/model/generalization. No newmodel forwards/updates.

Continue same alloriginalTRAIN qualification, no repeats/sweeps/failure removal, collectVAL/TEST only afteroriginal4/8gate. Prior44input checks notfullmodel qualification, pendingnewcondition fullmodel/sampler paths before any preservedthree2000training. Old32baseline2/8andfullUtonia62.47%successfulcarrymassfailure retained; globalgoalactive/unproven.

Latest 2026-09-16T08:48:41+08:00: Explicit continuation after completed three-case report: ORIGINAL48 configuration acquisition qualification started under separately named response_surface_dataset_v1, continuous_palmar_v1 + response_gain. Original configs/splits EXACT32TRAIN/8VAL/8TEST and original4/8groups xfoursettings>=100label gate retained. Reuse only identical-source/config completed TRAIN5000/5007/5024; low-level diagnostic metadata retained, no retroactive heldout claim. Planned29newTRAIN69600controls then, ONLY if TRAIN qualifies,16VAL/TEST38400; maximum108000new+7200reuse=115200saved. All failures retained; no sweep/repeats/newmodel optimization. No automatic training queued; formal new-condition model qualification still required.

Sole297713/server31/step0 collectorPGID3944359 launched00:44:53UTC; directcompute child3956596 Rl running5002 atlastcheck. Completed current orderedprefix2/48: reused5000 and NEW5001 bothPASS; new5001 all10criteria,1001airborneholdlabels,bothband100%,peak24.8025N,holdloss1.61575cm. Currentrecordedcontrols4800=2400new+2400reuse; remainingtwo declaredreusedcases encounteredlaterinoriginalorder. No fullgroup/32TRAIN qualification claimed.

Extended existing qualify_surface_input CLI only to accept explicit saved TRAINcases; original encoder/collector/controller/physics unchanged. CPUnew-condition6windows on5000/5007/5024 completed44/44checks PASS; load decode1.85512e-6N/area7.4393e-13m2/shear1.33712e-7N, geometryblind/forcezero/targetpoison/subdivision/worldtransform checks retained. CPU3953709 exit0 at00:47:01UTC; no full-model forward/optimizer/physics for this input check. Report response_surface_input_v1/INPUT_REPORT.json, not accuracy/fullcorpus/model qualification.

Previous three-case720framevideo/report remains complete and all negative fullUtonia carry evidence retained. Goal reliableprediction/arbitraryshape/demo benefit active/unproven. Continue same collector without duplicate/restart/release; finish allTRAIN before gate, inspect outcomes and renders, then new-condition full-model/input/sampler qualification before any preserved planned training.

Latest 2026-09-16T08:41:13+08:00: Response-gain fixed-three-case acquisition diagnostic COMPLETE/POSITIVE. All original numeric physical/stability checks pass under explicit continuous_palmar_v1 sensor.5024 previously no lift ->32.88s lift/final19.062516cm/557 airborne labels (507 stricter collector hold-window frames);5000 retains PASS/final20.205356cm;5007 prior target-stability FAIL ->PASS/final14.998047cm. Both-hand target-band fractions100% all three.5007 final height lower than original16.471206cm and hold loss1.720050cm vs.585055cm, retained;5007 changes sensor ANDgain, no isolated gain effect. Original/new trajectories not exact checkpoint forks.

24tests PASS;7200 actual controls/0 model forwards/0 optimizer updates, all7200 causal gain/response/sample-count replay EXACT0. DefaultOFF adaptive response uses past observed force/command displacement; no material-stiffness or nonlinear-stability proof. CollectionPGID3855488 exit0 at00:31:53UTC. RendererPGID3866449/Python3888248 exit0 at00:39:22UTC and directcompute renderer absent. Actual full-mesh720frame72s10fps video decoded all720, exact3x240 case-clock order, 7 pageasset links valid. Seven images individually viewed with explicit scope in AGENT_INSPECTION; not every video frame visually reviewed. Same original meshes/poses, local interface separation diagnostic only, no learned prediction.

Report/index at response_gain_diagnostic_v1 complete; old fullUtonia frozen carry failure (original10physicalPASS hold mass force62.47%,geometry41.93%,zero30.16%) linked, not superseded by physical acquisition success. Original32 acquisition coverage2/8 vs4required unchanged; threecases cannot qualify full new-condition dataset. No extra trial/training queued. Sole297713/server31/step0 retained; goal reliable predictor/arbitrary shape/demo benefit active/unproven. NEXT full original-configuration acquisition qualification under explicitly new sensor/controller before declared matched Utonia training; no relaxed gates or automatic success claim.

Latest 2026-09-16T08:31:19+08:00: Force-response bounded controller diagnostic IN PROGRESS. Saved34case/68hand response audit completedCPU3831227 exit0 at00:20:42UTC, no physics; continuous5024 30..40s speed.1822/.1875mm/s, positive0.5ssecant medians2779/2214N/m ->local approximate14.39/18.06s timeconstants, NOT materialstiffness/causalID. New defaultOFF response_gain usescausal .1s positiveforce/displacement secants,>=5samples in2s/recentmax retainedwhenquiet, gain<=.00015 and<=.1/(dt*K), gainrise<=5%/step, dropimmediate, abovebaseonlyvalidplane. No nonlinearstabilityclaim. New sensorcontinuous_palmar_v1 explicit; originalphysics/align/lift24..40/thresholds unchanged.24tests PASS.

Declared3cases5024,5000,5007 x2400=7200controls/0modelupdates undercollectionPGID3855488 sole297713/server31/step0. First5024 COMPLETEPASS/lift32.88s/hold507frames/band100%both/peak12.3787N; previouscontinuous caseFAIL retained.5000 COMPLETEPASS/lift24s/band100%both/peak12.6048N. Third5007 LIVE Python3876285 Rl directcompute latestframe800/time16.02s; endpointpending.5007 referenceoldanatomical27, so sensor ANDgain differ, no isolatedgaincausalclaim.

Declared finish child waitingcollectionexit0 thenreport causalallframe gain replay and720frame actualmesh video; report.py/render_cases.py/finish.sh compiled. No extra trial/training beyond3caseprotocol. Pending5007fulloutcome/all7200replay/finalrenderinspection/fullreport. Original32coverage2of8vs4notreplaced, fullUtonia unchanged andnewtrainingnotstarted. Goalactive/unproven.

Latest 2026-09-16T08:18:21+08:00: Continuous-palmar fixedpair REPORT/ACTUALVIDEO COMPLETE; mixedsensing result, physicalpairFAIL. DefaultOFF continuous_palmar_v1 preservesoriginalassignedIDs andallphysicalfieldchannels, routesonlyadditionaldeclaredpalmarface observationtonearest54aggregationbin; anatomical27baseline retained.18tests/256saved32integrationchecks PASS. Final4800savedcontrols/4801executed (1first-control proxyreadbackfailure) and0model forwards/updates. BothJSONserialization andfirst-controlreadback incidents archived, fixesonlyreport/parallelbaselineview; no completecase repeated.

5024stillnolift/nohold/final8.6404/8.5863N matchesfullcontactscalarload yetbelow9Nreadyminimum;5000allnewconditioncriteriaPASS/lift24s/1001airbornelabels/final20.235855cm. SAME new5000trajectory late44..48s201frames original27shear mass13.057629% versuscontinuous2.162537%; single-case sensoranalytic positive, NOTtrainedUtonia/heldout/policybenefit. New/oldtrajectoryprefixNOTexact,5024maxposecomponent.01520 includesquaternion, notcmtranslation; same-statecoveragecomparison distinctfromclosed-loopcausalclaim. Both full4800frame routing/integrationreadbacks PASS maxload2.54e-6N/shear6.36e-7N.

Finalactualmesh480frame48s10fps2x video decodedall480/nonconstant; fixedcase/frameorderchecked. Fivefinalimages individuallyviewed/AGENT_INSPECTION,6pageassetlinksvalid. RendererPGID3767718/Python3785919 exit0 at00:17:09UTC, directcomputechildabsent; collectorPGID3762478 exit0 at00:11:42UTC. Sole297713/server31/step0retained. Reportsatcontinuous_palmar_diagnostic_v1/index.html. Currentcomparisoncomplete, noadditionaltrial/trainingqueued. NEXT addressactualforcebuild-up/qualified acquisition; original32coverage2of8vs4unchanged, continuousfullcorpusnotcollected. OriginalfullUtonia/remainingdeclaredtraining unchanged; reliablepredictor/arbitraryshape/demo goalactive/unproven.

Latest 2026-09-16T08:15:10+08:00: Explicit continuous_palmar_v1 sensor adapter implemented DEFAULT OFF, originalanatomical27 unchanged. OriginalassignedIDs retained, newpalmarfaces nearestXZfootprint for54binaggregation; originaldensepositions/area/pressure/shear unchanged, no fullsolver/objecttruth fedtocontroller/model. Same newtrajectories record anatomical baseline channels andoriginalpad labels.18tests PASS, original32/3040frames256integrationchecks PASS,50944newcoveredfaceframes. All originalmodel/training budgets unchanged.

Fixed5024+5000 physics COMPLETE4800savedcontrols/4801executed (onefirst-control readbackerror),0modelupdates. Initialqualification JSONnumpybool serialization failed beforephysics, fixedbuiltinbool; next first-control baselineproxy lackedtactile auditattribute, failedafter1step0saved; originallogs/incidents preserved, fixedshallowcopyfieldview. No completecase repeated. FinalcollectorPGID3762478 exit0 at00:11:42UTC.5024FAIL/nolift/final8.6404/8.5863Nnowmatchesallcontactscalarload, so insufficientphysicalbuild-up remains;5000PASS/lift24s/1001airborne labels/final20.235855cm. SameNEW5000trajectory late44s201frames: anatomical shear mass13.057629% vscontinuous2.162537%, analytic sensing comparison NOT Utonia/no heldoutclaim. Prefixnotbitexact,5024handposecomponentmax.01520 includesquaternion; cannotclaimstrictcausalold/newrollout. Both4800frame routing/readback PASS, maxload2.54e-6N/shear6.36e-7N.

Report/COMPARISON saved. Actual480framepair rendererPGID3767718/Python3785919 LIVE Rl directcompute atlastcheck; comparisonplot andframe0120 inspected, restvideo/posters/finaldecode/link/inspection checks pending. Sole297713/server31/step0 retained. Finish declaredvideo andfullreport, no newtrial/training queued. Original32coverage2of8vs4 gate stillfails, continuousfullcorpusnotcollected. Reliablepredictor goalactive/unproven.

Latest 2026-09-16T07:57:38+08:00: Saved32 uncovered-skin localization COMPLETE,3040originalclocks/248398face-frame records:197454assigned,50944palmaroutsidefootprints,0oppositehalfspace. Originalfield27rectangle/actualscene(-1,+1)halfspace replay all32 EXACTlabels, no sensor/physics/modelchange. Unassignedsupport partition matchesprior max6.217e-15N. Late44 all14/32,112frames palmaroutside19.933007%weight; original10PASS80frames18.383763%; otherhalfspace0. Coordinatehalfspace NOTgeneral anatomicaldorsal proof.

Original10PASS late44 uncovered136handframes: normal-load-weighted nearestpatch sharespalm_r1_c0 53.713567%,palm_r2_c0 46.165903%,ring_distal .120531%;99.879469%nearwristpalm. Weightisnormal-load NOTverticalsupport. Perhandframe weightedp50distance.095858..2.138088mm,p95 .157162..2.759108mm, distancesplanarXZ/nongeodesic/no promise2.76mmcoversall. Fouractualmesh/footprintimages fixed5000/5001 at47.64s individuallyinspected,report/index7linkscomplete. CPUauditPGID3700576 exit0 at23:52:50UTC;rendererPGID3708582 exit0 at23:54:54UTC, directcomputePIDs absent. Sole297713/server31/step0retained.

NEXT explicitly separatecontinuouspalmar sensorcondition/adapter from original27padbaseline; do not silentlyroute pad-1/fullsolver/objecttruth into existingmodel. Newcoverage improvement notyet tested; controlleroriginal2of8vs4gate stillfails, no pendingtrainingstarted, fullofficialUtonia unchanged. Reliablecarryingpredictor/shape/demo goalactive/unproven, allpriorfailuresretained.

Latest 2026-09-16T07:50:33+08:00: Saved32 force-deficit decomposition COMPLETE,3040originalclocks/no newphysics/model/optimization. Original10physicalPASS late44s80frames signeddeficit21.367408%weight decomposes unassigned18.383763pp,normal2.411962pp,fieldprojection.571683pp,padreadout.000000577pp. Currentobserved shear masserror21.370703% vs ALLSURFACE validation-only3.579695% vs fullsolvervalidation.018890%; all14of32/112frames23.486165/3.976554/.033227%. Missing18retained, no fullsurface truth fedtomodel, no coverage promotion/calibration. Mainobserved deficit hereiscoverage, not primarilyprojection; neuralforce65.68% stillworse than observed21.37 so not sole model-failure explanation.

Independent originaltraction replaymax1.524415e-7N; allfacesnormal scalarclosure1.524526e-5N; assignedpadreadout1.073271e-6N; decomposition8.88e-16N.96oldmetricrows agree5.68e-14pp. CPUauditPGID3672800 exit0 at23:46:40UTC, actualfullmesh twofixed5000/5001 at47.64s rendererPGID3680549 exit0 at23:48:23UTC. Threefinalimages individuallyinspected/AGENT_INSPECTION,report/index6linksvalid. Directcompute bothPIDs absent,sole297713/server31/step0retained. NEXT locateunassigned contacts in knownhandgeometry beforedeclaringnew sensorcoverage; no silentlyrouting fullsolver/unobservedfield into originalobservations. Original27pads/hand baseline/2of8vs4 gate andallfrozenmodel negatives retained; reliablepredictor goalactive/unproven.

Latest 2026-09-16T07:43:13+08:00: Bounded acquisition gain-schedule pair COMPLETE/overallFAIL. DefaultOFF normalized_acquisition doubles12N prelift gain atloads<=6N, fades tooriginal at9N;24N/nearband/postlift/unlatchedapproach/readiness/physics unchanged.13controller tests PASS. Two2400case traces/4800controls,0model forwards/updates; collectionPGID3567499 exit0 at23:32:43UTC.5024 stillnolift/0airborne labels:late44means9.066117/8.754937N versusoriginal8.540255/8.380588, readinessalways0;5000alloriginalpasses/final20.159395cm versus20.380382cm. Gainreadback errors9.54e-7/7.15e-7. Preinterventionphysicalprefix NOTbitexact; handposes/distance exact but5024 objectposemax9.35e-5/load.08985N;5000pose5.96e-7/load.000612N. No strictsameprefix causalclaim. Modification notpromoted, original32cases/coverage2of8vs4required unchanged.

Actualmesh paired480frame/48s/10fps2x video COMPLETE, finishPGID3584330 exit0 at23:36:44UTC; all480decoded/exactepisode-clockorder,6pageassets exist. Fiveimages individuallyviewed andAGENT_INSPECTION saved; not everyvideoframe humanreview. Invalidfit180deg sentinel andnohold stability sentinel explicitlynot physicalerror measurements. REPORT/index under acquisition_schedule_diagnostic_v1 complete. Directcompute currentchildPIDs absent; sole297713/server31/step0 retainedRUNNING. No extra trial/training queued. Currentstage complete/report; reliablepredictor goalactive/unproven. NEXT distinguish acquisitioncoverage frompredictionfailure; priorfull32frozen andforce-observability evidence remains authoritative. No blindgain sweep, originalqualificationgate notrelaxed, fullofficialUtonia unchanged.

Latest 2026-09-16T07:22:30+08:00: Saved32 mechanics-observability diagnostic COMPLETE, no new physics/model forwards/optimization. Existing95clocks each/3040total, observation-only shear versus CADnormal*load+shear versus existing surface_data pad-local fittednormal*load+shear; same recorded shear heldfixed, no target calibration/clipping/true interface-normal inputs. Fullsolverforce strictlyvalidation. Original10physicalPASS late44s/80airborne frames errors21.370703/63.068299/24.042735%, validationfullsolver .018890%. All14covered/32 latehold112frames errors23.486165/58.639683/23.075305%, validation .033227%; missing18retained. CAD5001 all8late estimatesnegative; current geometric normal is NOT automatically actual force direction. DoesNOTprove neural model internally multipliesCADnormal/load or isolate its failurecause.

Matched oldmodel readback on same10cases/80clocks checked from savedCONTEXT: geometry/contact/force/zero mass42.230065/40.908167/65.683923/29.661513%. Existingobservations have useful bearing signal not captured by oldmodel; naivephysicsstillfails, no newpredictor improvement. Active dense/sparse load maxerror3.65492e-6N, shearcomponentdifference.000638547N, maxexcluded subthresholdload.000968746N. No force-direction calibration or globalrescale. Pressure-normal and traction approximations/partialpalmar limitations retained.

CPUaudit PGID3507830/Python3507836 exit0 at23:17:17UTC. Two actual fullmesh force-vector illustrations5000(firstsentinel)/5001(posthocnegativeexample) fixed47.64s complete; arrow0.05m/N, shiftedorigins explicitlynottrueforceapplicationpoints, truegravityvalidationonly.5001 actual20.89cm airborne/.454kg but CAD -.426kg, shear.412kg/fitted.344kg. RenderPGID/process under mechanics_observability_v1 exit0 at23:20:19UTC. All3 images viewed individually; reports/index/MATCHED_SAVED_READBACK andAGENT_INSPECTION complete. No newtraining/collection, original4of8coveragegate and budgetsunchanged. Sole297713/server31/step0 retained; currentchildren terminal.

NEXT explicit geometry/normalforce/tangentialbearing/unknownsupport semantics and uncertainty in auxiliary supervision, after resolvingoriginal acquisition2/8coverageversus4required; do not treat CAD-vector diagnostic as trainedUtonia or silentlyuse solvertruth asinput. Reliablecarrying predictor goalactive/unproven; preserve original full32 frozenfailures and earlierphysicalcontroller/SDF/inversion negatives.

Latest 2026-09-16T07:11:41+08:00: Frozen full officialUtonia original32TRAIN transfer evaluation and actual-versus-prediction rendering COMPLETE, overall carrying predictor FAIL. All32 x95clocks x4modes=12160savedoutputs,224matchedclock/targetchecks PASS; recovery inference exit0 at23:03:06UTC. Reused original10cases unchanged,16reloadforwards EXACT0; interrupted unsaved0..380 forwards separately retained, no new physics/optimization. Full138407503 models frozen at original2000 endpoints, no reduced architecture.

Full32 equalepisode mass geometry/contact/force/zero41.466807/41.212734/64.241197/35.108767%. Actualairbornehold14/32cases481windows: center10.759144/6.197106/11.065234/10.475048cm, rotation79.288276/87.175955/81.863585/85.461064deg, size8.355681/12.376126/10.430976/19.863295%, mass37.196586/37.106667/49.165651/35.558603%. Original10physicalPASS cases367holdwindows still mass41.933686/40.865325/62.472924/30.161653%; successfulacquisition doesNOTsolve prediction. Forcezero worsenssize, not universalchannelremovalfix. Late44s means true mass.341835.. .965837kg versusforce.845754..1.095023kg, geometry.660359.. .686437kg. No heldoutnewTEST orcausalinternalmechanism claim; rotation originalasset-axis/nosymmetryreduction.

Fourfixed videos5000/5005/5011/5014 all95framesdecoded/nonconstant (380total), 10fps5x;64snapshots for all32 complete. Actualfullmeshes/rawpredictions/sameclocks/cameras; no smoothing/truthcorrection or arbitraryshapereconstruction. Humanreview eight sheets covering64snapshots +16fullsize video sourceframes +2plots, scope explicit inAGENT_INSPECTION.json (not everyvideoframe humanview). Sourcevideo5000 lifts20.41cm at47.64s but0.454kg->1.041kg;5014tippingfailure retained despite lowercentererror. Controller hold phase doesnot implytrueairbornehold, page/report clarify. All7static+64dynamicpageassetlinks exist. Finalreport/render PGID3378043/Python3437290 exit0 at23:09:22UTC, no remaining child; sole297713/server31/step0 retained.

REPORT.md now explains training-action transfer gap, partial27palmar coverage and approximate pressure-distributed/projected shear, forcezero intervention and failedacquisitionversusfailedprediction; all original limits retained. Rendering/reports at frozen_surface_carry_v1/index.html. STOP this completed frozencomparison and report; no appended collection/training/controller sweep. Reliablecarryingpredictor goalactive/unproven. NEXT address explicit physicalsignals/observability and original acquisitioncoverage2/8vs4 gate before pendingthree2000/newTEST; do not replay completed frozen evaluation or silently relaxgate. Prior oldgrip/native mass negatives and all SDF/deadband/inversion failures retained.

Latest 2026-09-16T06:53:42+08:00: Original32-case frozen full-Utonia transfer evaluation IN PROGRESS,17/32 cases complete through5016; no new physics or optimization. External cancellation297659 by0 at06:41:03 (stepSIGKILL06:41:34) interrupted original evaluator. Ten complete cases5000..5009/50files and all32cache preserved; unfinished5010 discarded-forward count unknown0..380. INCIDENT/RECOVERY retain this accounting. Empty queue checked before sole replacement297713/server31/explicitstep0; first allocation attempt withoutQoS rejected before allocation, correct userQoS granted. No duplicate or voluntary release.

Recovery PGID3350460/Python3350467 directly verified Rl,93.5%CPU elapsed6:08. All32cache/50completedfiles unchanged; complete three endpoint/model records match;16 fixed reload forwards EXACT0. Original95clocks x32cases x4modes=12160savedoutputs, plus16replay and0..380interrupted unsaved executions. Prior immutable8-case partial is not final32 evidence. Three official full138407503-parameter models unchanged, frozen2000 endpoints/H32; forcezero preserves all other input features. Original acquisition TRAIN fixtures, not new heldout TEST.

Four actual-versus-raw-prediction full-mesh previews complete/inspected at5000+5005 x23.14/47.64s, previewPGID3355142 exit0 22:47:44UTC. Successful lift5000 final20.41cm but mass0.454kg truth versus1.041kg force prediction (129.2% error); singleframeexample, not aggregate.5005 no lift, failedcase retained. PREVIEW_INSPECTION.json records viewed images. Predictive mesh uses knownfamily geometry with raw pose/size, not arbitraryshape reconstruction.

Guarded report/render childPGID3378043/bash3378050 queued on SAMEstep, confirmed waiting for evaluator exit0 and complete12160outputs. Then full32report plus predeclared5000/5005/5011/5014 videos and64fixedsnapshots; no new forward/optimization or collection. finish.sh syntax passed; initial local preparation attempt used unavailable python executable and performed no writes/launches, corrected to python3. NEXT verify full inference/report/render completion and inspect outputs; no additional experiment queued. Original surfacecoverage2/8 vs4required and allprior mass/native/controller negatives retained; reliablecarrying predictor goalactive/unproven.

Latest 2026-09-16T06:27+0800: SDF resolution diagnostic COMPLETE/physicaloverallFAIL,4800newcontrols/0modelupdates. First saved-original32 deadline audit:17nolift,0reach1sready after40s;5029max.32s,others0. Independent one-observation-lag readiness replay EXACT0. Initialdraft mistakenlyshifted commandclock-.02s, preservedINITIAL_CLOCK then sourcecorrected: commandtime==savedtimestamp. No timingchange basedondraft. CAD-only hull enumerationunder originalsignedY>.4 andthreedistinctdigitpadvertices finds0candidates bothhands; originaltriadwristpad3/middle20/little26 retained. This doesNOTprove multi-digit softcontactimpossible; no rule relaxation/newhandpose.

OfficialobjectSDF saved-point128vs256 check COMPLETE PGID3163380 exit0 at22:12:04UTC, fixed5011/5005/5000 x50frames/twohands,0physics/forwards.5011 objectSDF-to-mesh angularerrors left18.0311->9.4119deg,right5.2499->1.9243; leftunsigneddistanceerror.517887->.121655mm.5005/5000 alreadysmall <.1deg.256residualerror/positiveSDFpointfractions retained; old128contacts NOTnew256contacttrajectory. Objecttruth strictlyvalidationonly. No512sweep.

Added optional object_sdf_resolution128(default)/256 to SupportScene and diagnostic-only collectorCLI. Only objectofficial Mesh.build_sdf resolutionchanges; hand128/fullmeshes/material/solver/dt/servo unchanged; protocol explicitly sets original_full_mesh_and_physics FALSE for256 andrecordschange. No deadband/sustained/SDFnormal correction enabled. Fixed5011+5000 PGID3172687 exit0 at22:18:36UTC.5011FAIL/nolift/noholdlabels/finalclearance-.012579cm,finalpalmar3.3296/2.4835N versus24Ntargets;5000PASS/final20.240078cm versusold20.380382cm/1001labels. Not allmetricnonregression ornewpredictorbenefit; originaldata unchanged.

New observed-normal opposition metric covers34traces xfixed23..24/39..40s=68windowrecords. acos(-nL dot nR) onlyoriginalbothfitvalid; missingcountsretained. Original5011fixed23:50/50valid,79.8105deg mean;39s40.8448deg.5010fixed23:49/50,55.2602deg.10passing cases all50valid/<2.011deg, BUT manyfails alsoopposed (5014 .111deg/5005 .906deg); NOTnewsuccessgate ortrueobjectnormal/forceclosureproof.256case5011bothwindows0/50valid despitepositiveforces, not zeroerror. Fixedactual24s transparentfullmesh rendersobservationsusedby24.02scontroller:128bothvalid81.3295deg;256leftinvalid/rightvalid/nullopposition, nofabricatedarrow.

sdf_resolution_diagnostic_v1 REPORT/NUMERICAL_REPORT/index/480frameactualmeshvideo COMPLETE, all480decoded/nonconstant; fourfinalimages inspected/hashboundAGENT_INSPECTION.json. Video48s/10fps/two48s traces2x,lastsample47.82s+48sfinalposters. Directionrender exit0 at22:23:23UTC, report/videoPGID3198325 exit0 at22:23:33. No extra experimentqueued; allchildren terminal, sole297659/server31/step0 retained.

NEXT return to full officialUtonia frozen evaluation on saved original32TRAIN trajectories to relate geometry/mass estimation errors to observedcontact validity/opposition, before furthercontrollertrials. Use existing training_mixed_geometry_mass_surface_v1 threefullmodel endpoints and qualified originalinput path; no newoptimization/reducedmodel. Preserve originalsurface32/48corpus76800saved/10passes/14labelcases/12052labels/coverage2of8 versus4required FAIL, pendingthree2000/newTESTpredictionrenders, oldgripmass69.862%/native66.50% negatives and all previousSDF/deadband/inversion failures. Reliablecarryingpredictor goalactive/unproven; frozen evaluation mustnotbeclaimednewtraining orheldout acceptance.

Latest 2026-09-16T06:03+08:00: Fixed alignment-deadband physical pair COMPLETE/overallFAIL,4800newcontrols/0modelupdates. Source diagnosis: six original aligned/no-lift cases each800frames24..40s stillrotate1.34..3.27deg whileunderload. Added defaultOFF diagnostic-only alignment_deadband: from24s suppressalignment inside original5deg readinesscone, unchangedforcegain.000025/nointegral/physics/path/readiness.10controller tests PASS (7old+3early-command/force-preservation/outside-cone invariants). collect_dense_grip now records optionalalignment_active andexplicit surfacecontroller sourceSHA. No SDFnormal correction/modelreplacement.

Pair5005+5000 PGID3072657 exit0 at21:57:55UTC.5005FAIL/no lift/noholdlabels, finalclearance-.012514cm; aligned24..40s bothrotationpath EXACT0deg vsoriginal2.5569/2.9146deg. Meanpalmarloads15.485/22.037N vsoriginal15.130/17.086,target24. Finalpalmar16.216/30.073N whilevalidationfullnormal32.917/31.715N; partialcoverage matters, NOTcompleteforceinput orstableholdmeans.5000PASS alloriginalcriteria/final20.384203cm/1001labels (original20.380382cm). Pre24s1199frames NOTbitexact: maxpadload1.16436N5005/.062911N5000; full pose/control differences retained. Tests prove algorithm branch invariants, NOTidenticalphysicalprefix orstrictcausalattribution. No promotion/extra gaintrials/training.

Report/actualmesh video COMPLETE: alignment_deadband_diagnostic_v1/REPORT.md,index.html, renders/alignment_deadband_pair.mp4.480frames10fps48s,two original48s traces2x, every10controls,lastsample47.82s+separate48sfinalposters. All480decoded/nonconstant; final4rowplot,2posters,2sourceframes inspected/hashbound AGENT_INSPECTION.json. Actualfullmeshes/poses,fixedsamecameras; farhandnaturallyoccludedbybox, notdeleted. Localinterfacewidth is diagnostic, NOTUtoniaprediction/correctshape. Report/renderPGID3096994 exit0 at22:01:54UTC; label-layout cleanupCPU exit0 at22:02:35. Allchildren terminal, sole297659/server31/step0retained.

Prior full32 inversion3200valid/13of64regress/5011SDFassumptionfailures preserved. Original32/48corpus76800saved/10passes/14labelcases/12052labels/coverage2of8 versus4required stillFAIL; diagnosticcasesneverreplace originals. OfficialfullUtonia138407503 unchanged, pendingthree2000/fullTESTpredictorrenders unexecuted; oldcontrolledgripmass69.862%/native66.50% negatives unresolved, goalactive/unproven. NEXT prioritizeobserved contactcoverage/layout andacquisition-readiness deadline limits; no blinddirection/gainchanges ornewtraining before originaldataqualification.

Latest 2026-09-16T05:49+08:00: Full original32TRAIN pad-local interface-normal inversion COMPLETE, fixed23.00..23.98s/50frames/twohands =3200/3200 valid outputs,0newphysics/model/optimizer. All64 hand/case rows retained;51/64 meet complete-output/+0.5deg diagnostic nonregression,13regress. Mean acute (abs-dot, NOT signed) angle4.082977->3.850839deg does NOT establish broad improvement. Original two-case200rows valid/patches/area/errors EXACT200/200. alltrain PGID3024726 exit0 at21:43:04UTC; report exit0 at21:43:46. Largest regression5011left26.356537->32.272019deg (+5.915482);5008..5011 left raw errors alreadylarge. No estimator/controller change or relaxed collection gate.

Post-hoc worst5011 all50frames/twohands patch diagnosis and officialSDF validation COMPLETE,0physics/model/updates. Leftpad3/20 localplane vs grad_hand-grad_object mean11.5913/8.1375deg; objectSDFgrad vsactualmesh16.7568/26.2364deg, objectgradientnorm.86389/.87443. Rightpad30 corresponding9.0136/5.2499deg/.95765; leftpad23 only4points/ineligible has.0697deg SDFmesh difference. Evidence that finite padplane and unit-gradient/true-mesh assumptions are inaccurate here, NOT universalfailurecause or correction fix. ObjectSDF/pose/mesh strictlyvalidationonly, savedestimates untouched; originalofficialbuildconfig128,objband+/-.006/margin.004. worst_sdf exit0 at21:46:51UTC.

All32 comparisonplot and5011 actualfullmesh/raw-corrected-truth directionrender COMPLETE/viewed/hashbound ALLTRAIN_IMAGE_INSPECTION.json; render exit0 at21:45:25UTC. Firstframe23srender metrics differ from50frameprimary; trutharrow usescontactcenter nearestnormal, anglesarea-averageallpoints. Full report sustained_grip_diagnostic_v1/contact_layout/ALLTRAIN_REPORT.md,index.html updated, prior2case positives andfirstwholehandfailure retained. NEXT inspect local contact geometry/SDF discretization and uncertainty before promotingdirectionto controller/Utonia; do notblindlyexpand corrected-controllerphysics or training. Prior32/48corpus76800saved/10passes/14labelcases/12052labels/coverage2of8 versus4required remainsFAIL; noVALTEST/three2000trained. OfficialfullUtonia138407503 unchanged, oldcontrolledgripmass69.862%/native66.50% negatives unresolved. Reliablecarryingpredictor goalactive/unproven; sole297659/server31/step0 retained, currentdiagnosticchildren terminal.

Latest 2026-09-16T05:35+08:00: NEW local geometric positive result, NOT carrying/predictor success. Saved contact-layout diagnosis covers original32TRAIN plus two sustained-feedback cases/34traces/85fixed windows,0physics/model/updates; CPU PGID2874963 exit0 at21:10:02UTC.5014 prelift bothhands onlypalm_r1_c0,11.8/11.9mm2 area,2.42/2.44mm large-tangentRMS, wrist-side edge contacts verified onfullhandCAD (X.13..1.54mm); not generalwholepalmsupport or pureendcap misassignment proof.5000 has4/3pads,62.3/27.7mmRMS. Sevenboth-singlepad baselinecases allfail; ten>=3/3padcases alsofail, so no newacceptance gate. Tenpassing final4s fullhandupforce/mg=.99979..1.00005 but observedverticalshear/mg=.6482.. .9101; partialcoverage plus perhandpressure-distributed/projected traction, not a universalcalibrationfactor.

KnownCAD andactualfullmesh diagnostics COMPLETE:5014 candidatefingervertices outsideactualobject by10.5..11.5mm while5000tipsdistance.21.. .57mm. Localinterface-plane extrapolation is not fullobjectgeometry. Interface-to-object nearestnormalerrors at24s5014 5.377/5.608deg versus5000 .444/1.166deg; interfacepoints themselves within.13.. .24mm ofactualsurface. OfficialNewton sdf_hydroelastic.py defines interface as equalpressure p=-kh*signed_depth, not originalobjectsurface. Originalcurrentmaterialkh equal, hand geometry known; objectstates/normals validationonly.

Two fixed50frame23.00..23.98s/case/hand officialhandSDF inversion diagnostics COMPLETE, nolearnedmodel. ExactofficialMesh.build_sdf hand128/narrowband+/-.004/margin.002 andtexture_sample_sdf_grad; knownkh ratio1, localobjectunitSDFgradient assumption. Tangential handSDFgradient yields objecttangentgradient; recovernormalcomponent, no clampifnorm>1. FirstWHOLEHAND inversionFAIL, reportretained:5014 angles5.4095/5.5933 ->.9703/1.1081deg, but5000 .44535/1.15269 ->4.69229/.50194deg. Finite multi-region interface is not one local tangentplane.

Second anatomical-pad-local inversion then fixedarea-weighted normalaggregation PASS originalfourcriteria,200/200handframes valid; same5014 .97025/1.10812deg,5000 .27775/.57309deg. Each5014 <2deg and>=50%improvement; both5000 nonregressionwithin.5deg. Original>=6points/.2mmsecondRMS/.35thicknessratio;5000left retainedarea89.748..89.779%,others~100%; no frames/cases removed. PATCH_INVERSION_RESULT.json exit0 at21:24:10UTC. Learned predictor/controller untouched; nounknownmaterial/whole-shape/generalizationclaim. Singlefirstframe23s visualvector extraction is not primaryacceptance; inherited50frame summaryfalse in itslog is explicitlyinapplicable, VISUAL_NORMALS.json onlyvectors.

All5 finalimages inspected:full34layoutplot,5014+5000fullhandcontactcloseups,5014+5000actualmesh raw/corrected/truthnormalviews. Finaldirectionrender exit0 at21:34:12UTC. Previous5000constantgray failuresretained; contextrecreationdidnotfix, originaleyez-.1026m meantgroundocclusion. Finalboundingframingkeepscameraaboveground andotherhandarrows hidden; initialframingimagesretained. Fullreport and HTML at sustained_grip_diagnostic_v1/contact_layout/REPORT.md,index.html; all localassetlinkschecked. Mesa renderingfulloriginalmeshes, objecttrutharrowsvalidationonly, no fabricatedmotion. Finishthisuseful offlinecomparison/report boundary; NEXT fulloriginal32TRAIN fixedwindow validation before controllerinput integration/newphysics/training. No additionalexperimentqueued.

Prior sustained-feedback two-case4800controls remainsoverallFAIL:5014 peak9.875N butmotion.32/4s/noairbornehold;5000PASS/finalclearance20.228cm. Prelifttrajectoriesnotbitexact (.2453/.0635N maxpadload differences), no strictcausalclaim. Originalcorpus32/48 incomplete/all32TRAINcomplete,10controllerpasses/14airborneholdlabelcases/12052labels,coverage2/8 versus4required;76800saved=74400new+2400reuse. Allfailedcases andtarget-speedcoupling evidence retained,noVALTEST/newoptimization. OfficialfullUtonia138407503 unchanged; prior58/39/55 arecompatibilityonly. Three2000/fullbatch1/forcezero/all8TEST predictionrenders unexecuted; oldcontrolledgrip mass69.862%/native66.50% negatives unresolved, support4.49%bounded. Reliablecarryingpredictor goalactive/unproven. Sole297659/server31/step0retained, allcurrentchildren terminal.

Latest 2026-09-16T03:11:31+08:00: Surface-feedback bounded diagnostics COMPLETE, three x2400controls, no optimizer/model changes. Original surface-feedback controller lifts actual5007 but oldseven mean criteria hide alternating force (newSTABILITY_ADDENDUM retains originalpassedtrue and declaresnotstable). Measured stiffness led gain.00015->.000025; oscillation suppressed, targettracking stillfails. PI2s withlimits tested, alsofails targettracking; no further controller trial queued. Allthree actualhold bilateral/clearance100%, center rise16.53/17.16/17.31cm, peak51.21/28.25/38.38N; fullholdtargetband0/0%,65.46/100%,82.87/67.35%. Same relative+5..+10s251frames comparison: p95frameforcechanges33.18/45.32N versus.0255/.0656N versus.0233/.0417N, meshdrift8.84/2.69/4.93mm. Recordedpalmar/fullnormal ratiosabout59%left/81..87%right; allsolverfullforce truth remainsvalidation-only. Differentclock/path/controlchanges prevent attributingallphysicalimprovement uniquelytolocalplanechannel.

New24s/240frames/10fps rendered tactile_plane_carry.mp4 COMPLETE on lowgainfull48s trace, actualmesh/centroid/observedlocalplanes and1Dseparation displayed; 63/63displayholdframes havewidthestimate, meanerror7.445mm/max8.499mm. Allframesdecoded/nonblack, fourpreviews/decodedholdimages inspected. Initialinlineencode blackatframe0 retained asplane_video exit1; PNGthenencode child1279254 exit0 at19:08:37UTC. Plane/PCA/forcebalance are explicit geometry/analyticdiagnostics, NOT Utonia predictions or arbitraryshapereconstruction. Full report grip_transfer_v1/REPORT.md andgrip_motion_dataset_v1/index.html updated. Allphysicschildren terminal; sole297659/server31/step0 retained. Completednewphysics18600controls plusinterrupted451..1200; noextraoptimization. Original48corpus notexpanded (TRAIN5000PASS/5007FAIL), trainingnotqueued, global reliablepredictor goalstillactive. Next usefulwork: dense/estimatedsurfacegeometry adapter with2mmvoxel informationloss accounted, partialsensor forceobservability andmasssupervision, full officialUtonia; do notcontinueblindcontroller tuning orclaimoldmass/native failures fixed.

Latest 2026-09-16T02:54:33+08:00: User continuation remains active; reliable carrying predictor NOT established. Completed four matched Newton same-geometry .5/.9kg x12/24N side-grip cases all lift17.46..17.77cm. Frozen geometry/contact/force/forcezero hold mass errors30.428/30.006/69.862/23.046%; force center8.006cm,rotation34.212deg. No new optimization. Full48sec/960frame20fps true-mesh rendering COMPLETE, all frames decoded/nonblack and original393frame prefix preserved, four-case contact sheet inspected. Report: experiments/object_predictor_v1/grip_transfer_v1/REPORT.md, video renders/newton_grip_transfer.mp4.

Old297460 externally CANCELLEDby0 at02:31:03,step SIGKILL02:31:34; preserved completed cases/predictions/renderPNG. Emptyqueue verified before sole replacement297659/server31/explicitstep0. Only interrupted encoding and unsaved dense diagnostic recovered; neither old physical case nor model evaluation rerun. Dense recovery1200controls COMPLETE,17798surface samples; still physical failure peak966.17N, no valid lift. Total completed new controls11400 plus interrupted451..1200, before current2400diagnostic. Saved source interface positions/areas (no true normals) recover area-PCA local plane directions: fixedprelift14.5..16s76frames/hand, left/right35/27medianfaces, onlylittle_distal, area13.13/11.21mm2; median error to validation-only actual object mesh normals.794/.638deg. Full1200 integration maxpadload1.34e-5N/shear5.09e-5N/centroid3.53e-8m; allfinite. Three fullmesh/contact closeups inspected. This is ideal simulated tactile local-plane geometry, NOT learned shape or hardware accuracy.

Varied48case corpus remains gated after5000PASS/5007FAIL; no-yaw and force-readiness attempts also failed. New SurfaceGripController uses only hand proprioception, padload, observed assigned tactile positions/areas; no oracle normals/object pose. Two geometric control tests pass (contact pivot/rate cap, no forced lift). Single TRAIN5007 surface-feedback diagnostic2400controls/48s launched18:53:10UTC viaretained PGID3546445/Python3547019; directcompute Rl84.1%CPU verified, logframe550. Maximum3deg/s alignment, earliestlift24/latest40 gated on observed resolvedplanes/alignment/load stability; no forced lift. Original full physics/hand+box meshes unchanged. Finish this bounded diagnostic and actual render/metrics before extending acquisition; no formal48 expansion or new training queued. Preserve prior negatives, no policy/native/material/arbitraryshape claim, sole allocation retained.

Latest 2026-09-16T01:56:28+08:00: User explicitly resumed with goal "继续推进" after prior rendered stop/report. Reliable carrying estimation remains objective; diagnosis is not final success. Added exact-mesh GripScene acquisition adapter reusing existing Newton ProbeScene/SupportScene physics. Four fixed geometry/seed310017 cases .5/.9kg x12/24N targets all completed1200controls each/4800total and pass bilateral/clearance/rise/peak/target criteria. Hold both contact100%, real fullmesh clearance>10cm100%, center rises17.46..17.77cm; full resolved support .5kg~4.904N/.9kg~8.827N independent of squeeze. No policy or model training. Partial-palmar readings and unassigned/full hand forces explicitly saved separately; only original OBSERVATION_KEYS feed estimator. Data4cases retained, including no filtering; protocol primary31+25k clocks20..24s preserved, additional31+5k visual clocks declared before evaluation.

Sole297460/server05/step0 live. Frozen three-arm evaluation PGID3709733 started17:53:06UTC; geometry complete, contact live by directcompute child3718956. Geometry hold eachcase mass~.685kg, center7..8cm, rotation17.6..19.5deg, size2.4..2.6%; force result pending. Render_grip_transfer.py ready but not executed; comparison report and rendered48s video still pending. NATIVE_FORCE_SEMANTICS.json quantifies rawTacSL normal median230.80N vsPhysX mapped54pad normalnormsum53.49N, ratio4.33; this is sensor/solver definition mismatch, not calibrated conversion. Normal mapping passes; independent friction matrices differ max1.907e-6N, original1e-6checkfalse retained and documented in DISCREPANCY, no silent pass or model-input replacement. Measured Newton palmar vertical shear carries weight signal (diagnostic approx.461/.477/.877/.805kg acrossfour), not learned success; fullhand truth excluded. Continue matched frozen evaluation/render, then evidence-guided generalized acquisition/training; full original goal unfinished, no duplicate allocation/release.

Latest 2026-09-16T01:19:27+08:00: USER-REQUESTED RENDERED VISUALIZATION COMPLETE; STOP AND REPORT. Prior curves/point-cloud images did not meet the user's intended rendering requirement. New rendered_carry_v1/index.html, carry_state_estimation.mp4 (160 frames/25fps/6.4s,0.25x), support_mass_comparison.mp4 (204frames/10fps/20.4s), GIF and Chinese RENDER_REPORT.md deliver full 35 G1 URDF visuals, full scanned box15626vertices/31248faces, actual hands/contact markers and predicted pose/size/mass. Existing native official-policy80frame carry trace1000 replayed (not new physics); full frozen force Utonia consumes all80 causal observations,49predictions after31warmup. Native mean errors center11.185cm/rotation65.221deg/size21.330%/mass66.502%: CLEAR TRANSFER FAILURE, not successful carrying estimator or policy improvement. Actual lift54.840cm, true0.5kg vs finalpred0.820kg. Different backend/scale/contact action confounded; no single cause proven.

Initial raw native inference failed rigid-site1um guard due float32 centroid rounding; raw log retained. Rendering-only adapter restores first-observed rigid mount with current hand pose, bounded2um (maxworld1.916/1.185um), does not alter contact/forces/truth/baseadapter/model. All49 new predictions saved with source+endpoint hashes. Support video includes ALL sixTEST30..35/all34windows percase, priorgeometry30.696% vsforce4.490% mass result; worst32 retained. Label dimensions are asset-axis AABB extents of oblique scan; visual truth guide uses oriented mesh bounds, not inflated label cuboid. EGL works on retained297460/server05/step0; no new Isaac/Vulkan launch. Render+verify normalexit0; all14decode/pose/clock/finite/fullmesh checksPASS, decoded carry4frames/all6supportcases inspected/hashbound. RENDER_COMPLETION_STOP.json records completed requested rendering, zero newoptimization/physics, broadergoalstillunproven. Retainallocation; no more experiments without new userdirection. All previous negatives and history preserved.

Latest 2026-09-15T23:42+08:00: FIRST USEFUL OBJECT-PREDICTOR RESULT COMPLETE; USER STOP BOUNDARY REACHED. Full original objective remains broader/unproven; do not append training, collection, sweeps or policy experiments. FIRST_USEFUL_RESULT_REPORT.md contains the complete Chinese report and figures; FIRST_USEFUL_RESULT_STOP.json records verified evidence and stop instruction. Original pipeline3003510 normalexit0 at15:05:12UTC; direct compute process group now absent. Sole297460/server05/step0 retained, no child restart/release.

All three full138407503 Utonia arms2000/6000total, fullmodel+Adam51checks and225samplingchecks perarm complete. Final batch1 acquisition metrics: primary probe geometry/contact/force center2.4154/2.1888/2.3884cm, rotation7.3571/6.3980/7.5352deg,size8.2441/6.3704/7.5165%; supportmass30.6962/31.0990/4.4896%, forcezero37.5696%. Additional12mass47.1992/47.7829/7.2741%, forcezero36.4102%. All four predeclared MEAN criteria pass, not all cases: force size3031/3032=12.54/15.19%; supportmass5/6 under10%, worst32=17.05%; additionalmass8/12 under10%, worst2010=21.45%. Probe all-window mass remains38.8941%, no static probe weight claim. Complete known-mesh primary distance1.8490/1.6084/1.8253cm; known-family pose/size reconstruction only. All57savedmetric/36shapealignment checks pass; all14training+41evaluation/3D/summary images inspected/hash-bound. Both streaming audits pass, consume200/7500frames and match34/150savedwindows max1.19e-7/1.79e-7; no physical rollout. Full force-model forward median~94ms excludes input preparation. All earlier failures/negative results/native Vulkan/sensor-scope limitations retained. No arbitrary shape, material, pretrained-vs-scratch, deployment or demo-following benefit established. Stop and report this useful bounded result; further work needs new user direction.

Latest 2026-09-15T22:36+08:00: All three mixed-surface Utonia arms COMPLETE, exactly2000 updates each /6000 total. Force arm full TEST equal-episode errors: center2.894413cm, rotation6.049603deg, size5.334539%, mass21.704085%, versus contact-only4.018458cm/4.315703deg/4.287803%/36.485901% and geometry2.668081cm/5.195466deg/5.476084%/36.856332%. This is mixed evidence: strong mass improvement, not all-task superiority or acquisition-specific acceptance. Every arm sampling225/225, final all-three full-model/Adam/protocol/initial/batch/2000-clock audit51/51 passes. All14 training-stage learning/comparison/whole-episode images inspected and hash-bound in training_mixed_geometry_mass_surface_v1/AGENT_INSPECTION.json. Force distinguishes support masses; support32 remains biased high. Probe dimensions often remain near typical training sizes;3031/3032 biases and3035 bad probe mass retained. No arbitrary-shape or policy claim.

Same pipelinePGID3003510/sole297460/server05/step0 now performs declared batch1 evaluations, no remaining optimization. Latest directly verified child3125166 live Rl; geometry mixed+fresh-mass reports saved. Geometry batch1 mixed2.676094cm/5.191988deg/5.468270%/36.847660%, fresh mass47.1992%; model-only median87.31ms. Contact/force batch1, force-zero, fresh mass, acquisition-specific and full-mesh reports, streaming checks and corresponding images remain pending. Added CPU-only plot_mixed_summary.py (compiled, no execution or model forwards yet): once MIXED_METRICS.json exists, render its four declared primary metrics/thresholds and force-zero intervention to primary_state_comparison.png and inspect it. Existing automatic pipeline is unchanged; this optional report-rendering command must be run after source readiness through the retained compute launcher. User's stop/report boundary remains: finish this first useful verified comparison and visuals, stop further experimentation, provide full report; no extra training, collection or sweeps. Global objective remains active/unproven pending final evidence.

Latest 2026-09-15T22:07+08:00: User explicitly requests stopping additional experimentation once the first useful verified comparison and visualizations are ready, then giving a full report; otherwise wait for running results. USER_REPORT_STOP_BOUNDARY.json records this steering. Finish the current matched three-arm pipeline and already-declared evaluation/visualizations; do not append training, collection or sweeps.

Mixed-surface contact arm COMPLETE 2000 updates, endpoint reload EXACT 0, actual sampling audit 225/225 pass. Mixed TEST equal-episode errors: position 4.018458 cm, rotation 4.315703 deg, size 4.287803%, mass 36.485901%, versus geometry 2.668081 cm / 5.195466 deg / 5.476084% / 36.856332%. Position regresses; this is mixed evidence, not overall touch benefit or final acquisition-specific acceptance. Third force arm has started under the same pipeline PGID 3003510, trainer 3084334, sole 297460/server05/step0; full initial-model SHA matches both completed arms. No force endpoint yet. Current pipeline automatically finishes at declared streaming readbacks; no extra experiment has been queued. Final full-model/Adam comparison, batch1/force-zero/fresh-mass/acquisition-specific/3D reports and image inspection remain pending. Preserve prior negatives, incidents and scope limitations; global predictor objective remains unfinished.

Latest 2026-09-15T21:49+08:00: Mixed-surface contact arm has passed500 actual updates and saved VAL500; same pipelinePGID3003510/sole297460/server05/step0 continues. Direct compute trainer3045069 Rl107%CPU elapsed6:39, log step506 verified. Intermediate mixed VAL500 contact position2.988480cm/rotation14.452030deg/size4.955356%/mass42.063376% versus geometry3.380028cm/8.753779deg/5.756294%/41.839424%: position/size lower, rotation worse, mass not improved. Not an overall touch benefit, final TEST result or selection decision. INTERMEDIATE_VAL500_COMPARISON.json preserves source hashes and three checks: full initial-model SHA exact, all first500 actual batch episode/frame/clock records exact, both original500 clocks complete. No model/source/protocol/budget change or extra training from this readback.

Geometry remains COMPLETE2000/test/reload EXACT0/sampling225 checks; contact2000 and force arm still pending. This turn verified live training and completed the saved matched VAL500 comparison. Continue exact serial training, full all-arm model+Adam audit, all learning/test plots and declared batch1/forcezero/freshmass/acquisition-specific/3D/streaming reports. All prior collection/schema/token/CPU-path incidents, fullmodel fidelity, partial palmar sensor scope, mass/native negatives and unfinished global objective retained. No duplicate allocation, release, restart or new budget.

Latest 2026-09-15T21:43+08:00: Mixed-surface geometry arm COMPLETE2000 updates, full model+Adam saved, TEST report complete, endpoint reload output difference EXACT0. Actual sampling readback225/225 passes, all2000 original clocks/batches/exposures retained. Mixed TEST equal-episode errors: position2.668081cm, rotation5.195466deg, size5.476084%, mass36.856332%; this is the geometry baseline, not force benefit or acquisition-specific final acceptance. Intermediate VAL1000 position3.203826cm/rotation7.304306deg/size4.5061%/mass39.685%; VAL1500 position2.632508cm but rotation7.713158deg/size5.25094%/mass41.3621%, nonmonotonic trend retained; no checkpoint selection.

Same pipeline PGID3003510 on sole297460/server05/step0 automatically entered geometry_contact trainer Python3045069; direct compute process confirmed live Dl elapsed1s at transition. New contact PROTOCOL now present, full138407503 parameters and complete initial-model SHA EXACT to geometry arm. No contact optimizer update or endpoint claimed yet. Force arm not started. This turn completed first-arm training/test/reload/sampler evidence after verified waits; shared-login FS lag observed, direct compute ps/log authoritative, no restart. Final all-three fullmodel/Adam audit, learning/TEST plots, batch1/forcezero/freshmass/acquisition-specific/3D/streaming reports remain required. Full original scope and criteria unchanged; all earlier collection/schema/token/CPU-path incidents, partial palmar sensor scope, prior mass/native negatives and unfinished global goal retained. Continue exact serial pipeline; no new budget, duplicate allocation or release.

Latest 2026-09-15T21:30+08:00: Mixed-surface predictor training continues on sole 297460/server05/step0, pipeline PGID3003510, geometry trainer Python3005122. Direct compute ps confirms Rl105%CPU, elapsed10:20, actual optimizer log through update1000; no completed2000 endpoint or later arm yet. First VAL500 saved: mixed equal-episode position3.380cm/rotation8.754deg/size5.756%/mass41.839%. Read-only FP64 stratification gives probe primary140..148s position2.1257cm/rotation11.5988deg/size8.0531%, support mass26.8654%; intermediate VAL only, not TEST success or checkpoint selection.

New CPU progress readback completed11/11 checks on saved step500: full138407503 named Adam bindings, all138407449 active moment elements/clocks finite and exact, sole unused54-element official mask token unchanged, saved protocol exact. First CPU launch failed exit2 because script lived in login-local /tmp; PROGRESS_READBACK_PATH_INCIDENT.json preserved, identical script moved to shared experiments path, r1 CPU exit0 at13:29:33UTC. No model forward, repeated training or training interruption. PROGRESS_READBACK.json stores exact observed checkpoint clock and all VAL groups. Final all-arm endpoint audit remains required. Original three2000 budget, full model/data/normal policy/optimizer unchanged; collection36/270000/all72plots, all prior schema/token incidents, sensor-scope limits and mass/native negatives retained. Continue exact pipeline through all arms and declared saved/batch1/forcezero/freshmass/3D/streaming readbacks; global goal unfinished, no allocation release/duplicate/restart.

Latest 2026-09-15T21:22+08:00: Object predictor multi-face collection COMPLETE: 36 configurations/270000 saved controls, 22 recorded-pad coverage/peak passes and 14 failures (TRAIN14/24, VAL4/6, TEST4/6). All 72 physical/3D images inspected/hash-bound in AGENT_INSPECTION.json; all source hashes, 36 controller audits and original 26 pre-interruption traces verified. Recovery collector exited0 at13:08:41UTC; prior externally killed3026 remains separately accounted (total executed277051..277500). Full history coverage COMPLETE: TRAIN192 windows112each-hand/166any, VAL48/32/48, TEST48/31/40; total288/175/254, each-hand readings may be at different times. No gaps/failures removed; prior TRAIN24 records exact.

Mixed 72-case corpus and all223 sampling checks complete; old H8 GPU streaming replay passes max1.123823e-7. First pipeline2991445 exited1 at13:10:20UTC before GPU/updates: legacy support report lacks complete key. Source-specific full36/per-frame/balance/exactmetadata validation fixed builder; incident archived. Schema r1 pipeline2995084 exited1 at13:13:56 before updates: full-model all-gradient presence check failed. Full three-mode named diagnostic2999076 exited1 at13:16:23: ONLY absent gradient is official54-element backbone.embedding.mask_token, no nonfinite gradients; official Embedding uses token only with mask, absent in task input. All original failed reports retained. Qualification now explicitly checks sole unused token and unchanged release value, all used gradients finite. Training records complete optimizer ID/name/size bindings and token hash; endpoint audit expects full138407503 bound parameters,138407449 moment elements and unchanged unused token. No model/loss/optimizer behavior/budget changes.

Current token r2 pipeline PGID3003510 launched13:18:14UTC on sole297460/server05/step0; corrected full GPU qualification42/42 PASS, all three full released tensors/Adam bindings/five-stage gradients/repeated outputs exact. Geometry trainer Python3005122 LIVE, direct compute Rl122%CPU elapsed2:20, actual optimizer log through step70 at21:21:52. Full138407503, same72case/splits/32history/normaladapter/loss, balanced2+2 batch, three2000=6000updates; no learned endpoint yet. Active script train_mixed_geometry_mass_surface_resume_token.sh reuses hash-verified completed data/history/sampler/old replay, then serial arms/full endpoint+Adam/batch1/forcezero/freshmass/3D/streaming checks. Continue exact pipeline and inspect all required plots; do not rerun completed collection/training or relaunch original immutable builder. Goal unfinished, prior mass10%/native failures and partial palmar sensor scope preserved; no duplicate allocation or release.

Latest 2026-09-15T20:51+08:00: Object predictor corpus is 33/36 complete (247500 saved controls), with 20 recorded-pad coverage/peak passes and 13 failures. All 66 physical/3D images are inspected and hashed. Third TEST 3032 passes: bilateral X/Y/top frames 601/554/811, peak 4.1138 N, final X loads 0.198/0.200 N, Y 0.409/0.531 N, top 0.980/1.013 N; all final contact-toggle fractions zero. Initial impulses/asymmetric onset remain visible; saved controller replay passes. Same recovery PGID 2908987/Python 2909028 is live on sole 297460/server05/step0 (direct compute Rl, 99.7% CPU, elapsed 46:54), now fourth TEST 3033/frame2400. This turn completed 3032 inspection after verified waits. No model/source/protocol changes or training. Remaining three TEST cases, full history coverage, guarded mixed_surface GPU qualification, three 2000-update arms and all readbacks/plots. Original failures, interruption accounting, partial palmar sensor scope, prior mass/native negatives and unfinished global objective remain; no duplicate allocation, release or restart.

Latest 2026-09-15T20:44+08:00: Object predictor collection is 32/36 complete (240000 saved controls), with 19 recorded-pad coverage/peak passes and 13 failures. All 64 physical/3D images are inspected and hashed. Second TEST 3031 passes: bilateral X/Y/top frames 508/1023/894, peak 6.9791 N, final X loads 0.317/0.285 N, Y 0.470/0.391 N, top near 1 N; all final contact-toggle fractions zero. Initial asymmetric X onset and impulses are retained; saved controller replay passes. Same recovery PGID 2908987/Python 2909028 is live on sole 297460/server05/step0 (direct compute Rl, 99.7% CPU, elapsed 39:54), running third TEST 3032/frame1650. This turn completed 3031 inspection after verified waits. No model/source/protocol changes or training. Remaining four TEST cases, full history coverage, guarded mixed_surface full GPU qualification, three 2000-update arms and all readbacks/plots. All original failures, interruption accounting, partial palmar sensor scope, prior mass/native negatives and unfinished global objective remain; no duplicate allocation, release or restart.

Latest 2026-09-15T20:37+08:00: Object predictor corpus is 31/36 complete (232500 saved controls), 18 recorded-pad coverage/peak passes and 13 failures; all 62 physical/3D images are inspected and hashed. First TEST 3030 fails Y coverage: bilateral X/Y/top frames 44/15/845, peak 26.0717 N. X pad readings end at zero with temporary body-origin rise/shift (not a floating conclusion); final Y left 0/right 0.1277 N and right contact-toggle fraction 0.23116; top settles near 1 N. Saved controller replay passes. Same recovery PGID 2908987/Python 2909028 remains live on sole 297460/server05/step0 (direct compute Rl, 99.7% CPU, elapsed 32:41), now second TEST 3031/frame700. This turn completed first TEST inspection after verified waits. No source/model/protocol changes or new training. Remaining five TEST cases, full history coverage, guarded mixed_surface GPU qualification, three 2000-update arms and all readbacks/plots. All original split denominators, external interruption accounting, partial palmar sensor scope, prior mass/native negatives and unfinished global objective remain; no duplicate allocation, release or restart.

Latest 2026-09-15T20:31+08:00: Object predictor collection is 30/36 complete (225000 saved controls); all 24 TRAIN and six VAL configurations are collected. Recorded-pad coverage/peak: TRAIN 14 pass/10 fail, VAL 4 pass/2 fail, total 18 pass/12 fail. All 60 physical/3D images are inspected and hashed. Last VAL 3029 fails Y coverage: bilateral X/Y/top frames 741/0/1109; left has seven brief Y readings versus right 966, with no simultaneous readings. Its empty Y 3D panel is retained. Peak 3.4635 N; final Y left 0/right 0.198 N; saved controller replay passes. Same recovery PGID 2908987/Python 2909028 is live on sole 297460/server05/step0 (direct compute Rl, 99.6% CPU, elapsed 26:32), now first TEST 3030/frame1000. This turn completed 3029 inspection after verified waits; no model/source/protocol changes or new training. Remaining six TEST configurations, full history coverage, guarded mixed_surface GPU qualification, three 2000-update arms and all evaluation/readbacks/plots. Prior external interruption accounting, sensor-scope limits, negative mass/native evidence and unfinished global objective remain; no duplicate allocation, release or restart.

Latest 2026-09-15T20:25+08:00: Object predictor corpus has 29/36 complete configurations and 217500 saved controls; 18 pass recorded-pad coverage/peak criteria and 11 fail. All 58 physical/3D images are inspected and hashed. Fifth VAL case 3028 fails Y coverage: bilateral X/Y/top frames 760/4/1071, peak 25.5369 N, final Y left 0/right 1.2266 N with lateral object displacement. Its saved controller replay passes; the failure remains in VAL. Recovery collector PGID 2908987/Python 2909028 is live on sole allocation 297460/server05/step0 (direct compute ps: Rl, 99.6% CPU, elapsed 20:13), running last VAL case 3029 at frame 1100. This turn completed case 3028 inspection after verified waits. No model/protocol/source changes or new training. Remaining: last VAL and six TEST cases, full history coverage, guarded mixed_surface full-model GPU qualification, then three 2000-update arms and all readbacks/plots. External interruption accounting (277500 total execution upper bound), partial palmar sensor scope, prior mass/native negatives and unfinished global objective remain; no duplicate allocation, release or restart.

Latest 2026-09-15T20:18+08:00: objectpredictor corpus28/36 complete210000savedcontrols,18recorded-padcoverage/peakPASS10FAIL,all56physical+3Dimages inspected/hashsaved. LatestfourthVAL3027PASS both769/715/877,peak6.3679N,last4sX.183/.248N,Y.468/.407N,top~1N,alltoggle0; controllerreplayPASS. Same recoveryPGID2908987/Python2909028 directcomputeRl99.6%CPUelapsed13:31,nextcase3028/frame750 onsole297460/server05/step0. This turn verifiedwaits andcompleted3027inspection; no source/model/protocol changes ortraining. Originalexternalinterruption+142loggednotbitexactcomparison andexecutionupper277500 retained. Finishremaining2VAL+6TEST/full36coverage/guardedmixed_surfaceGPUqualification/three2000; partialpalmar sensor scope/alloldmass/native negatives/globalgoalunfinished retained,no duplicate/release/restart.

Latest 2026-09-15T20:12+08:00: recoverycase3026 COMPLETE7500frames/PASS both534/465/1041,peak2.7684N,last4sallcontacttoggle0; bothphysical3Dplots inspected/hashsaved,controllerreplayPASS. Corpus27/36 complete202500savedcontrols,17recorded-padPASS10FAIL,all54imagesinspected. Interruptedoriginal3026retainedseparately7051..7500unknownexactcontrols. NEW saved142loggedprefixcomparison allold0..7050samples:onlyinitialbitexact,phaseclocksallagree,maxpositioncomponent5.73695e-5m/load.0388561N; notallframephysicalequivalence orindependentrepetition. Same recoveryPGID2908987/Python2909028 directcomputeRl99.3%CPUelapsed7:22,nextcase3027/frame1050 onsole297460/server05/step0. No source/model/protocol/newtraining changes; originalcompleted26preserved. Continue remaining9/full36history/guardedmixed_surfaceGPUqualification/three2000. Externalinterruptionbound277500totalexecution,partialpalmarsensor scope,oldmass/native negatives/globalgoalunfinished retained;no duplicate/release/restart.

Latest 2026-09-15T20:05+08:00: EXTERNAL RESOURCE INTERRUPTION AUDITED. Sole297320 CANCELLEDby0 at20:01:04,step0signal9 ended20:01:34; collector stoppedcase3026afterloggedframe7050,no partialtrace/snapshot. All26completecases/fulltraces+records+audits hashed andunchanged,all52imagesinspected;16recorded-padPASS10FAIL. Archived incompletecase/log/process/partialreports undermultiface_dataset_v1/interruptions/job297320_episode3026 withINCIDENT.json. Exactinterruptedcontrolsunknown7051..7500; separateboundedallowance7500,finalsavedcohort270000,totalexecutionupper277500. Emptyqueue verified before sole replacement297460 grantedserver05/step0/H200GPU2; guarded recoveryPGID2908987/Python2909028 launched20:04:30,all26hashes/fivephysicssources PASS,originalcase3026seed/config resumedfromstart thenremainingnine,75000remainingcontrols. DirectcomputePythonRl85%CPUelapsed22s/frame150 verified;not astatecheckpointresume,0completedcasesrecollected/0newoptimization. Original36split/failuredenominators/model/trainingprotocolunchanged,oldartifactsretained. Finishremaining10/20plots/full36history then guardedmixed_surface GPUqualification/three2000; priornegative mass/native/sensorscope limitations andglobalgoalunfinished retained,no duplicate/release.

Latest 2026-09-15T20:00+08:00: object predictor collection26/36 complete195000controls,24TRAIN+2VAL;16recorded-padcoverage/peakPASS10FAIL,all52physical+3Dimages inspected/hashsaved. SecondVAL3025PASS both613/650/830,peak4.3231N,last4s allcontacttoggle0; earlyYloadfluctuations retained,controllerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:50:59,case3026/frame5200 onsole297320/server05/step0. This turn completed3025inspection; no source/model/protocol changes ortraining. Full36coverage/guarded mixed_surface GPUqualification/three2000 stillpending. Partialpalmar-sensor scope, failedcases, allTRAINhistory limitations andoldnegative mass/native results retained; globalgoalunfinished,no duplicate/release/restart.

Latest 2026-09-15T19:50+08:00: object predictor collection25/36 complete187500controls,24TRAIN+firstVAL;15recorded-padcoverage/peakPASS10FAIL,all50physical+3Dimages inspected/hashsaved. FirstVAL3024PASS both688/625/980,peak13.2494N,last4s allcontacttoggle0,loadfluctuations retained;controllerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:40:55,secondVAL3025/frame900 onsole297320/server05/step0. This turn completed3024inspection after verifiedwait; no source/model/protocol changes ortraining. AllTRAINhistory192windows112eachhand/166any andsensor-scope/3020groundedtilt audits retained. Remaining5VAL+6TEST/full36coverage/guarded mixed_surface GPUqualification/three2000 pending. Globalgoal/allpriornegative mass/native evidence preserved;no duplicate/release/restart.

Latest 2026-09-15T19:47+08:00: all24TRAIN fixed32history coverage COMPLETE CPU2875861 exit0 at11:44:53UTC,192primarywindows; eachhand/all3directions112/192,anyhand/all3directions166/192. Qualified14cases111/112eachhand112/112any; failed10cases1/80eachhand54/80any. Eachhandreadings mayoccuratdifferenttimes,NOTsimultaneouscontact. First12records EXACT toprioraudit; source-bound summarysaved. No window/case removal, no sampler/protocol/modelchange;0physics/model/updates. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:37:23,firstVAL3024/frame4400 onsole297320/server05/step0. Still24/36complete180000controls/14recorded-padPASS10FAIL/all48physicalimagesinspected. Full6VAL+6TEST/full36coverage andguarded mixed_surface GPUqualification/three2000 pending. Partialpalmarzero scope/3020groundedtilt/allpriornegatives/globalgoalunfinished retained,no duplicate/release/restart.

Latest 2026-09-15T19:43+08:00: object predictor collection24/36 complete180000controls,all24TRAIN nowcollected,14recorded-padcoverage/peakPASS10FAIL; all48physical+3Dimages inspected/hashsaved. Latest3023FAIL both577/2/980,Yleft2/right726frames,finalYleft0/right.702N; X/top sustained,peak7.4827N,controllerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:33:48,firstVALcase3024/frame50 onsole297320/server05/step0. Turn completed3023/allTRAIN inspection after verifiedwaits; no source/model/protocol changes ortraining. All6VAL+6TEST stillneeded, thenfull36historycoverage/guarded mixed_surface GPUqualification/three2000. Source-limitedpartialpalmar reads and3020groundedtilt/unknownfullforces retained; allpriornegatives/globalgoalunfinished,no duplicate/release/restart.

Latest 2026-09-15T19:37+08:00: object predictor collection23/36 complete172500controls,14recorded-padcoverage/peakPASS9FAIL; all46physical+3Dimages inspected/hashsaved. Latest3022PASS both795/913/973,peak4.9759N,last4s alltoggle0,top~1N,controllerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:27:42,case3023/frame450 onsole297320/server05/step0. Turn completed3022inspection after verifiedwaits; no source/model/protocol changes ortraining. Prior3020groundedtilt/fullmesh20exactchecks andpartialpalmar-sensor scope retained; sensorzerosNOTfullhandseparation. Full36/historycoverage/guarded mixed_surface GPUqualification/three2000 pending; globalgoal/allpriornegative mass/native evidence preserved,no duplicate/release/restart.

Latest 2026-09-15T19:31+08:00: full savedgroundgeometry3020 COMPLETE CPU2856150 exit0 at11:29:11UTC,all7500poses/full15626vertices/31248faces,20independent fullXYZheightchecks exact0; bothchronology/fixed44s3Dplots inspected/hashsaved. Phase0fullmesh minZ-.769..-.101mm whilebodyorigin rises18.72mm androtationfrom1s up7.848deg; fixed44s minZ-.178mm/rotation7.679deg/bothpadloads0. Supportsgroundedtilt,NO floatingevidence; unrecordedtotalhandforces/groundreactions remainunavailable. Sensor-scopeaudit andREADME updated,0physics/model/updates; fivepinnedphysicsfiles untouched. Collection22/36complete165000controls,13recorded-padcoveragePASS9FAIL,all44physical+3Dimages inspected. Latest3021FAIL Xboth7/Y523/top1067,peak10.7883N,controllerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:21:28,case3022/frame650 onsole297320/server05/step0. Full36/historycoverage/guarded mixed_surface fullGPUqualification/three2000 pending; allpriornegatives/globalgoalunfinished retained,no duplicate/release/restart.

Latest 2026-09-15T19:26+08:00: object predictor collection21/36 complete157500controls,13contact/peakPASS8FAIL; all42physical+3Dimages inspected/hashsaved. Latest3020FAIL both30/4/838,Yrightlast4s toggle.5025/top~1N,peak12.8587N; controllerreplayPASS. Xbodyorigin rises~2cm withnearzero recordedpadloads; groundclearance/rotationnotyetquantified. NEW SENSOR_SCOPE_SOURCE_AUDIT: fullhandcolliders but27palmarpads/hand; uncoveredfaces pad-1 excludedfromobservation ANDfeedback. observation computes resolvedtotalhandload/unassignedfaces butqualify_probe onlychecksoverflow anddoesnotsavethem. Zero recordedpadload isNOT proof no fullhandphysicalcontact; priorfirst17slowrecovery diagnosis strictly sensor-definedloss, measuredspeedvalid butphysicalseparationnotproven. Sourcehashesrecorded, no physics/model/protocol changes. NEXT savedfullmesh/pose groundclearance audit distinguishoriginheight/tilt, no fabricatedforce truth. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:16:37,case3021/frame2500 onsole297320/server05/step0. Full36/historycoverage/guarded mixed_surface qualification/three2000 pending; globalgoal/allpriornegatives retained, no duplicate/release/restart.

Latest 2026-09-15T19:17+08:00: object predictor collection20/36 complete150000controls,13contact/peakPASS7FAIL; all40physical+3Dimages inspected/hashsaved. Latest3019declaredPASS both511/605/1228,peak6.9623N,BUT Xleft loaddecays to~0,last4smean.002684N/right.576N; coveragePASSnotendcontactstability. Y/top sustained,savedcontrollerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:08:19,case3020/frame200 onsole297320/server05/step0. Turn completed3019inspection after verifiedwaits; no source/model/protocol changes ortraining. Priorfirst17slowrecovery diagnosis explicitlybounded(first17failure equivalence notgeneralized),first12history limitationretained; full36/coverage/guarded mixed_surface GPUqualification/three2000 pending. Globalgoal andpriornegative evidence retained; no duplicate/release/restart.

Latest 2026-09-15T19:12+08:00: object predictor collection19/36 complete142500controls,12contact/peakPASS7FAIL; all38physical+3Dimages inspected/hashsaved. Latest3018FAIL both805/0/1046,Yright4brief/left1154frames,finalYright0/left.854N; emptyY3Dpanel retained. X/top sustained,toppeak14.6572N then~1N; savedcontrollerreplayPASS. Mass1.2kg boundary noted,not demonstratedfailurecause. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed2:02:28,case3019/frame850 onsole297320/server05/step0. Turn completed3018inspection after verifiedwaits; no source/model/protocol changes ortraining. First17slowrecovery andfirst12historydiagnostics remainbounded, full36/coverage/guarded mixed_surface GPUqualification/three2000 stillpending. Globalgoal andpriornegative mass/native evidence preserved; no duplicate/release/restart.

Latest 2026-09-15T19:05+08:00: object predictor collection18/36 complete135000controls,12contact/peakPASS6FAIL; all36physical+3Dimages inspected/hashsaved. Latest3017PASS both982/1225/1311,peak5.7704N,last4s contacttoggle0 inallphases,top~1N; savedcontrollerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed1:55:44,case3018/frame450 onsole297320/server05/step0. This turn completed3017inspection after verifiedwait; no source/model/protocol changes ortraining. Savedfirst17slowlatchedrecovery diagnosis remainsdescriptive, no fasterrecoverybenefit proof orcurrentcontrollerchange. Full36/historycoverage andguarded mixed_surface fullGPUqualification/three2000 stillpending. Allpriornegatives/globalgoalunfinished retained; no duplicate/release/restart.

Latest 2026-09-15T19:03+08:00: NEW saved contact-loss diagnosis completed CPU2821449 exit0 at11:01:00UTC,first17episodes/all102phase-hand records,0controller/physics/model/updates. All6physicalfailed cases exactly match cases with>=1s terminal latchedcontactloss:11phase-hand runs10.38..17.76s,continuedinward speed~.05mm/s versus6mm/s beforefirsttouch,travel.518.. .887mm. Savedpose versus existingfeedbackrule displacement max1.3279e-8m. CONTACT_LOSS_FIRST_SEVENTEEN.json pins allinput/controller/helperhashes; fivephysicsfiles unchanged. Descriptive actualslowrecovery, NOT proof fasterrecovery restorescontact orpredictorbenefit; no controller/protocol/modelchanges. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed1:53:31,case3017/frame5500 onsole297320/server05/step0. Still17/36complete127500controls/11PASS6FAIL/all34images inspected; full36/historycoverage/guarded mixed_surface qualification andthree2000 remainpending. Globalgoalunfinished andallpriornegative evidence retained; no duplicate/release/restart.

Latest 2026-09-15T18:58+08:00: object predictor collection17/36 complete127500controls,11contact/peakPASS6FAIL; all34physical+3Dimages inspected/hashsaved. Latest3016FAIL: both467/0/990,Yleft7brief/right784frames,finalYleft0/right.946N; emptyY3Dpanel retained. X/top sustained,toppeak11.3635N then~1N each; savedcontrollerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed1:49:05,case3017/frame200 onsole297320/server05/step0. This turn completed3016inspection after verified waits; no source/model/protocol changes ortraining. Fixedfirst12history audit3009missingYright1/8 retained; full36coverage/guarded mixed_surface GPUqualification/three2000 stillpending. Allpriorfailures/globalobjectiveunfinished retained; no duplicate/release/restart.

Latest 2026-09-15T18:53+08:00: object predictor collection16/36 complete120000controls,11contact/peakPASS5FAIL; all32physical+3Dimages inspected/hashsaved. Latest3015FAIL: both441/13/705, Ybelow25required; leftfinalYload0/right.251N, rightYpeak16.905N andsmalllateraldisplacement. X/top sustained, savedcontrollerreplayPASS, no repeatphysics. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed1:43:11,case3016/frame800 onsole297320/server05/step0. This turn completed3015failure inspection after verified waits; no source/model/protocol changes ortraining. First12history audit limitation3009missingYright1/8 retained; full36coverage andguarded mixed_surface fullGPUqualification/three2000 pending. Globalobjective unfinished, allpriornegative mass/native evidence preserved; no duplicate/release/restart.

Latest 2026-09-15T18:46+08:00: object predictor collection15/36 complete112500controls,11contact/peakPASS4FAIL; all30physical+3Dimages inspected/hashsaved. Latest3014FAIL: both0/4/827; sideleft only5/4contactframes andfinal side load0, right~.28N; peak8.5453N. Empty X simultaneous3Dpanel retained. SavedcontrollerreplayPASS, no repeatphysics. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed1:36:35,case3015/frame600 onsole297320/server05/step0. Fivephysicalsource andallpreviousimagehashes unchanged. Turn completed newphysicalfailure inspection after verified waits; no model/protocol changes ortraining. First12history coverage and3009missingYright1/8 retained, full36 required. Guarded mixed_surface fullGPUqualification/three2000 notstarted; globalgoal andmass/native limitations unfinished. Continueexactcollector withoutduplicate/release/restart.

Latest 2026-09-15T18:39+08:00: object predictor collection14/36 complete105000controls,11contact/peakPASS3FAIL; all28physical+3Dimages inspected/hashsaved. Case3013both774/901/985,peak10.5611N atfirsttopcontact then~1N each;last4s allcontacttoggle0; savedcontrollerreplayPASS. Samecollector Python2665211 directcompute Rl99.9%CPU elapsed1:29:52,case3014/frame250 onsole297320/server05/step0. This turn verified waits andcompleted3013inspection; no physics/model/protocol changes ortraining. First12 historyaudit limitation3009Yrightmissing1/8 remains; full36coverage pending. Guarded mixed_surface fullGPUqualification/three2000 notstarted. Continueexactcollector, preserve alloldfailures/globalgoalunfinished; no duplicate/release/restart.

Latest 2026-09-15T18:31+08:00: fixed32 history audit on first12 saved probe cases completed CPU2783563 exit0 at10:29:18UTC, zero model/physics/updates. New evidence: physically qualifying but side-chattering3009 retains allthree directions BOTHhands7/8 primarywindows; at146.64s/frame7331 sampledYright0 despite512contacts in fullprefix. Anyhand8/8. Other8 physically passing cases BOTHhands8/8; failed3001/3002/3006 BOTHhands0/8. Saved HISTORY_COVERAGE_FIRST_TWELVE.json completefalse; full36 stillrequired. No window removal/history/protocol change. Same collector Python2665211 live Rl99.9%CPU elapsed1:21:21/frame5300 case3012 onsole297320/server05/step0. Still12/36completed,9PASS3FAIL,24imagesinspected; newmodel training notstarted. Continue exactcollection then guarded mixed_surface qualification/three2000; allpriorfailures/globalobjectiveunfinished retained. Follow-up same turn:13/36 now complete97500controls,10PASS3FAIL, all26physical+3Dimages inspected/hashsaved. Case3012both630/439/984,peak5.6925N,last4s toggles0; Xloads asymmetrical0.166/0.422N andsmalltemporarylateraldisplacement retained. Samecollector directcompute Rl99.9%CPU elapsed1:25:13,case3013/frame2300. The history audit remains explicitly first12, not13 orfull36.

Latest 2026-09-15T18:27+08:00: object predictor multiface collection 12/36 complete, 90000 controls, 9 contact/peak PASS and 3 FAIL. Case3011 both-hand frames689/1046/937, peak7.9233N; last4s contact toggles0 in all phases, saved controller replay PASS. All24 physical/3D images inspected and hashed in AGENT_INSPECTION.partial.json. Same Python2665211 live on sole297320/server05/step0, Rl99.9%CPU elapsed1:17:52; case3012 frame1150. Five pinned physics source hashes unchanged. This turn completed case3011 inspection after verified waits; no physics restart, model/protocol changes, or new training. Await full36 before guarded mixed_surface full-model qualification/three2000. Prior failed cases, mass10%failure, native failures and unfinished overall objective retained.

Latest 2026-09-15T18:21+08:00: objectpredictor same collector PGID2665146/Python2665211 LIVE sole297320/server05/step0;11/36 configurations complete82500controls,8physicalPASS/3FAIL, all22physical+3Dimages inspected/hashsaved. Latest3010both435/625/709,peak3.2174N,last4s allphase/hand contacttoggle0 andsavedcontrollerreplayPASS. Case3011running, directcomputeps Rl99.9%CPU elapsed1:11:33/frame1250 confirmed. Prior3009sidechatter and3001/3002/3006 failures retained. No model/source/protocol changes or newtraining this continuation; guarded mixed_surface three2000 remains prepared pending full36/inspection/fullhistorycoverage andGPUqualification. Overallpredictor goal unfinished; preserve mass10%failure/nativefailures/allocation, no duplicate/restart/release.

Latest 2026-09-15T18:15: objectpredictor corpus LIVE sole297320/server05/step0 PGID2665146/Python2665211; first10/36complete75000controls,7PASS/3FAIL, all20physical+3Dimages inspected/hashsaved. Latest3008both726/1075/1099,peak2.861N PASS;3010running. No newGPUqualification/training/modelendpoint; fullmixed_surface_v1 three2000 remainsprepared andbudgetunchanged. Addedfullknownmesh posttraining reporting: existingfull15626vertex nearest-distance metric plusper-acquisition grouping andfixedper-episode3Doverlay/pointcloud artifacts (probe first140..148ssample;supportfirstsample; no besterrorselection;rawstatecriteriaunchanged). New shape_report_qualification_v1 firstCPUfailed ModuleNotFoundError:newton beforemetrics;incident/log preserved0physics/updates. ExistingNewtonpy312+officialWarp/Newtonpath resolves sourceimport: qualifier_r1 PGID2739927 exit0 at10:07:14UTC,24alignmentchecks+13independent fullsavedmetric/cloud exactchecks pass onpriorold8frame realoutputs; all6old3Doverlays inspected. Tight-export margin corrected, same13checks exit0 at10:08:57UTC; finallayoutsample30 inspected, initialall6hashes retained. Preparedpipeline nowruns fullmeshgeometry CPU stages inNewtonenvironment, fullmodelstagesremainpy311. Notnewshapecompletion or32modelbenefit; alloldmass/nativefailures retained. Continue exact36/inspection/fullhistorycoverage then guarded mixed_surface input/fullGPUpreflight/oldendpointreplay/three2000 andallreadbacks/plots; no duplicate/release/manualgate. Latest3009both192/317/1110,peak9.086N qualifies, BUTsideforceschatter(final4stoggle19.1..38.7%),topstable; qualificationcounts/peakpass isnotstabilityevidence. LoginFSloglagobservedthisturn; directcomputeps2665211Rl99.9%CPU andfreshcomputelogconfirmednextcase, no restart. No code/protocol/trainingchanges inthiscontinuation.

Latest 2026-09-15T18:00: objectpredictor fullhand CADnormal candidate now INTEGRATED as optional normal_policy=hand_surface in dataset cachedgeometry/onlineencoder/runtime/evaluation/preflight; defaultstored keepsoldnormalinputs. CPUintegration PGID2718989 exit0 at09:49:41UTC242/242:15actualTRAINhistories/5traces, cachedvsruntime EXACT0, onlynormalcolumnschange, samealllabels/contact+forcegeometry, geometryfeaturesnotactileleak, allfull32x54x20/sparsebatchesfinite, wrongCADsignature rejects. No neuralGPUqualification or newtrained32endpoint yet. Newprospective mixed_geometry_mass_surface_v1 SUPERSEDES unexecuted mixed_geometry_mass_v1, all27oldprotocolfields exact, totalbudgetstill6000=3x2000/full138407503, same72cases/splits/hist32/age150/strides50and5/2+2batch/head1e-5/CSR/noDP/losses. New fieldsnormalpolicy/fullhand+querycodeSHA. Oldpreparedlauncherdisabledexplicitly; replacement train_mixed_geometry_mass_surface_prepared.sh requires complete36/hashes/agentinspection/adapter242, thencorpus/sampler/old8endpointactualGPUreplay/fullmixed3armGPUpreflight, thenthree2000/fullmodel+Adam+initialmodelhash/batch1+forcezero+freshmass+supportandprobe3030streaming. PREPARATION ONLY, notlaunched. Collectioncontinuessole297320/server05/step0 PGID2665146/Python2665211; first8/36 complete60000controls,5PASS/3FAIL, all16physical+3Dimages inspected/hashrecorded; sixth3005allthreecontacts/peakunder6N;3006Yphase0simultaneouscontactsFAIL,emptyYpanelretained. Allfivephysicalsourcehashes unchanged; querycodehash frozen bynewprotocol. Explicitattributionlimit: allarms shareforcefeedback acquisition; geometry-only predictor seesforce-conditionedhandmotion, nottouch-freeacquisition. Oldmass10%fail/nativefourfailures/overallgoalunfinished retained; no duplicate/release/manualgate. Latest3007both630/551/1013,peak1.1844N PASS; next3008physicsrunning. CPUfirst7historycoverage exit0 at09:57:53UTC: all4physicalpassing cases retainall3directionsbothhands8/8primarywindows; failed3001/3002/3006retain0/8bothhandwindows(anyhand0/2/8). No code/protocol/training changes this continuation; exactcollectorhandleverified live.

Latest 2026-09-15T17:45: objectpredictor collection LIVE sole297320/server05/step0 PGID2665146/Python2665211; fivecases3000..3004 complete37500controls,3PASS/2FAIL, all10physical/3Dimages inspected/hashsaved.3003both741/903/1283,3004both825/988/1114 pass;3001/3002 negatives retained. New saved hand-orientation audit41/41, CPUexit0 at09:38:24UTC: all36support+first4probe saved hand_normals_w are rotated local palmar axes EXACT(max0), not localCADsurface normals; docstring corrected without changing inputs. Anatomical-site vertex-normal vsaxis loadweighted probe angles36..62deg descriptive, NOT truecontact force error or negative-training cause. New full-original-hand-CAD nearesttriangle+barycentric vertex-normal candidate implemented OUTSIDE currentdataset/model/physics: fiveactualTRAINtraces30200frames, basic15/15 andcontact-containing32history/rigidworldtransform15/15 pass, cachedhistoryexact0/rigidmax5.619e-8. Surfacecentroid distancep95<=.437mm,max1.639mm inthese5; source+fullhandmeshhashes snapshotted. CPU2711119 exit0 at09:41:47UTC, equivarianceexit0 at09:44:03UTC. Candidate NOTwiredtoencoder/training; no normalpolicy revision or newmodel endpoint. Current declared mixed72/full138407503/three2000updates still PREPARATION ONLY, no newupdates. Isaacheadlesssource+official5.1requirements audit recorded; render() already NO_GUI_OR_RENDERING noop, coreplay stillapp.update; no verified fix/no fifth launch. Continueexact36cohort/inspection/fullcoverage and complete finalmixed input/fullGPUqualification before declaredtraining; consider qualified CADnormal candidate explicitly before training, never silently reinterpretpalmaraxes. Alloldmass10%fail/nativefailures/globalgoalunfinished retained; noallocationrelease/duplicate/GPUconcurrenttraining.

Latest 2026-09-15T17:32: object predictor corpus still LIVE on sole297320/server05/step0, PGID2665146/Python2665211, no restart. First3/36 cases saved/22500controls:3000 passes all phases;3001 FAIL both4/6/909;3002 FAIL both7/28/928 and58.0475N peak. Allsix actual physical/3D images inspected and hashed in AGENT_INSPECTION.partial.json. Fixed32-history primary140..148s coverage: bothhands all3directions8/8,0/8,0/8 respectively; anyhand8/8,0/8,2/8. Source/controller clocks exact, no selected-window removal or physical rerun. Mixed72-case protocol PREPARED before training:36probe+36oldmasssupport, split48/12/12,32clock-only history/time_scale150, per-source strides50/5, balanced2+2 batch4, full138407503/2000updates per3arms(6000total), unchanged losses/head1e-5/CSR/noDropPath. New sampler CPU35/35 on2realprobe+24TRAINsupport, PGID2685312 exit0 at09:26:24UTC; independent prior-real metric CPU21/21 exit0 at09:30:32UTC max3.06e-12. Guarded builder/full-corpus sampler audit/fullmixed GPUpreflight/fullendpoint+Adam audit/stratifiedreport and serial training/eval script IMPLEMENTED but not executed: train_mixed_geometry_mass_prepared.sh requires completed36 and hash-bound agent inspection, not user approval. No learned32endpoint or new training yet; collection source five hashes unchanged. Existing fresh mass raw12.6565/calibrated11.5610% still10%FAIL. Continue exact36cohort, inspect allphysicalplots and fullhistorycoverage, qualify finalmixed data/fullGPU then declared three2000 arms and all saved/batch1/forcezero/freshmass/streaming readbacks. No claim mass/arbitraryshape/material/policy prerequisite complete; preserve all prior negatives and allocation.

2026-09-15T17:10 active object-predictor continuation made actual progress. Multi-face
stable qualification all3configs COMPLETE:22500controls, phase contact and<=30Npeak
criteria allpass, allphysical+3Dplots inspected, all22500controller-pose replays match
within2.98e-8m and touch flags exact. A480/982/832bothframes, nominal610/601/1011,
B935/600/1136; respectivepeakmax3.24/6.01/13.43N. Fixedlast4sphase contact toggling0
allhands/allcases. First1500r1 and gentleA failures retained; no neural benefit claim.
32frame episode_uniform_recent clock-only path: CPU291/291, full original137253744
pretrained/total138407503 GPU15/15, repeats0, old8frame39window replay max1.124e-7.
Historyqualification2615162 exit0 at08:42:33UTC; originalCPUimportexit143 retained.
New multicase corpus COLLECTING on sole297320/server05/explicitstep0, tmux%1,
PGID2665146/collect_multiface_dataset, launched09:09:01UTC.36newconfigurations/24train6val6test,
IDs3000..3035, seed310330, geometryseeds310430..310465, mass.25..1.2kg and independent
XYZscale.85..1.15, each7500/max270000actualcontrols. Sourcehashes and split fixedbefore
physics, TRAINincludesrangeboundaries; allfailedcases retained. Firstcase3000running,
no dataset endpoint or new training yet. Eachcase saves physical/3Dplots and full
controller readback. Do not edit the five hashed physics sourcefiles while collection
is live. Preparation/model/training work outside those files can continue.
Next: finish exactcorpus/physicalinspection/coverage readback, declare and execute
matched full32frame training/evaluation; keep mass observability explicit (ground
probing is not a weight measurement). Original mass10% failures, pose/shape limitations
and freshIsaacLab renderer block remain. Full objective not complete; GPU retained.

2026-09-15 object predictor is the active prerequisite; see
[sensor-to-object work](../scripts/sugar/object_predictor/README.md). Full official
Utonia 137,253,744 pretrained / 137,542,639 total parameters, full Adam, no replacement
backbone. Three matched three-arm rounds completed 600 updates per arm. Original
readout LR instability and scatter reload failures retained. CSR gives exact saved
prediction replays. Frozen TRAIN/eval audit found a 7.844x size-loss gap from original
stochastic depth; disabling existing regularizers during task training removes it.
Third-round Newton batch-one center 24.32/15.39/15.50 cm, size 8.16/10.01/9.88%, mass
27.33/28.71/28.11% (geometry/contact/force+area). Still no overall force benefit or
policy-ready predictor. All backend evaluations, streaming and full known-mesh
metrics complete. Exact-initial comparison retained as 61/63; separate numerical
comparison 63/63 at existing1e-5, maximum initial difference7.33e-6; all labels and
every training batch exact. Original bitwise failure is not reclassified.

Controlled mass calibration completed: all three full600 models/Adam, all batch1
calibration and mixed-Newton evaluations, streaming and physical mass plots. Held-out
mass errors geometry/contact/force are32.65/32.19/12.03%; force-zero preserving contact
and area50.01%. Force information is learned, but <=10% target fails (3/4 criteria).
Mixed-geometry transfer remains negative: force mass40.62%, center21.56cm.
VAL-only one-log-bias calibration and FRESH physical TEST are complete NEGATIVE.
Force scale1.151142; all12new masses/2400controls/2340contact, seed310120/IDs2000..2011,
fixed geometryseed310017. All12 support checks pass. Fresh468batch1 windows: raw
geometry/contact/force mass49.91/49.39/12.66%; calibrated68.51/68.45/11.56%, force-zero
37.38%. Both raw/calibrated3/4criteria: <=10%FAIL. Lowest.2515kg has persistent
calibrated49.53%error, not only transient; TRAINmin.2749kg and VALmin.5644kg, documented
range limitation without filtering TEST. Qualified409windows: directphysics1.546%,
calibratednetwork9.717%; primaryallwindowsunchanged. All12masscurves/scatter/physical
coverage inspected. Independentreadback44/44; full200framestreaming39windows max1.124e-7,
reset/duplicate checks pass. Pipeline2523763 exit0 at08:09:43UTC,0newoptimizerupdates.
Runtime MASS_ESTIMATOR_RUNTIME.json explicitly marks experimental/not-ready/failed10%.
Original failures preserved; no arbitrary pose/shape/material/policy claim.
Old297084 externallyCANCELLED15:46:03 afteralloldchildrencomplete; initialreplacement
invalidQOS creatednojob. Sole297320/server05/explicitstep0 retained in tmux%1; allGPU
children terminal. Overall prerequisite still not policy-ready. Further work should
address physical observability and varied stable contact coverage, not test-bias sweeps.

Mixed-data conservative support diagnostic has only33TESTwindows in one episode:
direct normal sum/g mass error2.76%, third learned force model13.96%. Geometry-only
input is exactly identical across1170same-clock TRAIN/TEST comparisons, so independent
object properties are unavailable to that control. Voxel force averaging loses>1%
in only3of2463mixed contact frames (1TRAIN/2VAL/0TEST), a retained rare limitation,
not the primary mass failure explanation. Four fresh native IsaacLab starts failed
Vulkan; last initialized all54forcefields. Historical80-frame native trace uses
verified legacy force separation and distinct penalty calibration. No fresh native
rollout or native fine-tuning. Keep allocation and all prior demo-following negatives.

Latest 2026-09-15T12:09:18: actual_state_expert01512 full matched rollout CLOSED MIXED. Demo original/repeat318total/160generated versusold321/163; alternate400/242 versusold321/163,1of2distinctdemo branchesfull400 (1/3includingrepeat). Zero original/repeat231/73 versus208/50; alternate355/197 versus209/51,0/3full400. All1853newactual/905generated controls, full frozen models, all905GPUbatch1actor outputs BIT-EXACT, both49routingchecks/all46physicalpanels complete. Context20checks and old/new40checks pass; all4common-window body/box RMS lower butoriginaldemo terminates3steps earlier. InitialCPUbatched thresholdfailure and twozero-readback launcher/parser incidents retained,0repeatedphysics/optimization; finalzeroCUDAreadback exit0 at12:07:05. Earlierprediction remainsTRAIN7/9versus8/9/reused2/2,9of11MSEworse,113strictchecks. Usefulalternate execution reference only, known48position assistance/oneTRAINpair/one seed/noSMPbenefit. AllGPUchildren terminal, sole297084/server06/step0 retained. NEXT_ORIGINAL_FAILURE_PROTOCOL declares saved last32validpreterminal+fullthreshold chronology/officialbodyterm/currentgeneratedplans/same-worldknownplanqueries audit before new optimization or recovery budget; not yet executed. No old513/release/duplicate/manual gate.

Latest 2026-09-15T12:01:41: NEW ACTUAL demo rollouts complete MIXED: original/repeat318total/160generated (old321/163), ee_body_pos terminated; alternate400total/242generated (old321/163), full400PASS, newdemo1/3 includingrepeat, notoverall success. All23physical/reference/XYZ/command panels inspected; alternate late boxdistance~.23m, original~.08m, known48position assistance retained. Full562saved GPUbatch1 actor actions BIT-EXACT to execution, full601629frozen audit10checks pass. Initial CPUbatched alternate max1.0490417e-5 exceeded1e-5; old49checkreadback48/49 and failed pipeline1469554 exit1 at11:54:19 preserved separately. CUDAexact replay reconciles canonical49checks, no tolerance relaxation/physics rerun. New audit_then_zero pipeline PGID2452106 launched12:00:06 onsole297084/server06/step0: exactaudit+zero CPU40preflight pass, zero originalphysicsnowrunning, thenrepeat/alternate and CUDAexactreadback. Strict prediction comparison113checks complete, TRAIN7/9 versusold8/9/reused2/2,9of11MSEworse; do not confuse oneclosedloopbranch improvement withoffline orSMPbenefit. Finish exactzero andallphysicalplots/context+oldnewcomparison; no newtraining/old513/release/duplicate.

Latest 2026-09-15T11:46:51: actual_state_expert01512 completed overnight on296322/server06: pipeline3689692 exit0 atSep15 00:40:19, both full8327408 arms512/fullAdam, CPU1084/H200BF161084 and29loss readback checks pass, all11phase5condition32draws saved. All88horizon panels now inspected. Prediction NEGATIVE: TRAIN7/9 versusoldmeasured32 8/9,277and298fail;reused218/2582/2. CorrectMSE worse9/11 (197and218 improve), no offline benefit. No new rollout overnight: automatic pipeline ended before agent plot inspection/strictcomparison; oldallocationexternallyCANCELLED04:01:04, stepended04:01:33, completedtraining untouched. Emptyuserqueue verified then sole replacement297084/server06/explicitstep0 granted, tmux curiosity_actual_rollout_20260915/%1. New generator_actual_state_rollouts_job297084 PGID1469554 launched11:46:11: strict fullmodel/Adam/exposure/initialsamples comparison first, then serial newdemo+zero CPUrouting/3x400physics/fullreadbacks,max2400controls,0newupdates. No physicalendpoint yet; continue exactchild and inspect bothphysicalsets then context/oldnewcomparison. Previousdemo163 versuszero50/51generated steps andboth0/3 remain. Retain known48position assistance, allprior failures and soleallocation; no independentdeployment/SMPclaim.

Latest 2026-09-14T21:12:43: External resource wait verified for third consecutive goal turn: sole296322 PENDING AssocGrpGRES, live salloc3326311/tmux curiosity_actual_state_20260914/%0; no allocation/GPUpreflight/training artifacts. CPU1084 and real-trace comparator16checks complete, posttraining rollout/readback/comparison glue ready. Further meaningful training/physics progress requires external quota availability. Keep pending job and exact buffered srun/automatic BF16+both512 pipeline alive; no duplicate/release/manual gate. Active work is resource-blocked, objective not complete.

Latest 2026-09-14 21:11: sole296322 remains PENDING AssocGrpGRES, explicit compute entry and fullBF16/both512 pipeline buffered in live tmux curiosity_actual_state_20260914/%0; no GPU child/training endpoint yet. Posttraining actual-state rollout pipeline now implemented: demo+zero serial CPUrouting/3x400physics/fullsavedreadback each,max2400controls total,no newoptimization. Full new/old same-condition comparator implemented and qualified on prior real demo/zero traces16checks: all saved common intervals/durations/terminal/box metrics exact, expected condition/model mismatches detected; no newrollouts inqualification. Empty jointlyvalid following/XYZ now records completed negative evidence with no fabricatedcurve. Existing training-mode CPU1084checks remain valid; loss/model unchanged. Next wait sameallocation, complete GPUqualification/training/88panel+strictcomparison, execute generator-actual-state-rollouts, inspect bothphysicalsets, then context and previous-model comparisons; preserve all oldfailures/known48position assistance/noSMPclaim.

Latest 2026-09-14 21:07: matched ACTUAL generated rollouts complete: demo original/repeat/alternate321total/163generated each versus zero208/50,208/50,209/51; both0/3full400. Zero pipeline1309409 exit0 at20:52:42 and CPU49checks exit0 at20:53:32; all23zero panels inspected. MATCHED_ZERO_COMPARISON all21checks pass; common49/50posthandoff frames boxcoordinateRMSE demo.000959/.001649m versuszero.031422/.035682m, but no independent seeds/deployment/SMP benefit. Actual-state expert corpus355examples/4uniquegroups and16checks complete20:55:32; targets are desired existing expert8x36command proposals on actual visited states, NOT counterfactual physical futures or proven recoveries. New matched_generator_actual_state_expert01512 protocol+full8327408glue complete, original18/144/q+.25rank+.1frozenreplay unchanged; extra144expert rows/update fixed.1 BOTHarms,73728extra rows/arm, same releaseinit/normalizer/seed272084/both512. Both eval-module CPU1084checks and actual model.train() Dropout CPU1084checks PASS; CPU3589460 normalexit0,0updates. Disabled-aux fulloriginal loss/allgradients/forwards/RNG exact in both initial+learned arms; activeoldcomponents/RNG exact, finite nonzeroexpert correction and all512exposure schedules exact. BF16/training notstarted. Old296018 externallyCANCELLED21:01:06/stepended21:01:35 after all oldchildrencomplete; currentqueue verifiedempty before sole replacement296322 PENDING AssocGrpGRES,tmux curiosity_actual_state_20260914/%0. Explicit srun step+guarded pipeline command buffered to autostart aftergrant; no duplicate/release/restart. Complete true CPU/BF16, both512/all11x5x32/all88panels/loss+strictcomparison then unchanged bounded actual generated rollout. Preserve allprior failures, known48position assistance, SMP numerical repeat caveat; no manual gate.

Latest 2026-09-14 20:49:43: FIRST ACTUAL GENERATED-COMMAND ROLLOUTS COMPLETE NEGATIVE. measured32_generated_command_rollout158_r1 pipeline1295431 exit0 at20:44:43: original/repeat/alternate each321controls,163generated from158,963actual/489generated total,0/3full400. Original/repeat ee_body_pos termination; alternate obj_pos. Full8327408 Generator and601629 Tracker frozen, known48position assistance explicit. Original/repeat all41trace arrays andsha256 EXACT; allprefixes tooldnumericbaseline EXACT. CPU40routingpreflight, livefirstsample max9.54e-7, full489actoroutputs CPUreadback max9.54e-6; all49savedroutingchecks pass CPUreadback exit0 at20:45:39. All23physical/reference/XYZ/command panels inspected. Shared162valid posthandoff frames prefer own reference on all4box/body criteria, but400budgetfail makesoverallnegative. Full official SMP pipeline1306377 exit0 at20:47:22:153matchedposthandoff windows158..310; Carry energy new/baseline1.09495(original)/1.05962(alternate), allwindowsstillCarry-over-Kick. Wholeprior5674161params includesstoredmodelcomponents, frozen; no SMPbenefit.16declared savedmetric checks pass; extra repeat-derived bitwise check FAILS (raw41arrays exact, feature4.77e-7/score1.67e-6 numericaldifference), preservedwithout rerun. Userpriority continues actual matchedzero control: CPU40preflight exit0 at20:47:40; generator_actual_zero_rollout_job296018 PGID1309409 onsole296018/server34/step0 running same3x400/max1200 andoriginal158handoff usingexistingzero512endpoint. Complete actualcontrol/readbacks/plots/comparison before correctivetraining; no oldbudgetextension, fabricatedlabels, independentdeployment/SMPclaim or allocationrelease. First missinglast_command incident preserved separately; correctedrealissuedhistory getter andartifact checks used in all completedr1arms.

Latest 2026-09-14 20:40:03: measured32 matched full8327408 both512 completed pipeline1267703 exit0 at20:34:31, full11phase5condition32draw/all88panels inspected/loss readback; strict comparison1290975 exit0 at20:36:22 all106checks. All11correctMSE lower than prior self18 and all8oldpassingTRAIN nonregression pass; TRAIN8/9/reused2/2 unchanged,277 .021517207846 versus .022297633812 (-3.50%), overall prediction floor stillNEGATIVE. User explicitly prioritized actualrollout; separate reference-assisted generated-command diagnostic declared, not relaxed prediction/deployment/SMP criterion. FullCPU routing40checks pass actual savedinputs/fullofficialmodels/lag5/originalinterpolation. First pipeline1290987 original/repeat failed at158beforegeneratedaction: officialgetter last_command absent when built-in generator disabled; Kit incorrectly returned0 aftertraceback. Preservedlogs/incident, exactpipeline stopped duringthirdsceneinit, no replacementtraining. Fixed getter receives actualissued t-5 thenrestoresfield; endpointartifacts checked irrespectiveexitcode, partialtraces persisted onfuturefailures, firstlivepred comparedtosavedprimary beforeaction. New measured32_generated_command_rollout158_r1 PGID1295431 onsole296018/server34/step0 running three400stepattempts/max1200controls/max726generated, unchanged full607 Tracker and48knownposition assistance, no updates. Continue exactrollouts/physical+routing readbacks/plots, preserve all failures. No independentdeployment/generated-only48/SMPclaim; retain allocation.

Latest 2026-09-14 20:26:11: measured32 partitioned-leaf CPU134/134 and actual H200 BF16 406/406 PASS. Prior r1 pipeline1260198 then failed before workspace.run/optimizer construction because observer accessed optimizer too early; zero updates, original artifacts preserved with exact hashes in pre_optimizer_binding_failure and PRE_OPTIMIZER_BINDING_INCIDENT.json. Observer now records actual Adam name/ID bindings at first training forward after official initialization; no model/loss changes. Current r2 generator_measured_robot_pipeline_r2_job296018 PGID1267703 on sole296018/server34/step0: zero_context completed512 exit0, demo_geometry logged246/512; both runtime bindings contain full8327408. Passed preflights reused, not rerun. Continue both512/all11phase5condition32draw/fullmodel+Adam/loss readback, inspect all88 panels then strict comparison to old self18 with unchanged9/9TRAIN+reused2/2+all8oldpassingTRAIN nonregression. No new evaluation endpoint, generated physics or SMP benefit yet; no old budget extension/allocation release/duplicate.

Latest 2026-09-14 20:19: measured32 AMP issue diagnosed and fixed without relaxing exactness. Full144row original/repeat/wide68-slice/partitioned-leaf diagnostic1250016 exit0 at20:13:14: onlyoldrobot firstweight differed9177entries,max7.8082e-6; originalrepeat and partitionedleaf allold gradients exact. Saved5checks confirms old equalsBF16-rounded slicedgradient. Same affine now retains original36leaf weight plus8192zero measured_weight in same module/full8327408; no hidden layers/tokens, old weights never reshaped. Explicit optimizer name-to-ID bindings handle the oneextra Parameter. Width68 CPU134/failedBF16/rootpipeline records preserved. New partitioned_affine_preflight_r1 CPU1254114 exit0 at20:16:54,134/134 fullchecks. New retained pipeline generator_measured_robot_pipeline_r1_job296018 PGID1260198 live; actualBF16 child1260697 has passed initialzero fulloldgradients/all9TRAIN original sample controls and continues remaining fullstates/timechecks before automatic both512/all11x5x32/fullmodelAdam/lossreadback. No training endpoint yet. Sole296018/server34/step0 retained; no originalupdate513/physics/SMPclaim.

Latest 2026-09-14 20:13: measured32 first pipeline1243545 TERMINATED exit1 at20:10:41 before training: actualBF16 child1243975 failed only zero_context_initial_every_original_parameter_gradient_exact; CPU134pass, allfullforward/32primary-control and RNG checks pass,0trainingupdates. Preserve GENERATED_BF16_PREFLIGHT.json.partial and WIDTH68_BF16_INCIDENT.json. Diagnose fullnamed gradients using original/repeat/wide68-slices and same-affine partitioned-leaf control, tag generator_measured_robot_bf16_diagnostic_job296018 launched1250015; suspected AMP cast/gradient accumulation ordering, not established untilactualresult. No tolerance relaxation or oldpipeline restart. Retain sole296018/server34/step0; fix input-affine compatibility then complete new CPU/H200 preflight before both512/fixed evaluation. No model reduction/newloss/physics/SMPclaim.

Latest 2026-09-14 20:10: measured32 implementation COMPLETE: existingrobot affine36to68 uses original contiguous36-column contribution plus32new columns, no newhiddenlayer/token/denoiser; zero8192new coefficients/full8327408, construction RNG restored, original normalizefields and fullmodel restoration/factory/training/readback/comparison adapted. CPU1239054 exited0 at20:09:36,134/134 checks: botharms initial+oldself learned endpoints, actual144row complete oldloss/records/every oldgradient/forwards/RNG exact, nonzeronewcolumn gradients, all16aux/all50q/clocks, full Adam grouping and original firstphase sampler replay. Sole296018/server34/step0 retained pipeline generator_measured_robot_pipeline_job296018 PGID1243545 launched20:10:00; actualBF16 stage then bothseparate512/all11phase5condition32draws/all88panels/fullmodel+Adam/lossreadback automatically. No endpoint yet. Strict comparison retains self18 baseline8/9, requires9/9 plusreused2/2 andall8oldpassingTRAIN MSE nonregression. No terminal-loss replacement, source/teacher/weight/seed/normalizer change, old513, generatedphysics or SMPclaim. Complete remaining GPUpreflight/training/evaluation/visual/readback evidence without manual authorization.

Latest 2026-09-14 20:04: full terminal-objective qualification COMPLETE NEGATIVE: GPU1209076 exited0 at19:57:50,144new student paths+144exact full-gradient replays+108controls/6336official sampler+192component forwards,1204checks/all9panels and20independent saved checks (CPU1222632 exited0 at19:59:08). Fixed.1 terminal replacement supports5/9TRAIN phase means in bothstudents, below predeclared7/9 despite8/8global seed descent and277positive;178/197/221 negative both. Reject replacement without weight sweep/newtraining. Next actual causal-input audit1226494 exited0 at20:01:51,157checks/all9plots and23saved checks (CPU1228663 exited0 at20:02:54): all18joint29+gravity3 exactly pre-action TRACE/raw/branch/modelobs, samewithinpair; current full4encoders ignore them underlastcommand mode. Currentjoint versuscommand(t-5) RMS.05584..12772rad includesclockoffset, notsame-time trackingerror. New matched_generator_branch_measured_robot32512 PROTOCOL+PREPARATION only: existingrobot input36to68,zero8192newcoefficients/full8327408, preserveallold8319216/normalizer/RNG, same18/144/seed272084/.25rank+.1oldselfteacher replay/both512/all11x5x32. No newmodelmodules; adapter/factory/restoration/full144CPU+H200BF16 implementation NOTyetdone, no newtraining. Require oldparameter forward/gradient/RNG preservation and nonzeronewcolumn gradients before training. Baseline self18 TRAIN8/9; require9/9+reused2/2+all8oldpassingTRAIN nonregression. Sole296018/server34/step0 retained, allchildren terminal. Preserve all prior negatives; no generatedphysics/SMPclaim or old513.

Latest 2026-09-14 19:55: four-endpoint TRAIN residual/encoder audits COMPLETE: centroid1202426 exit0 at19:49:03,708checks/all9panels and182 independent saved checks; full4encoder1203665 exit0 at19:50:26,46checks/all9panels and43 saved checks. All4 phase277 mean biases exceed unchanged.01260763 threshold; self84=.01495079,self400=.01436304 even without variance. Relative277/298 geometry branch separation increases at both studentseeds;197/221 decreases; descriptive only, not causal or missing-information proof. No exact joint-token aliases among18TRAIN. Prospective fixed.1 full-official16step terminal-MSE replacement qualification prepared before new gradients, CPU1208720 exited0 at19:54:43. GPU tag generator_terminal_objective_qualification_job296018 launched via sole296018/server34/step0 Held lock, launcher1209075: oldself gradients reused, newself144 fresh-for-this-student paths+144exact gradient replays,108controls,6336official sampler+192component forwards. Both frozen learned students only,0updates/physics; q+rank+.1terminal replaces local frozen-replay auxiliary, no model modules. Fixed mean/6of8seed/7of9phase including277 raw-descent criteria; full144row CPU/H200BF16 and separate matched protocol required only if passes. Prior0/3 conflict replication and all overallprediction negatives retained; no coefficient/teacher/solver/thirdseed sweep or old513. Retain allocation; no generatedphysics/SMPclaim.

Latest 2026-09-14 19:47: fixed-teacher student-seed replication COMPLETE: pipeline1171742 exited0 at19:45:21, all4fullmodels each512 updates/fullAdam/all11phase5condition32draws/all176horizon panels inspected. Strict comparison1200381 exited0 at19:46:43,203/203 checks. Relative improvement repeats: TRAIN7/9 to8/9 at both seeds272084/272400; equal-phase TRAIN MSE decreases10.67%/19.02%. Both reused218/258 pass, but overall prediction NEGATIVE:277 accuracy fails and oldpassingTRAIN nonregression fails. Newseed self277=.02055072784,298=.01296106167;197/221/261 regress. Fixed teachers retained, only student RNG replicated, not end-to-end teacher replication/generalization. Next four-endpoint9TRAIN centroid saved-draw audit launched tag generator_four_endpoint_centroid_job296018,launcher1202425; then independent readback and full4encoder TRAIN condition audit. No thirdseed/newtrainingbudget/generatedphysics/SMPclaim. Sole296018/server34/step0 retained.

# 当前科学状态

Latest 2026-09-14 19:35: seed272400 ranked18 variant COMPLETE NEGATIVE: both512/fullmodel+Adam/all11phase5condition32draws/all88horizon panels viewed and hashed inspection records,55saved metricchecks+23training-loss/exposure checks pass; TRAIN7/9,reused2/2,277MSE.02677622065 [22,32] failsaccuracyandbranchpreference,298.01649124175 [32,32] failsaccuracy. Equalphase9TRAINmean.007652937674 (oldseed ranked18 .007314971545); not a cross-variant benefit result. Ranked pipeline1173312/lossreadback1185435 exited0. Second self18 pipeline1185719 andzero trainer1185849 live19:34:46,183actual logrows throughglobal_step182,under original groupPGID1171742/Python1171751/sole296018/server34/step0. No secondvariant endpoint yet; finish both512/all11x5x32/88panels/55saved+23loss and implemented strict bothseed fullmodel/Adam/initial/batch/budget comparison before relative-improvement conclusion. All176horizon images require hash-bound agent inspection records; first11 done. Next frozenfourendpoint TRAIN centroid/condition representation glue implemented and compiled,NOTexecuted: same32saved samples/9TRAIN/all4models, exactold18centroid rows, then16batched officialencoder calls18rows/3x256tokens, fullmodel+normalizer/RNG/swap/zero controls,0denoiser/draw/update/physics. Require plots+independent savedreadback;no thirdseed/teachercoefficientsolver sweep/old513/toy/newmodel/generatedphysics/SMPclaim/human gate/release. Preserveallpriornegativeobjectives and fullscope.

Latest 2026-09-14 19:23: fixed-teacher seed272400 both actual CPU AND H200BF16 preflights pass76/76each (four reports),full8319216/18TRAIN144rows/botharms initial+learned/fullgradient/forwards/RNG/16aux+50q. Retained pipeline PGID1171742/Python1171751 live onsole296018/server34/step0; ranked18 pipeline child1173312 and zero_context trainer1173470 live19:22:31,14actual optimizer log rows throughglobal_step13 verified. Self18 training notstarted and no endpoint yet. Serial declared four512arms/2048totalnewupdates,eachfullmodel+Adam/all11phase5condition32draw/88panels/lossreadback; then176panels and implemented strict two-seed comparison required. Relativeimprovement hypothesis fixedbeforetraining: lower equalphase9TRAIN MSE AND more fullcriterionTRAINpasses; previousseedoverallnegative preserved regardlessnewseed. Fixedteachers/normalizer/data/fullinit/weights/budgets/primary32seeds unchanged;only studenttrainingseed272084to272400. Existingcheckpoint profile complete271checks+738saved/18plots,277neverpassesall9states; no latecheckpointrescue/old513/thirdseed/teachercoefficientsolver sweep/generatedphysics/SMPclaim/manualgate/release. Continue exactpipeline and next predeclared allfourfrozenTRAIN residual/condition representation comparison after all evidence.

Latest 2026-09-14 19:21: fixed-teacher seed272400 replication both actual CPU preflights complete76/76each,18realTRAIN144rows/full8319216/botharms initial+source-learned fullgradient/forwards/RNG/all16aux+50q checks; CPU group exit0 at19:20:38. New generator_fixed_teacher_seed_pipeline_job296018 PGID1171742 Python1171751 launched19:20:53/liveverified on sole296018/server34/step0. Serial pipeline requires both actualH200BF16 preflights, then ranked18andself18 eachzero+demo512/fullmodel+Adam/all11phase5condition32draws/88panels and full loss-exposure readback. BF16/endpoints NOTyetcomplete atthis observation;2048newupdatesdeclared,0old513/physics. Compare bothseeds under unchangedfullprediction criteria, plus preregistered newseed lower9TRAINmean AND moreTRAINpasses for relative-improvement replication;originaloverallnegative cannotbe averagedaway. Both176panels and strict fullstate/batch/budget/Adam/savedreadback remain required,then fixed allfourTRAINresidual/representation comparison. All9savedcheckpoint curves271checks/738readback/18plots complete,277neverpasses (.884291 final versus.5). Preserveallpriornegatives,seed/noise distinctions,frozenoriginalteachers/32evalseeds,allocation;no thirdseed/teachercoefficientsolver sweep/human gate/generatedphysics/SMPclaim/release.

Latest 2026-09-14 19:18: all existing self18 checkpoint learning curves completed PGID1153593 exit0 at19:13:53:4032new full16step paths+72exact initial/512controls,65664officialforwards,271checks/all18learning+horizon panels inspected; independent savedreadback738checks exit0 at19:14:39. All9recorded states fail277 accuracy (mean ratios7.14863,2.09240,1.92251,1.80393,1.56873,1.24832,.982485,.914664,.884291; criterion<=.5),no pass-to-fail transition. Machine decision selects one fixed-teacher student-training seed replication. Group generator_fixed_teacher_seed_replication272400 prepared19:17:08, new ranked18/self18 seed272400 variants, eachzero+demo arms/512updates/full8319216/18TRAIN144rows/same releaseinit/oldnorm/optimizer/schedule/.25rank+.1aux; only studenttraining RNG changes. All originalteachers/replayarrays andprimary32seeds fixed; not end-to-end teacher-seed replication. Total2048newupdates predeclared, no original513. Separate prospective hypothesis19:17:55 requires lower equalphase TRAIN MSE AND more fullcriterion TRAIN passes; old7to8/mean.00731497to.00653448 evidence remains overallnegative. No bestseed/averaging rescue/thirdseed. Actual two-variant CPU preflight group generator_fixed_teacher_seed_cpu_job296018 launcher1168524 nowrunning; BF16/training NOTstarted. After CPU+BF16, allfour fullmodel+Adam/endpoints,all11x5x32 pervariant/176panels/loss-exposure+savedreadback/strict bothseed comparison, then allfour frozenTRAIN residual/representation comparison. Sole296018/server34/step0 retained; no human gate/generatedphysics/SMPclaim/release.

Latest 2026-09-14 19:07: actual496-vs512 full sampler comparison completed PGID1145307 exit0 at18:58:34:144new496paths+36exact512controls/all9TRAIN8freshseeds/99checks/9horizon panels; independent savedreadback162checks exit0 at19:00:52. FivephaseMSEs regress slightly,277 +0.02785%; large277 residual already present496,notmainlylast16updates. Actualdelta79checks and all9projectionplots complete; endpoint projections not historical Adam-unit direction. All8original/replayed epoch fullmodel+Adam checkpoints exact25checks at18:59:31; actualupdate clocks1,65,129,193,257,321,385,449. Full32primaryseed existing-checkpoint curve protocol prepared: firstupdate fullmodel exactlyinitial,4032newpaths+72exactreferencecontrols/65664officialforwards,0trainingupdates/physics. New generator_existing_checkpoint_profile_job296018 PGID1153593 Python1153600 launched19:06:11/liveverified on sole296018/server34/step0. All9TRAIN atall8savedstates+512,initialand512arraysreusedafterexactcontrols;9learningcurves+9earliest/finalhorizon panels+savedreadback required before automaticnextbranch. No selectedcheckpoint,old513,newtrainingadmission,physics/SMPclaim,release ormanualgate. Preserve allpriorprimary8/9+reused2/2 andfreshconflictreplication0/3 negatives.

Latest 2026-09-14 18:57: exact optimizertrace r1 completed all512training updates/fullendpoint+Adam/all512batches; PGID1139943 exit1 at18:54:43 onlyduring own496snapshot readback becauseweights_only default rejects originalOmegaConf optimizer metadata. SNAPSHOT_READBACK_INCIDENT retained; explicit trustedown snapshot load and separate --readback-existing CPU launcher1144009 exit0 at18:56:23:570checks includingALL512lossfields,allRNGhooks/full512snapshotmoments and exactFP64last16delta telescope pass. No repeatedtraining afterreadbackincident; exactly512extraexecuted replayupdates,0newunique/oldendpointextension. New CPU actualdisplacement versus frozen endpointgradient readback tag generator_optimizer_delta_readback_job296018 launcher1144539 running. Then originalfull496vs512 predeclared144new496paths+36exact512freshcontrols/all9TRAIN8seeds/9horizon panels/savedreadback. Existing officialepoch checkpoints every64 also retained (epoch index not updatecount); do not infer intermediate clocks or selectcheckpoint. Allpriornegativefresh0/3,primary8/9+2/2/native/physical/camera retained. Sole296018/server34/step0 preserved;no gate/old513/generatedphysics/SMPclaim/release.

Latest 2026-09-14 18:52: first optimizertrace PGID1137548 exited1 at18:50:31 beforeoptimizer creation ortraining update: recorder incorrectly expectedoptimizer inworkspace constructor,official run createsit later. Failure directory/record/RECORDER_INITIALIZATION_INCIDENT preserved,0extra executedupdates,originalendpoint untouched. Fixed passivehook registration insideoriginal model.get_optimizer factory;officialrun/optimizer.step unchanged. Separate r1 protocol+directory matched_generator_branch_self_replay01_gaps512_optimizer_trace_r1,tag generator_self_optimizer_trace_r1_job296018 PGID1139943 Python1139949 launched18:51:53/livepsverified onsole296018/server34/step0. Sameexact512replay budget andallmatching requirements,actual497..512deltas/496+512snapshots thenpredeclared9TRAIN8freshseed496vs512comparison;no endpoint yet. Keep352Adamreadback/0of3freshconflictreplication negatives,allpriorprimary/physics failures,retainedallocation. Neverrestart onobservationfailure/release/duplicate/old513/checkpointselect/generatedphysics/SMPclaim/human gate.

Latest 2026-09-14 18:50: storedAdam CPUdiagnostic exit0 at18:46:56,9integritychecks/all287officialbindings/fullmodel+moment exact/9plots/352independent readback checks at18:47:54 complete. Per-unitlr historicalmoment opposes178(-.0384573)/197(-5.42686),overall+2.69349 and7/8fresh global seed meanspositive;actualendpointlr0,not actualupdate512 evidence. Freshrawobjective conflict replication remainsNEGATIVE0/3,notnewauxiliary admission. New separate exact original512 replay generator_self_optimizer_trace_job296018 PGID1137548 Python1137554 launched18:50:14 onsole296018/server34/step0,liveps verified. Replaydir matched_generator_branch_self_replay01_gaps512_optimizer_trace,fullreleaseinit/18TRAIN144rows/seed272084/oldnorm/samerank.25+aux.1 andteacher/officialBF16loop. Passive originaloptimizerhooks record actual497..512FP64 parameterdisplacements andfullmodel+Adam post-update496/512 snapshots;all512post/16pre RNG checks.512additionalexecuted replayupdates,0newuniqueupdates/oldendpoint extension. Require fulloriginal model+Adam/all512batches+ALLlossfields/initial exact and16delta telescope before endpoint-projected direction readback and full496-versus512 frozen9TRAIN/8freshseeds:144new496paths+36exact512controls,reference512freshoutputs reused,9horizon panels. No primary32rerun/checkpointselection/old513/physics/SMPclaim/human gate/teacherweightsolver sweep/release. Primaryself18stilloverallnegative8/9+reused2/2;allpriornegatives retained.

Latest 2026-09-14 18:46: fresh fullsampler gradient PGID1118712 exit0 at18:42:49;144new16step TRAIN paths+144exactautogradreplays+36primarycontrols,5184officialforwards+144localt0,1800checks/9plots and288CPU namedmeanchecks complete. Fresh277 has3negative full-vs-local individualdots (272306both,272307b0);no allpositive claim. Saved conflictreadback exit0 at18:44:05,76mean/hash checks/all9plots: prior178/197meannegative but4/8negative each;298fresh8/8positive,0/3predeclared conflicts replicate. New277/318meansnegative4/8and5/8,notrobust substitute. No extra seeds/relaxedcriteria/newauxiliary; unselected uniformterminalcompatibility draft removedwithout execution. Next storedAdam versus freshfullsampler CPUdiagnostic generator_fresh_sampler_adam_job296018 launcher1125779 started onsole296018/server34/step0: official fullparambinding/512moments andunchangedfullstate,all9phase+8global seed means+original objective components,0forward/gradient/step. Endpointlr0/per-unitlr historicaldirection only,not actualupdate512/513 orbenefit. Requireall9plots andindependent savedreadback,then exact executed-optimizer trajectory replay onlyifhistoricalphaseconflict,otherwise boundednoise-state sensitivity. Currentprimaryself18stilloverallnegative TRAIN8/9+reused2/2;allpriorrankremoval/physics/native/camera failures and retainedallocation preserved;no human gate/teacheriteration/weightsolver sweep/SMPclaim.

Latest 2026-09-14 18:41: self18 objective-vs-sampler64row audit PGID1116460 exit0 at18:37:52;204checks/fullstate/RNG/scheduler/zero controls/9plots and3200independent namedmeanreadback checks exit0 at18:39:23 complete. Overall raw total descentdot+.00168731,rank-.00323597;phase totals178-.000196338,197-.00304541,298-.00103815 conflict while277+.01066343. Rank helps277 andhurts others;priorrankremovalnegative retained,not a removal recommendation. New fresh8TRAIN fullsampler gradient replication PGID1118712 launched18:40:01 onsole296018/server34/step0:seeds272300..307 priorinteger search absent,disjointprimary32/replay8;144new official16step paths+144exactautogradreplays+36primarycontrols,5184officialforwards+144localt0,0updates/physics. Samefrozen64row objective mean compared toperrow terminalgradient,all9phase+8seed means saved. Require fullstate/allstep/outputs/RNG/scheduler/9plots+CPUmeanreadback then freshconflict readback. Prior178/197/298 replicated onlyif freshmean<-1e-8 and>=6of8seedmeansnegative;no seedextension. Codeforfreshsavedconflict readback implemented. Currentself18primaryoverallnegative8/9+2/2;retain alloldnegatives,allocation,no human gate/teacheriteration/weightsolver sweep/old513/SMPclaim.

Latest 2026-09-14 18:37: full self18 q+rank.25+replay.1 versus savedfinalsampler gradient implementation complete; new64row audit generator_self_objective_vs_sampler_job296018 PGID1116460 launched18:36:45 onsole296018/server34/step0. Full18realTRAIN/4trainingreplayseeds272230..233/all16times, separate3components andfullgradient, exactcomponent-sum controls/priorbatchedpointtol/zero-rank controls/fullstate/RNG/scheduler,all9phase versus fullsampler meanvectors andall4seedmeans required. Fullsampler reference4primaryseeds272090..093 explicitly differs; finite-sample mean comparison,notper-noise pairing oractualBF16/Adam. Finishall9plots/namedmeanCPUreadback then machinebranch: anycomponent/total conflict -> boundedfreshTRAINseed replication; allsupport -> storedAdamdirection check.0updates/newpaths/physics,newmodules,ortrainingbudget. Currentfullsampler72path887checks+160readback remains complete,primaryself18overallnegative8/9+2/2. Retainallocation andallnegativeevidence,no gate/teacheriteration/weightsolver sweep/old513/SMPclaim.

Latest 2026-09-14 18:34: full officialsampler gradient PGID1112141 exit0 at18:33:39;72TRAIN primarypath replays/all16states+epsilon+clean+prev+physicaloutputs exact withautograd,1152official forwards+72localt0,887checks/fullstate/RNG/scheduler/nineplots complete. Independent CPU namedmean readback160checks exit0 at18:34:29; all9phase+overall full/local meanvectors preserved. All72individual fullmodel dotspositive, but phaseaveraged158cos-.0958339 and197-.0431562 conflict; overallcos+.5308792,fullnorm.1179788/local.0085933,277cos.5557415. This proves faithful full16step gradient feasibility and local-vs-full objective mismatch,nottraining/Adambenefit. NEXT_OBJECTIVE_COMPARISON_PROTOCOL declares64full18TRAIN q+rank.25+aux.1 component/fullgradient rows onfirst4replayseeds/all16times versus savedfullsampler means; explicitlydifferent q/replay vsprimaryseedsets,FP32coupled diagnostic,require freshTRAINreplication before method admission. Code/execution ofnext64rowcomparison notyetdone. No newtraining/paths/physics,primaryself18stilloverallnegative TRAIN8/9,reused2/2;allpriornegatives preserved. Sole296018/server34/step0 retained,allcurrentchildren terminal;continue implementation/autochecks withno gate,teacheriteration/weightsolver sweep/old513/SMPclaim/release.

Latest 2026-09-14 18:32: paired-path transfer PGID1108939 exit0 at18:30:12;4224fullpointforwards+2zero/244checks+275independent savedreadback/all11plots complete. Bothown epsilon/clean exact; self-on-ranked18 finalclean MSE improvesall11, but ownstate feedback offsets benefit in9/11,net6/11four-seed finals regress. TRAIN197direct-.000169407/feedback+.000677136;261-.000404804/+.001809735;277-.000197091/+.002110890;318-.000274832/+.001264970. Descriptive oldpath-anchored algebra,notunique cause;277fourseed sign differsprimary32,unchanged overallnegative TRAIN8/9,reused2/2. Next full originalsampler autograd audit launched generator_full_sampler_gradient_job296018 launcher1112140 onsole296018/server34/step0:9realTRAIN/4existingprimaryseeds/2branches=72exactpaths,1152official16step forwards+72localt0,0updates/newpaths/physics. Require all16state/epsilon/clean/prev+physicaloutput exact withgrad,fullstate/RNG/scheduler/finiteness,full-vs-local gradients/nineplots/namedmeanreadback. Original policy.predict_action and scheduler called unchanged,no replacement sampler/model. This is feasibility/direction diagnosis,notnewtrainingbudget orbenefit;next compareexisting q+rank+aux onlyafterintegrity. No teacheriteration/weightsolver sweep/old513/physics/SMPclaim/manualgate/release.

Latest 2026-09-14 18:29: self18 frozenTRAIN replay pointprobe PGID1102834 exit0 at18:23:28;2304teacher exactpoints+4608learner+2zero,38checks+58independentreadback/all9plots complete. Correctepsilon+clean means improveall9,277epsilon .7084704to.6162851,clean .03192571to.02751393; no ownpath success inference. New paired fullranked18/self18 official16step recorder PGID1107361 exit0 at18:28:11:352exactfirst4primaryseed replays/all11correctwrong/bothbranches,5632forwards,114checks each+244independent savedreadback/all22plots inspected. InitialGaussian/times/targets and complete transition chains exact. All11ownpath time-means improve, but six final4seedmeans regress;277 subset .02061041to.02252421 differs from primary32improvement,do not substitute. Next fixedpath transfer preparation generator_self_path_transfer_prepare_job296018 launcher1108610 running:4224pointforwards+2zero, exact bothown epsilon/clean andself-on-ranked18,0newsamples/updates/physics. Complete all11plots/fullstates/RNG/scheduler/savedreadback before boundedTRAINnextdecision. Sole296018/server34/step0 retained;no teacheriteration/weightsolver sweep/old513/generatedphysics/SMPclaim/human gate.

Latest 2026-09-14 18:23: self18 all50time frozen diagnostic PGID1099685 exit0 at18:20:11;2fullmodels22cases4commonnoise/17checks+17independent savedreadback/48savedcomponents/all11plots complete,exactold22noise andall3oldmetric arrays preserved with18+4forwardchunks. Correctepsilon andclean means improveall11phases despite6generation MSEregressions;277epsilon.02573177to.02106131(ratio.818494),197ratio.739016,221.683460,261.865790,318.963759. Not a supervised-to-generation success equivalence. New frozen training-replay point probe generator_self_replay_training_probe_job296018 launched1102833 onsole296018/server34/step0: fullranked18teacher+fullself18learner,actual18TRAIN/same8seeds16times,2304exactteacher epsilon+clean replays then4608correct/wrong learnerpointforwards,2zero controls,0newpaths/gradients/updates/physics. Complete9TRAINplots/fullstate/RNG/scheduler/savedarrays before own-path orobjectiveinteraction diagnostic;no teacheriteration/weightorsolver sweep/old512extension/generatedphysics/SMPclaim/human gate/release. Self18 primary remains overallnegative TRAIN8/9,checks2/2,all11both32/32 but277accuracyfails and4oldpassingTRAINregress;centroid594checks/11plots meanbiasalonefails277 retained.

Latest 2026-09-14 18:19: self_replay01_gaps512 completeNEGATIVE TRAIN8/9,reused2/2,all11both32/32,277accuracyfails and4oldTRAINMSEregress. Saved3reference31availablephases/all5variants/32draw centroid readback exited0 at18:16:22,594checks/all155sourceNPZhashes/all288perphase decomposition identities/11plots inspected,0model forwards/updates/draws/physics.277 selfcommonbias.00813585,contrastbias.00681494,variance.00734684;meanbiasalonefails .5meanthreshold. Contrastgain.63695 vsoldgap.43654 andfit14.85586;off-axiscontrasterroralsoincreased,not ignoredprompt/varianceonly. Next fulloldranked18/self18 all22cases4commonnoise50time frozen diagnostic preparation tag generator_self_replay_denoising_prepare_job296018 running; new adapter preserves exactold22noise/all3arrays andold18+4forwardchunks. Require11plots/fullstate/scheduler/savedreadback before nextboundedconditioning/generated-state diagnosis; no newtrainingbudget/teacheriteration/solverorweightsweep/old513. Sole296018/server34/step0 retained;allprimaryGPUchildren terminal,no human gate/physics/SMPclaim.

Latest 2026-09-14 18:16: self_replay01_gaps512 COMPLETE overallNEGATIVE;pipelinePGID1081748 exit0 at18:12:50,both512/fullmodel+Adam/all11phase5condition32draws/all88panels/saved55checks/loss23checks/strict102checks(comparisonexit0 18:14:16). TRAIN8/9 versusold7/9,298 restored;reused218/2582/2 preserved;all11phases bothbranches32/32.277onlyphasefails: MSE.02229763381,meanbaseline ratio.884291>fixed.5 despiteallotherconditioncriteria pass. FourpreviouspassingTRAIN197/221/261/318 MSE regress;5/11allMSE improve,6worse. No threshold change orfullprediction/physics/SMPbenefit claim.277drawmeanMSE.01495079057 exceedsfixedthreshold.01260763127 evenwithoutvariance(.00734684375);meanbias67.05%. New readonly3reference/allavailablephase/all5variant/32draw centroid/common-vs-contrast+variance decomposition launched generator_self_replay_centroid_readback_job296018 launcher1096767 onsole296018/server34/step0. Require everyoriginalmetric/target/seed/normalizer/decomposition/sourcehash,11plots,savedarrays;0model forwards/updates/draws/physics. No automaticteacher/weight/solver sweep oroldbudgetextension. Allprior negatives andsource68position7/8 retained;keepallocation,nohuman gate.

Latest 2026-09-14 18:06: self_replay01_gaps512 both512full model+Adam endpoints COMPLETE,zerochild1083085 anddemo1084921 exit0;pipelinePGID1081748/Python1081754 now frozenall11phase evaluation onsole296018/server34/step0. No final scientific result yet. Separate retainedCPU generator_self_replay_loss_readback_job296018 launcher1088357 checksall512workspace losses/batches/auxiliary clocks/73728rows/4096caseexposures whileofficial evaluation continues. Completeall5conditions32draws/all88horizon panels/savedreadback/strict LATENT_REPLAY_COMPARISON. Samefull8319216/18cases144rows/releaseinit/oldnorm/.25rank+.1aux,only frozenreplayteacher refreshed;newzeroendpoint legitimatelydiffers. Preserveoldrankedgap7/9+2/2 andrankremoval2/9+1/2 negatives,source68position7/8 andallothernative/camera failures. No newtrainingbudget/old513/physics/SMPclaim/release/duplicate/human gate.

Latest 2026-09-14 18:01: full own-versus-old replay gradient PGID1075846 exited0 at17:56:00;128rows/416checks/2304exact ownpoint epsilon+clean/allfirst64oldphase losses+fullgradient group norms/fullstate/RNG/zero-controls/4plots complete. Overall old/candidatecos+.99125684; candidate descent oldobjective+.42937299,ownaux+.03229213;all8seedmeans pass18criteria,2/128individual ownauxdots negative,denoiser t6/t3 conflict remains. IndependentCPU meanreadback r1 exit0 17:58:03,64checks; initial readback parser dispatch failure beforemodel retained,0gradients/rerun. Separate self_replay01_gaps512 preparation10checks passes,only frozenaux teacher oldfull025 to ranked18endpoint; samefull8319216/releaseinit/18realTRAIN144rows/oldnorm/.25rank+.1auxBOTHarms/seed272084/both512/schedule/73728rows/4096percase/all11phasecriteria. CPU andactualH200BF16preflights81/81each; harmless old14 scope display text documented separately,actual18/144 and all checks correct. New generator_self_replay01_gaps512_job296018 PGID1081748 Python1081754 launched18:00:31 onsole296018/server34/step0,record/liveprocess verified; no endpoint yet. Bothfullmodel+Adam/all11phase5condition32draws/all88panels/loss-exposure readback/strict LATENT_REPLAY_COMPARISON required. Newzeroauxinputs differ,so no oldzeroendpoint equality;disabledaux oldbase+rank fullgradient/RNGexact. Never updateold513,replaycompletephases,release/duplicate/manual gate or claimgeneratedphysics/SMPbenefit. Rankremovalnegative TRAIN2/9+checks1/2 andrankedgap7/9+2/2 remain.

Latest 2026-09-14 17:54: current ranked18 TRAIN own recorder PGID1074076 exited0 at17:52:28;144new official16step paths+144exactfinalreplays+36originalprimary replays,402checks/fullstate/scheduler/recorderRNG/exactoldteacher target+initialGaussian/seeds/times,all9plots inspected. Same8TRAIN seeds paired witholdteacher,not32draw replacement. New221/261 ownfinalMSE.00084652/.00680399 vsoldteacher.04239231/.03715870;277.02151420 remains high,158/178/245 own8seedmeansworse. New full128row8seed16time old-versus-own correction gradient audit launched generator_current_gap_own_gradient_job296018 launcher1075845 onsole296018/server34/step0;2304own exactpoint epsilon/clean replays,first64oldcomponent/fullgradient reference checks,fullstate/RNG/zero-controls/allplots andindependent savedmean readback required. EvalFP32 coupledq16times diagnostic,not actualBF16/Adam. Fixed .1 replacement direction must descend oldfull objective and own correction foroverall+all8seedmeans before preparing anynewmatchedstage. Rankremoval complete negative103strictchecks andexactloggingrepair retained;no newtrainingbudget/update513/physics/SMPclaim/human gate/release.

Latest 2026-09-14 17:51: latent_only_gaps512 COMPLETE NEGATIVE;both512/all11phase5condition32draws/all88panels,TRAIN2/9(221/245),reusedchecks1/2(218),all11MSE worse versusrankedgaps512. Fullzero model+Adam/all512batches/losses exact. Logging defect fully resolved by separate full512demo replay PGID1070112 exit0 at17:50:05: fullmodel+Adam+initial/all512batches/nonbasefields exact,6checks. CPU andactualBF16 loggingfix78/78each; provenance-explicit repaired lossreadback26checks and strictcomparison103checks,comparisonexit0 at17:51:15. Originalrawlogs/failedreport retained;512extra replaycompute,0newuniqueupdates/no endpoint extension. Retain rankedgaps512 TRAIN7/9/checks2/2 foundation,stilloverallnegative. New current-model TRAIN own diffusion recorder generator_current_gap_train_replay_job296018 launched1074075 onsole296018/server34/step0: all9TRAIN18cases/same8replayseeds,144newpaths+144exactreplays+36originalprimarycontrols,official16step/full8319216,0updates/physics. Require all9plots/fullstate/scheduler/recorderRNG/exactoldteacher targets+initialnoise before predeclared128row old-versus-own replay correction gradient audit. No newtrainingbudget/weight sweep/generatedphysics/SMPclaim/manual gate/release/duplicate.

Latest 2026-09-14 17:48: latent_only_gaps512 original pipeline PGID1056184 exited0 at17:44:35; both512 fullmodel+Adam/all11phase5condition32draws/all88horizon panels complete. Rank removal primary NEGATIVE: TRAIN2/9(221/245),reusedchecks1/2(218),all11MSE worse than ranked gaps512(TRAIN7/9/checks2/2). Zero fullmodel+Adam/all512batches/losses exact. Raw demo base metadata defect retained in LOSS_RECORDING_INCIDENT and failed readback; actual objective currentbase+.1aux unaffected by copied logging metadata, pending exact replay proof. Logging-only fix CPU78/78 andactualH200BF16 PGID1069543 exit0 at17:47:36 78/78 pass. Independent demo_log_replay512 launched tag generator_latent_only_demo_log_replay_job296018, launcher1070111 onsole296018/server34/step0. Require fulloriginal model+Adam/initial/all512batches/nonbase records exact, corrected arithmetic;512additional executed replaysteps,0newuniqueupdates,no original endpoint extension. Then provenance-explicit trainingreadback and strict rank-ablation comparison; preserve original logs/samples/failed report and ranked gaps512 foundation. No generatedphysics/SMP benefit or human gate; retain allocation.

Latest 2026-09-14 17:44: latent_only_gaps512 both512fullmodel+Adam complete;zero fullmodel/Adam/all512batch+loss exact. Primary all11phase eval continues onPGID1056184;early158/178/197and258/261fail,218/221/245pass,56panels inspected sofar. Actualreadback CPUchild exited1: rank0demo base metadata stale afterfirstrow because last_paired_loss reused frompreviousaux record; actualreturn usescurrentbase+.1currentaux. Rawlogs and TRAINING_LOSS_READBACK_FAILED.json preserved; no finalstrict scientificclaim untilrepair. Fixed logging branch torequire nonzerorank beforeusing pairedmetadata, added5repeatedcombined arithmeticpreflightchecks. Separate fullCPU loggingfix preflight running;afteroriginalevalfinish executeactualBF16 loggingfix preflight then separate demo_log_replay512 fromsamefullreleaseinit. Require exact original fullmodel+Adam/all512batches/totals/auxiliary/nonbasefields; use freshbase records with explicitprovenance. This is512additional executed replaysteps,0newuniqueupdates,not oldendpoint extension;zero/evaluation notrerun. Alloriginalresults retained,sole296018/server34/step0 preserved,no manual gate or generatedphysics/SMPclaim.

Latest 2026-09-14 17:33: fullcomponent Adam-direction r1 PGID1051233 exited0 at17:28:23;397checks/all64reference stats exact/287official optimizerbindings/fullmodel+Adam unchanged/16panels andsavedmeanvectors inspected. Endpointlr0; unit-lr historicalmoment fullmean descentdots base43.1329,rank13.5169,aux9.73742,all4seedmeanspositive. No actualAdam harm or update513 claim. Firstmetadata ListConfig JSON failure occurred before gradients andisretained. One separate ranking-presence ablation prepared: matched_generator_branch_latent_only_gaps512,ONLY rank.25to0; full8319216/releaseinit/18realTRAIN/144batch/oldnorm/oldfull025teacher/.1auxBOTHarms/seed272084/both512updates/schedule/all11phasecriteria fixed,lesscompute withoutwrong-rankforwards explicitlynot equalFLOPs. CPU andactualH200BF16 preflights73/73each pass botharms initial+learned base/fullgradient/RNG/forward/auxiliary checks,including exactoldranked differentiablebase. Pipeline generator_latent_only_gaps512_job296018 PGID1056184 launched17:32:50 onsole296018/server34/step0. Zeroarmmustpass exactoldfullmodel AND fullAdam/all512loss+batchcheck beforedemoarm; thenall11phase5condition32draws/88panels/savedreadbacks/newstrict RANK_ABLATION_COMPARISON. No newendpointyet; fullpreviousgaps512strictnegative TRAIN7/9,reusedchecks2/2 retained. Neverextendoldbudgets/release/duplicate/askhumanauthorization orclaimnative/generatedphysics/SMPbenefit.

Latest 2026-09-14 17:27: full18case three-loss gradient audit PGID1046730 exited0 at17:23:40;all64rows/330checks/12panels inspected,old phase scalars exact andthree weighted fullgradient sum validated. Mean full base/rank cosine-.040585,base/aux+.581483;denoiserbase/rank-.635741 versus targetencoder+.507023. All4seedmeans/fullmean directions descendallthree;individualfull-direction adverse base0/64,rank32/64,aux8/64. No ranking-removal benefit or actualAdam harm claim. Next stored fullAdam512 direction probe launched17:26:19 PGID1050488,tag generator_gap_adam_direction_job296018 onsole296018/server34/step0. Official parameter grouping andcomplete ID/name/shape/step/moment binding;bias-corrected frozenmoment+variance plusendpointdecay unit-lr direction,64samegradient rows/oldstatistics exact/fullstate/plots. Nooptimizer.step,update513,actualstep512reconstruction or newgradient inserted; meanfullvectors saved forreuse. Finishstate/plot evidence before next bounded objective ablation. Strictgaps512negative TRAIN7/9,reusedchecks2/2 andallprior negatives retained; no release/duplicate/human gate/generatedphysics/SMPclaim.

Latest 2026-09-14 17:20: gaps512 strictnegative retained TRAIN7/9/reusedchecks2/2. FullTRAIN teacher/old/new replay probe PGID1038181 exited0 at17:15:36;2304teacher point epsilon/clean exact,48checks/81savedreadback/nineplots pass,277/298auxepsilon ratios1.097636/1.123686. Full current18case old14/new4 objective gradient PGID1041944 exited0 at17:18:35;64rows/202checks/fourplots/fullstate/RNG/zero-controls/weightedfullgradient identity pass. Mean full cosine+.736203,3/64negative subgroup dots,but64/64combined directions andall4seedmeans descendboth; no simple blanket data-conflict explanation. Scalar rank value dominates this coupled16time FP32 diagnostic,not parametergradient/Adam harm proof. DECISION and NEXT_COMPONENT_GRADIENT_PROTOCOL.json declarefullcurrent18case3losscomponent gradients,first4TRAINseeds/all16times64rows,exactold phase-loss/totalgradient/fullstate controls before another budget; code/execution not yetstarted. Sole296018/server34/step0 retained,allcurrent children normalterminal. All11phase88panels/90strictchecks complete,6of7oldTRAIN MSEregress,277/298fail; nooldupdate513/native/generatedphysics/SMPbenefit or human gates.

Latest 2026-09-14 17:14: gaps512 complete strict NEGATIVE retained: TRAIN7/9,reusedchecks2/2,6of7oldTRAIN MSEregress. Full old/new all22case/4common-noise/all50time probe PGID1025274 exited0 at17:10:53;17checks,exactold18noise plus all3old denoising arrays,44saved-generation checks and all11curves inspected. New correct epsilon improves218/221/258/261 and298/318, but worsens158/178/197/245/277;277ratio1.1299536,298ratio.8074693. Thus no single monotone supervised/generation explanation.158/178 newcontact weightederror.001771/.000861 with increased drawvariance;277/298 joint/velocity regress and branch-response projections.437/.693. Next full teacher+old14+new18 exact TRAIN replay probe PGID1038181 launched17:13:49,tag generator_gap_training_replay_job296018 on sole296018/server34/step0;2304teacher exactpoint replays then9216learnedcorrect/wrong pointforwards,all18TRAIN/8seeds/16times,fullstates/RNG/scheduler/zero-controls/nineplots,0newpaths/updates/physics. Complete andinspect before choosing generated-path versus objective-interaction diagnostic. Official Zero-WAM online recheck17:11 still docs/license/readme,code/model/data release pending; reuse ideas with fullofficialSUGAR+SMP remains active,notauthorimplementation. Neverextend512/release/duplicate/usehuman gates or claimnative/generatedphysics/SMPbenefit.

Latest 2026-09-14 17:09: gaps512 full endpoint COMPLETE, overall strict prediction objective NEGATIVE. Recovery PGID1015484 exited0 at17:03:52 on sole296018/server34/step0;100cross-job replay calls exact,47boundary checks and59preserved-file hashes pass,only6missing phases sampled,0trainingupdates/physics. Both512 fullmodel+Adam,all11phase5condition32draws,saved readback,88horizon panels inspected and90strict comparison checks pass. TRAIN7/9 (277/298fail),reused218/258checks2/2 now pass withMSE.001422627945/.008223874494 andboth[32,32],ratios.0371178/.248339 tooldlatent. Old7TRAIN sixMSE regress;277.0268072784[22,32],298.0154765733[32,32]; bothpreviouslypass nowfail. New221/261pass. Do not remove no-regression criteria or infer native/generatedphysics/SMP benefit. Next bounded zero-update full2model/22case/4common-noise/all50time diagnostic preparing; preserveexactold18noise and forward batch replay,includeall11plots,old221/261primarysamples unavailable explicitly. Initialcomparison import incident and CPUdiagnostic path-string assertion retained; latter corrected to exact common actual action/obs tensors rather than directory identity. No new training budget or oldupdate513. Retain allocation and all prior physical237/358,native,position7of8,camera negatives; no human gates.

Latest 2026-09-14 17:00: matched_generator_branch_latent_replay01_gaps512 both full8319216model arms COMPLETE512updates/fullAdam/frozenoldnormalizer; actual loss/exposure readback21checks passes all2304seed/time/case points32exposures,73728rows/arm. Original pipeline PGID300245 interrupted after5complete phases158/178/197/218/221,all pass per-phase criteria;218MSE.001422627945 [32,32],new221MSE.000874994439.158/178MSE regress versusoldlatentfit despitepassing; no all-stage or fullprediction conclusion yet. Job295673 CANCELLED by2059 at15:54:40,step0 exit0:9 at15:54:42; tmuxgone anduserqueueempty verified,not agent release. Sole replacement296018/server34/explicitstep0 acquired8CPU64GiB/day,tmuxcuriosity_gap_eval_recovery_20260914 pane%0; pending node constraint removed within samejob,allgpu nodes H200. Loginfilesystem reads recovered. Frozen recovery CPU77checks verifies fullmodel+Adam512,all25savedvariant arrays/metrics/actualtargets and59immutable file hashes. InitialCPU audit foundone tiny221zero horizon discrepancy fromalgebraic offset cancellation; originalFP32 normalize-before-subtract fixes it with unchanged1e-5/1e-12tolerances,incidentretained. New generator_gap_eval_recovery_job296018 PGID1015484 runs100exact first2seed boundary replays for all5conditions/5completedphases/2branches on replacementH200,fullstate/scheduler/hashes; only afterpass,archiveempty interrupted245folder andcomplete missing245/258/261/277/298/318 throughoriginalofficial evaluate_branch. All5completed arrays/plots remain byte-exact,0trainingupdates/physics; unretained in-flight245drawcountunknown and extra compute explicitlyrecorded. Finish all11phase savedreadback/88horizon panels/strictcomparison; do not reruntraining or complete phases, extend512,release/duplicate allocation, claimnative/generalization/generatedphysics/SMPbenefit or addhuman gates. Actualphysical corpus remains11/13 with237/358failures,broadernative/position7of8/camera negatives remain.


Latest 2026-09-14 15:46: generator_actual_branch_coverage_gaps2 COMPLETE: PGID274154 exited0 at15:28:35, all6new400steprollouts/2400controls, fullfrozen607/commonworld/independent-following/export and all28physical/reference/XYZ panels pass inspection. Required11/11 available,cumulative11/13 retains237/358failures; prior9/11 unchanged. New18case model protocol40checks preserves allold18branch arrays/provenance includingreused218/258. Sameoldfull025 teacher gap replay PGID285751 exited0 at15:34:45:32newpaths+32exactreplays,112oldpaths reused,28oldprimarysample controls exact,all106fullstate/scheduler/data checks and9TRAINplots inspected. No newphysics/updates inreplay. Dataset/factory/preflight/exposure/comparison glue now supports18cases without new model modules; full144row CPU PGID292139 exited0 at15:43:53 and actualH200BF16 PGID293147 exited0 at15:45:08 both74/74checks,including botharms initial+learned old112subset full replay loss/gradient/RNG exact. New matched_generator_branch_latent_replay01_gaps512 pipeline generator_gap_latent512_job295673 PGID300245 launched onsole295673/server35/step0; live recorded zero_context child. Both512updates then all11phase5condition32draws/fullmodel+Adam/savedreadback/88horizon panels and strict comparison required; no endpoint yet. Same0.25rank+0.1latent objective/release initialization/oldnormalizer/seed272084/AdamW/schedule; batch112to144 and rows57344to73728 explicitlychanged,4096exposures/case,not equalFLOPs or identical old q-noise trajectories. Old latent_replay01512 remains TRAIN7/7 versusfull0255/7 butreusedchecks0/2 overallnegative; all broader native/position7of8/camera negatives remain. Sharedfilesystem errno24 onlogin persists; retainedcompute reads/writes and exactmonitoring work. Never restart on observation failure, release/duplicate allocation, extend old512budgets, claim generatedphysics/SMPbenefit or use manual authorization. Continue exactpipeline through all evidence and automatedscientific next step.


Latest 2026-09-14 15:28: full latent_replay01512 completed both512/full model+Adam/all9phase5condition32draws/72panels/85strict checks: TRAIN7/7 versus oldfull0255/7, reused218/258checks0/2, overall NEGATIVE. Retain new TRAIN fitting reference only. Frozen all50time denoising13checks/all9plots and full joint-condition coverage54checks/all18queries complete; correct epsilon improves all9phases, learned joint neighborhoods still favor original for check alternates. Two-largest-TRAIN-gap static screen first failed cross-host orientation floating parity about1e-16rad; separate r1 passes42checks at1e-12numeric tolerance, exact physical decisions unchanged, selecting221/261 by clocks/static compatibility. New generator_actual_branch_coverage_gaps2 preparation PGID270393 exited0 15:14:04, all9old formatter groups exact. Actual collector generator_gap_coverage_job295673 PGID274154 Python274160 launched15:19:28 onsole295673/server35/step0; exactly6new400stepattempts,max2400controls,0updates,27oldrollouts reused.221all3x400 and fullstate/commonworld/independent-following/export pass; all14physical/reference/XYZ panels inspected, transient box-reference distance about0.21m thenrecovers.261 still requires complete endpoint/visual inspection. Retain237/358failures, current9/11 prior corpus unchanged, at most11/13 after newstage. Login sharedfilesystem errno24 remains; recovered monitoring by backgrounding only terminal launcher supervisor; detached collection PGID274154 and live PhysX child were not suspended or restarted. All observations/edits now use retained compute. Next complete both phases and all physical evidence; only then prepare separate18case full Generator coverage experiment with same0.25rank+0.1latent objective, sameoldfull025teacher/normalizer, exactold14replay plusnew4case replay, explicitly144batch/exposures/all11phase evaluation. No new Generator yet, no endpoint extension/generatedphysics/SMPclaim. Never release or duplicate allocation.


2026-09-14 14:46：`matched_generator_branch_latent_replay01512`已完整结束，整体判据仍为负。
两组各512更新、完整模型与Adam、九阶段／五条件／32draw、全部72面板、保存回读和
85项严格比较均完成，pipeline PGID179256于14:38:18退出0。七个TRAIN阶段首次全部
通过（原5/7→7/7），七阶段误差都降低；两个复用检查仍0/2。保留新终点作为TRAIN拟合
基准，不把这个结果称为阶段泛化、生成物理执行或SMP收益。

| 阶段 | 原full025→回放辅助 MSE | 当前原／替代选对 | 当前判据 |
| --- | --- | --- | --- |
| 158 TRAIN | 0.000558→0.000465 | 32／32 | 通过 |
| 178 TRAIN | 0.000378→0.000307 | 32／32 | 通过 |
| 197 TRAIN | 0.003062→0.001794 | 32／32 | 通过 |
| 218 复用检查 | 0.039437→0.038327 | 32／0 | 失败 |
| 245 TRAIN | 0.003010→0.002781 | 32／32 | 通过 |
| 258 复用检查 | 0.030865→0.033115 | 32／32 | 失败 |
| 277 TRAIN | 0.023596→0.010784 | 32／32 | 通过 |
| 298 TRAIN | 0.017171→0.008090 | 32／32 | 通过 |
| 318 TRAIN | 0.007766→0.003474 | 32／32 | 通过 |

[完整严格比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_latent_replay01512/frozen_evaluation/LATENT_REPLAY_COMPARISON.json)。
两组相同辅助时刻和实际batch记录确认，每个1792种seed／时刻／真实案例组合恰32次。
登录节点一度反复仅读出372行示范组日志，首次日志回读检查因此失败；原计算节点读取
完整512行，完整回读也在原节点执行并通过，未改写日志或重复训练。
[训练损失与实际暴露回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_latent_replay01512/TRAINING_LOSS_READBACK.json)。

随后全冻结共同噪声检查PGID190166于14:42:59退出0，18实际案例／四噪声／全部50时刻、
13项检查和九曲线通过，原full025三种误差数组精确复现。九阶段监督epsilon均值都降低，
但218／258仍高达0.21358／0.30956。保存32draw分解显示，277／298方差分别从
0.009805／0.009105降到0.002156／0.001918，均值误差也下降。检查218／258误差中
98%以上是均值偏差，258方差虽更低但均值误差更大；分解仅是描述证据，不是替代预测。
[冻结去噪比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_latent_replay01512/frozen_evaluation/rank_component_audit/DENOISING_COMPARISON.json)。

下一只读CPU条件覆盖检查已启动：比较完整full025与新回放终点的条件编码，14个当前
拟合案例、3897个此前native候选和4125个全部可用native候选分开统计。后两者不是当前
TRAIN，含源96／90的可用子集也不是独立验证；邻居由输入距离选择、标签只作事后描述。
完成全状态、全部自邻居及四个复用检查查询后，再选择有界覆盖或表示假设。0更新、
0采样、0物理；所有GPU子进程正常结束，唯一295673/server35/显式step0保留。

2026-09-14 14:28：实际回放训练连接已完成，CPU与H200 BF16各69项预检全部通过。
原112行的obs／action／配对标签逐项精确，xt／time／seed仅作为额外TRAIN字段；
新目标调用原完整监督加排序后，再加入0.1真实目标纠正，恢复原下一调用的随机数状态。
两组均执行辅助项，无新增模型模块，完整8,319,216参数保留。初态与已学习终点的原loss、
全部梯度、前向、噪声时刻、RNG、禁用辅助项等值全部通过；所有16回放时刻与50加噪
时刻的完整112行反向、循环计数、冻结全状态也通过。
[CPU预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_latent_replay01512/GENERATED_CPU_PREFLIGHT.json)、
[实际BF16预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_latent_replay01512/GENERATED_BF16_PREFLIGHT.json)。

训练pipeline已于14:26:52启动，PGID179256，Python179264，零条件子进程179571
于14:27现场确认运行；tag `generator_latent_replay01512_job295673`，同唯一
295673/server35/显式step0。固定两组各512更新，随后自动完成九阶段／五条件／32draw，
保存完整模型与Adam、57,344条辅助输入暴露与时刻记录、全部72面板和指标回读。
`compare_generator_latent_replay.py`已实现两组旧／新完整比较：初态／batch／预算／统计
精确，原初态采样精确；新零组接受辅助监督，因此不要求终点或采样等同旧零组。
没有新终点或收益结论，不延长旧预算，也无生成物理／SMP证据；所有负结果和计算分配保留。

2026-09-14 14:17：新八seed生成状态梯度检查完整结束，PGID166891于14:15:22退出0。
128组／526项检查全部通过，完整模型、normalizer、RNG和零条件对照正确；四面板与
全部七阶段指标已查看。全模型两方向均值cosine +0.82703，单独128组仍有23组负点积。
按预先固定0.1计算，八个seed均值全部同时使原目标与辅助目标一阶下降，整体点积分别
+0.87709／+2.29346，机器判据全部通过。这仅是完整已学习终点的FP32原始梯度证据，
不是实际Adam／BF16训练效果，也不改变此前所有生成预测负结果。
[完整结果](../experiments/demo_following/demo_future_smp_v1/generator_train_diffusion_replay8/generated_state_gradient_audit/RESULT.json)、
[固定判据决策](../experiments/demo_following/demo_future_smp_v1/generator_train_diffusion_replay8/generated_state_gradient_audit/DECISION.json)。

独立`matched_generator_branch_latent_replay01512`协议已准备，13项原设置逐项一致：
完整8,319,216参数、发布初态、旧normalizer、14个原真实TRAIN案例、112行batch、
seed272084、原50时刻加噪、0.25softplus排序、原AdamW／调度、每组固定512更新。
新增项为0.1生成状态epsilon监督，两组使用完全相同的冻结full025 TRAIN回放和真实目标。
每个原重复行按index//14绑定八seed，TRAIN前向计数mod16轮转原16采样时刻，
512更新恰覆盖每个seed／time／case组合32次，总57,344辅助行、每真实案例4,096次。
不通过采样求导，不改原基础噪声RNG，也不把扩散潜变量冒充物理状态。
教师已训练512更新与112次新增路径采样是额外计算，不宣称同FLOPs比较。

新零条件组的辅助梯度非零，所以预期它的模型／Adam会改变；必须让两组接受同样监督。
要求禁用辅助项时旧完整loss／梯度／RNG精确复现，并检查新两组真实batch、预算、
初态、normalizer一致，保存完整模型与Adam。不能沿用旧排序实验的零组终点等值要求。
[新协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_latent_replay01512/PROTOCOL.json)。

下一步继续实现实际回放dataset／原完整模型辅助loss与训练连接，完成初态和已学习终点的
CPU及实际H200 BF16全模型预检，然后自动完成两组512更新、九阶段／五条件／32draw、
全部72面板、保存指标与原full025比较。上述训练适配和实际预检尚未执行，不把协议当结果。
所有已记录子进程正常结束，唯一295673/server35/显式step0保留；不扩展旧预算，不产生
生成物理或SMP收益声明。首次参数入口失败继续保留，修复未覆盖或重复任何成功计算。

2026-09-14 14:14：新TRAIN扩散轨迹采集已于14:07:10完整结束，PGID165048退出0。
112条新路径、112次精确复现、28次旧样本重放全部完成；197项检查、七曲线、
完整模型不变与保存指标精确回读通过。14个原真实TRAIN案例、八个新seed、16个原官方
时刻，输入形状8×16×14×8×36；没有218／258标签或主要评估seed进入该数据。
[采集结果](../experiments/demo_following/demo_future_smp_v1/generator_train_diffusion_replay8/RESULT.json)。

首次新梯度调用因bootstrap直接调用main而未执行__main__下的参数解析，在模型计算前
遇到旧输出目录并退出1；没有覆盖旧结果，0新增梯度或更新。失败记录保留，解析已移入main，
实际bootstrap的--help已验证。独立r1调用PGID166891正在同唯一295673/server35/step0执行
128组完整新TRAIN梯度检查，tag `generator_fresh_generated_gradient_r1_job295673`。

在读取新梯度终点前固定辅助权重0.1，依据旧四seed均值的辅助梯度范数约为原目标的33.18%；
新判据要求平均联合方向同时下降、全部八seed均值保持原目标下降、至少六个也下降生成状态目标。
没有系数扫描，不把原始梯度等同Adam效果。完整检查和四面板后，由
`decide_generator_latent_replay`自动产生机器决策；通过才准备独立每组512更新协议。
审查训练接口时明确：原始生成状态监督在零条件下仍有梯度，所以新匹配的两组必须接受
相同回放／真实标签／0.1辅助损失。原提案中零组跳过辅助项的表述已被保存的
MATCHED_CONTROL_AMENDMENT.json取代，旧协议保留溯源；不能再要求新增零组终点复现旧模型。
应要求禁用辅助项时旧完整loss／梯度／RNG精确复现，以及新两组数据／预算完全匹配。
当前无新训练协议实例、训练改动或更新；旧终点和全部负结果保留，0生成物理或SMP收益。

2026-09-14 14:07：生成状态TRAIN梯度诊断已完整结束，PGID159048于14:01:26退出0。
64组／270项检查通过，逐点epsilon与clean精确复现，全模型／normalizer／RNG不变、
零条件梯度精确为0；完整四面板已查看。全模型平均监督加排序方向与生成状态修正方向
cosine +0.72077，四个seed均值中三个为正，但64组中28组方向冲突。
TRAIN277／298生成状态epsilon误差均值0.75927／1.00487，对照标准加噪为0.04210／0.04421。
这是均匀16推理时刻、FP32 eval模式的原始梯度证据，不是全50训练时刻、实际Adam更新或收益。
[完整梯度结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/own_reverse_process/path_feedback_readback/generated_state_gradient_audit/RESULT.json)。

下一步已启动独立TRAIN扩散状态采集：14:06:20，PGID165048，
`generator_train_diffusion_replay8_job295673`，同唯一295673/server35/显式step0。
完整full025冻结终点、原16步官方采样，七阶段14案例、八个新seed272230..237，
与全部32个主要评估seed分离。112条新路径、112次新路径精确复现和28次旧样本重放；
只记录扩散潜变量，监督目标仍为原实际未来，不把它们当成新增物理轨迹。
158／178已通过，完成其余阶段、保存回读、全状态和七曲线后，自动执行全部八seed／
16时刻的128组梯度检查，再确定一个有界的辅助损失实验；未声明新系数或训练预算。
218／258检查标签不进入梯度或回放数据，不扩展旧512预算，也无生成物理或SMP收益。

2026-09-14 13:48：hinge自身反向记录PGID131414于13:34:33退出0，144次已有样本
精确重放、65项检查和九曲线完成。随后路径分解PGID146858于13:42:59退出0；
4,608次保存转换用原官方DDPMScheduler.step精确复现，两模型每次转换前CUDA RNG
精确一致，136项检查和全部九曲线通过，0去噪前向／0新路径／0优化／0物理。

对四个诊断seed的正确路径，277最终MSE：原0.0223962，hinge在原xt上0.0222957，
hinge自身路径0.0267143；直接模型差异−0.0001005，状态变化项+0.0044186。
298对应0.0271570／0.0268241／0.0289304，直接−0.0003328，状态项+0.0021063。
这是以原模型状态为基准的精确代数分解，不是唯一因果解释；四seed与完整32draw有差别，
例如158／178的四seed误差下降却不改变其32draw主结果回退。不能挑选seed／中间输出或部署混合模型。
[完整路径分解](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/own_reverse_process/path_feedback_readback/RESULT.json)。

下一完整模型TRAIN生成状态梯度诊断的实际输入已准备：原full025七阶段／14案例、
原前四推理seed、全部16推理时刻，保存xt形状4×16×14×8×36、原真实目标14×8×36，
以及同次初始Gaussian；配对初始噪声／形状／有限值检查通过，0新模型前向。
[下一64组梯度协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/own_reverse_process/path_feedback_readback/NEXT_GRADIENT_PROTOCOL.json)。
检查保留完整full025终点在实际生成扩散状态上纠正未来误差的梯度，与原监督加排序方向
是否兼容。xt为扩散潜变量，不是虚构物理状态；只使用原真实TRAIN未来和因果观测。
原官方加噪对照使用同次初始Gaussian，不把它称为后续步噪声或相同输入分布。
要求逐点epsilon精确重放、全模型／normalizer／实际案例／有限值／转换恒等式和零条件检查；
新梯度代码与执行尚未完成，不把已准备数组或协议称为结果，也不据局部方向直接声称训练收益。

当前所有已记录GPU子进程均正常结束，唯一295673/server35/显式step0保持，继续实现上述
有界梯度诊断。无新训练预算、无update513、无系数扫描、无生成物理或独立SMP收益。

2026-09-14 13:34：同一生成状态的模型迁移诊断完整结束，PGID124737于13:25:11退出0。
103项检查通过，144次hinge自身最终样本精确重放、2,304个固定xt前向和全部九曲线完成，
合计4,608次完整去噪前向，完整状态／RNG／目标与原参考绑定通过。保存指标按原两级
均值顺序精确复算；首次合并轴求均值的严格等值检查失败，最大float64舍入差2.78e-17，
已在回读中记录，未重跑模型或修改数组。在原full025生成状态上，hinge的九阶段平均
clean误差均更低；TRAIN277／298分别为原0.85349／0.81682，最终一步0.99551／0.98774。
因此不能把监督改善与生成退化直接归因为“生成状态上去噪更差”；还需比较各自路径变化。
[同状态完整结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/recorded_path_transfer/RESULT.json)、
[保存回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/recorded_path_transfer/SAVED_READBACK.json)。

下一有界hinge自身反向轨迹记录已于13:33:49启动，PID／PGID131414，
tag generator_hinge_own_reverse_job295673，同唯一295673/server35/显式step0保持。
九阶段／正确错配／两分支／原前四seed，共144次已有最终样本精确重放，完整官方16步
predict_action和scheduler.step不变；记录xt／epsilon／clean／prev_sample，不做优化。
完成后使用已保存full025路径和hinge在full025状态上的前向，检查共同初始噪声／时刻／
逐步随机项，再分解模型改动的直接预测变化与其引起的后续状态变化。全部九阶段／
完整状态／保存样本／曲线必须通过；四seed诊断不能替代32draw主要结果，也不产生新物理标签。
[自身路径协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/own_reverse_process/PROTOCOL.json)。

2026-09-14 13:24：matched_generator_branch_paired_hinge025512完整结束，为负结果。
PGID109991于13:12:04退出0，两组各512更新、九阶段／五条件／32draw、全部72面板、
保存指标和58项严格前驱比较均完成。零示范完整模型与完整Adam精确复现原full025；
实际batch／预算／初态／统计和条件交换均通过。TRAIN仍5/7、复用检查0/2，六个TRAIN
误差更高，七个全部阶段误差更高。hinge不作为下一研究基础，保留原full-rank0.25基准。

| 阶段 | 原softplus→hinge MSE | 原／替代选对（各32次） | hinge全判据 |
| --- | --- | --- | --- |
| 158 TRAIN | 0.000558→0.000702 | 32／32 | 通过 |
| 178 TRAIN | 0.000378→0.000637 | 32／32 | 通过 |
| 197 TRAIN | 0.003062→0.002536 | 31／32 | 通过 |
| 218 已使用检查 | 0.039437→0.040902 | 32／0 | 失败 |
| 245 TRAIN | 0.003010→0.003067 | 32／32 | 通过 |
| 258 已使用检查 | 0.030865→0.030842 | 32／31 | 失败 |
| 277 TRAIN | 0.023596→0.026006 | 32／32 | 失败 |
| 298 TRAIN | 0.017171→0.019836 | 32／32 | 失败 |
| 318 TRAIN | 0.007766→0.009640 | 32／32 | 通过 |

[完整终点](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/RESULT.json)、
[严格比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/PAIRED_COMPARISON.json)。

保存分解全部通过：277／298均值偏差和采样方差都上升，沿真实分支方向的条件响应
投影从0.73471／0.87386降为0.64474／0.78198。四个旧通过阶段误差回退；
局部梯度冲突减轻不能推断训练收益。两个完整终点的相同18案例／四保存噪声／全50时刻
监督去噪诊断PGID120603于13:17:45退出0，全部完整状态、输入、噪声、scheduler和九曲线
检查通过；原full025全部三类误差数组精确复现旧诊断。hinge九阶段正确epsilon误差
均降低，277／298为原0.68964／0.55461倍，却生成更差。再次区分带真实目标的加噪
监督输入与模型自己生成的状态；还不能把差异直接判定为分布偏移的唯一原因。
[共同噪声诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/rank_component_audit/RESULT.json)、
[全时刻比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/frozen_evaluation/rank_component_audit/DENOISING_COMPARISON.json)。

下一有界诊断正在准备：完整hinge模型读取原full025生成途中保存的同一xt，比较原已精确
审计的full025预测。九阶段／前四seed／两分支／16时刻／正确错配，共2,304点前向；
另先精确复现hinge自身144个保存最终样本（2,304去噪前向），合计4,608次完整去噪前向。
相同模型、normalizer和目标绑定必须通过；固定点计算要求RNG与全状态不变。
这是在另一完整模型访问的状态上诊断，不是新的物理标签、部署轨迹或新增独立样本。
保持唯一295673/server35/显式step0，不追加旧512、不改系数／间隔、不释放配额。
当前仍0新PhysX、无生成闭环或独立SMP收益。

2026-09-14 13:02：固定hinge梯度诊断完整通过，PGID100652于12:49:54退出0。
两个完整模型状态×四保存噪声×八时刻共64组，79项检查通过，全部逐阶段误差与梯度已检查。
旧full025终点32组基础损失和全梯度范数精确复现；初态与终点各4/4seed均值满足预设
base下降／hinge不增判据。终点全模型base/rank内积−0.05621（旧softplus−0.67645），
去噪器组合方向的base内积为+0.35727，局部下降。只是eval模式原始梯度，不是Adam收益。

独立matched_generator_branch_paired_hinge025512已准备并启动。保持完整8,319,216参数、
发布初始化、原14真实TRAIN案例／112batch、冻结normalizer、seed272084、IID噪声、
goalzero／完整历史、0.25权重与0.1间隔，只把softplus改为hinge。两组仍各512更新，
不续训旧终点；两组全部九阶段／五条件／32draw／72面板与旧full025严格比较自动串行。
原TRAIN全部通过、复用检查全部通过和旧通过阶段MSE不回退判据不变。
[新协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/PROTOCOL.json)。

实际CPU与H200 BF16预检均39/39通过，包括发布初态和学习终点的基础全梯度／全部
前向／噪声／后续RNG精确、达到间隔导数严格为零、未达到间隔导数仅浮点舍入、
完整状态不变、全50时刻完整反向有限；零条件全部梯度精确复现。CPU／GPU用各自
随机序列，数值不用于精度优劣比较。BF16预检PGID108005于13:00:47退出0。
[CPU预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/PAIRED_CPU_PREFLIGHT.json)、
[BF16预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_hinge025512/PAIRED_PREFLIGHT.json)。

当前管线tag generator_hinge025512_job295673于13:01:31启动，PID／PGID109991，
唯一295673/server35/显式step0保持。训练／冻结终点尚未完成，不以预检或局部梯度
声称生成改善；不追加512预算、不释放配额、不并发另一物理或训练流程。
全部旧负结果／实际237与358失败保持，0新PhysX，尚无生成闭环或独立SMP收益。

2026-09-14 12:50：固定xt条件诊断于12:44:05退出0（PGID96843）。完整模型2,304次
前向、九阶段、四原seed、两分支、16时刻，全部正确epsilon／官方clean重建精确，
完整状态／RNG／因果字段检查通过，九曲线全部查看，保存指标和条件胜出计数复算一致。
失败TRAIN277／298的两分支各64/64个seed×time组合都因正确示范而降低误差；最终
正确／错配误差比0.61792／0.49474，但准确性仍失败。因此当前不是简单的示范完全失效，
也不能用条件敏感性替代生成质量。[固定xt完整结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/fixed_xt_condition/RESULT.json)。

新保存损失回读：原full025共同噪声诊断14TRAIN案例×4seed×50时刻=2,800组，
91.82%已满足原0.1排序间隔；这些组仍贡献标量softplus导数总和的83.24%。
这是标量误差导数，不是参数梯度占比或实际Adam受损证据。已使用检查只报告，不参与选择。
[排序间隔回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/rank_margin_readback/RESULT.json)。

下一完整TRAIN梯度诊断已于12:49:09启动，PID／PGID100652，
tag generator_hinge_gradient_job295673，同唯一295673/server35/显式step0保留。
完整发布初态与保留full025终点，同14案例、原四噪声seed、八时刻共64组；
仅将排序softplus替换为固定0.1间隔的hinge，权重仍0.25，基础官方IID去噪不变。
无优化器步、新采样或物理步。要求完整状态、零条件梯度与有限值；两个初态／终点的
平均组合方向均使base下降、hinge不增，每状态至少3/4seed均值兼容且初始几何梯度非零，
才准备独立匹配512预算协议及实际CPU／BF16全模型预检。该判据仅是局部梯度支持，
不称训练有效；所有旧512终点保持，不追加更新、不扫权重或间隔。
[完整梯度协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/rank_margin_readback/hinge_gradient_audit/PROTOCOL.json)。

2026-09-14 12:43：官方反向过程记录已于12:39:22退出0（PGID93057），
九阶段／16与50步／正确错配／两分支／前四seed的288次最终样本全部精确复现；
128项检查通过，完整8,319,216参数状态不变，九曲线全部查看。18份保存逐步数组的
误差复算和最终clean／prev_sample精确一致也通过。16步的失败TRAIN277／298，
四seed均值在所有中间步的最小误差仍为最终误差99.55%／91.47%，说明这四条路径的
最后几步偏移不能解释大部分残差；这不是选择提前输出，也不能替代原32draw结果。
[反向完整结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/RESULT.json)、
[保存数组回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/SAVED_REVERSE_READBACK.json)。

下一有界固定xt条件诊断已送至同一295673/server35/step0：
tag generator_recorded_condition_job295673。仅读取原16步正确示范的已保存噪声状态，
保持状态／时刻／因果观测相同，只切换示范；完整官方encoder与denoiser共2,304次前向，
要求所有原正确epsilon和官方clean重建精确复现、RNG和完整状态不变。
全部九阶段报告，但方法判断仅依据TRAIN；错配输入只是敏感性干预，不是新物理标签。
无新采样路径、训练或物理步。完成九曲线后检查早／晚时刻的示范效用，再声明下一匹配目标。
[固定xt协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/fixed_xt_condition/PROTOCOL.json)。

2026-09-14 12:38：50步官方DDPM诊断已完整结束，PGID2921795于9月13日19:18:57退出0。
九阶段／五条件／32draw、每阶段每条件两个旧16步样本精确复现、完整状态检查、
保存指标回读与全部72面板查看完成。严格比较全部完整性通过；TRAIN仍5/7、复用检查0/2。
277／298误差下降8.71%／2.07%，但仍不满足原完整判据；318上升12.28%，
因此50步没有改善整体通过率，保留原full-rank0.25的16步终点作为研究基准。
这不证明任意求解器都无效，仅否定此次官方16到50步改动足以解决问题。
[严格采样比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/frozen_evaluation/SOLVER_COMPARISON.json)。

原294775在全部采样结束后被外部取消，Slurm结束时间9月13日22:01:03，step0于22:01:33结束。
9月14日确认用户无其它排队／运行作业后，唯一替代295673/server35/显式step0已获得，
8CPU／64GiB／1GPU／一天，同tmux curiosity_generator_h200_20260913的%722保持。
零更新反向过程记录于12:37:44启动，PID／PGID93057，tag generator_reverse_probe_job295673。
用完整官方predict_action与原scheduler.step，记录每步输入、epsilon、预测干净样本和输出；
九阶段、16／50步、正确／错配、两分支、原前四seed共288次保存样本重放，必须与旧终点精确一致。
不改变计算、不使用标签指导生成。完成后检查九条反向曲线、完整状态与逐步偏差／方差，
据失败TRAIN阶段误差发生时刻决定下一有界诊断；不追加旧512训练预算。
[反向过程协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/PROTOCOL.json)。

今天再次在线检查[Zero-WAM官方仓库](https://github.com/robbyant-research/Zero-WAM)：
仍只有README／LICENSE／图片等，代码、模型、数据仍列为预计9月15日前发布。
当前路线继续用其示范条件未来监督思想、完整官方SUGAR与官方SMP；不称官方Zero-WAM复现。
SMP目前是实际全身／物体状态的任务先验，不能直接把36维参考命令当其状态输入，
也不能以Carry类别分数代替对具体示范的跟随。生成闭环、独立SMP收益、原始位置7/8和相机问题均未解决。

以下为带日期的历史记录；其运行状态不覆盖上述记录。

2026-09-13 19:03：encoder_rank025512已完整结束，为负结果。PGID2813722于18:40:21
退出0，两组各512更新、九阶段／五条件／32draw、保存指标回读、全部72面板和严格
前驱比较均完成。完整模型／Adam／初态／实际batch／冻结统计／条件交换均通过；
零示范完整模型与完整Adam精确复现前驱。九阶段正确生成误差全部高于原full-rank0.25，
TRAIN由5/7降为4/7，检查仍0/2。这条梯度范围改动不作为后续基础，原full-rank0.25
完整终点继续保留；不以单个终点的局部梯度方向推断实际训练收益。

| 阶段 | full-rank→encoder-only MSE | 原／替代选对（各32次） | encoder-only全判据 |
| --- | --- | --- | --- |
| 158 TRAIN | 0.000558→0.002808 | 32／32 | 通过 |
| 178 TRAIN | 0.000378→0.008116 | 29／31 | 通过 |
| 197 TRAIN | 0.003062→0.012205 | 25／20 | 失败 |
| 218 已使用检查 | 0.039437→0.041710 | 32／0 | 失败 |
| 245 TRAIN | 0.003010→0.003629 | 32／32 | 通过 |
| 258 已使用检查 | 0.030865→0.035355 | 32／3 | 失败 |
| 277 TRAIN | 0.023596→0.034033 | 24／27 | 失败 |
| 298 TRAIN | 0.017171→0.022456 | 31／31 | 失败 |
| 318 TRAIN | 0.007766→0.009525 | 32／32 | 通过 |

[完整终点](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_encoder_rank025512/RESULT.json)、
[严格比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_encoder_rank025512/frozen_evaluation/PAIRED_COMPARISON.json)。

实现保留全部8,319,216参数。排序分支通过完整官方functional_call只向已有target_state_net
传梯度，基础去噪仍训练所有原参数。实际CPU与H200 BF16预检在发布初态／旧学习终点
均验证前向值、基础全梯度、target排序梯度、后续RNG精确；其它rank梯度严格为零、
完整状态不变，全部50时刻反向有限。该实现检查通过不等于方法有效。
[CPU预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_encoder_rank025512/PAIRED_CPU_PREFLIGHT.json)、
[BF16预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_encoder_rank025512/PAIRED_PREFLIGHT.json)。

完整共同噪声对照于18:51:07退出0（PGID2882453）：两个完整终点、同18真实案例、四组
保存噪声、全50时刻、正确／错配条件，状态／输入／normalizer／scheduler检查和全部
九曲线查看通过；full025复算的全部指标数组精确复现上轮。encoder-only在九阶段的
全时刻正确epsilon误差均降低，例如277／298为full-rank的0.77981／0.57544倍，但生成
误差全部上升。目标已进入加噪输入，监督拟合改善不能替代从纯噪声生成的质量。
[共同噪声诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_encoder_rank025512/frozen_evaluation/rank_component_audit/RESULT.json)、
[全时刻比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_encoder_rank025512/frozen_evaluation/rank_component_audit/DENOISING_COMPARISON.json)。

下一有界零更新采样诊断已启动：matched_generator_branch_rank025_solver50使用原完整
full-rank0.25两组终点的只读引用，仅将官方policy.num_inference_steps从16改为50。
沿原官方DDPM scheduler走49到0全部时刻，不写新求解器、不改裁剪／归一化／权重／
真实数据；不是训练追加。每阶段每条件先精确复现原16步的前两个保存draw，再进行
全部九阶段／五条件／32draw，检查完整权重不变和保存指标，查看全部72面板。
同seed保证沿相同官方初始化调用；不同步数的反向随机路径和算力不同，不称逐时刻
噪声或FLOPs匹配。50为完整训练时间网格，不搜索最优步数；原完整判据不变。
[采样诊断协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_rank025_solver50/PROTOCOL.json)。

当前唯一294775/server43/显式step0保持，tmux curiosity_generator_h200_20260913；
tag generator_solver50_job294775于19:02:27启动，PID／PGID2921795，
计算Python子进程2921805已确认存活。158完成并通过完整检查／分支判据，MSE0.00057785；
其余阶段继续，尚无50步完整终点。原294457外部取消历史不变，保留全部旧模型／Adam／实际237和358失败。
生成物理闭环、原始位置物理前提、相机兼容性和独立SMP收益仍未解决；本轮0新PhysX。

以下为带日期的历史记录；其中“下一步／运行中”不覆盖上述当前状态。

2026-09-13 18:15：0.25完整匹配实验及严格回读已完成，PGID1861526于17:50:10退出0。
完整官方Generator两组各512更新，九阶段／五条件／32draw、保存指标复算和全部72面板
查看完成。完整模型／Adam／冻结统计／实际batch／初始及零示范生成样本均匹配；
零示范完整模型与Adam精确复现0.1对照。九阶段正确示范MSE全部低于0.1，旧通过阶段
无退化；TRAIN从4/7升至5/7（新增197），两个已使用检查仍0/2。科学目标仍未通过。

| 阶段 | 0.1→0.25 MSE | 原／替代选对（各32次） | 全判据 |
| --- | --- | --- | --- |
| 158 TRAIN | 0.001930→0.000558 | 32／32 | 通过 |
| 178 TRAIN | 0.006073→0.000378 | 32／32 | 通过 |
| 197 TRAIN | 0.011233→0.003062 | 30／32 | 通过 |
| 218 已使用检查 | 0.039970→0.039437 | 32／0 | 失败 |
| 245 TRAIN | 0.003363→0.003010 | 32／32 | 通过 |
| 258 已使用检查 | 0.036939→0.030865 | 32／32 | 失败 |
| 277 TRAIN | 0.034225→0.023596 | 32／32 | 失败 |
| 298 TRAIN | 0.021623→0.017171 | 32／32 | 失败 |
| 318 TRAIN | 0.009102→0.007766 | 32／32 | 通过 |

[完整终点](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank025512/RESULT.json)、
[严格比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank025512/frozen_evaluation/PAIRED_COMPARISON.json)。

零更新分解全部通过：218／258采样方差仅占误差1.59%／2.10%，主要为预测均值偏差；
277／298分别41.55%／53.03%，有较大随机波动。277／298的误差中87.91%／90.97%与
真实两分支差异方向正交，因此选对分支不足以保证命令准确。输出裁剪下限仅解释这两
TRAIN失败误差24.15%／23.46%，原均值基线门槛仍可达。0.25两组各512条基础／排序／
总损失记录均有限且分解正确；不同更新的随机噪声／时刻不同，不以首尾排序值推断收敛。
[保存分量回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank025512/frozen_evaluation/rank_component_audit/SAVED_COMPONENTS.json)。

三个完整终点的共同噪声诊断于18:03:25退出0（PGID2142827）：同18实际案例、四组
保存噪声、全部50时刻、正确／错配两条件，完整状态／normalizer／scheduler／输入检查
全部通过，九阶段曲线全部查看。0.25的全50时刻正确epsilon误差在九阶段均高于0.1；
例如277为0.02486对0.01896，298为0.03126对0.01758。生成分支选择改善与监督去噪精度
退化并存。目标进入加噪输入，不称新生成结果。
[完整共同噪声诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank025512/frozen_evaluation/rank_component_audit/RESULT.json)、
[全时刻比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank025512/frozen_evaluation/rank_component_audit/DENOISING_COMPARISON.json)。

随后0.25终点的完整TRAIN梯度诊断于18:12:20退出0（PGID2244649）。14真实TRAIN案例、
四保存噪声、八预声明时刻0／3／9／15／24／33／42／49，共32组；完整状态、零示范
梯度和有限值检查全部通过。24/32组基础／排序全梯度内积为负，四seed均值中3/4为负。
平均去噪器余弦−0.58827，示范target encoder余弦+0.43875。全模型组合方向的基础损失
一阶变化仍为下降，不能称实际Adam更新必然损害精度；局部去噪器子空间则存在冲突。
八时刻平均不是全部50时刻训练梯度，也没有使用训练BF16／dropout状态或应用优化器。
[完整梯度诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank025512/paired_objective_audit/RESULT.json)。

下一项协议已生成：保持完整模型、14案例、0.25系数与两组各512预算，仅将排序梯度
限制在已有target_state_net；基础去噪仍更新全部原模型参数。该组合在32组原始梯度
诊断均保留基础损失一阶下降方向，但不保证实际训练收益。需要可微的官方去噪器
冻结参数路径传回条件梯度，不添加网络；额外前向计算不称等FLOPs。新目录
matched_generator_branch_encoder_rank025512尚未创建、代码适配／BF16预检／训练均未
执行。下一执行先落实梯度范围和同噪声／同遮罩／后续RNG／全状态预检，再运行新的
匹配512实验；不得延长旧终点或继续扫系数。
[下一匹配协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank025512/NEXT_MATCHED_PROTOCOL.json)。

Slurm记录原294457于17:51:04被外部取消（CANCELLED by 0），晚于原实验完整退出；
没有重跑旧训练。唯一替代294775/server43/显式step0保持RUNNING，tmux仍为
curiosity_generator_h200_20260913；两个诊断子进程均已退出，计算shell保留。
本轮所有诊断0优化、0新生成采样、0PhysX。生成闭环、原始位置物理前提、相机兼容性
与独立SMP收益仍未解决，所有旧模型／Adam／实际失败保留。

以下为带日期的历史记录；其中“下一步／运行中”不覆盖上述当前状态。

2026-09-13 17:29：成对排序0.1完整实验结束，PGID463616于17:22:55退出0。
两组完整官方Generator各512更新，全部九阶段／五条件／32draw采样、保存指标回读和
72时距面板查看完成。完整模型／Adam／冻结统计／共同初态／实际batch／条件交换均
通过；新零示范的完整模型与完整Adam逐项复现基础实验。跨实验九阶段初始／零示范
生成样本、实际目标／噪声seed和更新配置精确。新增目标的正确示范MSE在全部九阶段
降低，仍只有TRAIN4/7（158／178／245／318）通过，已使用检查0/2。完整科学结果仍负。

| 阶段 | 基础→排序MSE | 原／替代选对（各32次） | 排序判据 |
| --- | --- | --- | --- |
| 158 TRAIN | 0.007020→0.001930 | 32／32 | 通过 |
| 178 TRAIN | 0.010492→0.006073 | 31／31 | 通过 |
| 197 TRAIN | 0.015620→0.011233 | 24／21 | 失败 |
| 218 检查 | 0.044240→0.039970 | 32／0 | 失败 |
| 245 TRAIN | 0.003936→0.003363 | 32／32 | 通过 |
| 258 检查 | 0.038311→0.036939 | 32／4 | 失败 |
| 277 TRAIN | 0.043547→0.034225 | 21／26 | 失败 |
| 298 TRAIN | 0.035577→0.021623 | 31／31 | 失败 |
| 318 TRAIN | 0.018659→0.009102 | 32／32 | 通过 |

298虽分支选择通过，误差仍为两未来均值基线的0.749倍，未达到原0.5门槛；197／277
仍有均值偏差与选择不稳定。正确／错配响应增强不是完整行为跟随。0新PhysX控制步、
0生成闭环、0SMP收益。所有旧512终点、原实际237／358失败均保留。
[完整结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank512/RESULT.json)、
[严格跨实验比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank512/frozen_evaluation/PAIRED_COMPARISON.json)。

依据全部七TRAIN误差降低、通过数2→4、旧通过阶段保持，以及原TRAIN梯度审计候选
列表，下一有界实验仅改变排序权重0.1→0.25。选择规则不读取218／258结果。新计划
matched_generator_branch_paired_rank025512保持14案例、112行batch、同完整初态／旧
统计／seed272084／AdamW1e-4／warmup16／各512更新。两种排序目标都已有相同错配
前向结构，本次只改变系数，不扩大数据或旧预算。全部旧结果和两次实际失败继续保留。
最终协议已生成，025运行目录／系数适配／BF16预检／训练尚未执行，不称已启动。
[NEXT匹配协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank512/NEXT_MATCHED_PROTOCOL.json)。
唯一294457/server43/step0仍保留，当前本轮子进程均退出；下一步落实上述有界匹配实验。

2026-09-13 17:16：完整成对条件梯度审计已完成，PGID3060843于17:02:21退出0。
发布初态／dense终点×四噪声seed×四时刻15／24／33／42，均使用同14个真实TRAIN案例，
共同噪声数组完整保存。32组完整梯度与七阶段误差已查看；全部模型／旧统计保持、有限
梯度、仅几何交换检查通过，零示范对照误差相等且梯度精确为零。初态排序梯度只在新
示范输入列出现，终点也传入完整去噪器。初态／终点全梯度余弦均值分别0.00349／0.08623，
示范列为0.13321／0.25536；部分单时刻仍冲突，不能称每次更新都共同改善。

预声明候选0.1／0.25／0.5／1／2／4在两个状态的均值及四seed均值均支持局部共同下降，
按最小合格规则选择0.1。该判据只是原始梯度的局部SGD方向，不是实际AdamW更新或
生成收益。没有新模型、优化或物理。
[完整梯度审计](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_dense_fit512/paired_objective_audit/RESULT.json)。

新matched_generator_branch_paired_rank512保持同14案例、112行batch、旧normalizer、
完整发布初态、seed272084、AdamW1e-4、warmup16及两组各512更新。唯一训练目标变化为
L=原IID去噪MSE+0.1×成对排序项；排序比较同一真实目标加噪输入在正确／错配示范下的
epsilon误差，间隔按真实两未来差异和当前alpha缩放。成对标签单独存放，不进入obs；
错配前向复用对应随机遮罩，随后恢复原随机数流。零示范排序梯度恒零，使用原损失。
这是本地完整模型目标适配，不称官方Zero-WAM IFP或SMP实现。额外计算增加FLOPs，
仅匹配更新／案例曝光，不称等算力。

完整CPU输入预检及H200实际BF16目标预检均通过：基础前向／噪声／时刻／后续RNG、
零示范损失／梯度精确，全部50训练时刻反向有限，模型状态不改。GPU预检PGID见记录，
于17:12:45退出0。正式有界pipeline于17:13:25启动，PID／PGID463616，tag
 generator_paired_rank512_job294457，唯一294457/server43/step0。当前零示范组512完成，
完整终点权重与完整Adam逐项复现旧dense对照；示范组训练中。终点将自动完成全部
九阶段／五条件／32draw及保存回读；尚无本轮完整生成结论。
[匹配协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank512/PROTOCOL.json)、
[BF16预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_paired_rank512/PAIRED_PREFLIGHT.json)。

2026-09-13 16:55：14案例终点的零更新范围／去噪／同阶段比较均完成。
完整冻结去噪PGID1214705于16:49:32退出0，四噪声seed×50时刻×18实际案例×正确／
错配；完整模型状态不变、无参数梯度、仅几何交换及全部有限值检查通过，九面板已查看。
197无输出裁剪误差；258／277／298／318裁剪下限仅占实际误差10.74%／13.09%／11.32%／
11.31%，全部阶段原均值基线门槛仍可达。245已通过且裁剪占其残余误差52.83%，不能
以裁剪占比高低替代跟随判据。

t24正确条件epsilon MSE两分支：197为0.00379／0.00187；218为0.00791／0.03677，
258为0.01643／0.02693；后两者正确／错配仍较接近。245为0.00197／0.00308，而错配
为0.04571／0.04401，具有明确监督条件区分。不是生成指标，真实目标进入了加噪输入。
[完整去噪诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_dense_fit512/frozen_evaluation/denoising_probe/RESULT.json)。

同六阶段比较的实际输入／标签与完整初始生成样本均精确一致。218／258主生成误差为
旧8案例的0.4170／0.3545倍，但替代选择仍0／1；178由通过变失败，298／318仍失败。
仅为描述比较：训练案例／batch／总计算量改变；两去噪诊断案例顺序和形状不同，即使
seed名称相同也不能声称逐案例噪声相同。主生成32draw比较则初始样本精确复现。
[同阶段比较](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_dense_fit512/frozen_evaluation/DENSE_COMPARISON.json)。

下一项有界工作是完整官方模型在14个TRAIN实际成对未来上的条件梯度诊断：固定共同
噪声，比较正确与错配条件，检查成对未来目标能否提供有效示范梯度，再预声明新的
匹配训练目标。这是本地方法研究，不称官方Zero-WAM IFP或SMP latent；不新增简化
模型、不追加旧512终点、不再仅凭增加相邻阶段数继续训练。该梯度诊断尚未执行。
当前唯一294457/server43/step0仍RUNNING，全部本轮训练／采样／去噪子进程退出，
计算shell保留。完整生成闭环、原始位置执行前提和独立SMP收益仍未解决。

2026-09-13 16:44：14案例完整Generator诊断已全部结束，PGID3465800于16:43:02退出0。
两组各512更新，完整模型／Adam／冻结统计／共同初态／实际batch／条件交换检查均通过。
九阶段全部五条件32次采样与保存指标复算完成，72个时距面板全部查看。TRAIN仅158／245
通过，为2/7；178／197／277／298／318失败。检查218／258均失败，为0/2。各阶段正确
示范MSE及两分支选对次数如下（每分支32次）：

| 阶段 | 用途 | MSE | 原／替代选对 | 全判据 |
| --- | --- | --- | --- | --- |
| 158 | TRAIN | 0.007020 | 30／32 | 通过 |
| 178 | TRAIN | 0.010492 | 27／31 | 失败 |
| 197 | TRAIN | 0.015620 | 21／20 | 失败 |
| 218 | 已使用检查 | 0.044240 | 32／0 | 失败 |
| 245 | TRAIN | 0.003936 | 32／32 | 通过 |
| 258 | 已使用检查 | 0.038311 | 31／1 | 失败 |
| 277 | TRAIN | 0.043547 | 16／23 | 失败 |
| 298 | TRAIN | 0.035577 | 22／28 | 失败 |
| 318 | TRAIN | 0.018659 | 24／30 | 失败 |

新增245能明确响应正确／错配示范，但197／277拟合仍失败，不能把补数据当作稳定条件
迁移。218／258绝对误差比旧8案例降低，替代选择仍失败；178从此前通过变为失败。
两实验数据数和batch总计算量同时增加，不解释为固定总算力因果收益。所有旧负结果与
实际237／358失败保留。0新PhysX、0生成执行、0SMP收益。
[14案例完整负结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_dense_fit512/RESULT.json)。

下一步零更新输出范围回读与完整冻结九阶段去噪诊断，并与旧同阶段读数比较：区分
裁剪下限、监督去噪拟合及示范敏感性。原512终点不追加，唯一294457/server43/step0
保留，训练和采样子进程已退出；诊断准备中。

2026-09-13 16:32：245三条真实400步已完成，新增1,200控制步、0优化。原采集pipeline
于16:20:20退出1：物理／参考／官方export子进程均退出0，但phase0的官方formatter会
添加开头重复行，适配器遗漏一行索引。已保留失败输出并修正为计入官方前缀，确保
八个未来标签全部来自真实行；245标签及t-5历史逐项精确，八旧阶段16案例全数组回归
一致。只读恢复没有重跑PhysX；旧pipeline退出码不改。

245全部14个物理／参考／XYZ面板已查看，原三维指标复算一致。原／替代箱体自身对
其它参考RMSE比为0.06628／0.14612；替代初始Z误差约0.289m随后恢复，仍存在摆动和
跟踪误差。当前九所需阶段9/9通过，累计11尝试9/11，237／358失败保留，原dense8/9
负结果不改。这是已知96／90两示范的真实状态覆盖，不是生成控制或独立泛化。
[完整采集与恢复结果](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_compatible/RESULT.json)。

最终14案例协议已生成，七TRAIN阶段158／178／197／245／277／298／318与已反复使用的
218／258检查未来标签帧不交。完整官方Generator两组各512更新，每批112行、每案例
4,096次曝光；总噪声行57,344/组，比旧实验32,768增加，不称固定总算力比较。
完整CPU预检全部通过：14案例与112行平衡批次、完整8,319,216参数／旧统计、IID噪声／时间、
目标零化／梯度／完整初始16步采样均正确。两组各512更新的pipeline已于16:34:48启动，
PID／PGID3465800，tag generator_dense_fit512_job294457，唯一294457/server43/step0。
随后自动执行九阶段全部五条件32次采样及保存回读；当前训练中，尚无新终点。
本次在线复查[Zero-WAM官方仓库](https://github.com/robbyant-research/Zero-WAM)，仍仅README／许可／图片，代码／模型／数据仍列为预计9月15日前发布。
[NEXT协议](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_compatible/NEXT_MATCHED_PROTOCOL.json)。

2026-09-13 16:17：dense三阶段采集全部结束，PGID2705656于15:52:31退出0。
九条新实际轨迹共3,438控制步，0更新；197／277各三条400步通过，237原／重复400步、
替代238步终止。当前九所需阶段8/9，累计十个尝试8/10，失败237／358全部保留。
197自身／其它参考箱体RMSE比为0.05258／0.07728，277为0.03658／0.08567；两组全部
物理／参考／XYZ曲线及237物理曲线共32面板已查看，三维指标复算一致。277短暂参考
误差峰约0.113m，仍非精确跟踪。237无共同有效切换后帧，没有导出合格分叉标签。
原14案例dense协议未生成、未训练；不能把8/9改为全通过。
[dense完整负结果](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_dense/RESULT.json)。

237替代在动作执行前已经触发官方ee_body_pos和obj_pos；箱体仅Z方向相差0.320952m，
两腕相对参考误差0.319515／0.341994m，超过原0.3m界限。已将无有效跟随帧的通用说明
修正为“排除终止帧后无共同有效帧”，避免误称参考从未选中；实际选中并执行过一个动作。
原判据和轨迹不改，不能把这例当作控制恢复失败或有效条件跟随。

只读静态审计使用完整官方参考变换／终止公式和保留真实状态，在197／237／277两组
共六条的终止布尔、对齐身体／物体数组及无切换基线状态全部复算通过。使用尚未在318
切换的完整实际前缀扫描原先223..253邻域，按原规则选取标签时钟不交且两参考所有
原终止条件均false、最靠近238的245；不改阈值或参考Z。静态兼容不是后续物理成功。
[切换兼容审计](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_dense/switch_admission_audit/RESULT.json)。

新的generator_actual_branch_coverage_compatible已准备并在唯一294457/server43/step0
启动，tag generator_compatible_branch_coverage_job294457。仅245原／重复／替代三条
400步，最多1,200新控制步、0更新；24条旧轨迹只读复用，全部16旧分叉formatter回归
精确，两来源未来标签帧仍与218／258不交。累计尝试11阶段，保留237／358失败，最多
9/11；dense原8/9失败不变。当前245实际采集中，尚无新的完整组结论。
[245配对协议](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_compatible/PROTOCOL.json)。

后续完整14案例适配代码已准备但未执行：七TRAIN阶段158／178／197／245／277／298／318，
保留218／258检查；完整官方模型和旧normalizer、IID／目标零化／历史、lr1e-4／seed272084／
warmup16不变，各512更新。每案例八次噪声重复组成112行batch，每案例仍4,096曝光；
比旧8案例batch64总噪声行从32,768增至57,344，不能冒充固定总算力单变量比较。必须
先完成245实际与所有曲线，再写最终协议、合并实际数组并运行完整112行预检；没有
准备成功的模型运行目录或新Generator更新。原237不能进入训练标签。

2026-09-13 15:45：完整联合条件覆盖审计完成，当前8实际拟合案例、前轮3,897普通片段、
全部4,125可用片段分别统计；fit512与replay656两个完整模型的旧normalizer／全权重
保持和3×256条件token形状均通过，所有拟合案例能检索自身。全可用集合含96／90
和重复使用检查的标签，只是可用数据，不是当前模型的TRAIN或新验证。

218替代分支在全可用库的纯几何最近片段来自90，命令MSE0.00552；但fit512学习条件
的联合最近片段来自96，命令MSE0.09635且更偏原分支。258替代分支同样出现几何对应
正确、联合状态更偏原分支的现象。前轮3,897片段联合邻居也偏原分支。这是距离描述，
不证明唯一原因；支持补充缺失状态附近的真实共同世界参考选择干预，而非混合几何
检索控制器。首次CPU审计因重复适配已扩展encoder停止，0优化／采样／物理；原协议
和失败记录保留。修正为各自完整官方初态后分别严格加载，隔离r1审计全部通过。
[联合条件覆盖](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_interpolation_fit512/frozen_evaluation/joint_condition_coverage_r1/RESULT.json)。

新generator_actual_branch_coverage_dense已于15:40:06在唯一294457/server43/step0启动，
PID／PGID2705656，tag generator_dense_branch_coverage_job294457。固定最近时钟规则
从198／238／278附近选择197／237／277：只依据两来源未来帧不与218／258重叠，
平距取较早帧，不依据目标误差或物理结果。新增每阶段原／重复／替代各400步，恰好
9条、最多3,600新PhysX步、0更新；六旧阶段18条真实轨迹只读复用，旧12案例官方
formatter回归全部精确。完整冻结model607、初态／物理／因果对齐／独立跟随判据不变。

累计尝试阶段分母为10，含旧358失败；新三阶段全部通过时最多9/10。当前197原／重复
已400步完成，替代仍运行，尚无新完整组结论。先完成九条与所有曲线、XYZ、实际标签
时钟检查；全部通过后才准备独立的14真实案例完整Generator覆盖协议，不照搬旧8案例
固定数。没有新Generator训练，不释放分配或开并行物理。
[三阶段实际覆盖协议](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_dense/PROTOCOL.json)。

2026-09-13 15:27：几何阶段替换诊断完整结束，PGID2016609于15:23:54退出0。
四个失败阶段×实际输入及四个同来源TRAIN几何×正确／错配各8次完整采样，全部20模式
保留；完整模型状态不变、仅声明字段变化、实际与同阶段替换逐样本复现原采样均通过。
218／258所有正确几何donor的替代选择均0/8。218换158几何时正确／错配输出变化MSE
从0.0000826增至0.04493，却仍偏原未来；只观察条件响应不能作为跟随收益。晚期两个
训练阶段对不同donor有响应，但跨阶段输入不物理一致，不择优或改写原判据。
[几何阶段诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_interpolation_fit512/frozen_evaluation/geometry_phase_probe/RESULT.json)。

本轮新增完整模型训练两组各512更新、全部六阶段采样和三项零更新诊断已经完成，
0新PhysX步、0生成控制、0SMP收益。下一步先只读核查12个实际案例与完整native TRAIN
在当前状态／历史／示范几何／未来命令及完整学习条件token上的联合覆盖，依据证据
预声明需要补充的真实配对；不把更多噪声重复当作更多示范，不追加旧终点，也不把
失败且反复使用的218／258当作未使用测试。下一核查尚未运行。
当前唯一294457/server43/step0仍RUNNING，全部本轮子进程退出，保留计算shell。

2026-09-13 15:24：输出裁剪下界回读完成，六阶段保存预测均在实际DDPM的[-1,1]
范围内，指标精确复算。298／318真实目标有3.47%／2.95%坐标超界，不可避免MSE为
0.004028／0.002110，占当前采样误差11.50%／14.67%；258为0.004113，占3.81%。
任何阶段都没有因裁剪而必然无法达到均值基线判据，不能把裁剪当成唯一原因或直接
更换归一化来改判。
[输出范围回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_interpolation_fit512/frozen_evaluation/OUTPUT_RANGE_READBACK.json)。

完整冻结去噪诊断也已结束，PGID1396801于15:19:38退出0，四噪声seed×全部50个
训练去噪时刻×12实际案例×正确／错配几何；0更新、0生成采样、0物理。全部完整
模型状态／有限值／仅几何交换检查通过，六面板曲线已查看。t25正确条件的两分支
平均去噪epsilon误差，早期158／178约0.00144／0.00138，晚期298／318约0.00582／
0.00466，检查218／258约0.06298／0.05631；后两阶段正确／错配差异很小。检查阶段
在目标加噪的监督去噪中已失败，不能只归因于最终采样。t49清洁重建误差受极低alpha
放大，但原16步采样从t45开始；不把t49发散单独归因于当前推理失败。
[完整去噪诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_interpolation_fit512/frozen_evaluation/denoising_probe/RESULT.json)。

下一零更新完整模型输入诊断已启动：在四个失败阶段218／258／298／318，保留真实
当前物体／历史／目标字段，分别替换四个TRAIN阶段158／178／298／318的同来源几何，
并各做正确／错配8次完整采样；实际输入与同阶段替换必须精确复现原32次采样前8次。
所有donor分别报告，不择优。跨阶段混合不具物理一致性，只是条件敏感性诊断；不会
作为新反事实标签、检索控制器或训练判据。唯一294457/server43/step0串行执行，tag
 generator_geometry_phase_probe_job294457；仍无新优化或物理。

2026-09-13 15:16：八案例专门拟合完整结束，PGID3814619于15:13:12退出0。
完整官方Generator两组各512更新，全部模型／Adam／共同初态／实际batch／输入交换
检查与六阶段保存指标复算通过；全部48个时距面板已查看。TRAIN158／178通过，
MSE0.006188／0.007997，正确分支次数30/32、32/32及28/32、31/32；298／318失败，
MSE0.035017／0.014381，正确次数19/32、26/32及25/32、30/32。318虽明显使用条件，
仍未达到均值基线和逐分支选择门槛。TRAIN为2/4，不能称八案例全部拟合通过。

检查218／258仍失败，MSE0.106095／0.108062，正确次数32/0及32/1；正确／错配曲线
基本重合，替代关节／速度偏差仍大。没有延长旧预算或新物理。该结果支持早期拟合
可改善，但没有建立稳定阶段迁移；不与混合实验冒充单变量收益比较。
[八案例完整负结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_interpolation_fit512/RESULT.json)。

下一步零更新审计固定normalizer和实际DDPM最终输出裁剪的可达误差下界，逐六阶段
复算保存预测与真实目标；不改normalizer、标签、采样器或阈值。随后根据结果定位
完整去噪／条件使用问题，再决定不同的有界实验。唯一294457/server43/step0计算shell
保留，训练／冻结采样子进程已退出，目前无活动GPU子任务。

2026-09-13 15:08：八真实案例的完整预检全部通过：12个TRAIN／检查数组对应、64行
平衡批次、完整8,319,216参数及冻结旧统计、IID噪声与原时间、归一化目标零化、
有限反向和完整初始16步采样通过。main前一次yaml.parser导入失败由既有import-only
bootstrap恢复，未重复优化。pipeline于15:06:30在294457/server43/step0启动，
PID／PGID3814619，tag generator_interpolation_fit512_job294457；两组各512更新后
自动完成六阶段冻结采样及保存回读。当前尚无终点，未新开物理或其它训练。
[八案例预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_interpolation_fit512/PREFLIGHT.json)。

本次再次在线核查[Zero-WAM官方仓库](https://github.com/robbyant-research/Zero-WAM)，
仍是README／许可／图片，代码、模型和数据仍列为预计9月15日前发布，尚无可用实现。

2026-09-13 15:06：内插重放已于14:53:32完整结束，PGID1098565退出0。两组完整官方8,319,216参数
Generator各656更新；完整模型／Adam／共同初态／实际batch和对前轮初态与batch的精确
检查均通过。固定八来源542片段正确MSE0.034210、零条件0.035090、错配0.035967；
相对零条件改善2.51%、相对错配4.89%，均未达不变的5%门槛。六阶段全部失败：
158／178／298／318拟合分支正确次数分别22/23、24/20、18/17、14/25（每分支32次）；
218／258仍32/0。全部六阶段与native曲线已查看，保存采样指标复算通过。晚期拟合
曲线也只有较弱条件分离，不能把失败只归因于未拟合阶段。完整负结果与旧终点保留。
[内插完整负结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_interpolation_replay16/RESULT.json)。

另补看实际298／318／358两分支的全部XYZ曲线，原完整三维参考误差逐项复算一致。
298／318主要在Y方向分离并跟随自身参考；358仍失败。累计实际阶段6/7不变，曲线
不是新相机视频。没有新PhysX控制步或生成闭环证据。

下一有界诊断matched_generator_branch_interpolation_fit512：仅用同八个真实拟合案例，
完整官方模型和旧冻结统计不变，IID／归一化目标零化／完整历史保留；每批64行为八案例
各八次噪声重复，两组从完整发布初态和新Adam各512更新，lr1e-4／seed272084／warmup16。
每真实案例4,096次曝光与旧四案例256诊断相同；这是专门拟合诊断，不是与混合实验仅改
一个变量的收益比较，也不是新独立示范。全部六阶段五条件各32次冻结采样自动执行，
四个TRAIN判据与两个已使用的218／258检查分别报告。不延长旧656终点、不改阈值。
当前准备与完整CPU预检阶段；唯一294457/server43/step0仍RUNNING，计算shell保留。

2026-09-13 14:47：内插重放的完整预检通过：全部十二实际分叉数组精确，八拟合分叉组成
64行平衡损失检查；5,193行实际混合数据及整轮所有batch精确，首个128行批覆盖所有
八分叉。完整8,319,216参数与冻结旧统计保持，目标零化／原噪声时间／IID噪声／有限
反向梯度／完整16步初始采样全部通过。main前一次torch._utils瞬时导入失败由既有
import-only bootstrap恢复，没有重复优化或主程序写入。内插两组各656更新的pipeline
已在唯一294457/server43/step0启动，tag generator_interpolation_replay_job294457；
随后自动执行全部native与六阶段冻结采样／保存回读。当前尚无新终点，不追加或并行
启动其它训练，保留分配与旧负结果。
[内插完整预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_interpolation_replay16/PREFLIGHT.json)。

2026-09-13 14:43：298组三条实际400步已完整结束，共1,200新PhysX控制步，PGID3538166于
14:37:52正常退出。完整权重／初态／共同世界／物理保持和独立跟随全部通过，原／替代
箱体自身对其它参考RMSE比0.02853／0.09292，身体比0.08627／0.13098。八个物理／参考
面板已查看：替代当前参考误差峰值约0.122m，终点约0.035m，仍非精确跟踪。当前需用
六组全部合格，累计七个尝试阶段为6/7，358失败仍保留；之前六阶段5/6结论未改变。
本次新晚期可行性总计九条真实400步、3,600新步（318／358六条＋298三条），0优化。
[298和累计覆盖结果](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_bracket/RESULT.json)。

新的matched_generator_branch_interpolation_replay16协议已预声明：完整发布初态／
旧冻结normalizer／IID与归一化目标零化／AdamW5e-5／seed272051／16epoch／656更新
每组不变。3,897普通片段和1,296分叉噪声行合计5,193行不变，八真实分叉各重复162次；
仅以298／318四个新晚期分叉替换一半早期重复曝光。拟合158／178／298／318，保留
已使用的218／258检查，来源未来标签帧实际复核不交；全部六阶段五条件32次采样和
原八来源542片段五条件8次采样均须执行。对前轮完整初态和实际batch顺序另做精确
比较；原验证／分叉判据不改。是已知来源内插诊断，非未使用测试或独立示范泛化。
[内插匹配协议](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_bracket/NEXT_MATCHED_PROTOCOL.json)。

CPU协议准备曾在写出合并实际数组后遇到PyTorch内部unittest.case瞬时导入失败；0模型
更新／0物理。已使协议准备仅需NumPy，逐项复核并只读复用已写数组；完整模型导入
仍由既有main前bootstrap处理，未安装或替换依赖。当前模型运行目录准备／完整预检
正在执行，新训练尚未启动。唯一294457/server43/step0保留，所有采集子进程均退出。

2026-09-13 14:34：318／358晚期采集已完成六条400步、2,400新物理步，PGID400147于
14:18:56正常退出，0更新。318通过；358虽然抬箱、初态／共同世界及全部400步通过，
独立参考跟随失败：替代分支对自身／原参考的箱体RMSE比1.3106、全身比1.1662。
42个切换后控制帧内，箱体X仍近原参考而Z有所改变；两组全部16个物理／参考面板已
查看。六阶段采集判据为5/6、两个新晚期1/2，明确失败；358未导出合格分叉，不启动
原先要求两新组都通过的训练协议。
[六阶段完整负结果](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage6/RESULT.json)。

下一独立有界可行性实验generator_actual_branch_coverage_bracket已准备并启动：
只新增298阶段原／重复／替代三条400步，最多1,200物理步，0优化；切换后102控制帧，
不延长旧358终点。复用已通过158／178／218／258／318五组只读数据；全部十个分叉
官方formatter回归精确。累计尝试仍包含失败358，最多6/7可用，不能改写旧5/6负结果。
新的拟合候选158／178／298／318与固定218／258检查的未来来源帧预查不交。
只有新298通过不变的完整物理／独立参考跟随判据并检查曲线，才准备使用六个合格组的
新内插协议；数据覆盖分母和358失败在新协议中单列。不改原模型、损失、预算或验证
判据来制造通过。当前同294457/server43/step0，tag generator_bracket_coverage_job294457，
三条真实采集中；没有新Generator训练。
[298配对协议](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage_bracket/PROTOCOL.json)。

2026-09-13 14:18：完整实际训练数据覆盖回读完成，3,897普通片段＋四真实分叉各计一次，
没有新模型或采样。后续218／258两分支各自最近的纯几何片段都更接近对应命令；加入
共同历史／物体状态的等字段权重距离后，替代分支最近片段改为更接近原未来。距离只是
描述性邻近关系，不是新预测器或唯一根因。258两分支即使按目标标签挑选最接近的存档
命令，MSE0.023370／0.019195仍高于两标签均值0.017524；这是存档集合oracle，不能
作为模型能力下界或可部署指标。该证据支持继续补真实共同世界分叉的阶段覆盖。
[实际覆盖回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_replay16_r1/frozen_evaluation/REPLAY_COVERAGE_READBACK.json)。

新generator_actual_branch_coverage6已于14:10:17在294457/server43/step0启动，
PID／PGID400147。旧158／178／218／258四组12条真实轨迹只读复用，旧八分叉数组
官方formatter重新导出逐项精确；新增318／358两个晚期阶段，各原／重复／替代400步，
恰好六条、最多2,400新PhysX控制步、0优化更新。同完整冻结model607、同初态／种子／
物理／历史，不改实际世界；只改变参考选择和既有因果参考对齐。未来拟合阶段为
158／178／318／358，保留218／258作为已使用的阶段内插检查；两来源未来标签帧集合
预查不交，仍不是新motion或未使用测试。318组三条已400步完成且全部物理／共同世界／
独立跟随／官方标签验收通过，八个物理／参考面板已查看；替代参考误差峰值约0.185m，
不能称精确跟踪。358组仍在运行，尚无6/6最终数据结论。
[晚期实际覆盖协议](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage6/PROTOCOL.json)。

正在将完整dataset／preflight／phase evaluator中的四分叉固定数改为协议声明数。
若两新组全部通过并完成曲线检查，下一匹配实验保持3,897普通行＋1,296分叉噪声行、
原模型／初态／优化设置／16epoch／656更新不变，仅用四个晚期分叉替换一半早期重复
曝光。当前未准备最终训练协议、未启动新Generator；先完成全部真实端点与实际数组
回读。旧656更新终点不延长，生成物理与SMP收益仍未建立。

2026-09-13 14:01：修正后的混合真实数据实验完整结束，PGID1462218于13:53:08正常退出。
两组完整官方Generator各656更新，全部权重／Adam／共同初态／实际batch／预算与冻结
统计检查通过；3,897个其它来源片段＋四个早期分叉，5,193行每epoch，仍单训练seed。
固定542片段／8来源／五条件各8次采样：正确0.034115、零条件0.035439、错配0.037765、
同模型零条件0.039977。正确相对错配改善9.66%（6/8来源），相对训练零条件仅3.73%
（8/8），未达预声明5%；整体条件判据失败，不能靠明显错配差异改判。

四阶段五条件各32次完整采样全部结束：158／178的正确MSE0.017782／0.017736，
正确选择27/30及26/26次，均未通过TRAIN分叉判据；218／258为0.060478／0.074144，
两阶段均原分支32/32、替代0/32，后续迁移仍失败。保存均值误差分别0.056677／0.066116，
远大于采样方差0.003801／0.008028；不能把失败简单归因于随机采样。全部native四图
和四阶段各八图已查看：早期关节／速度有条件响应但接触时序差；后续正确／错配几乎
重合，替代关节／速度误差明显。两个保存采样readback精确复算全部指标。
[混合完整终点](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_replay16_r1/RESULT.json)。

随后零更新的完整模型目标梯度诊断也完成，PGID2959213于13:58:47正常退出。四个新噪声
seed各遍历全部3,897个普通TRAIN片段和四分叉各32重复；完整模型状态保持，所有字段／
行数／有限梯度检查通过。完整模型梯度cos为-0.286/-0.364/+0.329/+0.441；新增几何列
cos四次均负（-0.250至-0.372），但按实际约25%分叉权重合并后的梯度，与两种目标的
梯度点积四次全部为正。因此存在局部方向冲突，却不支持“当前混合方向持续牺牲分叉”
这个更强解释。它是单终点fp32一阶诊断，不是实际Adam轨迹、唯一根因或调权收益证据。
[完整目标梯度诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_replay16_r1/objective_probe/RESULT.json)。

不追加此656更新预算，不开启生成物理或声称SMP收益。下一步优先核查实际分叉覆盖与
原始示范表示的对应，再预声明新的真实配对覆盖；不能仅凭更多噪声重复当成更多示范。
整体路线仍是完整SUGAR执行＋示范条件未来监督，官方SMP仅作独立任务级运动先验，
没有作者Zero-WAM代码时不把本地重建继续冒充官方实现。原始位置来源7/8与新相机缺失
仍未解决。唯一294457/server43/显式step0在本次观察仍RUNNING，所有本轮子进程均退出，
计算shell保留；没有取消、重复申请或新活动训练。首次拼批失败的2次消耗仍单列保留。

2026-09-13 13:47：强化混合预检全部通过：5,193行逐原始字段精确、整轮随机batch
全部拼接、两种首行顺序一致，移除两个未读取字段后的完整编码token精确；两组完整
8,319,216参数、IID噪声／时间、128行反向梯度与完整16步初始采样一致。
原294136被Slurm于13:41:05外部取消（CANCELLED by0），显式step于13:41:35退出；
不是本Agent释放。确认无其它分配后仅申请一个替代294457/server43/显式step0，
同tmux／pane%722保留。修正后隔离实验已于13:47:04启动，PID／PGID1462218，tag
`generator_branch_replay_job294457`，先zero_context后demo_geometry各656更新，随后
完整native与四阶段冻结评估／保存回读。当前尚无新终点；两次旧失败更新保留且单列。
[完整混合预检](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_replay16_r1/PREFLIGHT.json)。

2026-09-13 13:44：混合训练首个进程2813683于13:40:32退出1；zero_context在两次已
记录更新后因混合batch字段不一致触发KeyError，demo_geometry未开始，没有有效终点。
原始日志、初态及失败报告完整保留；这两次消耗单列，未保存的中间Adam不可声称恢复。
原分叉档案有joint_pos／project_gravity，而官方use_last_action格式省略这两个未读取
字段。已统一混合dataset输出为官方格式，原始档案不变；新增整轮所有batch、两种首行
顺序以及完整编码token等价检查。新隔离运行matched_generator_branch_replay16_r1
仍采用原656更新／组协议，从共同完整发布初态开始，当前强化预检中。后续运行发生异常
时会保存完整当前模型／Adam，仍不冒充精确scheduler／sampler续跑。
[首次执行失败记录](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_replay16/EXECUTION_FAILURE.json)、
[隔离修正协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_replay16_r1/PROTOCOL.json)。

2026-09-13 13:38：完整Generator阶段迁移实验已结束，两组各256更新，全部完整权重、
Adam、实际batch与输入／采样检查通过；PGID2719057于13:08:48正常退出。158阶段
正确MSE0.007509、分支选择30/32与32/32，判据通过；178阶段0.012463、28/32与30/32，
但未通过两标签均值基线阈值。后续218／258阶段MSE0.105185／0.271703，均32/32选择
原分支、0/32选中替代分支，迁移失败。全部四阶段各八个时距面板已查看，保存采样复算
精确；后续误差主要在采样均值，不能只归因于随机方差。不追加此四样本256更新预算。
[阶段迁移完整结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_phase_iid256/RESULT.json)。

完整冻结输入替换诊断亦结束，PGID2748970于13:16:02正常退出。后续两阶段分别保留
实际输入，或换用178阶段历史／物体／两者；正确和错配几何各8次完整16步采样。
实际对照逐样本精确复现旧采样，所有权重与字段检查通过；所有替换仍8/8选择原未来。
这些混合输入不构成物理一致的反事实状态，距离仅为诊断；结果不支持把历史变化视为
唯一失败原因，也未恢复示范条件迁移。
[冻结输入诊断](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_phase_iid256/frozen_evaluation/input_branch_probe/RESULT.json)。

下一有界实验matched_generator_branch_replay16已准备：55个其它TRAIN来源的3,897个
真实片段，加158／178四个真实分叉；普通96／90来源全部228片段排除，以保留后续来源
标签帧不交。每个分叉每epoch重复324次，合计5,193行、约25%分叉权重；重复不是独立
轨迹。两组同完整发布初态、旧冻结TRAIN normalizer、局部IID及归一化目标零化、完整
历史，固定16epoch／656更新。旧统计曾见96／90几何，不能称全新示范或未使用测试。
固定八来源542片段验证和四阶段分叉检查分别完整执行，只有两者均通过才进入生成控制
前提检查。此前位置来源7/8和相机缺失仍保留；本轮0新物理步、无SMP收益结论。
[混合真实数据协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_branch_replay16/PROTOCOL.json)。
当前完整实际数据／128行混合batch模型预检正在CPU执行，尚未启动新训练。唯一分配
294136/server29/step0保留，旧子进程均已退出；预检通过后直接串行训练及全部冻结回读。

2026-09-13 12:55：真实分叉阶段覆盖已完整结束，PGID2656801于12:51:44正常退出。
新增9条PhysX轨迹全部400步，共3,600真实控制步；复用158组三条旧model607轨迹明确
不计新增步。158／178／218／258四组全部通过完整冻结权重、初态／前缀／共同世界、
物理保持、独立参考跟随、官方formatter和因果输入／标签检查，4/4数据可用。
全部四组各八个物理／参考曲线面板已查看：重复与原轨迹重合，替代轨迹随各自参考
改变；218／258替代分支切换后约0.18／0.17m短暂箱体参考偏差后恢复，仍非精确跟踪。
保留完整实际状态、实际29-D动作及独立8×36参考命令，不是生成未来或SMP收益。
[完整真实分叉覆盖结果](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage4/RESULT.json)。

下一项有界完整Generator阶段迁移实验已预声明，并准备实际数组：158／178两组的
4个真实分叉用于拟合，218／258两组的4个真实分叉用于冻结检查。逐原始source核对
两者未来标签帧集合完全不交；仍是两个已知TRAIN来源内部的后续阶段迁移，不是独立
示范或未使用的测试集。两组同完整发布初态、旧冻结normalizer和IID噪声适配，统一
去除9维目标但保留完整因果历史，各256更新；每批4真实案例各重复16次，不算64条
独立数据。每个未拟合阶段分别要求正确／错配／零条件及两标签均值判据、每分支至少
28/32正确选择，训练阶段结果单列。全部完整模型／Adam／实际batch和32次采样保留。
[NEXT_MATCHED_PROTOCOL](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage4/NEXT_MATCHED_PROTOCOL.json)。
新数组和源时钟分离检查已完成，完整模型多阶段dataset／evaluator适配与训练尚未执行；
不把协议或数据准备当成新的模型终点。下一步直接完成这些适配、完整预检和既定有界
训练评估。此前固定验证集上的条件负结果、原始位置来源7/8问题和新相机缺失仍保留。
唯一294136/server29/step0保留；当前所有本轮训练／采集／采样子进程均已退出，不取消
分配、不重复申请，当前进入下一有界模型实验的准备。

2026-09-13 12:39：真实分叉采集已于12:38:24在294136/server29/step0启动，
记录PID／PGID2656801，tag generator_branch_coverage_job294136。完整旧分叉全部
数组formatter回归精确；计算节点直接核对新增入口后才执行。复用158组的三个旧
model607实际终点均400步，完整物理／独立参考跟随及精确共同世界／因果输入／
8×36命令／原始几何检查通过，计为复用的2个实际分叉样本，0新物理步。
当前子进程switch178_original PID2657218开始首条新实际PhysX采集；之后自动串行
完成178／218／258各三组及全部readback。尚无9条新轨迹的完整结论，不追加或重复
启动训练／采集，不释放保留GPU。

2026-09-13 12:36：完整冻结分叉迁移诊断已完成，五条件各32次／两真实分叉，0更新、
0新物理步。共同初态、数据顺序、同噪声零条件相等及换示范交换输出均通过。
正确示范MSE0.047078，两个来源分别31/32、2/32选中对应未来；比两标签均值基线
差2.212倍，分叉判据失败。全部两行四组时距曲线已查看，替代分叉速度／接触时序
误差明显。更广模型大多保留原分支，而专门的两样本IID拟合可通过；优先补充真实
共同世界分叉覆盖，不能据此声称已找到唯一根因或新增独立motion泛化证据。
[冻结分叉迁移](../experiments/demo_following/demo_future_smp_v1/matched_generator_goal_removed_iid16/frozen_branch_transfer/RESULT.json)。

前两次分叉入口调用在模型加载前因unknown stage退出，0采样；原记录保留。已原子
持久化脚本／模块，并在server29直接确认入口后成功运行，PGID2628324于12:31:31
正常退出。没有重复成功采样、模型更新或取消分配。

下一项generator_actual_branch_coverage4已准备：完整冻结model607，原TRAIN96/90
配对，复用已有158帧三个实际终点，另固定178／218／258三个切换时刻，各执行原／
重复／替代400步，共9条新rollout、至多3,600真实控制步，0优化更新。每组同初态、
完整权重、物理、因果历史与种子；只改变选择后的参考及既有单次因果参考坐标变换，
不改真实世界。全部组自动做实际物理／共同前缀／世界相等与独立参考跟随检查，
只有通过者用官方formatter恢复精确phase3分叉标签，失败仍在4组分母。
这是同两个TRAIN来源的阶段覆盖，不是四对独立示范或训练replicate；全部三个新阶段
必须通过才满足本次数据可用判据。采集器不启动新Generator训练。先完成旧真实数据的
完整formatter回归，再进入同一保留H200的串行采集。
[真实分叉覆盖协议](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_coverage4/PROTOCOL.json)。

2026-09-13 12:28：目标移除两组各528更新及全部冻结采样、保存数据回读完成，
12:26:25正常退出。完整权重／Adam、前轮共同完整初态、实际batch顺序和预算全部通过。
正确／零条件／错配／同模型零条件MSE0.034503／0.035129／0.035250／0.036260。
正确比零条件改善1.78%（8/8来源），比错配改善2.12%（7/8），均未达到预声明5%；
相对保留目标的前轮IID正确条件反而变差11.41%，只有1/8来源改善。全部四组时距
曲线已查看，角速度收益混合；原示范条件和绝对改善判据仍失败。不能说目标提示完全
无影响，也不能把移除它视为有效部署改进。不增加该预算或扫描遮蔽概率。
[目标移除结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_goal_removed_iid16/RESULT.json)。

已启动下一项零更新冻结分叉迁移诊断：用这两个完整新终点、共同世界158帧的两份真实
分叉输入，五种条件各32次完整采样，复用原分叉辨别判据。两个精确phase3样本未纳入
4,125训练片段，但同来源相邻片段已参与TRAIN，不能称motion-held-out／独立测试。
该诊断只区分是否连已知来源的共同世界分叉也未学会，新增物理步和训练更新都为0。
同294136/server29/step0保留，tag generator_goal_removed_branch_transfer_job294136；
当前结果待完成，原2591420已正常退出。

2026-09-13 12:24：目标移除完整预检通过，32个真实TRAIN输入、两组完整权重与恢复
采样精确；训练／推理目标9维为零，物体／历史字段保持，全部16步采样不受目标交换影响。
matched_generator_goal_removed_iid16已于12:23:36在294136/server29/step0启动，
记录PID／PGID2591420，先零条件后几何条件各528更新，再自动冻结五条件评估与回读。
当前零条件组已实际进入优化；尚无终点结论，不追加或重复启动。

2026-09-13 12:20：此前共同世界分叉实验及更广IID噪声实验均已完成。
完整官方Generator为8,319,216参数，12层／256宽，8×36命令，完整16步DDPM；
保留全部模型与AdamW。这里的噪声变更是明确的本地方法适配，不是原封不动的官方损失。

真实共同世界的两个TRAIN分叉各重复32次组成平衡batch，两组各256更新，归一化目标
9维明确零化；仅有两个真实样本，不能把重复数当独立数据。原全局noise_assignment下，
正确示范MSE0.027238，32次采样分别21／22次选择对应未来，原判据失败。
只去掉跨样本、依赖目标的噪声重排后，正确MSE0.003786，分别30／32次选择正确未来，
错配分别32／30次选择另一个未来；相对两标签均值、零条件、错配、同模型零条件的
误差比0.17794／0.09176／0.04785／0.09981，分叉辨别判据通过。
两轮共同完整初态、实际batch顺序、256更新及条件交换检查均通过，全部时距曲线已查看。
[原噪声分叉结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_actual_branch256/RESULT.json)、
[IID分叉结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_actual_branch_iid256/RESULT.json)。
完整官方欧氏距离全局分配的CPU新噪声探针中，沿两标签差方向用固定零阈值辨别分叉，
准确率从50.63%升至94.19%；这是新探针，不是历史训练噪声回放或唯一根因证明。
原终点32组同噪声配对有21组在两个prompt下选择同一未来，支持噪声携带分叉提示的解释。
[噪声耦合回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_actual_branch256/frozen_evaluation/NOISE_COUPLING_READBACK.json)。

更广IID匹配实验仍使用原4,125个真实TRAIN片段、完整历史与既有目标，各528更新；
不混入上述两个分叉样本。共同初态、实际数据顺序、预算及完整模型／Adam终点全部通过。
全部542验证片段、8个来源、五种条件各8次采样完成，正确／零条件／错配／同模型零条件
MSE为0.030970／0.030974／0.030901／0.030991。比原噪声正确条件改善8.205%，8/8
来源胜出；但相对零条件／错配比0.999871／1.002221，新增示范条件收益仍失败。
全部四组时距曲线已查看；保存采样复算一致，均值误差仍未优于零／错配。
[更广IID结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_demo_geometry_iid16/RESULT.json)。
这说明当前分叉TRAIN拟合可行及生成拟合改善，不能证明验证示范遵循、生成闭环或SMP收益。

下一项有界诊断为matched_generator_goal_removed_iid16：同完整初态、4,125片段、
每组528更新和独立噪声，仅在训练／全部评估统一零化归一化9维目标，保留完整因果历史。
原目标字段存档不变，不能把移除输入称原始目标来源兼容。先做完整模型实际数据／16步
采样输入检查，再串行执行匹配训练及冻结评估。正确示范须通过原条件判据，并单列相对
前轮既有目标IID结果的绝对误差改善；不能靠削弱对照声称部署改善。不追加该预算或
扫描遮蔽强度；若仍负，转入实际共同世界分叉覆盖诊断。当前尚无该诊断训练结果。

唯一分配294136／server29／显式step0保留，同tmux／pane%722。原分叉PGID2394483、
IID分叉PGID2480209及更广IID的PGID2529674均正常退出，后者12:13:12结束。
当前无活动GPU训练／物理子进程，正在CPU准备目标移除诊断；不释放或重复申请GPU。

2026-09-13 11:35：第二组有界Generator实验`matched_generator_history_dropout16`已完成，
两组完整8,319,216参数模型各16epoch／528更新，所有模型／Adam、共同初态、实际batch
和掩码检查通过；与上一轮初态和数据顺序也完全一致。只在训练时以独立RNG遮蔽50%
归一化历史命令，验证始终完整，官方模型／损失／训练循环保留，明确为本地方法适配。
正确示范MSE0.038188，零条件0.038034、错配0.038204、同模型零条件0.038177；
正确／零条件比率1.00404，正确／错配0.999560，均仅4/8来源胜出；比上一轮正确条件
差13.188%，8个来源全部变差。原判据和额外前轮改善判据均失败；全部时距曲线与保存
采样分解已查看，不继续调遮蔽概率或增加同一预算。
[遮蔽实验结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_history_dropout16/RESULT.json)。

原定16条位置来源物理对照也全部完成，5,702个实际控制步：实测参考8/8、原始示范7/8，
保留失败教师后的全10条覆盖为8/10、7/10。source68在138步因anchor_pos终止，未持续
抬箱；终止前躯干距实测／原始参考分别0.3016／0.2104m。全部16条实际48维输入逐帧
重建最大误差3.87e-7，实际动作／来源／时钟回读通过；八行三组物理曲线已全部查看。
原始位置替换前提失败，不是生成命令或SMP结果，也不能归为位置字段接线错误。
[物理结果](../experiments/demo_following/demo_future_smp_v1/original_position_feedback_floor/RESULT.json)、
[全部输入／物理回读](../experiments/demo_following/demo_future_smp_v1/original_position_feedback_floor/PHYSICAL_READBACK.json)、
[物理曲线](../experiments/demo_following/demo_future_smp_v1/original_position_feedback_floor/PHYSICAL_READBACK.png)。

完整冻结Generator分支诊断已完成：TRAIN／验证各24个实际样本，四个噪声seed，官方
50步训练扩散的0／25／49时刻。历史命令对中间时刻输出的影响远强于新增示范；新增
几何有梯度但没有稳定去噪收益。它是完整模型只读诊断，不是新训练或生成控制证明。
[输入分支证据](../experiments/demo_following/demo_future_smp_v1/matched_generator_demo_geometry16/frozen_evaluation/input_branch_probe/RESULT.json)。

数据审计发现4,125个TRAIN样本的既有物体＋历史输入逐行不完全相同；精确唯一性本身
不能证明缺少有用条件信息。更具体地，既有真实96／90配对在158帧具有完全相同世界、
物体输入和t-5命令，但不同未来命令；既有目标输入最大差0.681845，目标已经提供来源
提示。旧stride5导出首个选后片段从160帧开始，未纳入精确分叉点。
现已通过同一个官方formatter、原5帧间隔和phase3恢复158..193的两份真实分叉样本，
实际世界／因果输入精确、8×36命令标签与t-5历史逐项相同于原trace；29维实际动作
单独保存，原始示范几何来自原始来源时钟，未修改旧数据或生成新物理轨迹。
[条件数据审计](../experiments/demo_following/demo_future_smp_v1/matched_generator_demo_geometry16/frozen_evaluation/CONDITION_COVERAGE.json)、
[真实分叉数据](../experiments/demo_following/demo_future_smp_v1/generator_actual_branch_point/RESULT.json)。
下一项是预声明有界完整官方Generator的共同世界分叉拟合诊断：两组同一初态，仅示范
几何条件不同，显式统一零化归一化目标输入以去除既有目标提示，并比较独立噪声的正确／
错配／零条件与两标签均值基线。这项诊断尚未训练；即使两个TRAIN未来可分也不代表
泛化、生成闭环或SMP收益。当前不新增其他Generator预算。

分配294136／server29／显式step0仍保留于同一tmux。原PGID2005132在11:14:13正常
结束；遮蔽实验PGID2304440在11:24:10正常结束。没有活动训练／物理子进程，GPU没有
释放，也没有重复请求；当前继续真实分叉诊断准备和后续有界执行。

2026-09-13 10:59：全部76条冻结TRAIN采集及完整回读通过，28,155个实际控制步，
57条可用、19条学生失败；四条教师失败仍保留，全TRAIN覆盖口径57/80。全部四组／
76条高度曲线已查看；失败仍排除训练标签。4,043个新片段加82个既有TRAIN片段，
共4,125个；两组完整官方Generator已各完成16epoch／528更新，完整权重及AdamW
终点检查通过。共同完整初态、实际batch顺序、数据和预算逐项相同。

全部542验证片段、五种条件、每片段八次完整16步采样已完成：

| 条件 | 等权来源归一化MSE |
|---|---:|
| 发布模型 | 0.081724 |
| 训练零示范条件 | 0.033674 |
| 训练正确示范条件 | 0.033738 |
| 同模型错配示范 | 0.033687 |
| 同模型零化示范 | 0.033798 |

正确示范误差为重复上一命令的45.57%，8/8来源胜出；但相对训练零条件／错配条件
比率1.00191／1.00151，仅2/8和3/8胜出，预声明示范条件收益判据失败。四组时距曲线
已查看，适配模型几乎重合。这是完整生成器适配改善，尚非所选示范控制生成、真实生成
闭环或SMP增益；不改变判据或继续盲目追加预算。
[完整生成匹配结果](../experiments/demo_following/demo_future_smp_v1/matched_generator_demo_geometry16/RESULT.json)、
[时距／命令组曲线](../experiments/demo_following/demo_future_smp_v1/matched_generator_demo_geometry16/frozen_evaluation/HORIZON_ERRORS.png)。

当前294136／server29／step0、PGID2005132保留并运行原定16条零更新位置来源物理
对照。source8两组均完整通过，后续来源继续；这项检查不含Generator，不能作为生成
闭环证据。另以保存的完整终点与全部采样进行CPU条件影响／采样方差回读，零新采样、
零新训练；据此选择后续诊断，尚无新的Generator预算。
保存采样的CPU回读已完成：原指标全部复现；正确条件MSE分解为八次均值误差0.023316
加采样方差0.010422，均值误差仍不优于训练零条件0.023200或错配0.023275。
同噪声换错示范／零化示范引起的预测变化MSE仅约采样方差的1.33%／1.85%（等权来源
比率均值）。新增168列确实学习，权重范数0.265585；控制组保持精确零，不能把弱收益
归结为条件接线完全断开，也不能只归因于采样噪声。下一步用完整冻结模型隔离输入分支
影响与训练信号，在证据明确前不追加新训练预算。
[保存条件影响回读](../experiments/demo_following/demo_future_smp_v1/matched_generator_demo_geometry16/frozen_evaluation/CONDITIONING_READBACK.json)。

Generator既有9维目标位姿的来源已完成8条／2,982帧回读：它来自冻结Refiner实测参考库
的终点，位置逐项误差0、旋转矩阵最大误差2.39e-7；与原始demo同一终点时钟的箱体
位置相差2.4–12.4cm。官方采集、导出和Generator输入链路一致，未修改正在采集的数据。
当前两组16epoch及五种冻结条件仍使用同一个既有目标，能够检验新增几何的条件收益，
但不能证明脱离Refiner参考库。后续只用原始demo运行时，除48维位置反馈外，还必须
显式替换并评估此9维目标；已预声明的16条位置对照不包含Generator，不能覆盖该问题。
证据追加在既有[接线回读](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/generator_demo_geometry_preflight/TRACKER_COUPLING_READBACK.json)
的`generator_goal_provenance`；这是零更新、零新物理步的数据来源审计。

2026-09-13 09:44：当前唯一分配294136／server29／显式step0，保留同一tmux／%722。
记录的PID／PGID2005132正在执行`generator-data-and-matched`：从source36恢复剩余
46条实际采集，完成后自动完整回读，再串行运行两组16epoch Generator和冻结采样。
前30条、24条可用及11,203个已完成真实步均保留；当前尚未开始Generator更新。

293870／server51于09:31:07被Slurm `CANCELLED by 0`，step0于09:31:37结束。
source36仅初始化、无实际rollout，其空输出与日志单独保留；恢复核对全部30条完整
结果、真实trace和教师查询，未重跑完成来源。旧排入的回读／训练没有启动；训练命令
落回登录shell后被launcher正确拒绝。新连续流程在计算子进程内部串行执行并传播失败，
不再依赖交互shell排入后续命令。见[中断与恢复证据](../experiments/demo_following/demo_future_smp_v1/tracker_train_generator_corpus76/INTERRUPTION_JOB_293870.json)。

294136上的相机重试也在首例初始化发生Vulkan设备丢失，09:41:59退出，0个新相机终点。
同一节点此前视频成功不能证明当前配置稳定兼容；当前gpu分区查询只列H200。
[官方Isaac Sim 5.1要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
明确无RT Core的GPU不受支持。保留全部失败尝试，新model607视频仍缺失；继续可运行
的实际PhysX采集和完整Generator工作，不能把相机初始化失败计为物理失败或视觉通过。

当前采集进度快照：30/76条已完成、24条可用，共11,203个真实控制步；已由PGID2005132从中断处
恢复，最终口径以完整RESULT为准。完整官方Generator下一阶段已经预声明并准备：
两组各16个epoch，batch128、seed272051、AdamW lr5e-5、32步warmup，具体更新数为
`16 * ceil(最终实际TRAIN片段数 /128)`。LR／batch／短预算明确是本地适配设置。
完整原模型、官方训练循环和扩散损失保留；逐batch记录实际索引、保留每个epoch模型及
完整AdamW，并核对两组完整初态、实际数据顺序与预算一致。预声明
[匹配Generator协议](../experiments/demo_following/demo_future_smp_v1/matched_generator_demo_geometry16/PROTOCOL.json)。

既有8条实际验证轨迹已用官方formatter生成542个冻结片段，独立于TRAIN，未拟合统计。
正确／错配示范只改变8×21原始几何，其余输入和观察到的未来标签逐项精确；错配采用
固定循环下一来源、原始归一化时钟，不做窗口搜索。它不是另一条反事实物理标签。
[冻结验证数据](../experiments/demo_following/demo_future_smp_v1/generator_native_validation8/RESULT.json)、
[542片段接口检查](../experiments/demo_following/demo_future_smp_v1/generator_native_validation8/DATA_INTERFACE_READBACK.json)。

当前计算子进程已按序接入：剩余采集→完整数据回读→两组Generator训练及冻结采样，
统一tag `generator_data_and_matched_job294136`；Generator尚未启动。每条训练入口要求完整
实际数据核验通过。冻结评估包括发布模型、训练零条件、正确／错配／零化几何条件，
每片段全部8次完整16步采样，与重复上一条命令基线比较；不挑最好样本、不按验证挑
checkpoint。完整训练与采样执行尚未验证，当前无生成性能结论；新模型相机及真实
生成命令接入、SMP独立增益仍待完成。

生成命令接线的CPU检查已完成，但尚未接入实际rollout：
`generator_tracker_routing.py`保留完整846-D actor，当前36与未来288维只从完整官方
Generator保存输出派生，原474维实际状态／历史保持精确，过期计划直接拒绝。
完整601,629参数actor对8个已记录输入的CPU回读最大误差7.15e-7；改变被替换的数值
命令前缀不影响接线结果。395个实际控制帧的命令历史及79个官方10Hz样本一致。
官方formatter的`last_action`是5个控制步之前的36维命令，因此逐控制步重规划需要
保存五步真实已下发命令；不能误用上一控制步，也不能混用29维电机动作。

48维位置反馈仍明确来自冻结Refiner实测参考库的未来anchor／box位置，尚非Generator
预测。该依赖必须单独解决后才能主张完整生成输入；现有接线检查仅为接口证据，使用
保存的完整发布模型输出，不是新模型生成结果、物理成功或SMP收益。
[接线与时钟证据](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/generator_demo_geometry_preflight/TRACKER_COUPLING_READBACK.json)。

原始位置替换的完整离线回读已完成：全部8条／2,982个实际帧，原48维参考输入重建
最大误差1.94e-7，完整actor回读最大误差9.54e-6。原始与Refiner箱体参考平均差距
6.7–10.1cm、最大22.6cm；仅替换位置字段，策略输出RMS变化0.050–0.101，不能当作
数值等价替换。八行[位置／动作变化曲线](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/generator_demo_geometry_preflight/ORIGINAL_REFERENCE_POSITION_COMPARISON.png)
均已检查。这是冻结actor离线查询，未证明替换后的物理表现。

下一项16条冻结物理对照已写入Generator协议并接在其预测终点之后：两组同完整model607，
同一原native seed／世界／物理／预算，仅48维位置来源不同，0训练更新、共5,964个
计划控制步。原始位置适配使用官方坐标变换与原始demo帧0..7，既不读取Refiner实测
位置库，也不外推尾部；仍明确是已知demo几何，不称作生成位置。原510观察和288已知
命令保留，因此这项检查仍非生成闭环。接口代码已准备，GPU分支及该物理对照尚未执行；
两组都8/8通过才证明这个执行前提成立。若Generator预测与该前提均通过，再进入真实
生成命令handoff；否则依据失败组件继续完整模型诊断。

2026-09-13恢复会话时重新检查[Zero-WAM官方仓库](https://github.com/robbyant-research/Zero-WAM)：
当前可见文件仍为README／许可／图片，Release Plan写明代码、模型和数据预计9月15日前
发布，实际发布尚未核实。保留完整本地重建负结果，当前以完整官方SUGAR与SMP推进
项目方法；官方材料发布后再作兼容性审计，预计日期不作为可用实现的证明。

2026-09-13 最新完整结果：扩大TRAIN覆盖组在固定8条教师可用验证上**8/8完整通过**，
只训练旧配对的匹配对照为**0/8**；两组旧配对原／重复／替代均400步，独立参考跟随
检查均通过。失败教师source18／48仍保留，全10条覆盖口径为8/10对0/10。

| 验证source | 对照实际步数／终止 | 覆盖组实际步数／终止 |
|---|---|---|
| 8 | 253／obj_pos | 400／无 |
| 28 | 279／obj_pos | 400／无 |
| 38 | 306／obj_pos | 400／无 |
| 58 | 123／obj_ori | 326／无，原定短预算 |
| 68 | 85／anchor_pos | 400／无 |
| 78 | 321／obj_pos | 400／无 |
| 88 | 163／obj_pos及obj_ori | 256／无，原定短预算 |
| 98 | 154／ee_body_pos | 400／无 |

两组各304个真实环境，从同一完整model479／Adam恢复，各128更新、933,888条真实
转换；完整model607／Adam回读通过，actor／critic时钟12160／2160。共同训练初态
与模型／Adam精确，所有冻结验证初态与startup精确；教师冻结，critic沿原500边界更新。
两组24步真实预检各7,296条转换，无重置／更新。训练与全部冻结评估在04:46:31正常
结束，PGID781922已退出；当时保留293772/server29/step0，随后外部取消如下。

八行物理曲线及两组TRAIN配对曲线全部查看。覆盖组可持续抓取、抬升，并在部分参考
上放回箱体；仍有约0.20m后段位置距离及0.50rad朝向误差。这是单训练seed、既有固定
验证集上的执行基础通过，不是精确跟踪、生成未来、人类视频条件、SMP增益或最终目标
完成。source68共同窗口在交互前结束，双方零箱体误差不能作为跟随改善证据。当前
朝向回读使用原始导出参考在每个记录帧的旋转，未误用最终目标target_object_quat_w。

当前算力变化：293772/server29被Slurm在04:51:03标记CANCELLED by0，step0于
04:51:37终止。相机流水线PGID937754随分配退出，首条source8尚未进入rollout，
没有完成的新相机视频；旧尝试与日志保留。04:46:31之前已完成的无相机结果和模型不受
影响。确认没有其他活动作业后只提交了一个新请求293870，tmux
`curiosity_generator_h200_20260913`／pane%722，8CPU／64GiB／1GPU／一天，当前
已获配server51／step0。08:43:16相机重试在初始化时遇到Vulkan设备丢失，
08:44:51退出；没有新相机轨迹，转入不依赖相机的真实采集，保留原相机待完成事项。

CPU侧已准备下一阶段`tracker_train_generator_corpus76`：固定完整model607，全部76条
教师可用TRAIN来源、原始真实长度和未来尾部预算，共29,525个计划控制步、0优化更新。
每条从自己的native世界开始；完整890-D教师只作同状态查询，绝不接管执行。仅完整
物理通过的Tracker轨迹调用官方Generator formatter；36-D参考命令标签、29-D实际
执行动作和实际状态分别保存。保留4条教师失败在80分母，验证／测试不进入数据。
已完成配对导出器的共用接口重构，原两组全部RAW／IL数组回归精确相同（各41个片段）。
76条原始示范几何条件库已生成5,390个8×21片段，仅从原始TRAIN数值示范派生，不含
29-D关节目标、36-D命令、实际未来、动作、力或奖励。原始native时钟不重采样；采集器
在每条物理完整通过并经官方转换后，逐来源／逐帧绑定实际标签。条件库当前不含已执行
Tracker标签，不能计为完成的生成训练数据。76条冻结采集已在08:46:20启动，完整结果待采集结束；新Generator训练
尚未启动。准备产物见[冻结TRAIN采集协议](../experiments/demo_following/demo_future_smp_v1/tracker_train_generator_corpus76/PROTOCOL.json)
和[原始示范条件库](../experiments/demo_following/demo_future_smp_v1/tracker_train_generator_corpus76/original_demo_geometry_context/RESULT.json)。

证据：[完整匹配结果](../experiments/demo_following/demo_future_smp_v1/matched_train_coverage128/RESULT.json)、
[全部物理指标](../experiments/demo_following/demo_future_smp_v1/matched_train_coverage128/PHYSICAL_READBACK.json)、
[八行物理曲线](../experiments/demo_following/demo_future_smp_v1/matched_train_coverage128/PHYSICAL_READBACK.png)。

全部80条TRAIN冻结Refiner采集已于03:57:15完成，33,220个真实控制步，76条可用。
失败source40／46／60／66分别在6／284／342／327步末端位置／箱体位置／箱体朝向／
箱体位置终止；全部保留在80分母并排除训练参考。完整数据回读通过：来源与原始参考
时钟、29-D实际动作、官方转换和原始教师前缀对应正确，失败未混入训练库，76条持续
抬箱事件均在400帧窗口内。全部80条分组物理曲线已检查；这证明教师数据可用范围，
不代表学生能力、生成未来或相机视觉成功。原子发布新收尾检查时修正过来源ID假设：
trace保存原始来源ID，非内部槽位0；原采集数据与时钟没有因此修改。

匹配实验：两组均保留完整846-D Tracker和现有定位依赖输入，仅改变TRAIN参考分布。
对照只用旧示范配对；实验组152个环境用旧配对，152个覆盖76条native来源、各两次。
官方初始分配为env_id取模，304个槽位保证每条来源实际进入训练。各128更新480..607，
每组933,888条实际转换；原BCPPO在500进入critic warmup，教师冻结、critic此后可训练。
保持原训练阶段，不重置Adam或时钟。实际训练重置沿用官方参考结束／物理失败；
对照的首次参考结束在第450帧。配置episode_length_s=8不额外添加400帧终止项，
400帧冻结评估窗口与训练参考长度不同。终点两组均执行旧配对原／重复／替代以及固定8条
验证，保留两个教师失败在10条覆盖分母；8/8完整执行与TRAIN独立跟随均通过才算此
执行基础阶段通过。仍须比较匹配对照，不能把单例改善改判为总体成功。

完整官方Generator的条件适配状态恢复接口已补齐。官方wrapper固定重建9-D目标编码器，
不能直接加载扩展后的177-D权重；新接口只恢复已审计输入适配，并严格加载完整模型和
原有归一化状态。CPU零更新保存／回读检查：两分支均8,319,216参数，全部状态精确，
4个真实观察的条件与完整16步采样最大差均0；不重拟合统计、不构造优化器。
[状态恢复证据](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/generator_demo_geometry_preflight/STATE_RESTORE_READBACK.json)。
`generator_dataset.py`已提供官方workspace可用的真实片段数据适配，复用官方单样本
格式化并只额外传入原始示范几何。现有82个真实TRAIN片段的旧观察／完整8×36标签
精确，替换监督标签不改变任何输入，完整normalizer与已审计适配器相同；未完成的新
原生采集会被拒绝读取。它保留旧统计，仅拟合实际纳入TRAIN的新几何字段。
[数据接口证据](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/generator_demo_geometry_preflight/DATA_ADAPTER_READBACK.json)。
完整官方训练入口的适配已补齐：直接继承未改写的`TrainGeneratorWorkspace.run()`，
通过官方wrapper载入完整发布模型后仅扩展既有输入，显式保留AdamW并同步完成保存。
两组CPU零更新检查均为8,319,216个完整参数，初态一致，模型／原归一化及新建空AdamW
保存回读精确，归一化参数不进入优化器。发布checkpoint本身不含AdamW；该检查不证明
已训练优化器矩的恢复，也尚未实现官方循环局部scheduler／sampler的精确中断续跑。
[官方workspace接口证据](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/generator_demo_geometry_preflight/WORKSPACE_READBACK.json)。
新原生实际采集正继续，source1／2已各400步通过并完成73个实际标签片段的绑定；失败
source0／3保留并排除训练标签。全部76条完成后，已在当前连续计算子进程内接入
`train-reference-corpus --tracker-readback-only`，逐轨迹核查真实动作／来源／时钟、
完整成功标签及完整数据集，输出全部四组物理曲线。收尾回读尚未执行；实际Generator
训练仍未启动，不能由接口检查声称生成性能改善。

后续Generator接入仍有明确兼容性问题：完整官方8×36输出插值覆盖36个控制帧，当前
Tracker每步要求向前35帧的完整计划，沿用20帧重规划时仅age0具有全部所需帧。
当前288-D计划和48-D位置反馈仍来自数值参考，只打开use_generator无法使完整输入
成为生成结果。不得用数值未来填补缺失尾部后声称推理成功。SMP需要实际10×216身体／
物体运动特征，不能将36-D命令直接当作该接口。先解决执行覆盖，再验证独立原始
示范几何条件与冻结SMP的贡献；Generator新训练尚未开始。

证据：[全部采集](../experiments/demo_following/demo_future_smp_v1/train_reference_coverage80/RESULT.json)、
[完整数据回读](../experiments/demo_following/demo_future_smp_v1/train_reference_coverage80/READBACK.json)、
[分组曲线1](../experiments/demo_following/demo_future_smp_v1/train_reference_coverage80/PHYSICAL_GROUP_0.png)、
[分组曲线2](../experiments/demo_following/demo_future_smp_v1/train_reference_coverage80/PHYSICAL_GROUP_1.png)、
[分组曲线3](../experiments/demo_following/demo_future_smp_v1/train_reference_coverage80/PHYSICAL_GROUP_2.png)、
[分组曲线4](../experiments/demo_following/demo_future_smp_v1/train_reference_coverage80/PHYSICAL_GROUP_3.png)、
[新匹配协议](../experiments/demo_following/demo_future_smp_v1/matched_train_coverage128/PROTOCOL.json)、
[槽位回读](../experiments/demo_following/demo_future_smp_v1/matched_train_coverage128/BANK_READBACK.json)、
[共同初始输入回读](../experiments/demo_following/demo_future_smp_v1/matched_train_coverage128/COMMON_INITIAL_INPUT_READBACK.json)、
[生成接入回读](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/generator_demo_geometry_preflight/TRACKER_COUPLING_READBACK.json)。

2026-09-13 最新验证结论：固定10条适配外示范已全部完成教师评估；8条完整执行通过，
source18／48分别268／298步箱体位置终止。两组完整model479在8条教师可用参考上的
16个冻结学生终点全部失败，**两组均0/8成功（全10条覆盖口径均0/10）**。匹配初态、
所有startup字段逐项一致，实际数值有限；全部物理曲线已检查。单一TRAIN配对和相机
通过不代表适配外native跟踪通过。

| 验证source | 零位置反馈步数 | 位置反馈步数 | 两组终止原因 |
|---|---:|---:|---|
| 8 | 257 | 258 | 箱体位置／箱体位置 |
| 28 | 107 | 194 | anchor位置／箱体位置 |
| 38 | 35 | 306 | anchor位置／箱体位置 |
| 58（预算326） | 115 | 111 | 箱体朝向／箱体朝向 |
| 68 | 73 | 245 | anchor位置／箱体朝向 |
| 78 | 65 | 249 | anchor位置／箱体朝向 |
| 88（预算256） | 126 | 160 | 箱体位置／箱体位置 |
| 98 | 122 | 186 | anchor位置／anchor位置 |

在7/8参考上位置反馈延长了执行，但不能改判为成功；部分共同早期窗口的箱体误差为0，
只是双方均未开始搬箱，不是该片段已有示范跟随。source8的固定同世界完整教师查询已
启动。首次因未裁取的原始教师bank长度不匹配，在物理0步被官方接口拒绝；失败组已
退出并保留。匹配bank使用原始source8的0:436帧与已核实的原始96参考时间线，所有
原始数值／物体／contact回读精确，未重采样或用实测关节替代参考。修正后的两组查询
已完成：257／258步旧trace所有字段和startup精确复现，两条完整890-D教师接口误差0，
教师与学生均冻结。学生动作对同状态教师的首帧MSE为0.03631／0.03247，所有有效帧
均值0.14884／0.07046；250帧后分别0.51189／0.47501，仅6／7个有效帧。原发布Tracker
的同状态反事实首帧MSE为0.16193，不能由动作标签误差推出其实际执行优劣。
完整发布510-D Tracker的同一8条参考实际基线已完成：360／259／290／271／337／
400／159／212步，只有source78完整通过，合计1/8（全10条覆盖1/10）。全部初态逐项
匹配，冻结无训练。不同输入接口和训练历史意味着这不是单因素遗忘估计；原发布模型
和适配模型的基础执行覆盖均有限。PGID356677已正常退出。

原始示范几何条件准备与完整Generator零更新检查也已完成：每组41个8×21片段，只含
原始数值示范箱体位姿和手／脚相对箱体位置，不含29-D目标关节、36-D命令、实际未来
或奖励。原始时钟逐项对应既有reference trace；不同示范条件在82个片段中均不同。
保留完整官方12×256 Generator、原normalizer和16步DDPM，只向既有target encoder
首层增加168个零输入列。全部旧状态精确保留，两组初始化相同，完整去噪／采样最大
差3.10e-6／1.38e-5，前后向有限且无参数更新；零／真条件新列梯度范数0／0.27137。
这只是8,319,216参数完整模型的项目条件适配接口验证，不是生成性能改善、SMP latent
或Zero-WAM复现。没有Generator训练；先从物理失败和同状态教师证据决定下一实验。

证据：[全部冻结验证](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/heldout_native_validation/RESULT.json)、
[完整物理回读](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/heldout_native_validation/PHYSICAL_READBACK.json)、
[物理曲线](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/heldout_native_validation/PHYSICAL_READBACK.png)、
[同世界教师查询](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/heldout_native_validation/SOURCE8_TEACHER_QUERY_RESULT.json)、
[发布Tracker固定基线协议](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/heldout_native_validation/RELEASED_FLOOR_PROTOCOL.json)、
[原始示范条件](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/reference_feedback/evaluation/original_demo_geometry_context/RESULT.json)、
[完整Generator零更新检查](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/generator_demo_geometry_preflight/RESULT.json)。

以下为此前TRAIN配对与阶段记录，运行状态以顶部最新核查为准。

2026-09-13 会话恢复：用户再次明确沿Zero-WAM与SMP的思想结合继续提升。当前路线是
保留本地Zero-WAM完整负结果，不盲目续训；先完成下述冻结Tracker的实际视觉和适配外
示范检查，再以完整官方SUGAR Generator／Tracker为基础研究示范条件未来片段，冻结
官方SMP承担任务级运动先验。当前Tracker使用已知数值参考未来，尚未证明生成未来
或人类视频条件能力。后续比较须分离未来条件与SMP贡献，实际执行动作仍为29-D，
真实rollout未来只作训练目标／评价，不进入部署输入。新Generator训练尚未启动。
293772/server29/step0的原／替代相机任务均已完成：各400个实际控制步、200帧H.264／
25fps／8秒，无重置，连续十帧抬箱通过，最大抬升0.549019／0.583387m。每段9张
首中末采样帧已检查，完整G1、抓取／抬箱／转身搬运可见，未见明显几何破坏；灰色地面
与背景限制深度参照。全部帧已解码，不声称逐帧人工看完。相机两组158帧所有前缀及
切换世界精确相同，242个共同后续帧的箱体匹配／另一参考比0.06266／0.15951，身体
0.13640／0.19302。相机轨迹与无相机轨迹不逐帧相同，只证明各自rollout。
此前五次初始化失败仍保留；节点变化后成功，未定位底层唯一根因，也未修改共享驱动。

冻结适配外native跟踪流水线`heldout_native_validation`已于2026-09-13 01:04:36在
同一保留step启动，PGID149332。十条固定验证示范先全部运行完整Refiner；通过的真实
轨迹用官方formatter转换后，两组model479各自冻结评估。首条source8的教师已完成
436步无重置并持续抬箱，其余仍在串行执行。八条学生预算400步，source58／88因真实
长度与36帧未来尾部预留，预算为326／256步；不补帧、不剔除教师失败，分别报告全部
10条覆盖和8条400步子集。这是适配外native跟踪前置检查，不是公共历史切换或全局
未见示范泛化；本轮训练更新为0。

证据：[相机配对回读](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/CAMERA_PAIR_READBACK.json)、
[原组实际视频](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/reference_feedback/video_server29_original/ACTUAL_WORLD.mp4)、
[替代实际视频](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/reference_feedback/video_server29_alternate/ACTUAL_WORLD.mp4)、
[固定验证协议](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/heldout_native_validation/PROTOCOL.json)。

通过物理检查的TRAIN配对已调用官方Tracker-to-Generator formatter导出：每组400条
实际记录、81条10Hz记录、41个不跨切换／不含补帧的8步片段。重建物体观测对现场
输入最大误差1.19e-6，记录command误差0，29-D实际动作和物体状态逐项等于原trace。
进一步回读发现，两组全部82个Generator目标片段与当前已知8×36参考条件包逐元素
相同，MSE与最大误差均为0。将该完整包直接喂给生成器再报告拟合，只验证答案复制，
不证明从示范语义／因果历史推断未来。后续条件设计须区别原始示范任务／箱体／交互
信息与输出命令；若提供完整目标命令，必须报告直接复制基线并限定为执行上界。
该发现不否定已知参考下的Tracker物理结果。本轮没有Generator或其他模型新训练。
见[官方转换](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/reference_feedback/evaluation/official_il_data/RESULT.json)、
[条件与目标逐元素回读](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/reference_feedback/evaluation/official_il_data/CONDITION_TARGET_READBACK.json)。

2026-09-13 最新：`matched_reference_feedback96`两组各96次在线更新、全部六个冻结
终点、精确回读与完整官方SMP评分均已完成。**位置反馈组通过这个TRAIN配对的400步
物理与独立参考跟随检查，对照替代组仍失败；新相机证据已补齐，未见示范结论仍待验证。**

| 完整官方846-D Tracker | 原／重复／替代实际步数 | 原／替代最大抬升 | 配对结果 |
|---|---|---|---|
| 零参考位置反馈 | 400／400／379 | 0.548086／0.601499m | 替代obj_pos终止 |
| 官方参考位置反馈 | 400／400／400 | 0.548602／0.582623m | 全部通过 |

两组完整初始参数、Adam及实际训练初态一致，原有状态／矩逐项保留、新48列及矩从0
开始；各147456个真实transition，累计BCPPO480、Adam9600，teacher／critic冻结。
完整601629参数actor与Adam终点有限，控制组新列始终为0，实验组新列实际学习。
各组重复轨迹、158帧所有公共前缀字段和切换世界精确相同。独立跟随在实验组242个
有效切换后帧全部通过：箱体匹配／另一参考比0.0750／0.1949，身体0.1442／0.2302。

在两组相同的220个有效帧上，实验／控制箱体误差比0.481／0.578，身体0.571／0.673；
300:378后期Y误差比0.422／0.336。物理、参考跟随和横向分解图均已查看。以上是
同一TRAIN配对和一个适配seed的结果；新增48-D输入依赖当前因果位姿定位，不是发布
Tracker原接口或既有触觉无物体状态合同的验证，也不是Zero-WAM生成式模型的成功。

完整官方SMP保持冻结。相同原组400帧／391个窗口、同batch及评分噪声，控制／实验
Carry能量0.201034／0.196981，全部窗口偏向Carry；这个任务先验不能代表具体示范距离。

第五次相机初始化诊断仅关闭renderer.asyncInit，仍在Kit任务场景创建前设备丢失／
退出139，实际物理步数0，无新视频；精确失败进程已退出。继续排查渲染并准备未参与
本轮适配的示范检查，不追加已完成96预算或把TRAIN通过称泛化。293435/server20由
Slurm记录为2026-09-12 23:56:04 CANCELLED by0（step0于23:56:34结束）；当前
保留GPU随后也变化：293437/server43由Slurm记录为2026-09-13 00:41:05 CANCELLED by0。
确认无活动作业后新申请293772，已获批并显式进入server29/step0，tmux
`curiosity_reference_feedback_h200_20260913`；标准配置的原组实际相机重试正在该新节点
执行。未修改系统驱动或共享缓存，未主动释放旧分配。

证据：[完整结果](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/RESULT.json)、
[完整终点](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/CHECKPOINT_READBACK.json)、
[匹配物理与横向分解](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/MATCHED_PHYSICAL_READBACK.json)、
[独立参考跟随](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/reference_feedback/evaluation/REFERENCE_FOLLOWING_READBACK.json)、
[冻结SMP](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/SMP_COMMON_PHASE/RESULT.json)、
[相机诊断](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/CAMERA_NEXT_DIAGNOSTIC.json)。

以下为此前诊断与训练过程；运行状态以上述最新核查为准。

2026-09-12 最新终点：`online_tracker_future128`完成完整官方OnPolicyRunner／BCPPO
128次在线更新及全部三个冻结评估，**整体仍失败，但两个示范的执行时间延长**。
完整父模型／Adam256继续到384，时钟5120→7680；64真实环境×24步×128，实际
196608个新transition。完整终点577053个actor参数及Adam均有限，teacher／critic冻结。

| 冻结未来Tracker | 原／重复／替代步数 | 原／替代最大抬升 | 终止原因 |
|---|---|---|---|
| 父model255 | 313／313／286 | 0.557283／0.611453m | 箱体朝向 |
| 在线128后model383 | 392／392／354 | 0.550936／0.577762m | 箱体位置 |

两示范均连续抬箱十帧。原／重复所有trace字段逐项相同，158帧公共前缀、startup和
切换世界相同。在195个共同有效切换后帧内，箱体匹配／另一参考误差比为0.242／0.473，
身体为0.376／0.577，四项均更接近所选参考；完整400步判据失败，不能称可靠跟随。
两张物理／跟随曲线已查看。逐轴检查显示末个有效帧箱体XYZ偏差：原组
(0.0796,0.2734,0.0228)m，替代组(-0.0719,0.2893,0.0116)m；主要为后期Y漂移。
当前／未来command包含局部参考速度而没有参考路径位置误差，尚待观测与教师查询
诊断，不能把该结构观察直接宣布为唯一根因。在线训练参考从episode起点选定，与
冻结评估158帧切换不同，在线／离线对比不是单因素因果结论。

零更新同世界教师查询已完成两组：所有旧trace／startup字段逐项相同，full890-D
教师两接口误差为0。有效帧动作MSE原／替代为0.017690／0.022163；原组前100帧
0.003871、350帧后0.079500，替代300:350为0.077778，后期偏差增加。

完整官方actor的离线反事实平移检查也已完成，未创建物理轨迹或更新参数：将实际
机器人／箱体及完整历史整体平移±0.05／±0.15m，固定所选参考。所有局部历史项按
官方公式平移不变，箱体局部位置独立重算；student输入／动作最大差约1.9e-6／4.8e-6。
±0.15m时完整教师标签MSE差约0.01585–0.01623，对理想相同输入的等权两标签拟合
下界约0.004。这支持参考路径平移不可观测这一结构限制，不证明反事实教师能实际
恢复或它是唯一失败原因。不是新物理数据，未把新字段偷偷加入现有控制器。

当前`matched_reference_feedback96`完整匹配流水线已实现并在293437/server43/step0
运行。两组实际零更新preflight全部通过：完整初始参数／Adam、原观测、初始动作及
64环境×24步的完整身体／root／动作／参考时钟逐项相同，均零终止；官方48-D位置
函数独立回读最大误差约1.04e-7。零组新增输入确实为0，实验组为实际官方位置项。
控制组已开始96次训练，随后实验组96次及全部六个冻结终点自动串行执行。
两组均保留完整官方798输入actor并增加相同48个零列，隐藏宽度512/256/128不变；
控制输入零，实验输入官方8帧anchor／object参考相对位置，明确依赖当前因果位姿定位。
同一model383／完整Adam384开始，各96在线更新到480，仍在500前BC阶段，随后全部
六个冻结终点。不是原发布Tracker接口或既有触觉无物体状态合同的验证。保留293437／
293435及全部checkpoint／Adam，不盲目追加原输入设置。相机故障仍未解决，无新视频。

证据：[训练与终点](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/RESULT.json)、
[保存模型回读](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/training/CHECKPOINT_READBACK.json)、
[物理回读](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/evaluation/READBACK.json)、
[独立跟随](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/evaluation/REFERENCE_FOLLOWING_READBACK.json)、
[逐轴与精确重复](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/evaluation/EXACT_REPLAY_AXIS_READBACK.json)。
[同状态教师](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/ON_STUDENT_STATE_READBACK.json)、
[平移输入诊断](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/reference_translation_diagnostic/RESULT.json)、
[下一匹配协议](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/PROTOCOL.json)、
[两组实际preflight](../experiments/demo_following/demo_future_smp_v1/matched_reference_feedback96/MATCHED_PREFLIGHT.json)。

以下为该在线实验的启动与零更新检查记录，运行状态以上述终点为准。

首次训练启动在保存初始checkpoint时遇到官方runner尚未初始化logger的接口错误；
保存下来的完整模型和Adam与父状态逐项相同，新增更新为0。失败目录保留于
`online_tracker_future128/failed_startup_logger`。适配层现在先调用官方幂等日志初始化，
再保存初始状态；已通过原保留step重新启动，launcher为`online_future128_pipeline_logger_fixed`。
已确认原生runner日志推进到累计update269，开始实际在线训练；最终物理判据仍待全部冻结终点。

在线零更新检查发现并修正了一个冷启动参考缓存问题：64个环境的相对身体参考缓存
初始全零，第一次控制步全触发ee_body_pos。仅按官方MotionCommand末尾同一公式
初始化派生参考缓存后，初始判错64→0，世界body/root、参考时钟及policy／future／
teacher／critic观测全部逐项不变。64环境×24步实测终止64→0，模型和完整Adam均未
更新；actor及890-D教师两条接口误差为零。这项修正针对原生在线冷启动，不推翻此前
已主动刷新参考的冻结评估或解释所有离线失败。先前两份preflight和所有失败保留。

证据：[初始化修正](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/preflight_initialized/REFERENCE_CACHE_INITIALIZATION.json)、
[实际零更新检查](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/preflight_initialized/RESULT.json)、
[在线实验协议](../experiments/demo_following/demo_future_smp_v1/online_tracker_future128/PROTOCOL.json)。

2026-09-12 最新结果：`matched_tracker_coverage128`两组训练及六个冻结终点全部完成，
**数据覆盖方案整体失败且实验组提前退步**。两组完整初始模型／Adam状态／共同数据初始
拟合逐项相同，各新增128次更新（256→384），optimizer clock5120→7680，各196608次
记录使用，教师和critic逐项冻结。原模型、矩与全部新终点保留。

| 同一未来输入完整Tracker | 共同四轨迹动作MSE | 原／重复／替代实际步数 | 物理结论 |
|---|---:|---:|---|
| 重复旧轨迹控制组 | 0.0211899 | 298／298／233 | 箱体位置终止，替代未持续抬箱 |
| 加入真实恢复轨迹组 | 0.0000893632 | 119／119／119 | 均anchor_pos终止，未到158切换点 |

加入恢复轨迹组通过原固定数据拟合标准；新两条轨迹误差为控制组的0.00258／0.00214，
旧两条误差为控制组的2.34／2.23倍，但仍低于原绝对拟合阈值。拟合曲线与两组实际物理
曲线均已查看。实际实验组没有到达选择替代示范的时点，故其具体示范跟随误差比值为
不可计算，不能从0个有效帧报告成功或条件差异。不直接延长这个离线设置。

新终点的早期教师查询也已完成：119步原轨迹和startup逐项复现，初始训练状态及
完整890-D教师输入／动作接口逐项相同。首帧动作MSE5.28e-6，全段同状态教师MSE
0.046847；0:20／20:50／50:100／100:119帧分别为0.00869／0.01890／0.06983／0.07067。
这个结果支持在线实际状态分布需要进一步覆盖，不证明唯一根因。未追加训练。
冻结完整官方SMP评分已完成。同一前118帧／109个窗口、相同batch形状和评分噪声，
控制／实验组Carry能量为0.114919／0.134301，两组109/109窗口都偏向Carry。任务先验
对运动偏离有响应，但不能当作早期身体跟踪或搬箱成功判据，也不是具体示范距离。

相机在两节点及CUDA mask／官方默认渲染设置共四次尝试均于Kit渲染初始化失败，
无新任务场景或相机rollout。已保存四次失败，停止重复同样初始化；相机问题继续独立
排查。已有完整无相机训练和行为证据不因此改判，正面跟随或Generator推进仍须实际视觉。
已接入完整官方OnPolicyRunner／BCPPO和已知未来参考观察组，零更新检查与启动修正
见顶部最新状态。原生训练从episode起点选择整条参考，与冻结评估158帧才切换的时间
安排不同，明确作为探索性在线状态覆盖诊断。

证据：[匹配初始化](../experiments/demo_following/demo_future_smp_v1/matched_tracker_coverage128/MATCHED_INITIALIZATION.json)、
[共同数据拟合](../experiments/demo_following/demo_future_smp_v1/matched_tracker_coverage128/ACTION_FIT_READBACK.json)、
[完整物理回读](../experiments/demo_following/demo_future_smp_v1/matched_tracker_coverage128/MATCHED_PHYSICAL_READBACK.json)。

2026-09-12 最新冻结物理结论：`matched_tracker_future_plan/evaluation_preconverted_warmed`
六个终点已全部完成，整体**未通过**；以下最新结果覆盖后文过程中的“正在执行”。

| 完整 Tracker 输入 | 原96／重复步数 | 替代90步数 | 原／替代最大抬升 | 终止原因 |
|---|---:|---:|---:|---|
| 当前输入＋零未来计划 | 400／400 | 233 | 0.597250／0.005051 m | 替代箱体位置 |
| 当前输入＋8×36未来计划 | 313／313 | 286 | 0.557283／0.611453 m | 两示范均箱体朝向 |

未来组两示范均连续抬箱十帧，原／重复轨迹逐项相同，158帧公共前缀和切换世界一致。
在共同有效的127个切换后帧内，两条轨迹的箱体／身体误差均更偏向所选示范：匹配／
另一参考误差比值，箱体为0.214／0.486，身体为0.586／0.638。控制组替代示范未抬箱，
箱体也未更接近所选参考。但未来组没有保留原示范400步行为，两个终止均不通过原判据，
故不能称可靠跟随或泛化成功。两组物理曲线已查看；尚无该匹配终点的新相机证据。

运行修复采用官方URDF转换器生成的完整G1 USD，保留所有共同spawn／刚体／关节／碰撞
参数。400步旧教师支持轨迹和startup全部字段逐项复现，原示范现场未来计划与参考派生
计划最大差1.43e-6。进程内URDF转换路径复现Vulkan故障，跳过该转换恢复完整物理；
更底层唯一原因未确定。未来计划形状的首次数值路径出现2.98e-8输入差、5.96e-8动作差；
加入只读预热后连续三次输入／动作精确稳定，严格参考恢复检查保持。控制组三条轨迹
加预热前后所有记录字段精确相同，完整匹配评估没有新增训练更新。

实际学生状态诊断也已完成：原／替代补录分别逐项复现313／286步终点，startup相同，
完整890-D教师输入／动作的两条官方接口逐项相同，现场未来计划与训练计划最大差9.54e-7。
初始训练状态／教师标签逐项相同，第0帧动作MSE仅8.32e-6；独立执行后误差快速增长，
两条实际学生轨迹的同世界教师动作MSE为0.066582／0.122793。动作与身体偏差曲线已查看。
这支持实际状态覆盖不足的诊断方向，不是唯一根因证明，也不是跨状态动作对比。

固定第100帧实际交给冻结Refiner的两条恢复均通过400步，抬升0.544441／0.580224m。
所有接管前记录、接管世界和两组158帧公共前缀逐项相同；100步后实际执行动作就是教师
动作，未重置或重写状态。此结果只证明这两条教师支持轨迹可用，学生独立终点仍然失败。
失败学生相机录制首次在渲染版Kit初始化、任务场景创建前发生Vulkan设备丢失，
子进程退出-11且已确认退出，没有物理动作或视频。清除CUDA mask仍在同一渲染初始化阶段失败。现于另一个已保留
节点293435/server20串行执行同一相机任务，输出另存；293437保留空闲。无相机评估和
教师恢复的完整结果不受影响。

下一训练协议已预先写定但尚未实现／启动：两组从同一完整未来Tracker model255和
Adam矩／时钟开始，各128次BCPPO更新到384，仍在原500步BC边界以内。控制组重复
旧轨迹，实验组加入上述实际恢复轨迹；等量1536记录项／196608次使用，唯一差别是状态
覆盖。父checkpoint没有RNG状态，因此两组明确使用相同新seed272042，不称精确随机
续训。仍须原／重复／替代400步独立冻结评估，拟合通过不能代替物理通过。

证据：[匹配物理回读](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/MATCHED_PHYSICAL_READBACK.json)、
[未来组独立跟随指标](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/future_plan/evaluation_preconverted_warmed/REFERENCE_FOLLOWING_READBACK.json)、
[完整400步加载等价](../experiments/demo_following/demo_future_smp_v1/tracker_bcppo_refined_pair_low_exploration/PRECONVERTED_REPLAY_READBACK.json)、
[只读预热等价](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/WARMUP_CONTROL_REPLAY_READBACK.json)、
[实际学生状态回读](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/ON_STUDENT_STATE_READBACK.json)、
[实际教师恢复回读](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/FUTURE_TRACKER_RECOVERY_READBACK.json)、
[下一匹配数据覆盖协议](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/MATCHED_COVERAGE_NEXT_PROTOCOL.json)。

2026-09-12 最新推进：`matched_tracker_future_plan`两组各256次更新已完成，完整模型与
Adam状态保留。两组均为完整官方ActorCritic，仅actor首层增加相同288个输入列并置零；
初始全部参数逐项相同，真实样本的初始动作精确复现发布Tracker。控制组输入零未来计划，
实验组输入8×36已知参考计划，其余数据／seed／预算／FP32优化相同。各使用两条真实
教师支持轨迹的768个transition记录，不是新在线rollout或32个真实环境。

- 当前输入组／未来计划组整体教师动作MSE为`0.000336606 / 0.0000824001`。
  切换帧两项MSE分别为`0.0424264/0.1277163`与`0.000093250/0.000027463`，
  未来组通过完整拟合标准，当前组失败；各5120次optimizer clock、教师与critic逐项冻结。
  这支持固定数据上未来意图信息有效，不证明物理执行或泛化。
- 未来组置零／换成同phase另一示范的未来计划后，整体MSE升为`0.0147088/0.00669570`；
  错示范切换帧误差为`0.314252/0.319996`。动作差值图已检查，未来组跟随两条教师
  动作差值，当前组因相同输入而差值为零。
- 计划来自已知参考bank与当时选择／时钟，直接调用官方纯张量旋转函数。两条实际当前
  command及替代组400个现场8步计划的最大回读误差为`1.43e-6`，低于原`1e-5`标准。
  原组完整计划是参考派生，未现场记录；未伪造物理未来或新增轨迹。两组用相同派生方式。
- 新分配为`293437/server43/step0`，tmux`curiosity_future_plan_physx_20260912`；
  `293435/server20/step0`也保留。两节点的PhysX初始化均发生Vulkan`ERROR_DEVICE_LOST`，
  首个新冻结终点尚未开始rollout；这些是运行环境失败，不是新的物理策略终点。关闭
  Fabric GPU互操作和在单卡cgroup中清除CUDA mask均未解决。纯CUDA完整训练正常完成，
  server20 ECC计数为零，尚未定位唯一原因。所有已失败子进程组精确终止，分配未取消。
  停用完整Fabric读回也失败；官方空场景已完成16次step调用，尚待核实关闭过程与
  含任务资产场景的区别，不能据此宣称SUGAR物理环境恢复。
  不追加训练或称跟随成功。

证据：[匹配初始化](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/MATCHED_INITIALIZATION.json)、
[动作拟合回读](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/ACTION_FIT_READBACK.json)、
[参考计划接口](../experiments/demo_following/demo_future_smp_v1/tracker_bcppo_refined_pair_low_exploration/KNOWN_REFERENCE_PLAN_AUDIT.json)、
[物理运行失败](../experiments/demo_following/demo_future_smp_v1/matched_tracker_future_plan/PHYSICAL_RUNTIME_FAILURE.json)。

2026-09-12 会话续接核对（以下最新结果覆盖后文过程记录中的“正在运行”）：

- 低初始探索噪声的完整 Tracker 已完成64更新，冻结终点125步因`anchor_pos`终止，
  同样失败。第100步无重置切换至冻结 Refiner 后可完成400步，说明该实际状态仍可恢复。
- 使用这条真实教师支持轨迹的384个transition，完整官方Tracker／原BCPPO做256次
  固定数据BC更新；这是离线拟合诊断。FP32终点教师动作MSE为0.00011443，拟合通过。
  初次TF32教师批量精度检查未通过且零更新退出，失败产物保留；不把精度差异当作唯一原因。
- 冻结model255独立控制的原96组与重复组均完成400步，最大抬升0.535911m；切换90组
  第231步因`obj_pos`终止，仅抬升0.039034m。公共158帧及切换世界逐项相同，重复轨迹
  差异为零；整体配对判据失败。已检查物理曲线，尚无该终点的新相机视频或泛化证据。
- 第二条真实教师支持轨迹也已完成400步，抬升0.589765m。两组均在100步由同一个
  Tracker交给同一个冻结Refiner，158步选择不同示范。切换时510-D Tracker输入和当前
  36-D参考command逐项相同，教师29-D动作MSE差为0.316414；对这两个等权标签，任何
  相同输入的确定性输出至少有0.079104的平均MSE。此为一个实际TRAIN分叉点的输入
  歧义，不证明全任务不可控。未来8×36参考计划确实不同；替代组现场记录与参考command
  回读误差为零，原组现场计划提取仍须补查。
- 下一项有界实验：先补齐原组因果未来计划接口检查，再以两条相同真实轨迹、相同完整
  官方Tracker初始化／预算，比较当前command与额外未来计划输入。通过独立冻结PhysX
  配对、接触／参考跟随与实际视频后，再推进完整官方Generator的示范条件未来监督及
  冻结SMP先验组合。尚未实现未来输入适配，也未训练新Generator；不追加旧Zero-WAM预算。
- 资源核查：293231被Slurm记录为`CANCELLED by 0`，结束于2026-09-12 20:21:04，
  step0于20:21:34结束；当前`squeue -j 293231`为空。最后采集已有完整RESULT与退出
  状态，不能再把旧tmux视为有效GPU分配。进一步GPU工作需先取得并进入新的计算step。

证据：[完整Tracker配对终点](../experiments/demo_following/demo_future_smp_v1/tracker_teacher_supported_fixed_replay_bc_fp32/evaluation/READBACK.json)、
[当前输入／未来目标审计](../experiments/demo_following/demo_future_smp_v1/tracker_bcppo_refined_pair_low_exploration/CURRENT_INPUT_FUTURE_TARGET_AUDIT.json)。

以下为本轮按阶段保留的过程记录，运行状态以上述最新核查为准。

2026-09-12：四轮匹配预测器实验均已完成，整体未通过。第四轮相对直接预测网络的远期
误差改善约 14–32%，但全部 12 个未来几何优于解析对照的判据失败；Kick 1.4s 几何误差
反而增加约 22–24%。最新审计发现现有四种评分示范共享同一条实际未来轨迹，不能当作
更换执行示范后的条件未来监督。冻结 Refiner 的首次实际 PhysX 配对诊断已完成，
公共前缀精确匹配，但 Carry45→96 直接切换因参考状态不兼容而失败。Native96已完成
450帧零重置；按固定phase可行性选择的96→90三组匹配诊断已通过数据配对检查，
实际视频与独立参考跟随回读也已完成。官方完整Generator／Tracker的checkpoint审计已
通过。发布Tracker在原始参考上102步终止，使用官方格式的Refiner实际motion后能抬箱，
但330步箱体位置终止，未通过400步标准。对应原始教师示范bank的两组400步兼容性检查
已通过。首个完整Tracker64更新BCPPO终点退步到217步、未抬箱；同世界状态的教师接口
检查逐项通过。现启动只改变初始探索噪声的匹配64更新实验，尚无新方法物理跟随成功结论。
保留 H200 为 `293231/server35/step0`，tmux 为 `curiosity_demo_future_20260912`。
既有 11,386,010 参数预测器主干和全部参数保留，只添加 30,030 参数的三个未来任务头组；
这是项目预测器的辅助监督适配，不是新的官方 SMP／Zero-WAM 实现或生成式世界模型。
官方 MimicKit/TinyMDM checkpoint 已严格加载，冻结 ESM/SDS API 检查通过。

- step96 收尾：全部八例／64 张生成帧已逐帧检查，严重块状破坏已消失，但姿态、箱体
  对齐和时序偏差仍在；slot03／04 的 GT 区间本身接近静止。64 张冻结 VAE 对照全部与
  此前已检查的 step64 对照解码 RGB 相同。见[视觉记录](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911_step96/VISUAL_INSPECTION.json)。
- 数据：使用原始 PhysX corpus 和原 numeric demo，未来结束时距为 `0.2/0.4/0.8/1.4 s`；
  160／20／19 motion 的共同有效历史数为 `18542/2137/2085`，历史和未来均不跨 reset。
  最近时距重新生成的旧三种条件标签与原标签的最大绝对差不足 `1e-6`。倒放参考在完整
  时间线上反转后重新计算速度符号和剩余接触时长；它是诊断条件，不默认物理可行。
- 冻结预测器诊断：validation/test 的最近时距 full motion-macro MAE 为
  `0.103665/0.097163`；去掉 demo 后为 `0.197851/0.193171`，同任务其他 motion
  的历史替换后为 `0.152809/0.127679`。两类输入都被使用。同任务条件差值的符号一致率
  仅 `0.623670/0.604246`。指标在模型的 `log1p(target/scale)` 空间计算，与旧报告的
  decoded normalized MAE 不是同一指标，不据此声称比旧模型更好。
- 匹配实验：`aux_detached` 与 `aux_attached` 使用相同完整预训练权重、四种条件、
  四时距标签、mini-batch 次序、dropout seed、fresh AdamW、`lr=1e-5` 和固定八轮预算。
  唯一区别是远期任务头是否向共享主干反传；控制组的远期头也训练。主干／辅助头分别
  按 norm1 裁剪，防止辅助头裁剪范数直接改变主干有效步长。主干始终接受最近时距监督。
  原预测器没有可恢复的 optimizer，因此两组都从相同新 optimizer 开始，不声称精确续训。
- 真实 checkpoint／真实 TRAIN batch 的零更新 preflight 通过：两组初值和前向精确相同，
  最近输出精确复现原模型；控制组远期梯度到主干严格为零，实验组为非零，两组未来头
  均可训练。已确认控制组首个更新实际应用，后续步数和终点以 trace／RESULT 为准。

自动判据与后续分支见[实验协议](../experiments/demo_following/demo_future_smp_v1/PROTOCOL.json)。
先完成两组终点和示范／历史消融，要求分任务、分 split 的远期 MAE 与同任务条件差值误差
至少改善 5%，最近时距退步不超过 2%；未通过先诊断，不把预测器进展当作物理跟随。
旧 validation/test 已用于历史开发，本次是训练 motion-disjoint 的探索性比较，并非全新
未看过的确认性测试。后续策略验证仍须独立真实行为指标和匹配训练／评估种子。

首个匹配终点：两组各恰好 4,640 次真实更新、593,344 次样本使用，权重和 AdamW 均保存，
全部 validation/test 与四种示范／历史输入模式完成评估。attached/detached 比值为：

| 分组 | 远期 MAE | 最近时距 MAE | 同任务远期条件差值误差 |
|---|---:|---:|---:|
| validation Carry | 0.937187 | 1.081059 | 0.973391 |
| validation Kick | 0.941064 | 1.014847 | 0.997616 |
| test Carry | 0.909812 | 1.122577 | 0.976744 |
| test Kick | 0.927842 | 1.021945 | 1.006310 |

远期平均预测改善，但所有同任务差值标准均失败，三个分组近期退步超标，不能接纳为
更好的具体示范表示。Carry test 近期左右手接触误差分别为控制组的 `2.20/2.24` 倍。
只读梯度诊断在八个固定真实 TRAIN batch 上完成：不含倒放时，demo projection 的
近期／远期梯度 7/8 为负余弦，均值 `-0.2643`；完整主干总梯度均为正余弦，均值
`0.1790`。局部梯度竞争是证据，不是唯一原因证明。
见[首轮结果](../experiments/demo_following/demo_future_smp_v1/matched_future_supervision/RESULT.json)、
[逐目标回读](../experiments/demo_following/demo_future_smp_v1/matched_future_supervision/READBACK.json)、
[梯度诊断](../experiments/demo_following/demo_future_smp_v1/matched_future_supervision/GRADIENT_CONFLICT_PROBE.json)。

第二个有界匹配实验从同一 attached step4640 的完整权重／AdamW 状态开始，两组都增加
对已完成 detached checkpoint 最近时距输出的固定蒸馏约束，权重 5；只有实验组增加
同一历史下 correct／same-task alternate 条件差值的监督。每个 batch 保留四条件成组，
差值使用 TRAIN-only 标准差（下限 0.05）归一化，目标保留真实正负号；不强制原 demo
必然胜出。两组各固定四轮，原科学判据保持；失败不直接延长该设置预算。该实验仍是
预测器适配，尚未启动策略训练或物理 rollout。

第二轮实际终点：两组各追加恰好 2,320 次更新，累计 step6960；初值、AdamW moments／
clocks 精确一致，新增样本使用各 296,672 次。supervised/control 比值如下：

| 分组 | 远期 MAE | 最近时距 MAE | 同任务远期条件差值误差 |
|---|---:|---:|---:|
| validation Carry | 0.971255 | 1.022431 | 0.894486 |
| validation Kick | 0.997114 | 0.996760 | 0.955873 |
| test Carry | 0.965477 | 1.028607 | 0.921765 |
| test Kick | 0.979491 | 0.981630 | 0.940836 |

同任务差值误差改善约 4.4–10.6%，但所有远期整体改善标准失败，Carry 近期退步仍超标。
两轮各 Carry／Kick 的固定样本差值图均已查看，Kick 预测仍明显抹平实际时序差值。
见[第二轮结果](../experiments/demo_following/demo_future_smp_v1/matched_same_task_gap/RESULT.json)。

随后完成的零更新诊断改变了下一步优先级：

- 当前状态保持对照排除含未来信息的剩余接触时长，使用当前真实 body/object/contact
  测量值；在四时距、两任务和两 split 上均优于训练模型。它额外使用机体位姿，因此
  是信息接口诊断，不能称为由 121-D 输入部署的模型或完整动力学基线。
- 在 12 个真实记录上构造共旋转 90° 的坐标例子，机器人相对箱体几何误差变化低于
  `1.2e-7`，固定 numeric demo 下的世界坐标标签却全部改变。完整 121-D 的旋转不变性
  依据源码公式推导，未声称逐项重建了输入；这些也不是新物理 rollout 或已观测重复输入。
  该结果指出参考与 rollout 世界坐标之间缺少显式关系，不证明它是全部失败的唯一原因。
- 官方 `compute_disc_obs` 与已有 G1/box adapter 在两示范的六个窗口上通过共同 yaw／
  平移不变性检查，仍为 `10 × 216` 特征。原 199-motion corpus 缺少其所需的完整 body
  quaternion／velocity、joint position／velocity／names，不能直接声称已在这些轨迹上
  运行该精确 SMP 特征，也不能伪造缺失状态。
- 更直接的标签 bug：旧 `reference_contact_roles` 用机体索引 `6/12` 判断 Kick 左右脚；
  现有源顺序和实际 trace 名称均对应髋部／膝部，左右踝部为 `21/22`。99 个 Kick 中
  96 个有角色变化，12,857 个 active reference frame 中 5,050 个改变，即 `39.278%`。
  该问题影响参考脚部 contact／duration 和相应监督；实际 PhysX 接触力与执行动作没有
  被这些参考标签替代。此前模型对旧标签的拟合／校准统计仍可复现，但不能直接用来证明
  修正语义下的脚部预测有效。本轮两组比较也只成立于其原错误标签定义。

证据：[当前状态诊断](../experiments/demo_following/demo_future_smp_v1/CURRENT_STATE_INFORMATION_PROBE.json)、
[坐标诊断](../experiments/demo_following/demo_future_smp_v1/COORDINATE_CONTRACT_PROBE.json)、
[官方特征与标签审计](../experiments/demo_following/demo_future_smp_v1/FEATURE_CONTRACT_AUDIT.json)。
修正标签与坐标数据已完成，见下方第三轮说明。新标签需要 TRAIN-only 归一化与
重新训练／校准；不直接替换冻结奖励输入。
原模型、优化器、数据与原标签保留，两个失败设置均不直接追加预算。

第三轮使用单独的 `dataset_reference_roles_v2`／`dataset_heading_local_v3`，原标签与
所有结果保留。199 个源示范首帧、每帧 34 条官方 URDF 关节原点关系的最大误差为
`3.32e-7 m`，机体顺序与全部实际 trace 名称一致；全帧刚体一致性假设未通过，最大残差
`0.018215 m`，此现象已记录，不用运动学替换原数据。修正脚部标签影响 TRAIN／validation／
test 的 `11010/1394/880` 个基础历史，所有非脚部目标逐项相同。

局部坐标适配对每个参考／实际十帧窗口，使用其首帧 pelvis 朝向旋转四类连续特征；调用
官方 MimicKit heading／quaternion 函数，但不是官方末帧 torso 朝向的 216-D SMP 表示。
事件标签逐项保持，构造的 yaw 检查通过。它保留局部运动与事件，**不衡量绝对世界路线或
绝对朝向跟随**；未来实际姿态只用于监督标签，部署输入仍是原 121-D 因果历史。
归一化只使用 TRAIN；target scale 改为条件／时距上的中位数和明确下限，是本轮局部方法
变体，不能与旧 p90 归一化的 MAE 数字直接比较。两组归一化完全一致。

第三轮 `matched_corrected_canonical` 从同一完整原 seed271303 权重开始，替换上述
TRAIN-only 归一化 buffer，采用相同 fresh AdamW、八轮和所有输入条件；唯一组间差别
仍是未来辅助梯度是否进入共享主干。实际 preflight 的控制组辅助主干梯度严格为零、
实验组非零；初值和前向逐项相同。两组各恰好 4,640 次更新、593,344 次样本使用，
全部终点、消融、更新回读与固定样本图均完成。原匹配科学判据保持，不直接追加失败预算。
完整权重与 optimizer 保留，官方 SMP 保持冻结，尚无策略训练或物理跟随结果。

第三轮 attached/detached 比值：

| 分组 | 远期 MAE | 最近时距 MAE | 同任务远期条件差值误差 |
|---|---:|---:|---:|
| validation Carry | 0.951110 | 1.100749 | 0.964161 |
| validation Kick | 0.951332 | 1.016558 | 1.003363 |
| test Carry | 0.936343 | 1.162337 | 0.985043 |
| test Kick | 0.935611 | 1.027976 | 0.998105 |

所有同任务差值标准失败，三个近期保持标准失败；test Kick 的历史消融也未达到 2% 退化。
固定 Carry／Kick source008 图已查看：Carry 的实际差值幅度明显被低估，Kick 的预测
几乎恒为零。训练集回读同样明显差于当前状态对照，不能只归因于未见过的示范。

随后直接调用官方 `URDFCharModel` 做了 139,300 个实际帧的当前几何恢复审计，输入只用
121-D 当前观测、官方 URDF 和 nominal joint defaults。1e-3 精确误差标准失败，位置平均
误差约几毫米、最大约两厘米。源码 startup default-joint 随机偏移范围为 ±0.01rad，旧
corpus 未保存这项校准；它是残差的可检验解释，不把已失败的精确恢复改写为通过。

几何保持对照另外排除了全部 contact／duration／regime，仅评估四类连续几何。原对照
使用已知 episode duration 推进未来 phase；这虽然因果，但比网络的单个当前 phase
多了时钟速率信息。因此又完成了严格 current-phase-only 对照：四个头均只使用网络
已有的当前 phase，不读取未来 phase 或 duration。它在所有 split／task 的 0.2／0.4s
上仍优于网络；例如 test Carry／Kick 的 0.2s 误差分别为网络的 `0.5580/0.1945`，0.4s
为 `0.8379/0.5392`。0.8／1.4s 并非全部优于网络，不能沿用“所有时距均优于网络”的
强说法。差异说明当前可恢复几何路径值得显式保留，未来时钟速率也应作为明确的因果
接口变量，而不是隐含于标签生成器。

下一轮将预声明相同完整预测器、相同初始化与预算，测试保留当前几何后学习未来残差；
时钟速率若加入，必须两组都提供，且只来自已知 episode clock。官方 SMP 仍只负责运动
先验，现有 11.386M 网络仍是项目的条件预测器，不称为官方 Zero-WAM／SMP latent。
证据：[第三轮终点](../experiments/demo_following/demo_future_smp_v1/matched_corrected_canonical/RESULT.json)、
[训练集／信息诊断](../experiments/demo_following/demo_future_smp_v1/matched_corrected_canonical/FIT_INFORMATION_PROBE.json)、
[精确恢复审计](../experiments/demo_following/demo_future_smp_v1/OBSERVABLE_GEOMETRY_PROBE.json)、
[相同当前 phase 的几何对照](../experiments/demo_following/demo_future_smp_v1/OBSERVABLE_GEOMETRY_HOLD_PROBE_CURRENT_PHASE_ONLY.json)。

第四轮 `matched_geometry_residual` 已实现并通过零更新 preflight。两组都使用完整
canonical attached 权重，保留全部 Transformer、方差与事件头，只将四类连续几何目标的
mean-output 行同时置零，并加入相同零初始化的 768 参数 clock-rate projection；采用
相同 fresh AdamW 和八轮预算。实验组在几何输出上加上官方 FK 得到的当前状态基线，
控制组加零，因此两组权重、方差和非几何初始前向相同，**几何初始输出按实验定义不同**。
不把不同初始几何输出隐瞒为完全相同前向，也不声称是 optimizer 精确续训。

preflight 已确认：实验组初始几何输出精确等于独立当前状态对照；去掉 demo 后该路径
不再携带选定 demo 身份；去掉历史会改变几何路径；两组完整主干／几何头均有有限梯度。
已知 source duration 推导的因果 clock rate 在所有 TRAIN／validation／test 上逐项复现
原 phase 数组和参考窗口，不输入未来实际状态。除原匹配判据外，还要求实验组在 0.4／
0.8／1.4s 上比同输入几何基线改善至少 5%，0.2s 退步不超过 2%，才能推进物理准备。
第四轮现已完成：每组恰好 4,640 次更新、593,344 次样本使用，完整权重和 AdamW 保留；
终点、消融、四张固定样本图与零更新残差信息审计均完成。原 20 项网络相对比较检查通过，
但全部 12 项未来几何超过解析对照至少 5% 的检查失败，因此整体失败。test Carry／Kick
远期总体误差分别为直接网络的 0.861865／0.676366，同任务差值误差为 0.869091／0.545717。
然而修正能量的 97.7–99.6% 在同任务两个参考间是共同的，具体示范区分主要来自解析路径。
MSE 与 MAE 的不同表现已记录，不回改预声明 MAE 判据，也不直接延长这个设置。

源码回读确认：原 corpus 的官方 Generator+Tracker 根据当前状态、最终箱体目标等生成
动作，选定源 motion 提供初始化和末帧目标；GeneratorObs 没有完整选定示范的时间线输入。
多条件标签构建对同一实际未来更换评分参考，这是兼容性监督，不能证明条件执行未来变化。
官方 Refiner 的 policy 直接读取未来参考关节、anchor 和箱体状态，适合进一步审计具体示范
执行能力。现有 step10000 权重已由归档恢复记录确认是项目按官方 SUGAR 流水线训练的产物，
并非作者公开发布权重；来源记录见 archived TODO 的 2026-08-21 recovery 条目。
随后执行的新配对 PhysX 数据诊断见下文。原始语料全部保留。

证据：[第四轮终点](../experiments/demo_following/demo_future_smp_v1/matched_geometry_residual/RESULT.json)、
[残差信息审计](../experiments/demo_following/demo_future_smp_v1/matched_geometry_residual/RESIDUAL_INFORMATION_AUDIT.json)、
[数据条件与 Refiner 源码审计](../experiments/demo_following/demo_future_smp_v1/DATA_CONDITIONING_AUDIT.json)。

冻结 Refiner 实际数据诊断现已完成首组。保留官方 890-D observation、完整
`512/256/128` ActorCritic 和项目按官方流水线训练的 step10000 权重；只有评估条件固定为
friction1／restitution0／mass1、无外力推动和 reset 位姿／关节扰动。原示范45和重复组各
450 帧零重置，最大抬升 0.708140m，切换点后双手实际接触比例 0.7984。公共前197帧的
完整状态／动作／教师观测和全部已记录 startup 参数逐项相同，切换帧197的世界状态也
精确相同；因此共同历史得到了直接验证。

但替代96组在第198次 transition 即终止，没有任何可接纳的切换后未来。执行替代动作
之前，箱体参考距离已为1.803093m、torso参考距离为1.452345m，原45分别为0.017528m／
0.055500m；原位置终止阈值为0.30m。它证明参考当前状态不兼容，不能把 reset 后画面或
动作差异算成不同示范执行成功，也不能据此断言实际摔倒。终止 transition 的 post-state
已排除。物理回读图已检查，原45与重复组重合，替代组在切换前结束。

本次记录了完整35机体位姿／速度、绝对关节位置／速度、关节名称／默认值、物体状态、
实际29-D动作和命名手脚过滤接触力。按实际名称映射后，官方 `compute_disc_obs` 与已有
G1/box adapter 已在真实轨迹上生成 `10×216` 特征，冻结官方 TinyMDM API 已实际评分。
45／重复各441窗口全部更偏向Carry类；96切换组只有公共前缀的188窗口，没有切换后评分。
这说明特征接口可用，不能称作具体示范距离或跟随收益；完整模型状态逐项保持。

随后固定phase197/659，仅用一次因果 yaw／水平平移对齐参考、保留实际世界与原始文件的
只读诊断筛查80个TRAIN Carry参考。没有替代参考满足全部原始位置／朝向约束；没有为了
取得候选而放宽阈值。完整原45/96失败保留。下一步按自动分支从96自身初始状态运行同一
冻结教师450帧，区分 native 示范执行能力与公共历史切换可行性；Native96现已完成450帧零重置，最大抬升0.543218m，不能当作同初始状态策略比较。
随后固定native96的frame158/530筛查同一80个TRAIN参考，有5个满足保守当前位姿约束且
2.8秒参考箱体位移差至少0.15m；按当前兼容性最大归一化误差最小选中90。新的96→90
原示范／重复／替代组已串行启动，各组在frame158应用相同一次因果yaw和箱体XY参考
对齐规则，保留高度和完整未来相对运动，不修改实际世界；仍保留原终止阈值和450帧预算。
它是探索性TRAIN数据可行性选择，不能当作独立未见示范验证。

96/90匹配数据终点已完成：三组各450帧零重置，教师全程冻结。公共158帧的完整状态、
动作、教师观测、startup参数和切换瞬间世界状态逐项相同。重复组箱体坐标轨迹RMS差为0，
替代组为0.221341m。原示范／替代的切换后参考箱体坐标RMSE为0.077126／0.066052m，
双手实际接触比例0.8801／0.9247；两者均连续抬升至少十帧。它首次提供了本路线中同一
实际历史下不同示范导致不同实际未来的完整数据。

物理回读图已查看：替代组较早抬箱与放下，重复组和原组重合。两实际轨迹×两参考的交叉几何回读已通过：箱体匹配／不匹配参考误差比为
0.374956／0.268052，14机体位置为0.317700／0.241346。所有比较共享292个有效时刻、
相同固定phase和坐标规则，不做窗口最小化；该指标属于探索性后续诊断，不冒充原主判据
或未见示范评估。完整实际状态经官方SMP特征／现有冻结先验API评分，各组
441窗口均更偏向Carry类；先验状态逐项保持，但类偏好不是具体示范距离。实际相机视频各完成450控制帧、零重置；每条225帧／25fps／9秒H264均完整解码，
各五个固定时刻的画面及全部统计图已查看，机器人和箱体完整可见。视频只证明各自运行，
不宣称重放无相机统计轨迹。两者抬箱高度仍系统性低于参考，绝对跟随并不精确。

最新证据：[96/90配对结果](../experiments/demo_following/demo_future_smp_v1/refiner_aligned96_90_feasibility/READBACK.json)、
[实际先验评分](../experiments/demo_following/demo_future_smp_v1/refiner_aligned96_90_feasibility/official_smp/RESULT.json)。
[独立参考跟随矩阵](../experiments/demo_following/demo_future_smp_v1/refiner_aligned96_90_feasibility/REFERENCE_FOLLOWING_READBACK.json)、
[视觉检查](../experiments/demo_following/demo_future_smp_v1/refiner_aligned96_90_feasibility/VISUAL_INSPECTION.json)、
[示范96实际视频](../experiments/demo_following/demo_future_smp_v1/refiner_aligned96_90_feasibility/video_original/ACTUAL_WORLD.mp4)、
[示范90实际视频](../experiments/demo_following/demo_future_smp_v1/refiner_aligned96_90_feasibility/video_alternate/ACTUAL_WORLD.mp4)。

下一阶段优先沿官方SUGAR完整Generator／Tracker的条件生成接口核查和适配，使用真正的
示范—实际未来配对，避免默认追加第五轮标量兼容性预测器预算。源码配置为12层／8头／
256宽度、8步36-D运动command，仍与实际29-D执行动作严格区分。现有条件只有当前物体／
机器人状态和最终物体目标，没有完整选定示范。新增示范条件与未来监督必须明确属于项目
适配，并保持官方完整架构、checkpoint、normalizer及求解路径的可核查基线。此阶段尚未
做新Generator训练；官方SMP继续作为任务级运动先验，不能替代示范
身份。另发现源码在单帧等长条件下的robot-state mask切片为空，尚须核查checkpoint中
实际mask buffer与配置，不能据此断言发布模型训练错误或推理失效。
见[官方Generator接口审计](../experiments/demo_following/demo_future_smp_v1/OFFICIAL_GENERATOR_INTERFACE_AUDIT.json)。

本轮训练／物理／视频子进程均已退出，allocation293231及其step0 shell仍保留。参考跟随
回读曾因计算节点读取旧launcher内容而两次在分发阶段退出，未执行科学评估；原子重新
发布相同launcher内容并核对计算节点所见后，实际回读完成。失败分发日志均保留。

随后完成发布Generator／Tracker零更新审计：Carry和Kick两套均严格加载；完整Generator
各8,275,200参数、12层／8头／256宽度，8步36-D运动command；Tracker actor各429,597
参数、510-D输入／29-D动作。固定seed采样逐项复现，297个参数张量获得有限反向梯度，
模型状态逐项保持。每任务八个原始PhysX记录的下一步实际Tracker动作最大复现误差为
2.861e-6／7.153e-7。官方wrapper实际使用DDPM、存储配置target为DDIM；本次保留官方
加载／求解路径，不擅自切换。发布checkpoint全部12块均关闭attention mask，因此源码
默认mask切片疑点不适用于这些权重。初次审计因错误假设mask启用而退出，失败日志保留；
修正审计配置分支后v2完成，不是模型训练失败或参数修复。
证据：[完整发布权重审计](../experiments/demo_following/demo_future_smp_v1/official_generator_audit_v2/RESULT.json)。

当前开始`tracker_aligned96_90_feasibility`：使用发布完整Tracker及其原PhysX任务、原始
参考、相同一次因果参考对齐和名义条件，先运行原96组450帧。如果无法到切换点或连续
抬升，先诊断执行能力；通过再运行重复／替代组及独立回读。Tracker观察历史仅由真实
env.step推进，反事实参考查询不推进历史。新增记录完整实时接触力子步历史，供以后按
官方Refiner-to-motion规则生成参考标签；实际29-D动作与36-D参考command独立保存。

上述原始参考Tracker终点失败：102步触发原`anchor_pos`终止，最大抬升仅0.002025m，
尚未到158切换点，未运行重复／替代组。终止前躯干参考距离0.311378m，躯干高度仍为
0.8437m；这是接近箱体时位置跟随落后，不能称为已经摔倒。首次初始化还因未提供训练
教师bank而构建不了未使用的teacher观察组；停用该训练专用组后才实际执行，失败产物
保留。510-D actor输入、完整权重、物理条件与所有终止阈值保持。

随后按官方流程补录已通过的Refiner96/90 pair：两组各450步，机体／物体／关节状态、
教师观察和实际动作与此前轨迹逐项相同。完整实时过滤接触力历史为450×1×4×3×3，
两组公共前158帧的这项新字段也逐项相同。调用官方`process_refiner_rollout.py`转换
两条actual motion，按原规则使用各手传感器历史最大力>0.1N的双手标签；Carry不做
initial-frame stabilization。两条参考接触标签分别256／270帧。这里物理测得姿态被
转换为参考motion，绝不称为29-D执行动作或训练时live tactile输入。

`tracker_refined96_90_feasibility`预声明400步，位于450个真实参考帧内部，避免末端
padding／reference-complete影响。原96组最大抬升0.584497m，连续抬升检查通过，但
第330步触发原`obj_pos`终止；最后动作前箱体距离参考0.307157m，主要为Y方向误差。
400步标准保持失败，未启动重复／替代或Generator训练。实际轨迹回读图已生成。

下一项教师兼容性诊断也有明确结果：让原冻结Refiner直接跟踪这些measured refined
motions，在268步因`obj_ori`终止，最大抬升0.262977m。不能以此作为有效BCPPO教师。
官方`SUGAR/train.sh`本来就分别使用refined student motion和对应原始teacher示范。
现已从原numeric demo、记录的source-frame时钟和一次参考刚体变换构建对应450帧原始
teacher bank，两组参考箱体／机体位置与已执行记录的最大误差均5.96e-8m；没有用实际
执行姿态替代原教师目标。该分段时间线在切换前八帧的教师future context和裁剪末端可能
不同于先前在线切换，因此只报告当前参考位置匹配，不声称整个890-D教师观察等价。
正在`aligned_original_teacher_feasibility`运行完整冻结Refiner400步兼容性检查，先原组，
通过再替代组；两者通过后才准备有界完整Tracker BCPPO适配。所有失败条件与旧数据保留。

对应原始教师bank的实际检查现已完成：原组／替代各400帧零重置，最大抬升
0.538593／0.585202m；冻结状态与连续抬升标准均通过。原始与measured参考的区别因此
影响当前教师兼容性，但尚未证明是全部Tracker误差的唯一原因。

已启动`tracker_bcppo_refined_pair_v1`：直接调用仓库SUGAR `train.py`、完整ActorCritic
和BCPPO，使用发布Tracker完整权重初始化、冻结项目官方Refiner，student读取两条
measured refined motion，teacher读取对应原始示范bank。预声明16环境、seed272041、
每更新24步、64次更新、lr1e-4、相同名义物理及fresh optimizer。全部预算处于官方前500
更新的KL蒸馏阶段；不是新Generator、SMP奖励或世界模型训练。数据规模、LR和固定预算
属于明确项目适配，不称为作者完整训练复现。当前训练日志已确认真实更新正在应用；
严格加载记录与终点分别见`training/actor_critic_warm_start.json`和model63／终点回读。
终点自动冻结，按原400步／seed272012条件先评估原组，通过再重复／替代组和独立回读。
训练使用准备好的固定参考bank，评估使用与发布Tracker基线相同的一次因果参考对齐。

首个BCPPO终点现已完成：索引0--63恰好64个runner更新，24,576次实际transition，
每参数optimizer clock为1,280（5epochs×4mini-batch×64），全权重有限、critic逐项
保持，actor八个参数张量实际改变。mean std从0.423986变为0.419428。冻结model63在
217步因`ee_body_pos`终止，最大抬升0.008244m、未连续抬箱；比发布Tracker的330步／
0.584497m退步。两checkpoint评估的全部startup数组逐项相同，同一因果对齐规则保持；
不同实际行为可以导致不同的对齐变换。失败终点未运行重复／替代或直接追加预算。

针对该失败的同世界教师接口诊断完成217帧，BCPPO官方teacher观察与切换到原teacher
bank后直接调用完整Refiner的890-D观察／动作最大误差均为0，冻结状态保持。加入这些
只读查询后，实际机体／物体／观察／执行动作又逐项复现217步终点。该轨迹上适配／发布
actor对教师动作MSE为0.050937／0.151407；更接近教师仍未带来更好实际执行。最初诊断
因相对teacher路径在切换cwd后解析错误而未初始化，已修复路径解析，失败产物保留。

现已启动`tracker_bcppo_refined_pair_low_exploration`：使用相同完整发布actor／critic
均值权重、seed272041、fresh optimizer、数据、teacher、lr1e-4、64更新和冻结400步
终点，只将初始per-joint探索std替换为统一0.05。这明确改变初始探索分布，不能称为
全部参数逐项相同；KL教师std目标保持，训练采样历史的变化是这个因素的作用结果。
这是对训练状态分布的有界诊断，不假定噪声是唯一原因，也不把首个失败延长为128更新。
此前首组training log的滚动平均episode长度从1升到约216，不能据此断言所有训练样本
都没有到达交互阶段。下一步仍由匹配冻结物理结果决定。

证据：[64更新回读](../experiments/demo_following/demo_future_smp_v1/tracker_bcppo_refined_pair_v1/UPDATE_AUDIT.json)、
[匹配失败终点](../experiments/demo_following/demo_future_smp_v1/tracker_bcppo_refined_pair_v1/MATCHED_ENDPOINT_READBACK.json)、
[同世界教师接口](../experiments/demo_following/demo_future_smp_v1/tracker_bcppo_refined_pair_v1/teacher_interface_audit/TEACHER_INTERFACE_AUDIT.json)。

证据：[原始参考Tracker](../experiments/demo_following/demo_future_smp_v1/tracker_aligned96_90_feasibility/FLOOR_DECISION.json)、
[官方refined转换](../experiments/demo_following/demo_future_smp_v1/refiner_aligned96_90_feasibility/refined_motion_export/RESULT.json)、
[refined参考Tracker](../experiments/demo_following/demo_future_smp_v1/tracker_refined96_90_feasibility/FLOOR_DECISION.json)、
[对应原始teacher bank](../experiments/demo_following/demo_future_smp_v1/refiner_aligned96_90_feasibility/aligned_original_teacher_bank/RESULT.json)。

证据：[冻结教师配对回读](../experiments/demo_following/demo_future_smp_v1/refiner_pair_feasibility/READBACK.json)、
[切换边界审计](../experiments/demo_following/demo_future_smp_v1/refiner_pair_feasibility/SWITCH_BOUNDARY_AUDIT.json)、
[实际状态官方SMP评分](../experiments/demo_following/demo_future_smp_v1/refiner_pair_feasibility/official_smp/RESULT.json)、
[固定phase参考可行性](../experiments/demo_following/demo_future_smp_v1/refiner_pair_feasibility/TRAIN_REFERENCE_FEASIBILITY.json)。

活动入口：[数据构建](../scripts/sugar/demo_following/demo_future/data.py)、
[冻结诊断](../scripts/sugar/demo_following/demo_future/audit_frozen.py)、
[匹配训练](../scripts/sugar/demo_following/demo_future/train_matched.py)、
[保留计算节点入口](../scripts/sugar/demo_following/run_demo_future_held.sh)。

## 2026-09-12 会话恢复时的历史快照

2026-09-12 会话恢复核对：本次用户要求恢复上个 session 的任务并讨论后续路线，提出结合
Zero-WAM 与 SMP 思路。以下最新观测覆盖下方 9 月 11 日运行中状态；新方法尚未实现或训练。

- 原任务是拉取 sugar 更新、区分真实 bug 修复与数据／GPU／方法变体，使用原始 PhysX
  数据和完整模型做生成式 overfit，直至指标和实际画面正常。该科学目标仍未达成。
- 磁盘 trace 确认 step96 段恰好追加索引 64--95 的 32 次更新，累计 96 次。完整模型、
  优化器、全部八例 TRAIN 视频、动作回读、冻结 VAE 对照及 112 次动作 flow 前向均已保存。
  本次只核对产物及数值，尚未补做 step96 的全部逐帧视觉检查，不能声称画面正常。
- step96 相对最初基线的匹配噪声 video/action/IFP 比值为
  `0.580787 / 0.242752 / 0.198857`；独立噪声为
  `0.019361 / 0.138865 / 0.306106`。prompt loss 判据通过，但原整体拟合判据失败。
- 实际八例动作 MSE 为 `0.081108–0.115207`，8/8 优于全零，0/8 优于逐关节常数
  均值 oracle；动作时序误差仍为零变化基线的 `8.37–32.33` 倍。真实未来视频替换后的
  动作 MSE 相对变化绝对值最大 `0.8163%`，仍为 0/8 优于常数均值。
- `t=0.02` 平均速度 MSE 从 step64 的 `1.372083` 降至 `1.140739`，但仍高于
  step32 的 `1.096990`；不能沿用“比 step64 持续退步”的说法，也不能说低噪声已拟合好。
- Slurm `sacct` 显示 291647 在 2026-09-12 06:51:05 为 CANCELLED，step0 于
  06:51:34 结束；本次 `squeue -u yanhongru` 为空，原 tmux 会话已不存在。取消原因未查明，
  本次未取消作业或申请新 GPU。旧文中的“保留 H200／正在训练”不是当前资源状态。

权威终点：[step96 结果](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911_step96/OVERFIT_RESULT.json)、
[动作回读](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911_step96/ACTION_RECONSTRUCTION.json)、
[真实未来诊断](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911_step96/INVERSE_DYNAMICS_PROBE.json)。

路线讨论依据：截至本次核查，[Zero-WAM 官方仓库](https://github.com/robbyant-research/Zero-WAM)
仍只有说明／图及发布计划，代码、模型、数据预计 2026-09-15 前发布，尚不能当作可运行实现。
建议保留本地重建的完整失败与改善证据，把后续研究重点放在官方 MimicKit/SMP 的冻结运动
先验、具体示范条件、跨多个未来时距的任务监督与 SUGAR 物理执行的结合。此为新方法建议，
不是官方 Zero-WAM 复现，也不是重启旧 tactile／HOST／BPP 的执行指令。官方 TinyMDM 的
task-class 条件和 ESM/SDS 分数不能充当已验证的 selected-demo latent 或精确示范距离。
先检验同一可行历史下更换示范能否改变多步预测与实际行为；只比较 Carry/Kick 分类不足以
证明同任务新示范跟随。新方法尚无实验结论。

## 2026-09-11 交接历史

记录日期：2026-09-11。最新用户目标：拉取 GitHub sugar 新代码、区分 bug 修复与数据／GPU
适配，并继续 overfit 直至指标和实际可视化正常。已拉取 4bcf8582；旧结果均保留。

当前使用原始 PhysX 199 轨迹／160-20-19 split、原八个 TRAIN 案例、完整 10.681B 模型，
不使用新提交中的运动学栅格替代数据。单 H200 291647/server31/step0 已在 tmux 保留。
初始 23 项 CPU 回归及当前代码的官方完整 30 层数值检查通过；后续 CPU 回归为 67 项。
最新运行：累计 64 次更新及全部终点诊断完成但仍未通过；23:36 已启动从 step64 精确
恢复到 step96 的有界 32-update 段，继承原优化／目标变体并试用已通过 H200 benchmark
的 CPU 搬运缓冲区。最新 CPU 回归 71 项通过。恢复与首个新增更新须以实际运行记录为准。
23:42 实际完整状态恢复已通过；23:47 已确认首个新增更新完成，累计 65 次。真实首步
耗时 277.5 秒，仍需继续观察后续步数及原标准终点评估。
本轮 32 步拟合、动作采样及动态视觉检查仍未通过，详见下方实测结果。
首个新 endpoint 为 overfit_resampled_noise_20260911，32 次噪声／时间重采样更新后自动
评估并渲染所有八例。此 endpoint 不等于目标完成，失败后继续依据证据分析和 overfit，
不自动启动 formal 或 physics。
详见[代码对比与完成条件](../experiments/demo_following/paper_zero_wam_v1/bugfix_audit_20260911/CODE_COMPARISON.json)。

9 月 11 日 12:49 初始八例／八噪声评估已完成：匹配条件的 video/action/IFP loss 为
`0.1872653 / 1.4499187 / 0.6574702`。teacher-action 隔离及生成视频到动作的连接通过，
正确 demo 的损失优势未通过。这是**更新前基线**，不能称为本轮 overfit 结果。
14:32 最新完整包级 CPU 回归 56 项通过，包括独立动作诊断的 JSON 配置 tuple 恢复与
动作时序检查；未修改正在运行的训练代码、模型或配置。这不是整个模型正确性或 overfit
成功的证明。
12:57 已在同一计算 shell 的当前前台任务之后排入
`run_paper_zero_wam_bugfix_endpoint_held.sh`：终点完成后导出全部 64 张冻结 VAE 对照帧；
先保存逐关节和时序动作回读；若采样动作未优于零／均值或时序基线，再做同一完整
checkpoint 的真实未来条件诊断。读取使用已有的仅导入重试入口，不重试训练或推理。
该诊断于 17:33 自动启动，launcher PID/PGID `3545495`，日志前缀为
`logs/held_291647_bugfix_endpoint`，不要重复排队。动作回读和 64 张 VAE 对照已完成。
16:53 在 server31 核实首段 32 次真实更新已全部完成，索引连续为 0--31，warmup 后
LR 为 `1e-5`。32 次更新均有限且 GPU/CPU AdamW 主参数回读一致；八个样本记录始终
相同，共使用 256 个不同噪声种子。前两次检查的 64 个 cross-attention 模块均有非零
梯度。22/32 步触发既定梯度裁剪；第 32 次范数 `1.09558`，未触发裁剪。平均每步约
`450.82 s`，主要耗时仍为 CPU 梯度搬运／累加。第 32 次单次 video/action/IFP loss 为
`0.0137297 / 1.0364978 / 0.1997930`；由于每步噪声／时间不同，不把它们与平均初始
探针直接相除来宣称通过。同一进程进入终点评估，尚未得到新画面或完整科学结论。

17:12 新 `FITTING_RESULT.json` 已发布：本段拟合和整体 prompt 判据仍未通过。
两组八次噪声采样的终点／初始比值如下（CarryBox、KickBox 分别检查也未整体通过）：

| 32 步新终点／初始 loss | 视频 | 动作 | IFP |
|---|---:|---:|---:|
| 匹配噪声采样组 | 0.627887 | 0.781634 | 0.333485 |
| 独立噪声采样组 | 0.055699 | 0.733700 | 0.433932 |

IFP 比值达到既定 0.5 阈值，动作在两组均未达到；视频只在独立采样组达到。视频结果
对采样组仍敏感，八次平均并不消除采样方差；不能只引用更有利的一组。完整 checkpoint
正在保存，实际动作与逐帧画面尚待渲染／诊断，因此还不能定位唯一根因或宣称生成正常。

17:16 模型与优化器保存完成，`OVERFIT_RESULT.json` 已发布。prompt 细分结果：视频在
两项任务的 wrong-task／reversed／same-task-alternate 比较中均更偏好正确示范，正差值
从更新前 2/6 项变为 6/6 项（`0.000873–0.001990`）；IFP 仍为 3/6 项，三个负差值
为约 `-3.20e-6` 至 `-1.11e-5`，因此整体 prompt 判据失败。teacher-action 隔离和
生成未来到动作的连接均通过。它们说明条件路径在工作，不证明动作重建或语义跟随成功。

17:40 全部八例的 64 张生成帧及 64 张 VAE 对照已检查：大块伪影和机器人／箱体消失
明显改善，但 slot01／02／06／07 仍有明确转身、起身、伸腿推箱或姿态时序偏差。
冻结 VAE 对照均接近真值；新导出中的 24 张已看过对照经直接 RGB 像素相等检查确认，
另 40 张逐帧补看，未以少量代表帧代替全序列。八例动作 MSE 为 `1.016–1.109`，全部
差于零动作基线 `0.203–0.295`；动作时序误差为零变化基线的 `151.4–533.5` 倍。
逐帧证据见[本轮视觉检查](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911/VISUAL_INSPECTION.json)。

只读 RGB 分析显示前景仅占 `2.22%–4.00%`：八例整图误差均优于保留最后已观测帧的
因果静止基线，但前景时序误差均差于零变化基线（`1.16–2.19` 倍）。这不是新增的
训练目标或降低后的通过标准，而是防止整图 MSE 掩盖动态偏差的诊断。
见[RGB 时序回读](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911/RGB_TEMPORAL_READBACK.json)。

18:00 真实未来条件诊断已正常退出，八例均严格复现已保存采样动作。换成真实未来后，
仍为 `0/8` 动作优于零值基线，单例 MSE 变化的绝对值至多 `0.000744`、相对变化至多
`0.0684%`。这八例不支持将当前动作失败主要归因于生成视频；下一步检查动作专家自身
的去噪拟合。当前没有第 33 次更新，也没有 formal／physics 启动。
见[完整逆动力学诊断](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911/INVERSE_DYNAMICS_PROBE.json)。

17:50 已把同一完整 checkpoint 的 `--action-flow-only` 诊断排在当前前台进程之后，
由同一保留 H200 和 pipeline lock 串行运行，日志前缀为
`logs/held_291647_bugfix_action_flow`，不要重复排队。它要求八例真实未来诊断完成且
严格回放成立，然后在 `t=0.02/0.1/0.25/0.5/0.75/0.9/1.0` 各用两组固定噪声，
共 112 次完整主干前向，读取动作速度误差及配对噪声响应，不训练或改变采样结果。
速度目标使用训练时原始精度减法；噪声／真值分量投影只是统计诊断，不是替代模型。
17:52 最新 67 项 CPU 回归通过。该 GPU 诊断已于 18:00:35 自动启动，PGID `1308711`、
Python PID `1308770`；18:01 正在加载完整 checkpoint，结果尚未发布。

18:11 已核实动作 flow 诊断于 18:06:04 正常退出，八例／七个时间／两组噪声共 112 次
完整前向全部完成。`t=1` 配对噪声响应增益仅 `0.01147–0.02121`（理想实数速度目标
约为 1）；低时间增益更小。输出在噪声／负真值／常数的统计投影中主要对应负真值，
并未学好速度目标中的噪声项。这与真实未来条件无法改善动作采样一致，但不证明唯一
代码根因。见[动作 flow 诊断](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911/ACTION_FLOW_PROBE.json)。

当前续训段为明确标注的**动作接口学习率实验**：精确恢复 step32 的完整 FP32 模型和
CPU AdamW 状态，只将 `action_encoder.{weight,bias}` 与
`action_head.projection.{weight,bias}` 四个新形状接口张量的 LR 乘以 100，实际为
`1e-3`；其余全宽视频／动作专家和 IFP 参数仍为 `1e-5`，全部继续训练。不重置矩、
时钟或噪声种子，不改变数据、结构、损失或既定判据；IFP detach 仍是原有本地变体。
这是检验接口优化过慢假设，不是已确认 bug 修复或论文官方配方。父终点和原始基线
保留，新目录为 `overfit_resampled_noise_20260911_step64`，仅追加索引 32--63 的
32 次更新，然后原判据评估、全部八例渲染、动作回读及必要的真实未来诊断。
接口 LR、变更边界和依据写入配置／诊断记录及逐步 trace，后续续训显式继承此变体。
70 项 CPU 回归通过，包含具备历史矩时与原生分组 AdamW 更新及状态逐元素相等检查；
它不代表真实全模型续训或科学目标通过。
18:16 实际父终点只读检查通过，step64 段已于 18:15:39 启动：保留分配
`291647/server31/step0`，launcher PID/PGID `2391636`，训练 Python PID/PGID
`2404474`，日志前缀 `logs/held_291647_bugfix_step64`。完整恢复及首个更新随后通过
下述实际检查；训练和全部渲染成功退出后，通过 `&&` 自动进入已有终点诊断脚本，日志
前缀 `logs/held_291647_bugfix_step64_endpoint`。不要重复启动任一子进程。
18:21 真实 `OVERFIT_RESUME.json` 已发布：step32 的 1770 份 AdamW 状态、参数顺序、
矩和时钟检查通过，完整 FP32 权重的 GPU／CPU 主参数回读一致，恢复本身新增更新为 0。
这补齐了此前只有 CPU fixture 的全模型恢复验证；仍不是新训练段拟合或生成成功。
18:29 已核实首个新增更新（全局索引 32，即第 33 次）真正应用：1770 次原生 AdamW
参数调用，所有参数有限，GPU／CPU 主参数回读一致。基础 LR `1e-5`，四个接口的实际
LR 均为 `0.001`；全局梯度范数 `1.58191`，未裁剪，耗时 `441.59 s`。八条样本记录
与父段相同，新噪声种子 `291957--291964` 与父段 256 个种子无重复。
该行 video/action/IFP loss 为 `0.0536248 / 1.0862179 / 0.1964481`，来自此次更新前
的前向，不能用它宣称接口 LR 变更已有效，更不能当作终点评估。
18:53 已在 step64 训练及终点诊断之后排入一次串行纯张量搬运测试，日志前缀
`logs/held_291647_step64_gradient_transfer`，尚未运行。比较 256 MiB／1 GiB 梯度存储
的原路径与单块 CPU 缓冲区复用，按原／复用／复用／原顺序各累积八次；检查精确和、
存储独立性与实测耗时。只有两种大小均通过精确检查且中位耗时下降至少 10%，才支持
在未来训练段试用。当前训练仍未启用复用；这不是模型实验或 overfit 成功证据。
22:19 已核实本段恰好完成 32 次新增更新（全局索引 32--63，累计 64/64 次），
没有追加第 65 次。每次均为 1770 次原生参数更新，全部训练参数有限、GPU／CPU
回读一致，均使用与父阶段完全相同的八条样本记录；两段累计 512 个噪声种子无重复。
本段平均每步 `444.79 s`，27/32 步触发既定裁剪，全局梯度峰值 `54.0276` 出现在
索引 59。最终梯度范数为 `24.0133`，裁剪系数 `0.0832872`，动作／视频／IFP
实际更新范数为 `0.154873 / 0.118913 / 0.047033`。梯度峰值未导致参数非有限或
同步放大的参数更新，仍需结合终点评估分析，不证明唯一根因或收敛。
最终训练 video/action/IFP loss 为 `0.022089 / 0.530898 / 0.135659`，不是固定
噪声评估。实际 PID `2404474` 仍在原 H200 上执行，已转入同进程终点评估流程；
22:37 `FITTING_RESULT.json` 已发布：整体仍未通过。相对最初模型，固定噪声
video/action/IFP 比值为 `0.588277 / 0.342161 / 0.263995`，独立噪声为
`0.033442 / 0.274474 / 0.364055`。固定噪声视频在 Carry／Kick 两任务分别为
`0.582186 / 0.594501`，均超过原阈值 0.5；独立噪声组三项及分任务拟合均通过，
但示范条件检查仍失败。相对 step32 父终点，动作损失分别降至 `0.437751 / 0.374095`，
固定噪声视频仅降至 `0.936915`。这证明当前完整模型的动作拟合有进展，不证明采样
动作、视频或示范遵循正常，也不单独证明学习率是唯一根因。此时正在保存终点，
八例实际渲染和后续诊断尚未完成；不追加训练或宣称整个 overfit 通过。
22:42 `OVERFIT_RESULT.json` 已在完整权重和优化器保存后发布，八例渲染流程已开始。
示范检查的六项视频损失差均为正，IFP 为 5/6（父终点为 3/6）；唯一负项是 CarryBox
倒放示范的 IFP 差值 `-2.53697e-5`。教师动作隔离和生成未来到动作路径检查通过。
正向连接和小幅示范损失差不证明实际语义遵循；仍需检查所有采样动作和生成画面。

22:59 已确认训练／渲染子进程于 22:58:52 正常退出，累计更新仍为 64，全部八例
TRAIN 视频已生成。64 张实际生成帧已逐帧检查：没有旧版严重块状破坏，转身、箱体位移
和 slot06 末尾伸腿推箱更接近真值，但仍有姿态、幅度、脚步和时序偏差。slot03／04
目标区间本身接近静止，不能将其相似画面当作完整搬箱／踢箱成功。补充纠正父终点
slot01 的描述：直接重看父帧 0／3／7 后确认原本已有腿部动作，主要是转身和前移不足，
不能概括成完全停住。新逐帧记录见
[step64 视觉检查](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911_step64/VISUAL_INSPECTION.json)。
同一前景／因果静止基线的 RGB 回读中，八例整图及前景误差均优于保留最后观测帧；
前景时序误差比值为 `0.506/0.778/0.789/0.720/0.866/0.826/0.856/1.267`，7/8
优于零变化，slot07 仍失败。GT 面板逐像素对应原 manifest，原渲染 MSE 复算一致。
这仍是诊断，不是更改后的完成标准；见
[step64 RGB 时序回读](../experiments/demo_following/paper_zero_wam_v1/overfit_resampled_noise_20260911_step64/RGB_TEMPORAL_READBACK.json)。
八例真实采样动作 MSE 为 `0.168–0.219`，8/8 优于全零，但 0/8 优于逐关节常数
均值 oracle（`0.0464–0.0904`）；动作时序误差仍为零变化基线的 `15.90–70.92` 倍。
因此动作重建和整体 overfit 仍失败，不能用 teacher-forced loss 或全零基线掩盖。
既有 endpoint 脚本于 22:58:53 自动启动，launcher PID/PGID `1708131`，日志前缀
`logs/held_291647_bugfix_step64_endpoint`；动作回读已完成，全部冻结 VAE 对照及必要
的真实未来条件诊断继续执行。没有第 65 次更新或 formal／physics。
23:01 全部 64 张本轮冻结 VAE 对照已导出，并逐一比较解码 RGB 数组：64/64 与
此前逐帧检查过的父终点对照像素完全相等。复用的是已验证相同的视觉证据，不声称又
人工看了 64 张新对照；冻结 VAE 仍接近 GT。完整比较和命令保存在本轮视觉检查 JSON。
23:11 可选 CPU 缓冲区复用已接入后续续训入口，仍默认关闭，71 项 CPU 回归通过。
`--cpu-gradient-staging-benchmark <completed.log>` 要求同型号 H200 的已正常退出测试，
八次累加的精确值／独立存储通过，并从两种大小的实测耗时重新计算中位数比值；两者
均低于 0.9 才启用，否则速度不达标时自动使用原路径。配置和每步 trace 记录实际开关
及缓冲区大小。当前 GPU benchmark 尚未完成，未宣称 GPU 提速或全模型等价性已验证。
23:27 已核实真实未来条件诊断于 23:26:12 正常退出。全部八例严格复现原始采样动作；
换成真实视频后，动作 MSE 相对变化的绝对值至多 `0.7396%`，仍为 0/8 优于逐关节
常数均值，动作时序误差仍是零变化的 `15.92–70.90` 倍。因此生成视频误差不是这八例
动作失败的主要解释，动作专家自身的去噪拟合仍需检查。
H200 搬运 benchmark 随后于 23:26:49 正常退出，两种大小、原／复用／复用／原顺序
全部通过精确和及独立存储检查。256 MiB／1 GiB 的复用／原路径中位耗时比为
`0.333134 / 0.365459`，满足既定低于 0.9 的试用条件；这是纯张量工程结果，不是整步
训练速度或整个模型数值等价性已经通过。下一段可显式使用该日志启用既有复用入口，
再观察真实全模型的步耗时、有限更新和 GPU／CPU 回读。
同一完整 step64 checkpoint 的动作去噪探针于 23:27:56 启动，launcher PID/PGID
`3779006`、Python PID `3780506`，日志前缀 `logs/held_291647_step64_action_flow`。
原有终点和 benchmark 均已退出、保留计算 shell 已回到提示符后才发送一次命令；
不要重复排队。该探针固定真实视频，在七个时间点各用两组噪声，不训练或改变原采样。
23:34 核实上述 112 次完整模型前向于 23:33:33 正常退出。纯噪声端 `t=1` 配对响应
增益从父终点的 `0.01147–0.02121` 提高到 `0.60903–0.80432`，七个时间点中
`t>=0.25` 的五个点均为 8/8 速度 MSE 改善。但 `t=0.02` 平均速度 MSE 从
`1.096990` 上升到 `1.372083`，8/8 都退步；`t=0.1` 平均也略升。低噪声端仍未
拟合好，不能将中高噪声进步或平均损失下降描述成整条动作去噪路径通过。
依据八例实际动作／时序误差及中高噪声响应均明显改善，下一次只追加 32 次更新，
检查进步能否扩展到低噪声和原始所有指标。新段不改变模型、数据、目标、学习率、
矩、噪声时钟或原始初始基线，只试用已验证的搬运复用；IFP detach 仍为明确标注的
本地变体，不能称论文目标完全一致。若下一终点低噪声退步持续，不再沿该设置直接
续训，先做 matched／zero／alternate 视频与动作历史条件比较及优化器分支诊断。
机器可读的有界续训判据保存在既有 CODE_COMPARISON.json。
step96 段于 23:36:14 在 `291647/server31/step0` 启动，launcher PID/PGID
`185809`、torchrun PID `185815`，日志前缀 `logs/held_291647_bugfix_step96`。
完整模型／优化器恢复及首个新增更新尚待实际记录；此时不能宣称已完成第 65 次更新。
同一命令通过 `&&` 串行接入 step96 endpoint 脚本，日志前缀
`logs/held_291647_bugfix_step96_endpoint`。新训练目录为
`overfit_resampled_noise_20260911_step96`，旧 step32／64 全部保留，无 formal／physics。
23:42 `OVERFIT_RESUME.json` 已发布：恢复 step64 的 1770 份优化器状态，参数顺序、
AdamW 矩和时钟均精确，GPU／CPU 主参数回读完全一致；恢复新增更新为 0。
实际训练 Python PID/PGID 为 `198219`，已进入首个新增更新。配置确认接口 LR 倍率
继承为 100，CPU staging 已启用；尚不能以此宣称全模型步耗时改善或新段科学通过。
23:47 已确认全局索引 64（累计第 65 次）真正应用：1770 次原生参数更新、全参数
有限及 GPU／CPU 回读一致；样本与两段父实验完全相同，累计 520 个噪声种子无重复。
基础 LR 仍为 `1e-5`、四个接口为 `1e-3`，梯度范数 `9.28557`、裁剪系数 `0.215388`。
实际复用缓冲区 `1,309,253,632` 字节，首步耗时 `277.492 s`，其中梯度搬运／累加
`76.965 s`（单独设备到主机复制 `53.908 s`），前反向 `143.684 s`。父段对应平均
为 `444.790 / 237.479 / 188.193 / 142.632 s`。真实模型上观察到提速，但这是单步
观察，不是稳定均值，也不是同一全梯度的双路径回放或新段拟合成功证明。
23:49 已将 step96 的条件动作去噪探针排在当前训练／终点流程之后，日志前缀
`logs/held_291647_step96_action_flow`，尚未运行，不要重复排队。沿用既有判据：
若采样动作已通过所有基线则跳过；否则要求真实未来诊断完整且八例严格回放；若
真实未来动作也未通过，才在同一完整 step96 模型上重测七个时间点。入口只检查结果并
诊断，不添加训练更新；训练进程本身保持原设置。

历史准备记录（已由上述真实续训验证更新）：15:09 续训入口已准备但当时未启动：
`train_single_gpu --resume-overfit-endpoint <parent>`
要求父终点完成评估、八例渲染、64 张 VAE 对照及必要的逆动力学诊断，再精确恢复 FP32
模型和 CPU AdamW，每段只追加 32 次更新，写入独立 `..._step64` 等目录。保留原始初始
基线，同时报告新段相对父终点的变化。首段没有热加载这些入口修改。
15:58 最新 64 项 CPU 回归通过，包含权重／状态文件恢复及恢复后下一步的
逐元素一致性测试；当时尚待父终点完成的真实 10.681B 恢复已于 18:21 验证。
可选 CPU 搬运缓冲区复用仅通过算术／所有权测试和 CPU 小规模性能对比，尚未验证 GPU
搬运提速，未启用到训练；证据记录在现有代码对比 JSON 中。

旧修复版八例的逐关节回读再次证实动作失败：动作 MSE `0.648–0.752`，逐关节常数均值
oracle 的 MSE 仅 `0.0464–0.0904`；相邻帧动作变化误差为零变化基线的 `89.5–352.6` 倍。
这不是新实验结果；详见本地 `bugfix_audit_20260911/HISTORICAL_ACTION_RECONSTRUCTION.json`。

已验证的对比边界：旧梯度裁剪实际触发 24/32 步，而非“从未触发”；真实 attention mask
阻断了 action-target 到视频 token 的所有路径，因此替换这些 token 不能解释视频条件修复。
学习率／初始化／IFP detach 是待实测的优化或方法变更，不能直接等同于已经定位的 bug。
论文的 IFP 目标是监督主干表示，detach 会改变这条训练路径；保留为明确标注的变体诊断，
不宣称忠实复现。[Zero-WAM §3.3](https://arxiv.org/html/2608.26103v2#S3.SS3)
本轮启动时的 `DIAGNOSTIC_CONTRACT.json` 把 `objective_changes` 错标为 false；以实际
`ifp_trunk_gradient=false` 配置和上述方法说明为准。原始记录保留，后续记录的标签已修正。

以下是 9 月 10 日已结束的历史结果，不是当前新实验的结论。

## 修复版 overfit：完成但未通过

本地论文重建保留 10,680,751,069 个参数：独立的 30 层／3072 宽视频和动作专家，
以及四个完整宽度 IFP heads。这不是 Zero-WAM 作者的官方实现。

修复了继承的空文本投影／cross-attention、61 帧全跨度示范支持及反转支持一致性，
并接回官方 RoPE／FlashAttention。199 条新示范缓存完成检查，机器人目标与动作保持不变。
官方完整视频路径与 IFP 数值对照通过；这不等于整个模型或训练目标已被证明正确。

相同 8 个训练案例、固定噪声和时间、batch 8，恰好完成 32 次真实更新：

| 终点／初始 loss | 视频 | 动作 | IFP |
|---|---:|---:|---:|
| 固定噪声／时间 | 1.666156 | 0.222756 | 0.162461 |
| 独立噪声／时间 | 3.097468 | 0.581112 | 0.854718 |

整体拟合和示范区分判据均失败。全部 8 个实际 TRAIN 案例已检查首／中／末帧：
生成画面块状伪影明显，机器人和箱体结构破坏或缺失；8 个冻结 VAE 对照接近真值。
这不是 held-out 展示，也不是闭环物理 rollout。

- [完整结果与原因分析](../experiments/demo_following/paper_zero_wam_v1/overfit_repaired_fixed_noise_20260910/OUTCOME_ANALYSIS.json)
- [执行完成核对](../experiments/demo_following/paper_zero_wam_v1/overfit_repaired_fixed_noise_20260910/COMPLETION_AUDIT.json)
- [全部 TRAIN 视频与帧](../experiments/demo_following/paper_zero_wam_v1/overfit_repaired_fixed_noise_20260910/training_videos)
- [逐案例视觉检查](../experiments/demo_following/paper_zero_wam_v1/overfit_repaired_fixed_noise_20260910/VISUAL_INSPECTION.json)
- [实现修复审计](../experiments/demo_following/paper_zero_wam_v1/REPAIR_IMPLEMENTATION_AUDIT_20260910.json)

## 原因边界与未执行建议

不能把失败简单归因于训练规模或模型太小，也没有找到唯一根因。固定时间／噪声拟合
不等于生成式 overfit；动作训练使用真实未来，部署依赖生成未来，存在误差传播风险。
修复后视频分支损失反而上升，需要先隔离视频、逆动力学与 IFP 的训练路径。

本轮训练计算约 7.1 小时，修复版 attention 适配器有严重效率退化，尚未通过性能剖析
定位唯一瓶颈。建议依次验证官方完整视频路径、真实未来到动作、再到联合生成式 overfit；
这些是尚未执行的方法建议，不是自动训练队列。

## 既有 endpoint 与执行边界

- [step-700 完整分析](../experiments/demo_following/paper_zero_wam_v1/formal/STEP700_OUTCOME_ANALYSIS.json)：
  390 组／1,950 次评分及 8 段预测视频完成，未通过示范跟随。
- [修复前固定噪声诊断](../experiments/demo_following/paper_zero_wam_v1/overfit_debug_fixed_noise_20260909/OUTCOME_ANALYSIS.json)：
  保留原始失败证据，不覆盖。
- 没有第 33 步、全量第 701 步、物理 rollout 或额外参数／种子扫描。
- 训练和渲染进程已退出。整理时单 H200 288297 / server23 仍在 tmux
  `curiosity_pzw_overfit_h200_20260909` 保留；它是带日期的观测，不保证将来仍存活。
- HOST 为负结果，BPP 与 tactile 为未激活历史路线；归档旧计划不会自动重启它们。

旧详细方法、预算、计划与 TODO 见[归档说明](archive.md)。
w_overfit_h200_20260909` 保留；它是带日期的观测，不保证将来仍存活。
- HOST 为负结果，BPP 与 tactile 为未激活历史路线；归档旧计划不会自动重启它们。

旧详细方法、预算、计划与 TODO 见[归档说明](archive.md)。
