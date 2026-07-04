(function () {
  "use strict";

  const canvas = document.getElementById("scene");
  const ctx = canvas.getContext("2d");

  const ids = [
    "robotHeight",
    "armReach",
    "torqueCap",
    "boxMass",
    "boxWidth",
    "comOffset",
    "friction"
  ];

  const controls = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
  const phaseText = document.getElementById("phaseText");
  const strategyText = document.getElementById("strategyText");
  const effortText = document.getElementById("effortText");
  const balanceText = document.getElementById("balanceText");
  const slipText = document.getElementById("slipText");
  const toggleRun = document.getElementById("toggleRun");
  const resetRun = document.getElementById("resetRun");

  const state = {
    running: true,
    t: 0,
    lastTime: 0
  };

  const phases = [
    { name: "approach", end: 4.2 },
    { name: "probe", end: 7.4 },
    { name: "repose", end: 10.4 },
    { name: "lift", end: 13.0 },
    { name: "carry", end: 22.0 }
  ];

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function lerp(a, b, p) {
    return a + (b - a) * clamp(p, 0, 1);
  }

  function smoothstep(p) {
    const q = clamp(p, 0, 1);
    return q * q * (3 - 2 * q);
  }

  function readParams() {
    return {
      robotHeight: Number(controls.robotHeight.value),
      armReach: Number(controls.armReach.value),
      torqueCap: Number(controls.torqueCap.value),
      boxMass: Number(controls.boxMass.value),
      boxWidth: Number(controls.boxWidth.value),
      comOffset: Number(controls.comOffset.value),
      friction: Number(controls.friction.value)
    };
  }

  function getPhase(t) {
    for (const phase of phases) {
      if (t < phase.end) return phase.name;
    }
    return "carry";
  }

  function phaseProgress(t, name) {
    const index = phases.findIndex((phase) => phase.name === name);
    const start = index === 0 ? 0 : phases[index - 1].end;
    const end = phases[index].end;
    return clamp((t - start) / (end - start), 0, 1);
  }

  function planPosture(params, t) {
    const load = params.boxMass / 16;
    const width = params.boxWidth;
    const reachStress = clamp((width - params.armReach) / 0.55, 0, 1);
    const comStress = Math.abs(params.comOffset);
    const torqueDemand = params.boxMass * (0.52 + width * 0.24 + comStress * 0.5);
    const capacity = 7.6 * params.torqueCap * params.robotHeight;
    const overload = clamp((torqueDemand - capacity) / 6.2, 0, 1);
    const slipRisk = clamp(load * 0.62 + reachStress * 0.22 + (1 - params.friction) * 0.72, 0, 1);

    let strategy = "front carry";
    if (overload > 0.46 || load > 0.72) strategy = "chest support";
    if (load > 0.52 && params.friction < 0.5) strategy = "low carry";
    if (Math.abs(params.comOffset) > 0.26) strategy = "asymmetric carry";
    if (overload > 0.72 && params.friction < 0.42) strategy = "abort posture";

    const supportBonus = strategy === "chest support" ? 0.22 : 0;
    const lowBonus = strategy === "low carry" ? 0.14 : 0;
    const asymPenalty = strategy === "asymmetric carry" ? 0.08 : 0;
    const balance = clamp(
      0.86 - overload * 0.38 - slipRisk * 0.2 + supportBonus + lowBonus - asymPenalty,
      0,
      1
    );
    const effort = clamp(torqueDemand / Math.max(capacity, 0.01), 0, 1.8);

    const holdHeight =
      strategy === "low carry" ? 0.56 :
      strategy === "chest support" ? 0.78 :
      strategy === "asymmetric carry" ? 0.68 :
      0.72;

    const torsoLean =
      strategy === "chest support" ? -0.2 :
      strategy === "low carry" ? 0.08 :
      strategy === "asymmetric carry" ? -params.comOffset * 0.38 :
      -0.08;

    const stance = clamp(0.22 + overload * 0.22 + slipRisk * 0.12 + Math.abs(params.comOffset) * 0.18, 0.22, 0.56);
    const gaitSpeed = clamp(1 - overload * 0.45 - slipRisk * 0.25, 0.28, 1);
    const probePulse = Math.sin(t * 6.4) * (getPhase(t) === "probe" ? 1 : 0);

    return {
      strategy,
      effort,
      balance,
      slipRisk,
      holdHeight,
      torsoLean,
      stance,
      gaitSpeed,
      probePulse,
      liftAllowance: clamp(1.1 - overload * 0.55 - slipRisk * 0.2, 0.05, 1)
    };
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    const w = Math.max(640, Math.floor(rect.width * scale));
    const h = Math.max(420, Math.floor(rect.height * scale));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function world(ctx2d) {
    const w = canvas.width;
    const h = canvas.height;
    const ground = h * 0.78;
    const scale = Math.min(w / 9.4, h / 5.8);
    ctx2d.setTransform(scale, 0, 0, scale, w * 0.09, ground);
    return { w, h, ground, scale };
  }

  function line(a, b, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function circle(p, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawRobot(x, params, posture, phase) {
    const h = params.robotHeight;
    const footY = 0;
    const hip = { x, y: -1.05 * h };
    const torsoTop = {
      x: x + posture.torsoLean * 0.8,
      y: -1.78 * h
    };
    const head = { x: torsoTop.x, y: torsoTop.y - 0.2 * h };
    const stance = posture.stance;
    const walk = phase === "approach" || phase === "carry" ? Math.sin(state.t * 6 * posture.gaitSpeed) * 0.08 : 0;

    const leftFoot = { x: x - stance - walk, y: footY };
    const rightFoot = { x: x + stance + walk, y: footY };
    const leftKnee = { x: x - stance * 0.4, y: -0.52 * h + walk * 0.2 };
    const rightKnee = { x: x + stance * 0.4, y: -0.52 * h - walk * 0.2 };

    const shoulder = { x: torsoTop.x + 0.02, y: torsoTop.y + 0.22 * h };
    const reach = 0.58 * params.armReach;
    const handY = -posture.holdHeight * h - 0.28 + posture.probePulse * 0.02;
    const handCenter = { x: x + 0.82 + posture.torsoLean * 0.5, y: handY };
    const leftHand = { x: handCenter.x, y: handCenter.y - 0.17 };
    const rightHand = { x: handCenter.x, y: handCenter.y + 0.17 };
    const elbow = { x: shoulder.x + reach * 0.55, y: shoulder.y + (handY - shoulder.y) * 0.45 };

    line(leftFoot, leftKnee, "#7fb3d5", 0.055);
    line(leftKnee, hip, "#7fb3d5", 0.055);
    line(rightFoot, rightKnee, "#7fb3d5", 0.055);
    line(rightKnee, hip, "#7fb3d5", 0.055);
    line(hip, torsoTop, "#e8edf2", 0.075);
    line(shoulder, elbow, "#e8edf2", 0.052);
    line(elbow, leftHand, "#e8edf2", 0.052);
    line(elbow, rightHand, "#e8edf2", 0.052);
    circle(head, 0.13 * h, "#f2d0a7");
    circle(leftFoot, 0.06, "#6ab7d8");
    circle(rightFoot, 0.06, "#6ab7d8");

    const comX = lerp(hip.x, torsoTop.x, 0.45) + posture.torsoLean * 0.25;
    circle({ x: comX, y: -1.18 * h }, 0.045, posture.balance > 0.55 ? "#62c77b" : "#df6b5f");
    line({ x: comX, y: -1.18 * h }, { x: comX, y: 0 }, "rgba(98,199,123,0.42)", 0.018);

    return { leftHand, rightHand, handCenter };
  }

  function drawBox(box, params, posture, carried) {
    const boxW = params.boxWidth;
    const boxH = 0.52;
    const x = box.x;
    const y = box.y;
    const tilt = carried ? params.comOffset * 0.18 * (1 - posture.balance) : 0;

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(tilt);
    ctx.fillStyle = "#b78345";
    ctx.strokeStyle = "#f0c27a";
    ctx.lineWidth = 0.024;
    ctx.fillRect(-boxW / 2, -boxH, boxW, boxH);
    ctx.strokeRect(-boxW / 2, -boxH, boxW, boxH);
    ctx.strokeStyle = "rgba(255,255,255,0.32)";
    ctx.beginPath();
    ctx.moveTo(-boxW / 2, -boxH * 0.52);
    ctx.lineTo(boxW / 2, -boxH * 0.52);
    ctx.stroke();
    circle({ x: params.comOffset * boxW, y: -boxH * 0.52 }, 0.045, "#df6b5f");
    ctx.restore();
  }

  function drawGauge(x, y, width, label, value, color) {
    ctx.fillStyle = "rgba(255,255,255,0.12)";
    ctx.fillRect(x, y, width, 0.08);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, width * clamp(value, 0, 1), 0.08);
    ctx.fillStyle = "#f2f5f7";
    ctx.font = "0.13px system-ui";
    ctx.fillText(label, x, y - 0.05);
  }

  function drawScene() {
    resizeCanvas();
    const params = readParams();
    const phase = getPhase(state.t);
    const posture = planPosture(params, state.t);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, "#161b24");
    gradient.addColorStop(1, "#101218");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    world(ctx);
    ctx.font = "0.16px system-ui";
    ctx.fillStyle = "#2a323e";
    ctx.fillRect(-0.9, 0, 10.5, 0.08);
    ctx.fillStyle = "#2d3641";
    ctx.fillRect(3.7, -0.76, 1.25, 0.08);
    ctx.fillRect(3.84, -0.68, 0.08, 0.68);
    ctx.fillRect(4.72, -0.68, 0.08, 0.68);

    const approach = smoothstep(phaseProgress(state.t, "approach"));
    const carryP = phase === "carry" ? phaseProgress(state.t, "carry") : 0;
    const robotX = phase === "approach" ? lerp(0.3, 3.0, approach) : lerp(3.0, 6.8, carryP);
    const liftP = phase === "lift" ? smoothstep(phaseProgress(state.t, "lift")) : phase === "carry" ? 1 : 0;
    const reposeP = phase === "repose" ? smoothstep(phaseProgress(state.t, "repose")) : (phase === "lift" || phase === "carry") ? 1 : 0;

    posture.holdHeight = lerp(0.42, posture.holdHeight, reposeP);
    posture.stance = lerp(0.22, posture.stance, reposeP);
    posture.torsoLean = lerp(0, posture.torsoLean, reposeP);

    const hands = drawRobot(robotX, params, posture, phase);
    const carried = phase === "lift" || phase === "carry";
    const tableBox = { x: 4.33, y: -0.76 };
    const carriedBox = {
      x: hands.handCenter.x + params.comOffset * 0.12,
      y: lerp(tableBox.y, hands.handCenter.y + 0.25, liftP * posture.liftAllowance)
    };
    drawBox(carried ? carriedBox : tableBox, params, posture, carried);

    if (phase === "probe") {
      const probeX = 4.33 + Math.sin(state.t * 6.4) * 0.13;
      line({ x: hands.handCenter.x, y: hands.handCenter.y }, { x: probeX, y: -0.98 }, "#e8b94f", 0.03);
    }

    if (posture.strategy === "chest support" && carried) {
      ctx.fillStyle = "rgba(98,199,123,0.18)";
      ctx.fillRect(robotX + 0.42, -1.35 * params.robotHeight, 0.42, 0.46);
    }

    drawGauge(0.0, -3.55, 1.3, "effort", posture.effort / 1.8, posture.effort > 1 ? "#df6b5f" : "#e8b94f");
    drawGauge(1.55, -3.55, 1.3, "balance", posture.balance, posture.balance > 0.55 ? "#62c77b" : "#df6b5f");
    drawGauge(3.1, -3.55, 1.3, "slip", posture.slipRisk, posture.slipRisk > 0.6 ? "#df6b5f" : "#6ab7d8");

    phaseText.textContent = phase;
    strategyText.textContent = posture.strategy;
    effortText.textContent = posture.effort.toFixed(2);
    balanceText.textContent = posture.balance.toFixed(2);
    slipText.textContent = posture.slipRisk.toFixed(2);
  }

  function animate(now) {
    if (!state.lastTime) state.lastTime = now;
    const dt = Math.min((now - state.lastTime) / 1000, 0.05);
    state.lastTime = now;
    if (state.running) {
      state.t += dt;
      if (state.t > phases[phases.length - 1].end) state.t = 0;
    }
    drawScene();
    requestAnimationFrame(animate);
  }

  toggleRun.addEventListener("click", () => {
    state.running = !state.running;
    toggleRun.textContent = state.running ? "Pause" : "Start";
  });

  resetRun.addEventListener("click", () => {
    state.t = 0;
    state.running = false;
    toggleRun.textContent = "Start";
    drawScene();
  });

  for (const control of Object.values(controls)) {
    control.addEventListener("input", drawScene);
  }

  window.addEventListener("resize", drawScene);
  toggleRun.textContent = "Pause";
  requestAnimationFrame(animate);
})();
