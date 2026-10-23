/* Chiller Twin front-end: fault injection, playback, diagnosis report. */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  const FAULTS = {
    fwc: { code: "FWC", name: "Reduced condenser water flow", color: "--f-fwc" },
    fwe: { code: "FWE", name: "Reduced evaporator water flow", color: "--f-fwe" },
    cf:  { code: "CA",  name: "Condenser fouling", color: "--f-cf" },
    ro:  { code: "RO",  name: "Refrigerant overcharge", color: "--f-ro" },
    rl:  { code: "RL",  name: "Refrigerant leak / undercharge", color: "--f-rl" },
    nc:  { code: "NC",  name: "Non-condensables in refrigerant", color: "--f-nc" },
    eo:  { code: "EO",  name: "Excess oil", color: "--f-eo" },
    normal: { code: "OK", name: "Normal", color: "--f-normal" },
  };
  const ORDER = ["fwc", "fwe", "cf", "ro", "rl", "nc", "eo"];
  const LANES = ["eo", "nc", "rl", "ro", "cf", "fwe", "fwc", "normal"]; // top -> bottom
  const css = getComputedStyle(document.documentElement);
  const color = (f) => css.getPropertyValue(FAULTS[f].color).trim();

  const S = {
    cfg: null, hours: 2, values: {}, probes: {}, result: null, running: false,
    sensors: {}, sevFrac: {},
  };

  /* ---------------- helpers ---------------- */
  const pad = (n) => String(n).padStart(2, "0");
  const hhmm = (s) => `${Math.floor(s / 3600)}:${pad(Math.floor((s % 3600) / 60))}`;
  const hhmmss = (s) => `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(Math.floor(s % 60))}`;
  const dur = (s) => { s = Math.round(s); const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60); return h ? `${h}h ${pad(m)}m` : `${m}m`; };
  const chip = (f, extra = "") => `<span class="chip" style="--fc:${color(f)}">${FAULTS[f].code}${extra}</span>`;
  function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), 4500); }

  setInterval(() => { $("#clock").textContent = new Date().toLocaleTimeString("en-GB"); }, 1000);

  /* ---------------- severity model ---------------- */
  function severityOf(slider, v) {
    const L4 = slider.levels["4"];
    const frac = ((v - slider.nominal) * slider.direction) / ((L4 - slider.nominal) * slider.direction);
    return { frac: Math.max(0, frac), sev: frac < 0.125 ? 0 : frac < 0.375 ? 1 : frac < 0.625 ? 2 : frac < 0.875 ? 3 : 4 };
  }
  function armed() {
    const out = {};
    for (const sl of S.cfg.sliders) {
      const { sev, frac } = severityOf(sl, S.values[sl.var]);
      S.sevFrac[sl.fault] = sev ? frac : 0;
      if (sev) out[sl.fault] = sev;
    }
    return out;
  }

  /* ---------------- probes ---------------- */
  function buildProbes() {
    const byVar = {};
    for (const sl of S.cfg.sliders) (byVar[sl.var] ||= []).push(sl);
    const host = $("#probes");
    host.innerHTML = "";
    for (const [v, sliders] of Object.entries(byVar)) {
      const lay = Chiller.PROBES[v];
      const nominal = sliders[0].nominal;
      let min = Math.min(...sliders.map((s) => s.min)), max = Math.max(...sliders.map((s) => s.max));
      if (sliders.length > 1) { // bidirectional probe (RO up / RL down)
        min = Math.min(...sliders.map((s) => Math.min(s.levels["4"], s.nominal))) - 0.6;
        max = Math.max(...sliders.map((s) => Math.max(s.levels["4"], s.nominal))) + 1.2;
      }
      const step = (max - min) / 400;
      const el = document.createElement("div");
      el.className = "probe";
      el.id = `probe-${v}`;
      el.style.left = (lay.x / 1200) * 100 + "%";
      el.style.top = (lay.y / 640) * 100 + "%";
      const pos = (x) => ((x - min) / (max - min)) * 100;
      const ticks = [`<div class="tick nom" style="left:${pos(nominal)}%"></div>`];
      for (const sl of sliders) for (let k = 1; k <= 4; k++) {
        const x = sl.nominal + (k / 4) * (sl.levels["4"] - sl.nominal);
        ticks.push(`<div class="tick" style="left:${pos(x)}%;background:${color(sl.fault)}"><span>${k}</span></div>`);
      }
      el.innerHTML = `
        <div class="probe-top"><span class="probe-name">${sliders[0].label}</span><span class="probe-var">${v}</span></div>
        <div class="probe-val"><span class="pv">${nominal.toFixed(1)}</span><small>${sliders[0].unit}</small></div>
        <div class="rng">
          <div class="track"></div><div class="fill"></div>${ticks.join("")}<div class="knob"></div>
          <input type="range" id="slider-${v}" aria-label="${sliders[0].label}" min="${min}" max="${max}" step="${step}" value="${nominal}">
        </div>
        <div class="rng-foot"><span>${lay.component}</span><span class="nom">nominal ${nominal.toFixed(1)}</span></div>
        <div class="probe-tags"></div>`;
      host.appendChild(el);
      S.values[v] = nominal;
      S.probes[v] = { el, sliders, min, max, nominal, pos };
      const input = el.querySelector("input");
      input.addEventListener("input", () => { S.values[v] = +input.value; refresh(); });
      input.addEventListener("dblclick", () => { input.value = nominal; S.values[v] = nominal; refresh(); });
    }
  }

  function paintProbe(v) {
    const p = S.probes[v];
    const val = S.values[v];
    const el = p.el;
    let active = null;
    const tags = [];
    for (const sl of p.sliders) {
      const { sev } = severityOf(sl, val);
      if (sev) { active = sl.fault; tags.push(chip(sl.fault, ` · SL${sev}`)); }
    }
    if (!tags.length) tags.push(p.sliders.map((s) => `<span class="chip ghost">${FAULTS[s.fault].code} ${s.direction > 0 ? "▲" : "▼"}</span>`).join(""));
    el.querySelector(".probe-tags").innerHTML = tags.join("");
    el.querySelector(".pv").textContent = val.toFixed(1);
    const fc = active ? color(active) : "";
    el.style.setProperty("--fc", fc || "var(--ice)");
    el.classList.toggle("armed", !!active);
    const a = p.pos(p.nominal), b = p.pos(val);
    const fill = el.querySelector(".fill");
    fill.style.left = Math.min(a, b) + "%";
    fill.style.width = Math.abs(b - a) + "%";
    el.querySelector(".knob").style.left = b + "%";
    Chiller.setLeader($("#chiller"), v, fc);
  }

  /* ---------------- guide ---------------- */
  function buildGuide() {
    $("#guide-list").innerHTML = ORDER.map((f) => {
      const sl = S.cfg.sliders.find((s) => s.fault === f);
      const up = sl.direction > 0;
      return `<li data-f="${f}" data-var="${sl.var}" style="--fc:${color(f)}">
        <span class="g-code">${FAULTS[f].code}</span>
        <div><div class="g-name">${FAULTS[f].name}</div>
          <div class="g-how">${up ? "Raise" : "Lower"} <b>${sl.var}</b> on ${Chiller.PROBES[sl.var].component.toLowerCase()} · SL4 ≈ ${sl.levels["4"].toFixed(1)} ${sl.unit}</div></div>
        <span class="g-dir">${up ? "▲" : "▼"}<span class="g-sev"></span></span>
      </li>`;
    }).join("");
    $$("#guide-list li").forEach((li) => li.addEventListener("click", () => {
      const p = S.probes[li.dataset.var].el;
      p.classList.remove("flash"); void p.offsetWidth; p.classList.add("flash");
    }));
  }

  /* ---------------- sensors (idle twin) ---------------- */
  function idleSensors(jitter = 0) {
    const b = S.cfg.baseline, v = S.values, f = S.sevFrac;
    const j = () => (Math.random() - 0.5) * jitter;
    const TCI = b.TCI + j(), TEO = b.TEO + j() * 0.3;
    const TCO = TCI + v.CDW_DT, TEI = TEO + v.CHW_DT;
    const TRC = TCO + v.TCA + j() * 0.3;
    const TRE = b.TRE - 1.2 * (f.fwe || 0) - 0.8 * (f.rl || 0) + j() * 0.2;
    const PRC = v.PRC + j();
    const PRE = 34.6 + (TRE - 39.5) * 0.75;
    const kW = b.kW * (1 + 0.06 * (f.cf || 0) + 0.12 * (f.nc || 0) + 0.05 * (f.ro || 0) + 0.04 * (f.fwc || 0)) + j() * 1.5;
    return { TCI, TCO, TEI, TEO, TRC, TRE, PRC, PRE, kW, TR_dis: 112 + (PRC - 95) * 0.9, PO_net: v.PO_net };
  }

  function paintKpis(sn) {
    $("#k-kw").textContent = sn.kW.toFixed(1);
    $("#k-lift").textContent = (sn.PRC - sn.PRE).toFixed(1);
    const carnot = (sn.TRE + 459.67) / Math.max(1, sn.TRC - sn.TRE);
    $("#k-cop").textContent = (carnot * 0.55).toFixed(2);
  }

  function refresh() {
    Object.keys(S.probes).forEach(paintProbe);
    const a = armed();
    const keys = Object.keys(a);
    $("#k-armed").textContent = keys.length;
    $("#armed-list").innerHTML = keys.length
      ? `<span class="muted">Armed:</span> ` + ORDER.filter((f) => a[f]).map((f) => chip(f, ` SL${a[f]}`)).join("")
      : `<span class="muted">No fault armed — the machine will run healthy.</span>`;
    $$("#guide-list li").forEach((li) => {
      const s = a[li.dataset.f];
      li.classList.toggle("armed", !!s);
      li.querySelector(".g-sev").textContent = s ? ` SL${s}` : "";
    });
    if (!S.running) {
      S.sensors = idleSensors();
      Chiller.update($("#chiller"), { sev: S.sevFrac, sensors: S.sensors, mode: "idle" });
      paintKpis(S.sensors);
    }
  }

  /* ---------------- run + playback ---------------- */
  function setLocked(on) {
    $$(".probe input, #hours button, #btn-reset").forEach((el) => (el.disabled = on));
    $("#btn-run").disabled = on;
  }
  function pill(state, label) { const p = $("#machine-pill"); p.className = "pill " + state; p.querySelector("span").textContent = label; }

  async function run() {
    const faults = armed();
    setLocked(true);
    S.running = true;
    pill("run", "RUNNING");
    $("#btn-run span").textContent = "Running…";
    let res;
    try {
      const r = await fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ hours: S.hours, faults }) });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      res = await r.json();
    } catch (e) {
      toast("Simulation failed: " + e.message);
      S.running = false; setLocked(false); pill("", "IDLE"); $("#btn-run span").textContent = "Run chiller";
      return;
    }
    res.injected = faults;
    S.result = res;
    await playback(res);
    pill("done", "COMPLETE");
    $("#step-2").disabled = false;
    setTimeout(() => { buildReport(res); show("report"); S.running = false; setLocked(false); $("#btn-run span").textContent = "Run chiller"; $("#guide-card").hidden = false; $("#telemetry").hidden = true; $("#run-overlay").hidden = true; refresh(); }, 900);
  }

  function playback(res) {
    const cols = res.columns, rows = res.rows, N = rows.length;
    const ci = Object.fromEntries(cols.map((c, i) => [c, i]));
    const wins = res.diagnosis.windows;
    const total = N * S.cfg.dt_s;
    const ms = 6500 + S.hours * 1100;
    $("#guide-card").hidden = true;
    $("#telemetry").hidden = false;
    $("#run-overlay").hidden = false;
    $("#infer-log").innerHTML = "";
    const cv = $("#strip"), ctx = cv.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    cv.width = cv.clientWidth * dpr; cv.height = 150 * dpr;
    let shown = 0;
    const series = [["TCO", css.getPropertyValue("--warm")], ["TCI", css.getPropertyValue("--amber")], ["TEI", css.getPropertyValue("--ice")], ["TEO", css.getPropertyValue("--cyan")]];
    let lo = Infinity, hi = -Infinity;
    for (const r of rows) for (const [k] of series) { lo = Math.min(lo, r[ci[k]]); hi = Math.max(hi, r[ci[k]]); }
    lo -= 1; hi += 1;

    return new Promise((resolve) => {
      const t0 = performance.now();
      function frame(now) {
        const p = Math.min(1, (now - t0) / ms);
        const idx = Math.max(1, Math.floor(p * N));
        const row = rows[idx - 1];
        const scen = res.scenario[idx - 1];
        const sensors = Object.fromEntries(cols.map((c) => [c, row[ci[c]]]));
        const sev = {};
        if (scen !== "normal") sev[scen] = Math.max(0.35, res.injected[scen] / 4);
        Chiller.update($("#chiller"), { sev, sensors, mode: "running" });
        paintKpis(sensors);
        // strip chart: last 240 samples
        const W = cv.width, H = cv.height, span = 240, s0 = Math.max(0, idx - span);
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = "rgba(148,163,184,.08)"; ctx.lineWidth = 1;
        for (let g = 1; g < 4; g++) { ctx.beginPath(); ctx.moveTo(0, (H * g) / 4); ctx.lineTo(W, (H * g) / 4); ctx.stroke(); }
        // shade fault region
        for (let i = s0; i < idx; i++) if (res.scenario[i] !== "normal") {
          ctx.fillStyle = color(res.scenario[i]) + "22";
          ctx.fillRect(((i - s0) / span) * W, 0, W / span + 1, H);
        }
        for (const [k, c] of series) {
          ctx.strokeStyle = c.trim(); ctx.lineWidth = 1.6 * dpr; ctx.beginPath();
          for (let i = s0; i < idx; i++) {
            const x = ((i - s0) / span) * W, y = H - ((rows[i][ci[k]] - lo) / (hi - lo)) * H;
            i === s0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
        // buffer + inference log
        const buf = idx < 128 ? idx : 64 + ((idx - 128) % 64);
        $("#buf-n").textContent = `${buf} / 128`;
        $("#buf-fill").style.width = (buf / 128) * 100 + "%";
        while (shown < wins.length && wins[shown].end_idx <= idx) {
          const w = wins[shown++];
          const li = document.createElement("li");
          li.style.setProperty("--fc", color(w.fault));
          li.innerHTML = `<span>#${pad(shown)}</span><span>${hhmm(w.t1_s)}</span><span class="lbl">${FAULTS[w.fault].code} · ${FAULTS[w.fault].name}</span><span class="cf">${(w.confidence * 100).toFixed(0)}%</span>`;
          $("#infer-log").prepend(li);
        }
        $("#t-clock").textContent = `${hhmm(idx * S.cfg.dt_s)} / ${hhmm(total)}`;
        $("#ro-pct").textContent = Math.round(p * 100) + "%";
        $("#ro-fill").style.strokeDashoffset = 326.7 * (1 - p);
        if (p < 1) requestAnimationFrame(frame); else resolve();
      }
      requestAnimationFrame(frame);
    });
  }

  /* ---------------- report ---------------- */
  function buildReport(res) {
    const d = res.diagnosis, dt = S.cfg.dt_s;
    const total = Object.values(d.by_fault_seconds).reduce((a, b) => a + b, 0) || 1;
    const faultSegs = d.segments.filter((s) => s.fault !== "normal");
    const detected = ORDER.filter((f) => (d.by_fault_seconds[f] || 0) > 0);
    const injected = ORDER.filter((f) => res.injected[f]);

    $("#r-count").textContent = faultSegs.length;
    $("#r-count-sub").textContent = `${detected.length} fault type${detected.length === 1 ? "" : "s"} · ${d.n_windows} windows scored`;
    $("#r-time").textContent = dur(d.fault_seconds);
    $("#r-time-sub").textContent = `${((d.fault_seconds / total) * 100).toFixed(1)}% of ${dur(total)} analysed`;
    $("#r-health").textContent = (d.health_ratio * 100).toFixed(1) + "%";
    $("#r-health").style.color = d.health_ratio > 0.8 ? "var(--ok)" : d.health_ratio > 0.5 ? "var(--amber)" : "var(--bad)";
    requestAnimationFrame(() => ($("#r-health-bar").style.width = d.health_ratio * 100 + "%"));

    // verdict: injected vs detected + time-to-detect
    const hit = injected.filter((f) => detected.includes(f));
    const missed = injected.filter((f) => !detected.includes(f));
    const extra = detected.filter((f) => !injected.includes(f));
    const ttd = injected.map((f) => {
      const i0 = res.scenario.indexOf(f);
      const w = d.windows.find((w) => w.fault === f && w.end_idx > i0);
      return w ? `${FAULTS[f].code} ${dur(Math.max(dt, w.t1_s - i0 * dt))}` : null;
    }).filter(Boolean);
    let cls = "ok", icon = "✓", title;
    if (!injected.length) { title = extra.length ? "Healthy run — model raised transient alerts" : "Healthy run confirmed — no fault detected"; cls = extra.length ? "part" : "ok"; icon = extra.length ? "!" : "✓"; }
    else if (!missed.length) title = `Model recognised ${hit.length === 1 ? "the injected fault" : `all ${hit.length} injected faults`}`;
    else if (hit.length) { cls = "part"; icon = "!"; title = `Model recognised ${hit.length} of ${injected.length} injected faults`; }
    else { cls = "bad"; icon = "✕"; title = "Injected fault was not recognised"; }
    $("#verdict").innerHTML = `
      <div class="v-icon ${cls}">${icon}</div>
      <div><div class="eyebrow">Verdict</div><h3>${title}</h3></div>
      <div class="v-rows">
        <div><span class="eyebrow">Injected</span><div class="chips">${injected.length ? injected.map((f) => chip(f, ` SL${res.injected[f]}`)).join("") : chip("normal")}</div></div>
        <div><span class="eyebrow">Detected</span><div class="chips">${detected.length ? detected.map((f) => chip(f)).join("") : chip("normal")}</div></div>
        ${ttd.length ? `<div><span class="eyebrow">Time to detect</span><div class="mono small">${ttd.join(" · ")}</div></div>` : ""}
        <div><span class="eyebrow">Run</span><div class="mono small">${S.hours}h · ${res.rows.length} samples @ ${dt}s</div></div>
      </div>`;

    // donut + list
    const slices = [...ORDER, "normal"].map((f) => [f, d.by_fault_seconds[f] || 0]).filter(([, s]) => s > 0);
    const R = 92, r = 62, cx = 110, cy = 110;
    let a0 = -Math.PI / 2, paths = "";
    for (const [f, s] of slices) {
      const a1 = a0 + (s / total) * Math.PI * 2 - (slices.length > 1 ? 0.02 : 0);
      const large = a1 - a0 > Math.PI ? 1 : 0;
      const p = (rad, ang) => `${cx + rad * Math.cos(ang)} ${cy + rad * Math.sin(ang)}`;
      const full = slices.length === 1;
      paths += full
        ? `<circle cx="${cx}" cy="${cy}" r="${(R + r) / 2}" fill="none" stroke="${color(f)}" stroke-width="${R - r}" data-f="${f}"/>`
        : `<path data-f="${f}" d="M${p(R, a0)} A${R} ${R} 0 ${large} 1 ${p(R, a1)} L${p(r, a1)} A${r} ${r} 0 ${large} 0 ${p(r, a0)}Z" fill="${color(f)}"/>`;
      a0 = a1 + (slices.length > 1 ? 0.02 : 0);
    }
    $("#donut").innerHTML = paths + `<text x="110" y="104" text-anchor="middle" class="svg-tag-k">ANALYSED</text><text x="110" y="128" text-anchor="middle" class="svg-tag" style="font-size:20px">${dur(total)}</text>`;
    $("#bd-list").innerHTML = slices.sort((a, b) => b[1] - a[1]).map(([f, s]) => {
      const ws = d.windows.filter((w) => w.fault === f);
      const conf = ws.reduce((a, w) => a + w.confidence, 0) / (ws.length || 1);
      return `<li data-f="${f}" style="--fc:${color(f)}"><span class="sw"></span>
        <span class="nm"><b>${FAULTS[f].code} · ${FAULTS[f].name}</b><span>${ws.length} windows · first at ${hhmm(ws[0].t0_s)} · conf ${(conf * 100).toFixed(0)}%</span></span>
        <span class="t">${dur(s)}</span><span class="p">${((s / total) * 100).toFixed(1)}%</span></li>`;
    }).join("");
    const hl = (f, on) => { $$("#donut [data-f]").forEach((el) => (el.style.opacity = on && el.dataset.f !== f ? 0.25 : 1)); $$("#bd-list li").forEach((li) => li.classList.toggle("hl", on && li.dataset.f === f)); };
    $$("#donut [data-f], #bd-list li").forEach((el) => { el.onmouseenter = () => hl(el.dataset.f, true); el.onmouseleave = () => hl(null, false); });

    buildTable(res);
    buildTimeline(res);
  }

  function predictedPerRow(res) {
    const N = res.rows.length, out = new Array(N).fill(null);
    for (const w of res.diagnosis.windows) for (let i = w.start_idx; i < w.end_idx; i++) out[i] = w; // later windows override
    return out;
  }

  const TABLE_COLS = ["TEI", "TEO", "TCI", "TCO", "TRE", "TRC", "TRC_sub", "Tsh_suc", "TCA", "kW", "PRE", "PRC", "P_lift", "PO_net", "TO_sump", "TR_dis", "CDW_DT", "CHW_DT"];
  function buildTable(res) {
    const ci = Object.fromEntries(res.columns.map((c, i) => [c, i]));
    const pred = predictedPerRow(res);
    let h = `<thead><tr><th>time</th><th>injected</th><th>predicted</th>${TABLE_COLS.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
    res.rows.forEach((r, i) => {
      const p = pred[i];
      h += `<tr><td>${hhmmss(r[0])}</td><td>${chip(res.scenario[i])}</td><td>${p ? chip(p.fault) : '<span class="muted">—</span>'}</td>${TABLE_COLS.map((c) => `<td>${r[ci[c]].toFixed(2)}</td>`).join("")}</tr>`;
    });
    $("#data-table").innerHTML = h + "</tbody>";
    $("#table-wrap").scrollTop = 0;
  }

  function download() {
    const res = S.result; if (!res) return;
    const pred = predictedPerRow(res);
    const head = ["time_s", "elapsed", "injected_fault", "predicted_fault", "confidence", ...res.columns.slice(1)];
    const lines = [head.join(",")];
    res.rows.forEach((r, i) => {
      const p = pred[i];
      lines.push([r[0], hhmmss(r[0]), FAULTS[res.scenario[i]].code, p ? FAULTS[p.fault].code : "", p ? p.confidence.toFixed(4) : "", ...r.slice(1)].join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `chiller_twin_${S.hours}h_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "")}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function buildTimeline(res) {
    const d = res.diagnosis, dt = S.cfg.dt_s;
    const W = 760, H = 520, L = 150, Rm = 18, T = 14, B = 70;
    const total = res.rows.length * dt;
    const laneH = (H - T - B) / LANES.length;
    const x = (t) => L + (t / total) * (W - L - Rm);
    let g = "";
    LANES.forEach((f, i) => {
      const y = T + i * laneH;
      g += `<rect x="${L}" y="${y + 2}" width="${W - L - Rm}" height="${laneH - 4}" rx="6" fill="rgba(255,255,255,${i % 2 ? 0.012 : 0.025})"/>`;
      g += `<circle cx="16" cy="${y + laneH / 2}" r="4" fill="${color(f)}"/>`;
      g += `<text x="28" y="${y + laneH / 2 + 4}" class="svg-tag" style="font-size:12px;font-weight:600">${FAULTS[f].code}</text>`;
      g += `<text x="62" y="${y + laneH / 2 + 4}" class="svg-tag-k" style="font-size:9.5px">${FAULTS[f].name.split(" ").slice(0, 2).join(" ").toUpperCase()}</text>`;
    });
    // time grid
    const stepS = total > 3 * 3600 ? 3600 : 1800;
    for (let t = 0; t <= total + 1; t += stepS) {
      g += `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B + 4}" stroke="rgba(148,163,184,.12)" stroke-dasharray="2 4"/>`;
      g += `<text x="${x(t)}" y="${H - B + 20}" text-anchor="middle" class="svg-tag-k">${hhmm(t)}</text>`;
    }
    g += `<text x="${(L + W - Rm) / 2}" y="${H - B + 38}" text-anchor="middle" class="svg-cap">ELAPSED TIME (h:mm)</text>`;
    // window bars: each window owns its stride slot, drawn with gaps (not connected)
    const stride = 64 * dt;
    d.windows.forEach((w, k) => {
      const lane = LANES.indexOf(w.fault);
      const t0 = w.t0_s, t1 = k === d.windows.length - 1 ? w.t1_s : Math.min(w.t1_s, t0 + stride);
      const x0 = x(t0) + 1.5, x1 = x(t1) - 1.5;
      const y = T + lane * laneH + laneH * 0.22;
      g += `<rect class="bar" data-k="${k}" x="${x0}" y="${y}" width="${Math.max(2, x1 - x0)}" height="${laneH * 0.56}" rx="4"
        fill="${color(w.fault)}" fill-opacity="${(0.35 + 0.65 * w.confidence).toFixed(2)}" style="animation:rise .4s ${k * 0.02}s both"/>`;
    });
    // injected ground-truth ribbon
    const ry = H - B + 48;
    g += `<text x="${L - 10}" y="${ry + 8}" text-anchor="end" class="svg-tag-k">INJECTED</text>`;
    let i0 = 0;
    for (let i = 1; i <= res.scenario.length; i++) {
      if (i === res.scenario.length || res.scenario[i] !== res.scenario[i0]) {
        const f = res.scenario[i0];
        g += `<rect x="${x(i0 * dt) + 1}" y="${ry}" width="${Math.max(2, x(i * dt) - x(i0 * dt) - 2)}" height="10" rx="3" fill="${color(f)}" opacity=".85"/>`;
        i0 = i;
      }
    }
    $("#timeline").innerHTML = g;
    $("#tl-meta").textContent = `${d.windows.length} windows · 128 × 10 s · stride 64`;
    const tip = $("#tip"), card = $(".timeline-card");
    $$("#timeline .bar").forEach((b) => {
      b.addEventListener("mousemove", (e) => {
        const w = d.windows[+b.dataset.k], rc = card.getBoundingClientRect();
        tip.hidden = false;
        tip.innerHTML = `<b style="color:${color(w.fault)}">${FAULTS[w.fault].code}</b> ${FAULTS[w.fault].name}<br>${hhmm(w.t0_s)} → ${hhmm(w.t1_s)}<br>confidence ${(w.confidence * 100).toFixed(1)}%`;
        tip.style.left = Math.min(e.clientX - rc.left + 14, rc.width - 220) + "px";
        tip.style.top = e.clientY - rc.top + 14 + "px";
      });
      b.addEventListener("mouseleave", () => (tip.hidden = true));
    });
  }

  /* ---------------- navigation ---------------- */
  function show(name) {
    $$(".screen").forEach((s) => s.classList.toggle("active", s.id === "screen-" + name));
    $("#step-1").classList.toggle("active", name === "sim");
    $("#step-2").classList.toggle("active", name === "report");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  function resetSliders() {
    for (const [v, p] of Object.entries(S.probes)) { S.values[v] = p.nominal; p.el.querySelector("input").value = p.nominal; }
    refresh();
  }

  /* ---------------- boot ---------------- */
  async function boot() {
    Chiller.mount($("#chiller"));
    try {
      S.cfg = await (await fetch("/api/config")).json();
    } catch (e) { toast("Cannot reach the twin backend."); return; }
    buildProbes();
    buildGuide();
    refresh();
    setInterval(() => { if (!S.running) { S.sensors = idleSensors(0.25); Chiller.update($("#chiller"), { sev: S.sevFrac, sensors: S.sensors, mode: "idle" }); paintKpis(S.sensors); } }, 1200);

    $$("#hours button").forEach((b) => b.addEventListener("click", () => { $$("#hours button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); S.hours = +b.dataset.h; }));
    $("#btn-run").addEventListener("click", run);
    $("#btn-reset").addEventListener("click", resetSliders);
    $("#btn-dl").addEventListener("click", download);
    $("#btn-back").addEventListener("click", () => show("sim"));
    $("#btn-new").addEventListener("click", () => { resetSliders(); pill("", "IDLE"); show("sim"); });
    $$(".step").forEach((s) => s.addEventListener("click", () => !s.disabled && show(s.dataset.screen)));
  }
  boot();
})();
