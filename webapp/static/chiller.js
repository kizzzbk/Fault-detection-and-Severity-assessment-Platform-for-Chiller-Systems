/* Animated centrifugal chiller (SVG, 1200 x 640 coordinate space).
 * Chiller.mount(svg) draws the machine; Chiller.update(state) drives the
 * fault-reactive visuals and live sensor tags. */
(function () {
  const NS = "http://www.w3.org/2000/svg";

  // Probe layout: card box (x, y in svg units, width = 234) + target point on the machine.
  const PROBES = {
    TCA:     { x: 14,  y: 16,  side: "L", target: [480, 300], component: "Condenser tubes" },
    CDW_DT:  { x: 14,  y: 226, side: "L", target: [300, 303], component: "Condenser water" },
    CHW_DT:  { x: 14,  y: 440, side: "L", target: [300, 474], component: "Chilled water" },
    PO_net:  { x: 276, y: 16,  side: "B", target: [548, 222], component: "Oil sump" },
    PRC:     { x: 952, y: 16,  side: "R", target: [790, 268], component: "Condenser shell" },
    TRC_sub: { x: 952, y: 250, side: "R", target: [822, 388], component: "Liquid line / EXV" },
  };
  const CARD_W = 234;

  function tubes(x0, x1, y0, y1, n, cls) {
    let s = "";
    for (let i = 0; i < n; i++) {
      const y = y0 + ((y1 - y0) * i) / (n - 1);
      s += `<line x1="${x0}" x2="${x1}" y1="${y}" y2="${y}" class="${cls}" />`;
    }
    return s;
  }

  function shell(id, x, y, w, h, label) {
    return `
      <g id="${id}">
        <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${h / 2}" fill="url(#steel)" stroke="#3a4656" stroke-width="1.5"/>
        <rect x="${x + 8}" y="${y + 6}" width="${w - 16}" height="${h * 0.22}" rx="${h * 0.11}" fill="url(#sheen)" opacity=".5"/>
        <rect x="${x - 22}" y="${y + 6}" width="34" height="${h - 12}" rx="8" fill="#232c38" stroke="#3a4656"/>
        <rect x="${x + w - 12}" y="${y + 6}" width="34" height="${h - 12}" rx="8" fill="#232c38" stroke="#3a4656"/>
        ${[0.18, 0.5, 0.82].map((f) => `<line x1="${x + w * f}" x2="${x + w * f}" y1="${y + 2}" y2="${y + h - 2}" stroke="#0d1117" stroke-width="2" opacity=".6"/>`).join("")}
        <text x="${x + w / 2}" y="${y + h + 16}" text-anchor="middle" class="svg-cap">${label}</text>
      </g>`;
  }

  function mount(svg) {
    svg.innerHTML = `
    <defs>
      <linearGradient id="steel" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#3b4655"/><stop offset=".45" stop-color="#262f3b"/><stop offset="1" stop-color="#151b23"/>
      </linearGradient>
      <linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".22"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
      <linearGradient id="motor" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#425063"/><stop offset="1" stop-color="#1b222c"/></linearGradient>
      <radialGradient id="volute" cx=".4" cy=".35"><stop offset="0" stop-color="#4a586b"/><stop offset="1" stop-color="#1a212b"/></radialGradient>
      <linearGradient id="warm" x1="0" x2="1"><stop offset="0" stop-color="#fb923c"/><stop offset="1" stop-color="#f97316"/></linearGradient>
      <linearGradient id="cool" x1="0" x2="1"><stop offset="0" stop-color="#38bdf8"/><stop offset="1" stop-color="#22d3ee"/></linearGradient>
      <linearGradient id="liquid" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#22d3ee" stop-opacity=".55"/><stop offset="1" stop-color="#0e7490" stop-opacity=".85"/></linearGradient>
      <linearGradient id="oil" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fbbf24"/><stop offset="1" stop-color="#b45309"/></linearGradient>
      <radialGradient id="ncgas" cx=".5" cy=".3" r=".7"><stop offset="0" stop-color="#facc15" stop-opacity=".7"/><stop offset="1" stop-color="#facc15" stop-opacity="0"/></radialGradient>
      <clipPath id="clipCond"><rect x="342" y="252" width="516" height="101" rx="50"/></clipPath>
      <clipPath id="clipEvap"><rect x="342" y="422" width="516" height="106" rx="53"/></clipPath>
      <clipPath id="clipSump"><rect x="546" y="207" width="108" height="31" rx="6"/></clipPath>
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6"/></filter>
      <pattern id="floor" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1a2230" stroke-width="1"/></pattern>
    </defs>

    <!-- floor -->
    <ellipse cx="610" cy="566" rx="360" ry="22" fill="#000" opacity=".45" filter="url(#glow)"/>
    <rect x="320" y="548" width="580" height="12" rx="3" fill="#1c232d" stroke="#2f3a48"/>
    ${[340, 480, 740, 880].map((x) => `<rect x="${x - 10}" y="558" width="20" height="10" fill="#151b23"/>`).join("")}

    <!-- supports -->
    ${[420, 780].map((x) => `<rect x="${x - 12}" y="352" width="24" height="72" fill="#1a212b" stroke="#2f3a48"/>`).join("")}
    ${[420, 780].map((x) => `<rect x="${x - 12}" y="528" width="24" height="22" fill="#1a212b" stroke="#2f3a48"/>`).join("")}

    <!-- ===== water circuits ===== -->
    <g id="cw">
      <path d="M256 282 H330" stroke="#3a1f10" stroke-width="16" fill="none" stroke-linecap="round"/>
      <path d="M256 326 H330" stroke="#3a1f10" stroke-width="16" fill="none" stroke-linecap="round"/>
      <path d="M256 282 H330" stroke="url(#warm)" stroke-width="5" fill="none" class="flow" id="cw-in"/>
      <path d="M330 326 H256" stroke="url(#warm)" stroke-width="5" fill="none" class="flow" id="cw-out"/>
      <path d="M260 282 l8 -5 v10z" fill="#fb923c" opacity="0"/>
    </g>
    <g id="chw">
      <path d="M256 452 H330" stroke="#0c2a36" stroke-width="16" fill="none" stroke-linecap="round"/>
      <path d="M256 498 H330" stroke="#0c2a36" stroke-width="16" fill="none" stroke-linecap="round"/>
      <path d="M256 452 H330" stroke="url(#cool)" stroke-width="5" fill="none" class="flow" id="chw-in"/>
      <path d="M330 498 H256" stroke="url(#cool)" stroke-width="5" fill="none" class="flow" id="chw-out"/>
    </g>

    <!-- ===== shells ===== -->
    ${shell("cond", 340, 250, 520, 105, "CONDENSER")}
    ${shell("evap", 340, 420, 520, 110, "EVAPORATOR")}

    <g clip-path="url(#clipCond)">
      <g id="cond-tubes">${tubes(350, 850, 266, 340, 8, "ctube")}</g>
      <rect id="cond-liquid" x="342" y="330" width="516" height="30" fill="url(#liquid)"/>
      <path id="cond-wave" d="" fill="#67e8f9" opacity=".35"/>
      <ellipse id="nc-gas" cx="760" cy="262" rx="120" ry="40" fill="url(#ncgas)" opacity="0"/>
      <g id="nc-dots" opacity="0">${Array.from({ length: 14 }, (_, i) => `<circle cx="${650 + (i * 37) % 200}" cy="${258 + (i * 13) % 26}" r="${2 + (i % 3)}" fill="#fde047" class="bubble" style="animation-delay:${-i * 0.23}s;animation-duration:${2.2 + (i % 4) * 0.3}s"/>`).join("")}</g>
    </g>
    <g clip-path="url(#clipEvap)">
      <g id="evap-tubes">${tubes(350, 850, 440, 512, 8, "etube")}</g>
      <rect x="342" y="482" width="516" height="50" fill="url(#liquid)" opacity=".5"/>
      <g id="evap-boil">${Array.from({ length: 18 }, (_, i) => `<circle cx="${370 + i * 27}" cy="${505 - (i % 3) * 6}" r="${1.5 + (i % 3)}" fill="#a5f3fc" class="bubble" style="animation-delay:${-i * 0.17}s"/>`).join("")}</g>
    </g>

    <text x="600" y="296" text-anchor="middle" class="svg-tag" id="tag-cond">TRC –  ·  PRC –</text>
    <text x="600" y="468" text-anchor="middle" class="svg-tag" id="tag-evap">TRE –  ·  PRE –</text>

    <!-- ===== refrigerant: suction (evap -> compressor) ===== -->
    <path d="M858 470 H930 V152 H780" stroke="#13303d" stroke-width="26" fill="none" stroke-linejoin="round"/>
    <path d="M858 470 H930 V152 H780" stroke="#7dd3fc" stroke-width="4" fill="none" class="flow" id="suction" opacity=".85" stroke-linejoin="round"/>
    <text x="944" y="320" class="svg-tag-k" transform="rotate(90 944 320)">SUCTION VAPOR</text>

    <!-- discharge (compressor -> condenser) -->
    <path d="M700 196 V252" stroke="#3a1414" stroke-width="24" fill="none"/>
    <path d="M700 196 V252" stroke="#f87171" stroke-width="4" fill="none" class="flow" id="discharge"/>
    <text x="712" y="232" class="svg-tag-k" id="tag-dis">TR_dis –</text>

    <!-- liquid line + EXV (condenser -> evaporator) -->
    <path d="M822 352 V424" stroke="#0c2a36" stroke-width="16" fill="none"/>
    <path d="M822 352 V424" stroke="#22d3ee" stroke-width="4" fill="none" class="flow slow" id="liqline"/>
    <g id="exv" transform="translate(822 388)">
      <path d="M-14 -11 L14 11 L14 -11 L-14 11 Z" fill="#1f2935" stroke="#7dd3fc" stroke-width="1.5"/>
      <rect x="-4" y="-24" width="8" height="12" fill="#2f3a48"/><rect x="-9" y="-30" width="18" height="7" rx="2" fill="#3a4656"/>
    </g>
    <g id="flash" opacity="0">${Array.from({ length: 6 }, (_, i) => `<circle cx="${818 + (i % 2) * 7}" cy="${418 - i * 5}" r="${1.6 + (i % 2)}" fill="#e0f2fe" class="bubble" style="animation-delay:${-i * 0.26}s"/>`).join("")}</g>

    <!-- ===== compressor + motor ===== -->
    <g id="compressor">
      <rect x="500" y="104" width="160" height="96" rx="18" fill="url(#motor)" stroke="#4b5a6e" stroke-width="1.5"/>
      ${Array.from({ length: 9 }, (_, i) => `<line x1="${518 + i * 15}" x2="${518 + i * 15}" y1="112" y2="192" stroke="#151b23" stroke-width="3" opacity=".55"/>`).join("")}
      <rect x="486" y="128" width="16" height="48" rx="4" fill="#2a3340" stroke="#4b5a6e"/>
      <text x="580" y="98" text-anchor="middle" class="svg-cap">MOTOR</text>
      <text x="580" y="158" text-anchor="middle" class="svg-tag" id="tag-kw">– kW</text>
      <circle cx="720" cy="152" r="58" fill="url(#volute)" stroke="#5b6b80" stroke-width="2"/>
      <circle cx="720" cy="152" r="44" fill="#0e131a" stroke="#2f3a48"/>
      <g class="rotor" id="impeller">
        ${Array.from({ length: 10 }, (_, i) => `<path d="M720 152 q 14 -18 36 -12" transform="rotate(${i * 36} 720 152)" stroke="#7dd3fc" stroke-width="3" fill="none" stroke-linecap="round" opacity=".9"/>`).join("")}
        <circle cx="720" cy="152" r="9" fill="#cbd5e1"/>
      </g>
      <circle cx="720" cy="152" r="44" fill="none" stroke="#22d3ee" stroke-width="1" opacity=".25" id="volute-glow"/>
      <text x="720" y="82" text-anchor="middle" class="svg-cap">COMPRESSOR</text>
    </g>

    <!-- oil sump -->
    <g id="sump">
      <rect x="544" y="205" width="112" height="35" rx="7" fill="#151b23" stroke="#4b5a6e"/>
      <g clip-path="url(#clipSump)"><rect id="oil-level" x="546" y="222" width="108" height="20" fill="url(#oil)" opacity=".85"/></g>
      <text x="600" y="227" text-anchor="middle" class="svg-tag-k" fill="#111" id="tag-oil">OIL –</text>
    </g>

    <!-- water tags -->
    <text x="258" y="270" class="svg-tag-k">TCI</text><text x="286" y="270" class="svg-tag" id="tag-tci">–</text>
    <text x="258" y="346" class="svg-tag-k">TCO</text><text x="286" y="346" class="svg-tag" id="tag-tco">–</text>
    <text x="258" y="440" class="svg-tag-k">TEI</text><text x="286" y="440" class="svg-tag" id="tag-tei">–</text>
    <text x="258" y="518" class="svg-tag-k">TEO</text><text x="286" y="518" class="svg-tag" id="tag-teo">–</text>
    <text x="1185" y="626" text-anchor="end" class="svg-cap">RP-1043 · 90 TON · R134a</text>

    <g id="leaders"></g>
    `;
    drawLeaders(svg);
  }

  function drawLeaders(svg) {
    const g = svg.querySelector("#leaders");
    g.innerHTML = Object.entries(PROBES).map(([v, p]) => {
      const [tx, ty] = p.target;
      let sx, sy, d;
      if (p.side === "L") { sx = p.x + CARD_W; sy = p.y + 52; d = `M${sx} ${sy} H${(sx + tx) / 2} L${tx} ${ty}`; }
      else if (p.side === "R") { sx = p.x; sy = p.y + 52; d = `M${sx} ${sy} H${(sx + tx) / 2 + 30} L${tx} ${ty}`; }
      else { sx = p.x + CARD_W; sy = p.y + 52; d = `M${sx} ${sy} H${tx} V${ty}`; }
      return `<path class="leader" data-var="${v}" d="${d}"/><circle class="leader-dot" data-var="${v}" cx="${tx}" cy="${ty}" r="4"/>`;
    }).join("");
  }

  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  function mix(c1, c2, t) {
    const p = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
    const a = p(c1), b = p(c2);
    return "#" + a.map((v, i) => Math.round(lerp(v, b[i], t)).toString(16).padStart(2, "0")).join("");
  }
  const fmt = (v, d = 1) => (v == null || isNaN(v) ? "–" : Number(v).toFixed(d));

  /* state: { sev: {fwc,fwe,cf,ro,rl,nc,eo} as 0..1, sensors: {...}, mode } */
  function update(svg, state) {
    const s = state.sev || {};
    const q = (id) => svg.querySelector(id);
    svg.classList.toggle("running", state.mode === "running");
    svg.classList.toggle("idle", state.mode === "idle");

    // CF: tubes foul from steel-cyan to rust brown
    const foul = clamp01(s.cf || 0);
    svg.querySelectorAll(".ctube").forEach((l, i) => {
      l.setAttribute("stroke", mix("#4fb3c9", "#9a4a14", foul));
      l.setAttribute("stroke-width", (2 + foul * 3.5).toFixed(1));
      l.setAttribute("opacity", ".75");
    });
    svg.querySelectorAll(".etube").forEach((l) => { l.setAttribute("stroke", "#4fb3c9"); l.setAttribute("stroke-width", 2); l.setAttribute("opacity", ".6"); });

    // RO raises / RL drains the condenser liquid pool
    const level = clamp01(0.28 + 0.45 * (s.ro || 0) - 0.26 * (s.rl || 0));
    const top = 353 - level * 100;
    q("#cond-liquid").setAttribute("y", top.toFixed(1));
    q("#cond-liquid").setAttribute("height", (360 - top).toFixed(1));
    q("#flash").setAttribute("opacity", clamp01((s.rl || 0) * 1.4).toFixed(2));

    // NC: yellow non-condensable pocket at top of condenser
    q("#nc-gas").setAttribute("opacity", clamp01((s.nc || 0) * 1.1).toFixed(2));
    q("#nc-dots").setAttribute("opacity", clamp01((s.nc || 0) * 1.2).toFixed(2));

    // EO: oil level climbs
    const oil = clamp01(0.45 + 0.5 * (s.eo || 0));
    q("#oil-level").setAttribute("y", (240 - oil * 33).toFixed(1));

    // FWC / FWE: slower water flow = slower dash + thinner stroke
    const speed = (f) => (state.mode === "running" ? 0.6 : 1.4) * (1 + 2.5 * clamp01(f));
    ["#cw-in", "#cw-out"].forEach((id) => { q(id).style.animationDuration = speed(s.fwc) + "s"; q(id).setAttribute("stroke-width", (5 - 3 * clamp01(s.fwc || 0)).toFixed(1)); });
    ["#chw-in", "#chw-out"].forEach((id) => { q(id).style.animationDuration = speed(s.fwe) + "s"; q(id).setAttribute("stroke-width", (5 - 3 * clamp01(s.fwe || 0)).toFixed(1)); });

    // discharge heats up with lift
    const lift = clamp01(Math.max(s.cf || 0, s.nc || 0, s.ro || 0, s.fwc || 0));
    q("#discharge").setAttribute("stroke", mix("#f87171", "#fde047", lift));
    q("#volute-glow").setAttribute("stroke", mix("#22d3ee", "#f87171", lift));
    q("#volute-glow").setAttribute("opacity", (0.25 + lift * 0.6).toFixed(2));

    const v = state.sensors || {};
    q("#tag-cond").textContent = `TRC ${fmt(v.TRC)}°F  ·  PRC ${fmt(v.PRC)} psig`;
    q("#tag-evap").textContent = `TRE ${fmt(v.TRE)}°F  ·  PRE ${fmt(v.PRE)} psig`;
    q("#tag-kw").textContent = `${fmt(v.kW)} kW`;
    q("#tag-dis").textContent = `TR_dis ${fmt(v.TR_dis, 0)}°F`;
    q("#tag-oil").textContent = `ΔP ${fmt(v.PO_net, 0)} psi`;
    q("#tag-tci").textContent = `${fmt(v.TCI)}°F`;
    q("#tag-tco").textContent = `${fmt(v.TCO)}°F`;
    q("#tag-tei").textContent = `${fmt(v.TEI)}°F`;
    q("#tag-teo").textContent = `${fmt(v.TEO)}°F`;
  }

  function setLeader(svg, variable, color) {
    svg.querySelectorAll(`[data-var="${variable}"]`).forEach((el) => {
      el.classList.toggle("armed", !!color);
      el.style.setProperty("--fc", color || "");
    });
  }

  window.Chiller = { PROBES, CARD_W, mount, update, setLeader };
})();
