(function () {
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const ARMS = ["blind_first", "control_always_ai", "disagreement_prompt", "withheld_ai"];
const ARM_VAR = { blind_first: "--series-1", control_always_ai: "--series-2", disagreement_prompt: "--series-3", withheld_ai: "--series-4" };
const DOMAINS = ["chest_xray_triage", "code_review"];
const STATUS = {
  healthy: { color: "--status-good", label: "healthy" },
  watch: { color: "--status-warning", label: "watch" },
  under_reliant: { color: "--status-serious", label: "under-reliant" },
  over_reliance_risk: { color: "--status-critical", label: "over-reliance risk" },
};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function fmt(x, d) { return (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d === undefined ? 3 : d); }
function pct(x, d) { return (x === null || x === undefined || Number.isNaN(x)) ? "—" : (Number(x) * 100).toFixed(d === undefined ? 1 : d) + "%"; }
function svgEl(tag, attrs) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function el(tag, attrs, text) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) {
    if (k === "class") e.className = attrs[k];
    else if (k === "style") e.setAttribute("style", attrs[k]);
    else e.setAttribute(k, attrs[k]);
  }
  if (text !== undefined) e.textContent = text;
  return e;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  arms: new Set(ARMS),
  domain: "all",
  selectedReviewer: null,
  sortB: { col: "risk_score", dir: -1 },
  sortD: { col: "n_cases_made_worse", dir: -1 },
  expandedB: new Set(),
};

function filteredReviewerIds() {
  return Object.keys(DATA.reviewers).filter((rid) => {
    const r = DATA.reviewers[rid];
    if (!state.arms.has(r.arm)) return false;
    if (state.domain !== "all" && r.domain !== state.domain) return false;
    return true;
  }).sort();
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------
const tooltipEl = document.getElementById("tooltip");
function showTooltip(x, y, html) {
  tooltipEl.innerHTML = "";
  tooltipEl.appendChild(html);
  tooltipEl.style.left = (x + 14) + "px";
  tooltipEl.style.top = (y + 14) + "px";
  tooltipEl.classList.add("show");
}
function hideTooltip() { tooltipEl.classList.remove("show"); }
function ttTitle(text) { const d = el("div", { class: "tt-title" }, text); return d; }
function ttRow(keyColor, label, value) {
  const row = el("div", { class: "tt-row" });
  const l = el("span", {});
  if (keyColor) {
    const key = el("span", { class: "tt-key", style: `background:${keyColor}` });
    l.appendChild(key);
  }
  l.appendChild(document.createTextNode(label));
  const v = el("span", { class: "v" }, value);
  row.appendChild(l); row.appendChild(v);
  return row;
}

// ---------------------------------------------------------------------------
// Stat row
// ---------------------------------------------------------------------------
function renderStats() {
  const row = document.getElementById("statRow");
  row.innerHTML = "";
  const nFlagged = DATA.interventions.filter((r) => r.state !== "healthy").length;
  const sv = DATA.self_validation_summary;
  const stats = [
    { label: "Reviewers", value: Object.keys(DATA.reviewers).length, sub: "60 across 4 arms, 2 domains" },
    { label: "Self-validation mean rho", value: fmt(sv.mean, 3), sub: `±${fmt(sv.std, 3)} across ${sv.n_seeds} seeds` },
    { label: "Flagged for intervention", value: nFlagged, sub: `of ${DATA.interventions.length} reviewers` },
    { label: "D4 cases made worse", value: Math.round(DATA.cost_groups.reduce((s, g) => s + g.cases_made_worse, 0)).toLocaleString(), sub: "cost side, aggregate" },
    { label: "D4 skill-gain units", value: fmt(DATA.cost_groups.reduce((s, g) => s + g.skill_gain_units, 0), 1), sub: "benefit side, aggregate" },
  ];
  stats.forEach((s) => {
    const box = el("div", { class: "stat" });
    box.appendChild(el("div", { class: "label" }, s.label));
    box.appendChild(el("div", { class: "value" }, String(s.value)));
    box.appendChild(el("div", { class: "sub" }, s.sub));
    row.appendChild(box);
  });
}

// ---------------------------------------------------------------------------
// Filter row
// ---------------------------------------------------------------------------
function renderFilters() {
  const row = document.getElementById("filterRow");
  row.innerHTML = "";

  const armGroup = el("div", { class: "filter-group" });
  armGroup.appendChild(el("span", { class: "fg-label" }, "Arm"));
  ARMS.forEach((arm) => {
    const active = state.arms.has(arm);
    const btn = el("button", { class: "toggle-btn" + (active ? " active" : ""), type: "button" });
    if (active) btn.style.background = `var(${ARM_VAR[arm]})`;
    btn.appendChild(el("span", { class: "sw", style: active ? "" : `color:var(${ARM_VAR[arm]})` }));
    btn.appendChild(document.createTextNode(arm.replace(/_/g, " ")));
    btn.addEventListener("click", () => {
      if (state.arms.has(arm) && state.arms.size === 1) return; // keep at least one
      state.arms.has(arm) ? state.arms.delete(arm) : state.arms.add(arm);
      renderAll();
    });
    armGroup.appendChild(btn);
  });
  row.appendChild(armGroup);

  const domGroup = el("div", { class: "filter-group" });
  domGroup.appendChild(el("span", { class: "fg-label" }, "Domain"));
  const domSel = el("select", { class: "plain" });
  [["all", "All domains"], ["chest_xray_triage", "chest_xray_triage"], ["code_review", "code_review"]].forEach(([v, label]) => {
    const opt = el("option", { value: v }, label);
    if (v === state.domain) opt.selected = true;
    domSel.appendChild(opt);
  });
  domSel.addEventListener("change", () => { state.domain = domSel.value; renderAll(); });
  domGroup.appendChild(domSel);
  row.appendChild(domGroup);
}

// ---------------------------------------------------------------------------
// Panel A — capability trajectory
// ---------------------------------------------------------------------------
function renderReviewerSelect() {
  const sel = document.getElementById("reviewerSelect");
  const ids = filteredReviewerIds();
  if (!state.selectedReviewer || !ids.includes(state.selectedReviewer)) state.selectedReviewer = ids[0];
  sel.innerHTML = "";
  ids.forEach((rid) => {
    const r = DATA.reviewers[rid];
    const opt = el("option", { value: rid }, `${rid} — ${r.arm.replace(/_/g, " ")}`);
    if (rid === state.selectedReviewer) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.onchange = () => { state.selectedReviewer = sel.value; renderTrajectory(); };
}

function renderTrajLegend() {
  const wrap = document.getElementById("trajLegend");
  wrap.innerHTML = "";
  ARMS.forEach((arm) => {
    const item = el("div", { class: "legend-item" });
    item.appendChild(el("span", { class: "legend-key", style: `background:var(${ARM_VAR[arm]})` }));
    item.appendChild(document.createTextNode(arm.replace(/_/g, " ")));
    wrap.appendChild(item);
  });
}

function renderTrajectory() {
  const svg = document.getElementById("trajChart");
  svg.innerHTML = "";
  const W = 980, H = 320, M = { l: 46, r: 16, t: 14, b: 30 };
  const plotW = W - M.l - M.r, plotH = H - M.t - M.b;

  const ids = filteredReviewerIds();
  const byReviewer = {};
  DATA.trajectory.forEach((row) => {
    if (!ids.includes(row.reviewer_id)) return;
    (byReviewer[row.reviewer_id] = byReviewer[row.reviewer_id] || []).push(row);
  });
  ids.forEach((rid) => byReviewer[rid] && byReviewer[rid].sort((a, b) => a.week - b.week));

  let yMin = Infinity, yMax = -Infinity;
  Object.values(byReviewer).forEach((rows) => rows.forEach((r) => {
    yMin = Math.min(yMin, r.interval_lo); yMax = Math.max(yMax, r.interval_hi);
  }));
  if (!isFinite(yMin)) { yMin = -1; yMax = 1; }
  const pad = (yMax - yMin) * 0.08;
  yMin -= pad; yMax += pad;

  const xScale = (wk) => M.l + ((wk - 1) / 23) * plotW;
  const yScale = (v) => M.t + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  // gridlines
  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) {
    const v = yMin + (i / nTicks) * (yMax - yMin);
    const y = yScale(v);
    svg.appendChild(svgEl("line", { x1: M.l, x2: W - M.r, y1: y, y2: y, class: "gridline" }));
    const lab = svgEl("text", { x: 4, y: y + 3, class: "axis-label" });
    lab.textContent = fmt(v, 2);
    svg.appendChild(lab);
  }
  [1, 6, 12, 18, 24].forEach((wk) => {
    const x = xScale(wk);
    const lab = svgEl("text", { x: x, y: H - 8, class: "axis-label", "text-anchor": "middle" });
    lab.textContent = "wk " + wk;
    svg.appendChild(lab);
  });
  svg.appendChild(svgEl("line", { x1: M.l, x2: W - M.r, y1: yScale(0), y2: yScale(0), class: "baseline" }));

  // background lines for every filtered reviewer except selected
  ids.forEach((rid) => {
    if (rid === state.selectedReviewer) return;
    const rows = byReviewer[rid]; if (!rows || !rows.length) return;
    const arm = DATA.reviewers[rid].arm;
    const d = rows.map((r, i) => `${i === 0 ? "M" : "L"}${xScale(r.week).toFixed(1)},${yScale(r.capability_estimate).toFixed(1)}`).join(" ");
    const path = svgEl("path", { d, fill: "none", style: `stroke:var(${ARM_VAR[arm]});stroke-width:1;opacity:0.22`, "stroke-linecap": "round" });
    path.addEventListener("mouseenter", (e) => {
      const box = svg.getBoundingClientRect();
      const tt = document.createDocumentFragment();
      tt.appendChild(ttTitle(rid + " — " + arm.replace(/_/g, " ")));
      showTooltip(e.clientX, e.clientY, tt);
    });
    path.addEventListener("mousemove", (e) => { tooltipEl.style.left = (e.clientX + 14) + "px"; tooltipEl.style.top = (e.clientY + 14) + "px"; });
    path.addEventListener("mouseleave", hideTooltip);
    path.addEventListener("click", () => { state.selectedReviewer = rid; renderReviewerSelect(); renderTrajectory(); });
    path.style.cursor = "pointer";
    svg.appendChild(path);
  });

  // selected reviewer: band + line + markers
  const selRows = byReviewer[state.selectedReviewer];
  if (selRows && selRows.length) {
    const arm = DATA.reviewers[state.selectedReviewer].arm;
    const colorVar = `var(${ARM_VAR[arm]})`;
    const top = selRows.map((r, i) => `${i === 0 ? "M" : "L"}${xScale(r.week).toFixed(1)},${yScale(r.interval_hi).toFixed(1)}`).join(" ");
    const bottom = selRows.slice().reverse().map((r) => `L${xScale(r.week).toFixed(1)},${yScale(r.interval_lo).toFixed(1)}`).join(" ");
    svg.appendChild(svgEl("path", { d: top + " " + bottom + " Z", style: `fill:${colorVar};opacity:0.10`, stroke: "none" }));

    const line = selRows.map((r, i) => `${i === 0 ? "M" : "L"}${xScale(r.week).toFixed(1)},${yScale(r.capability_estimate).toFixed(1)}`).join(" ");
    svg.appendChild(svgEl("path", { d: line, fill: "none", style: `stroke:${colorVar};stroke-width:2`, "stroke-linecap": "round", "stroke-linejoin": "round" }));

    selRows.forEach((r) => {
      const cx = xScale(r.week), cy = yScale(r.capability_estimate);
      const dot = svgEl("circle", { cx, cy, r: 4.5, style: `fill:${colorVar}`, stroke: cssVar("--surface-1"), "stroke-width": 2 });
      const hit = svgEl("circle", { cx, cy, r: 12, fill: "transparent", style: "cursor:pointer" });
      hit.addEventListener("mouseenter", (e) => {
        const tt = document.createDocumentFragment();
        tt.appendChild(ttTitle(`${state.selectedReviewer} · week ${r.week}`));
        tt.appendChild(ttRow(colorVar, "capability_estimate", fmt(r.capability_estimate, 3)));
        tt.appendChild(ttRow(null, "95% interval", `[${fmt(r.interval_lo, 2)}, ${fmt(r.interval_hi, 2)}]`));
        tt.appendChild(ttRow(null, "deferred_rate", pct(r.deferred_rate, 0)));
        tt.appendChild(ttRow(null, "committed_rate", pct(r.committed_rate, 0)));
        tt.appendChild(ttRow(null, "blind_sample_n", String(r.blind_sample_n)));
        showTooltip(e.clientX, e.clientY, tt);
      });
      hit.addEventListener("mousemove", (e) => { tooltipEl.style.left = (e.clientX + 14) + "px"; tooltipEl.style.top = (e.clientY + 14) + "px"; });
      hit.addEventListener("mouseleave", hideTooltip);
      svg.appendChild(dot); svg.appendChild(hit);
    });
  }
}

// ---------------------------------------------------------------------------
// Panel B — intervention table
// ---------------------------------------------------------------------------
const B_COLS = [
  { key: "reviewer_id", label: "Reviewer" },
  { key: "arm", label: "Arm" },
  { key: "domain", label: "Domain" },
  { key: "state", label: "State" },
  { key: "risk_score", label: "Risk" },
  { key: "capability_slope_per_week", label: "Slope/wk" },
  { key: "recent_deferred_rate", label: "Deferred" },
  { key: "intervention", label: "Intervention" },
];

function renderInterventionTable() {
  const ids = new Set(filteredReviewerIds());
  let rows = DATA.interventions.filter((r) => ids.has(r.reviewer_id));
  const { col, dir } = state.sortB;
  rows = rows.slice().sort((a, b) => {
    const av = a[col], bv = b[col];
    if (typeof av === "string") return dir * av.localeCompare(bv);
    return dir * ((av || 0) - (bv || 0));
  });

  const table = document.getElementById("interventionTable");
  table.innerHTML = "";
  const thead = el("thead"); const trh = el("tr");
  B_COLS.forEach((c) => {
    const th = el("th", { class: c.key === col ? "sorted" : "" }, c.label);
    if (c.key === col) th.setAttribute("data-arrow", dir === 1 ? "▲" : "▼");
    th.addEventListener("click", () => {
      if (state.sortB.col === c.key) state.sortB.dir *= -1; else state.sortB = { col: c.key, dir: -1 };
      renderInterventionTable();
    });
    trh.appendChild(th);
  });
  thead.appendChild(trh); table.appendChild(thead);

  const tbody = el("tbody");
  rows.forEach((r) => {
    const tr = el("tr", { style: "cursor:pointer" });
    tr.appendChild(el("td", {}, r.reviewer_id));
    const armTd = el("td", {});
    armTd.appendChild(el("span", { class: "badge", style: `background:color-mix(in srgb, var(${ARM_VAR[r.arm]}) 18%, transparent); color:var(${ARM_VAR[r.arm]})` }, r.arm.replace(/_/g, " ")));
    tr.appendChild(armTd);
    tr.appendChild(el("td", {}, r.domain));
    const stTd = el("td", {});
    const st = STATUS[r.state] || { color: "--text-muted", label: r.state };
    const badge = el("span", { class: "badge", style: "color:var(--text-primary); background:var(--page)" });
    badge.appendChild(el("span", { class: "dot", style: `background:var(${st.color})` }));
    badge.appendChild(document.createTextNode(st.label));
    stTd.appendChild(badge);
    tr.appendChild(stTd);
    tr.appendChild(el("td", { class: "num" }, String(r.risk_score)));
    tr.appendChild(el("td", { class: "num" }, fmt(r.capability_slope_per_week, 4)));
    tr.appendChild(el("td", { class: "num" }, pct(r.recent_deferred_rate, 0)));
    tr.appendChild(el("td", {}, r.intervention));
    tr.addEventListener("click", () => {
      state.expandedB.has(r.reviewer_id) ? state.expandedB.delete(r.reviewer_id) : state.expandedB.add(r.reviewer_id);
      renderInterventionTable();
    });
    tbody.appendChild(tr);

    if (state.expandedB.has(r.reviewer_id)) {
      const dtr = el("tr", { class: "detail-row" });
      const dtd = el("td", { colspan: String(B_COLS.length) });
      dtd.appendChild(el("div", {}, "Detail: " + r.intervention_detail));
      dtd.appendChild(el("div", { style: "margin-top:4px;" }, "Frequency: " + r.frequency));
      dtd.appendChild(el("div", { style: "margin-top:4px;" }, "Stop condition: " + r.stop_condition));
      dtd.appendChild(el("div", { style: "margin-top:4px;" },
        `under_reliance_rate ${pct(r.under_reliance_rate,1)} · appropriate_skepticism_rate ${pct(r.appropriate_skepticism_rate,1)} · over_reliance_rate ${pct(r.over_reliance_rate,1)}`));
      dtr.appendChild(dtd);
      tbody.appendChild(dtr);
    }
  });
  table.appendChild(tbody);
}

// ---------------------------------------------------------------------------
// Panel C — D3 predictions + self-validation
// ---------------------------------------------------------------------------
function renderD3Legend() {
  const wrap = document.getElementById("d3Legend");
  wrap.innerHTML = "";
  ARMS.forEach((arm) => {
    const item = el("div", { class: "legend-item" });
    item.appendChild(el("span", { class: "legend-dot", style: `background:var(${ARM_VAR[arm]})` }));
    item.appendChild(document.createTextNode(arm.replace(/_/g, " ")));
    wrap.appendChild(item);
  });
}

function renderD3Chart() {
  const svg = document.getElementById("d3Chart");
  svg.innerHTML = "";
  const W = 980, H = 280, M = { l: 44, r: 16, t: 14, b: 26 };
  const plotW = W - M.l - M.r, plotH = H - M.t - M.b;

  const ids = new Set(filteredReviewerIds());
  const rows = DATA.d3.filter((r) => ids.has(r.reviewer_id)).slice().sort((a, b) => b.predicted_exam_accuracy - a.predicted_exam_accuracy);
  if (!rows.length) return;

  const yMin = 0.3, yMax = 1.0;
  const yScale = (v) => M.t + (1 - (v - yMin) / (yMax - yMin)) * plotH;
  const xStep = plotW / Math.max(rows.length, 1);
  const xScale = (i) => M.l + (i + 0.5) * xStep;

  [0.4, 0.6, 0.8, 1.0].forEach((v) => {
    const y = yScale(v);
    svg.appendChild(svgEl("line", { x1: M.l, x2: W - M.r, y1: y, y2: y, class: "gridline" }));
    const lab = svgEl("text", { x: 4, y: y + 3, class: "axis-label" }); lab.textContent = pct(v, 0);
    svg.appendChild(lab);
  });

  rows.forEach((r, i) => {
    const x = xScale(i);
    const colorVar = `var(${ARM_VAR[r.arm]})`;
    svg.appendChild(svgEl("line", {
      x1: x, x2: x, y1: yScale(r.predicted_exam_accuracy_lo), y2: yScale(r.predicted_exam_accuracy_hi),
      style: `stroke:${colorVar};stroke-width:1.5;opacity:0.55`,
    }));
    const dot = svgEl("circle", { cx: x, cy: yScale(r.predicted_exam_accuracy), r: 3.4, style: `fill:${colorVar}` });
    const hit = svgEl("circle", { cx: x, cy: yScale(r.predicted_exam_accuracy), r: Math.max(xStep / 2, 6), fill: "transparent", style: "cursor:pointer" });
    hit.addEventListener("mouseenter", (e) => {
      const tt = document.createDocumentFragment();
      tt.appendChild(ttTitle(`${r.reviewer_id} — ${r.arm.replace(/_/g, " ")}`));
      tt.appendChild(ttRow(colorVar, "predicted accuracy", pct(r.predicted_exam_accuracy, 1)));
      tt.appendChild(ttRow(null, "95% interval", `[${pct(r.predicted_exam_accuracy_lo,1)}, ${pct(r.predicted_exam_accuracy_hi,1)}]`));
      showTooltip(e.clientX, e.clientY, tt);
    });
    hit.addEventListener("mousemove", (e) => { tooltipEl.style.left = (e.clientX + 14) + "px"; tooltipEl.style.top = (e.clientY + 14) + "px"; });
    hit.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(dot); svg.appendChild(hit);
  });
}

function renderRhoChart() {
  const svg = document.getElementById("rhoChart");
  svg.innerHTML = "";
  const W = 460, H = 120, M = { l: 20, r: 20, t: 14, b: 30 };
  const plotW = W - M.l - M.r;
  const xMin = 0.75, xMax = 0.92;
  const xScale = (v) => M.l + ((v - xMin) / (xMax - xMin)) * plotW;
  const midY = 55;

  svg.appendChild(svgEl("line", { x1: M.l, x2: W - M.r, y1: midY, y2: midY, class: "baseline" }));
  [0.75, 0.80, 0.85, 0.90].forEach((v) => {
    const x = xScale(v);
    const lab = svgEl("text", { x, y: H - 10, class: "axis-label", "text-anchor": "middle" }); lab.textContent = fmt(v, 2);
    svg.appendChild(lab);
  });

  const sv = DATA.self_validation_summary;
  const meanX = xScale(sv.mean);
  const color = cssVar("--div-benefit");
  svg.appendChild(svgEl("rect", { x: xScale(sv.mean - sv.std), y: midY - 14, width: xScale(sv.mean + sv.std) - xScale(sv.mean - sv.std), height: 28, style: `fill:${color};opacity:0.10` }));
  svg.appendChild(svgEl("line", { x1: meanX, x2: meanX, y1: midY - 20, y2: midY + 20, style: `stroke:${color};stroke-width:2` }));

  DATA.self_validation.forEach((s) => {
    const x = xScale(s.rho);
    const dot = svgEl("circle", { cx: x, cy: midY, r: 5, style: `fill:${color}`, stroke: cssVar("--surface-1"), "stroke-width": 2 });
    const hit = svgEl("circle", { cx: x, cy: midY, r: 14, fill: "transparent", style: "cursor:pointer" });
    hit.addEventListener("mouseenter", (e) => {
      const tt = document.createDocumentFragment();
      tt.appendChild(ttTitle("seed " + s.seed));
      tt.appendChild(ttRow(color, "ρ", fmt(s.rho, 4)));
      tt.appendChild(ttRow(null, "mean predicted", pct(s.mean_predicted, 1)));
      tt.appendChild(ttRow(null, "mean true", pct(s.mean_true, 1)));
      showTooltip(e.clientX, e.clientY, tt);
    });
    hit.addEventListener("mousemove", (e) => { tooltipEl.style.left = (e.clientX + 14) + "px"; tooltipEl.style.top = (e.clientY + 14) + "px"; });
    hit.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(dot); svg.appendChild(hit);
  });
}

function renderRhoSummary() {
  const sv = DATA.self_validation_summary;
  const wrap = document.getElementById("rhoSummary");
  wrap.innerHTML = "";
  wrap.appendChild(el("div", {}, `Mean ρ = ${fmt(sv.mean, 3)}, std = ${fmt(sv.std, 3)}`));
  wrap.appendChild(el("div", {}, `Range: [${fmt(sv.min, 3)}, ${fmt(sv.max, 3)}] across ${sv.n_seeds} independently generated seeds`));
  wrap.appendChild(el("div", { style: "color:var(--text-muted); font-size:12px; margin-top:6px;" },
    "Each seed: full A.2–A.5 pipeline rerun blind, then predicted vs. that seed's own true exam_accuracy."));
}

// ---------------------------------------------------------------------------
// Panel D — cost account
// ---------------------------------------------------------------------------
function renderCostChart() {
  const svg = document.getElementById("costChart");
  svg.innerHTML = "";
  const W = 460, H = 240, M = { l: 40, r: 10, t: 16, b: 46 };
  const plotW = W - M.l - M.r, plotH = H - M.t - M.b;
  const groups = DATA.cost_groups;
  const maxVal = Math.max(...groups.map((g) => Math.max(g.cases_made_worse, g.cases_made_better)), 1);
  const yScale = (v) => M.t + (1 - v / maxVal) * plotH;
  const bandW = plotW / groups.length;
  const barW = Math.min(20, bandW / 3);
  const costColor = cssVar("--div-cost");

  [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
    const y = M.t + (1 - f) * plotH;
    svg.appendChild(svgEl("line", { x1: M.l, x2: W - M.r, y1: y, y2: y, class: "gridline" }));
    const lab = svgEl("text", { x: 2, y: y + 3, class: "axis-label" }); lab.textContent = Math.round(f * maxVal);
    svg.appendChild(lab);
  });
  svg.appendChild(svgEl("line", { x1: M.l, x2: W - M.r, y1: M.t + plotH, y2: M.t + plotH, class: "baseline" }));

  groups.forEach((g, i) => {
    const cx = M.l + (i + 0.5) * bandW;
    [["cases_made_worse", 1, costColor], ["cases_made_better", 0.42, costColor]].forEach(([key, op, color], j) => {
      const x = cx - barW - 3 + j * (barW + 6);
      const v = g[key];
      const y = yScale(v);
      const h = M.t + plotH - y;
      const rect = svgEl("rect", { x, y, width: barW, height: Math.max(h, 0), rx: 4, ry: 4, style: `fill:${color};opacity:${op}` });
      rect.addEventListener("mouseenter", (e) => {
        const tt = document.createDocumentFragment();
        tt.appendChild(ttTitle(g.group));
        tt.appendChild(ttRow(color, key === "cases_made_worse" ? "cases made worse" : "cases made better", Math.round(v).toLocaleString()));
        tt.appendChild(ttRow(null, "n reviewers", String(g.n_reviewers)));
        showTooltip(e.clientX, e.clientY, tt);
      });
      rect.addEventListener("mousemove", (e) => { tooltipEl.style.left = (e.clientX + 14) + "px"; tooltipEl.style.top = (e.clientY + 14) + "px"; });
      rect.addEventListener("mouseleave", hideTooltip);
      rect.style.cursor = "pointer";
      svg.appendChild(rect);
    });
    const lab = svgEl("text", { x: cx, y: M.t + plotH + 16, class: "axis-label", "text-anchor": "middle" });
    lab.textContent = g.group;
    svg.appendChild(lab);
  });

  const legend = document.getElementById("costLegend");
  legend.innerHTML = "";
  legend.appendChild((() => { const it = el("div", { class: "legend-item" }); it.appendChild(el("span", { class: "legend-dot", style: `background:${costColor}` })); it.appendChild(document.createTextNode("cases made worse")); return it; })());
  legend.appendChild((() => { const it = el("div", { class: "legend-item" }); it.appendChild(el("span", { class: "legend-dot", style: `background:${costColor};opacity:0.42` })); it.appendChild(document.createTextNode("cases made better")); return it; })());
}

function renderBenefitChart() {
  const svg = document.getElementById("benefitChart");
  svg.innerHTML = "";
  const W = 460, H = 240, M = { l: 40, r: 10, t: 16, b: 46 };
  const plotW = W - M.l - M.r, plotH = H - M.t - M.b;
  const groups = DATA.cost_groups;
  const maxVal = Math.max(...groups.map((g) => g.skill_gain_units), 1);
  const yScale = (v) => M.t + (1 - v / maxVal) * plotH;
  const bandW = plotW / groups.length;
  const barW = Math.min(28, bandW / 2);
  const color = cssVar("--div-benefit");

  [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
    const y = M.t + (1 - f) * plotH;
    svg.appendChild(svgEl("line", { x1: M.l, x2: W - M.r, y1: y, y2: y, class: "gridline" }));
    const lab = svgEl("text", { x: 2, y: y + 3, class: "axis-label" }); lab.textContent = fmt(f * maxVal, 1);
    svg.appendChild(lab);
  });
  svg.appendChild(svgEl("line", { x1: M.l, x2: W - M.r, y1: M.t + plotH, y2: M.t + plotH, class: "baseline" }));

  groups.forEach((g, i) => {
    const cx = M.l + (i + 0.5) * bandW;
    const x = cx - barW / 2;
    const v = g.skill_gain_units;
    const y = yScale(v);
    const h = M.t + plotH - y;
    const rect = svgEl("rect", { x, y, width: barW, height: Math.max(h, 0), rx: 4, ry: 4, style: `fill:${color}` });
    rect.addEventListener("mouseenter", (e) => {
      const tt = document.createDocumentFragment();
      tt.appendChild(ttTitle(g.group));
      tt.appendChild(ttRow(color, "skill-gain units", fmt(v, 2)));
      tt.appendChild(ttRow(null, "mean exam acc. gain", pct(g.mean_exam_acc_gain_est, 1)));
      showTooltip(e.clientX, e.clientY, tt);
    });
    rect.addEventListener("mousemove", (e) => { tooltipEl.style.left = (e.clientX + 14) + "px"; tooltipEl.style.top = (e.clientY + 14) + "px"; });
    rect.addEventListener("mouseleave", hideTooltip);
    rect.style.cursor = "pointer";
    svg.appendChild(rect);
    const vlab = svgEl("text", { x: cx, y: y - 6, class: "axis-label", "text-anchor": "middle" });
    vlab.textContent = fmt(v, 1);
    svg.appendChild(vlab);
    const lab = svgEl("text", { x: cx, y: M.t + plotH + 16, class: "axis-label", "text-anchor": "middle" });
    lab.textContent = g.group;
    svg.appendChild(lab);
  });
}

const D_COLS = [
  { key: "reviewer_id", label: "Reviewer" },
  { key: "arm", label: "Arm" },
  { key: "risk_level", label: "Risk" },
  { key: "n_cases_made_worse", label: "Made worse" },
  { key: "n_cases_made_better", label: "Made better" },
  { key: "skill_gain_units", label: "Skill gain" },
  { key: "exam_acc_gain_est", label: "Exam gain" },
];

function renderCostTable() {
  const ids = new Set(filteredReviewerIds());
  let rows = DATA.cost_account.filter((r) => ids.has(r.reviewer_id));
  const { col, dir } = state.sortD;
  rows = rows.slice().sort((a, b) => {
    const av = a[col], bv = b[col];
    if (typeof av === "string") return dir * av.localeCompare(bv);
    return dir * ((av || 0) - (bv || 0));
  });

  const table = document.getElementById("costTable");
  table.innerHTML = "";
  const thead = el("thead"); const trh = el("tr");
  D_COLS.forEach((c) => {
    const th = el("th", { class: c.key === col ? "sorted" : "" }, c.label);
    if (c.key === col) th.setAttribute("data-arrow", dir === 1 ? "▲" : "▼");
    th.addEventListener("click", () => {
      if (state.sortD.col === c.key) state.sortD.dir *= -1; else state.sortD = { col: c.key, dir: -1 };
      renderCostTable();
    });
    trh.appendChild(th);
  });
  thead.appendChild(trh); table.appendChild(thead);

  const tbody = el("tbody");
  rows.forEach((r) => {
    const tr = el("tr");
    tr.appendChild(el("td", {}, r.reviewer_id));
    tr.appendChild(el("td", {}, r.arm.replace(/_/g, " ")));
    tr.appendChild(el("td", {}, r.risk_level));
    tr.appendChild(el("td", { class: "num" }, fmt(r.n_cases_made_worse, 1)));
    tr.appendChild(el("td", { class: "num" }, fmt(r.n_cases_made_better, 1)));
    tr.appendChild(el("td", { class: "num" }, fmt(r.skill_gain_units, 2)));
    tr.appendChild(el("td", { class: "num" }, pct(r.exam_acc_gain_est, 1)));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

// ---------------------------------------------------------------------------
// Wire up + theme
// ---------------------------------------------------------------------------
function renderAll() {
  renderStats();
  renderFilters();
  renderReviewerSelect();
  renderTrajLegend();
  renderTrajectory();
  renderInterventionTable();
  renderD3Legend();
  renderD3Chart();
  renderRhoChart();
  renderRhoSummary();
  renderCostChart();
  renderBenefitChart();
  renderCostTable();
}

document.getElementById("themeToggle").addEventListener("click", () => {
  const root = document.documentElement;
  const current = root.getAttribute("data-theme");
  const next = current === "dark" ? "light" : (current === "light" ? null : (window.matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark"));
  if (next) root.setAttribute("data-theme", next); else root.removeAttribute("data-theme");
  renderAll();
});

renderAll();
window.addEventListener("resize", () => { /* viewBox scaling handles responsiveness */ });
})();
