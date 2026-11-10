/* Forest Loss Atlas — static dashboard (no build step).
 * Data contract (web/data/): history.json, risk.json (scripts/build_web_data.py), metrics.json (deforest.evaluate).
 */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = new Intl.NumberFormat("en-US");
const compact = (v) => (Math.abs(v) >= 10000 ? new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v) : fmt.format(Math.round(v)));
const pct = (v, d) => {
  if (v == null || Number.isNaN(v)) return "—";
  const digits = d ?? (Math.abs(v) < 0.01 ? 2 : Math.abs(v) < 0.1 ? 1 : 0);
  return `${(v * 100).toFixed(digits)}%`;
};
const num = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(d));

const STYLE = {
  light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
};
const MAP_TABS = ["history", "risk"];

const state = {
  tab: "history",
  data: {},
  yearFrom: 2001,
  yearTo: 2023,
  horizon: 1,
  polygon: null,
  drawing: false,
  vertices: [],
  mapTab: "history",
};

/* ---------------------------------------------------------------- theme */
function currentTheme() {
  const t = document.documentElement.dataset.theme;
  if (t) return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
$("#theme-toggle").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) {}
  onThemeChange();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!document.documentElement.dataset.theme) onThemeChange();
});
function onThemeChange() {
  map.setStyle(STYLE[currentTheme()], { diff: false });
  renderPanel();
}
const seqColors = () => [1, 2, 3, 4, 5, 6, 7].map((i) => css(`--seq-${i}`));

/* ---------------------------------------------------------------- map */
const map = new maplibregl.Map({
  container: "map",
  style: STYLE[currentTheme()],
  bounds: [[-122.8, 48.9], [-113.5, 54.5]],
  fitBoundsOptions: { padding: 30 },
  attributionControl: { compact: true },
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

map.on("style.load", () => {
  map.addSource("cells", { type: "geojson", data: emptyFC() });
  map.addSource("roi", { type: "geojson", data: roiFeature() });
  map.addSource("draw", { type: "geojson", data: emptyFC() });
  const firstSymbol = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
  map.addLayer({ id: "cells-fill", type: "fill", source: "cells", paint: { "fill-color": "#000", "fill-opacity": 0.78 } }, firstSymbol);
  map.addLayer({ id: "roi-line", type: "line", source: "roi", paint: { "line-color": css("--ink-2"), "line-width": 1, "line-dasharray": [3, 3] } });
  map.addLayer({ id: "draw-fill", type: "fill", source: "draw", filter: ["==", "$type", "Polygon"], paint: { "fill-color": css("--series-1"), "fill-opacity": 0.08 } });
  map.addLayer({ id: "draw-line", type: "line", source: "draw", filter: ["!=", "$type", "Point"], paint: { "line-color": css("--series-1"), "line-width": 2 } });
  map.addLayer({ id: "draw-pts", type: "circle", source: "draw", filter: ["==", "$type", "Point"], paint: { "circle-radius": 4, "circle-color": css("--series-1"), "circle-stroke-color": css("--surface"), "circle-stroke-width": 2 } });
  if (state.data.history || state.data.risk) updateMapLayer();
  renderDraw();
});

const emptyFC = () => ({ type: "FeatureCollection", features: [] });
let ROI = [-122.8, 48.9, -113.5, 54.5];
function roiFeature() {
  const [x0, y0, x1, y1] = ROI;
  return { type: "Feature", geometry: { type: "LineString", coordinates: [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]] } };
}
function cellPolygon(meta, ix, iy) {
  const [ox, oy] = meta.roi, r = meta.res;
  const x0 = ox + ix * r, y0 = oy + iy * r;
  return [[[x0, y0], [x0 + r, y0], [x0 + r, y0 + r], [x0, y0 + r], [x0, y0]]];
}
const cellCenter = (meta, ix, iy) => [meta.roi[0] + (ix + 0.5) * meta.res, meta.roi[1] + (iy + 0.5) * meta.res];

function quantileBreaks(values, n = 7) {
  const v = values.filter((x) => x > 0).sort((a, b) => a - b);
  if (!v.length) return [];
  const out = [];
  for (let i = 1; i < n; i++) out.push(v[Math.min(v.length - 1, Math.floor((i / n) * v.length))]);
  return [...new Set(out)];
}
function stepExpr(breaks, colors) {
  const expr = ["step", ["get", "v"], colors[0]];
  breaks.forEach((b, i) => expr.push(b, colors[Math.min(i + 1, colors.length - 1)]));
  return expr;
}
const hIndex = () => state.data.risk.horizons.indexOf(state.horizon);

function layerModel() {
  const { history, risk } = state.data;
  if (state.mapTab === "risk" && risk) {
    const hi = hIndex();
    const feats = risk.cells.map((c) => ({ type: "Feature", properties: { v: c[3 + hi], n: c[2], kind: "risk" }, geometry: { type: "Polygon", coordinates: cellPolygon(risk, c[0], c[1]) } }));
    // classes as multiples of the regional mean, so the map reads "how much riskier than average"
    const mean = risk.mean_p[String(state.horizon)];
    const breaks = [0.5, 1, 2, 4, 8, 16].map((m) => m * mean);
    return { feats, breaks, fmt: (x) => pct(x), title: `Probability of loss, ${risk.labels[String(state.horizon)]}`, legendNote: `regional mean ${pct(mean)}` };
  }
  if (!history) return { feats: [], breaks: [], fmt: String, title: "" };
  const i0 = state.yearFrom - history.years[0], i1 = state.yearTo - history.years[0];
  const feats = [];
  for (const [ix, iy, counts] of history.cells) {
    let v = 0;
    for (let i = i0; i <= i1; i++) v += counts[i];
    if (v > 0) feats.push({ type: "Feature", properties: { v, kind: "history" }, geometry: { type: "Polygon", coordinates: cellPolygon(history, ix, iy) } });
  }
  return { feats, breaks: quantileBreaks(feats.map((f) => f.properties.v)), fmt: compact, title: `Loss pixels per cell, ${state.yearFrom}–${state.yearTo} (quantile classes)` };
}

let currentModel = null;
function updateMapLayer() {
  if (!map.getSource("cells")) return;
  currentModel = layerModel();
  const colors = seqColors();
  map.getSource("cells").setData({ type: "FeatureCollection", features: currentModel.feats });
  map.setPaintProperty("cells-fill", "fill-color", currentModel.breaks.length ? stepExpr(currentModel.breaks, colors) : colors[3]);
  renderLegend();
}

function renderLegend() {
  if (!currentModel) return;
  const el = $(state.mapTab === "risk" ? "#risk-legend" : "#history-legend");
  const { breaks, fmt: f, title, legendNote } = currentModel;
  const colors = seqColors().slice(0, breaks.length + 1);
  el.innerHTML = `<div class="title">${title}</div>
    <div class="ramp">${colors.map((c) => `<span style="background:${c}"></span>`).join("")}</div>
    <div class="ticks"><span>lower</span>${breaks.filter((_, i) => i % 2 === 1).map((b) => `<span>${f(b)}</span>`).join("")}<span>higher</span></div>
    ${legendNote ? `<div class="legend-note">${legendNote}</div>` : ""}`;
}

/* map hover */
const mapTip = $("#map-tip");
map.on("mousemove", "cells-fill", (e) => {
  if (state.drawing) return;
  const p = e.features[0].properties;
  mapTip.innerHTML = p.kind === "history"
    ? `<div class="t-label">${state.yearFrom}–${state.yearTo}</div><strong>${fmt.format(p.v)}</strong> loss pixels`
    : `<div class="t-label">${state.data.risk.labels[String(state.horizon)]} · ${p.n} grid points</div><strong>${pct(p.v)}</strong> probability of loss`;
  placeTip(mapTip, e.originalEvent.clientX, e.originalEvent.clientY);
});
map.on("mouseleave", "cells-fill", () => { mapTip.hidden = true; });

function placeTip(tip, x, y) {
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

/* ---------------------------------------------------------------- drawing */
const drawBtn = $("#draw-btn"), clearBtn = $("#clear-btn");
drawBtn.addEventListener("click", () => (state.drawing ? cancelDraw() : startDraw()));
clearBtn.addEventListener("click", () => { state.polygon = null; clearBtn.hidden = true; renderDraw(); renderPanel(); });
addEventListener("keydown", (e) => { if (e.key === "Escape" && state.drawing) cancelDraw(); });

function startDraw() {
  state.drawing = true; state.vertices = [];
  map.doubleClickZoom.disable();
  map.getCanvas().style.cursor = "crosshair";
  drawBtn.textContent = "Cancel"; drawBtn.classList.remove("primary");
  $("#draw-hint").hidden = false;
}
function cancelDraw() {
  state.drawing = false; state.vertices = [];
  map.doubleClickZoom.enable();
  map.getCanvas().style.cursor = "";
  drawBtn.textContent = "Draw area"; drawBtn.classList.add("primary");
  $("#draw-hint").hidden = true;
  renderDraw();
}
function finishDraw() {
  if (state.vertices.length >= 3) { state.polygon = [...state.vertices, state.vertices[0]]; clearBtn.hidden = false; }
  cancelDraw();
  renderPanel();
}
map.on("click", (e) => {
  if (!state.drawing) return;
  const v = state.vertices;
  if (v.length >= 3) {
    const first = map.project(v[0]);
    if (Math.hypot(first.x - e.point.x, first.y - e.point.y) < 10) return finishDraw();
  }
  v.push([e.lngLat.lng, e.lngLat.lat]);
  renderDraw();
});
map.on("dblclick", (e) => { if (state.drawing) { e.preventDefault(); finishDraw(); } });
map.on("mousemove", (e) => { if (state.drawing && state.vertices.length) renderDraw([e.lngLat.lng, e.lngLat.lat]); });

function renderDraw(cursor) {
  const src = map.getSource("draw");
  if (!src) return;
  const feats = [];
  if (state.drawing) {
    const pts = cursor ? [...state.vertices, cursor] : state.vertices;
    if (pts.length > 1) feats.push({ type: "Feature", geometry: { type: "LineString", coordinates: pts } });
    state.vertices.forEach((c) => feats.push({ type: "Feature", geometry: { type: "Point", coordinates: c } }));
  } else if (state.polygon) {
    feats.push({ type: "Feature", geometry: { type: "Polygon", coordinates: [state.polygon] } });
  }
  src.setData({ type: "FeatureCollection", features: feats });
}
function pointInRing([x, y], ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
const inSelection = (meta, ix, iy) => !state.polygon || pointInRing(cellCenter(meta, ix, iy), state.polygon);

/* ---------------------------------------------------------------- charts */
const chartTip = $("#chart-tip");
const svgNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(svgNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
function niceTicks(max, count = 4) {
  if (!(max > 0)) return [0, 1];
  const raw = max / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(+v.toFixed(10));
  if (ticks[ticks.length - 1] < max) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(10));
  return ticks;
}
function hover(node, html) {
  node.addEventListener("mousemove", (e) => { chartTip.innerHTML = html(); placeTip(chartTip, e.clientX, e.clientY); });
  node.addEventListener("mouseleave", () => (chartTip.hidden = true));
}

function barChart(container, { labels, values, highlight, yFmt, tipFmt }) {
  container.innerHTML = "";
  const W = Math.max(280, container.clientWidth), H = 190;
  const m = { t: 10, r: 6, b: 22, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const ticks = niceTicks(Math.max(...values, 1));
  const ymax = ticks[ticks.length - 1];
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Annual loss column chart" }, container);
  ticks.forEach((t) => {
    const y = m.t + ih - (t / ymax) * ih;
    el("line", { class: t === 0 ? "baseline" : "gridline", x1: m.l, x2: W - m.r, y1: y, y2: y }, svg);
    el("text", { x: m.l - 6, y: y + 3.5, "text-anchor": "end" }, svg).textContent = yFmt(t);
  });
  const band = iw / values.length, bw = Math.min(24, band - 2);
  values.forEach((v, i) => {
    const x = m.l + i * band + (band - bw) / 2;
    const h = (v / ymax) * ih, y = m.t + ih - h, r = Math.min(4, h, bw / 2);
    if (h > 0) el("path", { class: "bar", d: `M${x},${m.t + ih}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${m.t + ih}Z`, style: highlight(i) ? "" : `fill:${css("--baseline")}` }, svg);
    if (i % Math.ceil(values.length / 8) === 0 || i === values.length - 1)
      el("text", { x: m.l + i * band + band / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = labels[i];
    hover(el("rect", { x: m.l + i * band, y: m.t, width: band, height: ih, fill: "transparent" }, svg), () => tipFmt(i));
  });
}

/* horizontal bars: rows [{label, value, accent, note}] — one measure, values labelled at the bar end */
function hbars(rows, { max, fmtv, tip }) {
  const mx = max ?? Math.max(...rows.map((r) => r.value)) * 1.05;
  return `<div class="hbars">${rows.map((r, i) => `
    <div class="hb-row" data-i="${i}">
      <span class="hb-label">${r.label}</span>
      <span class="hb-track"><span class="hb-bar ${r.accent ? "accent" : ""}" style="width:${Math.max(0.5, (r.value / mx) * 100)}%"></span></span>
      <span class="hb-val">${fmtv(r.value)}</span>
    </div>`).join("")}</div>`;
}
function bindHbarTips(container, rows, tip) {
  $$(".hb-row", container).forEach((row) => hover(row, () => tip(rows[+row.dataset.i])));
}

function calibrationChart(container, bins) {
  container.innerHTML = "";
  const W = Math.max(260, container.clientWidth), H = 220, m = { t: 10, r: 10, b: 30, l: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const mx = niceTicks(Math.max(...bins.map((b) => Math.max(b.pred, b.obs))) * 1.05);
  const top = mx[mx.length - 1];
  const X = (v) => m.l + (v / top) * iw, Y = (v) => m.t + ih - (v / top) * ih;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Calibration: predicted vs observed loss rate" }, container);
  mx.forEach((t) => {
    el("line", { class: t === 0 ? "baseline" : "gridline", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, svg);
    el("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end" }, svg).textContent = pct(t);
    el("text", { x: X(t), y: H - 14, "text-anchor": "middle" }, svg).textContent = pct(t);
  });
  el("text", { x: m.l + iw / 2, y: H - 1, "text-anchor": "middle" }, svg).textContent = "predicted";
  el("line", { class: "diag", x1: X(0), y1: Y(0), x2: X(top), y2: Y(top) }, svg);
  bins.forEach((b) => {
    const c = el("circle", { class: "cal-dot", cx: X(b.pred), cy: Y(b.obs), r: 5 }, svg);
    hover(c, () => `<div class="t-label">${pct(b.share, 0)} of forest</div>predicted <strong>${pct(b.pred)}</strong> · observed <strong>${pct(b.obs)}</strong>`);
  });
}

function table(head, rows) {
  return `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
$$("[data-table-toggle]").forEach((b) =>
  b.addEventListener("click", () => {
    const t = $(`#${b.dataset.tableToggle}`);
    t.hidden = !t.hidden;
    b.textContent = t.hidden ? "Show table" : "Hide table";
  })
);
const tile = (label, value, note = "") => `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div>${note ? `<div class="note">${note}</div>` : ""}</div>`;

function segmented(container, items, selected, onPick) {
  container.innerHTML = items.map(([v, label]) => `<button role="radio" aria-checked="${v === selected}" data-v="${v}">${label}</button>`).join("");
  $$("button", container).forEach((b) => b.addEventListener("click", () => onPick(+b.dataset.v)));
}

/* ---------------------------------------------------------------- panels */
function renderHistoryPanel() {
  const h = state.data.history;
  if (!h) { $("#history-tiles").innerHTML = `<p class="lede">history.json not found.</p>`; return; }
  let series = h.totals, cellsIn = h.cells.length;
  if (state.polygon) {
    series = h.years.map(() => 0); cellsIn = 0;
    for (const [ix, iy, counts] of h.cells) if (inSelection(h, ix, iy)) { cellsIn++; counts.forEach((c, i) => (series[i] += c)); }
  }
  const i0 = state.yearFrom - h.years[0], i1 = state.yearTo - h.years[0];
  const inRange = series.slice(i0, i1 + 1);
  const total = inRange.reduce((a, b) => a + b, 0);
  const peakIdx = inRange.indexOf(Math.max(...inRange));
  const n = Math.min(5, Math.floor(inRange.length / 2));
  const early = n ? inRange.slice(0, n).reduce((a, b) => a + b, 0) / n : 0;
  const late = n ? inRange.slice(-n).reduce((a, b) => a + b, 0) / n : 0;
  const change = early > 0 ? (late - early) / early : null;
  $("#history-tiles").innerHTML =
    tile("Loss pixels", compact(total), `${state.yearFrom}–${state.yearTo}`) +
    tile("Peak year", total ? h.years[i0 + peakIdx] : "—", total ? `${compact(inRange[peakIdx])} pixels` : "") +
    tile(n ? `Last ${n} vs first ${n} yrs` : "Change", change == null ? "—" : `${change > 0 ? "+" : ""}${pct(change, 0)}`, "mean per year");
  $("#history-chart-title").textContent = `Annual loss — ${state.polygon ? `drawn area (${cellsIn} cells)` : "whole region"}`;
  barChart($("#history-chart"), {
    labels: h.years.map(String), values: series, highlight: (i) => i >= i0 && i <= i1, yFmt: compact,
    tipFmt: (i) => `<div class="t-label">${h.years[i]}</div><strong>${fmt.format(series[i])}</strong> loss pixels`,
  });
  $("#history-table").innerHTML = table(["Year", "Loss pixels"], h.years.map((y, i) => [y, fmt.format(series[i])]));
}

function renderRiskPanel() {
  const r = state.data.risk;
  if (!r) { $("#risk-tiles").innerHTML = `<p class="lede">risk.json not found — run the pipeline.</p>`; return; }
  $$("[data-base-year]").forEach((e) => (e.textContent = r.base_year));
  segmented($("#risk-horizons"), r.horizons.map((h) => [h, r.labels[String(h)]]), state.horizon, (h) => { state.horizon = h; updateMapLayer(); renderPanel(); });
  const hi = hIndex(), mean = r.mean_p[String(state.horizon)];
  let pts = 0, expected = 0, high = 0;
  for (const c of r.cells) {
    if (!inSelection(r, c[0], c[1])) continue;
    const p = c[3 + hi];
    pts += c[2]; expected += c[2] * p * r.km2_per_point;
    if (p >= 4 * mean) high += c[2];
  }
  const scope = state.polygon ? "drawn area" : "whole region";
  $("#risk-tiles").innerHTML =
    tile("Expected loss", `${compact(expected)} km²`, `${scope}, ${r.labels[String(state.horizon)]}`) +
    tile("Mean probability", pct(pts ? expected / (pts * r.km2_per_point) : 0), `forest at end of ${r.base_year}`) +
    tile("High-risk forest", pct(pts ? high / pts : 0, 1), `cells ≥ 4× regional mean`);
  const m = state.data.metrics?.horizons?.find((x) => x.h === state.horizon)?.test;
  $("#risk-note").innerHTML = m
    ? `<span class="status-icon" aria-hidden="true">i</span><div>On held-out areas and years (${m.years[0]}–${m.years[1]}), the top 10% of forest by this score contained <strong>${pct(m.models.xgboost.top_k["0.1"]?.recall, 0)}</strong> of the loss that followed. Random targeting would capture 10%. Details are in the Models tab.</div>`
    : `<span class="status-icon" aria-hidden="true">i</span><div>No evaluation metrics loaded.</div>`;
}

const MODEL_LABEL = { xgboost: "XGBoost", logistic: "Logistic baseline", distance: "Distance to recent loss", prevalence: "Random (base rate)" };

function renderModels() {
  const M = state.data.metrics;
  const body = $("#model-body");
  if (!M || M.schema !== 2) { body.innerHTML = `<p class="lede">No metrics.json (schema 2) found — run <code>python -m deforest.evaluate</code>.</p>`; return; }
  $("#models-lede").innerHTML = `${M.target}. ${M.weighting}`;
  const hs = M.horizons.map((h) => h.h);
  if (!hs.includes(state.horizon)) state.horizon = hs[0];
  segmented($("#model-horizons"), M.horizons.map((h) => [h.h, h.label]), state.horizon, (h) => { state.horizon = h; if (state.mapTab === "risk") updateMapLayer(); renderModels(); });
  const H = M.horizons.find((h) => h.h === state.horizon);
  const t = H.test;
  if (!t) { body.innerHTML = `<p class="lede">No test data for this horizon.</p>`; return; }
  const X = t.models.xgboost;
  const prRows = ["xgboost", "logistic", "distance", "prevalence"].map((k) => ({ key: k, label: MODEL_LABEL[k], value: t.models[k].pr_auc, accent: k === "xgboost" }));
  const ks = Object.keys(X.top_k);
  const temporal = H.test_temporal?.models?.xgboost?.pr_auc;

  body.innerHTML = `
    <div class="tiles">
      ${tile("PR-AUC", num(X.pr_auc), `${(X.pr_auc / t.prevalence).toFixed(1)}× random`)}
      ${tile("Loss caught in top 10%", pct(X.top_k["0.1"]?.recall, 0), "of forest area flagged")}
      ${tile("Base rate", pct(t.prevalence), `${fmt.format(t.positives)} loss events in test`)}
    </div>

    <figure class="chart-card">
      <figcaption><span class="chart-title">PR-AUC on held-out blocks, ${t.years[0]}–${t.years[1]}</span></figcaption>
      <div id="pr-bars">${hbars(prRows, { fmtv: (v) => num(v) })}</div>
      <p class="fig-note">Precision–recall area: higher is better. Random ranking scores the base rate.</p>
    </figure>

    <figure class="chart-card">
      <figcaption><span class="chart-title">Targeting the highest-risk forest</span></figcaption>
      <div class="table-wrap">${table(
        ["Area flagged", "Precision", "Recall", "Lift", "Recall (distance)"],
        ks.map((k) => [pct(+k, 0), pct(X.top_k[k].precision), pct(X.top_k[k].recall, 0), `${X.top_k[k].lift.toFixed(1)}×`, pct(t.models.distance.top_k[k].recall, 0)])
      )}</div>
    </figure>

    <figure class="chart-card">
      <figcaption><span class="chart-title">Calibration — predicted vs observed</span></figcaption>
      <div class="chart" id="cal-chart"></div>
      <p class="fig-note">Each dot is a tenth of the test forest. Dots on the diagonal mean the probabilities can be taken at face value. Brier score ${num(X.brier, 4)}.</p>
    </figure>

    <figure class="chart-card">
      <figcaption><span class="chart-title">By test year</span></figcaption>
      <div class="table-wrap">${table(["Year", "Base rate", "XGBoost PR-AUC", "Distance PR-AUC"], t.by_year.map((r) => [r.year, pct(r.prevalence), num(r.xgboost), num(r.distance)]))}</div>
    </figure>

    ${t.drivers && Object.keys(t.drivers).length ? `<figure class="chart-card">
      <figcaption><span class="chart-title">Fire vs other loss</span></figcaption>
      <div class="table-wrap">${table(["Driver", "Share of loss", "PR-AUC", "Recall @10%"], Object.entries(t.drivers).map(([k, d]) => [k === "fire" ? "Wildfire" : "Other (mostly harvest)", pct(d.share_of_loss, 0), num(d.pr_auc), pct(d["recall_at_0.1"], 0)]))}</div>
    </figure>` : ""}

    <figure class="chart-card">
      <figcaption><span class="chart-title">Leakage check</span></figcaption>
      <div class="table-wrap">${table(["Test rows", "XGBoost PR-AUC"], [["Held-out blocks (reported)", num(X.pr_auc)], ["Training blocks, later years", num(temporal)]])}</div>
      <p class="fig-note">A large gap would mean the model memorises places. The headline numbers always use the held-out blocks.</p>
    </figure>

    ${M.importance?.length ? `<figure class="chart-card">
      <figcaption><span class="chart-title">What the 1-year model relies on</span></figcaption>
      <div id="imp-bars">${hbars(M.importance.slice(0, 10).map((r) => ({ label: r.feature, value: r.gain, accent: true })), { fmtv: (v) => pct(v, 0) })}</div>
      <p class="fig-note">Share of total split gain.</p>
    </figure>` : ""}`;

  bindHbarTips($("#pr-bars"), prRows, (r) => `<div class="t-label">${r.label}</div>PR-AUC <strong>${num(r.value)}</strong> · ROC-AUC ${num(t.models[r.key].roc_auc)}`);
  calibrationChart($("#cal-chart"), t.calibration);
}

function renderPanel() {
  if (state.tab === "history") renderHistoryPanel();
  if (state.tab === "risk") renderRiskPanel();
  if (state.tab === "models") renderModels();
  renderLegend();
}

/* ---------------------------------------------------------------- tabs & controls */
$$(".tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    if (state.drawing) cancelDraw();
    state.tab = b.dataset.tab;
    $$(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $$(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== state.tab));
    if (MAP_TABS.includes(state.tab) && state.tab !== state.mapTab) { state.mapTab = state.tab; updateMapLayer(); }
    renderPanel();
  })
);

function initYearSelects(years) {
  const opts = years.map((y) => `<option value="${y}">${y}</option>`).join("");
  $("#year-from").innerHTML = opts; $("#year-to").innerHTML = opts;
  $("#year-from").value = state.yearFrom; $("#year-to").value = state.yearTo;
  const onChange = (e) => {
    let a = +$("#year-from").value, b = +$("#year-to").value;
    if (a > b) { if (e.target.id === "year-from") b = a; else a = b; }
    state.yearFrom = a; state.yearTo = b;
    $("#year-from").value = a; $("#year-to").value = b;
    updateMapLayer(); renderHistoryPanel();
  };
  $("#year-from").addEventListener("change", onChange);
  $("#year-to").addEventListener("change", onChange);
}

let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderPanel, 150); });

/* ---------------------------------------------------------------- boot */
const load = (name) => fetch(`data/${name}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
(async () => {
  const [history, risk, metrics] = await Promise.all([load("history"), load("risk"), load("metrics")]);
  state.data = { history, risk, metrics };
  const meta = history || risk;
  if (!meta) {
    $("#map-status").textContent = "No data found in web/data — run the pipeline, then serve web/ over HTTP.";
    return;
  }
  ROI = meta.roi;
  map.getSource("roi")?.setData(roiFeature());
  map.fitBounds([[ROI[0], ROI[1]], [ROI[2], ROI[3]]], { padding: 30, animate: false });
  $("#region-sub").textContent = `${meta.region ?? "Study region"} · tree-cover loss ${history ? `${history.years[0]}–${history.years.at(-1)}` : ""}${risk ? ` · risk ${risk.labels[String(risk.horizons.at(-1))]}` : ""}`;
  if (history) {
    state.yearFrom = history.years[0]; state.yearTo = history.years.at(-1);
    initYearSelects(history.years);
  }
  if (risk) state.horizon = risk.horizons[0];
  $("#map-status").hidden = true;
  updateMapLayer();
  renderPanel();
})();
