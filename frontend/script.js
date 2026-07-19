// ============================================================
// Config / state
// ============================================================
const FEATURE_KEYS = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];

// Two curated real-world-shaped scenarios so the demo doesn't require
// hand-typing 28 PCA components that have no human meaning on their own.
const SCENARIOS = {
  typical: {
    Amount: 42.5, Time: 51234,
    V1: 1.02, V2: -0.11, V3: 0.87, V4: 0.21, V5: -0.32, V6: 0.15, V7: 0.04,
    V8: -0.08, V9: 0.31, V10: 0.05, V11: -0.12, V12: 0.22, V13: -0.05,
    V14: 0.18, V15: -0.02, V16: 0.09, V17: -0.06, V18: 0.03, V19: -0.01,
    V20: 0.02, V21: -0.03, V22: 0.05, V23: -0.01, V24: 0.02, V25: 0.01,
    V26: -0.02, V27: 0.01, V28: 0.00
  },
  suspicious: {
    Amount: 0.0, Time: 406,
    V1: -2.3, V2: 1.4, V3: -3.1, V4: 2.2, V5: -1.1, V6: -0.5, V7: -2.9,
    V8: 1.1, V9: -0.8, V10: -2.5, V11: 2.0, V12: -3.0, V13: 0.4, V14: -3.6,
    V15: 0.2, V16: -1.7, V17: -1.9, V18: -0.3, V19: 0.1, V20: 0.05,
    V21: 0.3, V22: 0.7, V23: -0.02, V24: 0.1, V25: 0.2, V26: -0.1,
    V27: 0.05, V28: 0.02
  }
};

let latencyHistory = [];

// ============================================================
// DOM refs
// ============================================================
const el = (id) => document.getElementById(id);
const apiBaseInput = el("apiBase");
const scenarioSelect = el("scenarioSelect");
const amountInput = el("amountInput");
const timeInput = el("timeInput");
const rawGrid = el("rawGrid");
const rawDetails = el("rawDetails");
const runBtn = el("runBtn");
const healthPill = el("healthPill");
const avgLatencyEl = el("avgLatency");
const latencyReadout = el("latencyReadout");
const gaugeFill = el("gaugeFill");
const needle = el("needle");
const probValue = el("probValue");
const riskBadge = el("riskBadge");
const anomalyValue = el("anomalyValue");
const verdictValue = el("verdictValue");
const shapChart = el("shapChart");
const logFeed = el("logFeed");

const GAUGE_ARC_LENGTH = 251.2; // precomputed length of the semicircle path

// ============================================================
// Helpers
// ============================================================
function apiUrl(path) {
  const base = apiBaseInput.value.trim().replace(/\/$/, "");
  return base ? `${base}${path}` : path;
}

function buildRawGrid(values) {
  rawGrid.innerHTML = "";
  FEATURE_KEYS.filter((k) => k !== "Amount" && k !== "Time").forEach((key) => {
    const wrap = document.createElement("div");
    const input = document.createElement("input");
    input.dataset.key = key;
    input.value = values[key];
    input.placeholder = key;
    input.title = key;
    wrap.appendChild(input);
    rawGrid.appendChild(wrap);
  });
}

function currentPayload() {
  const payload = { Amount: parseFloat(amountInput.value) || 0, Time: parseFloat(timeInput.value) || 0 };
  rawGrid.querySelectorAll("input").forEach((inp) => {
    payload[inp.dataset.key] = parseFloat(inp.value) || 0;
  });
  return payload;
}

function applyScenario(name) {
  const s = SCENARIOS[name] || SCENARIOS.typical;
  amountInput.value = s.Amount;
  timeInput.value = s.Time;
  buildRawGrid(s);
  rawDetails.open = name === "custom";
}

// ============================================================
// Health check
// ============================================================
async function checkHealth() {
  try {
    const res = await fetch(apiUrl("/health"));
    if (!res.ok) throw new Error("bad status");
    healthPill.textContent = "online";
    healthPill.className = "status-pill online";
    healthPill.innerHTML = '<i class="dot"></i>online';
  } catch (e) {
    healthPill.innerHTML = '<i class="dot"></i>offline';
    healthPill.className = "status-pill offline";
  }
}

// ============================================================
// Gauge + risk badge
// ============================================================
function updateGauge(prob) {
  const offset = GAUGE_ARC_LENGTH - GAUGE_ARC_LENGTH * Math.min(Math.max(prob, 0), 1);
  gaugeFill.style.strokeDashoffset = offset;

  let color = "var(--signal)";
  if (prob >= 0.5) color = "var(--red)";
  else if (prob >= 0.2) color = "var(--amber)";
  gaugeFill.style.stroke = color;

  const angle = -90 + 180 * Math.min(Math.max(prob, 0), 1);
  needle.style.transform = `rotate(${angle}deg)`;

  probValue.textContent = `${(prob * 100).toFixed(2)}%`;
}

function setRiskBadge(level) {
  riskBadge.textContent = level;
  riskBadge.className = `risk-badge ${level}`;
}

// ============================================================
// SHAP tornado chart
// ============================================================
function renderShap(topFeatures) {
  if (!topFeatures || !topFeatures.length) {
    shapChart.innerHTML = '<div class="shap-empty">No feature data returned.</div>';
    return;
  }
  const maxAbs = Math.max(...topFeatures.map((f) => Math.abs(f.shap_value)), 1e-6);
  shapChart.innerHTML = "";

  topFeatures.forEach((f) => {
    const row = document.createElement("div");
    row.className = "shap-row";

    const label = document.createElement("span");
    label.className = "shap-feature";
    label.textContent = f.feature;

    const track = document.createElement("div");
    track.className = "shap-track";
    const centerLine = document.createElement("div");
    centerLine.className = "shap-center-line";
    track.appendChild(centerLine);

    const bar = document.createElement("div");
    const pct = (Math.abs(f.shap_value) / maxAbs) * 48; // max 48% of track width per side
    bar.className = `shap-bar ${f.shap_value >= 0 ? "pos" : "neg"}`;
    bar.style.width = `${pct}%`;
    track.appendChild(bar);

    const val = document.createElement("span");
    val.className = "shap-val";
    val.textContent = f.shap_value.toFixed(3);

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(val);
    shapChart.appendChild(row);
  });
}

// ============================================================
// Log feed
// ============================================================
function pushLogRow(result) {
  const empty = logFeed.querySelector(".log-empty");
  if (empty) empty.remove();

  const row = document.createElement("div");
  row.className = "log-row";
  const time = new Date().toLocaleTimeString();
  row.innerHTML = `
    <span>${time}</span>
    <span class="badge ${result.risk_level}">${result.risk_level}</span>
    <span>p(fraud)=${result.fraud_probability.toFixed(4)} &nbsp; anomaly=${result.anomaly_score.toFixed(3)}</span>
    <span>${result.latency_ms.toFixed(1)}ms</span>
  `;
  logFeed.prepend(row);

  const rows = logFeed.querySelectorAll(".log-row");
  if (rows.length > 25) rows[rows.length - 1].remove();
}

function updateAvgLatency(ms) {
  latencyHistory.push(ms);
  if (latencyHistory.length > 20) latencyHistory.shift();
  const avg = latencyHistory.reduce((a, b) => a + b, 0) / latencyHistory.length;
  avgLatencyEl.textContent = `${avg.toFixed(1)}ms`;
}

// ============================================================
// Run prediction
// ============================================================
async function runPrediction() {
  runBtn.disabled = true;
  runBtn.querySelector("span").textContent = "RUNNING...";
  verdictValue.textContent = "scoring...";

  const payload = currentPayload();
  const clientStart = performance.now();

  try {
    const res = await fetch(apiUrl("/predict"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    updateGauge(data.fraud_probability);
    setRiskBadge(data.risk_level);
    anomalyValue.textContent = data.anomaly_score.toFixed(4);
    verdictValue.textContent =
      data.risk_level === "HIGH" ? "flagged — likely fraud" :
      data.risk_level === "MEDIUM" ? "flagged for review" : "looks legitimate";
    latencyReadout.textContent = `latency: ${data.latency_ms.toFixed(2)}ms (round trip ${(performance.now() - clientStart).toFixed(1)}ms)`;
    renderShap(data.top_features);
    pushLogRow(data);
    updateAvgLatency(data.latency_ms);
    healthPill.innerHTML = '<i class="dot"></i>online';
    healthPill.className = "status-pill online";
  } catch (e) {
    verdictValue.textContent = "request failed — is the API running?";
    healthPill.innerHTML = '<i class="dot"></i>offline';
    healthPill.className = "status-pill offline";
  } finally {
    runBtn.disabled = false;
    runBtn.querySelector("span").textContent = "RUN PREDICTION";
  }
}

// ============================================================
// Wire up
// ============================================================
scenarioSelect.addEventListener("change", (e) => applyScenario(e.target.value));
runBtn.addEventListener("click", runPrediction);
apiBaseInput.addEventListener("change", checkHealth);

applyScenario("typical");
checkHealth();
setInterval(checkHealth, 15000);
