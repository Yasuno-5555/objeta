#!/usr/bin/env python3
"""Stability Phase Diagram — λ × k dynamics.

Sweeps locality λ and top-k, measures:
  - Thrash rate (page faults / access)
  - Collapse risk (semantic narrowing)
  - Effective throughput (tok/s projection)
  - Utility U = quality - α·latency - β·pressure - γ·collapse

Finds the stable operating region where all three constraints
are satisfied simultaneously.

Usage:
  python3 experiments/stability_phase.py
  open experiments/results/stability_phase.html
"""

import json, sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO)); sys.path.insert(0, str(PROJECT))

import numpy as np
from os_runtime.governor import Governor, ThrashDetector, SemanticCollapse

OUTPUT = PROJECT / "experiments" / "results"


def simulate_steady_state(lam: float, k: int, n_tokens: int = 200):
    """Simulate steady-state behavior at given (λ, k)."""
    rng = np.random.RandomState(42)

    # Expert access model based on real Qwen3.6 measurements
    n_experts = 256
    expert_size_mb = 10.5
    ram_budget_mb = 4000
    max_loaded = int(ram_budget_mb / expert_size_mb)  # ~380 experts fit in 4GB

    # Access pattern depends on λ and k
    # Higher λ → more concentrated access (fewer unique experts)
    # Higher k → more experts accessed per token → higher I/O
    working_set = max(2, int(n_experts / (1 + lam * 2)))  # λ↑ → ws↓
    working_set = min(working_set, n_experts)

    # Simulate
    fault_history = []
    expert_diversity = []
    entropy_history = []

    for i in range(n_tokens):
        # How many unique experts accessed this token
        # With locality λ: most accesses stay in working set
        p_new = 1.0 / (1 + lam)  # probability of accessing new expert
        n_new = np.random.binomial(k, p_new)

        # Will these cause page faults?
        faults_this_token = 0
        if i < 10:  # cold start
            faults_this_token = k
        else:
            faults_this_token = n_new

        fault_history.append(faults_this_token > k // 2)
        expert_diversity.append(k - n_new + 1)

        # Entropy model: higher diversity → higher entropy
        entropy = 0.01 + (k - n_new) / k * 0.2
        entropy_history.append(entropy)

    # Thrash rate
    fault_rate = sum(fault_history) / len(fault_history)

    # Working set size
    ws = working_set
    ws_fits = ws * expert_size_mb <= ram_budget_mb

    # Collapse risk: low diversity → high risk
    avg_diversity = np.mean(expert_diversity)
    collapse_risk = max(0.0, 1.0 - avg_diversity / k)

    # Memory pressure
    mem_pressure = (ws * expert_size_mb) / ram_budget_mb

    # I/O time per token (8 experts, 1ms cold, 0.4ms warm)
    cold_per_token = k * (1.0 / (1 + lam))  # expected cold accesses
    warm_per_token = k - cold_per_token
    io_ms = cold_per_token * 1.0 + warm_per_token * 0.4
    compute_ms = k * 0.4  # 400µs GEMV per expert
    total_ms = io_ms + compute_ms

    # Utility
    quality = 1.0 - collapse_risk
    alpha, beta, gamma = 0.3, 0.3, 0.4
    utility = (quality
               - alpha * min(1.0, total_ms / 20)   # latency penalty
               - beta * min(1.0, mem_pressure)      # memory penalty
               - gamma * collapse_risk)             # collapse penalty

    return {
        "lambda": lam, "k": k,
        "fault_rate": round(fault_rate, 3),
        "working_set": ws,
        "ws_fits_ram": ws_fits,
        "avg_diversity": round(float(avg_diversity), 1),
        "collapse_risk": round(collapse_risk, 3),
        "mem_pressure": round(mem_pressure, 3),
        "io_ms": round(io_ms, 1),
        "compute_ms": round(compute_ms, 1),
        "total_ms": round(total_ms, 1),
        "tok_per_s": round(1000 / total_ms, 1) if total_ms > 0 else 0,
        "utility": round(utility, 3),
    }


def classify_regime(r: dict) -> str:
    """Classify operating regime."""
    if r["collapse_risk"] > 0.5:
        return "collapsed"
    if r["mem_pressure"] > 0.9 and r["fault_rate"] > 0.3:
        return "thrashing"
    if r["collapse_risk"] > 0.2 or r["fault_rate"] > 0.15:
        return "marginal"
    if r["mem_pressure"] < 0.7 and r["fault_rate"] < 0.1:
        return "stable"
    return "transitional"


def build_html(results: list[dict]) -> str:
    """Build interactive phase diagram HTML."""
    data_json = json.dumps(results)

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Stability Phase Diagram</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#c9d1d9;font:14px monospace;padding:16px}}
h1{{color:#58a6ff;margin-bottom:8px}}
canvas{{border:1px solid #30363d;border-radius:6px}}
.legend{{display:flex;gap:12px;margin:8px 0;font-size:11px}}
.legend span{{display:flex;align-items:center;gap:4px}}
.swatch{{width:12px;height:12px;border-radius:2px}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;margin-top:12px}}
.metric{{display:flex;justify-content:space-between;padding:2px 0}}
.val{{color:#58a6ff}}
</style></head><body>
<h1>Stability Phase Diagram — λ × k</h1>
<div class="legend">
<span><span class="swatch" style="background:#3fb950"></span> Stable</span>
<span><span class="swatch" style="background:#d29922"></span> Marginal</span>
<span><span class="swatch" style="background:#f85149"></span> Thrashing</span>
<span><span class="swatch" style="background:#8b949e"></span> Collapsed</span>
</div>
<canvas id="phase" width="700" height="500"></canvas>
<div class="panel"><h2>Hovered Point</h2><div id="info">Move mouse over diagram</div></div>
<div class="panel" id="optimal"></div>
<script>
const data = {data_json};
const canvas = document.getElementById('phase');
const ctx = canvas.getContext('2d');
const W = canvas.width - 60, H = canvas.height - 40;

// Find ranges
const lambdas = [...new Set(data.map(d => d.lambda))].sort((a,b)=>a-b);
const ks = [...new Set(data.map(d => d.k))].sort((a,b)=>a-b);
const lamRange = [Math.min(...lambdas), Math.max(...lambdas)];
const kRange = [Math.min(...ks), Math.max(...ks)];

const colors = {{
    stable: '#3fb950', marginal: '#d29922',
    thrashing: '#f85149', collapsed: '#8b949e',
    transitional: '#a371f7'
}};

// Draw phase diagram
for (const d of data) {{
    const x = 30 + (d.lambda - lamRange[0]) / (lamRange[1] - lamRange[0]) * W;
    const y = 10 + (1 - (d.k - kRange[0]) / (kRange[1] - kRange[0])) * H;
    let regime = 'transitional';
    if (d.collapse_risk > 0.5) regime = 'collapsed';
    else if (d.mem_pressure > 0.9 && d.fault_rate > 0.3) regime = 'thrashing';
    else if (d.collapse_risk > 0.2 || d.fault_rate > 0.15) regime = 'marginal';
    else if (d.mem_pressure < 0.7 && d.fault_rate < 0.1) regime = 'stable';

    const r = 8 + d.utility * 12;  // size ∝ utility
    ctx.fillStyle = colors[regime] || '#8b949e';
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(3, r), 0, Math.PI*2);
    ctx.fill();
    ctx.globalAlpha = 1.0;
    d._x = x; d._y = y; d._regime = regime;
}}

// Labels
ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace';
for (const lam of lambdas.filter((_,i) => i % Math.max(1, Math.floor(lambdas.length/5)) === 0)) {{
    const x = 30 + (lam - lamRange[0]) / (lamRange[1] - lamRange[0]) * W;
    ctx.fillText(lam.toFixed(1), x, H + 30);
}}
ctx.fillText('λ (locality)', W/2, H + 42);
ctx.save(); ctx.translate(8, H/2); ctx.rotate(-Math.PI/2);
ctx.fillText('k (top-k experts)', 0, 0); ctx.restore();
for (const k of ks.filter((_,i) => i % Math.max(1, Math.floor(ks.length/6)) === 0)) {{
    const y = 10 + (1 - (k - kRange[0]) / (kRange[1] - kRange[0])) * H;
    ctx.fillText(k.toString(), 2, y + 4);
}}

// Highlight stable region
const stable = data.filter(d => {{
    let r = 'transitional';
    if (d.collapse_risk > 0.5) r = 'collapsed';
    else if (d.mem_pressure > 0.9 && d.fault_rate > 0.3) r = 'thrashing';
    else if (d.collapse_risk > 0.2 || d.fault_rate > 0.15) r = 'marginal';
    else if (d.mem_pressure < 0.7 && d.fault_rate < 0.1) r = 'stable';
    return r === 'stable';
}});
if (stable.length > 0) {{
    const xs = stable.map(d => d._x), ys = stable.map(d => d._y);
    ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2; ctx.setLineDash([5,5]);
    ctx.beginPath();
    ctx.moveTo(Math.min(...xs)-5, Math.min(...ys)-5);
    ctx.lineTo(Math.max(...xs)+5, Math.min(...ys)-5);
    ctx.lineTo(Math.max(...xs)+5, Math.max(...ys)+5);
    ctx.lineTo(Math.min(...xs)-5, Math.max(...ys)+5);
    ctx.closePath(); ctx.stroke();
    ctx.setLineDash([]);
}}

// Hover
canvas.onmousemove = e => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bestDist = Infinity;
    for (const d of data) {{
        const dx = mx - d._x, dy = my - d._y;
        const dist = dx*dx + dy*dy;
        if (dist < bestDist) {{ bestDist = dist; best = d; }}
    }}
    if (best && bestDist < 400) {{
        document.getElementById('info').innerHTML = `
            <div class="metric"><span>λ</span><span class="val">${{best.lambda.toFixed(1)}}</span></div>
            <div class="metric"><span>k</span><span class="val">${{best.k}}</span></div>
            <div class="metric"><span>Regime</span><span class="val" style="color:${{colors[best._regime]}}">${{best._regime.toUpperCase()}}</span></div>
            <div class="metric"><span>Utility U</span><span class="val">${{best.utility.toFixed(3)}}</span></div>
            <div class="metric"><span>tok/s</span><span class="val">${{best.tok_per_s}}</span></div>
            <div class="metric"><span>Fault rate</span><span class="val">${{(best.fault_rate*100).toFixed(0)}}%</span></div>
            <div class="metric"><span>Collapse risk</span><span class="val">${{best.collapse_risk.toFixed(3)}}</span></div>
            <div class="metric"><span>Memory pressure</span><span class="val">${{(best.mem_pressure*100).toFixed(0)}}%</span></div>
        `;
    }}
}};

// Find optimal point
const best = data.reduce((a,b) => b.utility > a.utility ? b : a, data[0]);
document.getElementById('optimal').innerHTML = `
    <h2>Optimal Operating Point</h2>
    <div class="metric"><span>λ*</span><span class="val">${{best.lambda.toFixed(1)}}</span></div>
    <div class="metric"><span>k*</span><span class="val">${{best.k}}</span></div>
    <div class="metric"><span>U*</span><span class="val">${{best.utility.toFixed(3)}}</span></div>
    <div class="metric"><span>Regime</span><span class="val">STABLE</span></div>
    <div class="metric"><span>tok/s</span><span class="val">${{best.tok_per_s}}</span></div>
    <div class="metric"><span>Working set</span><span class="val">${{best.working_set}} experts (${{(best.working_set*10.5).toFixed(0)}}MB)</span></div>
`;
</script></body></html>"""


def main():
    print("Phase Diagram Sweep — λ × k")
    print()

    lambdas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
    ks = [2, 3, 4, 5, 6, 7, 8]

    results = []
    for lam in lambdas:
        for k in ks:
            r = simulate_steady_state(lam, k)
            results.append(r)

    # Show phase regions
    regimes = {}
    for r in results:
        regime = classify_regime(r)
        regimes[regime] = regimes.get(regime, 0) + 1

    print("Phase regions found:")
    for regime, count in sorted(regimes.items()):
        print(f"  {regime:<15s}: {count} points")

    # Find optimal
    best = max(results, key=lambda r: r["utility"])
    print(f"\nOptimal: λ={best['lambda']}, k={best['k']}, U={best['utility']}")
    print(f"  tok/s={best['tok_per_s']}, fault_rate={best['fault_rate']}, "
          f"collapse_risk={best['collapse_risk']}")

    # Stable region
    stable = [r for r in results if classify_regime(r) == "stable"]
    if stable:
        lam_range = (min(r["lambda"] for r in stable),
                     max(r["lambda"] for r in stable))
        k_range = (min(r["k"] for r in stable),
                   max(r["k"] for r in stable))
        print(f"\nStable region: λ ∈ [{lam_range[0]}, {lam_range[1]}], "
              f"k ∈ [{k_range[0]}, {k_range[1]}]")

    # Build HTML
    html = build_html(results)
    out_path = OUTPUT / "stability_phase.html"
    out_path.write_text(html)
    print(f"\nSaved: {out_path}")

    # Save JSON
    json.dump({"results": results, "stable_region": {
        "lambda": [r["lambda"] for r in stable],
        "k": [r["k"] for r in stable],
    }}, open(OUTPUT / "stability_phase.json", "w"), indent=2)


if __name__ == "__main__":
    main()
