"""objeta OS — OpenAI-compatible API Server.

POST /v1/chat/completions  — chat completion with SSE streaming
POST /v1/completions        — text completion
GET  /v1/models             — available models
GET  /health                — health check

Every response includes objeta telemetry:
  collapse_events, avg_entropy, skip_rate, precision_mix, token_classes

Usage:
  python server.py [--port 8000] [--model tinyllama]
"""

from __future__ import annotations

import json, sys, time, uuid
from pathlib import Path
from typing import Optional

PROJECT = Path(__file__).parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO))
sys.path.insert(0, str(PROJECT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np

from os_runtime import OSRuntime, SchedulerConfig
from os_runtime.logging import RuntimeLogger, LogLevel

# ── FastAPI app ──

app = FastAPI(
    title="objeta OS Runtime",
    version="1.0.0",
    description="OpenAI-compatible LLM inference with observable OS telemetry",
)

# ── Global model registry ──

_models: dict[str, dict] = {}
_default_model: str | None = None


# ── OpenAI-compatible Schemas ──

class Message(BaseModel):
    role: str = "user"
    content: str = ""

class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message] = Field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 40
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 40
    stream: bool = False

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1716000000
    owned_by: str = "objeta"


# ── Model loading ──

def register_tinyllama(name: str = "tinyllama-1.1b"):
    """Register TinyLlama-1.1B-Chat (MLX backend)."""
    from runtime.models.llm import LLM, ModelConfig
    from runtime.models.loaders.model_loader import ModelLoader
    from transformers import AutoTokenizer

    MODEL_PATH = (
        "/Users/yasuno/.cache/huggingface/hub/"
        "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/"
        "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    )

    loader = ModelLoader(MODEL_PATH)
    config = ModelConfig(
        hidden_dim=2048, ffn_dim=5632, n_layers=22,
        n_heads=32, n_kv_heads=4, head_dim=64, vocab_size=32000,
    )
    weights = loader.load_weights()
    llm = LLM(weights, config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    _models[name] = {
        "llm": llm,
        "tokenizer": tokenizer,
        "family": "residual_transport",
        "n_layers": 22,
        "hidden_dim": 2048,
    }
    return name


def register_stories_moe(name: str = "stories15m-moe"):
    """Register stories15M_MOE (PyTorch backend)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_PATH = (
        "/Users/yasuno/.cache/huggingface/hub/"
        "models--ggml-org--stories15M_MOE/snapshots/"
        "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, device_map="cpu",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    _models[name] = {
        "model": model,
        "tokenizer": tokenizer,
        "family": "spherical_steering",
        "n_layers": 6,
        "n_experts": 4,
        "top_k": 2,
        "hidden_dim": 288,
    }
    return name


# ── Generation ──

def generate_chat(model_name: str, messages: list[Message],
                  max_tokens: int, temperature: float,
                  top_k: int, telemetry_callback=None
                  ) -> tuple[list[int], str, dict]:
    """Generate chat completion with OS telemetry."""
    entry = _models.get(model_name)
    if not entry:
        raise HTTPException(404, f"Model '{model_name}' not found")

    tokenizer = entry["tokenizer"]

    # Build prompt
    if "llm" in entry:
        # TinyLlama: use chat template
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    else:
        # stories15M: simple concatenation
        prompt = " ".join(m.content for m in messages)

    # OS config
    family = entry.get("family", "residual_transport")
    os_config = SchedulerConfig(
        family=family,
        backbone="attention" if family == "residual_transport" else "steering",
        fusion_ratio=0.5 if family == "residual_transport" else 1.0,
    )

    logger = RuntimeLogger(level=LogLevel.WARNING)  # quiet in server mode

    if "llm" in entry:
        # MLX path (TinyLlama)
        os = OSRuntime(entry["llm"], os_config, logger)
        tokens = os.generate(
            prompt, tokenizer=tokenizer,
            max_tokens=max_tokens, temperature=temperature, top_k=top_k,
        )
    else:
        # PyTorch path (stories15M MoE) — full scheduler integration
        import torch
        from os_runtime.scheduler import Scheduler
        from os_runtime.observation import compute_entropy, compute_steering

        model = entry["model"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        sched = Scheduler(os_config, entry.get("n_layers", 6))

        generated = list(input_ids[0].tolist())
        prev_hidden = None
        prev_token = generated[-1]

        logger.start_run()

        with torch.no_grad():
            for gen_idx in range(max_tokens):
                t0 = time.perf_counter()

                outputs = model(
                    torch.tensor([generated]),
                    output_router_logits=True,
                    output_hidden_states=True,
                )
                logits = outputs.logits[0, -1, :].cpu().numpy()
                hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()

                # Full observation
                entropy = compute_entropy(logits)
                top1 = int(np.argmax(logits))

                steering = 0.0
                if prev_hidden is not None:
                    steering = compute_steering(hidden, prev_hidden)
                prev_hidden = hidden.copy()

                # Classify through scheduler
                tc = sched.begin_token(
                    entropy, steering,
                    prev_token_id=prev_token,
                    predicted_token_id=top1,
                )

                # Per-layer dispatch
                layer_actions = []
                for l in range(entry.get("n_layers", 6)):
                    run_attn = sched.should_run_attn(l)
                    run_ffn = sched.should_run_ffn(l)
                    prec = sched.get_precision(l)
                    layer_actions.append({
                        "layer": l, "attn": run_attn,
                        "ffn": run_ffn, "prec": prec,
                    })

                forward_ms = (time.perf_counter() - t0) * 1000

                # Log token
                from os_runtime.logging import TokenLog, LayerAction as LA
                tlog = TokenLog(
                    token_idx=gen_idx,
                    token_id=top1,
                    entropy=entropy,
                    steering=steering,
                    top1_logit=float(logits[top1]),
                    is_repeat=(top1 == prev_token),
                    token_class=tc.value,
                    collapse_score={
                        "healthy": 0.0, "warning": 0.5, "critical": 1.0,
                    }.get(sched.state.collapse_status.value, 0.0),
                    collapse_status=sched.state.collapse_status.value,
                    precision=sched.state.precision,
                    layers_run=sched.layers_run,
                    layers_skipped=sched.layers_skipped,
                    skip_rate=sched.stats()["skip_rate"],
                    layer_actions=[
                        LA(layer=a["layer"], attn_ran=a["attn"],
                           ffn_ran=a["ffn"], precision_used=a["prec"])
                        for a in layer_actions
                    ],
                    forward_ms=forward_ms,
                )
                logger.log_token(tlog)

                # Sampling
                if temperature == 0:
                    next_token = top1
                else:
                    scaled = logits / max(temperature, 0.01)
                    scaled -= scaled.max()
                    probs = np.exp(scaled.astype(np.float64))
                    probs /= probs.sum()
                    if top_k > 0 and top_k < len(probs):
                        idx = np.argpartition(-probs, top_k)[:top_k]
                        p = probs[idx]; p /= p.sum()
                        next_token = int(idx[np.random.choice(len(idx), p=p)])
                    else:
                        next_token = int(np.random.choice(len(probs), p=probs))

                if next_token == tokenizer.eos_token_id:
                    break
                generated.append(next_token)
                prev_token = next_token

        logger.end_run()
        tokens = generated[len(input_ids[0]):]

    text = tokenizer.decode(tokens) if tokens else ""
    summary = logger.run_summary()

    objeta_telemetry = {
        "tokens_generated": len(tokens),
        "collapse_events": summary.get("collapse_events", 0),
        "avg_entropy": round(summary.get("avg_entropy", 0.0), 4),
        "avg_steering": round(summary.get("avg_steering", 0.0), 4),
        "skip_rate": round(summary.get("avg_skip_rate", 0.0), 3),
        "avg_precision": round(summary.get("avg_precision", 16.0), 1),
        "token_classes": summary.get("token_classes", {}),
    }

    return tokens, text, objeta_telemetry


# ── WebSocket telemetry stream ──

from fastapi import WebSocket, WebSocketDisconnect
import asyncio

_ws_clients: list[WebSocket] = []


async def broadcast_telemetry(data: dict):
    """Send telemetry to all connected dashboard clients."""
    for ws in _ws_clients[:]:
        try:
            await ws.send_json(data)
        except Exception:
            _ws_clients.remove(ws)


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        _ws_clients.remove(ws)


# ── Dashboard (single-file HTML) ──

@app.get("/dashboard")
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


from fastapi.responses import HTMLResponse

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>objeta OS — Telemetry Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font:14px 'SF Mono',monospace;padding:16px}
h1{font-size:18px;color:#58a6ff;margin-bottom:8px}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:12px}
.panel{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px}
.panel h2{font-size:13px;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
.metric{display:flex;justify-content:space-between;padding:2px 0;font-size:12px}
.metric .val{color:#58a6ff;font-weight:bold}
.timeline{display:flex;gap:3px;overflow-x:auto;padding:4px 0}
.token-bar{min-width:28px;height:60px;border-radius:3px;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;font-size:9px;padding:2px}
.token-bar .id{writing-mode:vertical-lr;font-size:7px;color:#8b949e;margin-bottom:2px}
.heatmap{display:grid;gap:1px}
.heatmap-row{display:flex;gap:1px;align-items:center}
.heatmap-label{width:24px;font-size:9px;color:#8b949e}
.heatmap-cell{width:18px;height:12px;border-radius:1px}
.collapse{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px;margin-bottom:4px;font-size:11px}
.collapse.WARNING{border-color:#d29922}
.collapse.CRITICAL{border-color:#f85149;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.legend{display:flex;gap:12px;font-size:10px;margin-top:4px;flex-wrap:wrap}
.legend span{display:flex;align-items:center;gap:4px}
.legend .swatch{width:10px;height:10px;border-radius:2px}
.status{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px}
.status.healthy{background:#1a3a1a;color:#3fb950}
.status.warning{background:#3a2a1a;color:#d29922}
.status.critical{background:#3a1a1a;color:#f85149}
pre{white-space:pre-wrap;word-break:break-all;font-size:11px;color:#8b949e}
</style>
</head>
<body>
<h1>⚡ objeta OS — Telemetry Dashboard</h1>
<div id="status">Connecting...</div>

<div class="grid">
  <div class="panel">
    <h2>Token Timeline</h2>
    <div class="legend">
      <span><span class="swatch" style="background:#3fb950"></span> fp16</span>
      <span><span class="swatch" style="background:#58a6ff"></span> q8</span>
      <span><span class="swatch" style="background:#d29922"></span> q4</span>
      <span><span class="swatch" style="background:#f85149"></span> q3</span>
    </div>
    <div id="timeline" class="timeline"></div>
  </div>
  <div class="panel">
    <h2>OS State</h2>
    <div id="telemetry"></div>
    <div style="margin-top:8px">
      <h2>Precision Mix</h2>
      <div id="precision"></div>
    </div>
  </div>
</div>

<div class="panel" style="margin-bottom:12px">
  <h2>Layer Heatmap (per-token)</h2>
  <div class="legend">
    <span><span class="swatch" style="background:#3fb950"></span> Full attn</span>
    <span><span class="swatch" style="background:#30363d"></span> Skip</span>
    <span><span class="swatch" style="background:#58a6ff"></span> Cached</span>
    <span><span class="swatch" style="background:#d29922"></span> Reduced</span>
  </div>
  <div id="heatmap" class="heatmap"></div>
</div>

<div id="collapses"></div>

<div class="panel">
  <h2>Generated Text</h2>
  <pre id="output"></pre>
</div>

<script>
const ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
let tokens = [];
let currentCollapse = null;
let layerHistory = [];

const PREC_COLORS = {16:'#3fb950', 8:'#58a6ff', 5:'#a371f7', 4:'#d29922', 3:'#f85149'};
const ATTN_COLORS = {'Full':'#3fb950','Cached':'#58a6ff','Reduced':'#d29922','Skip':'#30363d'};

function bar(token) {
  const h = Math.min(50, Math.max(10, token.steering * 80));
  const c = PREC_COLORS[token.precision] || '#8b949e';
  return `<div class="token-bar" style="height:${h+10}px;background:${c}"
    title="tok=${token.idx} id=${token.id} ent=${token.entropy.toFixed(3)} steer=${token.steering.toFixed(3)} prec=${token.precision}bit ${token.class} @ ${token.latency_ms}ms">
    <span class="id">${token.idx}</span></div>`;
}

function heatmapCell(layer, col) {
  const action = col === 'skip' ? 'Skip' : col === 'cached' ? 'Cached' : col === 'reduced' ? 'Reduced' : 'Full';
  const c = ATTN_COLORS[action] || '#30363d';
  return `<div class="heatmap-cell" style="background:${c}" title="L${layer} ${action}"></div>`;
}

ws.onopen = () => document.getElementById('status').innerHTML = '<span class="status healthy">CONNECTED</span>';

ws.onmessage = (e) => {
  const data = JSON.parse(e.data);

  // Token event
  if (data.token) {
    tokens.push(data.token);
    document.getElementById('timeline').innerHTML = tokens.map(bar).join('');
    document.getElementById('output').textContent = data.text || '';

    // Layer heatmap
    if (data.token.layer_actions) {
      layerHistory.push(data.token.layer_actions);
      if (layerHistory.length > 30) layerHistory.shift();

      let html = '';
      const nLayers = data.token.layer_actions.length;
      for (let l = 0; l < nLayers; l++) {
        html += '<div class="heatmap-row">';
        html += `<span class="heatmap-label">L${l}</span>`;
        for (let t = 0; t < layerHistory.length; t++) {
          const la = layerHistory[t][l];
          const attn = la.attn ? 'Full' : 'Skip';
          const c = ATTN_COLORS[attn] || '#30363d';
          html += `<div class="heatmap-cell" style="background:${c}" title="L${l} T${t} attn=${la.attn} ffn=${la.ffn} prec=${la.prec}"></div>`;
        }
        html += '</div>';
      }
      document.getElementById('heatmap').innerHTML = html;
    }

    // Telemetry
    const telem = data.telemetry || {};
    document.getElementById('telemetry').innerHTML = `
      <div class="metric"><span>Token class</span><span class="val">${data.token.class}</span></div>
      <div class="metric"><span>Entropy</span><span class="val">${data.token.entropy.toFixed(4)}</span></div>
      <div class="metric"><span>Steering</span><span class="val">${data.token.steering.toFixed(4)}</span></div>
      <div class="metric"><span>Precision</span><span class="val">${data.token.precision}bit</span></div>
      <div class="metric"><span>Collapse status</span><span class="val">${data.token.collapse || 'healthy'}</span></div>
      <div class="metric"><span>Skip rate</span><span class="val">${(telem.skip_rate*100).toFixed(0)}%</span></div>
    `;

    // Precision histogram
    if (telem.precision_mix) {
      document.getElementById('precision').innerHTML = Object.entries(telem.precision_mix)
        .map(([k,v]) => `<div class="metric"><span>${k}bit</span><span class="val">${(v*100).toFixed(0)}%</span></div>`).join('');
    }

    // Collapse events
    if (data.token.collapse && data.token.collapse !== 'healthy') {
      const cls = data.token.collapse.toUpperCase();
      currentCollapse = {cls, idx: data.token.idx, entropy: data.token.entropy, steering: data.token.steering};
      const el = document.getElementById('collapses');
      el.innerHTML += `<div class="collapse ${cls}">
        <span class="status ${data.token.collapse}">${cls}</span>
        token=${data.token.idx} ent=${data.token.entropy.toFixed(3)} steer=${data.token.steering.toFixed(3)}
      </div>`;
    }
  }

  // Status updates
  if (data.status) {
    document.getElementById('status').innerHTML = `<span class="status ${data.status}">${data.status.toUpperCase()}</span>`;
  }
};

ws.onclose = () => document.getElementById('status').innerHTML = '<span class="status critical">DISCONNECTED</span>';
</script>
</body>
</html>"""


# ── Endpoints ──

@app.get("/health")
async def health():
    return {"status": "ok", "models": list(_models.keys())}


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            ModelInfo(id=name).model_dump()
            for name in _models
        ],
    })


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    model = req.model or _default_model
    if not model:
        raise HTTPException(400, "No model specified")

    # Broadcast callback for real-time dashboard
    def on_token(tok_id: int, tok_text: str, tlog):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(broadcast_telemetry({
                "token": {
                    "idx": tlog.token_idx,
                    "id": tok_id,
                    "entropy": round(tlog.entropy, 4),
                    "steering": round(tlog.steering, 4),
                    "precision": tlog.precision,
                    "class": tlog.token_class,
                    "collapse": tlog.collapse_status,
                    "latency_ms": round(tlog.forward_ms, 1),
                    "layer_actions": [
                        {"layer": a.layer, "attn": a.attn_ran,
                         "ffn": a.ffn_ran, "prec": a.precision_used}
                        for a in (tlog.layer_actions or [])
                    ] if tlog.layer_actions else [],
                },
                "text": tok_text,
                "telemetry": {
                    "skip_rate": tlog.skip_rate,
                },
            }))
        except Exception:
            pass

    tokens, text, telemetry = generate_chat(
        model, req.messages, req.max_tokens,
        req.temperature, req.top_k,
        telemetry_callback=on_token,
    )

    if req.stream:
        return StreamingResponse(
            _stream_chat(model, tokens, text, telemetry),
            media_type="text/event-stream",
        )

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(tokens),
            "total_tokens": len(tokens),
        },
        "objeta": telemetry,
    })


async def _stream_chat(model: str, tokens: list[int],
                       text: str, telemetry: dict):
    """SSE streaming response."""
    from os_runtime.observation import compute_entropy  # noqa: F811

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Stream each token
    for i, tok in enumerate(tokens):
        token_text = ""
        try:
            entry = _models.get(model, {})
            if entry:
                token_text = entry["tokenizer"].decode([tok])
        except Exception:
            pass

        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk with telemetry
    final = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "objeta": telemetry,
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    model = req.model or _default_model
    if not model:
        raise HTTPException(400, "No model specified")

    msg = Message(role="user", content=req.prompt)
    tokens, text, telemetry = generate_chat(
        model, [msg], req.max_tokens,
        req.temperature, req.top_k,
    )

    if req.stream:
        return StreamingResponse(
            _stream_chat(model, tokens, text, telemetry),
            media_type="text/event-stream",
        )

    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(tokens),
            "total_tokens": len(tokens),
        },
        "objeta": telemetry,
    })


# ── Startup ──

@app.on_event("startup")
async def startup():
    global _default_model

    print("═" * 50)
    print("  objeta OS Runtime Server")
    print("═" * 50)
    print()

    # Register TinyLlama
    try:
        name = register_tinyllama()
        _default_model = name
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ tinyllama: {e}")

    # Register stories15M_MOE
    try:
        name = register_stories_moe()
        if not _default_model:
            _default_model = name
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ stories-moe: {e}")

    print()
    print(f"  Default model: {_default_model}")
    print(f"  Endpoints:")
    print(f"    POST /v1/chat/completions")
    print(f"    POST /v1/completions")
    print(f"    GET  /v1/models")
    print(f"    GET  /health")
    print()


# ── Main ──

if __name__ == "__main__":
    import argparse, uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.model:
        _default_model = args.model

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
