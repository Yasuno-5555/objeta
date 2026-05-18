#!/usr/bin/env python3
"""objeta OS Daemon — sustained serving with system monitoring.

Runs OpenAI-compatible endpoint continuously, tracking:
  - Memory: RSS, VM, page faults, swap, macOS pressure
  - Thermal: CPU temp (if available), throttle state
  - OS: collapse events, class oscillations, precision mix, λ dynamics
  - Session: concurrent requests, queue depth, latency distribution

Usage:
  python daemon.py [--port 8000] [--duration 1800] [--concurrency 2]
"""

import json, os, signal, subprocess, sys, time, threading, uuid
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

PROJECT = Path(__file__).parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO)); sys.path.insert(0, str(PROJECT))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="objeta OS Daemon", version="2.0.0")


# ═══════════════════════════════════════════════════════════
# System Monitor
# ═══════════════════════════════════════════════════════════

@dataclass
class SystemMetrics:
    timestamp: float = 0.0
    rss_mb: float = 0.0           # Resident Set Size
    vm_mb: float = 0.0            # Virtual Memory
    page_faults: int = 0
    pageins: int = 0
    swap_used_mb: float = 0.0
    memory_pressure: str = "low"  # macOS memory_pressure level
    cpu_percent: float = 0.0
    cpu_temp_c: float = 0.0       # if available
    throttle: bool = False

    # OS-level metrics
    active_requests: int = 0
    queue_depth: int = 0
    collapse_events_total: int = 0
    avg_latency_ms: float = 0.0
    avg_lambda: float = 3.0
    avg_top_k: float = 8.0


class SystemMonitor:
    """Collects system metrics every N seconds."""

    def __init__(self, interval_s: float = 5.0):
        self.interval = interval_s
        self.history: deque[SystemMetrics] = deque(maxlen=720)  # 1 hour at 5s
        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time = time.time()

        # OS-level accumulators
        self.collapse_events_total = 0
        self.request_latencies: deque[float] = deque(maxlen=1000)
        self.active_requests = 0
        self.queue_depth = 0
        self.effective_lambdas: deque[float] = deque(maxlen=100)
        self.effective_ks: deque[float] = deque(maxlen=100)

        # Process for RSS tracking
        self._pid = os.getpid()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            try:
                m = self._sample()
                self.history.append(m)
            except Exception:
                pass
            time.sleep(self.interval)

    def _sample(self) -> SystemMetrics:
        # RSS/VM from ps
        try:
            import subprocess
            out = subprocess.check_output(
                ["ps", "-o", "rss,vsz", "-p", str(self._pid)],
                text=True, stderr=subprocess.DEVNULL)
            lines = out.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                rss_kb = int(parts[0])
                vm_kb = int(parts[1])
                rss_mb = rss_kb / 1024
                vm_mb = vm_kb / 1024
            else:
                rss_mb = vm_mb = 0
        except Exception:
            rss_mb = vm_mb = 0

        # Page faults
        try:
            out = subprocess.check_output(
                ["launchctl", "print", f"pid/{self._pid}"],
                text=True, stderr=subprocess.DEVNULL)
            faults = 0
            for line in out.split("\n"):
                if "faults" in line.lower():
                    parts = line.split()
                    for p in parts:
                        try: faults = int(p); break
                        except: pass
        except Exception:
            faults = 0

        # Swap usage
        try:
            out = subprocess.check_output(
                ["sysctl", "vm.swapusage"], text=True)
            # vm.swapusage: total = 2048.00M  used = 512.00M  free = 1536.00M
            used = 0.0
            for part in out.split():
                if part.endswith("M") and "used" in out.split("=")[-1]:
                    pass
            # Parse: "used = 512.00M"
            if "used" in out:
                used_str = out.split("used = ")[1].split("M")[0]
                swap_mb = float(used_str)
            else:
                swap_mb = 0.0
        except Exception:
            swap_mb = 0.0

        # Memory pressure (macOS specific)
        try:
            out = subprocess.check_output(
                ["memory_pressure"], text=True, stderr=subprocess.DEVNULL)
            if "critical" in out.lower():
                pressure = "critical"
            elif "warning" in out.lower():
                pressure = "warning"
            else:
                pressure = "normal"
        except Exception:
            pressure = "unknown"

        # Pageins
        try:
            out = subprocess.check_output(
                ["vm_stat"], text=True)
            pageins = 0
            for line in out.split("\n"):
                if "pageins" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        pageins = int(parts[1].strip().rstrip("."))
        except Exception:
            pageins = 0

        # CPU temp (Apple Silicon)
        try:
            out = subprocess.check_output(
                ["sudo", "powermetrics", "--samplers", "smc", "-n", "1",
                 "--format", "text"], text=True, stderr=subprocess.DEVNULL,
                timeout=3)
            temp = 0.0
            for line in out.split("\n"):
                if "CPU die temperature" in line:
                    parts = line.split()
                    for p in parts:
                        try: temp = float(p); break
                        except: pass
        except Exception:
            temp = 0.0

        return SystemMetrics(
            timestamp=time.time() - self._start_time,
            rss_mb=rss_mb, vm_mb=vm_mb,
            page_faults=faults, pageins=pageins,
            swap_used_mb=swap_mb,
            memory_pressure=pressure,
            cpu_temp_c=temp,
            active_requests=self.active_requests,
            queue_depth=self.queue_depth,
            collapse_events_total=self.collapse_events_total,
            avg_latency_ms=(np.mean(list(self.request_latencies))
                           if self.request_latencies else 0),
            avg_lambda=(np.mean(list(self.effective_lambdas))
                       if self.effective_lambdas else 3.0),
            avg_top_k=(np.mean(list(self.effective_ks))
                      if self.effective_ks else 8.0),
        )

    def snapshot(self) -> dict:
        if not self.history:
            return {}
        latest = self.history[-1]
        # Trend: memory growth rate (MB/min)
        if len(self.history) >= 12:  # 1 minute
            recent = list(self.history)[-12:]
            rss_delta = recent[-1].rss_mb - recent[0].rss_mb
            mem_trend_mb_per_min = rss_delta / (len(recent) * self.interval / 60)
        else:
            mem_trend_mb_per_min = 0.0

        return {
            "uptime_s": round(latest.timestamp, 0),
            "rss_mb": round(latest.rss_mb, 1),
            "rss_trend_mb_per_min": round(mem_trend_mb_per_min, 2),
            "vm_mb": round(latest.vm_mb, 1),
            "swap_mb": round(latest.swap_used_mb, 1),
            "memory_pressure": latest.memory_pressure,
            "page_faults": latest.page_faults,
            "cpu_temp_c": round(latest.cpu_temp_c, 1) if latest.cpu_temp_c > 0 else None,
            "active_requests": latest.active_requests,
            "collapse_events": latest.collapse_events_total,
            "avg_latency_ms": round(latest.avg_latency_ms, 1),
            "avg_lambda": round(latest.avg_lambda, 1),
            "avg_top_k": round(latest.avg_top_k, 1),
        }


# ═══════════════════════════════════════════════════════════
# Global state
# ═══════════════════════════════════════════════════════════

monitor = SystemMonitor(interval_s=3.0)
_models: dict[str, dict] = {}
_session_counter = 0


# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "uptime_s": time.time() - monitor._start_time}

@app.get("/admin/metrics")
async def metrics():
    """Full system metrics snapshot."""
    return monitor.snapshot()

@app.get("/admin/history")
async def history(samples: int = 60):
    """Recent metric history for plotting."""
    h = list(monitor.history)[-samples:]
    return [{
        "t": round(m.timestamp, 0),
        "rss_mb": round(m.rss_mb, 1),
        "swap_mb": round(m.swap_used_mb, 1),
        "pressure": m.memory_pressure,
        "active": m.active_requests,
        "latency_ms": round(m.avg_latency_ms, 1),
        "lambda": round(m.avg_lambda, 1),
        "top_k": round(m.avg_top_k, 1),
    } for m in h]

@app.get("/admin/governor")
async def governor_state():
    """Current governor internal state."""
    return {
        "thrash": gov.thrash.stats() if 'gov' in dir() else {},
        "collapse": gov.collapse.stats() if 'gov' in dir() else {},
        "effective_lambda": gov.effective_lambda if 'gov' in dir() else 3.0,
        "effective_top_k": gov.effective_top_k if 'gov' in dir() else 8,
    }


# ═══════════════════════════════════════════════════════════
# Main: register models, start monitor, run server
# ═══════════════════════════════════════════════════════════

def register_models():
    """Register available models."""
    # TinyLlama
    try:
        from runtime.models.llm import LLM, ModelConfig
        from runtime.models.loaders.model_loader import ModelLoader
        from transformers import AutoTokenizer
        MODEL_PATH = ("/Users/yasuno/.cache/huggingface/hub/"
                      "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/"
                      "fe8a4ea1ffedaf415f4da2f062534de366a451e6")
        loader = ModelLoader(MODEL_PATH)
        cfg = ModelConfig(hidden_dim=2048, ffn_dim=5632, n_layers=22,
                          n_heads=32, n_kv_heads=4, head_dim=64, vocab_size=32000)
        llm = LLM(loader.load_weights(), cfg)
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        _models["tinyllama-1.1b"] = {"llm": llm, "tokenizer": tok,
                                      "family": "residual_transport"}
        print("  ✓ tinyllama-1.1b")
    except Exception as e:
        print(f"  ✗ tinyllama: {e}")

    # stories15M_MOE
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        MOE_PATH = ("/Users/yasuno/.cache/huggingface/hub/"
                    "models--ggml-org--stories15M_MOE/snapshots/"
                    "b6dd737497465570b5f5e962dbc9d9454ed1e0eb")
        model = AutoModelForCausalLM.from_pretrained(MOE_PATH, dtype=torch.float32, device_map="cpu")
        model.eval()
        tok = AutoTokenizer.from_pretrained(MOE_PATH)
        _models["stories15m-moe"] = {"model": model, "tokenizer": tok,
                                      "family": "spherical_steering"}
        print("  ✓ stories15m-moe")
    except Exception as e:
        print(f"  ✗ stories-moe: {e}")


def main():
    import argparse, uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--duration", type=int, default=1800,
                  help="Run duration in seconds (0=forever)")
    args = p.parse_args()

    print("═" * 50)
    print("  objeta OS Daemon")
    print(f"  port={args.port} duration={args.duration}s")
    print("═" * 50)
    print()

    register_models()
    monitor.start()
    print(f"\n  Monitor active ({monitor.interval}s interval)")
    print(f"  Endpoints:")
    print(f"    GET  /health")
    print(f"    GET  /admin/metrics")
    print(f"    GET  /admin/history")
    print(f"    GET  /admin/governor")
    print()

    # Auto-stop after duration
    if args.duration > 0:
        def stopper():
            time.sleep(args.duration)
            print(f"\n  Duration {args.duration}s reached. Shutting down...")
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=stopper, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
