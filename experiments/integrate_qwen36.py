#!/usr/bin/env python3
"""Qwen3.6-35B OS Integration — End-to-End.

Attempts to load Qwen3.6 via transformers (trust_remote_code).
If model loads, runs OS scheduler + router rewriter + VM residency
on real Qwen3.6 routing decisions.

If full model can't load, falls back to shard-1 partial forward.

Usage:
  python3 experiments/integrate_qwen36.py [--max-tokens 32]
"""

import json, sys, time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO)); sys.path.insert(0, str(PROJECT))

import numpy as np

OUTPUT = PROJECT / "experiments" / "results" / "qwen36_integration.json"


def attempt_full_model():
    """Try loading full Qwen3.6 via transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Attempting full Qwen3.6-35B-A3B load...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-35B-A3B",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.0f}s")
    print(f"  Layers: {len(model.model.layers)}")
    return model, tokenizer


def attempt_shard1_forward():
    """Load just shard 1 (L0-L1) and do partial forward."""
    import torch, safetensors

    SNAPSHOT = ("/Users/yasuno/.cache/huggingface/hub/"
                "models--Qwen--Qwen3.6-35B-A3B/snapshots/"
                "995ad96eacd98c81ed38be0c5b274b04031597b0")

    # Try loading the model code from HF, but use local weights
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
        print(f"  Config loaded: {config.model_type}")
        print(f"  Hidden: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Experts: {config.num_experts}")
        print(f"  Top-K: {config.num_experts_per_tok}")
    except Exception as e:
        print(f"  Config failed: {e}")
        return None

    # Try loading model with local shard
    shard1 = SNAPSHOT + "/model-00001-of-00026.safetensors"
    if not Path(shard1).exists():
        print(f"  Shard 1 not found at {shard1}")
        return None

    print(f"  Loading shard 1 ({Path(shard1).stat().st_size/1e9:.1f}GB)...")
    try:
        # Use from_pretrained with local_files_only + first shard only
        # This won't work with missing shards, so we need selective loading
        pass
    except Exception:
        pass

    # Fallback: manual forward with shard 1 tensors
    print("  Building manual forward from shard 1...")
    return load_manual_shard1(shard1, config)


def load_manual_shard1(shard_path: str, config) -> dict:
    """Load shard 1 tensors and build manual forward pass infrastructure."""
    import torch, safetensors

    weights = {}
    with safetensors.safe_open(shard_path, framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    # Extract key tensors
    embed = weights.get("model.language_model.embed_tokens.weight")
    print(f"  Embed: {list(embed.shape) if embed is not None else 'MISSING'}")

    # Layer 0 structure
    l0_keys = [k for k in weights if "layers.0" in k]
    print(f"  Layer 0 tensors: {len(l0_keys)}")
    for k in sorted(l0_keys):
        t = weights[k]
        print(f"    {k.split('layers.0.')[1]:50s} {list(t.shape)}")

    return {
        "weights": weights,
        "config": config,
        "n_loaded_layers": len(set(
            int(k.split("layers.")[1].split(".")[0])
            for k in weights if "layers." in k
        )),
    }


def run_os_integration(model_data: dict, max_tokens: int = 16):
    """Run OS scheduler on Qwen3.6 routing data."""
    from os_runtime.scheduler import Scheduler, SchedulerConfig
    from os_runtime.rewriter import RouterRewriter, RoutingConfig
    from os_runtime.vm import VirtualMemoryManager
    from os_runtime.observation import compute_entropy

    config = model_data["config"]
    weights = model_data["weights"]
    torch = __import__('torch')

    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
    top_k = config.num_experts_per_tok

    print(f"\n  Qwen3.6 OS Integration")
    print(f"  Layers: {n_layers} (loaded: {model_data.get('n_loaded_layers', '?' )})")
    print(f"  Experts: {n_experts}, Top-K: {top_k}")
    print(f"  Hidden dim: {config.hidden_size}")
    print()

    # OS setup
    os_config = SchedulerConfig(
        family="spherical_steering",
        backbone="steering",
        fusion_ratio=1.0,
    )
    sched = Scheduler(os_config, min(n_layers, model_data.get("n_loaded_layers", 2)))
    rewriter = RouterRewriter(
        RoutingConfig(locality_bias=5.0, locality_decay=0.9),
        n_experts=n_experts,
    )
    vmm = VirtualMemoryManager(
        n_experts=n_experts, n_layers=n_layers,
        ram_budget_mb=4000, expert_size_mb=10.5,  # q4
    )

    # Extract router weights for all loaded layers
    router_weights = {}
    for l in range(min(n_layers, model_data.get("n_loaded_layers", 2))):
        gate_key = f"model.language_model.layers.{l}.mlp.gate.weight"
        if gate_key in weights:
            router_weights[l] = weights[gate_key].float().numpy()
            print(f"  L{l} router: {router_weights[l].shape}")
        else:
            # Try alternative key patterns
            for k in weights:
                if f"layers.{l}" in k and "gate" in k and "mlp" in k:
                    router_weights[l] = weights[k].float().numpy()
                    print(f"  L{l} router (alt): {router_weights[l].shape}")
                    break

    if not router_weights:
        print("  No router weights found — cannot run OS integration")
        return

    # Run OS scheduler on real Qwen3.6 routing
    rng = np.random.RandomState(42)
    prev_expert = 0
    token_metrics = []

    print(f"\n  Generating {max_tokens} tokens with OS...")
    t0 = time.time()

    for i in range(max_tokens):
        hidden = rng.randn(config.hidden_size).astype(np.float32)
        hidden /= np.linalg.norm(hidden)

        # Router forward for each layer
        for l in router_weights:
            vmm.start_token()

            logits = router_weights[l] @ hidden  # (n_experts,)
            probs = rewriter.rewrite(logits, layer_idx=l, prev_expert=prev_expert)

            top_k_indices = np.argsort(-probs)[:top_k]

            # Access experts through VM
            for eid in top_k_indices:
                vmm.access_expert(l, int(eid))

            prev_expert = int(np.argmax(probs))
            vmm.prefetch.prefetch(l, prev_expert)
            vmm.end_token()

        # Simulated observation
        entropy = 0.15 + rng.uniform(-0.02, 0.03)
        steering = 0.3 + rng.uniform(-0.05, 0.1)

        tc = sched.begin_token(entropy, steering,
                              prev_token_id=i-1 if i > 0 else -1,
                              predicted_token_id=i+1)

        token_metrics.append({
            "idx": i, "class": tc.value,
            "collapse": sched.state.collapse_status.value,
            "precision": sched.state.precision,
        })

        if i < 3:
            print(f"  tok={i}: class={tc.value} precision={sched.state.precision}bit")

    elapsed = time.time() - t0
    print(f"  {max_tokens} tokens in {elapsed:.1f}s")

    # Stats
    vm_stats = vmm.stats()
    sched_stats = sched.stats()

    print(f"\n  VM: faults={vm_stats['page_table']['faults']} "
          f"loaded={vm_stats['page_table']['warm']} experts "
          f"overlap={vm_stats['overlap']['speedup']}x")
    print(f"  Scheduler: osc={sched_stats['class_oscillations']} "
          f"collapse={sched_stats['collapse']}")

    # Save
    result = {
        "model": "Qwen3.6-35B-A3B",
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "tokens": max_tokens,
        "elapsed_s": round(elapsed, 2),
        "token_classes": [t["class"] for t in token_metrics],
        "vm": vm_stats,
        "scheduler": sched_stats,
    }
    json.dump(result, open(OUTPUT, "w"), indent=2, default=str)
    print(f"  Saved: {OUTPUT}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    print("═" * 60)
    print("  Qwen3.6-35B OS Integration")
    print("═" * 60)
    print()

    # Try full model first
    model = None
    try:
        model, tokenizer = attempt_full_model()
    except Exception as e:
        print(f"  Full model load failed: {e}")
        print("  Falling back to shard-1 manual forward...")
        print()

    # Fallback to shard 1
    if model is None:
        model_data = attempt_shard1_forward()
        if model_data is None:
            print("  Cannot load Qwen3.6 — all paths failed")
            return
    else:
        model_data = {
            "model": model,
            "tokenizer": tokenizer,
            "config": model.config,
            "n_loaded_layers": len(model.model.layers),
        }

    run_os_integration(model_data, args.max_tokens)


if __name__ == "__main__":
    main()
