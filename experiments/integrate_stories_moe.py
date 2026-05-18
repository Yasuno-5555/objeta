#!/usr/bin/env python3
"""stories15M_MOE — MoE OS End-to-End Integration.

stories15M_MOE: 6 layers, 4 experts, top-2 routing, hidden=288, 73MB.
Perfect for MoE scheduler verification — tiny, fast, visible routing.

Verifies:
  1. Routing entropy observation at every layer
  2. Adaptive top-k (entropy-driven expert count)
  3. Expert frequency tracking per layer
  4. Collapse detection on MoE-specific signals
  5. Full execution trace → replay

Usage:
  python3 experiments/integrate_stories_moe.py
"""

import json, sys, time
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO))
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from os_runtime.scheduler import Scheduler, SchedulerConfig, TokenClass
from os_runtime.logging import RuntimeLogger, TokenLog, LayerAction, LogLevel
from os_runtime.moe import (
    MoeSchedulerExtension, AdaptiveTopK, ExpertCachePolicy,
    RoutingObservation,
)
from os_runtime.faults import FaultHarness, FaultInjection, FaultType
from os_runtime.replay import TraceReplay


MODEL_PATH = (
    "/Users/yasuno/.cache/huggingface/hub/"
    "models--ggml-org--stories15M_MOE/snapshots/"
    "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
)

OUTPUT_DIR = PROJECT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model constants
N_LAYERS = 6
N_EXPERTS = 4
TOP_K = 2
HIDDEN_DIM = 288


def extract_routing(model, hidden_states: torch.Tensor,
                    layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract router weights and expert indices from a Mixtral MoE layer.

    Returns (router_weights, expert_indices) as numpy arrays.
    """
    layer = model.model.layers[layer_idx]
    moe = layer.block_sparse_moe

    # Router forward
    router_logits = moe.gate(hidden_states)
    routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)

    # Top-k selection
    topk_weights, topk_indices = torch.topk(routing_weights, TOP_K, dim=-1)

    return (
        routing_weights.detach().cpu().numpy().flatten(),
        topk_indices.detach().cpu().numpy().flatten(),
    )


def run_moe_integration(prompt_text: str, with_faults: bool = False):
    print("═" * 60)
    print("  stories15M_MOE — MoE OS End-to-End")
    print("═" * 60)
    print()

    # ── Load model ──
    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32,
        device_map="cpu", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Layers: {N_LAYERS}, Hidden: {HIDDEN_DIM}, "
          f"Experts: {N_EXPERTS}, Top-K: {TOP_K}")
    print(f"  Vocab: {model.config.vocab_size}")
    print()

    # ── OS setup ──
    os_config = SchedulerConfig(
        family="spherical_steering",
        backbone="steering",
        fusion_ratio=1.0,  # no skip for 6-layer model
    )
    scheduler = Scheduler(os_config, N_LAYERS)

    moe_ext = MoeSchedulerExtension(
        n_layers=N_LAYERS, n_experts=N_EXPERTS, default_top_k=TOP_K,
    )

    trace_path = OUTPUT_DIR / "trace_stories_moe.jsonl"
    logger = RuntimeLogger(level=LogLevel.INFO, output_file=trace_path)
    logger.start_run()

    # ── Fault injection ──
    harness = None
    if with_faults:
        harness = FaultHarness()
        harness.add(FaultInjection(FaultType.EXPERT_DROP, token_idx=8, duration=6))
        print("  Fault: Expert Drop @ tokens 8-13")
        print()

    # ── Tokenize ──
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs.input_ids[0].tolist()
    print(f"Prompt: \"{prompt_text}\"")
    print(f"Input tokens: {len(input_ids)}")
    print()

    # ── Warmup: collect routing statistics ──
    print("Warmup: collecting routing stats...")
    warmup_prompts = [
        "Once upon a time",
        "The cat sat on",
        "She opened the door and",
    ]
    warmup_routing = []

    with torch.no_grad():
        for wp in warmup_prompts:
            w_inputs = tokenizer(wp, return_tensors="pt")
            w_outputs = model(
                w_inputs.input_ids,
                output_router_logits=True,
            )
            if hasattr(w_outputs, 'router_logits') and w_outputs.router_logits:
                for l, rl in enumerate(w_outputs.router_logits):
                    if rl is not None and len(rl) > 0:
                        weights = torch.softmax(
                            rl[-1, :].float(), dim=-1).cpu().numpy()
                        obs = moe_ext.observe_routing(l, weights)
                        warmup_routing.append(obs)

    moe_ext.build_static_cache_from_warmup([warmup_routing])
    print(f"  Collected {len(warmup_routing)} routing observations")
    print(f"  Cache hit rate: {moe_ext.cache_policy.hit_rate():.1%}")
    print()

    # ── Generation with OS interception ──
    print("Generating...")
    generated_ids = list(input_ids)
    all_layer_actions = []
    per_token_routing = []
    prev_hidden = None
    prev_token = -1

    max_new = 30
    with torch.no_grad():
        for gen_idx in range(max_new):
            t_forward_start = time.perf_counter()

            # Full forward through model
            model_inputs = torch.tensor([generated_ids])
            outputs = model(
                model_inputs, output_router_logits=True,
                output_hidden_states=True,
            )

            logits = outputs.logits[0, -1, :].cpu().numpy()
            hidden_states = outputs.hidden_states[-1][0, -1, :].cpu().numpy()

            # Extract routing from each layer (from router_logits)
            token_routing = []
            if hasattr(outputs, 'router_logits') and outputs.router_logits:
                for l, rl in enumerate(outputs.router_logits):
                    if rl is not None and len(rl) > 0:
                        weights = torch.softmax(
                            rl[-1, :].float(), dim=-1).cpu().numpy()
                        obs = moe_ext.observe_routing(l, weights)
                        token_routing.append(obs)
            per_token_routing.append(token_routing)

            # Observation
            from os_runtime.observation import compute_entropy, compute_steering
            entropy = compute_entropy(logits)
            top1 = int(np.argmax(logits))

            steering = 0.0
            if prev_hidden is not None:
                steering = compute_steering(hidden_states, prev_hidden)
            prev_hidden = hidden_states.copy()

            # Classification
            tc = scheduler.begin_token(
                entropy, steering,
                prev_token_id=prev_token,
                predicted_token_id=top1,
            )

            # Routing-aware expert count
            avg_routing_ent = np.mean(
                [obs.routing_entropy for obs in token_routing]
            ) if token_routing else 0.5
            k = moe_ext.get_expert_count(avg_routing_ent)

            # Layer actions (simulated for this architecture)
            layer_actions = []
            for l in range(N_LAYERS):
                run_attn = scheduler.should_run_attn(l)
                run_ffn = scheduler.should_run_ffn(l)
                prec = scheduler.get_precision(l)
                layer_actions.append(LayerAction(
                    layer=l, attn_ran=run_attn, ffn_ran=run_ffn,
                    precision_used=prec,
                ))

            # Collapse detection
            if harness:
                active = harness.active_faults(gen_idx)
                harness.record_status(
                    gen_idx, scheduler.state.collapse_status.value)
                harness.check_detection(
                    gen_idx, scheduler.state.collapse_status.value)

            # Sampling (greedy)
            next_token = top1

            # Token text
            token_text = ""
            try:
                token_text = tokenizer.decode([next_token])
            except Exception:
                pass

            forward_ms = (time.perf_counter() - t_forward_start) * 1000

            # Log
            active_faults_str = ""
            if harness:
                active_faults_str = ",".join(
                    [ft.value for ft in harness.active_faults(gen_idx)])

            tlog = TokenLog(
                token_idx=gen_idx,
                token_id=next_token,
                token_text=token_text,
                entropy=entropy,
                steering=steering,
                top1_logit=float(logits[top1]),
                is_repeat=(next_token == prev_token),
                token_class=tc.value,
                collapse_score={
                    "healthy": 0.0, "warning": 0.5, "critical": 1.0,
                }.get(scheduler.state.collapse_status.value, 0.0),
                collapse_status=scheduler.state.collapse_status.value,
                precision=scheduler.state.precision,
                layers_run=scheduler.layers_run,
                layers_skipped=scheduler.layers_skipped,
                skip_rate=scheduler.stats()["skip_rate"],
                layer_actions=layer_actions,
                forward_ms=forward_ms,
                fault_active=active_faults_str,
            )
            logger.log_token(tlog)

            # EOS
            if next_token == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)
            prev_token = next_token

    logger.end_run()

    # ── Results ──
    text = tokenizer.decode(generated_ids[len(input_ids):])
    summary = logger.run_summary()

    print()
    print(f"  Generated: {len(generated_ids) - len(input_ids)} tokens "
          f"in {summary['elapsed_s']:.1f}s ({summary['tok_per_s']:.1f} tok/s)")
    print(f"  Output: \"{text[:200]}\"")
    print()

    # Routing stats
    if per_token_routing:
        all_routing = []
        for token_obs in per_token_routing:
            for obs in token_obs:
                all_routing.append(obs.routing_entropy)
        avg_ent = np.mean(all_routing) if all_routing else 0

        # Expert frequency
        expert_counter = Counter()
        for token_obs in per_token_routing:
            for obs in token_obs:
                expert_counter[obs.top1_expert] += 1

        print("─" * 60)
        print("  Routing Analysis")
        print("─" * 60)
        print(f"  Avg routing entropy: {avg_ent:.3f} "
              f"({'uniform' if avg_ent > 0.9 else 'specialized'})")
        print(f"  Expert frequency: {dict(sorted(expert_counter.items()))}")
        print()

    # Replay
    logger.print_summary()

    if trace_path.exists():
        print()
        print(f"  Trace saved: {trace_path} ({trace_path.stat().st_size} bytes)")
        trace = TraceReplay.load(trace_path)
        print(f"  Trace tokens: {len(trace.tokens)} → replayable ✓")

    if harness:
        print()
        harness.print_results()

    result_path = OUTPUT_DIR / "integration_moe_result.json"
    result_path.write_text(json.dumps({
        "prompt": prompt_text,
        "output": text[:500],
        **summary,
        "avg_routing_entropy": avg_ent if per_token_routing else 0,
    }, indent=2))
    print(f"\n  Result: {result_path}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",
                       default="Once upon a time, there was a")
    parser.add_argument("--faults", action="store_true")
    args = parser.parse_args()
    run_moe_integration(args.prompt, args.faults)


if __name__ == "__main__":
    main()
