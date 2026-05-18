"""OSLLM — Scheduler-driven LLM wrapper.

The OS-augmented LLM wraps a base LLM and interposes the scheduler
at every layer forward pass. The scheduler decides:
  - Whether to run attention (Full/Reduced/Cached/Skip)
  - Whether to run FFN (Full/Adaptive/Sparse/Skip)
  - What precision to use (fp16/q8/q5/q4/q3)

Supports:
  - Enhanced structured logging (research data)
  - Replay mode (recorded trace playback)
  - Fault injection (collapse detector testing)
"""

import time
from typing import Any

import mlx.core as mx
import numpy as np

from .scheduler import (
    Scheduler, SchedulerConfig, TokenClass, CollapseStatus, Phase,
)
from .logging import RuntimeLogger, TokenLog, LayerAction, LogLevel
from .faults import FaultHarness, FaultType


class OSLLM:
    """OS-augmented LLM with trajectory-aware scheduling."""

    def __init__(self, base_llm: Any,
                 config: SchedulerConfig | None = None,
                 logger: RuntimeLogger | None = None,
                 fault_harness: FaultHarness | None = None):
        self.llm = base_llm
        self.cfg = base_llm.config
        self.os_config = config or SchedulerConfig()
        self.n_layers = self.cfg.n_layers
        self.scheduler = Scheduler(self.os_config, self.n_layers)
        self.logger = logger or RuntimeLogger()
        self.faults = fault_harness

        # Timing
        self._timing: dict[str, float] = {
            "attn_ms": 0.0, "ffn_ms": 0.0, "norm_ms": 0.0,
        }
        # Per-token layer action accumulator
        self._layer_actions: list[LayerAction] = []

    # ── Layer dispatch ──

    def forward_layer(self, layer_idx: int, x: mx.array,
                      position: int, seq_len: int,
                      token_class: TokenClass,
                      collapse_status: CollapseStatus) -> mx.array:
        """Scheduler-driven single-layer forward."""
        layer = self.llm.layers[layer_idx]
        policy = self.scheduler.policy_table[layer_idx]
        precision = self.scheduler.get_precision(layer_idx)

        # Apply fault injection
        attn_skip_override = False
        precision_override = None
        if self.faults:
            for ft in self.faults.active_faults(self.scheduler.token_count - 1):
                prec_mod, skip_mod = self.faults.apply_fault(
                    ft, precision, False)
                precision_override = prec_mod
                attn_skip_override = skip_mod
                self.faults.start_fault(ft, self.scheduler.token_count - 1)

        if precision_override is not None:
            precision = precision_override

        # Input norm
        t0 = time.perf_counter()
        x_norm = layer.forward_norm(x, layer.input_norm_weight)
        self._timing["norm_ms"] += (time.perf_counter() - t0) * 1000

        # Attention
        run_attn = self.scheduler.should_run_attn(layer_idx) and not attn_skip_override
        if run_attn:
            t0 = time.perf_counter()
            attn_out = layer.forward_attention(
                x_norm,
                self.llm.K_caches[layer_idx],
                self.llm.V_caches[layer_idx],
                position, seq_len,
            )
            self._timing["attn_ms"] += (time.perf_counter() - t0) * 1000
            x = x + attn_out

        # Post-attention norm + FFN
        t0 = time.perf_counter()
        post_normed = layer.forward_norm(x, layer.post_attn_norm_weight)
        self._timing["norm_ms"] += (time.perf_counter() - t0) * 1000

        run_ffn = self.scheduler.should_run_ffn(layer_idx)
        if run_ffn:
            t0 = time.perf_counter()
            ffn_out = layer.forward_ffn(post_normed)
            self._timing["ffn_ms"] += (time.perf_counter() - t0) * 1000
            x = x + ffn_out

        # Record layer action
        self._layer_actions.append(LayerAction(
            layer=layer_idx,
            attn_ran=run_attn,
            ffn_ran=run_ffn,
            precision_used=precision,
            phase=policy.phase.value,
        ))

        mx.eval(x)
        return x

    # ── Generation ──

    def generate(self, prompt: str | list[int], max_tokens: int = 512,
                 temperature: float = 0.7, top_k: int = 40,
                 tokenizer: Any = None) -> list[int]:
        """Generate tokens with full OS scheduling.

        Returns list of generated token IDs.
        """
        self.logger.start_run()

        # Reset state
        self.llm._reset_caches()
        self.scheduler.reset()
        for k in self._timing:
            if isinstance(self._timing[k], (int, float)):
                self._timing[k] = 0

        # Tokenize
        if isinstance(prompt, str):
            if tokenizer is None:
                raise ValueError("Tokenizer required for string prompt")
            input_ids = tokenizer.encode(prompt)
        else:
            input_ids = list(prompt)

        tokens = list(input_ids)
        prev_hidden = None
        prev_token_id = -1
        generated = []

        # Prefill — full compute, no skip
        for i, tid in enumerate(tokens):
            x_t = self.llm.embed_tokens(mx.array([tid])).reshape(-1)
            for li in range(self.n_layers):
                x_t = self._full_forward_layer(li, x_t, i, i + 1)
            prev_hidden = np.array(x_t).flatten()
            if i == len(tokens) - 1:
                prev_token_id = tid

        # Generation loop
        for gen_idx in range(max_tokens):
            self._layer_actions.clear()
            t_forward_start = time.perf_counter()

            seq_len = len(tokens)
            pos = seq_len - 1

            # Forward with scheduler
            x_t = self.llm.embed_tokens(
                mx.array([tokens[-1]])).reshape(-1)

            # Classification happens BEFORE forward for this token
            # so we know what class we're in
            tc = self.scheduler.state.token_class
            cs = self.scheduler.state.collapse_status

            for li in range(self.n_layers):
                x_t = self.forward_layer(li, x_t, pos, seq_len, tc, cs)

            forward_ms = (time.perf_counter() - t_forward_start) * 1000

            # Final norm + lm_head
            t_sample_start = time.perf_counter()
            nwf = self.llm.norm_weight.flatten()
            h_normed = self.llm.final_norm(
                x_t.flatten(), nwf).reshape(x_t.shape)
            logits = self.llm.lm_head @ h_normed
            logits_np = np.array(logits).flatten()

            # Observation
            from .observation import compute_entropy, compute_steering
            entropy = compute_entropy(logits_np)
            top1 = int(np.argmax(logits_np))

            steering = 0.0
            if prev_hidden is not None:
                h_curr = np.array(x_t).flatten()
                # Fault: hidden noise
                if self.faults and any(
                    ft == FaultType.HIDDEN_NOISE
                    for ft in self.faults.active_faults(gen_idx)
                ):
                    h_curr = self.faults.apply_hidden_noise(
                        h_curr, intensity=0.1)
                steering = compute_steering(h_curr, prev_hidden)
                prev_hidden = h_curr

            # Classify for NEXT token
            tc_next = self.scheduler.begin_token(
                entropy, steering,
                prev_token_id=prev_token_id,
                predicted_token_id=top1,
            )

            # Collapse score (0=healthy, 1=critical)
            collapse_score = {
                CollapseStatus.HEALTHY: 0.0,
                CollapseStatus.WARNING: 0.5,
                CollapseStatus.CRITICAL: 1.0,
            }.get(self.scheduler.state.collapse_status, 0.0)

            # Check fault detection
            if self.faults:
                self.faults.record_status(
                    gen_idx,
                    self.scheduler.state.collapse_status.value,
                )
                self.faults.check_detection(
                    gen_idx,
                    self.scheduler.state.collapse_status.value,
                )

            # Sampling
            if temperature == 0:
                next_token = top1
            else:
                scaled = logits_np / max(temperature, 0.01)
                scaled -= scaled.max()
                probs = np.exp(scaled.astype(np.float64))
                probs /= probs.sum()
                if top_k > 0 and top_k < len(probs):
                    idx = np.argpartition(-probs, top_k)[:top_k]
                    p = probs[idx]
                    p /= p.sum()
                    next_token = int(idx[np.random.choice(len(idx), p=p)])
                else:
                    next_token = int(np.random.choice(len(probs), p=probs))

            sample_ms = (time.perf_counter() - t_sample_start) * 1000

            # Token text for logging
            token_text = ""
            if tokenizer and next_token != tokenizer.eos_token_id:
                try:
                    token_text = tokenizer.decode([next_token])
                except Exception:
                    pass

            # Active faults
            active_faults = []
            if self.faults:
                active_faults = [
                    ft.value for ft in self.faults.active_faults(gen_idx)
                ]

            # Log
            tlog = TokenLog(
                token_idx=gen_idx,
                token_id=next_token,
                token_text=token_text,
                entropy=entropy,
                steering=steering,
                top1_logit=float(logits_np[top1]),
                is_repeat=(next_token == prev_token_id),
                token_class=tc_next.value,
                collapse_score=collapse_score,
                collapse_status=self.scheduler.state.collapse_status.value,
                precision=self.scheduler.state.precision,
                layers_run=self.scheduler.layers_run,
                layers_skipped=self.scheduler.layers_skipped,
                skip_rate=self.scheduler.stats()["skip_rate"],
                layer_actions=list(self._layer_actions),
                forward_ms=forward_ms,
                sample_ms=sample_ms,
                fault_active=",".join(active_faults) if active_faults else "",
                fault_type=active_faults[0] if active_faults else "",
            )
            self.logger.log_token(tlog)

            # EOS check
            if next_token == 2 or (tokenizer and next_token == tokenizer.eos_token_id):
                break

            tokens.append(next_token)
            prev_token_id = next_token
            generated.append(next_token)

        self.logger.end_run()
        return generated

    def _full_forward_layer(self, layer_idx: int, x: mx.array,
                            position: int, seq_len: int) -> mx.array:
        """Full forward through one layer (for prefill)."""
        layer = self.llm.layers[layer_idx]
        x_norm = layer.forward_norm(x, layer.input_norm_weight)
        attn_out = layer.forward_attention(
            x_norm,
            self.llm.K_caches[layer_idx],
            self.llm.V_caches[layer_idx],
            position, seq_len,
        )
        x = x + attn_out
        post_normed = layer.forward_norm(x, layer.post_attn_norm_weight)
        ffn_out = layer.forward_ffn(post_normed)
        x = x + ffn_out
        mx.eval(x)
        return x

    # ── Reporting ──

    def report(self) -> str:
        s = self.scheduler.stats()
        return (
            f"[OS] class={s['token_class']} "
            f"collapse={s['collapse']} "
            f"ent={s['entropy']:.2f} steer={s['steering']:.2f} "
            f"skip={s['skip_rate']*100:.0f}% "
            f"run={s['layers_run']} prec={s['precision']}"
        )

    def summary(self) -> dict:
        """Return full run summary for benchmarking."""
        return self.logger.run_summary()
