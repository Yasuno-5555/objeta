#!/usr/bin/env python3
"""Qwen3.6-35B-A3B text generation — mmap everything, fit in 8GB.

Based on transformers Qwen3_5MoeGatedDeltaNet reference implementation.
Uses Gated DeltaNet (NOT Mamba2) for linear attention layers.
Full GQA attention for the 10 full-attn layers (every 4th).
Rust parallel dispatch for MoE.
"""

import ctypes, json, math, os, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx

# Use objeta's executor wrapper
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
_lib = get_lib()

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"

# ── Rust C API ──
_lib.lko_moe_forward_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
_lib.lko_moe_forward_layer.restype = ctypes.c_int32


def rust_moe(x_np, router, gu_mmap, d_mmap, layer_idx):
    out = np.zeros(2048, dtype=np.float32)
    eidx = np.zeros(8, dtype=np.int32)
    ew = np.zeros(8, dtype=np.float32)
    _lib.lko_moe_forward_layer(
        router.ctypes.data_as(ctypes.c_void_p),
        gu_mmap.ctypes.data_as(ctypes.c_void_p), gu_mmap.nbytes,
        d_mmap.ctypes.data_as(ctypes.c_void_p), d_mmap.nbytes,
        x_np.ctypes.data_as(ctypes.c_void_p), 8, layer_idx,
        eidx.ctypes.data_as(ctypes.c_void_p),
        ew.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    return out


class MmapAttnWeights:
    """Memory-mapped fp16 attention weights for one layer."""

    def __init__(self, layer_idx: int):
        with open(BIN / f"layer_{layer_idx}_attn_f16.json") as f:
            self.meta = json.load(f)
        self._mmap = np.memmap(
            BIN / f"layer_{layer_idx}_attn_f16.bin",
            dtype=np.float16, mode='r')

    def get(self, name: str) -> np.ndarray:
        shape, offset, nbytes = self.meta[name]
        nelem = nbytes // 2
        return self._mmap[offset // 2 : offset // 2 + nelem].reshape(shape)

    def has(self, name: str) -> bool:
        return name in self.meta


class Qwen36Generator:
    def __init__(self):
        # Embedding: mmap fp32
        self.embed = np.memmap(BIN / "embed_tokens.bin", dtype=np.float32, mode='r')
        self.embed = self.embed.reshape(248320, 2048)
        self.final_norm_w = np.fromfile(BIN / "final_norm.bin", dtype=np.float32)
        # Pre-convert to MLX to avoid per-token numpy→MLX conversion
        self.final_norm_w_mx = mx.array(self.final_norm_w.astype(np.float32).flatten())

        print("Opening mmap attention weights...")
        self.attn = [MmapAttnWeights(l) for l in range(40)]
        n_linear = sum(1 for a in self.attn if a.has('linear_attn.in_proj_qkv.weight'))
        print(f"  Linear (GatedDeltaNet): {n_linear}, Full GQA: {40 - n_linear}")

        # Cache attention weights as numpy fp32 for Accelerate matmul
        print("Caching attention weights (numpy fp32)...", end=" ", flush=True)
        self._w = {}  # layer → {weight_name: numpy fp32 array}
        for l in range(40):
            a = self.attn[l]
            layer_cache = {}
            for key in a.meta:
                if key == '__metadata__':
                    continue
                layer_cache[key] = a.get(key)
            self._w[l] = layer_cache
        print("done")

        # GQA params
        self.n_heads, self.n_kv, self.hd = 16, 2, 256
        self.full_layers = {l for l in range(40) if l % 4 == 3}

        # DeltaNet params (from config)
        self.n_k_heads = 16
        self.n_v_heads = 32
        self.head_k_dim = 128
        self.head_v_dim = 128

        # MoE: per-layer lazy mmap
        self._gu = {}
        self._down = {}
        self._routers = {}

        # KV caches (full attention) and DeltaNet states
        self.kv: dict[int, tuple] = {}
        self.delta_states: dict[int, dict] = {}

    def _ensure_moe(self, l: int):
        if l not in self._gu:
            self._gu[l] = np.memmap(BIN / f"layer_{l}_gate_up.bin", dtype=np.uint8, mode='r')
            self._down[l] = np.memmap(BIN / f"layer_{l}_down.bin", dtype=np.uint8, mode='r')
            self._routers[l] = np.fromfile(
                BIN / f"layer_{l}_router.bin", dtype=np.float32).reshape(256, 2048)

    def _ensure_kv(self, l: int, max_seq: int):
        if l not in self.kv:
            self.kv[l] = (
                mx.zeros((self.n_kv, max_seq, self.hd), dtype=mx.float16),
                mx.zeros((self.n_kv, max_seq, self.hd), dtype=mx.float16),
            )

    def _ensure_delta_state(self, l: int):
        if l not in self.delta_states:
            self.delta_states[l] = {
                'conv_state': np.zeros((8192, 4), dtype=np.float32),
                'conv_ptr': 0,
                'S': mx.zeros((self.n_v_heads, self.head_k_dim, self.head_v_dim), dtype=mx.float32),
            }

    # ── RMS Norm ──
    def _rms_norm(self, x: mx.array, w: np.ndarray, eps: float = 1e-6) -> mx.array:
        wf = mx.array(w.astype(np.float32).flatten())
        xf = x.flatten()
        return (xf / mx.sqrt(mx.mean(xf ** 2) + eps)) * wf

    def _l2norm(self, x: mx.array, eps: float = 1e-6) -> mx.array:
        inv_norm = mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)
        return x * inv_norm

    def _softplus(self, x: mx.array) -> mx.array:
        return mx.log(1.0 + mx.exp(x))

    # ── Full GQA Attention ──
    def _full_attn(self, h_np: np.ndarray, l: int, pos: int, seq_len: int) -> np.ndarray:
        w = self._w[l]
        q_w = w['self_attn.q_proj.weight']
        k_w = w['self_attn.k_proj.weight']
        v_w = w['self_attn.v_proj.weight']
        o_w = w['self_attn.o_proj.weight']

        q_full = q_w @ h_np
        q = q_full[:self.n_heads * self.hd].reshape(self.n_heads, self.hd)
        q_gate = 1.0 / (1.0 + np.exp(-q_full[self.n_heads * self.hd:]))
        k = (k_w @ h_np).reshape(self.n_kv, self.hd)
        v = (v_w @ h_np).reshape(self.n_kv, self.hd)

        # RoPE (numpy)
        d2 = self.hd // 2
        rope_cos, rope_sin = self._rope
        c = rope_cos[pos, :d2][None, :]
        s = rope_sin[pos, :d2][None, :]
        q_e, q_o = q[:, :d2], q[:, d2:]
        k_e, k_o = k[:, :d2], k[:, d2:]
        q = np.concatenate([q_e * c - q_o * s, q_e * s + q_o * c], axis=-1)
        k = np.concatenate([k_e * c - k_o * s, k_e * s + k_o * c], axis=-1)

        Kc, Vc = self.kv[l]
        Kc[:, pos, :] = k
        Vc[:, pos, :] = v

        n_rep = self.n_heads // self.n_kv
        k_rep = np.repeat(Kc[:, :seq_len, :], n_rep, axis=0)
        v_rep = np.repeat(Vc[:, :seq_len, :], n_rep, axis=0)
        scale = 1.0 / math.sqrt(self.hd)
        scores = np.sum(q[:, None, :] * k_rep, axis=-1) * scale
        scores -= np.max(scores, axis=-1, keepdims=True)
        attn_w = np.exp(scores)
        attn_w /= np.sum(attn_w, axis=-1, keepdims=True)
        attn_out = np.sum(attn_w[:, :, None] * v_rep, axis=1).flatten()
        return o_w @ (attn_out * q_gate)

    # ── Gated DeltaNet ──
    def _delta_net(self, x: mx.array, a: MmapAttnWeights, l: int) -> mx.array:
        """Qwen3.6 GatedDeltaNet — reference: Qwen3_5MoeGatedDeltaNet in transformers."""
        h = x.flatten().astype(mx.float32)
        mx_w = self._mx[l]

        # Projections
        w_qkv = mx_w['linear_attn.in_proj_qkv.weight']
        w_z = mx_w['linear_attn.in_proj_z.weight']
        w_b = mx_w['linear_attn.in_proj_b.weight']
        w_a = mx_w['linear_attn.in_proj_a.weight']
        w_out = mx_w['linear_attn.out_proj.weight']
        w_conv = mx_w['linear_attn.conv1d.weight']  # (8192, 1, 4)
        w_conv = w_conv.reshape(8192, 4)  # squeeze the groups dim, as in reference
        w_norm = mx_w['linear_attn.norm.weight']  # (128,)
        dt_bias = mx_w['linear_attn.dt_bias']  # (32,)
        A_log = mx_w['linear_attn.A_log']  # (32,)

        mixed_qkv = w_qkv @ h  # (8192,)
        z = w_z @ h  # (4096,)
        b = w_b @ h  # (32,)
        a_vec = w_a @ h  # (32,)

        # Causal conv1d (kernel_size=4, groups=8192) — manual ring buffer
        state = self.delta_states[l]
        conv_state = state['conv_state']  # np (8192, 4)
        ptr = state['conv_ptr']

        qkv_np = np.array(mixed_qkv)
        conv_state[:, ptr] = qkv_np
        new_ptr = (ptr + 1) % 4
        state['conv_ptr'] = new_ptr

        # Apply conv filter: for each channel j, out[j] = sum_{t=0}^{3} w[j,t] * conv_state[j, (ptr-3+t)]
        # Reorder: indices from oldest to newest
        order = [(new_ptr - i + 4) % 4 for i in range(4)]  # PyTorch Conv1d: k=0 newest
        conv_hist = mx.array(conv_state[:, order])  # (8192, 4)
        qkv_conv = mx.sum(w_conv * conv_hist, axis=1)  # (8192,)
        qkv_act = qkv_conv * mx.sigmoid(qkv_conv)  # SiLU

        # Split: q (2048), k (2048), v (4096)
        q = qkv_act[:2048].reshape(self.n_k_heads, self.head_k_dim)  # (16, 128)
        k = qkv_act[2048:4096].reshape(self.n_k_heads, self.head_k_dim)  # (16, 128)
        v = qkv_act[4096:].reshape(self.n_v_heads, self.head_v_dim)  # (32, 128)

        z = z.reshape(self.n_v_heads, self.head_v_dim)  # (32, 128)

        # Repeat q,k to match v heads
        rep = self.n_v_heads // self.n_k_heads  # 2
        q = mx.repeat(q, rep, axis=0)  # (32, 128)
        k = mx.repeat(k, rep, axis=0)  # (32, 128)

        # Beta and g
        beta = mx.sigmoid(b)  # (32,)
        g = -mx.exp(A_log) * self._softplus(a_vec + dt_bias)  # (32,)

        # L2 normalize q and k
        q = self._l2norm(q)
        k = self._l2norm(k)

        # Scale q
        scale = 1.0 / math.sqrt(self.head_k_dim)
        q = q * scale

        # Delta rule recurrence (single step)
        S = state['S']  # (32, 128, 128)

        g_t = mx.exp(g).reshape(self.n_v_heads, 1, 1)  # (32, 1, 1)
        S = S * g_t  # decay

        # kv_mem = sum_j S[:, j, :] * k[:, j] = S^T @ k
        # S: (32, 128, 128), k: (32, 128)
        # S^T: (32, 128, 128), k.unsqueeze(-1): (32, 128, 1)
        # S^T @ k = sum over dim=1 of S * k: (32, 128)
        kv_mem = mx.sum(S * k[:, :, None], axis=1)  # (32, 128)

        delta = (v - kv_mem) * beta.reshape(self.n_v_heads, 1)  # (32, 128)

        # S += outer(k, delta) = k.unsqueeze(-1) * delta.unsqueeze(1)
        S = S + k[:, :, None] * delta[:, None, :]  # (32, 128, 128)

        # output_h = sum_k S[h, k, :] * q[h, k]  (contract over k_dim)
        output = mx.sum(S * q[..., None], axis=1)  # (32, 128)

        state['S'] = S

        # RMSNormGated per head (reference: Qwen3_5MoeRMSNormGated)
        # 1. RMS norm per head, 2. weight scale, 3. gate with SiLU(z)
        on = output  # (32, 128) — already reshaped as (n_v_heads, head_v_dim)
        rms = mx.sqrt(mx.mean(on ** 2, axis=-1, keepdims=True) + 1e-6)
        on_normed = (on / rms) * w_norm.reshape(1, self.head_v_dim)
        gated = on_normed * z * mx.sigmoid(z)  # SiLU gate: z * sigmoid(z)
        gated_flat = gated.reshape(-1)  # (4096,)

        return w_out @ gated_flat

    # ── Layer Forward ──
    def _forward_layer(self, h: mx.array, l: int, pos: int, seq_len: int):
        a = self.attn[l]

        # Input norm
        if a.has('input_layernorm.weight'):
            hn = self._rms_norm(h, a.get('input_layernorm.weight'))
        else:
            hn = h

        # Attention
        if l in self.full_layers:
            self._ensure_kv(l, seq_len)
            ao = self._full_attn(hn, a, l, pos, seq_len)
        elif a.has('linear_attn.in_proj_qkv.weight'):
            self._ensure_delta_state(l)
            ao = self._delta_net(hn, a, l)
        else:
            ao = mx.zeros(2048)

        h = (h.flatten() + ao).reshape(h.shape)

        # Post-attention norm
        if a.has('post_attention_layernorm.weight'):
            hn2 = self._rms_norm(h, a.get('post_attention_layernorm.weight'))
        else:
            hn2 = h

        # Shared Expert (sigmoid-gated, ffn_dim=512)
        shared_out = mx.zeros(2048)
        if a.has('mlp.shared_expert.gate_proj.weight'):
            mx_w = self._mx[l]
            gate_w_se = mx_w['mlp.shared_expert.gate_proj.weight']
            up_w_se = mx_w['mlp.shared_expert.up_proj.weight']
            down_w_se = mx_w['mlp.shared_expert.down_proj.weight']
            gate_gate_w = mx_w['mlp.shared_expert_gate.weight'].flatten()
            # gate_out = sigmoid(gate_gate @ x) * (down @ (SiLU(gate @ x) * (up @ x)))
            hn2_flat = hn2.flatten()
            gate_h = gate_w_se @ hn2_flat
            up_h = up_w_se @ hn2_flat
            hidden_se = gate_h / (1.0 + mx.exp(-gate_h)) * up_h  # SiLU(gate) * up
            se_raw = down_w_se @ hidden_se
            se_gate = mx.sigmoid(gate_gate_w @ hn2_flat)
            shared_out = se_raw * se_gate

        # MoE
        self._ensure_moe(l)
        moe_out = rust_moe(np.array(hn2), self._routers[l], self._gu[l], self._down[l], l)
        h = (h.flatten() + shared_out + mx.array(moe_out)).reshape(h.shape)
        return h

    # ── Full Forward ──
    def forward(self, token_id: int, position: int, seq_len: int) -> mx.array:
        emb = self.embed[token_id].astype(np.float32)
        h = mx.array(np.array(emb))
        for l in range(40):
            h = self._forward_layer(h, l, position, seq_len)
        hf = h.flatten()
        nw = mx.array(self.final_norm_w.astype(np.float32).flatten())
        rms = mx.sqrt(mx.mean(hf ** 2) + 1e-6)
        hn = (hf / rms) * nw
        lm_head = mx.array(self.embed[:].astype(np.float32))
        return lm_head @ hn

    # ── Generate ──
    def generate(self, prompt_ids: list[int], max_tokens: int = 30,
                 temperature: float = 0.7, top_k: int = 40):
        tokens = list(prompt_ids)
        n_prompt = len(tokens)

        print(f"Prefilling {n_prompt} tokens...")
        t0 = time.perf_counter()
        for i, tid in enumerate(tokens):
            logits = self.forward(tid, position=i, seq_len=i + 1)
            if i % 5 == 0 or i == n_prompt - 1:
                print(f"  [{i+1}/{n_prompt}] {time.perf_counter()-t0:.0f}s", flush=True)
        print(f"  Prefill done in {time.perf_counter()-t0:.1f}s")

        # First token
        if temperature == 0:
            next_token = int(mx.argmax(logits))
        else:
            probs = mx.softmax(logits * (1.0 / max(temperature, 0.01)))
            if top_k > 0:
                tk = mx.argsort(probs)[-top_k:]
                mask = mx.zeros_like(probs)
                mask[tk] = probs[tk]
                probs = mask / mx.sum(mask)
            next_token = int(mx.random.categorical(probs))

        generated = []
        t_start = time.perf_counter()

        for step in range(max_tokens):
            generated.append(next_token)
            pos = n_prompt + step
            logits = self.forward(next_token, position=pos, seq_len=pos + 1)

            if temperature == 0:
                next_token = int(mx.argmax(logits))
            else:
                probs = mx.softmax(logits * (1.0 / max(temperature, 0.01)))
                if top_k > 0:
                    tk = mx.argsort(probs)[-top_k:]
                    mask = mx.zeros_like(probs)
                    mask[tk] = probs[tk]
                    probs = mask / mx.sum(mask)
                next_token = int(mx.random.categorical(probs))

            if next_token in (0, 2, 248044):
                break

            if step % 3 == 0 or step == max_tokens - 1:
                elapsed = time.perf_counter() - t_start
                tok_s = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{step+1}/{max_tokens}] {tok_s:.2f} tok/s", flush=True)

        total_s = time.perf_counter() - t_start
        n_gen = len(generated)
        print(f"\n  {n_gen} tokens in {total_s:.1f}s ({n_gen/total_s:.2f} tok/s)")
        return generated


def main():
    print("=== Qwen3.6-35B-A3B Text Generation ===\n")

    gen = Qwen36Generator()

    from transformers import AutoTokenizer
    snap = sorted(os.listdir(
        "/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
    tok = AutoTokenizer.from_pretrained(
        f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}")
    print(f"  Vocab: {tok.vocab_size}\n")

    prompts = [
        "The meaning of life is",
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ]

    for prompt in prompts:
        # Apply Qwen3.6 chat template
        messages = [{"role": "user", "content": prompt}]
        chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(chat_text)
        print(f"\n── Prompt: \"{prompt}\" ──")
        print(f"   Chat template: {chat_text[:120]}...")
        gen_tokens = gen.generate(ids, max_tokens=20, temperature=0.7, top_k=40)
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        print(f"  Output: {text}")


if __name__ == "__main__":
    main()
