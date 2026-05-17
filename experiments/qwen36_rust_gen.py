#!/usr/bin/env python3
"""Qwen3.6-35B-A3B — Rust NEON GEMV + numpy ops. Zero MLX.

Key: all matmuls via Rust NEON (8 GFLOPS), complex logic in numpy.
"""

import ctypes, json, math, os, sys, struct, mmap, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
_lib = get_lib()

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HDIM, N_HEADS, N_KV, HEAD_DIM = 2048, 16, 2, 256
N_K_HEADS, N_V_HEADS, HK, HV = 16, 32, 128, 128

# ── Rust FFI setup ────────────────────────────────────────────────────────

# MoE dispatch
_lib.lko_moe_forward_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
_lib.lko_moe_forward_layer.restype = ctypes.c_int32

# Rust NEON GEMV
_lib.lko_q36_f32_gemv.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p,
]
_lib.lko_q36_f32_gemv.restype = ctypes.c_int32

_lib.lko_q36_rms_norm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
_lib.lko_q36_rms_norm.restype = ctypes.c_int32

def rust_gemv(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """W @ x via Rust NEON GEMV. W is (M,K) f32, x is (K,) f32."""
    M, K = W.shape
    y = np.zeros(M, dtype=np.float32)
    _lib.lko_q36_f32_gemv(
        W.ctypes.data_as(ctypes.c_void_p), M, K,
        x.ctypes.data_as(ctypes.c_void_p),
        y.ctypes.data_as(ctypes.c_void_p),
    )
    return y

def rust_moe(x_np, router, gu_mmap, d_mmap, layer_idx):
    out = np.zeros(HDIM, dtype=np.float32)
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

# ── Weight loading ────────────────────────────────────────────────────────

class AttnWeightLoader:
    def __init__(self, layer_idx):
        with open(BIN / f"layer_{layer_idx}_attn_f16.json") as f:
            self.meta = json.load(f)
        self._mmap = np.memmap(BIN / f"layer_{layer_idx}_attn_f16.bin", dtype=np.float16, mode='r')
    def get(self, name):
        shape, offset, nbytes = self.meta[name]
        nelem = nbytes // 2
        return self._mmap[offset//2 : offset//2 + nelem].reshape(shape).astype(np.float32)
    def has(self, name): return name in self.meta

# ── RoPE ──────────────────────────────────────────────────────────────────

def precompute_rope(max_seq, head_dim):
    theta = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = np.arange(max_seq)[:, None] * theta[None, :]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

def apply_rope(x, cos, sin, pos):
    d2 = x.shape[-1] // 2
    c, s = cos[pos, :d2][None, :], sin[pos, :d2][None, :]
    return np.concatenate([x[:,:d2]*c - x[:,d2:]*s, x[:,:d2]*s + x[:,d2:]*c], axis=-1)

def rms_norm(x, w, eps=1e-6):
    return (x / np.sqrt(np.mean(x**2) + eps)) * w

# ── Generator ─────────────────────────────────────────────────────────────

class Qwen36RustGen:
    def __init__(self):
        self.embed = np.memmap(BIN / "embed_tokens.bin", dtype=np.float32, mode='r').reshape(248320, HDIM)
        self.final_norm_w = np.fromfile(BIN / "final_norm.bin", dtype=np.float32)
        self._rope = precompute_rope(256, HEAD_DIM)

        print("Loading + concatenating weights (f32)...", end=" ", flush=True)
        self._w = {}
        for l in range(40):
            a = AttnWeightLoader(l)
            w = {}
            for key in a.meta:
                if key != '__metadata__':
                    w[key] = a.get(key)  # already fp32

            if l % 4 == 3:
                q_w = w['self_attn.q_proj.weight']
                k_w = w['self_attn.k_proj.weight']
                v_w = w['self_attn.v_proj.weight']
                w['_qkv_cat'] = np.concatenate([q_w, k_w, v_w], axis=0)
                w['_q_sz'] = q_w.shape[0]
                w['_k_sz'] = k_w.shape[0]

            if 'linear_attn.conv1d.weight' in w:
                w['linear_attn.conv1d.weight'] = w['linear_attn.conv1d.weight'].reshape(8192, 4)

            self._w[l] = w
        print("done")

        self._gu, self._down, self._routers = {}, {}, {}

    def _ensure_moe(self, l):
        if l not in self._gu:
            self._gu[l] = np.memmap(BIN / f"layer_{l}_gate_up.bin", dtype=np.uint8, mode='r')
            self._down[l] = np.memmap(BIN / f"layer_{l}_down.bin", dtype=np.uint8, mode='r')
            self._routers[l] = np.fromfile(BIN / f"layer_{l}_router.bin", dtype=np.float32).reshape(256, HDIM)

    def _full_attn(self, h, l, pos, seq_len, kv):
        w = self._w[l]
        W_qkv, q_sz, k_sz = w['_qkv_cat'], w['_q_sz'], w['_k_sz']
        n_h, n_kv, hd = N_HEADS, N_KV, HEAD_DIM

        qkv = rust_gemv(W_qkv, h)
        q_full = qkv[:q_sz]
        k_full = qkv[q_sz:q_sz + k_sz]
        v_full = qkv[q_sz + k_sz:]

        q = q_full[:n_h * hd].reshape(n_h, hd)
        q_gate = 1.0 / (1.0 + np.exp(-q_full[n_h * hd:]))
        k = k_full.reshape(n_kv, hd)
        v = v_full.reshape(n_kv, hd)

        cos, sin = self._rope
        q = apply_rope(q, cos, sin, pos)
        k = apply_rope(k, cos, sin, pos)

        Kc, Vc = kv
        Kc[:, pos, :] = k; Vc[:, pos, :] = v

        n_rep = n_h // n_kv
        k_rep = np.repeat(Kc[:, :seq_len, :], n_rep, axis=0)
        v_rep = np.repeat(Vc[:, :seq_len, :], n_rep, axis=0)

        scale = 1.0 / math.sqrt(hd)
        scores = np.sum(q[:, None, :] * k_rep, axis=-1) * scale
        scores -= np.max(scores, axis=-1, keepdims=True)
        attn_w = np.exp(scores); attn_w /= np.sum(attn_w, axis=-1, keepdims=True)
        attn_out = np.sum(attn_w[:, :, None] * v_rep, axis=1).flatten()

        return rust_gemv(w['self_attn.o_proj.weight'], attn_out * q_gate), (Kc, Vc)

    def _delta_net(self, h, l, state):
        w = self._w[l]
        mixed_qkv = rust_gemv(w['linear_attn.in_proj_qkv.weight'], h)
        z = rust_gemv(w['linear_attn.in_proj_z.weight'], h)
        b = rust_gemv(w['linear_attn.in_proj_b.weight'], h)
        a_vec = rust_gemv(w['linear_attn.in_proj_a.weight'], h)

        cs, ptr = state['conv_state'], state['conv_ptr']
        cs[:, ptr] = mixed_qkv
        new_ptr = (ptr + 1) % 4; state['conv_ptr'] = new_ptr
        order = [(new_ptr - i + 4) % 4 for i in range(4)]
        qkv_conv = np.sum(w['linear_attn.conv1d.weight'] * cs[:, order], axis=1)
        qkv_act = qkv_conv / (1.0 + np.exp(-qkv_conv))

        q = qkv_act[:2048].reshape(N_K_HEADS, HK)
        k = qkv_act[2048:4096].reshape(N_K_HEADS, HK)
        v = qkv_act[4096:].reshape(N_V_HEADS, HV)
        z = z.reshape(N_V_HEADS, HV)

        rep = N_V_HEADS // N_K_HEADS
        q = np.repeat(q, rep, axis=0); k = np.repeat(k, rep, axis=0)

        beta = 1.0 / (1.0 + np.exp(-b))
        g = -np.exp(w['linear_attn.A_log']) * np.log(1.0 + np.exp(a_vec + w['linear_attn.dt_bias']))

        q = q / (np.sqrt(np.sum(q**2, axis=1, keepdims=True)) + 1e-6)
        k = k / (np.sqrt(np.sum(k**2, axis=1, keepdims=True)) + 1e-6)
        q = q / math.sqrt(HK)

        S = state['S']
        S = S * np.exp(g).reshape(N_V_HEADS, 1, 1)
        kv_mem = np.sum(S * k[:, :, None], axis=1)
        delta = (v - kv_mem) * beta.reshape(N_V_HEADS, 1)
        S = S + k[:, :, None] * delta[:, None, :]
        output = np.sum(S * q[..., None], axis=1)
        state['S'] = S

        w_norm = w['linear_attn.norm.weight']
        rms = np.sqrt(np.mean(output**2, axis=1, keepdims=True) + 1e-6)
        on_normed = (output / rms) * w_norm.reshape(1, HV)
        gated = on_normed * z / (1.0 + np.exp(-z))
        return rust_gemv(w['linear_attn.out_proj.weight'], gated.reshape(-1))

    def _forward(self, h, l, pos, seq_len, kv, dstate):
        w = self._w[l]
        if 'input_layernorm.weight' in w:
            h = rms_norm(h, w['input_layernorm.weight'])

        if l % 4 == 3:
            ao, kv = self._full_attn(h, l, pos, seq_len, kv)
        elif 'linear_attn.in_proj_qkv.weight' in w:
            ao = self._delta_net(h, l, dstate)
        else:
            ao = np.zeros(HDIM, dtype=np.float32)

        h = h + ao

        if 'post_attention_layernorm.weight' in w:
            h = rms_norm(h, w['post_attention_layernorm.weight'])

        self._ensure_moe(l)
        moe_out = rust_moe(h, self._routers[l], self._gu[l], self._down[l], l)
        return h + moe_out, kv

    def generate(self, prompt_ids, max_tokens=20, temperature=0.7, top_k=40):
        kv_caches = [(np.zeros((N_KV, 256, HEAD_DIM), dtype=np.float32),
                      np.zeros((N_KV, 256, HEAD_DIM), dtype=np.float32)) for _ in range(40)]
        delta_states = [{'conv_state': np.zeros((8192, 4), dtype=np.float32),
                         'conv_ptr': 0,
                         'S': np.zeros((N_V_HEADS, HK, HV), dtype=np.float32)} for _ in range(40)]

        tokens = list(prompt_ids)
        n_prompt = len(tokens)
        print(f"Prefilling {n_prompt} tokens...")
        t0 = time.perf_counter()

        for i, tid in enumerate(tokens):
            h = self.embed[tid].astype(np.float32).copy()
            for l in range(40):
                h, kv_caches[l] = self._forward(h, l, i, i+1, kv_caches[l], delta_states[l])
            if i % 5 == 0 or i == n_prompt - 1:
                print(f"  [{i+1}/{n_prompt}] {time.perf_counter()-t0:.0f}s", flush=True)
        print(f"  Prefill done in {time.perf_counter()-t0:.1f}s")

        hn = rms_norm(h, self.final_norm_w)
        logits = self.embed @ hn

        if temperature == 0:
            next_token = int(np.argmax(logits))
        else:
            ls = logits / max(temperature, 0.01); ls -= np.max(ls)
            probs = np.exp(ls); probs /= np.sum(probs)
            if top_k > 0:
                tk = np.argpartition(-probs, top_k)[:top_k]
                probs = probs[tk] / np.sum(probs[tk])
                next_token = int(tk[np.random.choice(top_k, p=probs)])
            else:
                next_token = int(np.random.choice(len(probs), p=probs))

        generated = []
        t_start = time.perf_counter()

        for step in range(max_tokens):
            generated.append(next_token)
            pos = n_prompt + step
            if next_token == 2: break

            h = self.embed[next_token].astype(np.float32).copy()
            for l in range(40):
                h, kv_caches[l] = self._forward(h, l, pos, pos+1, kv_caches[l], delta_states[l])

            hn = rms_norm(h, self.final_norm_w)
            logits = self.embed @ hn

            if temperature == 0:
                next_token = int(np.argmax(logits))
            else:
                ls = logits / max(temperature, 0.01); ls -= np.max(ls)
                probs = np.exp(ls); probs /= np.sum(probs)
                if top_k > 0:
                    tk = np.argpartition(-probs, top_k)[:top_k]
                    tk_probs = probs[tk] / np.sum(probs[tk])
                    next_token = int(tk[np.random.choice(top_k, p=tk_probs)])
                else:
                    next_token = int(np.random.choice(len(probs), p=probs))

            if step % 5 == 0 or step == max_tokens - 1:
                elapsed = time.perf_counter() - t_start
                tok_s = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{step+1}/{max_tokens}] {tok_s:.2f} tok/s", flush=True)

        total_s = time.perf_counter() - t_start
        n_gen = len(generated)
        print(f"\n  {n_gen} tokens in {total_s:.1f}s ({n_gen/total_s:.2f} tok/s)")
        return generated


def main():
    print("=== Qwen3.6-35B-A3B (Rust NEON GEMV) ===\n")
    gen = Qwen36RustGen()

    from transformers import AutoTokenizer
    snap = sorted(os.listdir(
        "/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
    tok = AutoTokenizer.from_pretrained(
        f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}")
    print(f"  Vocab: {tok.vocab_size}\n")

    prompts = ["The meaning of life is"]
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(chat_text)
        print(f"\n── Prompt: \"{prompt}\" ──")
        gen_tokens = gen.generate(ids, max_tokens=20, temperature=0.7, top_k=40)
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        print(f"  Output: {text}")

if __name__ == "__main__":
    main()
