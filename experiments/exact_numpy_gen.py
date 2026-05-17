#!/usr/bin/env python3
"""EXACT reproduction of the working MLX generator in pure numpy.
Goal: produce coherent output at temp=0, then port to Rust 1:1.
"""
import ctypes, json, math, os, sys, time, struct, mmap
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
_lib = get_lib()

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HDIM = 2048

# ── Rust FFI ──
_lib.lko_moe_forward_layer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.lko_moe_forward_layer.restype = ctypes.c_int32
def rust_moe(x_np, router, gu_mmap, d_mmap, layer_idx):
    out=np.zeros(HDIM,dtype=np.float32); eidx=np.zeros(8,dtype=np.int32); ew=np.zeros(8,dtype=np.float32)
    _lib.lko_moe_forward_layer(router.ctypes.data_as(ctypes.c_void_p), gu_mmap.ctypes.data_as(ctypes.c_void_p), gu_mmap.nbytes, d_mmap.ctypes.data_as(ctypes.c_void_p), d_mmap.nbytes, x_np.ctypes.data_as(ctypes.c_void_p), 8, layer_idx, eidx.ctypes.data_as(ctypes.c_void_p), ew.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p))
    return out

# ── Weight loading (EXACT same as MLX version) ──
class AW:
    def __init__(self, l):
        with open(BIN/f"layer_{l}_attn_f16.json") as f: self.meta = json.load(f)
        self.mm = np.memmap(BIN/f"layer_{l}_attn_f16.bin", dtype=np.float16, mode='r')
    def get(self, n):
        s,o,nb = self.meta[n]; ne=nb//2
        return self.mm[o//2:o//2+ne].reshape(s).astype(np.float32)
    def has(self, n): return n in self.meta

embed = np.memmap(BIN/"embed_tokens.bin", dtype=np.float32, mode='r').reshape(248320, HDIM)
fnw = np.fromfile(BIN/"final_norm.bin", dtype=np.float32)

# ── Operations (EXACT same math as MLX) ──
def rms_norm(x, w): return (x / np.sqrt(np.mean(x**2) + 1e-6)) * w
def silu(x): return x / (1.0 + np.exp(-x))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def softplus(x): return np.log(1.0 + np.exp(x))

# ── Generator ──
class ExactGen:
    def __init__(self):
        self.attn = [AW(l) for l in range(40)]
        self.routers = [np.fromfile(BIN/f"layer_{l}_router.bin", dtype=np.float32).reshape(256, HDIM) for l in range(40)]
        self.gu = [np.memmap(BIN/f"layer_{l}_gate_up.bin", dtype=np.uint8, mode='r') for l in range(40)]
        self.down = [np.memmap(BIN/f"layer_{l}_down.bin", dtype=np.uint8, mode='r') for l in range(40)]

    def _full_attn(self, h, l, pos, kv):
        a = self.attn[l]
        qw, kw, vw, ow = a.get('self_attn.q_proj.weight'), a.get('self_attn.k_proj.weight'), a.get('self_attn.v_proj.weight'), a.get('self_attn.o_proj.weight')
        N_H, N_KV, HD = 16, 2, 256
        q_full = qw @ h  # (8192,) = Q(4096) + Q-gate(4096)
        n_q = N_H * HD  # 4096
        q = q_full[:n_q].reshape(N_H, HD)
        q_gate = sigmoid(q_full[n_q:])  # (4096,) — 1 gate per dimension
        k = (kw @ h).reshape(N_KV, HD)
        v = (vw @ h).reshape(N_KV, HD)
        # RoPE
        d2 = HD//2
        theta = 1.0 / (10000.0 ** (np.arange(0, HD, 2)/HD))
        c = np.cos(pos * theta).astype(np.float32)
        s = np.sin(pos * theta).astype(np.float32)
        q_e, q_o = q[:,:d2], q[:,d2:]
        k_e, k_o = k[:,:d2], k[:,d2:]
        q = np.concatenate([q_e*c - q_o*s, q_e*s + q_o*c], axis=-1)
        k = np.concatenate([k_e*c - k_o*s, k_e*s + k_o*c], axis=-1)
        Kc, Vc = kv
        Kc[:,pos,:]=k; Vc[:,pos,:]=v
        n_rep=N_H//N_KV; seq_len=pos+1
        k_rep=np.repeat(Kc[:,:seq_len,:],n_rep,axis=0)
        v_rep=np.repeat(Vc[:,:seq_len,:],n_rep,axis=0)
        scale=1.0/math.sqrt(HD)
        scores=np.sum(q[:,None,:]*k_rep,axis=-1)*scale
        scores-=np.max(scores,axis=-1,keepdims=True)
        attn_w=np.exp(scores); attn_w/=np.sum(attn_w,axis=-1,keepdims=True)
        attn_out=np.sum(attn_w[:,:,None]*v_rep,axis=1).flatten()
        # q_gate: (4096,) element-wise × attn_out (4096,) — 1 gate per dimension
        gated = attn_out * q_gate
        return ow @ gated, (Kc, Vc)

    def _delta_net(self, h, l, state):
        a = self.attn[l]
        N_KH, N_VH, HK, HV = 16, 32, 128, 128
        w_qkv = a.get('linear_attn.in_proj_qkv.weight')
        w_z = a.get('linear_attn.in_proj_z.weight')
        w_b = a.get('linear_attn.in_proj_b.weight')
        w_a = a.get('linear_attn.in_proj_a.weight')
        w_out = a.get('linear_attn.out_proj.weight')
        w_conv = a.get('linear_attn.conv1d.weight').reshape(8192, 4)
        w_norm = a.get('linear_attn.norm.weight')
        dt_bias = a.get('linear_attn.dt_bias')
        A_log = a.get('linear_attn.A_log')

        mixed_qkv = w_qkv @ h
        z = w_z @ h; b = w_b @ h; a_vec = w_a @ h

        cs, ptr = state['conv_state'], state['conv_ptr']
        cs[:, ptr] = mixed_qkv
        new_ptr = (ptr + 1) % 4; state['conv_ptr'] = new_ptr
        order = [(ptr+1)%4, (ptr+2)%4, (ptr+3)%4, ptr]  # weight[3]=newest (cross-corr)
        qkv_conv = np.sum(w_conv * cs[:, order], axis=1)
        qkv_act = silu(qkv_conv)

        q = qkv_act[:2048].reshape(N_KH, HK)
        k = qkv_act[2048:4096].reshape(N_KH, HK)
        v = qkv_act[4096:].reshape(N_VH, HV)
        z_rs = z.reshape(N_VH, HV)

        rep = N_VH // N_KH
        q = np.repeat(q, rep, axis=0); k = np.repeat(k, rep, axis=0)

        beta = sigmoid(b)
        g = -np.exp(A_log) * softplus(a_vec + dt_bias)

        # L2 norm q,k
        q = q / (np.sqrt(np.sum(q**2, axis=1, keepdims=True)) + 1e-6)
        k = k / (np.sqrt(np.sum(k**2, axis=1, keepdims=True)) + 1e-6)
        q = q / math.sqrt(HK)

        S = state['S']
        S = S * np.exp(g).reshape(N_VH, 1, 1)
        kv_mem = np.sum(S * k[:, :, None], axis=1)
        delta = (v - kv_mem) * beta.reshape(N_VH, 1)
        S = S + k[:, :, None] * delta[:, None, :]
        output = np.sum(S * q[..., None], axis=1)
        state['S'] = S

        # RMSNormGated
        rms = np.sqrt(np.mean(output**2, axis=1, keepdims=True) + 1e-6)
        on_normed = (output / rms) * w_norm.reshape(1, HV)
        gated = on_normed * z_rs * sigmoid(z_rs)
        return w_out @ gated.reshape(-1)

    def _forward(self, h, l, pos, kv, ds):
        a = self.attn[l]
        if a.has('input_layernorm.weight'):
            h = rms_norm(h, a.get('input_layernorm.weight'))

        if l % 4 == 3:
            ao, kv = self._full_attn(h, l, pos, kv)
        elif a.has('linear_attn.in_proj_qkv.weight'):
            ao = self._delta_net(h, l, ds)
        else:
            ao = np.zeros(HDIM, dtype=np.float32)

        h = h + ao

        if a.has('post_attention_layernorm.weight'):
            h = rms_norm(h, a.get('post_attention_layernorm.weight'))

        # Shared expert
        if a.has('mlp.shared_expert.gate_proj.weight'):
            gw = a.get('mlp.shared_expert.gate_proj.weight')
            uw = a.get('mlp.shared_expert.up_proj.weight')
            dw = a.get('mlp.shared_expert.down_proj.weight')
            gg = a.get('mlp.shared_expert_gate.weight').flatten()
            gate_h = gw @ h; up_h = uw @ h
            hidden = silu(gate_h) * up_h
            se_out = dw @ hidden
            se_gate = sigmoid(gg @ h)
            h = h + se_out * se_gate

        # MoE
        moe_out = rust_moe(h, self.routers[l], self.gu[l], self.down[l], l)
        return h + moe_out, kv

    def generate(self, prompt_ids, max_tokens=20, temperature=0.7, top_k=40):
        kv_caches = [(np.zeros((2, 256, 256), dtype=np.float32), np.zeros((2, 256, 256), dtype=np.float32)) for _ in range(40)]
        delta_states = [{'conv_state': np.zeros((8192, 4), dtype=np.float32), 'conv_ptr': 0, 'S': np.zeros((32, 128, 128), dtype=np.float32)} for _ in range(40)]

        tokens = list(prompt_ids); n_prompt = len(tokens)
        print(f"Prefilling {n_prompt} tokens..."); t0 = time.perf_counter()
        for i, tid in enumerate(tokens):
            h = embed[tid].astype(np.float32).copy()
            for l in range(40):
                h, kv_caches[l] = self._forward(h, l, i, kv_caches[l], delta_states[l])
            if i%5==0 or i==n_prompt-1: print(f"  [{i+1}/{n_prompt}] {time.perf_counter()-t0:.0f}s", flush=True)
        print(f"  Prefill done in {time.perf_counter()-t0:.1f}s")

        hn = rms_norm(h, fnw); logits = embed @ hn

        if temperature == 0: next_token = int(np.argmax(logits))
        else:
            ls = logits/max(temperature,0.01); ls-=np.max(ls); probs=np.exp(ls); probs/=np.sum(probs)
            if top_k>0: tk=np.argpartition(-probs,top_k)[:top_k]; probs=probs[tk]/np.sum(probs[tk]); next_token=int(tk[np.random.choice(top_k,p=probs)])
            else: next_token=int(np.random.choice(len(probs),p=probs))

        generated = []; t_start = time.perf_counter()
        for step in range(max_tokens):
            generated.append(next_token); pos=n_prompt+step
            if next_token==2: break
            h = embed[next_token].astype(np.float32).copy()
            for l in range(40): h, kv_caches[l] = self._forward(h, l, pos, kv_caches[l], delta_states[l])
            hn = rms_norm(h, fnw); logits = embed @ hn
            if temperature == 0: next_token = int(np.argmax(logits))
            else:
                ls = logits/max(temperature,0.01); ls-=np.max(ls); probs=np.exp(ls); probs/=np.sum(probs)
                if top_k>0: tk=np.argpartition(-probs,top_k)[:top_k]; tk_probs=probs[tk]/np.sum(probs[tk]); next_token=int(tk[np.random.choice(top_k,p=tk_probs)])
                else: next_token=int(np.random.choice(len(probs),p=probs))
            if step%5==0 or step==max_tokens-1: print(f"  [{step+1}/{max_tokens}] {time.perf_counter()-t_start:.0f}s", flush=True)

        total_s = time.perf_counter()-t_start; n_gen=len(generated)
        print(f"\n  {n_gen} tokens in {total_s:.1f}s ({n_gen/total_s:.2f} tok/s)")
        return generated

if __name__ == "__main__":
    from transformers import AutoTokenizer
    snap=sorted(os.listdir("/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
    tok=AutoTokenizer.from_pretrained(f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}")
    print(f"Vocab: {tok.vocab_size}\n")
    for prompt in ["The meaning of life is"]:
        msgs=[{"role":"user","content":prompt}]
        chat=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        ids=tok.encode(chat)
        print(f"── Prompt: \"{prompt}\" ──")
        gen = ExactGen().generate(ids, max_tokens=15, temperature=0, top_k=0)
        text=tok.decode(gen,skip_special_tokens=True)
        print(f"  Output: {text}")
