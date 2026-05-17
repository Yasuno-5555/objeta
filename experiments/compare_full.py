#!/usr/bin/env python3
"""Compare Python numpy vs Rust executor hidden state after full 40-layer forward."""
import ctypes, numpy as np, json, mmap, math, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
lib = get_lib()

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HDIM = 2048; N_KH, N_VH, HK, HV = 16, 32, 128, 128

# ── Weights ──
class AW:
    def __init__(self, l):
        with open(BIN/f"layer_{l}_attn_f16.json") as f: self.meta = json.load(f)
        self.mm = np.memmap(BIN/f"layer_{l}_attn_f16.bin", dtype=np.float16, mode='r')
    def g(self, n):
        s,o,nb = self.meta[n]; ne=nb//2
        return self.mm[o//2:o//2+ne].reshape(s).astype(np.float32)
    def h(self, n): return n in self.meta

def rm(x,w): return (x/np.sqrt(np.mean(x**2)+1e-6))*w

embed = np.memmap(BIN/"embed_tokens.bin", dtype=np.float32, mode='r').reshape(248320, HDIM)
fnw = np.fromfile(BIN/"final_norm.bin", dtype=np.float32)
routers = [np.fromfile(BIN/f"layer_{l}_router.bin", dtype=np.float32).reshape(256,HDIM) for l in range(40)]
gu = [np.memmap(BIN/f"layer_{l}_gate_up.bin", dtype=np.uint8, mode='r') for l in range(40)]
dw = [np.memmap(BIN/f"layer_{l}_down.bin", dtype=np.uint8, mode='r') for l in range(40)]

# MoE C API
lib.lko_moe_forward_layer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.lko_moe_forward_layer.restype = ctypes.c_int32
def moe(x, l):
    out = np.zeros(HDIM, dtype=np.float32); eidx=np.zeros(8,dtype=np.int32); ew=np.zeros(8,dtype=np.float32)
    lib.lko_moe_forward_layer(routers[l].ctypes.data_as(ctypes.c_void_p), gu[l].ctypes.data_as(ctypes.c_void_p), gu[l].nbytes, dw[l].ctypes.data_as(ctypes.c_void_p), dw[l].nbytes, x.ctypes.data_as(ctypes.c_void_p), 8, l, eidx.ctypes.data_as(ctypes.c_void_p), ew.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p))
    return out

# ── Python forward (1 layer) ──
def py_layer(h, l, pos, kv, ds):
    a = AW(l); w = {}
    for k in a.meta:
        if k != '__metadata__': w[k] = a.g(k)

    # input norm
    if 'input_layernorm.weight' in w: h = rm(h, w['input_layernorm.weight'])

    # attention
    if l%4==3:
        qw, kw, vw, ow = w['self_attn.q_proj.weight'], w['self_attn.k_proj.weight'], w['self_attn.v_proj.weight'], w['self_attn.o_proj.weight']
        q_full = qw @ h
        n_q = 16*256; q_gate = 1/(1+np.exp(-q_full[n_q:])); q = q_full[:n_q].reshape(16,256)
        k = (kw@h).reshape(2,256); v = (vw@h).reshape(2,256)
        # RoPE simplified for pos=0
        Kc,Vc = kv; Kc[:,pos,:]=k; Vc[:,pos,:]=v
        ao = np.zeros(HDIM, dtype=np.float32)
    elif 'linear_attn.in_proj_qkv.weight' in w:
        mqkv = w['linear_attn.in_proj_qkv.weight'] @ h
        z = w['linear_attn.in_proj_z.weight'] @ h
        b = w['linear_attn.in_proj_b.weight'] @ h
        a_vec = w['linear_attn.in_proj_a.weight'] @ h
        cs, ptr = ds['conv_state'], ds['conv_ptr']
        cs[:,ptr] = mqkv; np_ptr = (ptr+1)%4; ds['conv_ptr']=np_ptr
        order = [(np_ptr-i+4)%4 for i in range(4)]
        qkv_c = np.sum(w['linear_attn.conv1d.weight'].reshape(8192,4)*cs[:,order], axis=1)
        qkv_a = qkv_c/(1+np.exp(-qkv_c))
        q = qkv_a[:2048].reshape(N_KH,HK); k = qkv_a[2048:4096].reshape(N_KH,HK)
        v = qkv_a[4096:].reshape(N_VH,HV); z_rs = z.reshape(N_VH,HV)
        rep=N_VH//N_KH; q=np.repeat(q,rep,0); k=np.repeat(k,rep,0)
        beta = 1/(1+np.exp(-b))
        g = -np.exp(w['linear_attn.A_log'])*np.log(1+np.exp(a_vec+w['linear_attn.dt_bias']))
        q = q/(np.sqrt(np.sum(q**2,axis=1,keepdims=True))+1e-6)/math.sqrt(HK)
        k = k/(np.sqrt(np.sum(k**2,axis=1,keepdims=True))+1e-6)
        S = ds['S']; S = S*np.exp(g).reshape(N_VH,1,1)
        kv_mem = np.sum(S*k[:,:,None],axis=1)
        delta = (v-kv_mem)*beta.reshape(N_VH,1)
        S = S+k[:,:,None]*delta[:,None,:]; output=np.sum(S*q[...,None],axis=1); ds['S']=S
        rms=np.sqrt(np.mean(output**2,axis=1,keepdims=True)+1e-6)
        on_n=(output/rms)*w['linear_attn.norm.weight'].reshape(1,HV)
        gated=on_n*z_rs/(1+np.exp(-z_rs))
        ao = w['linear_attn.out_proj.weight']@gated.reshape(-1)
    else:
        ao = np.zeros(HDIM, dtype=np.float32)

    h = h + ao
    if 'post_attention_layernorm.weight' in w: h = rm(h, w['post_attention_layernorm.weight'])
    # shared expert
    if 'mlp.shared_expert.gate_proj.weight' in w:
        gate_h = w['mlp.shared_expert.gate_proj.weight']@h
        up_h = w['mlp.shared_expert.up_proj.weight']@h
        hidden = gate_h/(1+np.exp(-gate_h))*up_h
        se = w['mlp.shared_expert.down_proj.weight']@hidden
        seg = 1/(1+np.exp(-(w['mlp.shared_expert_gate.weight'].flatten()@h)))
        h = h + se*seg
    h = h + moe(h, l)
    return h, kv

# ── Init Rust ──
lib.lko_runner_init(str(BIN).encode(), 128)
lib.lko_runner_forward.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
lib.lko_runner_forward.restype = ctypes.c_int32

# ── Compare ──
tid = 1058
print(f"Token {tid} — Python vs Rust full forward\n")

# Python: 40 layers
py_h = embed[tid].copy()
py_kv = [(np.zeros((2,256,256),dtype=np.float32), np.zeros((2,256,256),dtype=np.float32)) for _ in range(40)]
py_ds = [{'conv_state':np.zeros((8192,4),dtype=np.float32),'conv_ptr':0,'S':np.zeros((N_VH,HK,HV),dtype=np.float32)} for _ in range(40)]
t0=time.perf_counter()
for l in range(40):
    py_h, py_kv[l] = py_layer(py_h, l, 0, py_kv[l], py_ds[l])
    if l in [0,2,7,14,21,39]:
        print(f"  Python L{l}: norm={np.linalg.norm(py_h):.4f}")

print(f"Python time: {time.perf_counter()-t0:.1f}s")
print(f"Python final norm: {np.linalg.norm(py_h):.4f}")

# Rust: 40 layers
rust_h = np.zeros(HDIM, dtype=np.float32)
t0=time.perf_counter()
lib.lko_runner_forward(tid, 0, 1, rust_h.ctypes.data)
print(f"\nRust time: {time.perf_counter()-t0:.1f}s")
print(f"Rust final norm: {np.linalg.norm(rust_h):.4f}")

cos = np.dot(py_h, rust_h)/(np.linalg.norm(py_h)*np.linalg.norm(rust_h)+1e-12)
print(f"cos(py, rust) = {cos:.6f}")

# Final RMSNorm + lm_head comparison
py_hn = rm(py_h, fnw)
rust_hn = rm(rust_h, fnw)
py_logits = embed @ py_hn
rust_logits = embed @ rust_hn
print(f"\nFinal logits:")
print(f"  Python top-5: {np.argsort(-py_logits)[:5]}")
print(f"  Rust   top-5: {np.argsort(-rust_logits)[:5]}")
print(f"  Python argmax: {np.argmax(py_logits)}")
print(f"  Rust   argmax: {np.argmax(rust_logits)}")
