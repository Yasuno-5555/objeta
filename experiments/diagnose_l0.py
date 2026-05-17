#!/usr/bin/env python3
"""Compare Rust L0 DeltaNet vs Python reference at pos=0. Uses verified conv1d logic."""
import ctypes, numpy as np, json, mmap, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib

lib = get_lib()
BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HDIM = 2048
N_KH, N_VH, HK, HV = 16, 32, 128, 128

# Init
lib.lko_metal_init.argtypes = [ctypes.c_char_p]
lib.lko_metal_init.restype = ctypes.c_int32
METALLIB = str(Path(__file__).parent.parent / "target" / "objeta.metallib")
lib.lko_metal_init(METALLIB.encode())

lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
lib.lko_runner_init.restype = ctypes.c_int32
assert lib.lko_runner_init(str(BIN).encode(), 128)

lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
lib.lko_runner_set_fusion_ratio(1.0)
lib.lko_runner_set_moe_on_deltanet(1)

lib.lko_runner_forward_n.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
lib.lko_runner_forward_n.restype = ctypes.c_int32

def rust_forward_n(token_id, pos, seq_len, n_layers):
    h = np.zeros(HDIM, dtype=np.float32)
    lib.lko_runner_forward_n(token_id, pos, seq_len, n_layers, h.ctypes.data)
    return h

# Python DeltaNet (exact from compare_py_rust.py)
class AttnWeights:
    def __init__(self, layer_idx):
        with open(BIN / f"layer_{layer_idx}_attn_f16.json") as f:
            self.meta = json.load(f)
        self._mmap = np.memmap(BIN / f"layer_{layer_idx}_attn_f16.bin", dtype=np.float16, mode='r')
    def get(self, name):
        shape, offset, nbytes = self.meta[name]
        nelem = nbytes // 2
        return self._mmap[offset//2 : offset//2 + nelem].reshape(shape).astype(np.float32)

def rms_norm(x, w):
    return (x / np.sqrt(np.mean(x**2) + 1e-6)) * w

def py_deltanet(h_in, a, state):
    """Exact Python DeltaNet from compare_py_rust.py."""
    w = {}
    for k in ['linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight',
              'linear_attn.in_proj_b.weight','linear_attn.in_proj_a.weight',
              'linear_attn.out_proj.weight','linear_attn.norm.weight',
              'linear_attn.dt_bias','linear_attn.A_log']:
        w[k] = a.get(k)
    w['conv'] = a.get('linear_attn.conv1d.weight').reshape(8192, 4)

    mqkv = w['linear_attn.in_proj_qkv.weight'] @ h_in
    z = w['linear_attn.in_proj_z.weight'] @ h_in
    b = w['linear_attn.in_proj_b.weight'] @ h_in
    a_vec = w['linear_attn.in_proj_a.weight'] @ h_in

    cs, ptr = state['conv_state'], state['conv_ptr']
    cs[:, ptr] = mqkv
    np_ptr = (ptr + 1) % 4; state['conv_ptr'] = np_ptr
    order = [(np_ptr - i + 4) % 4 for i in range(4)]
    qkv_conv = np.sum(w['conv'] * cs[:, order], axis=1)
    qkv_act = qkv_conv / (1.0 + np.exp(-qkv_conv))

    q = qkv_act[:2048].reshape(N_KH, HK)
    k = qkv_act[2048:4096].reshape(N_KH, HK)
    v = qkv_act[4096:].reshape(N_VH, HV)
    z_rs = z.reshape(N_VH, HV)

    rep = N_VH // N_KH
    q = np.repeat(q, rep, axis=0); k = np.repeat(k, rep, axis=0)
    beta = 1.0/(1.0+np.exp(-b))
    g = -np.exp(w['linear_attn.A_log']) * np.log(1.0+np.exp(a_vec+w['linear_attn.dt_bias']))
    q = q/(np.sqrt(np.sum(q**2,axis=1,keepdims=True))+1e-6) / math.sqrt(HK)
    k = k/(np.sqrt(np.sum(k**2,axis=1,keepdims=True))+1e-6)

    S = state['S']
    S = S * np.exp(g).reshape(N_VH, 1, 1)
    kv_mem = np.sum(S * k[:,:,None], axis=1)
    delta = (v - kv_mem) * beta.reshape(N_VH, 1)
    S = S + k[:,:,None] * delta[:,None,:]
    output = np.sum(S * q[...,None], axis=1)
    state['S'] = S

    rms = np.sqrt(np.mean(output**2, axis=1, keepdims=True) + 1e-6)
    on_n = (output/rms) * w['linear_attn.norm.weight'].reshape(1, HV)
    gated = on_n * z_rs / (1.0+np.exp(-z_rs))
    return w['linear_attn.out_proj.weight'] @ gated.reshape(-1), state

embed = np.memmap(BIN/"embed_tokens.bin", dtype=np.float32, mode='r').reshape(-1, HDIM)
final_norm = np.frombuffer((BIN/"final_norm.bin").read_bytes(), dtype=np.float32)

token_id = 1058  # "The"
print(f"=== L0 DeltaNet comparison: token={token_id}, pos=0 ===\n")

# Python
a0 = AttnWeights(0)
h_in = embed[token_id].copy()
in_w = a0.get('input_layernorm.weight')
h_norm = rms_norm(h_in, in_w)

py_state = {'conv_state': np.zeros((8192,4),dtype=np.float32), 'conv_ptr':0, 'S': np.zeros((N_VH,HK,HV),dtype=np.float32)}
py_out, py_state = py_deltanet(h_norm, a0, py_state)

# Python doesn't apply post-norm or MoE (they don't exist for L0)
# Python only has residual: h_out = h_norm + py_out
py_final = h_norm + py_out

# Rust L0 (no post-norm, no MoE for L0)
rust_h0 = rust_forward_n(token_id, 0, 1, 1)

print(f"  Python: norm={np.linalg.norm(py_final):.4f}  ||py_out||={np.linalg.norm(py_out):.4f}")
print(f"  Rust:   norm={np.linalg.norm(rust_h0):.4f}")
cos_py_rust = np.dot(py_final, rust_h0) / (np.linalg.norm(py_final) * np.linalg.norm(rust_h0) + 1e-12)
print(f"  cos(py, rust) = {cos_py_rust:.6f}")

# Check lm_head predictions
hn_py = rms_norm(py_final, final_norm)
hn_rust = rms_norm(rust_h0, final_norm)
logits_py = embed @ hn_py
logits_rust = embed @ hn_rust
print(f"  Python top-5: {np.argsort(logits_py)[-5:][::-1][:5]}")
print(f"  Rust top-5:   {np.argsort(logits_rust)[-5:][::-1][:5]}")
print(f"  Correct (1058): py={logits_py[1058]:.2f} rust={logits_rust[1058]:.2f}")

# ── Step by step comparison ──
print("\n=== Step-by-step intermediate comparison ===")

w_qkv = a0.get('linear_attn.in_proj_qkv.weight')
w_z = a0.get('linear_attn.in_proj_z.weight')
w_out = a0.get('linear_attn.out_proj.weight')
w_conv = a0.get('linear_attn.conv1d.weight').reshape(8192, 4)

mqkv = w_qkv @ h_norm
z = w_z @ h_norm
print(f"  1. mixed_qkv norm:      {np.linalg.norm(mqkv):.4f}")
print(f"  2. z norm:              {np.linalg.norm(z):.4f}")

# Conv1d at pos=0: state is all zeros except just-written column
cs = np.zeros((8192, 4), dtype=np.float32)
cs[:, 0] = mqkv  # ptr=0 → write to col 0
np_ptr = 1       # after write
order = [(np_ptr - i + 4) % 4 for i in range(4)]  # [1,0,3,2]
print(f"  3. conv1d order:        {order}")
qkv_conv = np.sum(w_conv * cs[:, order], axis=1)
print(f"  4. qkv_conv norm:       {np.linalg.norm(qkv_conv):.4f} (should = ||w_conv[:,1] * mqkv||)")

# Verify: qkv_conv should equal w_conv[:,1] * mqkv
expected = w_conv[:, 1] * mqkv
diff = np.linalg.norm(qkv_conv - expected)
print(f"     check w[:,1]*mqkv:   diff={diff:.6f} (expected 0)")

# The Rust order at ptr=0: order = [(0+1)%4, (0+2)%4, (0+3)%4, 0] = [1,2,3,0]
rust_order = [1, 2, 3, 0]
rust_conv = np.sum(w_conv * cs[:, rust_order], axis=1)
print(f"  5. Rust qkv_conv norm:  {np.linalg.norm(rust_conv):.4f} (should = ||w_conv[:,3] * mqkv||)")
rust_expected = w_conv[:, 3] * mqkv
rust_diff = np.linalg.norm(rust_conv - rust_expected)
print(f"     check w[:,3]*mqkv:   diff={rust_diff:.6f}")

cos_conv = np.dot(qkv_conv, rust_conv) / (np.linalg.norm(qkv_conv) * np.linalg.norm(rust_conv) + 1e-12)
print(f"     cos(py_conv, rust_conv): {cos_conv:.6f}")

# Which weight is correct? Let's check: at pos=0, PyTorch conv1d applies which weight to the new input?
# From HANDOFF: "weight[:,3] applied to newest input, weight[:,0] to oldest"
#
# PyTorch Conv1d with kernel_size=4, padding=0, groups=8192:
#   input shape: (8192, 1)  — one position at a time
#   At pos=0: only the current input matters (no past context with zero-padding)
#   Conv1d: y[c] = sum_{k=0}^{3} weight[c,k] * input[c, t-k]
#   With zero-padding for t-k < 0:
#     y[c] = weight[c,0]*0 + weight[c,1]*0 + weight[c,2]*0 + weight[c,3]*input[c,0]
#   So weight[c,3] * input[c,0] — weight[:,3] newest!
#
# But the RING BUFFER approach is different! The state has 4 columns, and the newest
# input is written to the current column. The order maps oldest→newest for summation.
#
# Python order: [(1-0+4)%4, (1-1+4)%4, (1-2+4)%4, (1-3+4)%4] = [1, 0, 3, 2]
#   w[:,0]*s[:,1] + w[:,1]*s[:,0] + w[:,2]*s[:,3] + w[:,3]*s[:,2]
#   = w[:,0]*0 + w[:,1]*mqkv + w[:,2]*0 + w[:,3]*0
#   = w[:,1]*mqkv  ← WRONG for PyTorch Conv1d semantics!
#
# Rust order: [1, 2, 3, 0]
#   w[:,0]*s[:,1] + w[:,1]*s[:,2] + w[:,2]*s[:,3] + w[:,3]*s[:,0]
#   = w[:,0]*0 + w[:,1]*0 + w[:,2]*0 + w[:,3]*mqkv
#   = w[:,3]*mqkv  ← CORRECT!

print(f"\n  Python uses w[:,1] = weight index 1 (second-oldest)")
print(f"  Rust uses   w[:,3] = weight index 3 (newest)")
print(f"  HANDOFF/A1: weight[:,3] should apply to newest → Rust is CORRECT")
print(f"  Python reference is WRONG!")
