#!/usr/bin/env python3
"""A1: HF reference DeltaNet vs our implementation — single layer."""
import numpy as np, json, math, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
import safetensors.torch as st

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
SHARD = "/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/model-00001-of-00026.safetensors"
HDIM, N_KH, N_VH, HK, HV = 2048, 16, 32, 128, 128

# ── Load weights ──────────────────────────────────────────────────────────

print("Loading weights...", end=" ", flush=True)
sf = {};
with st.safe_open(SHARD, framework="pt") as f:
    for k in f.keys(): sf[k] = f.get_tensor(k)
print(f"{len(sf)} tensors from shard")

# Our weights
class AW:
    def __init__(self, l):
        with open(BIN/f"layer_{l}_attn_f16.json") as f: self.meta = json.load(f)
        self.mm = np.memmap(BIN/f"layer_{l}_attn_f16.bin", dtype=np.float16, mode='r')
    def get(self, n):
        s,o,nb = self.meta[n]; ne=nb//2
        return self.mm[o//2:o//2+ne].reshape(s).astype(np.float32)

# ── Operations ────────────────────────────────────────────────────────────

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def silu(x): return x/(1.0+np.exp(-x))
def softplus(x): return np.log(1.0+np.exp(x))

def our_deltanet(h_np, a, state):
    t = {}
    w = {k: a.get(f'linear_attn.{k}') for k in [
        'in_proj_qkv.weight','in_proj_z.weight','in_proj_b.weight',
        'in_proj_a.weight','out_proj.weight','norm.weight','dt_bias','A_log']}
    w['conv'] = a.get('linear_attn.conv1d.weight').reshape(8192,4)

    mqkv = w['in_proj_qkv.weight'] @ h_np; t['mqkv'] = np.linalg.norm(mqkv)
    z = w['in_proj_z.weight'] @ h_np
    b = w['in_proj_b.weight'] @ h_np
    a_vec = w['in_proj_a.weight'] @ h_np

    cs, ptr = state['conv_state'], state['conv_ptr']
    cs[:,ptr] = mqkv; np_ptr = (ptr+1)%4; state['conv_ptr']=np_ptr
    order = [(ptr+1)%4, (ptr+2)%4, (ptr+3)%4, ptr]  # weight[3]=newest (cross-corr)
    qkv_c = np.sum(w['conv']*cs[:,order], axis=1); t['qkv_conv'] = np.linalg.norm(qkv_c)
    qkv_a = silu(qkv_c); t['qkv_act'] = np.linalg.norm(qkv_a)

    q = qkv_a[:2048].reshape(N_KH,HK); t['q_pre_l2'] = np.linalg.norm(q,axis=1).mean()
    k = qkv_a[2048:4096].reshape(N_KH,HK); t['k_pre_l2'] = np.linalg.norm(k,axis=1).mean()
    v = qkv_a[4096:].reshape(N_VH,HV)
    z_rs = z.reshape(N_VH,HV)

    rep=N_VH//N_KH; q=np.repeat(q,rep,0); k=np.repeat(k,rep,0)
    beta = sigmoid(b)
    g = -np.exp(w['A_log'])*softplus(a_vec+w['dt_bias']); t['g_mean']=np.mean(g)
    q=q/(np.sqrt(np.sum(q**2,axis=1,keepdims=True))+1e-6)/math.sqrt(HK)
    k=k/(np.sqrt(np.sum(k**2,axis=1,keepdims=True))+1e-6)

    S=state['S']; t['S_pre']=np.linalg.norm(S)
    S=S*np.exp(g).reshape(N_VH,1,1)
    kv_mem=np.sum(S*k[:,:,None],axis=1)
    delta=(v-kv_mem)*beta.reshape(N_VH,1)
    S=S+k[:,:,None]*delta[:,None,:]; output=np.sum(S*q[...,None],axis=1)
    state['S']=S; t['S_post']=np.linalg.norm(S)

    rms=np.sqrt(np.mean(output**2,axis=1,keepdims=True)+1e-6)
    on_n=(output/rms)*w['norm.weight'].reshape(1,HV)
    gated=on_n*z_rs*sigmoid(z_rs); t['gated']=np.linalg.norm(gated)
    final=w['out_proj.weight']@gated.reshape(-1); t['final']=np.linalg.norm(final)
    return final, state, t

def hf_deltanet(h_torch, sf, a, L):
    prefix = f"model.language_model.layers.{L}.linear_attn."

    def gw(name):
        k = prefix + name
        if k in sf: return sf[k].float()
        # fall back to our weights
        return torch.from_numpy(a.get(f'linear_attn.{name}')).float()

    w_qkv = gw('in_proj_qkv.weight'); w_z = gw('in_proj_z.weight')
    w_b = gw('in_proj_b.weight'); w_a = gw('in_proj_a.weight')
    w_out = gw('out_proj.weight'); w_norm = gw('norm.weight')
    dt_bias = gw('dt_bias'); A_log = gw('A_log')
    w_conv = gw('conv1d.weight').squeeze(1)  # (8192,4)

    t = {}
    with torch.no_grad():
        mqkv = w_qkv @ h_torch; t['mqkv'] = torch.norm(mqkv).item()
        z = w_z @ h_torch; b = w_b @ h_torch; a_vec = w_a @ h_torch

        # HF causal conv1d: (1, 8192, seq) with groups=8192
        x = mqkv.reshape(1, 8192, 1)  # batch=1, channels=8192, seq=1
        padded = F.pad(x, (3, 0))     # causal: left-pad 3 zeros → (1, 8192, 4)
        conv_out = F.conv1d(padded, w_conv.unsqueeze(1), groups=8192)  # (1, 8192, 1)
        qkv_c = conv_out.reshape(-1); t['qkv_conv'] = torch.norm(qkv_c).item()
        qkv_a = F.silu(qkv_c); t['qkv_act'] = torch.norm(qkv_a).item()

        q = qkv_a[:2048].reshape(N_KH,HK); t['q_pre_l2'] = torch.norm(q,dim=1).mean().item()
        k = qkv_a[2048:4096].reshape(N_KH,HK); t['k_pre_l2'] = torch.norm(k,dim=1).mean().item()
        v = qkv_a[4096:].reshape(N_VH,HV)
        z_rs = z.reshape(N_VH,HV)

        rep=N_VH//N_KH; q=q.repeat_interleave(rep,0); k=k.repeat_interleave(rep,0)
        beta=torch.sigmoid(b)
        g=-torch.exp(A_log.float())*F.softplus(a_vec.float()+dt_bias.float())
        t['g_mean']=g.mean().item()

        q=F.normalize(q,p=2,dim=-1,eps=1e-6)/math.sqrt(HK)
        k=F.normalize(k,p=2,dim=-1,eps=1e-6)

        S=torch.zeros(N_VH,HK,HV); t['S_pre']=torch.norm(S).item()
        S=S*torch.exp(g).reshape(N_VH,1,1)
        kv_mem=(S*k.unsqueeze(-1)).sum(dim=1)
        delta=(v-kv_mem)*beta.unsqueeze(-1)
        S=S+k.unsqueeze(-1)*delta.unsqueeze(1)
        output=(S*q.unsqueeze(-1)).sum(dim=1); t['S_post']=torch.norm(S).item()

        rms=torch.sqrt(torch.mean(output**2,dim=1,keepdims=True)+1e-6)
        on_n=(output/rms)*w_norm.reshape(1,HV)
        gated=on_n*z_rs*torch.sigmoid(z_rs); t['gated']=torch.norm(gated).item()
        final=w_out@gated.reshape(-1); t['final']=torch.norm(final).item()
    return final, S, t

# ── Run comparison ────────────────────────────────────────────────────────

L = 1
a = AW(L)
embed = np.memmap(BIN/"embed_tokens.bin", dtype=np.float32, mode='r').reshape(248320, HDIM)
h_in = embed[1058].copy()
in_w = a.get('input_layernorm.weight')
h_norm = h_in/np.sqrt(np.mean(h_in**2)+1e-6)*in_w
print(f"L{L}, input norm: {np.linalg.norm(h_norm):.4f}\n")

# Our
state = {'conv_state':np.zeros((8192,4),dtype=np.float32), 'conv_ptr':0, 'S':np.zeros((N_VH,HK,HV),dtype=np.float32)}
our_out, _, our_t = our_deltanet(h_norm, a, state)

# HF
h_torch = torch.from_numpy(h_norm).float()
hf_out, _, hf_t = hf_deltanet(h_torch, sf, a, L)

# ── Report ──
print(f"{'Metric':<20s} {'Ours':>10s} {'HF':>10s} {'Ratio':>8s}")
print("-"*50)
for key in our_t:
    our_v = our_t[key]; hf_v = hf_t.get(key, 0)
    ratio = our_v/max(hf_v, 1e-12)
    flag = "✓" if 0.8<ratio<1.25 else ("△" if 0.5<ratio<2.0 else "✗")
    print(f"{key:<20s} {our_v:>10.4f} {hf_v:>10.4f} {ratio:>7.2f}x {flag}")

cos = np.dot(our_out, hf_out.numpy())/(np.linalg.norm(our_out)*torch.norm(hf_out).item()+1e-12)
print(f"\ncos(our, hf) = {cos:.6f}")
print(f"Our  norm: {np.linalg.norm(our_out):.4f}")
print(f"HF   norm: {torch.norm(hf_out).item():.4f}")
