#!/usr/bin/env python3
"""Full 40-layer Qwen3.6 forward pass: Rust vs HuggingFace comparison.

Tracks hidden state cosine through every layer to find where divergence begins.
"""
import ctypes, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import safetensors.torch as st
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO))
sys.path.insert(0, str(PROJECT))

from experiments.qwen36_executor import get_lib

BIN = PROJECT / "models" / "qwen36_bin"
SNAPSHOT = "/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
HDIM, N_KH, N_VH, HK, HV = 2048, 16, 32, 128, 128
N_LAYERS = 40
GQA_LAYERS = {3,7,11,15,19,23,27,31,35,39}


def objeta_debug_enabled():
    value = os.environ.get("OBJETA_DEBUG", "")
    return value == "1" or value.lower() == "true"

# ── Weight loading ──────────────────────────────────────────────────────────

class AttnWeights:
    def __init__(self, l):
        with open(BIN / f"layer_{l}_attn_f16.json") as f:
            self.meta = json.load(f)
        self.mm = np.memmap(BIN / f"layer_{l}_attn_f16.bin", dtype=np.float16, mode='r')
    def get(self, n):
        s, o, nb = self.meta[n]
        ne = nb // 2
        return self.mm[o//2:o//2+ne].reshape(s).astype(np.float32)

def load_hf_tensors():
    """Load all tensors from HF shards. Lazy — only first shard for layer weights."""
    tensors = {}
    shard1 = f"{SNAPSHOT}/model-00001-of-00026.safetensors"
    with st.safe_open(shard1, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

# ── HF forward (reference) ──────────────────────────────────────────────────

def hf_forward_layer(h, sf, aw, L, kv_k, kv_v, pos, seq_len):
    """Run layer L of HF model. Returns (h_out, kv_k, kv_v)."""
    prefix = f"model.language_model.layers.{L}."

    def gw(name):
        k = prefix + name
        if k in sf:
            return sf[k].float()
        return torch.from_numpy(aw.get(name)).float()

    is_gqa = L in GQA_LAYERS
    has_deltanet = not is_gqa

    # Input norm
    in_w = gw('input_layernorm.weight')
    h_norm = F.rms_norm(h, (HDIM,), 1.0 + in_w, 1e-6)

    if objeta_debug_enabled() and L == 0:
        print(f"[HF DEBUG L0] h_orig norm: {torch.norm(h).item():.6f}, first 5: {h[:5].numpy()}")
        print(f"[HF DEBUG L0] h_norm norm: {torch.norm(h_norm).item():.6f}, first 5: {h_norm[:5].numpy()}")

    if has_deltanet:
        # DeltaNet
        w_qkv = gw('linear_attn.in_proj_qkv.weight')
        w_z = gw('linear_attn.in_proj_z.weight')
        w_b = gw('linear_attn.in_proj_b.weight')
        w_a = gw('linear_attn.in_proj_a.weight')
        w_out = gw('linear_attn.out_proj.weight')
        w_norm = gw('linear_attn.norm.weight')
        dt_bias = gw('linear_attn.dt_bias')
        A_log = gw('linear_attn.A_log')
        w_conv = gw('linear_attn.conv1d.weight').squeeze(1)

        mqkv = w_qkv @ h_norm
        z = w_z @ h_norm; b = w_b @ h_norm; a_vec = w_a @ h_norm

        x = mqkv.reshape(1, 8192, 1)
        padded = F.pad(x, (3, 0))
        conv_out = F.conv1d(padded, w_conv.unsqueeze(1), groups=8192)
        qkv_c = conv_out.reshape(-1)
        qkv_a = F.silu(qkv_c)

        q = qkv_a[:2048].reshape(N_KH, HK)
        k = qkv_a[2048:4096].reshape(N_KH, HK)
        v = qkv_a[4096:].reshape(N_VH, HV)
        z_rs = z.reshape(N_VH, HV)

        rep = N_VH // N_KH
        q = q.repeat_interleave(rep, 0); k = k.repeat_interleave(rep, 0)
        beta = torch.sigmoid(b)
        g = -torch.exp(A_log.float()) * F.softplus(a_vec.float() + dt_bias.float())
        
        if objeta_debug_enabled() and L == 0 and pos == 0:
            print(f"[HF DETAILED L0] mqkv norm: {torch.norm(mqkv).item():.6f}")
            print(f"[HF DETAILED L0] z norm: {torch.norm(z).item():.6f}")
            print(f"[HF DETAILED L0] b norm: {torch.norm(b).item():.6f}")
            print(f"[HF DETAILED L0] a_vec norm: {torch.norm(a_vec).item():.6f}")
            print(f"[HF DETAILED L0] qkv_a norm: {torch.norm(qkv_a).item():.6f}")
            print(f"[HF DETAILED L0] beta norm: {torch.norm(beta).item():.6f}")
            print(f"[HF DETAILED L0] exp(g) norm: {torch.norm(torch.exp(g)).item():.6f}")

        q = F.normalize(q, p=2, dim=-1, eps=1e-6) / math.sqrt(HK)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        S = torch.zeros(N_VH, HK, HV)
        S = S * torch.exp(g).reshape(N_VH, 1, 1)
        kv_mem = (S * k.unsqueeze(-1)).sum(dim=1)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        S = S + k.unsqueeze(-1) * delta.unsqueeze(1)
        output = (S * q.unsqueeze(-1)).sum(dim=1)

        rms = torch.sqrt(torch.mean(output ** 2, dim=1, keepdims=True) + 1e-6)
        on_n = (output / rms) * w_norm.reshape(1, HV)
        gated = on_n * z_rs * torch.sigmoid(z_rs)
        ao = w_out @ gated.reshape(-1)

        if objeta_debug_enabled() and L == 0 and pos == 0:
            print(f"[HF DETAILED L0] normalized q norm: {torch.norm(q).item():.6f}")
            print(f"[HF DETAILED L0] normalized k norm: {torch.norm(k).item():.6f}")
            print(f"[HF DETAILED L0] S_state norm: {torch.norm(S).item():.6f}")
            print(f"[HF DETAILED L0] output norm: {torch.norm(output).item():.6f}")
            print(f"[HF DETAILED L0] gated norm: {torch.norm(gated).item():.6f}")
            print(f"[HF DETAILED L0] ao norm: {torch.norm(ao).item():.6f}")
    else:
        # GQA
        w_q = gw('self_attn.q_proj.weight')
        w_k = gw('self_attn.k_proj.weight')
        w_v = gw('self_attn.v_proj.weight')
        w_o = gw('self_attn.o_proj.weight')

        n_heads, n_kv, hd = 16, 2, 256
        q_full = w_q @ h_norm
        q_chunks = q_full.reshape(n_heads, hd * 2)
        q = q_chunks[:, :hd]
        q_gate = torch.sigmoid(q_chunks[:, hd:]).reshape(-1)
        k = (w_k @ h_norm).reshape(n_kv, hd)
        v = (w_v @ h_norm).reshape(n_kv, hd)

        if objeta_debug_enabled() and L == 3 and pos == 0:
            print(f"  [HF GQA DEBUG] h norm = {torch.norm(h_norm).item():.6f}")
            print(f"  [HF GQA DEBUG] q_full norm = {torch.norm(q_full).item():.6f}")
            print(f"  [HF GQA DEBUG] q_proj norm = {torch.norm(q).item():.6f}, k_proj norm = {torch.norm(k).item():.6f}, v_proj norm = {torch.norm(v).item():.6f}")

        # Q/K RMSNorm matching Qwen3MoeAttention
        q_norm_w = gw('self_attn.q_norm.weight')
        k_norm_w = gw('self_attn.k_norm.weight')

        q_rms = torch.sqrt(torch.mean(q ** 2, dim=-1, keepdim=True) + 1e-6)
        q_normed = (q / q_rms) * (1.0 + q_norm_w.reshape(1, hd))

        k_rms = torch.sqrt(torch.mean(k ** 2, dim=-1, keepdim=True) + 1e-6)
        k_normed = (k / k_rms) * (1.0 + k_norm_w.reshape(1, hd))

        if objeta_debug_enabled() and L == 3 and pos == 0:
            print(f"  [HF GQA DEBUG] q_normed norm = {torch.norm(q_normed).item():.6f}, k_normed norm = {torch.norm(k_normed).item():.6f}")

        kv_k[:, pos, :] = k_normed; kv_v[:, pos, :] = v
        Kc = kv_k[:, :seq_len, :]; Vc = kv_v[:, :seq_len, :]

        n_rep = n_heads // n_kv
        scale = 1.0 / math.sqrt(hd)
        k_rep = Kc.repeat_interleave(n_rep, dim=0)
        v_rep = Vc.repeat_interleave(n_rep, dim=0)
        scores = torch.sum(q_normed.unsqueeze(1) * k_rep, dim=-1) * scale
        attn_w = torch.softmax(scores, dim=-1)
        attn_out = torch.sum(attn_w.unsqueeze(-1) * v_rep, dim=1).flatten()
        ao = w_o @ (attn_out * q_gate)

    h = h + ao

    # Post-attention norm
    post_w = gw('post_attention_layernorm.weight')
    hn2 = F.rms_norm(h, (HDIM,), 1.0 + post_w, 1e-6)

    # Shared expert (sigmoid-gated FFN)
    se_gate = gw('mlp.shared_expert.gate_proj.weight')
    se_up = gw('mlp.shared_expert.up_proj.weight')
    se_down = gw('mlp.shared_expert.down_proj.weight')
    se_gate_w = gw('mlp.shared_expert_gate.weight')

    gate = se_gate @ hn2
    up = se_up @ hn2
    hidden = F.silu(gate) * up
    se_out = se_down @ hidden
    
    se_gate_val = torch.sigmoid(torch.dot(se_gate_w.flatten(), hn2.flatten()))
    
    if objeta_debug_enabled() and L == 0 and pos == 0:
        print(f"[HF DETAILED L0] se_gate_val: {se_gate_val.item():.6f}")

    # MoE router
    # router = gw('mlp.gate.weight')
    # For now skip MoE dispatch (too complex to replicate exactly)
    # Just add shared expert output scaled by the gate

    h = h + se_out * se_gate_val
    return h, kv_k, kv_v



# ── Our forward (via Rust FFI) ──────────────────────────────────────────────

def init_rust():
    lib = get_lib()
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    assert lib.lko_runner_init(str(BIN).encode(), 256), "Rust init failed"

    lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
    lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
    lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
    lib.lko_runner_set_moe_enabled.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_enabled.restype = ctypes.c_int32
    # Disable routed MoE but enable shared expert on DeltaNet layers
    lib.lko_runner_set_fusion_ratio(1.0)
    lib.lko_runner_set_moe_on_deltanet(1)
    lib.lko_runner_set_moe_enabled(0)

    lib.lko_runner_forward.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
    lib.lko_runner_forward.restype = ctypes.c_int32

    lib.lko_runner_trace_layers.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
    lib.lko_runner_trace_layers.restype = ctypes.c_int32
    return lib

def rust_forward(lib, token_id, pos, seq_len):
    h = np.zeros(HDIM, dtype=np.float32)
    lib.lko_runner_forward(token_id, pos, seq_len, h.ctypes.data)
    return h


def hf_exact_layer_outputs(model, prefix_ids):
    """Capture exact per-layer hidden states from the real HF model via hooks."""
    captured = [None] * N_LAYERS
    handles = []

    def make_hook(layer_idx):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hidden[0, -1].float().detach().cpu().numpy()
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([prefix_ids], dtype=torch.long), output_hidden_states=False)
    finally:
        for h in handles:
            h.remove()

    assert all(x is not None for x in captured), "failed to capture some HF layer outputs"
    return captured

# ── Main comparison ─────────────────────────────────────────────────────────

def main():
    print("Loading HF weights...", end=" ", flush=True)
    sf = load_hf_tensors()
    print(f"{len(sf)} tensors")
    model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.bfloat16, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)

    # Monkey-patch MoE blocks to skip routed experts (like Rust's moe_enabled: 0)
    import types
    def patched_forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        
        shared_expert_gate_val = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
        shared_expert_output = shared_expert_gate_val * shared_expert_output
        
        expert_output = shared_expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output

    for layer in model.model.layers:
        layer.mlp.forward = types.MethodType(patched_forward, layer.mlp)

    # Load our weights
    aws = [AttnWeights(l) for l in range(N_LAYERS)]
    embed_hf = sf["model.language_model.embed_tokens.weight"].float()
    embed_ours = np.memmap(BIN / "embed_tokens.bin", dtype=np.float32, mode='r').reshape(248320, HDIM)
    vocab = len(embed_ours)
    print(f"Vocab: {vocab}")

    # Init Rust
    print("Init Rust executor...", end=" ", flush=True)
    lib = init_rust()
    print("OK")

    # Simple text tokens
    prompt = "The meaning of life is"
    prompt_ids = tok.encode(prompt)
    print(f"Prompt: {prompt} -> {len(prompt_ids)} tokens: {prompt_ids[:10]}")

    n_tokens = min(len(prompt_ids), 5)
    print(f"\nRunning {n_tokens} tokens through both implementations...")

    for pos in range(n_tokens):
        tid = prompt_ids[pos]
        seq_len = pos + 1

        print(f"\n── Token {pos} (id={tid}) seq_len={seq_len} ──")

        # HF exact forward on the growing prefix (stateful reference)
        prefix_ids = prompt_ids[:seq_len]
        h_hf = embed_hf[tid].clone()

        print(f"Initial Embedding Norm: HF={torch.norm(h_hf).item():.6f} | Ours={np.linalg.norm(embed_ours[tid]):.6f}")
        cos_emb = np.dot(h_hf.numpy(), embed_ours[tid]) / (torch.norm(h_hf).item() * np.linalg.norm(embed_ours[tid]) + 1e-12)
        print(f"Embedding Cosine Similarity: {cos_emb:.6f}")

        t0 = time.perf_counter()
        hf_layers = hf_exact_layer_outputs(model, prefix_ids)
        h_hf = hf_layers[-1]
        hf_time = time.perf_counter() - t0

        # Our forward with layer-by-layer tracing
        h_trace = np.zeros(N_LAYERS * HDIM, dtype=np.float32)
        lib.lko_runner_trace_layers(tid, pos, seq_len, N_LAYERS, h_trace.ctypes.data)
        ours_layers = h_trace.reshape(N_LAYERS, HDIM)

        print("\n=== Layer-by-layer comparison ===")
        for L in range(N_LAYERS):
            h_ours_L = ours_layers[L]
            h_hf_L = hf_layers[L]
            cos_L = np.dot(h_hf_L, h_ours_L) / (np.linalg.norm(h_hf_L) * np.linalg.norm(h_ours_L) + 1e-12)
            print(f"Layer {L:2d} (GQA={L in GQA_LAYERS}): cos={cos_L:.6f} | HF norm={np.linalg.norm(h_hf_L):.4f} Ours norm={np.linalg.norm(h_ours_L):.4f}")

        h_ours = ours_layers[-1]
        cos_val = np.dot(h_hf, h_ours) / (
            np.linalg.norm(h_hf) * np.linalg.norm(h_ours) + 1e-12)
        print(f"  cos(hf, ours) after 40L: {cos_val:.6f}")
        print(f"  HF norm: {np.linalg.norm(h_hf):.4f}  Ours norm: {np.linalg.norm(h_ours):.4f}")
        print(f"  HF time: {hf_time:.1f}s")

        # Also compare our forward without MoE for isolating the issue
        # (HF is also MoE-skipped in this test)

    # Final summary
    print("\n" + "=" * 60)
    print("If cos drops below 0.99 by first token, the forward pass has bugs.")
    print("If cos stays high for 1 token but drops across tokens, state handling is buggy.")
    print("=" * 60)


if __name__ == "__main__":
    main()
