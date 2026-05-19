#!/usr/bin/env python3
"""Compare intermediate layer components (after_attn, post_norm, shared, moe, after_mlp)
between Rust (via FFI) and HF (via forward hooks) for a specific target layer.
"""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.qwen36_executor import get_lib

BIN = PROJECT / "models" / "qwen36_bin"
SNAP_ROOT = Path("/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots")
SNAPSHOT = str(sorted(os.listdir(SNAP_ROOT))[-1])
MODEL_PATH = str(SNAP_ROOT / SNAPSHOT)
HDIM = 2048

def init_runner():
    lib = get_lib()
    assert lib is not None, "Rust library not found"
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
    lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
    lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
    lib.lko_runner_step.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_step.restype = ctypes.c_int32
    
    lib.lko_runner_trace_layer_components.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.lko_runner_trace_layer_components.restype = ctypes.c_int32

    assert lib.lko_runner_init(str(BIN).encode(), 256), "runner init failed"
    lib.lko_runner_set_fusion_ratio(1.0)
    lib.lko_runner_set_moe_on_deltanet(1)
    return lib

def main():
    target_layer = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    prompt = "The capital of France is"
    
    print(f"Target Layer: {target_layer}")
    
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Loading HF Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    model.eval()
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat_text)
    seq_len = len(ids)
    print(f"Sequence length: {seq_len} tokens")
    print(f"Token IDs: {ids}")
    
    # Print model structure for target layer mlp to find exact submodule names
    print("\nTarget Layer Submodules:")
    layer = model.model.layers[target_layer]
    for name, module in layer.named_children():
        print(f"  {name}: {module.__class__.__name__}")
    
    if hasattr(layer, "mlp"):
        print("  mlp submodules:")
        for name, module in layer.mlp.named_children():
            print(f"    mlp.{name}: {module.__class__.__name__}")

    # Set up HF Hooks
    hf_comp = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            t = t.detach().float().cpu()
            if t.ndim == 3:
                val = t[0, -1].numpy()
            elif t.ndim == 2:
                val = t[-1].numpy()
            elif t.ndim == 1:
                val = t.numpy()
            else:
                val = t.numpy()
            hf_comp[name] = val
        return hook

    # Register hooks
    hooks = []
    
    # 1. h_after_attn: input to post_attention_layernorm (before norm)
    # The pre-hook on post_attention_layernorm gets the residual+attn output
    def pre_hook(module, input):
        t = input[0].detach().float().cpu()
        if t.ndim == 3:
            val = t[0, -1].numpy()
        elif t.ndim == 2:
            val = t[-1].numpy()
        else:
            val = t.numpy()
        hf_comp['h_after_attn'] = val
        return None
    hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(pre_hook))
    
    # 2. h_norm2: output of post_attention_layernorm
    hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook('h_norm2')))
    
    # 3. mlp_out: output of mlp
    hooks.append(layer.mlp.register_forward_hook(make_hook('mlp_out')))
    
    # 4. shared_expert_out: output of shared expert
    if hasattr(layer.mlp, "shared_expert"):
        hooks.append(layer.mlp.shared_expert.register_forward_hook(make_hook('shared_expert_out')))
    
    # 5. shared_expert_gate: output of shared expert gate (if exists as module)
    if hasattr(layer.mlp, "shared_expert_gate"):
        hooks.append(layer.mlp.shared_expert_gate.register_forward_hook(make_hook('shared_expert_gate_out')))

    # 6. layer_out: final layer output (input to the next layer)
    hooks.append(layer.register_forward_hook(make_hook('layer_out')))

    # 7. GQA internal debug hooks
    is_gqa = (target_layer % 4 == 3)
    if is_gqa:
        hooks.append(layer.input_layernorm.register_forward_hook(make_hook('h_norm1')))
        if hasattr(layer.self_attn, 'q_proj'):
            hooks.append(layer.self_attn.q_proj.register_forward_hook(make_hook('q_proj_out')))
        if hasattr(layer.self_attn, 'k_proj'):
            hooks.append(layer.self_attn.k_proj.register_forward_hook(make_hook('k_proj_out')))
        if hasattr(layer.self_attn, 'v_proj'):
            hooks.append(layer.self_attn.v_proj.register_forward_hook(make_hook('v_proj_out')))
        if hasattr(layer.self_attn, 'q_norm'):
            hooks.append(layer.self_attn.q_norm.register_forward_hook(make_hook('q_norm_out')))
        if hasattr(layer.self_attn, 'k_norm'):
            hooks.append(layer.self_attn.k_norm.register_forward_hook(make_hook('k_norm_out')))

    # Run HF Forward
    print("\nRunning HF Forward...")
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        model(input_ids=input_ids)
    
    # Clean up hooks
    for h in hooks:
        h.remove()
        
    print("HF Hooked components extracted:")
    for k, v in hf_comp.items():
        print(f"  {k}: norm={np.linalg.norm(v):.6f}, shape={v.shape}")

    # Now run Rust
    print("\nInitializing Rust runner...")
    lib = init_runner()
    
    # Prefill all but the last token
    print("Rust prefilling...")
    hn = np.zeros(HDIM, dtype=np.float32)
    idx = np.zeros(64, dtype=np.int32)
    val = np.zeros(64, dtype=np.float32)
    for pos in range(seq_len - 1):
        tid = ids[pos]
        lib.lko_runner_step(tid, pos, pos + 1, hn.ctypes.data, len(idx), idx.ctypes.data, val.ctypes.data)
        
    # On the last token, trace target layer components
    last_pos = seq_len - 1
    last_tid = ids[last_pos]
    print(f"Tracing Rust components for last token (pos={last_pos}, tid={last_tid})...")
    
    rust_after_attn = np.zeros(HDIM, dtype=np.float32)
    rust_norm2 = np.zeros(HDIM, dtype=np.float32)
    rust_shared = np.zeros(HDIM, dtype=np.float32)
    rust_moe = np.zeros(HDIM, dtype=np.float32)
    rust_after_mlp = np.zeros(HDIM, dtype=np.float32)
    
    ret = lib.lko_runner_trace_layer_components(
        last_tid, last_pos, last_pos + 1, target_layer,
        rust_after_attn.ctypes.data,
        rust_norm2.ctypes.data,
        rust_shared.ctypes.data,
        rust_moe.ctypes.data,
        rust_after_mlp.ctypes.data
    )
    assert ret == HDIM, f"trace_layer_components failed: {ret}"
    
    # Print comparison
    print("\n=== COMPONENT COMPARISON ===")
    
    # For Shared and MoE:
    # In HF, shared_expert_out might need the gate applied. Let's compute it.
    # In HF Qwen MoE, shared_expert_gate is: sigmoid(gate(x)) * shared_expert(x)
    # We can inspect the parameters or the hook results if it was a module.
    hf_shared = None
    if 'shared_expert_out' in hf_comp:
        # If shared_expert_gate exists and we hooked it:
        if 'shared_expert_gate_out' in hf_comp:
            gate_val = hf_comp['shared_expert_gate_out']
            # If it's a sigmoid gate, it's just element-wise multiply or single scalar
            # In Qwen: gate_w is (1, 2048), so output is (1,) or (1, 1).
            # Let's check size
            if gate_val.size == 1:
                sig_gate = 1.0 / (1.0 + np.exp(-float(gate_val.item())))
                hf_shared = hf_comp['shared_expert_out'] * sig_gate
            else:
                hf_shared = hf_comp['shared_expert_out'] * gate_val
        else:
            # Let's check if there is an explicit gate parameter we can compute
            # We can extract mlp.shared_expert_gate weight from the model
            gate_module = getattr(layer.mlp, "shared_expert_gate", None)
            if gate_module is not None:
                gate_w = gate_module.weight.detach().float().cpu().numpy() # shape (1, 2048)
                h_norm2_val = hf_comp['h_norm2']
                # dot product
                g_val = np.dot(gate_w.flatten(), h_norm2_val)
                sig_gate = 1.0 / (1.0 + np.exp(-g_val))
                hf_shared = hf_comp['shared_expert_out'] * sig_gate
                print(f"  [HF Calculated] shared_expert_gate value: {sig_gate:.6f}")
            else:
                # Fallback to raw output
                hf_shared = hf_comp['shared_expert_out']
                
    # In HF, mlp_out = gated_shared + moe_out
    hf_moe = None
    if hf_shared is not None and 'mlp_out' in hf_comp:
        hf_moe = hf_comp['mlp_out'] - hf_shared

    # Component list to compare
    comps = [
        ('h_after_attn', rust_after_attn, hf_comp.get('h_after_attn')),
        ('h_norm2', rust_norm2, hf_comp.get('h_norm2')),
        ('shared', rust_shared, hf_shared),
        ('moe', rust_moe, hf_moe),
        ('h_after_mlp', rust_after_mlp, hf_comp.get('layer_out')), # layer_out should match h_after_mlp
    ]
    
    print(f"{'Component':<15} {'Rust Norm':<10} {'HF Norm':<10} {'Cosine':<10}")
    print("-" * 50)
    for name, r_val, h_val in comps:
        if h_val is None:
            print(f"{name:<15} {np.linalg.norm(r_val):<10.4f} {'N/A':<10} {'N/A':<10}")
            continue
        
        r_norm = np.linalg.norm(r_val)
        h_norm = np.linalg.norm(h_val)
        dot = np.dot(r_val, h_val)
        cos = dot / (r_norm * h_norm + 1e-12)
        print(f"{name:<15} {r_norm:<10.4f} {h_norm:<10.4f} {cos:<10.6f}")

if __name__ == "__main__":
    main()
