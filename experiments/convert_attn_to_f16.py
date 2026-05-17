"""Convert layer_*_attn.npz to mmap-friendly fp16 binary format.

Each layer gets:
  layer_{l}_attn_f16.bin — concatenated fp16 weights
  layer_{l}_attn_f16.json — {name: [shape, offset_bytes, size_bytes]}
"""

import json, struct, sys
from pathlib import Path
import numpy as np

BIN = Path("runtime/moe/converted/qwen36_bin")

# Keys to include, in order (for deterministic layout)
INCLUDE_PATTERNS = [
    'input_layernorm.weight',
    'post_attention_layernorm.weight',
    'linear_attn.in_proj_qkv.weight',
    'linear_attn.in_proj_z.weight',
    'linear_attn.out_proj.weight',
    'linear_attn.A_log',
    'linear_attn.conv1d.weight',
    'linear_attn.dt_bias',
    'linear_attn.in_proj_a.weight',
    'linear_attn.in_proj_b.weight',
    'linear_attn.norm.weight',
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight',
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'self_attn.k_norm.weight',
    'self_attn.q_norm.weight',
    'mlp.shared_expert.gate_proj.weight',
    'mlp.shared_expert.up_proj.weight',
    'mlp.shared_expert.down_proj.weight',
    'mlp.shared_expert_gate.weight',
]

for layer_idx in range(40):
    npz_path = BIN / f"layer_{layer_idx}_attn.npz"
    if not npz_path.exists():
        print(f"  SKIP layer {layer_idx}: no npz")
        continue

    data = np.load(npz_path, allow_pickle=True)
    meta = {}
    offset = 0
    total_bytes = 0

    with open(BIN / f"layer_{layer_idx}_attn_f16.bin", "wb") as f:
        for pattern in INCLUDE_PATTERNS:
            if pattern not in data:
                continue
            arr = data[pattern].astype(np.float16)
            raw = arr.tobytes()
            f.write(raw)
            nbytes = len(raw)
            meta[pattern] = [list(arr.shape), offset, nbytes]
            offset += nbytes
            total_bytes += nbytes

    with open(BIN / f"layer_{layer_idx}_attn_f16.json", "w") as f:
        json.dump(meta, f)

    is_linear = 'linear_attn.in_proj_qkv.weight' in data
    atype = "linear" if is_linear else "full"
    print(f"  Layer {layer_idx} ({atype}): {total_bytes/1024/1024:.1f} MB f16 ({len(meta)} tensors)")

print(f"\nDone. Total disk: ~{40 * total_bytes / 1024 / 1024 / 1024:.1f} GB")
