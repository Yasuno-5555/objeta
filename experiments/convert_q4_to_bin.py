#!/usr/bin/env python3
"""Convert Qwen3.6 q4 npz → flat binary per layer (clean, complete).

Output per layer:
  layer_{l}_gate_up.bin  — raw Q4_K_APPL bytes (mmap)
  layer_{l}_down.bin     — raw Q4_K_APPL bytes (mmap)
  layer_{l}_router.bin   — f32 [256, 2048]
  layer_{l}_attn.npz     — attention/norm/shared weights (SHORT keys, no prefix)
  embed_tokens.bin       — f32 [248320, 2048]
  final_norm.bin         — f32 [2048]
"""

import numpy as np
from pathlib import Path

SRC = Path("runtime/moe/converted/qwen36_q4")
DST = Path("runtime/moe/converted/qwen36_bin")
DST.mkdir(parents=True, exist_ok=True)

# Per-layer accumulators (built lazily across shards)
layer_q4 = {l: {} for l in range(40)}  # layer → {gate_up_packed, down_packed}
layer_attn = {l: {} for l in range(40)}  # layer → {short_key: array}
layer_router = {}  # layer → router array (takes first occurrence)

embedding = None
final_norm = None

for shard_path in sorted(SRC.glob("*.npz")):
    print(f"Reading {shard_path.name} ({shard_path.stat().st_size/1024/1024:.0f}MB)...")
    npz = np.load(shard_path, allow_pickle=True)

    for key in npz.keys():
        if key == 'layers' or key == 'shard_idx':
            continue

        # ── Q4 packed tensors ──
        if '__packed' in key:
            parts = key.split('.')
            if 'layers' not in parts: continue
            try:
                l = int(parts[parts.index('layers') + 1])
            except (ValueError, IndexError):
                continue
            if l >= 40: continue
            raw = bytes(npz[key])
            if 'gate_up_proj' in key:
                layer_q4[l]['gate_up'] = raw
            elif 'down_proj' in key:
                layer_q4[l]['down'] = raw
            continue

        # ── Router weights ──
        if 'mlp.gate.weight' in key and 'expert' not in key and 'shared' not in key:
            parts = key.split('.')
            try:
                l = int(parts[parts.index('layers') + 1])
            except (ValueError, IndexError):
                continue
            if l >= 40: continue
            if l not in layer_router:
                layer_router[l] = npz[key]
            continue

        # ── Embedding / final norm ──
        if 'embed_tokens' in key:
            embedding = npz[key]
            continue
        if key == 'model.language_model.norm.weight':
            final_norm = npz[key]
            continue

        # ── Attention / norm / shared expert (all small) ──
        pfx = "model.language_model.layers."
        if pfx in key:
            rest = key.split(pfx, 1)[-1]
            parts = rest.split('.')
            try:
                l = int(parts[0])
            except ValueError:
                continue
            if l >= 40: continue
            short_key = rest[len(str(l)) + 1:]  # strip "L."
            layer_attn[l][short_key] = npz[key]

    del npz

# ── Write binary files ──
print("\nWriting binary files...")
total_mb = 0
for l in range(40):
    if 'gate_up' in layer_q4[l]:
        raw = layer_q4[l]['gate_up']
        with open(DST / f"layer_{l}_gate_up.bin", "wb") as f:
            f.write(raw)
        total_mb += len(raw) / 1024 / 1024

    if 'down' in layer_q4[l]:
        raw = layer_q4[l]['down']
        with open(DST / f"layer_{l}_down.bin", "wb") as f:
            f.write(raw)
        total_mb += len(raw) / 1024 / 1024

    if l in layer_router:
        layer_router[l].astype(np.float32).tofile(DST / f"layer_{l}_router.bin")
        total_mb += layer_router[l].nbytes / 1024 / 1024

    if layer_attn[l]:
        np.savez_compressed(DST / f"layer_{l}_attn.npz", **layer_attn[l])

    if l % 10 == 0:
        print(f"  Layer {l} done ({len(layer_attn[l])} attn keys)")

if embedding is not None:
    embedding.astype(np.float32).tofile(DST / "embed_tokens.bin")
if final_norm is not None:
    final_norm.astype(np.float32).tofile(DST / "final_norm.bin")

n_files = len(list(DST.glob("*")))
print(f"\nDone! {n_files} files, {total_mb:.0f}MB binary + attn npz files")
print(f"Sample layer 0 attn keys:")
for k in sorted(layer_attn[0].keys()):
    print(f"  {k}: {layer_attn[0][k].shape}")
