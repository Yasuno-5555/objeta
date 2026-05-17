#!/usr/bin/env python3
"""Download Qwen3.6-35B-A3B shards incrementally, quantize, delete originals.

Strategy: process 1 shard at a time to stay within 51GB disk budget.
Peak usage per shard: ~4GB download + ~4GB bf16 + ~1GB q4 = ~9GB.
"""

import json, os, struct, sys, time
from pathlib import Path

CACHE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
OUTPUT = Path.home() / "projects/LKO/runtime/moe/converted/qwen36_q4"
N_SHARDS = 26

def get_shard_path(shard_idx):
    """Get the cached path for a shard, downloading if needed."""
    from huggingface_hub import hf_hub_download
    fname = f"model-{shard_idx+1:05d}-of-{N_SHARDS:05d}.safetensors"
    return hf_hub_download("Qwen/Qwen3.6-35B-A3B", fname)

def read_safetensors(path):
    """Read all tensors from a safetensors file. Returns dict of name -> numpy array."""
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    if '__metadata__' in header: del header['__metadata__']

    # Read all data into memory
    data = np.memmap(path, mode='r', dtype=np.uint8, offset=8+header_len)
    tensors = {}
    for key, info in header.items():
        start, end = info['data_offsets']
        shape = info['shape']
        dtype = info['dtype']
        arr = data[start:end]
        if dtype == 'BF16':
            raw = np.frombuffer(data[start:end], dtype=np.uint16)
            arr = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()
        elif dtype == 'F16':
            arr = arr.view(np.float16).reshape(shape).astype(np.float32)
        else:
            arr = arr.view(np.float32).reshape(shape)
        tensors[key] = arr.copy()
    return tensors

def quantize_rust(tensor, fmt="q4k_appl"):
    """Quantize using Rust SIMD quantizer. Returns bytes array + shape info."""
    import ctypes, numpy as np
    from runtime.executor import _lib

    # Ensure float32 contiguous
    w = np.array(tensor, dtype=np.float32, order='C')
    M, K = w.shape

    # Call Rust quantizer
    fn = getattr(_lib, f"lko_quantize_{fmt}")
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
                   ctypes.POINTER(ctypes.c_int64)]
    _lib.lko_free.argtypes = [ctypes.c_void_p]

    out_size = ctypes.c_int64(0)
    ptr = fn(w.ctypes.data_as(ctypes.c_void_p), M, K, ctypes.byref(out_size))

    if not ptr:
        raise RuntimeError(f"Rust quantizer {fmt} failed for shape ({M},{K})")

    total_bytes = out_size.value
    buf = (ctypes.c_uint8 * total_bytes).from_address(ptr)
    result = bytes(buf)  # Copy to Python bytes
    _lib.lko_free(ptr)

    # Compute block layout
    if fmt == "q4k_appl":
        block_bytes = 160
        block_size = 256
    elif fmt == "q4k_appl_v2":
        block_bytes = 144
        block_size = 256
    elif fmt == "q40":
        block_bytes = 18
        block_size = 32
    else:
        block_bytes = 160
        block_size = 256

    num_blocks = (K + block_size - 1) // block_size
    return result, M, K, num_blocks, block_bytes

def main():
    print("=" * 70)
    print("  Qwen3.6-35B-A3B: Incremental Download + Quantize")
    print(f"  Output: {OUTPUT}")
    print("=" * 70)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Count what's already done
    done_shards = set()
    for f in OUTPUT.glob("shard_*_layers_*.npz"):
        # shard_0_layers_0-1.npz
        parts = f.stem.split('_')
        done_shards.add(int(parts[1]))

    all_expert_sizes = {}  # layer_idx -> size MB

    for shard_idx in range(N_SHARDS):
        if shard_idx in done_shards:
            print(f"\nShard {shard_idx+1}/{N_SHARDS}: already processed, skipping")
            continue

        print(f"\n── Shard {shard_idx+1}/{N_SHARDS} ──")
        t0 = time.perf_counter()

        # 1. Download
        print("  Downloading...", end=" ", flush=True)
        path = get_shard_path(shard_idx)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"{size_mb:.0f}MB")

        # 2. Read tensors
        print("  Reading...", end=" ", flush=True)
        tensors = read_safetensors(path)
        print(f"{len(tensors)} tensors")

        # 3. Identify layers in this shard
        layers = set()
        for key in tensors:
            if 'layers.' in key:
                parts = key.split('layers.')
                layer_str = parts[1].split('.')[0]
                if layer_str.isdigit():
                    layers.add(int(layer_str))
        layers = sorted(layers)
        print(f"  Layers: {layers}")

        # 4. Quantize expert weights (the bulk)
        expert_keys = [k for k in tensors if 'mlp.experts' in k]
        non_expert_keys = [k for k in tensors if k not in expert_keys]

        q4_tensors = {}
        for key in expert_keys:
            t = tensors[key]
            orig_shape = t.shape
            # Flatten to 2D for quantizer: all dims except last → rows, last dim → cols
            if t.ndim > 2:
                K = t.shape[-1]
                M = int(np.prod(t.shape[:-1]))
                t_2d = t.reshape(M, K)
            elif t.ndim == 2:
                M, K = t.shape
                t_2d = t
            else:
                # Skip 1D tensors (biases, norms) — keep as fp32
                q4_tensors[key] = {"packed": t, "M": 1, "K": len(t), "num_blocks": 1, "block_bytes": 0}
                del tensors[key]
                continue

            packed_bytes, out_M, out_K, num_blocks, block_bytes = quantize_rust(t_2d, "q4k_appl")
            q4_tensors[key] = {
                "packed": np.frombuffer(packed_bytes, dtype=np.uint8),
                "M": M, "K": K, "orig_shape": orig_shape,
                "num_blocks": num_blocks, "block_bytes": block_bytes,
            }
            del tensors[key]  # Free memory

        # 5. Save quantized layer data
        out_file = OUTPUT / f"shard_{shard_idx}_layers_{'-'.join(map(str, layers))}.npz"
        save_data = {"layers": layers, "shard_idx": shard_idx}

        for key, qdata in q4_tensors.items():
            save_data[f"{key}__packed"] = qdata["packed"]
            save_data[f"{key}__M"] = np.int32(qdata["M"])
            save_data[f"{key}__K"] = np.int32(qdata["K"])
            save_data[f"{key}__num_blocks"] = np.int32(qdata["num_blocks"])
            save_data[f"{key}__block_bytes"] = np.int32(qdata["block_bytes"])

        # Also save non-expert tensors (small)
        for key in non_expert_keys:
            save_data[key] = tensors[key]

        np.savez_compressed(out_file, **save_data)
        out_size = os.path.getsize(out_file) / 1024 / 1024
        print(f"  Saved: {out_file.name} ({out_size:.0f}MB)")

        # 6. Track expert sizes
        for layer_idx in layers:
            gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
            if gate_key in q4_tensors:
                mb = q4_tensors[gate_key]["packed"].nbytes / 1024 / 1024
                all_expert_sizes[layer_idx] = (mb, q4_tensors[gate_key]["M"], q4_tensors[gate_key]["K"])

        # 7. Delete original shard
        os.remove(path)
        print(f"  Deleted original shard ({size_mb:.0f}MB freed)")

        elapsed = time.perf_counter() - t0
        print(f"  Time: {elapsed:.0f}s")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Complete!")
    print(f"{'=' * 70}")
    output_size = sum(f.stat().st_size for f in OUTPUT.glob("*.npz")) / 1024 / 1024
    print(f"  Total q4 size: {output_size:.0f}MB")
    print(f"  Layers processed: {len(all_expert_sizes)}")


if __name__ == "__main__":
    import numpy as np
    main()
