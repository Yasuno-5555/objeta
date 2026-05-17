"""Python bindings for the LKO Rust executor.

Loads `liblko_executor.dylib` and exposes C API via ctypes.

Usage:
    from runtime.executor import LKOExecutor
    exec = LKOExecutor()
    exec.register_buffer("wq", wq_ptr, wq_size)
    exec.set_config(hidden_dim=2048, n_heads=32, ...)
    exec.run_layer(x_ptr, ...)  # Must be implemented
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx


# Load the shared library
_lib_path = Path(__file__).parent / "target" / "debug" / "liblko_executor.dylib"
if not _lib_path.exists():
    _lib_path = Path(__file__).parent / "target" / "release" / "liblko_executor.dylib"

if _lib_path.exists():
    _lib = ctypes.cdll.LoadLibrary(str(_lib_path))
else:
    _lib = None


class LayerConfig(ctypes.Structure):
    _fields_ = [
        ("hidden_dim", ctypes.c_int32),
        ("ffn_dim", ctypes.c_int32),
        ("n_heads", ctypes.c_int32),
        ("n_kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("norm_eps", ctypes.c_float),
    ]


class LKOExecutor:
    """Python wrapper around the Rust LKO executor (via ctypes)."""

    def __init__(self):
        if _lib is None:
            raise RuntimeError(
                "liblko_executor.dylib not found. Build with: "
                "cd runtime/executor && cargo build"
            )
        # Set up function signatures
        _lib.lko_executor_create.restype = ctypes.c_void_p
        _lib.lko_executor_create.argtypes = []
        _lib.lko_executor_destroy.argtypes = [ctypes.c_void_p]
        _lib.lko_executor_register_buffer.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64,
        ]
        _lib.lko_executor_register_buffer.restype = ctypes.c_int32
        _lib.lko_executor_set_config.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(LayerConfig),
        ]
        _lib.lko_executor_set_config.restype = ctypes.c_int32

        self._handle = _lib.lko_executor_create()
        if not self._handle:
            raise RuntimeError("Failed to create LKO executor")

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.lko_executor_destroy(self._handle)

    def register_buffer(self, name: str, ptr: int, size_bytes: int):
        """Register a GPU weight buffer.

        Args:
            name: Identifier (e.g. "wq", "wk", "wv").
            ptr: GPU device pointer from MLX.
            size_bytes: Buffer size in bytes.
        """
        _lib.lko_executor_register_buffer(
            self._handle, name.encode(), ptr, size_bytes)

    def set_config(self, **kwargs):
        """Set layer configuration from keyword args."""
        config = LayerConfig(
            hidden_dim=kwargs.get("hidden_dim", 4096),
            ffn_dim=kwargs.get("ffn_dim", 11008),
            n_heads=kwargs.get("n_heads", 32),
            n_kv_heads=kwargs.get("n_kv_heads", 8),
            head_dim=kwargs.get("head_dim", 128),
            norm_eps=kwargs.get("norm_eps", 1e-5),
        )
        _lib.lko_executor_set_config(self._handle, ctypes.byref(config))


def executor_available() -> bool:
    """Check if the Rust executor shared library is available."""
    return _lib is not None


def quantize_q4k_appl_rust(w: mx.array) -> mx.array:
    """Quantize f32 matrix to Q4_K_APPL using Rust SIMD.

    Args:
        w: float32 matrix of shape (M, K).

    Returns:
        uint8 array of shape (M, num_blocks, 160).
    """
    if _lib is None:
        raise RuntimeError("Rust executor not built")

    import ctypes
    import numpy as np

    w_np = np.array(w.astype(mx.float32), dtype=np.float32, order='C')
    M, K = w_np.shape

    _lib.lko_quantize_q4k_appl.restype = ctypes.c_void_p
    _lib.lko_quantize_q4k_appl.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64),
    ]
    _lib.lko_free.argtypes = [ctypes.c_void_p]

    out_size = ctypes.c_int64(0)
    ptr = _lib.lko_quantize_q4k_appl(
        w_np.ctypes.data_as(ctypes.c_void_p),
        M, K, ctypes.byref(out_size),
    )
    if not ptr:
        raise RuntimeError("Rust quantizer failed")

    total_bytes = out_size.value
    buf = (ctypes.c_uint8 * total_bytes).from_address(ptr)
    result = np.frombuffer(buf, dtype=np.uint8).copy()
    _lib.lko_free(ptr)

    num_blocks = (K + 255) // 256
    return mx.array(result.reshape(M, num_blocks, 160))


def quantize_q5k_appl_rust(w: mx.array) -> mx.array:
    """Quantize f32 matrix to Q5_K_APPL using Rust SIMD.

    Args:
        w: float32 matrix of shape (M, K).

    Returns:
        uint8 array of shape (M, num_blocks, 192).
    """
    if _lib is None:
        raise RuntimeError("Rust executor not built")
    import ctypes, numpy as np
    w_np = np.array(w.astype(mx.float32), dtype=np.float32, order='C')
    M, K = w_np.shape
    _lib.lko_quantize_q5k_appl.restype = ctypes.c_void_p
    _lib.lko_quantize_q5k_appl.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64),
    ]
    _lib.lko_free.argtypes = [ctypes.c_void_p]
    out_size = ctypes.c_int64(0)
    ptr = _lib.lko_quantize_q5k_appl(
        w_np.ctypes.data_as(ctypes.c_void_p), M, K, ctypes.byref(out_size))
    if not ptr:
        raise RuntimeError("Rust Q5_K_APPL quantizer failed")
    total_bytes = out_size.value
    buf = (ctypes.c_uint8 * total_bytes).from_address(ptr)
    result = np.frombuffer(buf, dtype=np.uint8).copy()
    _lib.lko_free(ptr)
    num_blocks = (K + 255) // 256
    return mx.array(result.reshape(M, num_blocks, 192))


def quantize_q40_rust(w: mx.array) -> mx.array:
    """Quantize f32 matrix to Q4_0 using Rust SIMD.

    Args:
        w: float32 matrix of shape (M, K).

    Returns:
        uint8 array of shape (M, num_blocks, 18).
    """
    if _lib is None:
        raise RuntimeError("Rust executor not built")
    import ctypes, numpy as np
    w_np = np.array(w.astype(mx.float32), dtype=np.float32, order='C')
    M, K = w_np.shape
    _lib.lko_quantize_q40.restype = ctypes.c_void_p
    _lib.lko_quantize_q40.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64),
    ]
    _lib.lko_free.argtypes = [ctypes.c_void_p]
    out_size = ctypes.c_int64(0)
    ptr = _lib.lko_quantize_q40(
        w_np.ctypes.data_as(ctypes.c_void_p), M, K, ctypes.byref(out_size))
    if not ptr:
        raise RuntimeError("Rust quantizer failed")
    total_bytes = out_size.value
    buf = (ctypes.c_uint8 * total_bytes).from_address(ptr)
    result = np.frombuffer(buf, dtype=np.uint8).copy()
    _lib.lko_free(ptr)
    num_blocks = (K + 31) // 32
    shape = (M * num_blocks * 18,)
    return mx.array(result.reshape(M, num_blocks, 18))


def quantize_bulk(matrices: list, fmt: str = "q4k_appl") -> list:
    """Quantize multiple matrices, using Rust bulk or per-matrix SIMD.

    Args:
        matrices: list of (name, mx.array) tuples.
        fmt: "q40", "q4k_appl", "q4k_appl_v2", or "q5k_appl".

    Returns:
        list of (name, mx.array) with quantized weights.
    """
    if _lib is None:
        raise RuntimeError("Rust executor not built")
    import ctypes, numpy as np

    # Format parameters: (bytes_per_block, block_size, is_bulk)
    _fmt_params = {"q40": (18, 32, False),
                   "q4k_appl": (160, 256, True),
                   "q4k_appl_v2": (144, 256, True),
                   "q5k_appl": (192, 256, False)}
    bpb, block, is_bulk = _fmt_params.get(fmt, (160, 256, True))

    if is_bulk:
        # ── Bulk path: single C call for all matrices ──────────────
        all_f32 = []
        sizes = []
        for name, w in matrices:
            w_f32 = np.array(w.astype(mx.float32), dtype=np.float32, order='C').reshape(-1)
            all_f32.append(w_f32)
            sizes.extend([w.shape[0], w.shape[1]])

        concat = np.concatenate(all_f32)
        sizes_arr = (ctypes.c_int32 * len(sizes))(*sizes)

        fn_map = {"q4k_appl": "lko_quantize_q4k_appl_bulk",
                  "q4k_appl_v2": "lko_quantize_q4k_appl_v2_bulk"}
        quant_fn = getattr(_lib, fn_map[fmt])
        quant_fn.restype = ctypes.c_void_p
        quant_fn.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
        _lib.lko_free.argtypes = [ctypes.c_void_p]

        out_size = ctypes.c_int64(0)
        ptr = quant_fn(len(matrices), sizes_arr, concat.ctypes.data_as(ctypes.c_void_p),
                       ctypes.byref(out_size))
        if not ptr:
            raise RuntimeError(f"Rust bulk quantizer failed ({fmt})")

        buf = (ctypes.c_uint8 * out_size.value).from_address(ptr)
        flat_out = np.frombuffer(buf, dtype=np.uint8).copy()
        _lib.lko_free(ptr)

        result = []
        offset = 0
        for name, w in matrices:
            M, K = w.shape
            nb = (K + block - 1) // block
            nbytes = M * nb * bpb
            chunk = flat_out[offset:offset + nbytes].reshape(M, nb, bpb)
            result.append((name, mx.array(chunk)))
            offset += nbytes
        return result
    else:
        # ── Per-matrix path: call single-matrix C function in loop ─
        fn_map = {"q40": "lko_quantize_q40",
                  "q5k_appl": "lko_quantize_q5k_appl"}
        quant_fn = getattr(_lib, fn_map[fmt])
        quant_fn.restype = ctypes.c_void_p
        quant_fn.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
                             ctypes.POINTER(ctypes.c_int64)]
        _lib.lko_free.argtypes = [ctypes.c_void_p]

        result = []
        for name, w in matrices:
            w_f32 = np.array(w.astype(mx.float32), dtype=np.float32, order='C')
            M, K = w_f32.shape

            out_size = ctypes.c_int64(0)
            ptr = quant_fn(w_f32.ctypes.data_as(ctypes.c_void_p), M, K,
                           ctypes.byref(out_size))
            if not ptr:
                raise RuntimeError(f"Rust quantizer failed for {name}")

            buf = (ctypes.c_uint8 * out_size.value).from_address(ptr)
            arr = np.frombuffer(buf, dtype=np.uint8).copy()
            _lib.lko_free(ptr)

            nb = (K + block - 1) // block
            result.append((name, mx.array(arr.reshape(M, nb, bpb))))

        return result


def get_mlx_ptr(arr) -> int:
    """Get the GPU device pointer from an MLX array.

    On Apple Silicon (M1-M4), Metal buffers are in unified memory.
    np.array(..., copy=False) shares the Metal buffer directly,
    and .ctypes.data gives us the CPU-accessible pointer.

    Args:
        arr: An mlx.core.array that has been eval'd.

    Returns:
        GPU device pointer (also CPU-accessible) as int.
    """
    import numpy as np
    np_arr = np.array(arr, copy=False)
    return np_arr.ctypes.data


# ══════════════════════════════════════════════════════════════════
# KV Arena (Rust-backed)
# ══════════════════════════════════════════════════════════════════

class LKOKVArena:
    """Python wrapper around Rust KV Arena.

    Manages pre-allocated Metal buffers for K/V cache, with
    direct CPU memcpy writes that bypass MLX graph accumulation.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, max_seq_len: int):
        if _lib is None:
            raise RuntimeError("liblko_executor.dylib not found. Build it first.")

        _lib.lko_kv_arena_create.restype = ctypes.c_void_p
        _lib.lko_kv_arena_create.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        _lib.lko_kv_arena_destroy.argtypes = [ctypes.c_void_p]

        _lib.lko_kv_arena_register_k.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint64]
        _lib.lko_kv_arena_register_v.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint64]
        _lib.lko_kv_arena_is_ready.restype = ctypes.c_int32
        _lib.lko_kv_arena_is_ready.argtypes = [ctypes.c_void_p]

        _lib.lko_kv_arena_write.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p]

        _lib.lko_kv_arena_get_k_slice.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
        _lib.lko_kv_arena_get_v_slice.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]

        _lib.lko_kv_arena_copy_layer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        _lib.lko_kv_arena_copy_all.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]

        _lib.lko_kv_arena_set_seq_len.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        _lib.lko_kv_arena_get_seq_len.restype = ctypes.c_int32
        _lib.lko_kv_arena_get_seq_len.argtypes = [ctypes.c_void_p]

        _lib.lko_kv_arena_zero_active.argtypes = [ctypes.c_void_p]
        _lib.lko_kv_arena_zero_all.argtypes = [ctypes.c_void_p]

        self._handle = _lib.lko_kv_arena_create(n_layers, n_kv_heads, head_dim, max_seq_len)
        if not self._handle:
            raise RuntimeError("Failed to create LKO KV Arena")
        self._n_layers = n_layers
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and _lib:
            _lib.lko_kv_arena_destroy(self._handle)

    def register_from_mlx_arrays(self, k_caches: list, v_caches: list):
        """Register K/V buffers from pre-allocated MLX arrays.

        Args:
            k_caches: list of mx.array, shape (n_kv_heads, max_seq_len, head_dim), dtype=float16.
            v_caches: list of mx.array, same shape.
        """
        import numpy as np
        for layer_idx, (kc, vc) in enumerate(zip(k_caches, v_caches)):
            k_ptr = np.array(kc, copy=False).ctypes.data
            v_ptr = np.array(vc, copy=False).ctypes.data
            _lib.lko_kv_arena_register_k(self._handle, layer_idx, k_ptr)
            _lib.lko_kv_arena_register_v(self._handle, layer_idx, v_ptr)

    def is_ready(self) -> bool:
        return bool(_lib.lko_kv_arena_is_ready(self._handle))

    def write_kv(self, layer_idx: int, position: int, k_data, v_data):
        """Write K/V to the arena at position.

        Args:
            layer_idx: Layer index.
            position: KV cache position.
            k_data: numpy array or mx.array of shape (n_kv_heads, head_dim), dtype=float16.
            v_data: same.
        """
        import numpy as np
        k_np = np.array(k_data, copy=False)
        v_np = np.array(v_data, copy=False)
        _lib.lko_kv_arena_write(
            self._handle, layer_idx, position,
            k_np.ctypes.data_as(ctypes.c_void_p),
            v_np.ctypes.data_as(ctypes.c_void_p))

    def copy_from(self, src_arena: "LKOKVArena", up_to_position: int):
        """Copy all layers from src arena into self."""
        _lib.lko_kv_arena_copy_all(
            self._handle, src_arena._handle, up_to_position)

    def copy_layer_from(self, src_arena: "LKOKVArena", layer_idx: int, up_to_position: int):
        """Copy a single layer from src arena into self."""
        _lib.lko_kv_arena_copy_layer(
            self._handle, src_arena._handle, layer_idx, up_to_position)

    def set_seq_len(self, seq_len: int):
        _lib.lko_kv_arena_set_seq_len(self._handle, seq_len)

    @property
    def seq_len(self) -> int:
        return _lib.lko_kv_arena_get_seq_len(self._handle)

    def zero_active(self):
        """Zero the active region (up to seq_len) of all buffers."""
        _lib.lko_kv_arena_zero_active(self._handle)

    def zero_all(self):
        """Zero all buffers entirely."""
        _lib.lko_kv_arena_zero_all(self._handle)


# ══════════════════════════════════════════════════════════════════
# Speculative State Machine (Rust-backed)
# ══════════════════════════════════════════════════════════════════

class LKOSpecState:
    """Python wrapper around Rust speculative decode state machine."""

    def __init__(self, gamma: int = 4, vocab_size: int = 32000):
        if _lib is None:
            raise RuntimeError("liblko_executor.dylib not found. Build it first.")

        _lib.lko_spec_create.restype = ctypes.c_void_p
        _lib.lko_spec_create.argtypes = [ctypes.c_int32, ctypes.c_int32]
        _lib.lko_spec_destroy.argtypes = [ctypes.c_void_p]

        _lib.lko_spec_begin_prefill.argtypes = [ctypes.c_void_p]
        _lib.lko_spec_prefill_position.restype = ctypes.c_uint32
        _lib.lko_spec_prefill_position.argtypes = [ctypes.c_void_p]
        _lib.lko_spec_prefill_done.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        _lib.lko_spec_begin_draft.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        _lib.lko_spec_draft_token.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        _lib.lko_spec_draft_done.argtypes = [ctypes.c_void_p]

        _lib.lko_spec_get_verify.restype = ctypes.c_uint32
        _lib.lko_spec_get_verify.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]

        _lib.lko_spec_accept_reject.restype = ctypes.c_uint32
        _lib.lko_spec_accept_reject.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32)]

        _lib.lko_spec_current_token.restype = ctypes.c_uint32
        _lib.lko_spec_current_token.argtypes = [ctypes.c_void_p]
        _lib.lko_spec_position.restype = ctypes.c_uint32
        _lib.lko_spec_position.argtypes = [ctypes.c_void_p]
        _lib.lko_spec_is_done.restype = ctypes.c_int32
        _lib.lko_spec_is_done.argtypes = [ctypes.c_void_p]
        _lib.lko_spec_accept_rate.restype = ctypes.c_int32
        _lib.lko_spec_accept_rate.argtypes = [ctypes.c_void_p]

        self._handle = _lib.lko_spec_create(gamma, vocab_size)
        if not self._handle:
            raise RuntimeError("Failed to create LKO spec state")
        self._gamma = gamma

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and _lib:
            _lib.lko_spec_destroy(self._handle)

    def begin_prefill(self):
        _lib.lko_spec_begin_prefill(self._handle)

    def prefill_position(self) -> int:
        return _lib.lko_spec_prefill_position(self._handle)

    def prefill_done(self, first_token: int):
        _lib.lko_spec_prefill_done(self._handle, first_token)

    def begin_draft(self) -> tuple[int, int]:
        token = ctypes.c_uint32(0)
        pos = ctypes.c_uint32(0)
        _lib.lko_spec_begin_draft(self._handle, ctypes.byref(token), ctypes.byref(pos))
        return token.value, pos.value

    def draft_token(self, index: int, token: int):
        _lib.lko_spec_draft_token(self._handle, index, token)

    def draft_done(self):
        _lib.lko_spec_draft_done(self._handle)

    def get_verify(self) -> tuple[list[int], int]:
        buf = (ctypes.c_uint32 * self._gamma)()
        pos = ctypes.c_uint32(0)
        count = _lib.lko_spec_get_verify(self._handle, buf, ctypes.byref(pos))
        return list(buf[:count]), pos.value

    def accept_reject(self, target_logits, prefill_logits) -> tuple[int, int, list[int]]:
        import numpy as np
        t_np = np.array(target_logits, dtype=np.float32, order='C')
        p_np = np.array(prefill_logits, dtype=np.float32, order='C')
        accepted_buf = (ctypes.c_uint32 * (self._gamma + 1))()
        result = _lib.lko_spec_accept_reject(
            self._handle,
            t_np.ctypes.data_as(ctypes.c_void_p),
            p_np.ctypes.data_as(ctypes.c_void_p),
            accepted_buf)
        return result >> 16, result & 0xFFFF, list(accepted_buf[:result & 0xFFFF])

    def current_token(self) -> int:
        return _lib.lko_spec_current_token(self._handle)

    def position(self) -> int:
        return _lib.lko_spec_position(self._handle)

    def is_done(self) -> bool:
        return bool(_lib.lko_spec_is_done(self._handle))

    def accept_rate(self) -> float:
        return _lib.lko_spec_accept_rate(self._handle) / 100.0


# ══════════════════════════════════════════════════════════════════════
# Expert Store — per-layer expert weight offloading
# ══════════════════════════════════════════════════════════════════════

if _lib is not None:
    _lib.expert_store_create.restype = ctypes.c_void_p
    _lib.expert_store_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    _lib.expert_store_destroy.argtypes = [ctypes.c_void_p]
    _lib.expert_store_register_layer.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint64]
    _lib.expert_store_register_layer.restype = ctypes.c_int32
    _lib.expert_store_load_layer.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
    _lib.expert_store_load_layer.restype = ctypes.c_int32
    _lib.expert_store_get_buffer_ptr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _lib.expert_store_get_buffer_ptr.restype = ctypes.c_uint64
    _lib.expert_store_get_buffer_size.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _lib.expert_store_get_buffer_size.restype = ctypes.c_uint64
    _lib.expert_store_prefetch_async.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
    _lib.expert_store_prefetch_async.restype = ctypes.c_int32
    _lib.expert_store_prefetch_done.argtypes = [ctypes.c_void_p]
    _lib.expert_store_prefetch_done.restype = ctypes.c_int32
    _lib.expert_store_prefetch_wait.argtypes = [ctypes.c_void_p]
    _lib.expert_store_unload_layer.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _lib.expert_store_total_ssd_bytes.argtypes = [ctypes.c_void_p]
    _lib.expert_store_total_ssd_bytes.restype = ctypes.c_uint64


class ExpertStore:
    """Python wrapper around Rust ExpertStore for per-layer expert offloading."""

    def __init__(self, ssd_dir: str, n_layers: int):
        if _lib is None:
            raise RuntimeError("Rust executor not built. Run: cd runtime/executor && cargo build")
        self._handle = _lib.expert_store_create(
            ssd_dir.encode('utf-8'), n_layers)
        self._n_layers = n_layers

    def register_layer(self, layer_idx: int, filepath: str, size_bytes: int):
        _lib.expert_store_register_layer(
            self._handle, layer_idx, filepath.encode('utf-8'), size_bytes)

    def load_layer(self, layer_idx: int, metal_buffer_ptr: int) -> bool:
        return _lib.expert_store_load_layer(
            self._handle, layer_idx, metal_buffer_ptr) == 0

    def get_buffer_ptr(self, layer_idx: int) -> int:
        return _lib.expert_store_get_buffer_ptr(self._handle, layer_idx)

    def get_buffer_size(self, layer_idx: int) -> int:
        return _lib.expert_store_get_buffer_size(self._handle, layer_idx)

    def prefetch_async(self, layer_idx: int, metal_buffer_ptr: int) -> bool:
        return _lib.expert_store_prefetch_async(
            self._handle, layer_idx, metal_buffer_ptr) == 0

    def prefetch_done(self) -> bool:
        return bool(_lib.expert_store_prefetch_done(self._handle))

    def prefetch_wait(self):
        _lib.expert_store_prefetch_wait(self._handle)

    def unload_layer(self, layer_idx: int):
        _lib.expert_store_unload_layer(self._handle, layer_idx)

    def total_ssd_bytes(self) -> int:
        return _lib.expert_store_total_ssd_bytes(self._handle)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle is not None:
            _lib.expert_store_destroy(self._handle)
            self._handle = None
