"""Python bindings for the objeta Qwen3.6 MoE executor.

Loads the compiled Rust library and exposes the MoE dispatch C API.
"""

import ctypes
from pathlib import Path

_lib_name = "libobjeta_qwen36_executor.dylib"
_lib_path = None
for base in [
    Path(__file__).parent.parent / "target" / "release",
    Path(__file__).parent.parent / "target" / "debug",
]:
    candidate = base / _lib_name
    if candidate.exists():
        _lib_path = candidate
        break

if _lib_path:
    _lib = ctypes.cdll.LoadLibrary(str(_lib_path))
else:
    _lib = None

if _lib:
    _lib.lko_moe_forward_layer.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    _lib.lko_moe_forward_layer.restype = ctypes.c_int32
    _lib.lko_moe_init_freq_tracker.argtypes = [ctypes.c_int32]
    _lib.lko_moe_get_top_experts.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32]
    _lib.lko_moe_free_freq_tracker.argtypes = []


def get_lib():
    return _lib
