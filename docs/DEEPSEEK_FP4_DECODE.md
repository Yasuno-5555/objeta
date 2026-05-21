# DeepSeek FP4 Expert Decode Semantics

Confirmed from DeepSeek V4 Flash inference source code (2026-05-21).

## Source Evidence

- **`inference/kernel.py`**: `FP4 = "float4_e2m1fn"`, `FE8M0 = "float8_e8m0fnu"`, `fp4_block_size = 32`
- **`inference/convert.py`**: `FP4_TABLE` lookup table, `cast_e2m1fn_to_e4m3fn` with packing logic
- **`inference/model.py`**: `Linear.__init__` showing shape layout and scale attachment

## FP4 Numeric Format: E2M1FN

1 sign bit, 2 exponent bits, 1 mantissa bit. Bias = 1.

### Decode Table

| Bits (s,e2,e1,m0) | Value |
|---|---|
| 0b0000 | 0.0 |
| 0b0001 | 0.5 |
| 0b0010 | 1.0 |
| 0b0011 | 1.5 |
| 0b0100 | 2.0 |
| 0b0101 | 3.0 |
| 0b0110 | 4.0 |
| 0b0111 | 6.0 |
| 0b1000 | 0.0 (neg zero) |
| 0b1001 | -0.5 |
| 0b1010 | -1.0 |
| 0b1011 | -1.5 |
| 0b1100 | -2.0 |
| 0b1101 | -3.0 |
| 0b1110 | -4.0 |
| 0b1111 | -6.0 |

### Formula

- Normal (e > 0): `(-1)^s * 2^(e-1) * (1 + m/2)`
- Subnormal (e = 0): `(-1)^s * 2^0 * (0 + m/2)`

## Packing Order

Two FP4 values per I8 byte, packed along the last (K/input) dimension.

- **Low nibble** (bits 0-3): first FP4 value (lower column index)
- **High nibble** (bits 4-7): second FP4 value (higher column index)

From `convert.py`:
```python
low = x & 0x0F
high = (x >> 4) & 0x0F
x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)
```

## F8_E8M0 Scale Format

Unsigned 8-bit exponent with 0 mantissa bits. No sign bit.

- **Decode**: `value = 2^(raw_byte - 127)`
- **Bias**: 127 (IEEE 754 FP32 exponent bias)
- **Range**: 2^(-127) ≈ 5.9e-39 to 2^(128) ≈ 3.4e+38

From `kernel.py`:
```python
def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)
```

## Block Size

**32 logical FP4 elements per scale value** (confirmed: `fp4_block_size = 32`).

This means 16 I8 bytes share one F8_E8M0 scale value (32/2 = 16).

## Scale Application

For weight matrix `W` with logical shape `[out_features, in_features]` and scale `S` with shape `[out_features, in_features / 32]`:

```
W_dequant[i, j] = FP4_TABLE[W_packed[i, j // 2].nibble(j % 2)] * 2^(S[i, j // 32] - 127)
```

Where `nibble(0)` extracts low 4 bits and `nibble(1)` extracts high 4 bits.

## Matrix Orientation

- Weights stored row-major: `[out_features, in_features // 2]` in I8
- Scales stored row-major: `[out_features, in_features // 32]` in F8_E8M0
- Logical shape: `[out_features, in_features]`
- Each row's scale values are broadcast across groups of 32 columns

## Unresolved Questions

None — all semantics confirmed from DeepSeek source code.
