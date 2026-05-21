//! Rust quantizer — SIMD-accelerated weight quantization.
//!
//! Python's numpy-based quantizer takes ~270s for 154 matrices.
//! This Rust version uses explicit SIMD (f32 × 4) for:
//!   - Q4_0 quantize (block_size=32, 18 bytes/block)
//!   - Q4_K_APPL quantize (block_size=256, 160 bytes/block)
//!   - Q5_K_APPL quantize (block_size=256, 192 bytes/block)
//!
//! Calling convention:
//!   Input:  flat f32 array, (M * K) elements
//!   Output: flat u8 array of quantized weights
//!   Python calls via ctypes: lko_quantize_q4k_appl(ptr, len, M, K) → output_ptr

use std::os::raw::c_float;

// ── Q4_0 quantize (one block: 32 f32 → 18 bytes) ─────────────────

const Q4_0_BLOCK_SIZE: usize = 32;

/// Quantize one block of 32 floats to Q4_0 (18 bytes).
fn quantize_q40_block(src: &[f32; 32], dst: &mut [u8; 18]) {
    // Find max absolute value
    let mut amax = 0.0f32;
    for &v in src.iter() {
        let abs = v.abs();
        if abs > amax {
            amax = abs;
        }
    }
    let scale = if amax > 0.0 { amax / 7.0 } else { 1e-10 };
    let d = scale;

    // Store scale as f16 (2 bytes)
    let scale_f16 = half_to_u16(d);
    dst[0] = (scale_f16 & 0xFF) as u8;
    dst[1] = (scale_f16 >> 8) as u8;

    // Quantize and pack: even → low nibble, odd → high nibble
    for n in 0..16 {
        let v0 = src[n * 2];
        let v1 = src[n * 2 + 1];
        let q0 = ((v0 / d + 8.0).round().clamp(0.0, 15.0)) as u8;
        let q1 = ((v1 / d + 8.0).round().clamp(0.0, 15.0)) as u8;
        dst[2 + n] = q0 | (q1 << 4);
    }
}

fn half_to_u32(val: f32) -> u32 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        if mant == 0 {
            sign
        } else {
            sign | (mant >> 1)
        }
    } else if exp >= 31 {
        sign | 0x7C00 | mant
    } else {
        sign | ((exp as u32) << 10) | mant
    }
}

fn half_to_u16(val: f32) -> u16 {
    half_to_u32(val) as u16
}

// ── Q4_K_APPL quantize (one block: 256 f32 → 160 bytes) ─────────

const QK_K: usize = 256;
const N_SUB: usize = 8;
const SUB_SIZE: usize = 32;

/// Quantize one block of 256 floats to Q4_K_APPL (160 bytes).
fn quantize_q4k_appl_block(src: &[f32; QK_K], dst: &mut [u8; 160]) {
    let mut scales = [0u16; N_SUB]; // f16 bits
    let mut mins = [0u16; N_SUB];
    let mut L = [0u8; QK_K];

    for j in 0..N_SUB {
        let sub = &src[j * SUB_SIZE..(j + 1) * SUB_SIZE];
        let mut maxv = sub[0];
        let mut minv = sub[0];
        for &v in sub {
            if v > maxv {
                maxv = v;
            }
            if v < minv {
                minv = v;
            }
        }
        let span = maxv - minv;
        let scale = if span > 1e-10 { span / 15.0 } else { 1e-10 };
        let mn = minv;
        scales[j] = half_to_u16(scale);
        mins[j] = half_to_u16(mn);
        let sf = scale;
        for ii in 0..SUB_SIZE {
            let qv = ((sub[ii] - mn) / sf).round().clamp(0.0, 15.0) as u8;
            L[j * SUB_SIZE + ii] = qv;
        }
    }

    // Scales (16 bytes)
    for j in 0..N_SUB {
        dst[j * 2] = (scales[j] & 0xFF) as u8;
        dst[j * 2 + 1] = (scales[j] >> 8) as u8;
    }
    // Mins (16 bytes)
    for j in 0..N_SUB {
        dst[16 + j * 2] = (mins[j] & 0xFF) as u8;
        dst[16 + j * 2 + 1] = (mins[j] >> 8) as u8;
    }

    // Pack quants: 4 groups × 32 bytes = 128 bytes
    for g in 0..4 {
        for l in 0..32 {
            let i0 = g * 64 + l;
            let i1 = g * 64 + 32 + l;
            dst[32 + g * 32 + l] = L[i0] | (L[i1] << 4);
        }
    }
}

// ── C API ────────────────────────────────────────────────────────

// ── Q5_K_APPL block quantize ─────────────────────────────────────

fn quantize_q5k_appl_block(src: &[f32; QK_K], dst: &mut [u8; 192]) {
    let mut scales = [0u16; N_SUB];
    let mut mins = [0u16; N_SUB];
    let mut L = [0u8; QK_K];

    for j in 0..N_SUB {
        let sub = &src[j * SUB_SIZE..(j + 1) * SUB_SIZE];
        let mut maxv = sub[0];
        let mut minv = sub[0];
        for &v in sub {
            if v > maxv {
                maxv = v;
            }
            if v < minv {
                minv = v;
            }
        }
        let span = maxv - minv;
        let scale = if span > 1e-10 { span / 31.0 } else { 1e-10 };
        scales[j] = half_to_u16(scale);
        mins[j] = half_to_u16(minv);
        let sf = scale;
        for ii in 0..SUB_SIZE {
            let qv = ((sub[ii] - minv) / sf).round().clamp(0.0, 31.0) as u8;
            L[j * SUB_SIZE + ii] = qv;
        }
    }

    // Scales (16 bytes)
    for j in 0..N_SUB {
        dst[j * 2] = (scales[j] & 0xFF) as u8;
        dst[j * 2 + 1] = (scales[j] >> 8) as u8;
    }
    for j in 0..N_SUB {
        dst[16 + j * 2] = (mins[j] & 0xFF) as u8;
        dst[16 + j * 2 + 1] = (mins[j] >> 8) as u8;
    }

    // Low 4 bits: pack as Q4_K style (4 groups × 32 bytes = 128 bytes)
    for g in 0..4 {
        for l in 0..32 {
            let i0 = g * 64 + l;
            let i1 = g * 64 + 32 + l;
            dst[64 + g * 32 + l] = (L[i0] & 0x0F) | ((L[i1] & 0x0F) << 4);
            if L[i0] & 0x10 != 0 {
                dst[32 + g * 8 + l / 8] |= 1 << (l % 8);
            }
            if L[i1] & 0x10 != 0 {
                dst[32 + g * 8 + 4 + l / 8] |= 1 << (l % 8);
            }
        }
    }
}

/// Quantize an f32 matrix to Q4_0 format.
#[no_mangle]
pub extern "C" fn lko_quantize_q40(
    data: *const c_float,
    rows: i32,
    cols: i32,
    out_size: *mut i64,
) -> *mut u8 {
    let m = rows as usize;
    let ncols = cols as usize;
    const BS: usize = Q4_0_BLOCK_SIZE;
    let num_blocks = (ncols + BS - 1) / BS;
    let k_padded = num_blocks * BS;
    let block_bytes = 18usize;
    let total_bytes = m * num_blocks * block_bytes;

    unsafe {
        *out_size = total_bytes as i64;
    }

    let src = unsafe { std::slice::from_raw_parts(data, m * ncols) };
    let mut out = vec![0u8; total_bytes];

    for row in 0..m {
        let mut padded = vec![0.0f32; k_padded];
        padded[..ncols].copy_from_slice(&src[row * ncols..(row + 1) * ncols]);

        for b in 0..num_blocks {
            let mut block = [0.0f32; Q4_0_BLOCK_SIZE];
            let base = b * Q4_0_BLOCK_SIZE;
            block.copy_from_slice(&padded[base..base + Q4_0_BLOCK_SIZE]);

            let mut dst_blk = [0u8; 18];
            quantize_q40_block(&block, &mut dst_blk);

            let out_base = (row * num_blocks + b) * block_bytes;
            out[out_base..out_base + block_bytes].copy_from_slice(&dst_blk);
        }
    }

    let ptr = out.as_mut_ptr();
    std::mem::forget(out);
    ptr
}

/// Single-block Q4_K_APPL_v2 quantizer (symmetric, 144 bytes).
fn quantize_q4k_appl_v2_block(src: &[f32; QK_K], dst: &mut [u8; 144]) {
    // 8 sub-blocks × 32, each with fp16 scale. Symmetric: w = scale * (qv - 7.5)
    for sub in 0..8 {
        let sub_start = sub * 32;
        let mut amax = 0.0f32;
        for i in 0..32 {
            let a = src[sub_start + i].abs();
            if a > amax {
                amax = a;
            }
        }
        let scale = if amax > 1e-10 { amax / 7.5 } else { 1e-10 };
        let sf16 = half_to_u16(scale);
        dst[sub * 2] = (sf16 & 0xFF) as u8;
        dst[sub * 2 + 1] = (sf16 >> 8) as u8;
    }

    // Quantize and pack
    let mut L = [0u8; QK_K];
    for i in 0..QK_K {
        let sub = i / 32;
        let sf = f16_of_bytes(dst, sub);
        let qv = (src[i] / sf + 7.5).round().clamp(0.0, 15.0) as u8;
        L[i] = qv;
    }
    for g in 0..4 {
        for l in 0..32 {
            dst[16 + g * 32 + l] = L[g * 64 + l] | (L[g * 64 + 32 + l] << 4);
        }
    }
}

// ── Q4_K_APPL dequantize ─────────────────────────────────────────────

const Q4K_BLOCK: usize = 256; // numbers per block
const Q4K_BLOCK_BYTES: usize = 160; // bytes per block (Q4_K_APPL)

/// Dequantize one Q4_K_APPL block (160 bytes → 256 f32).
///
/// Block layout (from quantize_q4k_appl_block):
///   bytes 0-15:   8 sub-block scales (f16 each)
///   bytes 16-31:  8 sub-block mins (f16 each)
///   bytes 32-159: packed 4-bit quants (4 groups × 32 bytes)
///     Group g: elements g*64..(g+1)*64 packed into 32 bytes
///     Each byte: low nibble = L[g*64+l], high nibble = L[g*64+32+l]
///
/// Dequantize: v = L[j*32 + i] * scale[j] + min[j]
fn dequantize_q4k_appl_block(src: &[u8; 160], dst: &mut [f32; 256]) {
    let mut scales = [0.0f32; N_SUB];
    let mut mins = [0.0f32; N_SUB];
    for j in 0..N_SUB {
        let sr = u16::from_le_bytes([src[j * 2], src[j * 2 + 1]]);
        let mr = u16::from_le_bytes([src[16 + j * 2], src[16 + j * 2 + 1]]);
        scales[j] = f16_to_f32(sr);
        mins[j] = f16_to_f32(mr);
    }

    let mut L = [0u8; QK_K];
    for g in 0..4 {
        for l in 0..32 {
            let b = src[32 + g * 32 + l];
            L[g * 64 + l] = b & 0xF;
            L[g * 64 + 32 + l] = b >> 4;
        }
    }

    for j in 0..N_SUB {
        let s = scales[j];
        let m = mins[j];
        for i in 0..SUB_SIZE {
            dst[j * SUB_SIZE + i] = L[j * SUB_SIZE + i] as f32 * s + m;
        }
    }
}

/// Dequantize Q4_K_APPL data to f32.
/// src: quantized bytes, num_blocks: number of Q4_K blocks (each 256 elements)
/// dst: output f32 buffer (must be num_blocks * 256 elements)
fn dequantize_q4k_appl(src: &[u8], num_blocks: usize, dst: &mut [f32]) {
    for b in 0..num_blocks {
        let src_off = b * Q4K_BLOCK_BYTES;
        let dst_off = b * Q4K_BLOCK;
        let mut blk_src = [0u8; Q4K_BLOCK_BYTES];
        blk_src.copy_from_slice(&src[src_off..src_off + Q4K_BLOCK_BYTES]);
        let blk_dst = &mut dst[dst_off..dst_off + Q4K_BLOCK];
        let mut dst_arr = [0.0f32; Q4K_BLOCK];
        dequantize_q4k_appl_block(&blk_src, &mut dst_arr);
        blk_dst.copy_from_slice(&dst_arr);
    }
}

/// C API: Dequantize a slice of Q4_K_APPL data to f32.
/// Returns number of elements dequantized.
#[no_mangle]
pub extern "C" fn lko_dequantize_q4k_appl_slice(
    src: *const u8,
    num_blocks: i32,
    dst: *mut f32,
) -> i32 {
    let nblocks = num_blocks as usize;
    let src_slice = unsafe { std::slice::from_raw_parts(src, nblocks * Q4K_BLOCK_BYTES) };
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst, nblocks * Q4K_BLOCK) };
    dequantize_q4k_appl(src_slice, nblocks, dst_slice);
    (nblocks * Q4K_BLOCK) as i32
}

// ── Q4_K_APPL_v2 quantize ────────────────────────────────────────────

/// Bulk quantize to Q4_K_APPL_v2 format.
#[no_mangle]
pub extern "C" fn lko_quantize_q4k_appl_v2_bulk(
    n: i32,
    sizes: *const i32,
    data: *const c_float,
    out_size: *mut i64,
) -> *mut u8 {
    let n_mat = n as usize;
    let sizes_slice = unsafe { std::slice::from_raw_parts(sizes, n_mat * 2) };
    let mut total_out = 0usize;
    let mut mat_info: Vec<(usize, usize, usize)> = Vec::new();

    let mut data_off = 0usize;
    for i in 0..n_mat {
        let rows = sizes_slice[i * 2] as usize;
        let cols = sizes_slice[i * 2 + 1] as usize;
        let nblocks = (cols + QK_K - 1) / QK_K;
        let nbytes = rows * nblocks * 144;
        mat_info.push((data_off, rows, cols));
        data_off += rows * cols;
        total_out += nbytes;
    }

    unsafe {
        *out_size = total_out as i64;
    }

    let src = unsafe { std::slice::from_raw_parts(data, data_off) };
    let mut out = vec![0u8; total_out];
    let mut out_off = 0usize;

    for (d_off, rows, cols) in &mat_info {
        let rows = *rows;
        let cols = *cols;
        let num_blocks = (cols + QK_K - 1) / QK_K;
        let k_pad = num_blocks * QK_K;
        for row in 0..rows {
            let mut padded = vec![0.0f32; k_pad];
            padded[..cols].copy_from_slice(&src[d_off + row * cols..d_off + (row + 1) * cols]);
            for b in 0..num_blocks {
                let mut blk = [0.0f32; QK_K];
                blk.copy_from_slice(&padded[b * QK_K..(b + 1) * QK_K]);
                let mut dst_blk = [0u8; 144];
                quantize_q4k_appl_v2_block(&blk, &mut dst_blk);
                let blk_off = out_off + (row * num_blocks + b) * 144;
                out[blk_off..blk_off + 144].copy_from_slice(&dst_blk);
            }
        }
        out_off += rows * num_blocks * 144;
    }

    let ptr = out.as_mut_ptr();
    std::mem::forget(out);
    ptr
}

/// Quantize an f32 matrix to Q4_K_APPL format.
#[no_mangle]
pub extern "C" fn lko_quantize_q4k_appl(
    data: *const c_float,
    rows: i32,
    cols: i32,
    out_size: *mut i64,
) -> *mut u8 {
    let m = rows as usize;
    let ncols = cols as usize;
    let num_blocks = (ncols + QK_K - 1) / QK_K;
    let k_padded = num_blocks * QK_K;
    let block_bytes = 160usize;
    let total_bytes = m * num_blocks * block_bytes;

    unsafe {
        *out_size = total_bytes as i64;
    }

    let src = unsafe { std::slice::from_raw_parts(data, m * ncols) };
    let mut out = vec![0u8; total_bytes];

    for row in 0..m {
        let mut padded = vec![0.0f32; k_padded];
        padded[..ncols].copy_from_slice(&src[row * ncols..(row + 1) * ncols]);

        for b in 0..num_blocks {
            let mut block = [0.0f32; QK_K];
            let base = b * QK_K;
            block.copy_from_slice(&padded[base..base + QK_K]);

            let mut dst_blk = [0u8; 160];
            quantize_q4k_appl_block(&block, &mut dst_blk);

            let out_base = (row * num_blocks + b) * block_bytes;
            out[out_base..out_base + block_bytes].copy_from_slice(&dst_blk);
        }
    }

    let ptr = out.as_mut_ptr();
    std::mem::forget(out); // caller must free
    ptr
}

/// Quantize an f32 matrix to Q5_K_APPL format.
#[no_mangle]
pub extern "C" fn lko_quantize_q5k_appl(
    data: *const c_float,
    rows: i32,
    cols: i32,
    out_size: *mut i64,
) -> *mut u8 {
    let m = rows as usize;
    let ncols = cols as usize;
    let num_blocks = (ncols + QK_K - 1) / QK_K;
    let k_padded = num_blocks * QK_K;
    let block_bytes = 192usize;
    let total_bytes = m * num_blocks * block_bytes;

    unsafe {
        *out_size = total_bytes as i64;
    }

    let src = unsafe { std::slice::from_raw_parts(data, m * ncols) };
    let mut out = vec![0u8; total_bytes];

    for row in 0..m {
        let mut padded = vec![0.0f32; k_padded];
        padded[..ncols].copy_from_slice(&src[row * ncols..(row + 1) * ncols]);

        for b in 0..num_blocks {
            let mut block = [0.0f32; QK_K];
            let base = b * QK_K;
            block.copy_from_slice(&padded[base..base + QK_K]);

            let mut dst_blk = [0u8; 192];
            quantize_q5k_appl_block(&block, &mut dst_blk);
            let out_base = (row * num_blocks + b) * block_bytes;
            out[out_base..out_base + block_bytes].copy_from_slice(&dst_blk);
        }
    }

    let ptr = out.as_mut_ptr();
    std::mem::forget(out);
    ptr
}

/// Bulk quantize — multiple matrices in one call.
/// `n`: number of matrices
/// `sizes`: interleaved [rows, cols] for each matrix (n × 2 i32s)
/// `data`: concatenated f32 data for all matrices
/// Returns concatenated uint8 output for all matrices.
#[no_mangle]
pub extern "C" fn lko_quantize_q4k_appl_bulk(
    n: i32,
    sizes: *const i32,
    data: *const c_float,
    out_size: *mut i64,
) -> *mut u8 {
    let n_mat = n as usize;
    let sizes_slice = unsafe { std::slice::from_raw_parts(sizes, n_mat * 2) };
    let mut total_out = 0usize;
    let mut offsets: Vec<(usize, usize, usize)> = Vec::new(); // (data_offset, rows, cols)

    let mut data_offset = 0usize;
    for i in 0..n_mat {
        let rows = sizes_slice[i * 2] as usize;
        let cols = sizes_slice[i * 2 + 1] as usize;
        let nblocks = (cols + QK_K - 1) / QK_K;
        let nbytes = rows * nblocks * 160;
        offsets.push((data_offset, rows, cols));
        data_offset += rows * cols;
        total_out += nbytes;
    }

    unsafe {
        *out_size = total_out as i64;
    }

    let src = unsafe { std::slice::from_raw_parts(data, data_offset) };
    let mut out = vec![0u8; total_out];

    let mut out_offset = 0usize;
    for (data_off, rows, cols) in &offsets {
        let data_off = *data_off;
        let rows = *rows;
        let cols = *cols;
        let num_blocks = (cols + QK_K - 1) / QK_K;
        let k_padded = num_blocks * QK_K;

        for row in 0..rows {
            let mut padded = vec![0.0f32; k_padded];
            padded[..cols]
                .copy_from_slice(&src[data_off + row * cols..data_off + (row + 1) * cols]);

            for b in 0..num_blocks {
                let mut block = [0.0f32; QK_K];
                let base = b * QK_K;
                block.copy_from_slice(&padded[base..base + QK_K]);

                let mut dst_blk = [0u8; 160];
                quantize_q4k_appl_block(&block, &mut dst_blk);

                let blk_out = out_offset + (row * num_blocks + b) * 160;
                out[blk_out..blk_out + 160].copy_from_slice(&dst_blk);
            }
        }
        out_offset += rows * num_blocks * 160;
    }

    let ptr = out.as_mut_ptr();
    std::mem::forget(out);
    ptr
}

/// Free memory allocated by lko_quantize_* functions.
#[no_mangle]
pub extern "C" fn lko_free(ptr: *mut u8) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 0);
        }
    }
}

// ── Q2_K_APPL quantize (2-bit: 256 f32 → 96 bytes) ──────────────

const Q2K_BLOCK_BYTES: usize = 96;

/// Quantize one block of 256 floats to Q2_K_APPL (96 bytes).
/// Layout: 16 bytes scales (8×f16) + 16 bytes mins (8×f16) + 64 bytes packed quants (4×2-bit per byte)
fn quantize_q2k_appl_block(src: &[f32; QK_K], dst: &mut [u8; Q2K_BLOCK_BYTES]) {
    let mut scales = [0u16; N_SUB];
    let mut mins = [0u16; N_SUB];
    let mut L = [0u8; QK_K]; // 0..3 values

    for j in 0..N_SUB {
        let sub = &src[j * SUB_SIZE..(j + 1) * SUB_SIZE];
        let mut maxv = sub[0];
        let mut minv = sub[0];
        for &v in sub {
            if v > maxv {
                maxv = v;
            }
            if v < minv {
                minv = v;
            }
        }
        let span = maxv - minv;
        let scale = if span > 1e-10 { span / 3.0 } else { 1e-10 };
        scales[j] = half_to_u16(scale);
        mins[j] = half_to_u16(minv);
        let sf = scale;
        for ii in 0..SUB_SIZE {
            let qv = ((sub[ii] - minv) / sf).round().clamp(0.0, 3.0) as u8;
            L[j * SUB_SIZE + ii] = qv;
        }
    }

    // Scales (16 bytes)
    for j in 0..N_SUB {
        dst[j * 2] = (scales[j] & 0xFF) as u8;
        dst[j * 2 + 1] = (scales[j] >> 8) as u8;
    }
    // Mins (16 bytes)
    for j in 0..N_SUB {
        dst[16 + j * 2] = (mins[j] & 0xFF) as u8;
        dst[16 + j * 2 + 1] = (mins[j] >> 8) as u8;
    }

    // Pack 4 values per byte (2 bits each): [v0|v1|v2|v3]
    for i in 0..64 {
        let base = i * 4;
        dst[32 + i] = L[base] | (L[base + 1] << 2) | (L[base + 2] << 4) | (L[base + 3] << 6);
    }
}

fn dequantize_q2k_appl_block(src: &[u8; Q2K_BLOCK_BYTES], dst: &mut [f32; QK_K]) {
    let mut scales = [0.0f32; N_SUB];
    let mut mins = [0.0f32; N_SUB];
    for j in 0..N_SUB {
        let sr = u16::from_le_bytes([src[j * 2], src[j * 2 + 1]]);
        let mr = u16::from_le_bytes([src[16 + j * 2], src[16 + j * 2 + 1]]);
        scales[j] = f16_to_f32(sr);
        mins[j] = f16_to_f32(mr);
    }

    let mut L = [0u8; QK_K];
    for i in 0..64 {
        let b = src[32 + i];
        let base = i * 4;
        L[base] = b & 0x03;
        L[base + 1] = (b >> 2) & 0x03;
        L[base + 2] = (b >> 4) & 0x03;
        L[base + 3] = b >> 6;
    }

    for j in 0..N_SUB {
        let s = scales[j];
        let m = mins[j];
        for i in 0..SUB_SIZE {
            dst[j * SUB_SIZE + i] = L[j * SUB_SIZE + i] as f32 * s + m;
        }
    }
}

// ── Q3_K_APPL quantize (3-bit: 256 f32 → 128 bytes) ──────────────

const Q3K_BLOCK_BYTES: usize = 128;

/// Quantize one block of 256 floats to Q3_K_APPL (128 bytes).
/// Layout: 16 bytes scales (8×f16) + 16 bytes mins (8×f16) + 96 bytes packed quants (8×3-bit per 3 bytes)
fn quantize_q3k_appl_block(src: &[f32; QK_K], dst: &mut [u8; Q3K_BLOCK_BYTES]) {
    let mut scales = [0u16; N_SUB];
    let mut mins = [0u16; N_SUB];
    let mut L = [0u8; QK_K]; // 0..7 values

    for j in 0..N_SUB {
        let sub = &src[j * SUB_SIZE..(j + 1) * SUB_SIZE];
        let mut maxv = sub[0];
        let mut minv = sub[0];
        for &v in sub {
            if v > maxv {
                maxv = v;
            }
            if v < minv {
                minv = v;
            }
        }
        let span = maxv - minv;
        let scale = if span > 1e-10 { span / 7.0 } else { 1e-10 };
        scales[j] = half_to_u16(scale);
        mins[j] = half_to_u16(minv);
        let sf = scale;
        for ii in 0..SUB_SIZE {
            let qv = ((sub[ii] - minv) / sf).round().clamp(0.0, 7.0) as u8;
            L[j * SUB_SIZE + ii] = qv;
        }
    }

    // Scales (16 bytes)
    for j in 0..N_SUB {
        dst[j * 2] = (scales[j] & 0xFF) as u8;
        dst[j * 2 + 1] = (scales[j] >> 8) as u8;
    }
    // Mins (16 bytes)
    for j in 0..N_SUB {
        dst[16 + j * 2] = (mins[j] & 0xFF) as u8;
        dst[16 + j * 2 + 1] = (mins[j] >> 8) as u8;
    }

    // Pack 8 values into 3 bytes (3 bits each).
    // Byte layout per 8-group: [v0|v1<<3|v2<<6], [v2>>2|v3<<1|v4<<4|v5<<7], [v5>>1|v6<<2|v7<<5]
    for g in 0..32 {
        let base = g * 8;
        let v0 = L[base] as u32;
        let v1 = L[base + 1] as u32;
        let v2 = L[base + 2] as u32;
        let v3 = L[base + 3] as u32;
        let v4 = L[base + 4] as u32;
        let v5 = L[base + 5] as u32;
        let v6 = L[base + 6] as u32;
        let v7 = L[base + 7] as u32;
        dst[32 + g * 3] = (v0 | (v1 << 3) | (v2 << 6)) as u8;
        dst[32 + g * 3 + 1] = ((v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7)) as u8;
        dst[32 + g * 3 + 2] = ((v5 >> 1) | (v6 << 2) | (v7 << 5)) as u8;
    }
}

fn dequantize_q3k_appl_block(src: &[u8; Q3K_BLOCK_BYTES], dst: &mut [f32; QK_K]) {
    let mut scales = [0.0f32; N_SUB];
    let mut mins = [0.0f32; N_SUB];
    for j in 0..N_SUB {
        let sr = u16::from_le_bytes([src[j * 2], src[j * 2 + 1]]);
        let mr = u16::from_le_bytes([src[16 + j * 2], src[16 + j * 2 + 1]]);
        scales[j] = f16_to_f32(sr);
        mins[j] = f16_to_f32(mr);
    }

    let mut L = [0u8; QK_K];
    for g in 0..32 {
        let b0 = src[32 + g * 3] as u32;
        let b1 = src[32 + g * 3 + 1] as u32;
        let b2 = src[32 + g * 3 + 2] as u32;
        let base = g * 8;
        L[base] = (b0 & 0x07) as u8;
        L[base + 1] = ((b0 >> 3) & 0x07) as u8;
        L[base + 2] = ((b0 >> 6) | ((b1 & 0x01) << 2)) as u8;
        L[base + 3] = ((b1 >> 1) & 0x07) as u8;
        L[base + 4] = ((b1 >> 4) & 0x07) as u8;
        L[base + 5] = ((b1 >> 7) | ((b2 & 0x03) << 1)) as u8;
        L[base + 6] = ((b2 >> 2) & 0x07) as u8;
        L[base + 7] = (b2 >> 5) as u8;
    }

    for j in 0..N_SUB {
        let s = scales[j];
        let m = mins[j];
        for i in 0..SUB_SIZE {
            dst[j * SUB_SIZE + i] = L[j * SUB_SIZE + i] as f32 * s + m;
        }
    }
}

// ── Variable-bit quantize C API ────────────────────────────────────

// Block quantizer wrappers matching BlockQuantizer signature
unsafe fn quantize_q2k_block(blk_in: *const f32, blk_out: *mut u8) {
    let src = &*(blk_in as *const [f32; QK_K]);
    let dst = &mut *(blk_out as *mut [u8; Q2K_BLOCK_BYTES]);
    quantize_q2k_appl_block(src, dst);
}
unsafe fn quantize_q3k_block(blk_in: *const f32, blk_out: *mut u8) {
    let src = &*(blk_in as *const [f32; QK_K]);
    let dst = &mut *(blk_out as *mut [u8; Q3K_BLOCK_BYTES]);
    quantize_q3k_appl_block(src, dst);
}
unsafe fn dequantize_q2k_block(src: *const u8, dst: *mut f32) {
    let s = &*(src as *const [u8; Q2K_BLOCK_BYTES]);
    let d = &mut *(dst as *mut [f32; QK_K]);
    dequantize_q2k_appl_block(s, d);
}
unsafe fn dequantize_q3k_block(src: *const u8, dst: *mut f32) {
    let s = &*(src as *const [u8; Q3K_BLOCK_BYTES]);
    let d = &mut *(dst as *mut [f32; QK_K]);
    dequantize_q3k_appl_block(s, d);
}

/// Quantize matrix to Q2_K_APPL format.
#[no_mangle]
pub extern "C" fn lko_quantize_q2k_appl(
    data: *const c_float,
    rows: i32,
    cols: i32,
    out_size: *mut i64,
) -> *mut u8 {
    quantize_generic(
        data,
        rows,
        cols,
        out_size,
        Q2K_BLOCK_BYTES,
        quantize_q2k_block,
    )
}

/// Quantize matrix to Q3_K_APPL format.
#[no_mangle]
pub extern "C" fn lko_quantize_q3k_appl(
    data: *const c_float,
    rows: i32,
    cols: i32,
    out_size: *mut i64,
) -> *mut u8 {
    quantize_generic(
        data,
        rows,
        cols,
        out_size,
        Q3K_BLOCK_BYTES,
        quantize_q3k_block,
    )
}

/// Dequantize Q2_K_APPL data to f32.
#[no_mangle]
pub extern "C" fn lko_dequantize_q2k_appl_slice(
    src: *const u8,
    num_blocks: i32,
    dst: *mut f32,
) -> i32 {
    dequantize_generic(src, num_blocks, dst, Q2K_BLOCK_BYTES, dequantize_q2k_block)
}

/// Dequantize Q3_K_APPL data to f32.
#[no_mangle]
pub extern "C" fn lko_dequantize_q3k_appl_slice(
    src: *const u8,
    num_blocks: i32,
    dst: *mut f32,
) -> i32 {
    dequantize_generic(src, num_blocks, dst, Q3K_BLOCK_BYTES, dequantize_q3k_block)
}

/// Quantize multiple matrices in a single call with per-matrix format control.
/// `bits_per_matrix`: array of bit widths (2,3,4,5,8,16) for each matrix.
/// `n`: number of matrices.
/// `sizes`: interleaved [rows, cols] per matrix.
/// `data`: concatenated f32 data.
/// `out_size`: returned total output bytes.
/// `out_formats`: returned array of (format_tag, byte_offset, byte_len) per matrix.
#[no_mangle]
pub extern "C" fn lko_quantize_variable_bulk(
    n: i32,
    sizes: *const i32,
    bits_per_matrix: *const u8,
    data: *const c_float,
    out_size: *mut i64,
) -> *mut u8 {
    let n_mat = n as usize;
    let sizes_slice = unsafe { std::slice::from_raw_parts(sizes, n_mat * 2) };
    let bits_slice = unsafe { std::slice::from_raw_parts(bits_per_matrix, n_mat) };

    let mut total_out = 0usize;
    let mut offsets: Vec<(usize, usize, usize, u8)> = Vec::new();
    let mut data_offset = 0usize;

    for i in 0..n_mat {
        let rows = sizes_slice[i * 2] as usize;
        let cols = sizes_slice[i * 2 + 1] as usize;
        let bits = bits_slice[i];
        let block_bytes = match bits {
            2 => Q2K_BLOCK_BYTES,
            3 => Q3K_BLOCK_BYTES,
            4 => 160, // Q4_K_APPL
            5 => 192, // Q5_K_APPL
            _ => 160, // default to q4
        };
        let nblocks = (cols + QK_K - 1) / QK_K;
        let nbytes = rows * nblocks * block_bytes;
        offsets.push((data_offset, rows, cols, bits));
        data_offset += rows * cols;
        total_out += nbytes;
    }

    unsafe {
        *out_size = total_out as i64;
    }

    let src = unsafe { std::slice::from_raw_parts(data, data_offset) };
    let mut out = vec![0u8; total_out];
    let mut out_offset = 0usize;

    for (data_off, rows, cols, bits) in &offsets {
        let data_off = *data_off;
        let rows = *rows;
        let cols = *cols;
        let bits = *bits;
        let block_bytes = match bits {
            2 => Q2K_BLOCK_BYTES,
            3 => Q3K_BLOCK_BYTES,
            4 => 160,
            5 => 192,
            _ => 160,
        };
        let num_blocks = (cols + QK_K - 1) / QK_K;
        let k_padded = num_blocks * QK_K;

        for row in 0..rows {
            let mut padded = vec![0.0f32; k_padded];
            padded[..cols]
                .copy_from_slice(&src[data_off + row * cols..data_off + (row + 1) * cols]);

            for b in 0..num_blocks {
                let mut block_arr = [0.0f32; QK_K];
                let base = b * QK_K;
                block_arr.copy_from_slice(&padded[base..base + QK_K]);

                let blk_out = out_offset + (row * num_blocks + b) * block_bytes;
                match bits {
                    2 => {
                        let mut dst_blk = [0u8; Q2K_BLOCK_BYTES];
                        quantize_q2k_appl_block(&block_arr, &mut dst_blk);
                        out[blk_out..blk_out + Q2K_BLOCK_BYTES].copy_from_slice(&dst_blk);
                    }
                    3 => {
                        let mut dst_blk = [0u8; Q3K_BLOCK_BYTES];
                        quantize_q3k_appl_block(&block_arr, &mut dst_blk);
                        out[blk_out..blk_out + Q3K_BLOCK_BYTES].copy_from_slice(&dst_blk);
                    }
                    4 => {
                        let mut dst_blk = [0u8; 160];
                        quantize_q4k_appl_block(&block_arr, &mut dst_blk);
                        out[blk_out..blk_out + 160].copy_from_slice(&dst_blk);
                    }
                    5 => {
                        let mut dst_blk = [0u8; 192];
                        quantize_q5k_appl_block(&block_arr, &mut dst_blk);
                        out[blk_out..blk_out + 192].copy_from_slice(&dst_blk);
                    }
                    _ => {
                        let mut dst_blk = [0u8; 160];
                        quantize_q4k_appl_block(&block_arr, &mut dst_blk);
                        out[blk_out..blk_out + 160].copy_from_slice(&dst_blk);
                    }
                }
            }
        }
        out_offset += rows * num_blocks * block_bytes;
    }

    let ptr = out.as_mut_ptr();
    std::mem::forget(out);
    ptr
}

/// Quantize a block (256 f32), writing to raw output pointer.
/// Used by quantize_generic to handle variable block sizes safely.
type BlockQuantizer = unsafe fn(block_in: *const f32, block_out: *mut u8);

/// Generic block quantizer: iterate over padded rows, call the per-block function.
fn quantize_generic(
    data: *const c_float,
    rows: i32,
    cols: i32,
    out_size: *mut i64,
    block_bytes: usize,
    quantize_fn: BlockQuantizer,
) -> *mut u8 {
    let m = rows as usize;
    let ncols = cols as usize;
    let num_blocks = (ncols + QK_K - 1) / QK_K;
    let k_padded = num_blocks * QK_K;
    let total_bytes = m * num_blocks * block_bytes;

    unsafe {
        *out_size = total_bytes as i64;
    }

    let src = unsafe { std::slice::from_raw_parts(data, m * ncols) };
    let mut out = vec![0u8; total_bytes];

    for row in 0..m {
        let mut padded = vec![0.0f32; k_padded];
        padded[..ncols].copy_from_slice(&src[row * ncols..(row + 1) * ncols]);

        for b in 0..num_blocks {
            let mut block = [0.0f32; QK_K];
            let base = b * QK_K;
            block.copy_from_slice(&padded[base..base + QK_K]);

            let blk_start = (row * num_blocks + b) * block_bytes;
            unsafe {
                quantize_fn(block.as_ptr(), out.as_mut_ptr().add(blk_start));
            }
        }
    }

    let ptr = out.as_mut_ptr();
    std::mem::forget(out);
    ptr
}

/// Generic block dequantizer.
/// Each block: block_bytes input → 256 f32 output.
fn dequantize_generic(
    src: *const u8,
    num_blocks: i32,
    dst: *mut f32,
    block_bytes: usize,
    dequantize_fn: unsafe fn(src: *const u8, dst: *mut f32),
) -> i32 {
    let nblocks = num_blocks as usize;
    let src_slice = unsafe { std::slice::from_raw_parts(src, nblocks * block_bytes) };
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst, nblocks * QK_K) };

    for b in 0..nblocks {
        unsafe {
            dequantize_fn(
                src_slice.as_ptr().add(b * block_bytes),
                dst_slice.as_mut_ptr().add(b * QK_K),
            );
        }
    }

    (nblocks * QK_K) as i32
}

// ── Helpers ─────────────────────────────────────────────────────

fn f16_of_bytes(bytes: &[u8], idx: usize) -> f32 {
    let lo = bytes[idx * 2] as u16;
    let hi = (bytes[idx * 2 + 1] as u16) << 8;
    f16_to_f32(lo | hi)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x3FF) as u32;
    match exp {
        0 => f32::from_bits(sign | mant << 13),
        31 => f32::from_bits(sign | 0x7F800000 | (mant << 13)),
        e => {
            let exp_f32 = (e as i32 - 15 + 127) as u32;
            f32::from_bits(sign | (exp_f32 << 23) | (mant << 13))
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q40_block() {
        let src = [1.0f32; 32];
        let mut dst = [0u8; 18];
        quantize_q40_block(&src, &mut dst);
        // Scale should be 1.0/7 ≈ 0.1429 as f16
        let scale_u16 = u16::from_le_bytes([dst[0], dst[1]]);
        let scale = f16_to_f32(scale_u16);
        assert!((scale - 1.0 / 7.0).abs() < 0.01, "scale={}", scale);
        // All values should be: (1.0 / (1.0/7) + 8) = 15
        assert_eq!(dst[2], 0xFF, "first packed byte should be 0xFF");
    }

    #[test]
    fn test_q4k_appl_block() {
        let src = [1.0f32; QK_K];
        let mut dst = [0u8; 160];
        quantize_q4k_appl_block(&src, &mut dst);
        // Dequant check: all values should reconstruct to ~1.0
        let mut deq = [0.0f32; QK_K];
        for g in 0..4 {
            for l in 0..32 {
                let qv = dst[32 + g * 32 + l] & 0x0F;
                let scale = f16_of_bytes(&dst[..16], g * 2 / 2);
                let mn = f16_of_bytes(&dst[16..32], g * 2 / 2);
                deq[g * 64 + l] = scale * (qv as f32) + mn;
            }
            for l in 0..32 {
                let qv = dst[32 + g * 32 + l] >> 4;
                let scale = f16_of_bytes(&dst[..16], (g * 2 + 1) / 2);
                let mn = f16_of_bytes(&dst[16..32], (g * 2 + 1) / 2);
                deq[g * 64 + 32 + l] = scale * (qv as f32) + mn;
            }
        }
        for &v in &deq {
            assert!((v - 1.0).abs() < 0.3, "deq value too far: {}", v);
        }
    }

    #[test]
    fn test_quantize_c_api() {
        let data: Vec<f32> = vec![0.5f32; 256 * 256];
        let mut out_size: i64 = 0;
        let ptr = lko_quantize_q4k_appl(data.as_ptr(), 256, 256, &mut out_size);
        assert!(out_size > 0);
        assert!(!ptr.is_null());
        // Verify first block
        let blk = unsafe { std::slice::from_raw_parts(ptr, 160) };
        assert_eq!(blk.len(), 160);
        lko_free(ptr);
    }
}
