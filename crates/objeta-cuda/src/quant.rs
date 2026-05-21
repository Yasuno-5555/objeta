use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};

use crate::context::CudaDeviceInfo;
use crate::memory::DeviceBuffer;
use crate::stream::CudaStreamHandle;
use crate::telemetry::CudaEventTimer;
use crate::{cuda_map_err, CudaError, CudaErrorKind, Result};

const Q4_KERNEL_SRC: &str = include_str!("../kernels/q4_gemv.cu");
const Q5_KERNEL_SRC: &str = include_str!("../kernels/q5_gemv.cu");
const IQ3_KERNEL_SRC: &str = include_str!("../kernels/iq3_gemv.cu");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QuantFormat {
    Q4_0,
    Q5_0,
    IQ3_0,
}

impl QuantFormat {
    pub fn block_size(&self) -> usize {
        match self {
            QuantFormat::Q4_0 => 32,
            QuantFormat::Q5_0 => 32,
            QuantFormat::IQ3_0 => 32,
        }
    }

    pub fn block_bytes(&self) -> usize {
        match self {
            QuantFormat::Q4_0 => 18,
            QuantFormat::Q5_0 => 22,
            QuantFormat::IQ3_0 => 14,
        }
    }

    pub fn kernel_name(&self) -> &'static str {
        match self {
            QuantFormat::Q4_0 => "q4_gemv",
            QuantFormat::Q5_0 => "q5_gemv",
            QuantFormat::IQ3_0 => "iq3_gemv",
        }
    }

    pub fn format_label(&self) -> &'static str {
        match self {
            QuantFormat::Q4_0 => "q4_0",
            QuantFormat::Q5_0 => "q5_0",
            QuantFormat::IQ3_0 => "iq3_0",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QGemvShape {
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    pub block_bytes: usize,
}

impl QGemvShape {
    pub fn new(format: QuantFormat, rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            block_size: format.block_size(),
            block_bytes: format.block_bytes(),
        }
    }

    pub fn q4_0(rows: usize, cols: usize) -> Self {
        Self::new(QuantFormat::Q4_0, rows, cols)
    }

    pub fn q5_0(rows: usize, cols: usize) -> Self {
        Self::new(QuantFormat::Q5_0, rows, cols)
    }

    pub fn iq3_0(rows: usize, cols: usize) -> Self {
        Self::new(QuantFormat::IQ3_0, rows, cols)
    }

    pub fn blocks_per_row(&self) -> usize {
        self.cols / self.block_size
    }

    pub fn quantized_row_bytes(&self) -> usize {
        self.blocks_per_row() * self.block_bytes
    }

    pub fn quantized_matrix_bytes(&self) -> usize {
        self.rows * self.quantized_row_bytes()
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct QGemvTelemetry {
    pub h2d_ms: f32,
    pub kernel_ms: f32,
    pub d2h_ms: f32,
    pub unaccounted_ms: f32,
    pub total_ms: f32,
    pub bytes_read: usize,
    pub effective_gbps: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct QGemvNumerics {
    pub cosine_similarity: f32,
    pub relative_l2_error: f32,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct QGemvNumericsSuite {
    pub cosine_similarity: f32,
    pub relative_l2_error: f32,
    pub max_abs_error: f32,
    pub cuda_vs_cpu_quant: QGemvNumerics,
    pub quant_vs_fp32: QGemvNumerics,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QGemvReport {
    pub backend: &'static str,
    pub kernel: &'static str,
    pub format: QuantFormat,
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    pub block_bytes: usize,
    pub telemetry: QGemvTelemetry,
    pub numerics: QGemvNumericsSuite,
}

#[derive(Debug)]
struct QuantKernelModule {
    _module: Arc<CudaModule>,
    gemv: CudaFunction,
}

#[derive(Debug)]
pub struct QuantBackend {
    context: Arc<CudaContext>,
    device_info: CudaDeviceInfo,
    q4_module: Mutex<Option<QuantKernelModule>>,
    q5_module: Mutex<Option<QuantKernelModule>>,
    iq3_module: Mutex<Option<QuantKernelModule>>,
}

impl QuantBackend {
    pub fn new(context: Arc<CudaContext>, device_info: CudaDeviceInfo) -> Self {
        Self {
            context,
            device_info,
            q4_module: Mutex::new(None),
            q5_module: Mutex::new(None),
            iq3_module: Mutex::new(None),
        }
    }

    pub fn status(&self) -> Result<()> {
        Ok(())
    }

    pub fn compile_format(&self, format: QuantFormat) -> Result<()> {
        match format {
            QuantFormat::Q4_0 => {
                let _unused = self.q4_module()?;
            }
            QuantFormat::Q5_0 => {
                let _unused = self.q5_module()?;
            }
            QuantFormat::IQ3_0 => {
                let _unused = self.iq3_module()?;
            }
        }
        Ok(())
    }

    pub fn quantize_matrix(
        &self,
        format: QuantFormat,
        matrix: &[f32],
        shape: QGemvShape,
    ) -> Result<Vec<u8>> {
        quantize_matrix_cpu(format, matrix, shape)
    }

    pub fn q4_quantize_matrix(&self, matrix: &[f32], shape: QGemvShape) -> Result<Vec<u8>> {
        self.quantize_matrix(QuantFormat::Q4_0, matrix, shape)
    }

    pub fn q5_quantize_matrix(&self, matrix: &[f32], shape: QGemvShape) -> Result<Vec<u8>> {
        self.quantize_matrix(QuantFormat::Q5_0, matrix, shape)
    }

    pub fn gemv(
        &self,
        format: QuantFormat,
        stream: &CudaStreamHandle,
        qweights: &[u8],
        x: &[f32],
        shape: QGemvShape,
    ) -> Result<(Vec<f32>, QGemvTelemetry)> {
        validate_quant_inputs(format, qweights, x, shape)?;

        let total_timer = CudaEventTimer::start(stream.raw())?;
        let h2d_timer = CudaEventTimer::start(stream.raw())?;
        let d_qweights = stream.copy_from_slice(qweights)?;
        let d_x = stream.copy_from_slice(x)?;
        let h2d = h2d_timer.stop(format!("{}_h2d", format.format_label()), stream.raw())?;

        let mut d_y = stream.alloc_zeros::<f32>(shape.rows)?;
        let kernel_timer = CudaEventTimer::start(stream.raw())?;
        self.launch_kernel(format, stream.raw(), &d_qweights, &d_x, &mut d_y, shape)?;
        let kernel = kernel_timer.stop(format.kernel_name(), stream.raw())?;

        let d2h_timer = CudaEventTimer::start(stream.raw())?;
        let y = stream.copy_to_vec(&d_y)?;
        let d2h = d2h_timer.stop(format!("{}_d2h", format.format_label()), stream.raw())?;

        let total = total_timer.stop(format!("{}_total", format.format_label()), stream.raw())?;
        let bytes_read = shape.quantized_matrix_bytes() + shape.cols * std::mem::size_of::<f32>();
        let effective_gbps = if kernel.elapsed_ms > 0.0 {
            bytes_read as f32 / (kernel.elapsed_ms / 1_000.0) / 1.0e9
        } else {
            0.0
        };

        let unaccounted_ms = (total.elapsed_ms - h2d.elapsed_ms - kernel.elapsed_ms - d2h.elapsed_ms).max(0.0);
        Ok((
            y,
            QGemvTelemetry {
                h2d_ms: h2d.elapsed_ms,
                kernel_ms: kernel.elapsed_ms,
                d2h_ms: d2h.elapsed_ms,
                unaccounted_ms,
                total_ms: total.elapsed_ms,
                bytes_read,
                effective_gbps,
            },
        ))
    }

    pub fn q4_gemv(
        &self,
        stream: &CudaStreamHandle,
        qweights: &[u8],
        x: &[f32],
        shape: QGemvShape,
    ) -> Result<(Vec<f32>, QGemvTelemetry)> {
        self.gemv(QuantFormat::Q4_0, stream, qweights, x, shape)
    }

    pub fn q5_gemv(
        &self,
        stream: &CudaStreamHandle,
        qweights: &[u8],
        x: &[f32],
        shape: QGemvShape,
    ) -> Result<(Vec<f32>, QGemvTelemetry)> {
        self.gemv(QuantFormat::Q5_0, stream, qweights, x, shape)
    }

    pub(crate) fn launch_kernel(
        &self,
        format: QuantFormat,
        stream: &Arc<CudaStream>,
        qweights: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        shape: QGemvShape,
    ) -> Result<()> {
        let module_guard = match format {
            QuantFormat::Q4_0 => self.q4_module()?,
            QuantFormat::Q5_0 => self.q5_module()?,
            QuantFormat::IQ3_0 => self.iq3_module()?,
        };
        let module = module_guard.as_ref().expect("kernel module must exist");
        let rows = shape.rows as u32;
        let cols = shape.cols as u32;
        let blocks_per_row = shape.blocks_per_row() as u32;
        let row_bytes = shape.quantized_row_bytes() as u32;
        let cfg = LaunchConfig {
            grid_dim: (shape.rows as u32, 1, 1),
            block_dim: (shape.block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_map_err!(
            CudaErrorKind::Driver,
            format!("launch {} kernel", format.kernel_name()),
            unsafe {
                stream
                    .launch_builder(&module.gemv)
                    .arg(&qweights.raw)
                    .arg(&x.raw)
                    .arg(&mut y.raw)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&blocks_per_row)
                    .arg(&row_bytes)
                    .launch(cfg)
            }
        )?;
        Ok(())
    }

    fn q4_module(&self) -> Result<MutexGuard<'_, Option<QuantKernelModule>>> {
        self.ensure_module(
            &self.q4_module,
            QuantFormat::Q4_0,
            Q4_KERNEL_SRC,
            "objeta_cuda_q4_gemv.cu",
            "q4_gemv_f32_accum",
        )
    }

    fn q5_module(&self) -> Result<MutexGuard<'_, Option<QuantKernelModule>>> {
        self.ensure_module(
            &self.q5_module,
            QuantFormat::Q5_0,
            Q5_KERNEL_SRC,
            "objeta_cuda_q5_gemv.cu",
            "q5_gemv_f32_accum",
        )
    }

    fn iq3_module(&self) -> Result<MutexGuard<'_, Option<QuantKernelModule>>> {
        self.ensure_module(
            &self.iq3_module,
            QuantFormat::IQ3_0,
            IQ3_KERNEL_SRC,
            "objeta_cuda_iq3_gemv.cu",
            "iq3_gemv_f32_accum",
        )
    }

    fn ensure_module<'a>(
        &self,
        slot: &'a Mutex<Option<QuantKernelModule>>,
        format: QuantFormat,
        source: &str,
        source_name: &'static str,
        function_name: &'static str,
    ) -> Result<MutexGuard<'a, Option<QuantKernelModule>>> {
        let mut guard = slot.lock().map_err(|err| {
            CudaError::new(
                CudaErrorKind::Internal,
                format!("lock {} kernel module cache", format.format_label()),
                err.to_string(),
                file!(),
                line!(),
                module_path!(),
            )
        })?;
        if guard.is_none() {
            let ptx = compile_kernel_ptx(
                source,
                source_name,
                self.device_info.compute_capability_major,
                self.device_info.compute_capability_minor,
            )?;
            let module = cuda_map_err!(
                CudaErrorKind::Driver,
                format!("load {} PTX module", format.format_label()),
                self.context.load_module(ptx)
            )?;
            let gemv = cuda_map_err!(
                CudaErrorKind::Driver,
                format!("load {} kernel function {}", format.format_label(), function_name),
                module.load_function(function_name)
            )?;
            *guard = Some(QuantKernelModule {
                _module: module,
                gemv,
            });
        }
        Ok(guard)
    }
}

pub fn quantize_matrix_cpu(
    format: QuantFormat,
    matrix: &[f32],
    shape: QGemvShape,
) -> Result<Vec<u8>> {
    validate_dense_matrix(format, matrix, shape)?;
    let mut out = vec![0u8; shape.quantized_matrix_bytes()];

    match format {
        QuantFormat::Q4_0 => {
            for row in 0..shape.rows {
                let src = &matrix[row * shape.cols..(row + 1) * shape.cols];
                for block_idx in 0..shape.blocks_per_row() {
                    let src_block: &[f32; 32] = src
                        [block_idx * 32..(block_idx + 1) * 32]
                        .try_into()
                        .expect("fixed-size q4 block");
                    let dst_offset =
                        row * shape.quantized_row_bytes() + block_idx * shape.block_bytes;
                    let mut dst_block = [0u8; 18];
                    q4_quantize_block(src_block, &mut dst_block);
                    out[dst_offset..dst_offset + 18].copy_from_slice(&dst_block);
                }
            }
        }
        QuantFormat::Q5_0 => {
            for row in 0..shape.rows {
                let src = &matrix[row * shape.cols..(row + 1) * shape.cols];
                for block_idx in 0..shape.blocks_per_row() {
                    let src_block: &[f32; 32] = src
                        [block_idx * 32..(block_idx + 1) * 32]
                        .try_into()
                        .expect("fixed-size q5 block");
                    let dst_offset =
                        row * shape.quantized_row_bytes() + block_idx * shape.block_bytes;
                    let mut dst_block = [0u8; 22];
                    q5_quantize_block(src_block, &mut dst_block);
                    out[dst_offset..dst_offset + 22].copy_from_slice(&dst_block);
                }
            }
        }
        QuantFormat::IQ3_0 => {
            for row in 0..shape.rows {
                let src = &matrix[row * shape.cols..(row + 1) * shape.cols];
                for block_idx in 0..shape.blocks_per_row() {
                    let src_block: &[f32; 32] = src
                        [block_idx * 32..(block_idx + 1) * 32]
                        .try_into()
                        .expect("fixed-size iq3 block");
                    let dst_offset =
                        row * shape.quantized_row_bytes() + block_idx * shape.block_bytes;
                    let mut dst_block = [0u8; 14];
                    iq3_quantize_block(src_block, &mut dst_block);
                    out[dst_offset..dst_offset + 14].copy_from_slice(&dst_block);
                }
            }
        }
    }
    Ok(out)
}

pub fn q4_quantize_matrix_cpu(matrix: &[f32], shape: QGemvShape) -> Result<Vec<u8>> {
    quantize_matrix_cpu(QuantFormat::Q4_0, matrix, shape)
}

pub fn q5_quantize_matrix_cpu(matrix: &[f32], shape: QGemvShape) -> Result<Vec<u8>> {
    quantize_matrix_cpu(QuantFormat::Q5_0, matrix, shape)
}

pub fn iq3_quantize_matrix_cpu(matrix: &[f32], shape: QGemvShape) -> Result<Vec<u8>> {
    quantize_matrix_cpu(QuantFormat::IQ3_0, matrix, shape)
}

pub fn dense_gemv_cpu(matrix: &[f32], x: &[f32], shape: QGemvShape) -> Result<Vec<f32>> {
    if matrix.len() != shape.rows * shape.cols {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "dense_gemv_cpu",
            format!("matrix len {} != rows * cols {}", matrix.len(), shape.rows * shape.cols),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    if x.len() != shape.cols {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "dense_gemv_cpu",
            format!("vector len {} != cols {}", x.len(), shape.cols),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    let out = matrix
        .chunks(shape.cols)
        .map(|row| row.iter().zip(x.iter()).map(|(w, xv)| w * xv).sum())
        .collect();
    Ok(out)
}

pub fn gemv_cpu(
    format: QuantFormat,
    qweights: &[u8],
    x: &[f32],
    shape: QGemvShape,
) -> Result<Vec<f32>> {
    validate_quant_inputs(format, qweights, x, shape)?;
    let mut out = vec![0.0f32; shape.rows];
    for (row, out_cell) in out.iter_mut().enumerate() {
        let row_start = row * shape.quantized_row_bytes();
        let row_q = &qweights[row_start..row_start + shape.quantized_row_bytes()];
        *out_cell = match format {
            QuantFormat::Q4_0 => q4_dot_row(row_q, x, shape.cols, shape.blocks_per_row()),
            QuantFormat::Q5_0 => q5_dot_row(row_q, x, shape.cols, shape.blocks_per_row()),
            QuantFormat::IQ3_0 => iq3_dot_row(row_q, x, shape.cols, shape.blocks_per_row()),
        };
    }
    Ok(out)
}

pub fn q4_gemv_cpu(qweights: &[u8], x: &[f32], shape: QGemvShape) -> Result<Vec<f32>> {
    gemv_cpu(QuantFormat::Q4_0, qweights, x, shape)
}

pub fn q5_gemv_cpu(qweights: &[u8], x: &[f32], shape: QGemvShape) -> Result<Vec<f32>> {
    gemv_cpu(QuantFormat::Q5_0, qweights, x, shape)
}

pub fn iq3_gemv_cpu(qweights: &[u8], x: &[f32], shape: QGemvShape) -> Result<Vec<f32>> {
    gemv_cpu(QuantFormat::IQ3_0, qweights, x, shape)
}

pub fn q4_compare(reference: &[f32], actual: &[f32]) -> Result<QGemvNumerics> {
    compare_outputs(reference, actual)
}

pub fn q5_compare(reference: &[f32], actual: &[f32]) -> Result<QGemvNumerics> {
    compare_outputs(reference, actual)
}

pub fn compare_outputs(reference: &[f32], actual: &[f32]) -> Result<QGemvNumerics> {
    if reference.len() != actual.len() {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "compare quantized GEMV outputs",
            format!(
                "reference len {} != actual len {}",
                reference.len(),
                actual.len()
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    let mut dot = 0.0f64;
    let mut ref_norm = 0.0f64;
    let mut act_norm = 0.0f64;
    let mut sq_err = 0.0f64;
    let mut max_abs = 0.0f32;

    for (&r, &a) in reference.iter().zip(actual.iter()) {
        dot += (r as f64) * (a as f64);
        ref_norm += (r as f64) * (r as f64);
        act_norm += (a as f64) * (a as f64);
        let diff = a - r;
        sq_err += (diff as f64) * (diff as f64);
        max_abs = max_abs.max(diff.abs());
    }

    let cosine_similarity = if ref_norm > 0.0 && act_norm > 0.0 {
        (dot / (ref_norm.sqrt() * act_norm.sqrt())) as f32
    } else {
        1.0
    };
    let relative_l2_error = if ref_norm > 0.0 {
        (sq_err.sqrt() / ref_norm.sqrt()) as f32
    } else {
        0.0
    };

    Ok(QGemvNumerics {
        cosine_similarity,
        relative_l2_error,
        max_abs_error: max_abs,
    })
}

fn validate_dense_matrix(format: QuantFormat, matrix: &[f32], shape: QGemvShape) -> Result<()> {
    validate_shape_metadata(format, shape)?;
    let expected = shape.rows * shape.cols;
    if matrix.len() != expected {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate dense matrix size for {}", format.format_label()),
            format!("matrix len {} != rows*cols {}", matrix.len(), expected),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    if shape.cols % shape.block_size != 0 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate {} column alignment", format.format_label()),
            format!("cols {} must be a multiple of block_size {}", shape.cols, shape.block_size),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    Ok(())
}

fn validate_quant_inputs(
    format: QuantFormat,
    qweights: &[u8],
    x: &[f32],
    shape: QGemvShape,
) -> Result<()> {
    validate_shape_metadata(format, shape)?;
    if shape.cols % shape.block_size != 0 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate {} column alignment", format.format_label()),
            format!(
                "cols {} must be a multiple of block_size {}",
                shape.cols, shape.block_size
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    let expected_qbytes = shape.quantized_matrix_bytes();
    if qweights.len() != expected_qbytes {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate {} quantized matrix size", format.format_label()),
            format!("qweights len {} != expected {}", qweights.len(), expected_qbytes),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    if x.len() != shape.cols {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate {} GEMV vector size", format.format_label()),
            format!("x len {} != cols {}", x.len(), shape.cols),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    Ok(())
}

fn validate_shape_metadata(format: QuantFormat, shape: QGemvShape) -> Result<()> {
    let expected_block_size = format.block_size();
    let expected_block_bytes = format.block_bytes();
    if shape.block_size != expected_block_size || shape.block_bytes != expected_block_bytes {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate {} shape constants", format.format_label()),
            format!(
                "expected block_size={} block_bytes={}, got block_size={} block_bytes={}",
                expected_block_size, expected_block_bytes, shape.block_size, shape.block_bytes
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    Ok(())
}

fn q4_quantize_block(src: &[f32; 32], dst: &mut [u8; 18]) {
    let mut amax = 0.0f32;
    for &v in src {
        amax = amax.max(v.abs());
    }
    let scale = if amax > 0.0 { amax / 7.0 } else { 1.0e-10 };
    let scale_f16 = f32_to_f16_bits(scale);
    dst[0] = (scale_f16 & 0xFF) as u8;
    dst[1] = (scale_f16 >> 8) as u8;

    for i in 0..16 {
        let q0 = ((src[i * 2] / scale) + 8.0).round().clamp(0.0, 15.0) as u8;
        let q1 = ((src[i * 2 + 1] / scale) + 8.0).round().clamp(0.0, 15.0) as u8;
        dst[2 + i] = q0 | (q1 << 4);
    }
}

fn q5_quantize_block(src: &[f32; 32], dst: &mut [u8; 22]) {
    let mut amax = 0.0f32;
    for &v in src {
        amax = amax.max(v.abs());
    }
    let scale = if amax > 0.0 { amax / 15.0 } else { 1.0e-10 };
    let scale_f16 = f32_to_f16_bits(scale);
    dst[0] = (scale_f16 & 0xFF) as u8;
    dst[1] = (scale_f16 >> 8) as u8;
    dst[2..6].fill(0);

    for i in 0..32 {
        let q = ((src[i] / scale) + 16.0).round().clamp(0.0, 31.0) as u8;
        let low = q & 0x0F;
        let high = (q >> 4) & 0x01;

        let nibble_idx = i / 2;
        if i % 2 == 0 {
            dst[6 + nibble_idx] = low;
        } else {
            dst[6 + nibble_idx] |= low << 4;
        }

        if high != 0 {
            dst[2 + (i / 8)] |= 1 << (i % 8);
        }
    }
}

fn iq3_quantize_block(src: &[f32; 32], dst: &mut [u8; 14]) {
    let mut amax = 0.0f32;
    for &v in src {
        amax = amax.max(v.abs());
    }
    let scale = if amax > 0.0 { amax / 4.0 } else { 1.0e-10 };
    let scale_f16 = f32_to_f16_bits(scale);
    dst[0] = (scale_f16 & 0xFF) as u8;
    dst[1] = (scale_f16 >> 8) as u8;
    dst[2..14].fill(0);

    for i in 0..32 {
        let q = ((src[i] / scale) + 4.0).round().clamp(0.0, 7.0) as u8;
        let low = q & 0x03;
        let high = (q >> 2) & 0x01;

        dst[2 + (i / 4)] |= low << (2 * (i % 4));
        dst[10 + (i / 8)] |= high << (i % 8);
    }
}

fn q4_dot_row(row_q: &[u8], x: &[f32], cols: usize, blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for block_idx in 0..blocks_per_row {
        let block_start = block_idx * 18;
        let scale =
            f16_to_f32_bits(u16::from_le_bytes([row_q[block_start], row_q[block_start + 1]]));
        for lane in 0..32 {
            let col = block_idx * 32 + lane;
            let packed = row_q[block_start + 2 + (lane / 2)];
            let q = if lane % 2 == 0 {
                packed & 0x0F
            } else {
                packed >> 4
            };
            let w = (q as i32 - 8) as f32 * scale;
            if col < cols {
                sum += w * x[col];
            }
        }
    }
    sum
}

fn q5_dot_row(row_q: &[u8], x: &[f32], cols: usize, blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for block_idx in 0..blocks_per_row {
        let block_start = block_idx * 22;
        let scale =
            f16_to_f32_bits(u16::from_le_bytes([row_q[block_start], row_q[block_start + 1]]));
        for lane in 0..32 {
            let col = block_idx * 32 + lane;
            let low_byte = row_q[block_start + 6 + (lane / 2)];
            let low = if lane % 2 == 0 {
                low_byte & 0x0F
            } else {
                low_byte >> 4
            };
            let high = (row_q[block_start + 2 + (lane / 8)] >> (lane % 8)) & 0x01;
            let q = low | (high << 4);
            let w = (q as i32 - 16) as f32 * scale;
            if col < cols {
                sum += w * x[col];
            }
        }
    }
    sum
}

fn iq3_dot_row(row_q: &[u8], x: &[f32], cols: usize, blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for block_idx in 0..blocks_per_row {
        let block_start = block_idx * 14;
        let scale =
            f16_to_f32_bits(u16::from_le_bytes([row_q[block_start], row_q[block_start + 1]]));
        for lane in 0..32 {
            let col = block_idx * 32 + lane;
            let packed_low = row_q[block_start + 2 + (lane / 4)];
            let low = (packed_low >> (2 * (lane % 4))) & 0x03;
            let packed_high = row_q[block_start + 10 + (lane / 8)];
            let high = (packed_high >> (lane % 8)) & 0x01;
            let q = low | (high << 2);
            let w = (q as i32 - 4) as f32 * scale;
            if col < cols {
                sum += w * x[col];
            }
        }
    }
    sum
}

fn f16_to_f32_bits(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x03FF) as u32;
    let f_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut mantissa = mant;
                let mut exponent = -14i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03FF;
                sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
            }
        }
        0x1F => sign | 0x7F80_0000 | (mant << 13),
        _ => sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13),
    };
    f32::from_bits(f_bits)
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 255 {
        let nan_bits = if mant == 0 { 0 } else { 0x0200 };
        return sign | 0x7C00 | nan_bits;
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1F {
        return sign | 0x7C00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x80_0000;
        let shift = 14 - half_exp;
        let mut half = (mantissa >> shift) as u16;
        if (mantissa >> (shift - 1)) & 1 != 0 {
            half += 1;
        }
        return sign | half;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x0000_1000) != 0 {
        half += 1;
    }
    half
}

pub(crate) fn compile_kernel_ptx(
    source: &str,
    source_name: &'static str,
    major: i32,
    minor: i32,
) -> Result<Ptx> {
    let arch = match (major, minor) {
        (5, 0) => Some("compute_50"),
        (5, 2) => Some("compute_52"),
        (6, 0) => Some("compute_60"),
        (6, 1) => Some("compute_61"),
        (7, 0) => Some("compute_70"),
        (7, 5) => Some("compute_75"),
        (8, 0) => Some("compute_80"),
        (8, 6) => Some("compute_86"),
        (8, 9) => Some("compute_89"),
        (9, 0) => Some("compute_90"),
        _ => None,
    };
    let opts = CompileOptions {
        arch,
        include_paths: Vec::new(),
        name: Some(source_name.into()),
        ..Default::default()
    };
    cuda_map_err!(
        CudaErrorKind::Nvrtc,
        format!(
            "compile CUDA source {} for compute capability {}.{}",
            source_name, major, minor
        ),
        compile_ptx_with_opts(source, opts)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CudaBackendBuilder;

    fn seeded_f32s(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((state >> 32) as u32) | 1;
            let unit = (bits as f32) / (u32::MAX as f32);
            out.push((unit * 2.0) - 1.0);
        }
        out
    }

    fn run_case(
        format: QuantFormat,
        rows: usize,
        cols: usize,
        seed: u64,
    ) -> Result<QGemvReport> {
        let shape = QGemvShape::new(format, rows, cols);
        let matrix = seeded_f32s(rows * cols, seed ^ 0xA5A5_5A5A_1234_5678);
        let x = seeded_f32s(cols, seed ^ 0xD00D_F00D_1234_5678);

        let backend = CudaBackendBuilder::new().stream_count(2).build()?;
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0)?;

        let qweights = quant.quantize_matrix(format, &matrix, shape)?;
        let reference = gemv_cpu(format, &qweights, &x, shape)?;
        let (actual, telemetry) = quant.gemv(format, stream, &qweights, &x, shape)?;
        let cuda_vs_cpu_quant = compare_outputs(&reference, &actual)?;
        let dense_ref = dense_gemv_cpu(&matrix, &x, shape)?;
        let quant_vs_fp32 = compare_outputs(&dense_ref, &actual)?;

        let numerics = QGemvNumericsSuite {
            cosine_similarity: cuda_vs_cpu_quant.cosine_similarity,
            relative_l2_error: cuda_vs_cpu_quant.relative_l2_error,
            max_abs_error: cuda_vs_cpu_quant.max_abs_error,
            cuda_vs_cpu_quant,
            quant_vs_fp32,
        };

        Ok(QGemvReport {
            backend: "cuda",
            kernel: format.kernel_name(),
            format,
            rows,
            cols,
            block_size: shape.block_size,
            block_bytes: shape.block_bytes,
            telemetry,
            numerics,
        })
    }

    #[test]
    fn q4_rejects_non_multiple_cols() {
        let shape = QGemvShape::q4_0(4, 33);
        let matrix = vec![0.0f32; shape.rows * shape.cols];
        let err = q4_quantize_matrix_cpu(&matrix, shape).unwrap_err();
        assert!(err.source_message.contains("multiple of block_size"));
    }

    #[test]
    fn q4_cpu_reference_roundtrip_small() -> Result<()> {
        let shape = QGemvShape::q4_0(16, 32);
        let matrix = seeded_f32s(shape.rows * shape.cols, 42);
        let x = seeded_f32s(shape.cols, 7);
        let qweights = q4_quantize_matrix_cpu(&matrix, shape)?;
        let reference = q4_gemv_cpu(&qweights, &x, shape)?;
        let dense_reference = dense_gemv_cpu(&matrix, &x, shape)?;
        let numerics = compare_outputs(&dense_reference, &reference)?;
        println!(
            "q4 cpu ref rows={} cols={} cosine={:.6} rel_l2={:.6} max_abs={:.6}",
            shape.rows,
            shape.cols,
            numerics.cosine_similarity,
            numerics.relative_l2_error,
            numerics.max_abs_error
        );
        assert!(numerics.cosine_similarity > 0.995);
        Ok(())
    }

    #[test]
    fn q5_cpu_reference_roundtrip_small() -> Result<()> {
        let shape = QGemvShape::q5_0(16, 32);
        let matrix = seeded_f32s(shape.rows * shape.cols, 43);
        let x = seeded_f32s(shape.cols, 8);
        let qweights = q5_quantize_matrix_cpu(&matrix, shape)?;
        let reference = q5_gemv_cpu(&qweights, &x, shape)?;
        let dense_reference = dense_gemv_cpu(&matrix, &x, shape)?;
        let numerics = compare_outputs(&dense_reference, &reference)?;
        println!(
            "q5 cpu ref rows={} cols={} cosine={:.6} rel_l2={:.6} max_abs={:.6}",
            shape.rows,
            shape.cols,
            numerics.cosine_similarity,
            numerics.relative_l2_error,
            numerics.max_abs_error
        );
        assert!(numerics.cosine_similarity > 0.998);
        Ok(())
    }

    #[test]
    fn q4_correctness_and_seed_sweep() -> Result<()> {
        let cases = &[
            (1, 32),
            (16, 32),
            (128, 256),
            (1024, 4096),
        ];
        let seeds = &[0, 1, 2, 3, 4, 123];
        for &(rows, cols) in cases {
            for &seed in seeds {
                let report = run_case(QuantFormat::Q4_0, rows, cols, seed)?;
                println!(
                    "Q4 CUDA correctness check: rows={}, cols={}, seed={} -> cosine={:.6} rel_l2={:.6} max_abs={:.6}",
                    rows, cols, seed,
                    report.numerics.cosine_similarity,
                    report.numerics.relative_l2_error,
                    report.numerics.max_abs_error
                );
                assert!(
                    report.numerics.cosine_similarity > 0.990,
                    "Q4 fail: rows={}, cols={}, seed={}, cosine={}",
                    rows, cols, seed, report.numerics.cosine_similarity
                );
            }
        }
        Ok(())
    }

    #[test]
    fn q5_correctness_and_seed_sweep() -> Result<()> {
        let cases = &[
            (1, 32),
            (16, 32),
            (128, 256),
            (1024, 4096),
        ];
        let seeds = &[0, 1, 2, 3, 4, 123];
        for &(rows, cols) in cases {
            for &seed in seeds {
                let report = run_case(QuantFormat::Q5_0, rows, cols, seed)?;
                println!(
                    "Q5 CUDA correctness check: rows={}, cols={}, seed={} -> cosine={:.6} rel_l2={:.6} max_abs={:.6}",
                    rows, cols, seed,
                    report.numerics.cosine_similarity,
                    report.numerics.relative_l2_error,
                    report.numerics.max_abs_error
                );
                assert!(
                    report.numerics.cosine_similarity > 0.995,
                    "Q5 fail: rows={}, cols={}, seed={}, cosine={}",
                    rows, cols, seed, report.numerics.cosine_similarity
                );
            }
        }
        Ok(())
    }

    #[test]
    fn q5_is_no_worse_than_q4_on_same_case() -> Result<()> {
        let rows = 128;
        let cols = 256;
        let seed = 77;
        let shape_q4 = QGemvShape::q4_0(rows, cols);
        let shape_q5 = QGemvShape::q5_0(rows, cols);
        let matrix = seeded_f32s(rows * cols, seed ^ 0x1234_5678_9ABC_DEF0);
        let x = seeded_f32s(cols, seed ^ 0x0BAD_F00D_1234_5678);

        let dense_reference = dense_gemv_cpu(&matrix, &x, shape_q4)?;
        let q4 = q4_gemv_cpu(&q4_quantize_matrix_cpu(&matrix, shape_q4)?, &x, shape_q4)?;
        let q5 = q5_gemv_cpu(&q5_quantize_matrix_cpu(&matrix, shape_q5)?, &x, shape_q5)?;
        let q4n = compare_outputs(&dense_reference, &q4)?;
        let q5n = compare_outputs(&dense_reference, &q5)?;

        println!(
            "q4 cosine={:.6} rel_l2={:.6} | q5 cosine={:.6} rel_l2={:.6}",
            q4n.cosine_similarity,
            q4n.relative_l2_error,
            q5n.cosine_similarity,
            q5n.relative_l2_error
        );

        assert!(q5n.cosine_similarity >= q4n.cosine_similarity - 1.0e-6);
        assert!(q5n.relative_l2_error <= q4n.relative_l2_error + 1.0e-6);
        Ok(())
    }

    #[test]
    fn iq3_rejects_non_multiple_cols() {
        let shape = QGemvShape::iq3_0(4, 33);
        let matrix = vec![0.0f32; shape.rows * shape.cols];
        let err = iq3_quantize_matrix_cpu(&matrix, shape).unwrap_err();
        assert!(err.source_message.contains("multiple of block_size"));
    }

    #[test]
    fn iq3_cpu_reference_roundtrip_small() -> Result<()> {
        let shape = QGemvShape::iq3_0(16, 32);
        let matrix = seeded_f32s(shape.rows * shape.cols, 44);
        let x = seeded_f32s(shape.cols, 9);
        let qweights = iq3_quantize_matrix_cpu(&matrix, shape)?;
        let reference = iq3_gemv_cpu(&qweights, &x, shape)?;
        let dense_reference = dense_gemv_cpu(&matrix, &x, shape)?;
        let numerics = compare_outputs(&dense_reference, &reference)?;
        println!(
            "iq3 cpu ref rows={} cols={} cosine={:.6} rel_l2={:.6} max_abs={:.6}",
            shape.rows,
            shape.cols,
            numerics.cosine_similarity,
            numerics.relative_l2_error,
            numerics.max_abs_error
        );
        assert!(numerics.cosine_similarity > 0.970);
        Ok(())
    }

    #[test]
    fn iq3_correctness_and_seed_sweep() -> Result<()> {
        let cases = &[
            (1, 32),
            (16, 32),
            (128, 256),
            (1024, 4096),
        ];
        let seeds = &[0, 1, 2, 3, 4, 123];
        for &(rows, cols) in cases {
            for &seed in seeds {
                let report = run_case(QuantFormat::IQ3_0, rows, cols, seed)?;
                println!(
                    "IQ3 CUDA correctness check: rows={}, cols={}, seed={} -> cosine={:.6} rel_l2={:.6} max_abs={:.6}",
                    rows, cols, seed,
                    report.numerics.cosine_similarity,
                    report.numerics.relative_l2_error,
                    report.numerics.max_abs_error
                );
                assert!(
                    report.numerics.cosine_similarity > 0.970,
                    "IQ3 fail: rows={}, cols={}, seed={}, cosine={}",
                    rows, cols, seed, report.numerics.cosine_similarity
                );
            }
        }
        Ok(())
    }
}
