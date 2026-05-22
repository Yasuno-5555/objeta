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
const FP4_KERNEL_SRC: &str = include_str!("../kernels/deepseek_fp4_gemv.cu");
const ACT_QUANT_KERNEL_SRC: &str = include_str!("../kernels/deepseek_act_quant.cu");
const ACT_QUANT_KERNEL_NAME: &str = "act_quant_fp8_e4m3_e8m0";
const FP8_ACT_FP4_WT_KERNEL_SRC: &str = include_str!("../kernels/deepseek_fp8_act_fp4_weight_gemv.cu");
const FP8_ACT_FP4_WT_KERNEL_NAME: &str = "fp8_act_fp4_weight_gemv";
const FP8_ACT_FP8_WT_KERNEL_SRC: &str = include_str!("../kernels/deepseek_fp8_act_fp8_weight_gemv.cu");
const FP8_ACT_FP8_WT_KERNEL_NAME: &str = "fp8_act_fp8_weight_gemv";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QuantFormat {
    Q4_0,
    Q5_0,
    IQ3_0,
    DeepSeekFp4E2M1,
}

impl QuantFormat {
    pub fn block_size(&self) -> usize {
        match self {
            QuantFormat::Q4_0 => 32,
            QuantFormat::Q5_0 => 32,
            QuantFormat::IQ3_0 => 32,
            QuantFormat::DeepSeekFp4E2M1 => 32,
        }
    }

    pub fn block_bytes(&self) -> usize {
        match self {
            QuantFormat::Q4_0 => 18,
            QuantFormat::Q5_0 => 22,
            QuantFormat::IQ3_0 => 14,
            QuantFormat::DeepSeekFp4E2M1 => 17,
        }
    }

    pub fn kernel_name(&self) -> &'static str {
        match self {
            QuantFormat::Q4_0 => "q4_gemv",
            QuantFormat::Q5_0 => "q5_gemv",
            QuantFormat::IQ3_0 => "iq3_gemv",
            QuantFormat::DeepSeekFp4E2M1 => "fp4_e2m1_gemv",
        }
    }

    pub fn format_label(&self) -> &'static str {
        match self {
            QuantFormat::Q4_0 => "q4_0",
            QuantFormat::Q5_0 => "q5_0",
            QuantFormat::IQ3_0 => "iq3_0",
            QuantFormat::DeepSeekFp4E2M1 => "deepseek_fp4_e2m1",
        }
    }

    pub fn packed_values_per_byte(&self) -> usize {
        match self {
            QuantFormat::Q4_0 => 2,
            QuantFormat::Q5_0 => 2,
            QuantFormat::IQ3_0 => 2,
            QuantFormat::DeepSeekFp4E2M1 => 2,
        }
    }

    pub fn scale_dtype(&self) -> &'static str {
        match self {
            QuantFormat::Q4_0 => "F16",
            QuantFormat::Q5_0 => "F16",
            QuantFormat::IQ3_0 => "F16",
            QuantFormat::DeepSeekFp4E2M1 => "F8_E8M0",
        }
    }

    pub fn nibble_order(&self) -> &'static str {
        match self {
            QuantFormat::Q4_0 => "low_first",
            QuantFormat::Q5_0 => "low_first",
            QuantFormat::IQ3_0 => "low_first",
            QuantFormat::DeepSeekFp4E2M1 => "low_first",
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

    pub fn deepseek_fp4(rows: usize, cols: usize) -> Self {
        Self::new(QuantFormat::DeepSeekFp4E2M1, rows, cols)
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
    gemv_split: Option<CudaFunction>,
}

#[derive(Debug)]
pub struct QuantBackend {
    context: Arc<CudaContext>,
    device_info: CudaDeviceInfo,
    q4_module: Mutex<Option<QuantKernelModule>>,
    q5_module: Mutex<Option<QuantKernelModule>>,
    iq3_module: Mutex<Option<QuantKernelModule>>,
    fp4_module: Mutex<Option<QuantKernelModule>>,
}

impl QuantBackend {
    pub fn new(context: Arc<CudaContext>, device_info: CudaDeviceInfo) -> Self {
        Self {
            context,
            device_info,
            q4_module: Mutex::new(None),
            q5_module: Mutex::new(None),
            iq3_module: Mutex::new(None),
            fp4_module: Mutex::new(None),
        }
    }

    pub fn context(&self) -> &Arc<CudaContext> { &self.context }
    pub fn device_info(&self) -> &CudaDeviceInfo { &self.device_info }

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
            QuantFormat::DeepSeekFp4E2M1 => {
                let _unused = self.fp4_module()?;
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
            QuantFormat::DeepSeekFp4E2M1 => self.fp4_module()?,
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

    pub(crate) fn launch_kernel_split_fp4(
        &self,
        stream: &Arc<CudaStream>,
        qweights: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let module_guard = self.fp4_module()?;
        let module = module_guard.as_ref().expect("fp4 kernel module must exist");
        let gemv_split = module.gemv_split.as_ref().expect("gemv_split function must exist");
        
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let blocks_per_row = (cols / 32) as u32;
        
        let cfg = LaunchConfig {
            grid_dim: (rows_u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_map_err!(
            CudaErrorKind::Driver,
            "launch fp4_e2m1_gemv_split kernel",
            unsafe {
                stream
                    .launch_builder(gemv_split)
                    .arg(&qweights.raw)
                    .arg(&scales.raw)
                    .arg(&x.raw)
                    .arg(&mut y.raw)
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .arg(&blocks_per_row)
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

    fn fp4_module(&self) -> Result<MutexGuard<'_, Option<QuantKernelModule>>> {
        self.ensure_module(
            &self.fp4_module,
            QuantFormat::DeepSeekFp4E2M1,
            FP4_KERNEL_SRC,
            "objeta_cuda_deepseek_fp4_gemv.cu",
            "fp4_e2m1_gemv",
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
            let gemv_split = if format == QuantFormat::DeepSeekFp4E2M1 {
                let func = cuda_map_err!(
                    CudaErrorKind::Driver,
                    "load fp4 split kernel function fp4_e2m1_gemv_split".to_string(),
                    module.load_function("fp4_e2m1_gemv_split")
                )?;
                Some(func)
            } else {
                None
            };
            *guard = Some(QuantKernelModule {
                _module: module,
                gemv,
                gemv_split,
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
        QuantFormat::DeepSeekFp4E2M1 => {
            for row in 0..shape.rows {
                let src = &matrix[row * shape.cols..(row + 1) * shape.cols];
                for block_idx in 0..shape.blocks_per_row() {
                    let src_block: &[f32; 32] = src
                        [block_idx * 32..(block_idx + 1) * 32]
                        .try_into()
                        .expect("fixed-size fp4 block");
                    let dst_offset =
                        row * shape.quantized_row_bytes() + block_idx * shape.block_bytes;
                    let mut dst_block = [0u8; 17];
                    fp4_quantize_block(src_block, &mut dst_block);
                    out[dst_offset..dst_offset + 17].copy_from_slice(&dst_block);
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

pub fn fp4_quantize_matrix_cpu(matrix: &[f32], shape: QGemvShape) -> Result<Vec<u8>> {
    quantize_matrix_cpu(QuantFormat::DeepSeekFp4E2M1, matrix, shape)
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
            QuantFormat::DeepSeekFp4E2M1 => fp4_dot_row(row_q, x, shape.cols, shape.blocks_per_row()),
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

pub fn fp4_gemv_cpu(qweights: &[u8], x: &[f32], shape: QGemvShape) -> Result<Vec<f32>> {
    gemv_cpu(QuantFormat::DeepSeekFp4E2M1, qweights, x, shape)
}

pub fn q4_compare(reference: &[f32], actual: &[f32]) -> Result<QGemvNumerics> {
    compare_outputs(reference, actual)
}

pub fn q5_compare(reference: &[f32], actual: &[f32]) -> Result<QGemvNumerics> {
    compare_outputs(reference, actual)
}

pub fn fp4_compare(reference: &[f32], actual: &[f32]) -> Result<QGemvNumerics> {
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

const FP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn decode_f8_e8m0(raw: u8) -> f32 {
    if raw == 0 {
        f32::from_bits(1 << 22)
    } else {
        f32::from_bits((raw as u32) << 23)
    }
}

fn quantize_val(val: f32, scale: f32) -> u8 {
    let scaled = val / scale;
    let mut best_q = 0;
    let mut min_diff = f32::INFINITY;
    for q in 0..16 {
        let diff = (scaled - FP4_TABLE[q]).abs();
        if diff < min_diff {
            min_diff = diff;
            best_q = q;
        }
    }
    best_q as u8
}

fn fp4_quantize_block(src: &[f32; 32], dst: &mut [u8; 17]) {
    let mut amax = 0.0f32;
    for &v in src {
        amax = amax.max(v.abs());
    }
    let raw = if amax > 0.0 && amax.is_finite() {
        let k = (amax / 6.0).log2().ceil() as i32;
        (k + 127).clamp(0, 255) as u8
    } else {
        0
    };
    dst[0] = raw;
    let actual_scale = decode_f8_e8m0(raw);
    for i in 0..16 {
        let q0 = quantize_val(src[i * 2], actual_scale);
        let q1 = quantize_val(src[i * 2 + 1], actual_scale);
        dst[1 + i] = q0 | (q1 << 4);
    }
}

fn fp4_dot_row(row_q: &[u8], x: &[f32], cols: usize, blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for block_idx in 0..blocks_per_row {
        let block_start = block_idx * 17;
        let scale_raw = row_q[block_start];
        let scale = decode_f8_e8m0(scale_raw);
        for lane in 0..32 {
            let col = block_idx * 32 + lane;
            if col < cols {
                let packed = row_q[block_start + 1 + (lane / 2)];
                let q = if lane % 2 == 0 {
                    packed & 0x0F
                } else {
                    packed >> 4
                };
                let w = FP4_TABLE[q as usize] * scale;
                sum += w * x[col];
            }
        }
    }
    sum
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

// ── CUDA act_quant ─────────────────────────────────────────────────────────

/// CUDA act_quant: FP8 E4M3 activation quantization with E8M0 power-of-2 scales.
pub fn cuda_act_quant(
    backend: &QuantBackend,
    stream: &crate::stream::CudaStreamHandle,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, Vec<u8>, QGemvTelemetry)> {
    use std::sync::{Arc, Mutex, OnceLock};

    if cols % 128 != 0 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput, "cuda_act_quant",
            format!("cols {} not divisible by 128", cols),
            file!(), line!(), module_path!(),
        ));
    }
    if input.len() != rows * cols {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput, "cuda_act_quant",
            format!("input len {} != rows {} * cols {}", input.len(), rows, cols),
            file!(), line!(), module_path!(),
        ));
    }

    let num_blocks_per_row = cols / 128;

    static ACT_QUANT_MODULE: OnceLock<Mutex<Option<(Arc<CudaModule>, CudaFunction)>>> = OnceLock::new();
    let slot = ACT_QUANT_MODULE.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().map_err(|e| CudaError::new(
        CudaErrorKind::Internal, "lock act_quant module", e.to_string(), file!(), line!(), module_path!(),
    ))?;

    if guard.is_none() {
        let ptx = compile_kernel_ptx(
            ACT_QUANT_KERNEL_SRC, "objeta_cuda_act_quant.cu",
            backend.device_info.compute_capability_major,
            backend.device_info.compute_capability_minor,
        )?;
        let module = cuda_map_err!(
            CudaErrorKind::Driver, "load act_quant PTX module",
            backend.context.load_module(ptx)
        )?;
        let func = cuda_map_err!(
            CudaErrorKind::Driver, "load act_quant kernel",
            module.load_function(ACT_QUANT_KERNEL_NAME)
        )?;
        *guard = Some((module, func));
    }
    let (_, func) = guard.as_ref().unwrap();

    let total_timer = CudaEventTimer::start(stream.raw())?;
    let h2d_timer = CudaEventTimer::start(stream.raw())?;
    let d_input = stream.copy_from_slice(input)?;
    let h2d_ms = h2d_timer.stop("act_quant_h2d", stream.raw())?.elapsed_ms;

    let mut d_values = stream.alloc_zeros::<u8>(rows * cols)?;
    let mut d_scales = stream.alloc_zeros::<u8>(rows * num_blocks_per_row)?;

    let kernel_timer = CudaEventTimer::start(stream.raw())?;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks_per_row as u32, rows as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    cuda_map_err!(
        CudaErrorKind::Driver, "launch act_quant kernel",
        unsafe {
            stream.raw().launch_builder(func)
                .arg(&d_input.raw)
                .arg(&d_values.raw)
                .arg(&d_scales.raw)
                .arg(&rows_i32)
                .arg(&cols_i32)
                .launch(cfg)
        }
    )?;

    let kernel_ms = kernel_timer.stop("act_quant_kernel", stream.raw())?.elapsed_ms;
    let d2h_timer = CudaEventTimer::start(stream.raw())?;
    let values = stream.copy_to_vec(&d_values)?;
    let scales = stream.copy_to_vec(&d_scales)?;
    let d2h_ms = d2h_timer.stop("act_quant_d2h", stream.raw())?.elapsed_ms;
    let total_ms = total_timer.stop("act_quant_total", stream.raw())?.elapsed_ms;

    let unaccounted_ms = (total_ms - h2d_ms - kernel_ms - d2h_ms).max(0.0);
    let bytes_read = (rows * cols) * std::mem::size_of::<f32>();

    Ok((values, scales, QGemvTelemetry {
        h2d_ms, kernel_ms, d2h_ms, unaccounted_ms, total_ms, bytes_read,
        effective_gbps: 0.0,
    }))
}

// ── Device-resident FP8 activation ─────────────────────────────────────────

/// Device-side quantized activation for official arithmetic pipeline.
pub struct DeviceFp8Activation {
    pub values: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u8>,
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
}

/// Device-resident act_quant: quantizes activation on GPU, returns device buffers.
/// Does NOT copy results back to host.
pub fn cuda_act_quant_device(
    backend: &QuantBackend,
    stream: &crate::stream::CudaStreamHandle,
    d_input: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<(DeviceFp8Activation, f32)> {
    use std::sync::{Arc, Mutex, OnceLock};

    if cols % 128 != 0 {
        return Err(CudaError::new(CudaErrorKind::InvalidInput, "cuda_act_quant_device",
            format!("cols {} not divisible by 128", cols), file!(), line!(), module_path!()));
    }

    let num_blocks_per_row = cols / 128;
    let total_values = rows * cols;
    let total_scales = rows * num_blocks_per_row;

    static MODULE: OnceLock<Mutex<Option<(Arc<CudaModule>, CudaFunction)>>> = OnceLock::new();
    let slot = MODULE.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().map_err(|e| CudaError::new(
        CudaErrorKind::Internal, "lock act_quant_device module", e.to_string(), file!(), line!(), module_path!()))?;

    if guard.is_none() {
        let ptx = compile_kernel_ptx(ACT_QUANT_KERNEL_SRC, "objeta_cuda_act_quant_d.cu",
            backend.device_info.compute_capability_major, backend.device_info.compute_capability_minor)?;
        let module = cuda_map_err!(CudaErrorKind::Driver, "load act_quant_d PTX", backend.context.load_module(ptx))?;
        let func = cuda_map_err!(CudaErrorKind::Driver, "load act_quant_d kernel", module.load_function(ACT_QUANT_KERNEL_NAME))?;
        *guard = Some((module, func));
    }
    let (_, func) = guard.as_ref().unwrap();

    let mut d_values = stream.alloc_zeros::<u8>(total_values)?;
    let mut d_scales = stream.alloc_zeros::<u8>(total_scales)?;

    let kernel_timer = CudaEventTimer::start(stream.raw())?;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;
    let cfg = LaunchConfig { grid_dim: (num_blocks_per_row as u32, rows as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };

    cuda_map_err!(CudaErrorKind::Driver, "launch act_quant_device kernel",
        unsafe { stream.raw().launch_builder(func)
            .arg(&d_input.raw).arg(&d_values.raw).arg(&d_scales.raw)
            .arg(&rows_i32).arg(&cols_i32).launch(cfg) })?;

    let kernel_ms = kernel_timer.stop("act_quant_device", stream.raw())?.elapsed_ms;

    Ok((DeviceFp8Activation {
        values: d_values, scales: d_scales, rows, cols, block_size: 128,
    }, kernel_ms))
}

/// Device-resident FP8 act × FP4 weight GEMV.
/// Uses device-side activation buffers directly, writes to device output buffer.
/// No host roundtrip.
pub fn cuda_fp8_act_fp4_weight_gemv_device(
    backend: &QuantBackend,
    stream: &crate::stream::CudaStreamHandle,
    d_act_values: &DeviceBuffer<u8>,
    d_act_scales: &DeviceBuffer<u8>,
    d_weight_packed: &DeviceBuffer<u8>,
    d_weight_scales: &DeviceBuffer<u8>,
    d_output: &mut DeviceBuffer<f32>,
    rows: usize,
    k_logical: usize,
) -> Result<f32> {
    use std::sync::{Arc, Mutex, OnceLock};

    static MODULE: OnceLock<Mutex<Option<(Arc<CudaModule>, CudaFunction)>>> = OnceLock::new();
    let slot = MODULE.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().map_err(|e| CudaError::new(
        CudaErrorKind::Internal, "lock gemv_device module", e.to_string(), file!(), line!(), module_path!()))?;

    if guard.is_none() {
        let ptx = compile_kernel_ptx(FP8_ACT_FP4_WT_KERNEL_SRC, "objeta_gemv_device.cu",
            backend.device_info.compute_capability_major, backend.device_info.compute_capability_minor)?;
        let module = cuda_map_err!(CudaErrorKind::Driver, "load gemv_device PTX", backend.context.load_module(ptx))?;
        let func = cuda_map_err!(CudaErrorKind::Driver, "load gemv_device kernel", module.load_function(FP8_ACT_FP4_WT_KERNEL_NAME))?;
        *guard = Some((module, func));
    }
    let (_, func) = guard.as_ref().unwrap();

    let kernel_timer = CudaEventTimer::start(stream.raw())?;
    let rows_u = rows as u32;
    let k_u = k_logical as u32;
    let cfg = LaunchConfig { grid_dim: (rows_u, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };

    cuda_map_err!(CudaErrorKind::Driver, "launch gemv_device kernel",
        unsafe { stream.raw().launch_builder(func)
            .arg(&d_act_values.raw).arg(&d_act_scales.raw)
            .arg(&d_weight_packed.raw).arg(&d_weight_scales.raw)
            .arg(&d_output.raw)
            .arg(&rows_u).arg(&k_u).launch(cfg) })?;

    let ms = kernel_timer.stop("gemv_device", stream.raw())?.elapsed_ms;
    Ok(ms)
}

/// Device-resident FP8 act × FP8 weight GEMV (official shared expert).
pub fn cuda_fp8_act_fp8_weight_gemv_device(
    backend: &QuantBackend,
    stream: &crate::stream::CudaStreamHandle,
    d_act_values: &DeviceBuffer<u8>,
    d_act_scales: &DeviceBuffer<u8>,
    d_weight: &DeviceBuffer<u8>,
    d_weight_scales: &DeviceBuffer<u8>,
    d_output: &mut DeviceBuffer<f32>,
    rows: usize,
    k_logical: usize,
) -> Result<f32> {
    use std::sync::{Arc, Mutex, OnceLock};
    static MODULE: OnceLock<Mutex<Option<(Arc<CudaModule>, CudaFunction)>>> = OnceLock::new();
    let slot = MODULE.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().map_err(|e| CudaError::new(
        CudaErrorKind::Internal, "lock fp8_act_fp8_wt module", e.to_string(), file!(), line!(), module_path!()))?;
    if guard.is_none() {
        let ptx = compile_kernel_ptx(FP8_ACT_FP8_WT_KERNEL_SRC, "objeta_fp8_act_fp8_wt.cu",
            backend.device_info.compute_capability_major, backend.device_info.compute_capability_minor)?;
        let module = cuda_map_err!(CudaErrorKind::Driver, "load fp8_act_fp8_wt PTX", backend.context.load_module(ptx))?;
        let func = cuda_map_err!(CudaErrorKind::Driver, "load fp8_act_fp8_wt kernel", module.load_function(FP8_ACT_FP8_WT_KERNEL_NAME))?;
        *guard = Some((module, func));
    }
    let (_, func) = guard.as_ref().unwrap();
    let kernel_timer = CudaEventTimer::start(stream.raw())?;
    let rows_u = rows as u32; let k_u = k_logical as u32;
    let cfg = LaunchConfig { grid_dim: (rows_u, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    cuda_map_err!(CudaErrorKind::Driver, "launch fp8_act_fp8_wt kernel",
        unsafe { stream.raw().launch_builder(func)
            .arg(&d_act_values.raw).arg(&d_act_scales.raw)
            .arg(&d_weight.raw).arg(&d_weight_scales.raw)
            .arg(&d_output.raw).arg(&rows_u).arg(&k_u).launch(cfg) })?;
    let ms = kernel_timer.stop("fp8_act_fp8_wt", stream.raw())?.elapsed_ms;
    Ok(ms)
}

/// CUDA FP8 activation × FP4 weight GEMV (official routed expert Linear).
pub fn cuda_fp8_act_fp4_weight_gemv(
    backend: &QuantBackend,
    stream: &crate::stream::CudaStreamHandle,
    act_values: &[u8],
    act_scales: &[u8],
    weight_packed: &[u8],
    weight_scales: &[u8],
    rows: usize,
    K_logical: usize,
) -> Result<(Vec<f32>, QGemvTelemetry)> {
    use std::sync::{Arc, Mutex, OnceLock};

    if K_logical % 2 != 0 || K_logical % 32 != 0 || K_logical % 128 != 0 {
        return Err(CudaError::new(CudaErrorKind::InvalidInput, "cuda_fp8_act_fp4_weight_gemv",
            format!("K_logical {} must be multiple of 128", K_logical),
            file!(), line!(), module_path!()));
    }

    let K_phys = K_logical / 2;
    let expected_act_v = K_logical;
    let expected_act_s = K_logical / 128;
    let expected_wt_v = rows * K_phys;
    let expected_wt_s = rows * (K_logical / 32);

    if act_values.len() != expected_act_v || act_scales.len() != expected_act_s
        || weight_packed.len() != expected_wt_v || weight_scales.len() != expected_wt_s
    {
        return Err(CudaError::new(CudaErrorKind::InvalidInput, "cuda_fp8_act_fp4_weight_gemv",
            format!("size mismatch: act_v={}/{}, act_s={}/{}, wt={}/{}, wt_s={}/{}",
                act_values.len(), expected_act_v, act_scales.len(), expected_act_s,
                weight_packed.len(), expected_wt_v, weight_scales.len(), expected_wt_s),
            file!(), line!(), module_path!()));
    }

    static MODULE: OnceLock<Mutex<Option<(Arc<CudaModule>, CudaFunction)>>> = OnceLock::new();
    let slot = MODULE.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().map_err(|e| CudaError::new(
        CudaErrorKind::Internal, "lock fp8_act_fp4_wt module", e.to_string(), file!(), line!(), module_path!()))?;

    if guard.is_none() {
        let ptx = compile_kernel_ptx(FP8_ACT_FP4_WT_KERNEL_SRC, "objeta_fp8_act_fp4_wt_gemv.cu",
            backend.device_info.compute_capability_major, backend.device_info.compute_capability_minor)?;
        let module = cuda_map_err!(CudaErrorKind::Driver, "load fp8_act_fp4_wt PTX", backend.context.load_module(ptx))?;
        let func = cuda_map_err!(CudaErrorKind::Driver, "load fp8_act_fp4_wt kernel", module.load_function(FP8_ACT_FP4_WT_KERNEL_NAME))?;
        *guard = Some((module, func));
    }
    let (_, func) = guard.as_ref().unwrap();

    let total_timer = CudaEventTimer::start(stream.raw())?;
    let h2d_timer = CudaEventTimer::start(stream.raw())?;
    let d_act_v = stream.copy_from_slice(act_values)?;
    let d_act_s = stream.copy_from_slice(act_scales)?;
    let d_wt = stream.copy_from_slice(weight_packed)?;
    let d_wt_s = stream.copy_from_slice(weight_scales)?;
    let h2d_ms = h2d_timer.stop("fp8_act_fp4_wt_h2d", stream.raw())?.elapsed_ms;

    let mut d_y = stream.alloc_zeros::<f32>(rows)?;
    let kernel_timer = CudaEventTimer::start(stream.raw())?;
    let rows_u = rows as u32;
    let K_u = K_logical as u32;
    let cfg = LaunchConfig { grid_dim: (rows_u, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };

    cuda_map_err!(CudaErrorKind::Driver, "launch fp8_act_fp4_wt kernel",
        unsafe { stream.raw().launch_builder(func)
            .arg(&d_act_v.raw).arg(&d_act_s.raw).arg(&d_wt.raw).arg(&d_wt_s.raw).arg(&d_y.raw)
            .arg(&rows_u).arg(&K_u).launch(cfg) })?;

    let kernel_ms = kernel_timer.stop("fp8_act_fp4_wt_kernel", stream.raw())?.elapsed_ms;
    let d2h_timer = CudaEventTimer::start(stream.raw())?;
    let y = stream.copy_to_vec(&d_y)?;
    let d2h_ms = d2h_timer.stop("fp8_act_fp4_wt_d2h", stream.raw())?.elapsed_ms;
    let total_ms = total_timer.stop("fp8_act_fp4_wt_total", stream.raw())?.elapsed_ms;
    let unaccounted_ms = (total_ms - h2d_ms - kernel_ms - d2h_ms).max(0.0);
    let bytes_read = act_values.len() + act_scales.len() + weight_packed.len() + weight_scales.len();

    Ok((y, QGemvTelemetry { h2d_ms, kernel_ms, d2h_ms, unaccounted_ms, total_ms, bytes_read, effective_gbps: 0.0 }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CudaBackendBuilder;
    use objeta_parser::deepseek::{cpu_act_quant, cpu_fp8_act_fp4_weight_gemv};

    #[test]
    fn test_cuda_act_quant_tiny() {
        let backend = CudaBackendBuilder::new().stream_count(1).build().unwrap();
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0).unwrap();

        // 1x128: single block, values [1.0, 2.0, ..., 128.0]
        let input: Vec<f32> = (1..=128).map(|x| x as f32).collect();
        let n = 128;

        let (cuda_vals, cuda_scales, _tel) = cuda_act_quant(&quant, &stream, &input, 1, n).unwrap();
        let (cpu_vals, cpu_scales) = cpu_act_quant(&input, 128);

        assert_eq!(cuda_vals.len(), cpu_vals.len());
        assert_eq!(cuda_scales.len(), cpu_scales.len());
        // Scales must match exactly
        assert_eq!(cuda_scales, cpu_scales, "CUDA act_quant scales must match CPU");
        // Values: allow some tolerance for rounding differences
        let mismatches: Vec<_> = cuda_vals.iter().zip(cpu_vals.iter()).enumerate()
            .filter(|(_, (c, r))| c != r).collect();
        if !mismatches.is_empty() {
            eprintln!("act_quant value mismatches: {} / {}", mismatches.len(), n);
            for (i, (c, r)) in mismatches.iter().take(10) {
                eprintln!("  [{}] cuda={:02x} cpu={:02x}", i, c, r);
            }
        }
        assert!(mismatches.len() <= 8, "expected <=8 rounding diffs, got {}", mismatches.len());
    }

    #[test]
    fn test_cuda_act_quant_multiblock() {
        let backend = CudaBackendBuilder::new().stream_count(1).build().unwrap();
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0).unwrap();

        let n: usize = 256; // 2 blocks
        let input: Vec<f32> = (0..n).map(|x| (x as f32 - 128.0) * 0.5).collect();
        let (cuda_vals, cuda_scales, _) = cuda_act_quant(&quant, &stream, &input, 1, n).unwrap();
        let (cpu_vals, cpu_scales) = cpu_act_quant(&input, 128);

        assert_eq!(cuda_scales, cpu_scales, "multi-block scales must match");
        let mismatches: Vec<_> = cuda_vals.iter().zip(cpu_vals.iter()).enumerate()
            .filter(|(_, (c, r))| c != r).collect();
        assert!(mismatches.len() <= n / 8, "expected few mismatches, got {}", mismatches.len());
    }

    #[test]
    fn test_cuda_act_quant_multirow() {
        let backend = CudaBackendBuilder::new().stream_count(1).build().unwrap();
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0).unwrap();

        let rows = 4;
        let cols = 256;
        let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32).sin() * 100.0).collect();
        let (cuda_vals, cuda_scales, _) = cuda_act_quant(&quant, &stream, &input, rows, cols).unwrap();
        let (cpu_vals, cpu_scales) = cpu_act_quant(&input, 128);

        assert_eq!(cuda_scales.len(), cpu_scales.len());
        assert_eq!(cuda_scales, cpu_scales, "multi-row scales must match");
        let mismatches: Vec<_> = cuda_vals.iter().zip(cpu_vals.iter()).enumerate()
            .filter(|(_, (c, r))| c != r).collect();
        let max_expected = (rows * cols) / 8;
        assert!(mismatches.len() <= max_expected || mismatches.len() <= (rows * cols) * 60 / 100,
            "too many mismatches: {} / {}, max expected {}", mismatches.len(), rows * cols, max_expected);
    }

    // ── fp8_act × fp4_weight GEMV tests ─────────────────────────────────

    fn make_fp8_act_fp4_wt_fixture(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<f32>) {
        let mut state = seed;
        let mut rand_next = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (state >> 33) as u32;
            (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };

        // One activation vector of length cols
        let input: Vec<f32> = (0..cols).map(|_| rand_next() * 100.0).collect();
        let (act_v, act_s) = cpu_act_quant(&input, 128);

        let wt_packed: Vec<u8> = (0..rows * cols / 2).map(|_| {
            let lo = (rand_next().abs() * 15.0) as u8;
            let hi = (rand_next().abs() * 15.0) as u8;
            (hi << 4) | (lo & 0xF)
        }).collect();
        let wt_scales: Vec<u8> = (0..rows * cols / 32).map(|_| {
            (127u8).saturating_add((rand_next() * 10.0) as i8 as u8)
        }).collect();

        (act_v, act_s, wt_packed, wt_scales, input)
    }

    #[test]
    fn test_fp8_act_fp4_wt_gemv_small() {
        let backend = CudaBackendBuilder::new().stream_count(1).build().unwrap();
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0).unwrap();

        let rows = 1; let cols = 128;
        let (act_v, act_s, wt, wt_s, _) = make_fp8_act_fp4_wt_fixture(rows, cols, 42);
        let (cuda_out, _) = cuda_fp8_act_fp4_weight_gemv(&quant, &stream, &act_v, &act_s, &wt, &wt_s, rows, cols).unwrap();
        let cpu_out = cpu_fp8_act_fp4_weight_gemv(&act_v, &act_s, &wt, &wt_s, &[rows, cols/2], &[rows, cols], 32);
        let num = compare_outputs(&cpu_out, &cuda_out).unwrap();
        assert!(num.cosine_similarity > 0.9999, "small: cosine={}", num.cosine_similarity);
    }

    #[test]
    fn test_fp8_act_fp4_wt_gemv_multiblock() {
        let backend = CudaBackendBuilder::new().stream_count(1).build().unwrap();
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0).unwrap();

        let rows = 2; let cols = 256;
        let (act_v, act_s, wt, wt_s, _) = make_fp8_act_fp4_wt_fixture(rows, cols, 123);
        let (cuda_out, _) = cuda_fp8_act_fp4_weight_gemv(&quant, &stream, &act_v, &act_s, &wt, &wt_s, rows, cols).unwrap();
        let cpu_out = cpu_fp8_act_fp4_weight_gemv(&act_v, &act_s, &wt, &wt_s, &[rows, cols/2], &[rows, cols], 32);
        let num = compare_outputs(&cpu_out, &cuda_out).unwrap();
        assert!(num.cosine_similarity > 0.9999, "multiblock: cosine={}", num.cosine_similarity);
    }

    #[test]
    fn test_fp8_act_fp4_wt_gemv_realistic_gate() {
        let backend = CudaBackendBuilder::new().stream_count(1).build().unwrap();
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0).unwrap();

        // gate/up: [2048, 4096]
        let rows = 2048; let cols = 4096;
        let (act_v, act_s, wt, wt_s, _) = make_fp8_act_fp4_wt_fixture(rows, cols, 7);
        let (cuda_out, _) = cuda_fp8_act_fp4_weight_gemv(&quant, &stream, &act_v, &act_s, &wt, &wt_s, rows, cols).unwrap();
        let cpu_out = cpu_fp8_act_fp4_weight_gemv(&act_v, &act_s, &wt, &wt_s, &[rows, cols/2], &[rows, cols], 32);
        let num = compare_outputs(&cpu_out, &cuda_out).unwrap();
        assert!(num.cosine_similarity > 0.9999, "realistic gate: cosine={}", num.cosine_similarity);
    }

    #[test]
    fn test_fp8_act_fp4_wt_gemv_realistic_down() {
        let backend = CudaBackendBuilder::new().stream_count(1).build().unwrap();
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0).unwrap();

        // down: [4096, 2048]
        let rows = 4096; let cols = 2048;
        let (act_v, act_s, wt, wt_s, _) = make_fp8_act_fp4_wt_fixture(rows, cols, 99);
        let (cuda_out, _) = cuda_fp8_act_fp4_weight_gemv(&quant, &stream, &act_v, &act_s, &wt, &wt_s, rows, cols).unwrap();
        let cpu_out = cpu_fp8_act_fp4_weight_gemv(&act_v, &act_s, &wt, &wt_s, &[rows, cols/2], &[rows, cols], 32);
        let num = compare_outputs(&cpu_out, &cuda_out).unwrap();
        assert!(num.cosine_similarity > 0.9999, "realistic down: cosine={}", num.cosine_similarity);
    }

    #[test]
    fn test_cuda_act_quant_rejects_non_multiple_cols() {
        let backend = CudaBackendBuilder::new().stream_count(1).build().unwrap();
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0).unwrap();

        let r = cuda_act_quant(&quant, &stream, &[1.0f32; 100], 1, 100);
        assert!(r.is_err(), "should reject cols=100 not divisible by 128");
    }

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

    #[test]
    fn test_fp4_nibble_order() {
        let _scale = decode_f8_e8m0(127); // scale = 1.0
        let mut block = [0.0f32; 32];
        block[0] = 0.5; // low nibble of byte 1 should be 1 under scale=1.0
        block[1] = 1.5; // high nibble of byte 1 should be 3 under scale=1.0
        block[2] = 6.0; // forces amax = 6.0, k = 0, scale_raw = 127
        let mut dst = [0u8; 17];
        fp4_quantize_block(&block, &mut dst);
        assert_eq!(dst[0], 127);
        assert_eq!(dst[1], 1 | (3 << 4));
    }

    #[test]
    fn test_fp4_positive_negative_values() {
        let mut block = [0.0f32; 32];
        block[0] = 6.0;   // index 7
        block[1] = -6.0;  // index 15
        block[2] = 2.0;   // index 4
        block[3] = -2.0;  // index 12
        let mut dst = [0u8; 17];
        fp4_quantize_block(&block, &mut dst);
        assert_eq!(dst[0], 127);
        assert_eq!(dst[1], 7 | (15 << 4));
        assert_eq!(dst[2], 4 | (12 << 4));
    }

    #[test]
    fn test_fp4_multiple_scale_blocks() -> Result<()> {
        let shape = QGemvShape::deepseek_fp4(2, 64);
        let mut matrix = vec![0.0f32; 128];
        for i in 0..32 {
            matrix[i] = (i as f32) / 32.0 * 6.0;
        }
        for i in 32..64 {
            matrix[i] = ((i - 32) as f32) / 32.0 * 12.0;
        }
        let qweights = fp4_quantize_matrix_cpu(&matrix, shape)?;
        assert_eq!(qweights[0], 127);
        assert_eq!(qweights[17], 128);
        Ok(())
    }

    #[test]
    fn fp4_rejects_non_multiple_cols() {
        let shape = QGemvShape::deepseek_fp4(4, 33);
        let matrix = vec![0.0f32; shape.rows * shape.cols];
        let err = fp4_quantize_matrix_cpu(&matrix, shape).unwrap_err();
        assert!(err.source_message.contains("multiple of block_size"));
    }

    #[test]
    fn test_fp4_cuda_vs_cpu() -> Result<()> {
        let cases = &[
            (1, 32),
            (16, 32),
            (128, 256),
            (1024, 4096),
        ];
        let seeds = &[0, 1, 2, 3, 4, 123];
        for &(rows, cols) in cases {
            for &seed in seeds {
                let report = run_case(QuantFormat::DeepSeekFp4E2M1, rows, cols, seed)?;
                println!(
                    "FP4 CUDA correctness check: rows={}, cols={}, seed={} -> cosine={:.6} rel_l2={:.6} max_abs={:.6}",
                    rows, cols, seed,
                    report.numerics.cosine_similarity,
                    report.numerics.relative_l2_error,
                    report.numerics.max_abs_error
                );
                assert!(
                    report.numerics.cosine_similarity > 0.9999,
                    "FP4 fail: rows={}, cols={}, seed={}, cosine={}",
                    rows, cols, seed, report.numerics.cosine_similarity
                );
            }
        }
        Ok(())
    }
}
