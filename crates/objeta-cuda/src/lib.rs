mod attention;
mod context;
mod ffi;
mod memory;
mod moe;
mod quant;
mod stream;
mod telemetry;

pub use attention::AttentionBackend;
pub use context::{
    driver_version, runtime_version, CudaBackend, CudaBackendBuilder, CudaDeviceInfo,
};
pub use ffi::{BackendDeviceInfo, BackendInitOptions};
pub use memory::{DeviceBuffer, PinnedHostBuffer};
pub use moe::{
    MoeExecutor, ExpertWeights, ExpertWeightsFp32, DeepSeekFp4ExpertWeights, MoeTelemetry, selected_moe_cpu,
    selected_moe_cpu_fp32, selected_moe_cpu_native_fp4,
    execute_selected_moe_official_routed_fp4_cuda, DeepSeekFp8SharedExpertWeightsDevice,
    ExpertTensorKind, ExpertCacheKey, CudaExpertCache, ResidencyClass, BytesByTensorKind,
};
pub use quant::{
    compare_outputs, cuda_act_quant, cuda_act_quant_device, cuda_fp8_act_fp4_weight_gemv,
    cuda_fp8_act_fp4_weight_gemv_device, cuda_fp8_act_fp8_weight_gemv_device,
    dense_gemv_cpu, gemv_cpu,
    DeviceFp8Activation, q4_compare, q4_gemv_cpu,
    q4_quantize_matrix_cpu, q5_compare, q5_gemv_cpu, q5_quantize_matrix_cpu,
    iq3_gemv_cpu, iq3_quantize_matrix_cpu, fp4_compare, fp4_gemv_cpu,
    fp4_quantize_matrix_cpu, quantize_matrix_cpu, QGemvNumerics, QGemvNumericsSuite, QGemvReport, QGemvShape,
    QGemvTelemetry, QuantBackend, QuantFormat,
};
pub use stream::{CudaStreamHandle, CudaStreamPool};
pub use telemetry::{CudaEventTimer, KernelTiming};

use thiserror::Error;

pub type Result<T> = std::result::Result<T, CudaError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CudaErrorKind {
    Driver,
    Runtime,
    Nvrtc,
    Io,
    InvalidInput,
    Unsupported,
    Internal,
}

#[derive(Debug, Clone, Error, serde::Serialize, serde::Deserialize)]
#[error("{kind:?} error during {action} at {module}:{line} ({file}): {source_message}")]
pub struct CudaError {
    pub kind: CudaErrorKind,
    pub action: String,
    pub file: &'static str,
    pub line: u32,
    pub module: &'static str,
    pub source_message: String,
}

impl CudaError {
    pub fn new(
        kind: CudaErrorKind,
        action: impl Into<String>,
        source_message: impl Into<String>,
        file: &'static str,
        line: u32,
        module: &'static str,
    ) -> Self {
        Self {
            kind,
            action: action.into(),
            file,
            line,
            module,
            source_message: source_message.into(),
        }
    }
}

pub(crate) fn wrap_error<E: std::fmt::Display>(
    kind: CudaErrorKind,
    action: impl Into<String>,
    err: E,
    file: &'static str,
    line: u32,
    module: &'static str,
) -> CudaError {
    CudaError::new(kind, action, err.to_string(), file, line, module)
}

macro_rules! cuda_map_err {
    ($kind:expr, $action:expr, $expr:expr) => {
        $expr
            .map_err(|err| crate::wrap_error($kind, $action, err, file!(), line!(), module_path!()))
    };
}

pub(crate) use cuda_map_err;

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let delta = (a - b).abs();
            assert!(
                delta <= tol,
                "mismatch at {idx}: lhs={a}, rhs={b}, delta={delta}, tol={tol}"
            );
        }
    }

    #[test]
    fn cuda_device_info_dump() -> Result<()> {
        let devices = CudaBackend::discover_devices()?;
        assert!(!devices.is_empty(), "no CUDA devices discovered");
        let info = &devices[0];
        println!(
            "CUDA device: {} | CC {}.{} | total_vram={} MiB | driver={} | runtime={}",
            info.name,
            info.compute_capability_major,
            info.compute_capability_minor,
            info.total_global_mem_bytes / (1024 * 1024),
            info.driver_version_string(),
            info.runtime_version_string(),
        );
        assert!(info.total_global_mem_bytes > 0);
        Ok(())
    }

    #[test]
    fn cuda_pinned_copy_roundtrip() -> Result<()> {
        let backend = CudaBackendBuilder::new().build()?;
        let stream = backend.stream_pool().stream(0)?;

        let mut host = backend.alloc_pinned_host::<f32>(8)?;
        host.as_mut_slice()?
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]);

        let device = stream.copy_from_pinned(&host)?;
        let mut out = backend.alloc_pinned_host::<f32>(8)?;
        stream.copy_to_pinned(&device, &mut out)?;
        stream.synchronize()?;

        approx_eq(host.as_slice()?, out.as_slice()?, 1.0e-6);
        Ok(())
    }

    #[test]
    fn cuda_smoke_gemv() -> Result<()> {
        let backend = CudaBackendBuilder::new().stream_count(2).build()?;
        let stream = backend.stream_pool().stream(0)?;

        let matrix: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.0, 2.0, 3.0, -2.0, 1.0, 0.25,
        ];
        let vector: Vec<f32> = vec![0.5, -1.0, 2.0, 1.5];
        let expected = vec![
            1.0 * 0.5 + 2.0 * -1.0 + 3.0 * 2.0 + 4.0 * 1.5,
            -1.0 * 0.5 + 0.5 * -1.0 + 0.0 * 2.0 + 2.0 * 1.5,
            3.0 * 0.5 + -2.0 * -1.0 + 1.0 * 2.0 + 0.25 * 1.5,
        ];

        let d_matrix = stream.copy_from_slice(&matrix)?;
        let d_vector = stream.copy_from_slice(&vector)?;
        let mut d_out = stream.alloc_zeros::<f32>(3)?;

        let timer = CudaEventTimer::start(stream.raw())?;
        backend.smoke_gemv_f32(stream.raw(), &d_matrix, &d_vector, &mut d_out, 3, 4)?;
        let timing = timer.stop("smoke_gemv_f32", stream.raw())?;
        let out = stream.copy_to_vec(&d_out)?;

        println!(
            "kernel={} elapsed_ms={:.4} device={} driver={} runtime={}",
            timing.label,
            timing.elapsed_ms,
            backend.device_info().name,
            backend.device_info().driver_version_string(),
            backend.device_info().runtime_version_string(),
        );

        approx_eq(&out, &expected, 1.0e-5);
        Ok(())
    }
}
