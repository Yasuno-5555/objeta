use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use cudarc::runtime::result::{device, version};

use crate::memory::DeviceBuffer;
use crate::stream::CudaStreamPool;
use crate::{cuda_map_err, CudaError, CudaErrorKind, Result};

const SMOKE_KERNEL_SRC: &str = include_str!("../kernels/smoke.cu");

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub total_global_mem_bytes: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub driver_version: i32,
    pub runtime_version: i32,
}

impl CudaDeviceInfo {
    pub fn driver_version_string(&self) -> String {
        format_cuda_version(self.driver_version)
    }

    pub fn runtime_version_string(&self) -> String {
        format_cuda_version(self.runtime_version)
    }
}

#[derive(Debug, Clone)]
pub struct CudaBackendBuilder {
    device_ordinal: usize,
    stream_count: usize,
}

impl Default for CudaBackendBuilder {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            stream_count: 1,
        }
    }
}

impl CudaBackendBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = device_ordinal;
        self
    }

    pub fn stream_count(mut self, stream_count: usize) -> Self {
        self.stream_count = stream_count.max(1);
        self
    }

    pub fn build(self) -> Result<CudaBackend> {
        CudaBackend::new(self.device_ordinal, self.stream_count)
    }
}

#[derive(Debug)]
pub struct CudaBackend {
    device_info: CudaDeviceInfo,
    context: Arc<CudaContext>,
    stream_pool: CudaStreamPool,
    smoke_module: Mutex<Option<SmokeModule>>,
}

#[derive(Debug)]
struct SmokeModule {
    _module: Arc<CudaModule>,
    gemv_f32: CudaFunction,
}

impl CudaBackend {
    pub fn discover_devices() -> Result<Vec<CudaDeviceInfo>> {
        let count = cuda_map_err!(
            CudaErrorKind::Runtime,
            "cudaGetDeviceCount",
            device::get_count()
        )?;
        let driver = driver_version()?;
        let runtime = runtime_version()?;
        let mut devices = Vec::with_capacity(count as usize);
        for ordinal in 0..count as usize {
            let prop = cuda_map_err!(
                CudaErrorKind::Runtime,
                format!("cudaGetDeviceProperties(device={ordinal})"),
                device::get_device_prop(ordinal as i32)
            )?;
            devices.push(CudaDeviceInfo {
                ordinal,
                name: decode_device_name(&prop.name),
                total_global_mem_bytes: prop.totalGlobalMem,
                compute_capability_major: prop.major,
                compute_capability_minor: prop.minor,
                driver_version: driver,
                runtime_version: runtime,
            });
        }
        Ok(devices)
    }

    pub fn new(device_ordinal: usize, stream_count: usize) -> Result<Self> {
        let devices = Self::discover_devices()?;
        let device_info = devices
            .into_iter()
            .find(|info| info.ordinal == device_ordinal)
            .ok_or_else(|| {
                CudaError::new(
                    CudaErrorKind::InvalidInput,
                    format!("select CUDA device ordinal {device_ordinal}"),
                    "device ordinal is out of range",
                    file!(),
                    line!(),
                    module_path!(),
                )
            })?;
        let context = cuda_map_err!(
            CudaErrorKind::Driver,
            format!("create CUDA context for device {device_ordinal}"),
            CudaContext::new(device_ordinal)
        )?;
        let stream_pool = CudaStreamPool::new(context.clone(), stream_count)?;
        Ok(Self {
            device_info,
            context,
            stream_pool,
            smoke_module: Mutex::new(None),
        })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.device_info
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub fn stream_pool(&self) -> &CudaStreamPool {
        &self.stream_pool
    }

    pub fn alloc_pinned_host<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<crate::memory::PinnedHostBuffer<T>> {
        crate::memory::PinnedHostBuffer::new(self.context.clone(), len)
    }

    pub fn smoke_gemv_f32(
        &self,
        stream: &Arc<CudaStream>,
        matrix: &DeviceBuffer<f32>,
        vector: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if matrix.len() != rows * cols {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate smoke_gemv_f32 matrix shape",
                format!("matrix len {} != rows*cols {}", matrix.len(), rows * cols),
                file!(),
                line!(),
                module_path!(),
            ));
        }
        if vector.len() != cols {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate smoke_gemv_f32 vector shape",
                format!("vector len {} != cols {}", vector.len(), cols),
                file!(),
                line!(),
                module_path!(),
            ));
        }
        if out.len() != rows {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate smoke_gemv_f32 output shape",
                format!("output len {} != rows {}", out.len(), rows),
                file!(),
                line!(),
                module_path!(),
            ));
        }

        let smoke_guard = self.smoke_module(stream)?;
        let smoke = smoke_guard.as_ref().expect("smoke module cache populated");
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        cuda_map_err!(
            CudaErrorKind::Driver,
            "launch smoke_gemv_f32 kernel",
            unsafe {
                stream
                    .launch_builder(&smoke.gemv_f32)
                    .arg(&matrix.raw)
                    .arg(&vector.raw)
                    .arg(&mut out.raw)
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .launch(cfg)
            }
        )?;
        Ok(())
    }

    fn smoke_module(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<std::sync::MutexGuard<'_, Option<SmokeModule>>> {
        let mut guard = self.smoke_module.lock().map_err(|err| {
            CudaError::new(
                CudaErrorKind::Internal,
                "lock smoke module cache",
                err.to_string(),
                file!(),
                line!(),
                module_path!(),
            )
        })?;
        if guard.is_none() {
            let ptx = compile_smoke_ptx(
                self.device_info.compute_capability_major,
                self.device_info.compute_capability_minor,
            )?;
            let module = cuda_map_err!(
                CudaErrorKind::Driver,
                "load smoke PTX module",
                self.context.load_module(ptx)
            )?;
            let gemv_f32 = cuda_map_err!(
                CudaErrorKind::Driver,
                "load smoke_gemv_f32 function",
                module.load_function("smoke_gemv_f32")
            )?;
            let _ = stream;
            *guard = Some(SmokeModule {
                _module: module,
                gemv_f32,
            });
        }
        Ok(guard)
    }
}

pub fn runtime_version() -> Result<i32> {
    cuda_map_err!(
        CudaErrorKind::Runtime,
        "cudaRuntimeGetVersion",
        version::get_runtime_version()
    )
}

pub fn driver_version() -> Result<i32> {
    cuda_map_err!(
        CudaErrorKind::Runtime,
        "cudaDriverGetVersion",
        version::get_driver_version()
    )
}

fn decode_device_name(raw: &[i8]) -> String {
    let bytes: Vec<u8> = raw
        .iter()
        .copied()
        .take_while(|v| *v != 0)
        .map(|v| v as u8)
        .collect();
    String::from_utf8_lossy(&bytes).trim().to_string()
}

fn compile_smoke_ptx(major: i32, minor: i32) -> Result<Ptx> {
    let arch = compute_arch_literal(major, minor);
    let opts = CompileOptions {
        arch,
        include_paths: Vec::new(),
        name: Some("objeta_cuda_smoke.cu".into()),
        ..Default::default()
    };
    cuda_map_err!(
        CudaErrorKind::Nvrtc,
        format!(
            "compile smoke CUDA source for compute capability {}.{}",
            major, minor
        ),
        compile_ptx_with_opts(SMOKE_KERNEL_SRC, opts)
    )
}

fn compute_arch_literal(major: i32, minor: i32) -> Option<&'static str> {
    match (major, minor) {
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
    }
}

fn format_cuda_version(version: i32) -> String {
    let major = version / 1000;
    let minor = (version % 1000) / 10;
    let patch = version % 10;
    format!("{major}.{minor}.{patch}")
}
