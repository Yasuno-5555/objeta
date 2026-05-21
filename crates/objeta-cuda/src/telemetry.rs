use std::sync::Arc;

use cudarc::driver::{sys::CUevent_flags, CudaEvent, CudaStream};

use crate::{cuda_map_err, CudaErrorKind, Result};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelTiming {
    pub label: String,
    pub elapsed_ms: f32,
}

#[derive(Debug)]
pub struct CudaEventTimer {
    start: CudaEvent,
}

impl CudaEventTimer {
    pub fn start(stream: &Arc<CudaStream>) -> Result<Self> {
        let start = cuda_map_err!(
            CudaErrorKind::Driver,
            "record CUDA start event",
            stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        )?;
        Ok(Self { start })
    }

    pub fn stop(self, label: impl Into<String>, stream: &Arc<CudaStream>) -> Result<KernelTiming> {
        let end = cuda_map_err!(
            CudaErrorKind::Driver,
            "record CUDA end event",
            stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        )?;
        let elapsed_ms = cuda_map_err!(
            CudaErrorKind::Driver,
            "measure CUDA event elapsed time",
            self.start.elapsed_ms(&end)
        )?;
        Ok(KernelTiming {
            label: label.into(),
            elapsed_ms,
        })
    }
}
