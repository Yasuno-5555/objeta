use crate::{CudaError, CudaErrorKind, Result};

#[derive(Debug, Default)]
pub struct AttentionBackend;

impl AttentionBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn status(&self) -> Result<()> {
        Err(CudaError::new(
            CudaErrorKind::Unsupported,
            "initialize CUDA attention backend",
            "attention kernels are intentionally not implemented in the first skeleton",
            file!(),
            line!(),
            module_path!(),
        ))
    }
}
