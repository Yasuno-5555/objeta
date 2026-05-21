use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, DeviceRepr, PinnedHostSlice, ValidAsZeroBits};

use crate::{cuda_map_err, CudaErrorKind, Result};

#[derive(Debug)]
pub struct DeviceBuffer<T> {
    pub(crate) raw: CudaSlice<T>,
}

impl<T> DeviceBuffer<T> {
    pub(crate) fn from_raw(raw: CudaSlice<T>) -> Self {
        Self { raw }
    }

    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.len() == 0
    }

    pub fn num_bytes(&self) -> usize {
        self.raw.len() * std::mem::size_of::<T>()
    }
}

#[derive(Debug)]
pub struct PinnedHostBuffer<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub(crate) raw: PinnedHostSlice<T>,
}

impl<T> PinnedHostBuffer<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub fn new(context: Arc<CudaContext>, len: usize) -> Result<Self> {
        let raw = cuda_map_err!(
            CudaErrorKind::Driver,
            format!("allocate pinned host buffer len={len}"),
            unsafe { context.alloc_pinned::<T>(len) }
        )?;
        Ok(Self { raw })
    }

    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn num_bytes(&self) -> usize {
        self.raw.num_bytes()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    pub fn as_slice(&self) -> Result<&[T]> {
        let len = self.len();
        cuda_map_err!(
            CudaErrorKind::Driver,
            format!("map pinned host buffer len={len} as slice"),
            self.raw.as_slice()
        )
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [T]> {
        let len = self.len();
        cuda_map_err!(
            CudaErrorKind::Driver,
            format!("map pinned host buffer len={len} as mutable slice"),
            self.raw.as_mut_slice()
        )
    }
}
