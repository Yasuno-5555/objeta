use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::memory::{DeviceBuffer, PinnedHostBuffer};
use crate::{cuda_map_err, CudaError, CudaErrorKind, Result};

#[derive(Debug)]
pub struct CudaStreamPool {
    streams: Vec<CudaStreamHandle>,
}

impl CudaStreamPool {
    pub fn new(context: Arc<CudaContext>, stream_count: usize) -> Result<Self> {
        let mut streams = Vec::with_capacity(stream_count.max(1));
        streams.push(CudaStreamHandle::new(0, context.default_stream()));
        for idx in 1..stream_count.max(1) {
            let stream = cuda_map_err!(
                CudaErrorKind::Driver,
                format!("create CUDA stream {idx}"),
                context.new_stream()
            )?;
            streams.push(CudaStreamHandle::new(idx, stream));
        }
        Ok(Self { streams })
    }

    pub fn len(&self) -> usize {
        self.streams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    pub fn stream(&self, index: usize) -> Result<&CudaStreamHandle> {
        self.streams.get(index).ok_or_else(|| {
            CudaError::new(
                CudaErrorKind::InvalidInput,
                format!("select CUDA stream {index}"),
                format!("stream index out of range; pool len={}", self.streams.len()),
                file!(),
                line!(),
                module_path!(),
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct CudaStreamHandle {
    index: usize,
    raw: Arc<CudaStream>,
}

impl CudaStreamHandle {
    pub(crate) fn new(index: usize, raw: Arc<CudaStream>) -> Self {
        Self { index, raw }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn raw(&self) -> &Arc<CudaStream> {
        &self.raw
    }

    pub fn synchronize(&self) -> Result<()> {
        cuda_map_err!(
            CudaErrorKind::Driver,
            format!("synchronize CUDA stream {}", self.index),
            self.raw.synchronize()
        )
    }

    pub fn alloc_zeros<T>(&self, len: usize) -> Result<DeviceBuffer<T>>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        let raw = cuda_map_err!(
            CudaErrorKind::Driver,
            format!(
                "allocate zeroed device buffer len={len} on stream {}",
                self.index
            ),
            self.raw.alloc_zeros::<T>(len)
        )?;
        Ok(DeviceBuffer::from_raw(raw))
    }

    pub unsafe fn alloc_uninit<T>(&self, len: usize) -> Result<DeviceBuffer<T>>
    where
        T: DeviceRepr,
    {
        let raw = cuda_map_err!(
            CudaErrorKind::Driver,
            format!("allocate device buffer len={len} on stream {}", self.index),
            self.raw.alloc::<T>(len)
        )?;
        Ok(DeviceBuffer::from_raw(raw))
    }

    pub fn copy_from_slice<T>(&self, src: &[T]) -> Result<DeviceBuffer<T>>
    where
        T: DeviceRepr,
    {
        let raw = cuda_map_err!(
            CudaErrorKind::Driver,
            format!(
                "async H2D copy from slice len={} on stream {}",
                src.len(),
                self.index
            ),
            self.raw.clone_htod(src)
        )?;
        Ok(DeviceBuffer::from_raw(raw))
    }

    pub fn copy_from_pinned<T>(&self, src: &PinnedHostBuffer<T>) -> Result<DeviceBuffer<T>>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        let raw = cuda_map_err!(
            CudaErrorKind::Driver,
            format!(
                "async H2D copy from pinned host buffer len={} on stream {}",
                src.len(),
                self.index
            ),
            self.raw.clone_htod(&src.raw)
        )?;
        Ok(DeviceBuffer::from_raw(raw))
    }

    pub fn copy_to_vec<T>(&self, src: &DeviceBuffer<T>) -> Result<Vec<T>>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        cuda_map_err!(
            CudaErrorKind::Driver,
            format!(
                "async D2H copy to Vec len={} on stream {}",
                src.len(),
                self.index
            ),
            self.raw.clone_dtoh(&src.raw)
        )
    }

    pub fn copy_to_pinned<T>(
        &self,
        src: &DeviceBuffer<T>,
        dst: &mut PinnedHostBuffer<T>,
    ) -> Result<()>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        if src.len() != dst.len() {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate async D2H destination size",
                format!("device len {} != pinned host len {}", src.len(), dst.len()),
                file!(),
                line!(),
                module_path!(),
            ));
        }
        cuda_map_err!(
            CudaErrorKind::Driver,
            format!(
                "async D2H copy to pinned host len={} on stream {}",
                src.len(),
                self.index
            ),
            self.raw.memcpy_dtoh(&src.raw, &mut dst.raw)
        )?;
        Ok(())
    }

    pub fn copy_dtod<T>(&self, src: &DeviceBuffer<T>, dst: &mut DeviceBuffer<T>) -> Result<()>
    where
        T: DeviceRepr,
    {
        if src.len() != dst.len() {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate async D2D destination size",
                format!("src len {} != dst len {}", src.len(), dst.len()),
                file!(),
                line!(),
                module_path!(),
            ));
        }
        cuda_map_err!(
            CudaErrorKind::Driver,
            format!("async D2D copy len={} on stream {}", src.len(), self.index),
            self.raw.memcpy_dtod(&src.raw, &mut dst.raw)
        )?;
        Ok(())
    }
}

impl<T> From<CudaSlice<T>> for DeviceBuffer<T> {
    fn from(value: CudaSlice<T>) -> Self {
        DeviceBuffer::from_raw(value)
    }
}
