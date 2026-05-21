#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BackendInitOptions {
    pub device_ordinal: usize,
    pub stream_count: usize,
}

impl Default for BackendInitOptions {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            stream_count: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct BackendDeviceInfo {
    pub ordinal: usize,
    pub total_global_mem_bytes: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
}
