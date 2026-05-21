use serde::{Deserialize, Serialize};

/// Hardware target profile used by the specialization compiler.
///
/// # Adding a new target
/// Add a new match arm in `from_name()`.  Keep the `generic` fallback as a
/// safe last-resort; do not add unsafe/untested defaults there.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetHardware {
    pub name: String,
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub recommended_cache_bytes: u64,
    pub supports_gpu: bool,
    pub storage_tier: String,
    /// Executor backend keyword. Consumed by RuntimeProfile.
    pub preferred_backend: String,
    /// Quantization format preference list, highest quality first.
    /// The precision pass may use this to select format when a tier allows
    /// multiple formats (e.g. q5 or q4 for a warm expert).
    pub preferred_quant_formats: Vec<String>,
}

impl TargetHardware {
    pub fn from_name(name: &str) -> Self {
        match name {
            "m1-8gb" => Self {
                name: "m1-8gb".to_string(),
                ram_bytes: 8 * GB,
                vram_bytes: 0,
                // Keep within unified memory: leave ~5 GB for OS + model weights
                recommended_cache_bytes: 3 * GB,
                supports_gpu: false,
                storage_tier: "nvme".to_string(),
                preferred_backend: "fused_row_parallel".to_string(),
                // CPU path; iq3/iq2 are slow on Metal without a dedicated kernel
                preferred_quant_formats: vec!["q5".to_string(), "q4".to_string(), "q4_k".to_string()],
            },
            "m2-16gb" => Self {
                name: "m2-16gb".to_string(),
                ram_bytes: 16 * GB,
                vram_bytes: 0,
                recommended_cache_bytes: 4 * GB,
                supports_gpu: false,
                storage_tier: "nvme".to_string(),
                preferred_backend: "fused_row_parallel".to_string(),
                preferred_quant_formats: vec!["q5".to_string(), "q4".to_string(), "q4_k".to_string()],
            },
            "rtx3070-8gb-vram-32gb-ram" => Self {
                name: "rtx3070-8gb-vram-32gb-ram".to_string(),
                ram_bytes: 32 * GB,
                // 8 GB VRAM: can hold hot experts + kv-cache; cold experts spill to RAM
                vram_bytes: 8 * GB,
                // Hot experts live in VRAM; warm/cold can be swapped from RAM
                recommended_cache_bytes: 8 * GB,
                supports_gpu: true,
                storage_tier: "nvme".to_string(),
                preferred_backend: "cuda_fused".to_string(),
                // CUDA has good iq3/iq2 kernels; prefer them for cold experts
                preferred_quant_formats: vec![
                    "q5".to_string(),
                    "q4".to_string(),
                    "iq3".to_string(),
                    "iq2".to_string(),
                ],
            },
            "cpu-32gb" => Self {
                name: "cpu-32gb".to_string(),
                ram_bytes: 32 * GB,
                vram_bytes: 0,
                recommended_cache_bytes: 8 * GB,
                supports_gpu: false,
                storage_tier: "nvme".to_string(),
                preferred_backend: "cpu_parallel".to_string(),
                preferred_quant_formats: vec!["q4".to_string(), "q4_k".to_string(), "iq3".to_string()],
            },
            _ => Self {
                name: name.to_string(),
                ram_bytes: 16 * GB,
                vram_bytes: 0,
                recommended_cache_bytes: 4 * GB,
                supports_gpu: false,
                storage_tier: "nvme".to_string(),
                preferred_backend: "fused_row_parallel".to_string(),
                preferred_quant_formats: vec!["q4".to_string()],
            },
        }
    }

    pub fn default_expert_bytes(&self) -> u64 {
        match self.name.as_str() {
            "m1-8gb" => 2 * MB,
            "m2-16gb" => 3 * MB,
            "rtx3070-8gb-vram-32gb-ram" => 4 * MB,
            "cpu-32gb" => 4 * MB,
            _ => 2 * MB,
        }
    }
}

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_hardware_m1_8gb() {
        let t = TargetHardware::from_name("m1-8gb");
        assert_eq!(t.ram_bytes, 8 * GB);
        assert_eq!(t.vram_bytes, 0);
        assert!(!t.supports_gpu);
        assert_eq!(t.recommended_cache_bytes, 3 * GB);
        assert_eq!(t.preferred_backend, "fused_row_parallel");
    }

    #[test]
    fn target_hardware_rtx3070_differs_from_m1() {
        let m1 = TargetHardware::from_name("m1-8gb");
        let rtx = TargetHardware::from_name("rtx3070-8gb-vram-32gb-ram");
        // RTX has much more RAM and 8 GB VRAM
        assert!(rtx.ram_bytes > m1.ram_bytes);
        assert!(rtx.vram_bytes > 0);
        assert!(m1.vram_bytes == 0);
        assert!(rtx.supports_gpu);
        assert!(!m1.supports_gpu);
        // Different backends
        assert_ne!(rtx.preferred_backend, m1.preferred_backend);
        // RTX cache should be larger
        assert!(rtx.recommended_cache_bytes > m1.recommended_cache_bytes);
        // RTX supports iq3/iq2 kernels; verify they appear in preferred formats
        assert!(rtx.preferred_quant_formats.iter().any(|f| f == "iq3"));
    }

    #[test]
    fn target_hardware_generic_fallback() {
        let t = TargetHardware::from_name("unknown-device");
        assert_eq!(t.ram_bytes, 16 * GB);
        assert!(!t.preferred_quant_formats.is_empty());
    }

    #[test]
    fn target_hardware_cpu_32gb() {
        let t = TargetHardware::from_name("cpu-32gb");
        assert_eq!(t.ram_bytes, 32 * GB);
        assert_eq!(t.vram_bytes, 0);
        assert_eq!(t.preferred_backend, "cpu_parallel");
    }
}
