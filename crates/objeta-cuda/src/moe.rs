use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{CudaContext, CudaModule, CudaFunction, LaunchConfig, PushKernelArg};

use crate::{cuda_map_err, CudaError, CudaErrorKind, Result, DeviceBuffer};
use crate::context::CudaDeviceInfo;
use crate::quant::{QuantBackend, QuantFormat, QGemvShape, gemv_cpu, dense_gemv_cpu, cuda_act_quant_device, cuda_fp8_act_fp4_weight_gemv_device, cuda_fp8_act_fp8_weight_gemv_device};
use crate::stream::CudaStreamHandle;
use crate::telemetry::CudaEventTimer;

const MOE_KERNEL_SRC: &str = include_str!("../kernels/moe.cu");

#[derive(Debug, Clone)]
pub struct ExpertWeights {
    pub w_gate: Vec<u8>,
    pub w_up: Vec<u8>,
    pub w_down: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ExpertWeightsFp32 {
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DeepSeekFp4ExpertWeights {
    pub gate_weight: Vec<u8>,
    pub gate_scale: Vec<u8>,
    pub up_weight: Vec<u8>,
    pub up_scale: Vec<u8>,
    pub down_weight: Vec<u8>,
    pub down_scale: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExpertTensorKind {
    Gate,
    Up,
    Down,
    GateWeight,
    GateScale,
    UpWeight,
    UpScale,
    DownWeight,
    DownScale,
    SharedGateWeight,
    SharedGateScale,
    SharedUpWeight,
    SharedUpScale,
    SharedDownWeight,
    SharedDownScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExpertCacheKey {
    pub layer_id: usize,
    pub expert_id: usize,
    pub tensor_kind: ExpertTensorKind,
    pub quant_format: QuantFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyClass {
    RoutedDynamic,
    SharedPinned,
}

#[derive(Debug)]
pub struct CudaExpertCache {
    pub capacity_bytes: usize,
    pub resident_bytes: usize,
    pub pinned_bytes: usize,
    pub hit_count: usize,
    pub miss_count: usize,
    pub eviction_count: usize,
    pub bypass_oversized_experts: bool,
    pub cache_insert_attempt_count: usize,
    pub cache_insert_accept_count: usize,
    pub cache_insert_bypass_count: usize,
    pub oversized_tensor_bypass_count: usize,
    pub oversized_expert_bypass_count: usize,
    pub self_eviction_risk_count: usize,
    /// Pinned keys are never LRU-evicted. Shared-expert tensors live here.
    pinned: std::collections::HashSet<ExpertCacheKey>,
    map: std::collections::HashMap<ExpertCacheKey, (Arc<DeviceBuffer<u8>>, usize)>,
    order: Vec<ExpertCacheKey>,
}

impl CudaExpertCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
            pinned_bytes: 0,
            hit_count: 0,
            miss_count: 0,
            eviction_count: 0,
            bypass_oversized_experts: false,
            cache_insert_attempt_count: 0,
            cache_insert_accept_count: 0,
            cache_insert_bypass_count: 0,
            oversized_tensor_bypass_count: 0,
            oversized_expert_bypass_count: 0,
            self_eviction_risk_count: 0,
            pinned: std::collections::HashSet::new(),
            map: std::collections::HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn get(&mut self, key: &ExpertCacheKey) -> Option<Arc<DeviceBuffer<u8>>> {
        if self.map.contains_key(key) {
            self.hit_count += 1;
            self.touch(key);
            self.map.get(key).map(|(buf, _)| buf.clone())
        } else {
            self.miss_count += 1;
            None
        }
    }

    pub fn insert(&mut self, key: ExpertCacheKey, buf: DeviceBuffer<u8>) -> Arc<DeviceBuffer<u8>> {
        let size = buf.num_bytes();
        self.cache_insert_attempt_count += 1;

        if size > self.capacity_bytes {
            self.oversized_tensor_bypass_count += 1;
            self.cache_insert_bypass_count += 1;
            return Arc::new(buf);
        }

        self.cache_insert_accept_count += 1;
        
        // Remove if it already exists (updating size & order)
        if let Some((_, old_size)) = self.map.remove(&key) {
            self.resident_bytes -= old_size;
            if let Some(pos) = self.order.iter().position(|k| *k == key) {
                self.order.remove(pos);
            }
        }

        // Evict LRU elements if we exceed capacity
        while self.resident_bytes + size > self.capacity_bytes && !self.order.is_empty() {
            self.evict_lru();
        }

        let arc_buf = Arc::new(buf);
        self.map.insert(key, (arc_buf.clone(), size));
        self.resident_bytes += size;
        self.order.push(key);
        arc_buf
    }

    pub fn touch(&mut self, key: &ExpertCacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
    }

    /// Evict the least-recently-used NON-PINNED entry.
    pub fn evict_lru(&mut self) -> Option<(ExpertCacheKey, Arc<DeviceBuffer<u8>>)> {
        // Find the first non-pinned entry in LRU order
        let pos = self.order.iter().position(|k| !self.pinned.contains(k));
        if let Some(idx) = pos {
            let key = self.order.remove(idx);
            if let Some((buf, size)) = self.map.remove(&key) {
                self.resident_bytes -= size;
                self.eviction_count += 1;
                return Some((key, buf));
            }
        }
        None
    }

    /// Pin an already-cached tensor as shared-resident (never LRU-evicted).
    pub fn pin(&mut self, key: &ExpertCacheKey) {
        if self.map.contains_key(key) && !self.pinned.contains(key) {
            self.pinned.insert(key.clone());
            if let Some(&(_, size)) = self.map.get(key) {
                self.pinned_bytes += size;
            }
        }
    }

    /// Insert a tensor with pinned residency (never evicted).
    /// Must not already exist; panics if it does.
    pub fn insert_pinned(&mut self, key: ExpertCacheKey, buf: DeviceBuffer<u8>) -> Arc<DeviceBuffer<u8>> {
        let size = buf.num_bytes();
        // Free up space from non-pinned entries
        while self.resident_bytes + size > self.capacity_bytes && !self.order.is_empty() {
            if self.evict_lru().is_none() { break; }
        }
        let arc_buf = Arc::new(buf);
        if let Some((_, old_size)) = self.map.insert(key.clone(), (arc_buf.clone(), size)) {
            self.resident_bytes -= old_size;
        }
        self.resident_bytes += size;
        self.pinned_bytes += size;
        self.pinned.insert(key.clone());
        self.order.push(key);
        arc_buf
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.pinned.clear();
        self.resident_bytes = 0;
        self.pinned_bytes = 0;
    }

    /// Reset hit/miss/eviction counters but preserve resident tensors and pin state.
    pub fn reset_counters(&mut self) {
        self.hit_count = 0;
        self.miss_count = 0;
        self.eviction_count = 0;
        self.cache_insert_attempt_count = 0;
        self.cache_insert_accept_count = 0;
        self.cache_insert_bypass_count = 0;
        self.oversized_tensor_bypass_count = 0;
        self.oversized_expert_bypass_count = 0;
        self.self_eviction_risk_count = 0;
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct BytesByTensorKind {
    pub gate: usize,
    pub up: usize,
    pub down: usize,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct MoeTelemetry {
    pub selected_expert_count: usize,
    pub logical_expert_bytes_requested: usize,
    pub actual_expert_bytes_loaded: usize,
    pub resident_cache_bytes_reused: usize,
    pub weight_bytes_loaded: usize,
    pub weight_bytes_reused: usize,
    pub scale_bytes_loaded: usize,
    pub scale_bytes_reused: usize,
    pub resident_cache_resident_bytes: usize,
    pub resident_cache_capacity_bytes: usize,
    pub resident_cache_hit_count: usize,
    pub resident_cache_miss_count: usize,
    pub resident_cache_eviction_count: usize,
    pub resident_cache_insert_attempt_count: usize,
    pub resident_cache_insert_accept_count: usize,
    pub resident_cache_insert_bypass_count: usize,
    pub resident_cache_oversized_tensor_bypass_count: usize,
    pub resident_cache_oversized_expert_bypass_count: usize,
    pub resident_cache_self_eviction_risk_count: usize,
    pub dequantized_scratch_bytes: usize,
    pub h2d_ms: f32,
    pub gate_up_qgemv_ms: f32,
    pub activation_ms: f32,
    pub down_qgemv_ms: f32,
    pub accum_ms: f32,
    pub unaccounted_ms: f32,
    pub total_ms: f32,
    pub shared_expert_ms: f32,
    pub bytes_by_tensor_kind: BytesByTensorKind,
    pub bytes_per_expert: usize,
    pub selected_working_set_bytes: usize,
    pub routed_fp4_bytes_loaded: usize,
    pub routed_fp4_bytes_reused: usize,
    pub shared_fp8_bytes_loaded: usize,
    pub shared_fp8_bytes_reused: usize,
    pub total_logical_bytes: usize,
    pub total_loaded_bytes: usize,
    pub total_reused_bytes: usize,
}

#[derive(Debug)]
struct MoeModule {
    _module: Arc<CudaModule>,
    silu_mul: CudaFunction,
    weighted_accum: CudaFunction,
}

#[derive(Debug)]
pub struct MoeExecutor {
    context: Arc<CudaContext>,
    device_info: CudaDeviceInfo,
    module: Mutex<Option<MoeModule>>,
}

impl MoeExecutor {
    pub fn new(context: Arc<CudaContext>, device_info: CudaDeviceInfo) -> Self {
        Self {
            context,
            device_info,
            module: Mutex::new(None),
        }
    }

    pub fn status(&self) -> Result<()> {
        Ok(())
    }

    pub fn compile(&self) -> Result<()> {
        let _guard = self.ensure_module()?;
        Ok(())
    }

    fn ensure_module(&self) -> Result<MutexGuard<'_, Option<MoeModule>>> {
        let mut guard = self.module.lock().map_err(|err| {
            CudaError::new(
                CudaErrorKind::Internal,
                "lock MoE module cache",
                err.to_string(),
                file!(),
                line!(),
                module_path!(),
            )
        })?;
        if guard.is_none() {
            let ptx = crate::quant::compile_kernel_ptx(
                MOE_KERNEL_SRC,
                "objeta_cuda_moe.cu",
                self.device_info.compute_capability_major,
                self.device_info.compute_capability_minor,
            )?;
            let module = cuda_map_err!(
                CudaErrorKind::Driver,
                "load MoE PTX module",
                self.context.load_module(ptx)
            )?;
            let silu_mul = cuda_map_err!(
                CudaErrorKind::Driver,
                "load silu_mul function",
                module.load_function("silu_mul")
            )?;
            let weighted_accum = cuda_map_err!(
                CudaErrorKind::Driver,
                "load weighted_accum function",
                module.load_function("weighted_accum")
            )?;
            *guard = Some(MoeModule {
                _module: module,
                silu_mul,
                weighted_accum,
            });
        }
        Ok(guard)
    }

    pub fn execute_selected_moe_cuda(
        &self,
        quant: &QuantBackend,
        stream: &CudaStreamHandle,
        experts: &[ExpertWeights],
        selected_experts: &[(usize, f32)],
        x: &[f32],
        hidden: usize,
        intermediate: usize,
        out_dim: usize,
        layer_id: usize,
        mut cache: Option<&mut CudaExpertCache>,
    ) -> Result<(Vec<f32>, MoeTelemetry)> {
        // Validate inputs
        if x.len() != hidden {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate execute_selected_moe_cuda input x size",
                format!("x len {} != hidden {}", x.len(), hidden),
                file!(),
                line!(),
                module_path!(),
            ));
        }

        let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate, hidden);
        let shape_down = QGemvShape::new(QuantFormat::Q4_0, out_dim, intermediate);

        for &(e, _) in selected_experts {
            if e >= experts.len() {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert index",
                    format!("expert index {} out of bounds (len={})", e, experts.len()),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            let expert = &experts[e];
            if expert.w_gate.len() != shape_gate_up.quantized_matrix_bytes() {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert w_gate size",
                    format!("w_gate len {} != expected {}", expert.w_gate.len(), shape_gate_up.quantized_matrix_bytes()),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            if expert.w_up.len() != shape_gate_up.quantized_matrix_bytes() {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert w_up size",
                    format!("w_up len {} != expected {}", expert.w_up.len(), shape_gate_up.quantized_matrix_bytes()),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            if expert.w_down.len() != shape_down.quantized_matrix_bytes() {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert w_down size",
                    format!("w_down len {} != expected {}", expert.w_down.len(), shape_down.quantized_matrix_bytes()),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
        }

        let total_timer = CudaEventTimer::start(stream.raw())?;

        let mut h2d_ms = 0.0;
        let x_h2d_timer = CudaEventTimer::start(stream.raw())?;
        let d_x = stream.copy_from_slice(x)?;
        let x_h2d_timing = x_h2d_timer.stop("moe_x_h2d", stream.raw())?;
        h2d_ms += x_h2d_timing.elapsed_ms;

        // Allocate device buffers for intermediate values
        let mut d_gate = stream.alloc_zeros::<f32>(intermediate)?;
        let mut d_up = stream.alloc_zeros::<f32>(intermediate)?;
        let mut d_act = stream.alloc_zeros::<f32>(intermediate)?;
        let mut d_down = stream.alloc_zeros::<f32>(out_dim)?;
        let mut d_out = stream.alloc_zeros::<f32>(out_dim)?;

        let mut gate_up_qgemv_ms = 0.0;
        let mut activation_ms = 0.0;
        let mut down_qgemv_ms = 0.0;
        let mut accum_ms = 0.0;

        let mut actual_expert_bytes_loaded = 0;
        let mut resident_cache_bytes_reused = 0;
        let mut gate_bytes = 0;
        let mut up_bytes = 0;
        let mut down_bytes = 0;

        // Calculate unique experts selected to compute working set
        let mut unique_selected_experts = std::collections::HashSet::new();
        for &(e, _) in selected_experts {
            unique_selected_experts.insert(e);
        }
        let single_expert_bytes = shape_gate_up.quantized_matrix_bytes() * 2 + shape_down.quantized_matrix_bytes();
        let selected_working_set_bytes = unique_selected_experts.len() * single_expert_bytes;

        for &(e, weight) in selected_experts {
            let expert = &experts[e];
            gate_bytes += expert.w_gate.len();
            up_bytes += expert.w_up.len();
            down_bytes += expert.w_down.len();

            let gate_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::Gate,
                quant_format: QuantFormat::Q4_0,
            };
            let up_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::Up,
                quant_format: QuantFormat::Q4_0,
            };
            let down_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::Down,
                quant_format: QuantFormat::Q4_0,
            };

            let expert_bytes = expert.w_gate.len() + expert.w_up.len() + expert.w_down.len();
            let mut bypass_expert = false;
            if let Some(ref mut c) = cache {
                if expert_bytes > c.capacity_bytes {
                    if c.bypass_oversized_experts {
                        bypass_expert = true;
                        c.oversized_expert_bypass_count += 1;
                    } else {
                        c.self_eviction_risk_count += 1;
                    }
                }
            }

            let mut w_h2d_ms = 0.0;

            let d_w_gate = if bypass_expert {
                if let Some(ref mut c) = cache {
                    c.cache_insert_attempt_count += 1;
                    c.cache_insert_bypass_count += 1;
                }
                let t = CudaEventTimer::start(stream.raw())?;
                let buf = stream.copy_from_slice(&expert.w_gate)?;
                w_h2d_ms += t.stop("moe_gate_h2d", stream.raw())?.elapsed_ms;
                actual_expert_bytes_loaded += expert.w_gate.len();
                Arc::new(buf)
            } else if let Some(ref mut c) = cache {
                if let Some(buf) = c.get(&gate_key) {
                    resident_cache_bytes_reused += expert.w_gate.len();
                    buf
                } else {
                    let t = CudaEventTimer::start(stream.raw())?;
                    let buf = stream.copy_from_slice(&expert.w_gate)?;
                    w_h2d_ms += t.stop("moe_gate_h2d", stream.raw())?.elapsed_ms;
                    actual_expert_bytes_loaded += expert.w_gate.len();
                    c.insert(gate_key, buf)
                }
            } else {
                let t = CudaEventTimer::start(stream.raw())?;
                let buf = stream.copy_from_slice(&expert.w_gate)?;
                w_h2d_ms += t.stop("moe_gate_h2d", stream.raw())?.elapsed_ms;
                actual_expert_bytes_loaded += expert.w_gate.len();
                Arc::new(buf)
            };

            let d_w_up = if bypass_expert {
                if let Some(ref mut c) = cache {
                    c.cache_insert_attempt_count += 1;
                    c.cache_insert_bypass_count += 1;
                }
                let t = CudaEventTimer::start(stream.raw())?;
                let buf = stream.copy_from_slice(&expert.w_up)?;
                w_h2d_ms += t.stop("moe_up_h2d", stream.raw())?.elapsed_ms;
                actual_expert_bytes_loaded += expert.w_up.len();
                Arc::new(buf)
            } else if let Some(ref mut c) = cache {
                if let Some(buf) = c.get(&up_key) {
                    resident_cache_bytes_reused += expert.w_up.len();
                    buf
                } else {
                    let t = CudaEventTimer::start(stream.raw())?;
                    let buf = stream.copy_from_slice(&expert.w_up)?;
                    w_h2d_ms += t.stop("moe_up_h2d", stream.raw())?.elapsed_ms;
                    actual_expert_bytes_loaded += expert.w_up.len();
                    c.insert(up_key, buf)
                }
            } else {
                let t = CudaEventTimer::start(stream.raw())?;
                let buf = stream.copy_from_slice(&expert.w_up)?;
                w_h2d_ms += t.stop("moe_up_h2d", stream.raw())?.elapsed_ms;
                actual_expert_bytes_loaded += expert.w_up.len();
                Arc::new(buf)
            };

            let d_w_down = if bypass_expert {
                if let Some(ref mut c) = cache {
                    c.cache_insert_attempt_count += 1;
                    c.cache_insert_bypass_count += 1;
                }
                let t = CudaEventTimer::start(stream.raw())?;
                let buf = stream.copy_from_slice(&expert.w_down)?;
                w_h2d_ms += t.stop("moe_down_h2d", stream.raw())?.elapsed_ms;
                actual_expert_bytes_loaded += expert.w_down.len();
                Arc::new(buf)
            } else if let Some(ref mut c) = cache {
                if let Some(buf) = c.get(&down_key) {
                    resident_cache_bytes_reused += expert.w_down.len();
                    buf
                } else {
                    let t = CudaEventTimer::start(stream.raw())?;
                    let buf = stream.copy_from_slice(&expert.w_down)?;
                    w_h2d_ms += t.stop("moe_down_h2d", stream.raw())?.elapsed_ms;
                    actual_expert_bytes_loaded += expert.w_down.len();
                    c.insert(down_key, buf)
                }
            } else {
                let t = CudaEventTimer::start(stream.raw())?;
                let buf = stream.copy_from_slice(&expert.w_down)?;
                w_h2d_ms += t.stop("moe_down_h2d", stream.raw())?.elapsed_ms;
                actual_expert_bytes_loaded += expert.w_down.len();
                Arc::new(buf)
            };

            h2d_ms += w_h2d_ms;

            // gate = W_gate @ x
            // up = W_up @ x
            let gate_up_timer = CudaEventTimer::start(stream.raw())?;
            quant.launch_kernel(QuantFormat::Q4_0, stream.raw(), &*d_w_gate, &d_x, &mut d_gate, shape_gate_up)?;
            quant.launch_kernel(QuantFormat::Q4_0, stream.raw(), &*d_w_up, &d_x, &mut d_up, shape_gate_up)?;
            let gate_up_timing = gate_up_timer.stop("moe_gate_up_qgemv", stream.raw())?;
            gate_up_qgemv_ms += gate_up_timing.elapsed_ms;

            // act = silu(gate) * up
            let act_timer = CudaEventTimer::start(stream.raw())?;
            let cfg_silu = LaunchConfig {
                grid_dim: (((intermediate + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let intermediate_u32 = intermediate as u32;
            let moe_guard = self.ensure_module()?;
            let moe_mod = moe_guard.as_ref().expect("MoE module loaded");
            cuda_map_err!(
                CudaErrorKind::Driver,
                "launch silu_mul kernel",
                unsafe {
                    stream
                        .raw()
                        .launch_builder(&moe_mod.silu_mul)
                        .arg(&d_gate.raw)
                        .arg(&d_up.raw)
                        .arg(&mut d_act.raw)
                        .arg(&intermediate_u32)
                        .arg(&0.0f32)
                        .launch(cfg_silu)
                }
            )?;
            let act_timing = act_timer.stop("moe_activation", stream.raw())?;
            activation_ms += act_timing.elapsed_ms;

            // down = W_down @ act
            let down_timer = CudaEventTimer::start(stream.raw())?;
            quant.launch_kernel(QuantFormat::Q4_0, stream.raw(), &*d_w_down, &d_act, &mut d_down, shape_down)?;
            let down_timing = down_timer.stop("moe_down_qgemv", stream.raw())?;
            down_qgemv_ms += down_timing.elapsed_ms;

            // out += selected_weight * down
            let accum_timer = CudaEventTimer::start(stream.raw())?;
            let cfg_accum = LaunchConfig {
                grid_dim: (((out_dim + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_dim_u32 = out_dim as u32;
            cuda_map_err!(
                CudaErrorKind::Driver,
                "launch weighted_accum kernel",
                unsafe {
                    stream
                        .raw()
                        .launch_builder(&moe_mod.weighted_accum)
                        .arg(&d_down.raw)
                        .arg(&mut d_out.raw)
                        .arg(&weight)
                        .arg(&out_dim_u32)
                        .launch(cfg_accum)
                }
            )?;
            let accum_timing = accum_timer.stop("moe_accum", stream.raw())?;
            accum_ms += accum_timing.elapsed_ms;
        }

        // Copy back to host
        let d2h_timer = CudaEventTimer::start(stream.raw())?;
        let out = stream.copy_to_vec(&d_out)?;
        let d2h_timing = d2h_timer.stop("moe_d2h", stream.raw())?;
        let d2h_ms = d2h_timing.elapsed_ms;

        let total_timing = total_timer.stop("moe_total", stream.raw())?;
        let total_ms = total_timing.elapsed_ms;

        let unaccounted_ms = (total_ms - h2d_ms - gate_up_qgemv_ms - activation_ms - down_qgemv_ms - accum_ms - d2h_ms).max(0.0);

        let logical_expert_bytes_requested = gate_bytes + up_bytes + down_bytes;

        let (
            resident_cache_capacity_bytes,
            resident_cache_resident_bytes,
            resident_cache_hit_count,
            resident_cache_miss_count,
            resident_cache_eviction_count,
            resident_cache_insert_attempt_count,
            resident_cache_insert_accept_count,
            resident_cache_insert_bypass_count,
            resident_cache_oversized_tensor_bypass_count,
            resident_cache_oversized_expert_bypass_count,
            resident_cache_self_eviction_risk_count,
        ) = if let Some(ref c) = cache {
            (
                c.capacity_bytes,
                c.resident_bytes,
                c.hit_count,
                c.miss_count,
                c.eviction_count,
                c.cache_insert_attempt_count,
                c.cache_insert_accept_count,
                c.cache_insert_bypass_count,
                c.oversized_tensor_bypass_count,
                c.oversized_expert_bypass_count,
                c.self_eviction_risk_count,
            )
        } else {
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        };

        let telemetry = MoeTelemetry {
            selected_expert_count: selected_experts.len(),
            logical_expert_bytes_requested,
            actual_expert_bytes_loaded,
            resident_cache_bytes_reused,
            weight_bytes_loaded: actual_expert_bytes_loaded,
            weight_bytes_reused: resident_cache_bytes_reused,
            scale_bytes_loaded: 0,
            scale_bytes_reused: 0,
            resident_cache_resident_bytes,
            resident_cache_capacity_bytes,
            resident_cache_hit_count,
            resident_cache_miss_count,
            resident_cache_eviction_count,
            resident_cache_insert_attempt_count,
            resident_cache_insert_accept_count,
            resident_cache_insert_bypass_count,
            resident_cache_oversized_tensor_bypass_count,
            resident_cache_oversized_expert_bypass_count,
            resident_cache_self_eviction_risk_count,
            dequantized_scratch_bytes: 0,
            h2d_ms,
            gate_up_qgemv_ms,
            activation_ms,
            down_qgemv_ms,
            accum_ms,
            unaccounted_ms,
            total_ms,
            shared_expert_ms: 0.0,
            bytes_by_tensor_kind: BytesByTensorKind {
                gate: gate_bytes,
                up: up_bytes,
                down: down_bytes,
            },
            bytes_per_expert: single_expert_bytes,
            selected_working_set_bytes,
            routed_fp4_bytes_loaded: 0,
            routed_fp4_bytes_reused: 0,
            shared_fp8_bytes_loaded: 0,
            shared_fp8_bytes_reused: 0,
            total_logical_bytes: 0,
            total_loaded_bytes: 0,
            total_reused_bytes: 0,
        };

        // Assert invariant explicitly
        assert_eq!(
            telemetry.logical_expert_bytes_requested,
            telemetry.actual_expert_bytes_loaded + telemetry.resident_cache_bytes_reused,
            "MoE Byte invariant violated!"
        );

        Ok((out, telemetry))
    }

    fn load_tensor_to_device(
        &self,
        stream: &CudaStreamHandle,
        cache: &mut Option<&mut CudaExpertCache>,
        key: ExpertCacheKey,
        data: &[u8],
        bypass: bool,
        weight_bytes_loaded: &mut usize,
        weight_bytes_reused: &mut usize,
        scale_bytes_loaded: &mut usize,
        scale_bytes_reused: &mut usize,
        h2d_ms: &mut f32,
    ) -> Result<Arc<DeviceBuffer<u8>>> {
        let is_scale = matches!(
            key.tensor_kind,
            ExpertTensorKind::GateScale | ExpertTensorKind::UpScale | ExpertTensorKind::DownScale
        );
        if bypass {
            if let Some(c) = cache {
                c.cache_insert_attempt_count += 1;
                c.cache_insert_bypass_count += 1;
            }
            let t = CudaEventTimer::start(stream.raw())?;
            let buf = stream.copy_from_slice(data)?;
            *h2d_ms += t.stop("moe_fp4_h2d", stream.raw())?.elapsed_ms;
            if is_scale {
                *scale_bytes_loaded += data.len();
            } else {
                *weight_bytes_loaded += data.len();
            }
            Ok(Arc::new(buf))
        } else if let Some(c) = cache {
            if let Some(buf) = c.get(&key) {
                if is_scale {
                    *scale_bytes_reused += data.len();
                } else {
                    *weight_bytes_reused += data.len();
                }
                Ok(buf)
            } else {
                let t = CudaEventTimer::start(stream.raw())?;
                let buf = stream.copy_from_slice(data)?;
                *h2d_ms += t.stop("moe_fp4_h2d", stream.raw())?.elapsed_ms;
                if is_scale {
                    *scale_bytes_loaded += data.len();
                } else {
                    *weight_bytes_loaded += data.len();
                }
                Ok(c.insert(key, buf))
            }
        } else {
            let t = CudaEventTimer::start(stream.raw())?;
            let buf = stream.copy_from_slice(data)?;
            *h2d_ms += t.stop("moe_fp4_h2d", stream.raw())?.elapsed_ms;
            if is_scale {
                *scale_bytes_loaded += data.len();
            } else {
                *weight_bytes_loaded += data.len();
            }
            Ok(Arc::new(buf))
        }
    }

    pub fn execute_selected_moe_native_fp4_cuda(
        &self,
        quant: &QuantBackend,
        stream: &CudaStreamHandle,
        experts: &[DeepSeekFp4ExpertWeights],
        selected_experts: &[(usize, f32)],
        x: &[f32],
        hidden: usize,
        intermediate: usize,
        out_dim: usize,
        layer_id: usize,
        mut cache: Option<&mut CudaExpertCache>,
    ) -> Result<(Vec<f32>, MoeTelemetry)> {
        // Validate inputs
        if x.len() != hidden {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate execute_selected_moe_native_fp4_cuda input x size",
                format!("x len {} != hidden {}", x.len(), hidden),
                file!(),
                line!(),
                module_path!(),
            ));
        }

        let expected_gate_up_weight_bytes = intermediate * (hidden / 2);
        let expected_gate_up_scale_bytes = intermediate * (hidden / 32);
        let expected_down_weight_bytes = out_dim * (intermediate / 2);
        let expected_down_scale_bytes = out_dim * (intermediate / 32);

        for &(e, _) in selected_experts {
            if e >= experts.len() {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert index",
                    format!("expert index {} out of bounds (len={})", e, experts.len()),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            let expert = &experts[e];
            if expert.gate_weight.len() != expected_gate_up_weight_bytes {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert gate_weight size",
                    format!("gate_weight len {} != expected {}", expert.gate_weight.len(), expected_gate_up_weight_bytes),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            if expert.gate_scale.len() != expected_gate_up_scale_bytes {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert gate_scale size",
                    format!("gate_scale len {} != expected {}", expert.gate_scale.len(), expected_gate_up_scale_bytes),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            if expert.up_weight.len() != expected_gate_up_weight_bytes {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert up_weight size",
                    format!("up_weight len {} != expected {}", expert.up_weight.len(), expected_gate_up_weight_bytes),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            if expert.up_scale.len() != expected_gate_up_scale_bytes {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert up_scale size",
                    format!("up_scale len {} != expected {}", expert.up_scale.len(), expected_gate_up_scale_bytes),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            if expert.down_weight.len() != expected_down_weight_bytes {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert down_weight size",
                    format!("down_weight len {} != expected {}", expert.down_weight.len(), expected_down_weight_bytes),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            if expert.down_scale.len() != expected_down_scale_bytes {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert down_scale size",
                    format!("down_scale len {} != expected {}", expert.down_scale.len(), expected_down_scale_bytes),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
        }

        let total_timer = CudaEventTimer::start(stream.raw())?;

        let mut h2d_ms = 0.0;
        let x_h2d_timer = CudaEventTimer::start(stream.raw())?;
        let d_x = stream.copy_from_slice(x)?;
        let x_h2d_timing = x_h2d_timer.stop("moe_x_h2d", stream.raw())?;
        h2d_ms += x_h2d_timing.elapsed_ms;

        // Allocate device buffers for intermediate values
        let mut d_gate = stream.alloc_zeros::<f32>(intermediate)?;
        let mut d_up = stream.alloc_zeros::<f32>(intermediate)?;
        let mut d_act = stream.alloc_zeros::<f32>(intermediate)?;
        let mut d_down = stream.alloc_zeros::<f32>(out_dim)?;
        let mut d_out = stream.alloc_zeros::<f32>(out_dim)?;

        let mut gate_up_qgemv_ms = 0.0;
        let mut activation_ms = 0.0;
        let mut down_qgemv_ms = 0.0;
        let mut accum_ms = 0.0;

        let mut weight_bytes_loaded = 0;
        let mut weight_bytes_reused = 0;
        let mut scale_bytes_loaded = 0;
        let mut scale_bytes_reused = 0;
        let mut gate_bytes = 0;
        let mut up_bytes = 0;
        let mut down_bytes = 0;

        // Calculate unique experts selected to compute working set
        let mut unique_selected_experts = std::collections::HashSet::new();
        for &(e, _) in selected_experts {
            unique_selected_experts.insert(e);
        }
        let single_expert_bytes = expected_gate_up_weight_bytes * 2 + expected_gate_up_scale_bytes * 2 + expected_down_weight_bytes + expected_down_scale_bytes;
        let selected_working_set_bytes = unique_selected_experts.len() * single_expert_bytes;

        for &(e, weight) in selected_experts {
            let expert = &experts[e];
            let g_bytes = expert.gate_weight.len() + expert.gate_scale.len();
            let u_bytes = expert.up_weight.len() + expert.up_scale.len();
            let d_bytes = expert.down_weight.len() + expert.down_scale.len();
            gate_bytes += g_bytes;
            up_bytes += u_bytes;
            down_bytes += d_bytes;

            let gate_w_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::GateWeight,
                quant_format: QuantFormat::DeepSeekFp4E2M1,
            };
            let gate_s_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::GateScale,
                quant_format: QuantFormat::DeepSeekFp4E2M1,
            };
            let up_w_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::UpWeight,
                quant_format: QuantFormat::DeepSeekFp4E2M1,
            };
            let up_s_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::UpScale,
                quant_format: QuantFormat::DeepSeekFp4E2M1,
            };
            let down_w_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::DownWeight,
                quant_format: QuantFormat::DeepSeekFp4E2M1,
            };
            let down_s_key = ExpertCacheKey {
                layer_id,
                expert_id: e,
                tensor_kind: ExpertTensorKind::DownScale,
                quant_format: QuantFormat::DeepSeekFp4E2M1,
            };

            let expert_bytes = g_bytes + u_bytes + d_bytes;
            let mut bypass_expert = false;
            if let Some(ref mut c) = cache {
                if expert_bytes > c.capacity_bytes {
                    if c.bypass_oversized_experts {
                        bypass_expert = true;
                        c.oversized_expert_bypass_count += 1;
                    } else {
                        c.self_eviction_risk_count += 1;
                    }
                }
            }

            let d_w_gate = self.load_tensor_to_device(
                stream,
                &mut cache,
                gate_w_key,
                &expert.gate_weight,
                bypass_expert,
                &mut weight_bytes_loaded,
                &mut weight_bytes_reused,
                &mut scale_bytes_loaded,
                &mut scale_bytes_reused,
                &mut h2d_ms,
            )?;
            let d_w_gate_scale = self.load_tensor_to_device(
                stream,
                &mut cache,
                gate_s_key,
                &expert.gate_scale,
                bypass_expert,
                &mut weight_bytes_loaded,
                &mut weight_bytes_reused,
                &mut scale_bytes_loaded,
                &mut scale_bytes_reused,
                &mut h2d_ms,
            )?;

            let d_w_up = self.load_tensor_to_device(
                stream,
                &mut cache,
                up_w_key,
                &expert.up_weight,
                bypass_expert,
                &mut weight_bytes_loaded,
                &mut weight_bytes_reused,
                &mut scale_bytes_loaded,
                &mut scale_bytes_reused,
                &mut h2d_ms,
            )?;
            let d_w_up_scale = self.load_tensor_to_device(
                stream,
                &mut cache,
                up_s_key,
                &expert.up_scale,
                bypass_expert,
                &mut weight_bytes_loaded,
                &mut weight_bytes_reused,
                &mut scale_bytes_loaded,
                &mut scale_bytes_reused,
                &mut h2d_ms,
            )?;

            let d_w_down = self.load_tensor_to_device(
                stream,
                &mut cache,
                down_w_key,
                &expert.down_weight,
                bypass_expert,
                &mut weight_bytes_loaded,
                &mut weight_bytes_reused,
                &mut scale_bytes_loaded,
                &mut scale_bytes_reused,
                &mut h2d_ms,
            )?;
            let d_w_down_scale = self.load_tensor_to_device(
                stream,
                &mut cache,
                down_s_key,
                &expert.down_scale,
                bypass_expert,
                &mut weight_bytes_loaded,
                &mut weight_bytes_reused,
                &mut scale_bytes_loaded,
                &mut scale_bytes_reused,
                &mut h2d_ms,
            )?;

            // gate = W_gate @ x
            // up = W_up @ x
            let gate_up_timer = CudaEventTimer::start(stream.raw())?;
            quant.launch_kernel_split_fp4(
                stream.raw(),
                &*d_w_gate,
                &*d_w_gate_scale,
                &d_x,
                &mut d_gate,
                intermediate,
                hidden,
            )?;
            quant.launch_kernel_split_fp4(
                stream.raw(),
                &*d_w_up,
                &*d_w_up_scale,
                &d_x,
                &mut d_up,
                intermediate,
                hidden,
            )?;
            let gate_up_timing = gate_up_timer.stop("moe_gate_up_qgemv", stream.raw())?;
            gate_up_qgemv_ms += gate_up_timing.elapsed_ms;

            // act = silu(gate) * up
            let act_timer = CudaEventTimer::start(stream.raw())?;
            let cfg_silu = LaunchConfig {
                grid_dim: (((intermediate + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let intermediate_u32 = intermediate as u32;
            let moe_guard = self.ensure_module()?;
            let moe_mod = moe_guard.as_ref().expect("MoE module loaded");
            cuda_map_err!(
                CudaErrorKind::Driver,
                "launch silu_mul kernel",
                unsafe {
                    stream
                        .raw()
                        .launch_builder(&moe_mod.silu_mul)
                        .arg(&d_gate.raw)
                        .arg(&d_up.raw)
                        .arg(&mut d_act.raw)
                        .arg(&intermediate_u32)
                        .arg(&0.0f32)
                        .launch(cfg_silu)
                }
            )?;
            let act_timing = act_timer.stop("moe_activation", stream.raw())?;
            activation_ms += act_timing.elapsed_ms;

            // down = W_down @ act
            let down_timer = CudaEventTimer::start(stream.raw())?;
            quant.launch_kernel_split_fp4(
                stream.raw(),
                &*d_w_down,
                &*d_w_down_scale,
                &d_act,
                &mut d_down,
                out_dim,
                intermediate,
            )?;
            let down_timing = down_timer.stop("moe_down_qgemv", stream.raw())?;
            down_qgemv_ms += down_timing.elapsed_ms;

            // out += selected_weight * down
            let accum_timer = CudaEventTimer::start(stream.raw())?;
            let cfg_accum = LaunchConfig {
                grid_dim: (((out_dim + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_dim_u32 = out_dim as u32;
            cuda_map_err!(
                CudaErrorKind::Driver,
                "launch weighted_accum kernel",
                unsafe {
                    stream
                        .raw()
                        .launch_builder(&moe_mod.weighted_accum)
                        .arg(&d_down.raw)
                        .arg(&mut d_out.raw)
                        .arg(&weight)
                        .arg(&out_dim_u32)
                        .launch(cfg_accum)
                }
            )?;
            let accum_timing = accum_timer.stop("moe_accum", stream.raw())?;
            accum_ms += accum_timing.elapsed_ms;
        }

        // Copy back to host
        let d2h_timer = CudaEventTimer::start(stream.raw())?;
        let out = stream.copy_to_vec(&d_out)?;
        let d2h_timing = d2h_timer.stop("moe_d2h", stream.raw())?;
        let d2h_ms = d2h_timing.elapsed_ms;

        let total_timing = total_timer.stop("moe_total", stream.raw())?;
        let total_ms = total_timing.elapsed_ms;

        let unaccounted_ms = (total_ms - h2d_ms - gate_up_qgemv_ms - activation_ms - down_qgemv_ms - accum_ms - d2h_ms).max(0.0);

        let logical_expert_bytes_requested = gate_bytes + up_bytes + down_bytes;
        let actual_expert_bytes_loaded = weight_bytes_loaded + scale_bytes_loaded;
        let resident_cache_bytes_reused = weight_bytes_reused + scale_bytes_reused;

        let (
            resident_cache_capacity_bytes,
            resident_cache_resident_bytes,
            resident_cache_hit_count,
            resident_cache_miss_count,
            resident_cache_eviction_count,
            resident_cache_insert_attempt_count,
            resident_cache_insert_accept_count,
            resident_cache_insert_bypass_count,
            resident_cache_oversized_tensor_bypass_count,
            resident_cache_oversized_expert_bypass_count,
            resident_cache_self_eviction_risk_count,
        ) = if let Some(ref c) = cache {
            (
                c.capacity_bytes,
                c.resident_bytes,
                c.hit_count,
                c.miss_count,
                c.eviction_count,
                c.cache_insert_attempt_count,
                c.cache_insert_accept_count,
                c.cache_insert_bypass_count,
                c.oversized_tensor_bypass_count,
                c.oversized_expert_bypass_count,
                c.self_eviction_risk_count,
            )
        } else {
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        };

        let telemetry = MoeTelemetry {
            selected_expert_count: selected_experts.len(),
            logical_expert_bytes_requested,
            actual_expert_bytes_loaded,
            resident_cache_bytes_reused,
            weight_bytes_loaded,
            weight_bytes_reused,
            scale_bytes_loaded,
            scale_bytes_reused,
            resident_cache_resident_bytes,
            resident_cache_capacity_bytes,
            resident_cache_hit_count,
            resident_cache_miss_count,
            resident_cache_eviction_count,
            resident_cache_insert_attempt_count,
            resident_cache_insert_accept_count,
            resident_cache_insert_bypass_count,
            resident_cache_oversized_tensor_bypass_count,
            resident_cache_oversized_expert_bypass_count,
            resident_cache_self_eviction_risk_count,
            dequantized_scratch_bytes: 0,
            h2d_ms,
            gate_up_qgemv_ms,
            activation_ms,
            down_qgemv_ms,
            accum_ms,
            unaccounted_ms,
            total_ms,
            shared_expert_ms: 0.0,
            bytes_by_tensor_kind: BytesByTensorKind {
                gate: gate_bytes,
                up: up_bytes,
                down: down_bytes,
            },
            bytes_per_expert: single_expert_bytes,
            selected_working_set_bytes,
            routed_fp4_bytes_loaded: 0,
            routed_fp4_bytes_reused: 0,
            shared_fp8_bytes_loaded: 0,
            shared_fp8_bytes_reused: 0,
            total_logical_bytes: 0,
            total_loaded_bytes: 0,
            total_reused_bytes: 0,
        };

        // Assert invariant explicitly
        assert_eq!(
            telemetry.logical_expert_bytes_requested,
            telemetry.actual_expert_bytes_loaded + telemetry.resident_cache_bytes_reused,
            "MoE Byte invariant violated!"
        );

        Ok((out, telemetry))
    }
}

pub fn selected_moe_cpu(
    experts: &[ExpertWeights],
    selected_experts: &[(usize, f32)],
    x: &[f32],
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; out_dim];
    let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate, hidden);
    let shape_down = QGemvShape::new(QuantFormat::Q4_0, out_dim, intermediate);

    for &(e, weight) in selected_experts {
        if e >= experts.len() {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate expert index CPU",
                format!("expert index {} out of bounds (len={})", e, experts.len()),
                file!(),
                line!(),
                module_path!(),
            ));
        }
        let expert = &experts[e];
        let gate = gemv_cpu(QuantFormat::Q4_0, &expert.w_gate, x, shape_gate_up)?;
        let up = gemv_cpu(QuantFormat::Q4_0, &expert.w_up, x, shape_gate_up)?;

        let mut act = vec![0.0f32; intermediate];
        for i in 0..intermediate {
            let g = gate[i];
            let silu = g / (1.0f32 + (-g).exp());
            act[i] = silu * up[i];
        }

        let down = gemv_cpu(QuantFormat::Q4_0, &expert.w_down, &act, shape_down)?;

        for i in 0..out_dim {
            out[i] += weight * down[i];
        }
    }
    Ok(out)
}

pub fn selected_moe_cpu_fp32(
    experts: &[ExpertWeightsFp32],
    selected_experts: &[(usize, f32)],
    x: &[f32],
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; out_dim];
    let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate, hidden);
    let shape_down = QGemvShape::new(QuantFormat::Q4_0, out_dim, intermediate);

    for &(e, weight) in selected_experts {
        if e >= experts.len() {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate expert index CPU fp32",
                format!("expert index {} out of bounds (len={})", e, experts.len()),
                file!(),
                line!(),
                module_path!(),
            ));
        }
        let expert = &experts[e];
        let gate = dense_gemv_cpu(&expert.w_gate, x, shape_gate_up)?;
        let up = dense_gemv_cpu(&expert.w_up, x, shape_gate_up)?;

        let mut act = vec![0.0f32; intermediate];
        for i in 0..intermediate {
            let g = gate[i];
            let silu = g / (1.0f32 + (-g).exp());
            act[i] = silu * up[i];
        }

        let down = dense_gemv_cpu(&expert.w_down, &act, shape_down)?;

        for i in 0..out_dim {
            out[i] += weight * down[i];
        }
    }
    Ok(out)
}

// ── Shared FP8 expert device struct ──────────────────────────────────────────

/// Device-side shared expert weights (F8_E4M3 + F8_E8M0 tile scales).
pub struct DeepSeekFp8SharedExpertWeightsDevice {
    pub gate_weight: DeviceBuffer<u8>,
    pub gate_scale: DeviceBuffer<u8>,
    pub up_weight: DeviceBuffer<u8>,
    pub up_scale: DeviceBuffer<u8>,
    pub down_weight: DeviceBuffer<u8>,
    pub down_scale: DeviceBuffer<u8>,
}

// ── Official arithmetic: device-resident routed MoE ─────────────────────────

/// Official-arithmetic device-resident routed expert CUDA execution.
///
/// Uses act_quant → fp8_act×fp4_wt GEMV pipeline with no host roundtrips
/// between quantization and GEMV operations.
///
/// For each selected expert, the gate/up projections reuse the same quantized
/// hidden state (computed once). The intermediate activation after SwiGLU is
/// quantized separately before the down projection.
///
/// Returns (output_fp32, telemetry).
pub fn execute_selected_moe_official_routed_fp4_cuda(
    quant: &QuantBackend,
    moe_executor: &MoeExecutor,
    stream: &CudaStreamHandle,
    experts: &[DeepSeekFp4ExpertWeights],
    selected_experts: &[(usize, f32)],
    hidden: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    out_dim: usize,
    layer_id: usize,
    mut cache: Option<&mut CudaExpertCache>,
    shared_expert: Option<&DeepSeekFp8SharedExpertWeightsDevice>,
) -> Result<(Vec<f32>, MoeTelemetry)> {
    let total_timer = CudaEventTimer::start(stream.raw())?;

    // 1. Upload hidden state to device
    let h2d_start = CudaEventTimer::start(stream.raw())?;
    let d_hidden = stream.copy_from_slice(hidden)?;
    let h2d_ms = h2d_start.stop("official_h2d", stream.raw())?.elapsed_ms;

    // 2. act_quant(hidden) — computed once, reused for gate/up projections
    let (d_act, input_act_quant_ms) =
        cuda_act_quant_device(quant, stream, &d_hidden, 1, hidden_size)?;

    // Allocate intermediate device buffers
    let mut d_gate = stream.alloc_zeros::<f32>(intermediate_size)?;
    let mut d_up = stream.alloc_zeros::<f32>(intermediate_size)?;
    let mut d_act_inter = stream.alloc_zeros::<f32>(intermediate_size)?;
    let mut d_down = stream.alloc_zeros::<f32>(out_dim)?;
    let mut d_out = stream.alloc_zeros::<f32>(out_dim)?;

    let mut gate_up_ms = 0.0f32;
    let mut act_ms = 0.0f32;
    let mut inter_act_quant_ms = 0.0f32;
    let mut down_ms = 0.0f32;
    let mut accum_ms = 0.0f32;

    let mut weight_bytes_loaded = 0usize;
    let mut weight_bytes_reused = 0usize;
    let mut scale_bytes_loaded = 0usize;
    let mut scale_bytes_reused = 0usize;

    let single_expert_bytes = (intermediate_size * hidden_size / 2) * 2  // gate+up weights
        + (intermediate_size * hidden_size / 32) * 2                      // gate+up scales
        + (out_dim * intermediate_size / 2)                                // down weight
        + (out_dim * intermediate_size / 32);                              // down scale

    let mut unique_ids = std::collections::HashSet::new();
    for &(e, _) in selected_experts { unique_ids.insert(e); }
    let selected_working_set_bytes = unique_ids.len() * single_expert_bytes;

    for &(eid, weight) in selected_experts {
        let expert = &experts[eid];

        // Build cache keys
        let make_key = |tid: ExpertTensorKind| ExpertCacheKey {
            layer_id, expert_id: eid, tensor_kind: tid,
            quant_format: QuantFormat::DeepSeekFp4E2M1,
        };

        let expert_total = expert.gate_weight.len() + expert.gate_scale.len()
            + expert.up_weight.len() + expert.up_scale.len()
            + expert.down_weight.len() + expert.down_scale.len();
        let bypass = cache.as_ref().map_or(false, |c| expert_total > c.capacity_bytes && c.bypass_oversized_experts);

        // Helper: load tensor to device buffer, tracking hit/miss bytes
        let is_scale = |key: &ExpertCacheKey| -> bool {
            matches!(key.tensor_kind, ExpertTensorKind::GateScale | ExpertTensorKind::UpScale | ExpertTensorKind::DownScale)
        };
        let mut load = |cache: &mut Option<&mut CudaExpertCache>, key: ExpertCacheKey, data: &[u8], bypass: bool|
            -> Result<Arc<DeviceBuffer<u8>>> {
            let len = data.len();
            if bypass {
                if let Some(ref mut c) = cache {
                    c.cache_insert_attempt_count += 1;
                    c.cache_insert_bypass_count += 1;
                }
                if is_scale(&key) { scale_bytes_loaded += len; }
                else { weight_bytes_loaded += len; }
                Ok(Arc::new(stream.copy_from_slice(data)?))
            } else if let Some(ref mut c) = cache {
                if let Some(buf) = c.get(&key) {
                    if is_scale(&key) { scale_bytes_reused += len; }
                    else { weight_bytes_reused += len; }
                    Ok(buf)
                } else {
                    if is_scale(&key) { scale_bytes_loaded += len; }
                    else { weight_bytes_loaded += len; }
                    Ok(c.insert(key, stream.copy_from_slice(data)?))
                }
            } else {
                if is_scale(&key) { scale_bytes_loaded += len; }
                else { weight_bytes_loaded += len; }
                Ok(Arc::new(stream.copy_from_slice(data)?))
            }
        };

        let d_gate_w = load(&mut cache, make_key(ExpertTensorKind::GateWeight), &expert.gate_weight, bypass)?;
        let d_gate_s = load(&mut cache, make_key(ExpertTensorKind::GateScale), &expert.gate_scale, bypass)?;
        let d_up_w = load(&mut cache, make_key(ExpertTensorKind::UpWeight), &expert.up_weight, bypass)?;
        let d_up_s = load(&mut cache, make_key(ExpertTensorKind::UpScale), &expert.up_scale, bypass)?;
        let d_down_w = load(&mut cache, make_key(ExpertTensorKind::DownWeight), &expert.down_weight, bypass)?;
        let d_down_s = load(&mut cache, make_key(ExpertTensorKind::DownScale), &expert.down_scale, bypass)?;

        // Gate GEMV
        let ms = cuda_fp8_act_fp4_weight_gemv_device(
            quant, stream,
            &d_act.values, &d_act.scales,
            &d_gate_w, &d_gate_s, &mut d_gate,
            intermediate_size, hidden_size,
        )?;
        gate_up_ms += ms;

        // Up GEMV
        let ms = cuda_fp8_act_fp4_weight_gemv_device(
            quant, stream,
            &d_act.values, &d_act.scales,
            &d_up_w, &d_up_s, &mut d_up,
            intermediate_size, hidden_size,
        )?;
        gate_up_ms += ms;

        // SwiGLU activation: silu_mul(gate, up) on device
        moe_executor.compile()?;
        if let Some(moe_mod) = &*moe_executor.module.lock().map_err(|e| CudaError::new(CudaErrorKind::Internal, "lock moe module", e.to_string(), file!(), line!(), module_path!()))? {
            let t = CudaEventTimer::start(stream.raw())?;
            let cfg = LaunchConfig { grid_dim: ((intermediate_size as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            cuda_map_err!(CudaErrorKind::Driver, "launch silu_mul",
                unsafe { stream.raw().launch_builder(&moe_mod.silu_mul)
                    .arg(&d_gate.raw).arg(&d_up.raw).arg(&d_act_inter.raw).arg(&(intermediate_size as u32)).arg(&10.0f32).launch(cfg) })?;
            act_ms += t.stop("official_silu", stream.raw())?.elapsed_ms;
        }

        // act_quant(intermediate activation) for down projection
        let (d_act2, aq_ms) = cuda_act_quant_device(quant, stream, &d_act_inter, 1, intermediate_size)?;
        inter_act_quant_ms += aq_ms;

        // Down GEMV
        let ms = cuda_fp8_act_fp4_weight_gemv_device(
            quant, stream,
            &d_act2.values, &d_act2.scales,
            &d_down_w, &d_down_s, &mut d_down,
            out_dim, intermediate_size,
        )?;
        down_ms += ms;

        // Weighted accumulate: d_out += weight * d_down
        if let Some(moe_mod) = &*moe_executor.module.lock().map_err(|e| CudaError::new(CudaErrorKind::Internal, "lock moe module2", e.to_string(), file!(), line!(), module_path!()))? {
            let t = CudaEventTimer::start(stream.raw())?;
            let cfg = LaunchConfig { grid_dim: ((out_dim as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            cuda_map_err!(CudaErrorKind::Driver, "launch weighted_accum",
                unsafe { stream.raw().launch_builder(&moe_mod.weighted_accum)
                    .arg(&d_down.raw).arg(&d_out.raw).arg(&weight).arg(&(out_dim as u32)).launch(cfg) })?;
            accum_ms += t.stop("official_accum", stream.raw())?.elapsed_ms;
        }
    }

    // ── Shared expert (optional FP8) ──────────────────────────────────
    let mut shared_ms = 0.0f32;
    if let Some(shared) = shared_expert {
        let mut d_shared_gate = stream.alloc_zeros::<f32>(intermediate_size)?;
        let mut d_shared_up = stream.alloc_zeros::<f32>(intermediate_size)?;
        let mut d_shared_act = stream.alloc_zeros::<f32>(intermediate_size)?;
        let mut d_shared_down = stream.alloc_zeros::<f32>(out_dim)?;

        let ms = cuda_fp8_act_fp8_weight_gemv_device(quant, stream,
            &d_act.values, &d_act.scales,
            &shared.gate_weight, &shared.gate_scale,
            &mut d_shared_gate, intermediate_size, hidden_size)?;
        shared_ms += ms;

        let ms = cuda_fp8_act_fp8_weight_gemv_device(quant, stream,
            &d_act.values, &d_act.scales,
            &shared.up_weight, &shared.up_scale,
            &mut d_shared_up, intermediate_size, hidden_size)?;
        shared_ms += ms;

        // SwiGLU with clamp
        moe_executor.compile()?;
        if let Some(moe_mod) = &*moe_executor.module.lock().map_err(|e| CudaError::new(CudaErrorKind::Internal, "lock moe shared", e.to_string(), file!(), line!(), module_path!()))? {
            let t = CudaEventTimer::start(stream.raw())?;
            let cfg = LaunchConfig { grid_dim: ((intermediate_size as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            cuda_map_err!(CudaErrorKind::Driver, "launch silu_mul shared",
                unsafe { stream.raw().launch_builder(&moe_mod.silu_mul)
                    .arg(&d_shared_gate.raw).arg(&d_shared_up.raw).arg(&d_shared_act.raw)
                    .arg(&(intermediate_size as u32)).arg(&10.0f32).launch(cfg) })?;
            shared_ms += t.stop("shared_silu", stream.raw())?.elapsed_ms;
        }

        let (d_shared_act_q, aq_ms) = cuda_act_quant_device(quant, stream, &d_shared_act, 1, intermediate_size)?;
        shared_ms += aq_ms;

        let ms = cuda_fp8_act_fp8_weight_gemv_device(quant, stream,
            &d_shared_act_q.values, &d_shared_act_q.scales,
            &shared.down_weight, &shared.down_scale,
            &mut d_shared_down, out_dim, intermediate_size)?;
        shared_ms += ms;

        // Add shared output to final output (weight=1.0)
        if let Some(moe_mod) = &*moe_executor.module.lock().map_err(|e| CudaError::new(CudaErrorKind::Internal, "lock moe shared2", e.to_string(), file!(), line!(), module_path!()))? {
            let t = CudaEventTimer::start(stream.raw())?;
            let cfg = LaunchConfig { grid_dim: ((out_dim as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let one = 1.0f32;
            cuda_map_err!(CudaErrorKind::Driver, "launch weighted_accum shared",
                unsafe { stream.raw().launch_builder(&moe_mod.weighted_accum)
                    .arg(&d_shared_down.raw).arg(&d_out.raw).arg(&one).arg(&(out_dim as u32)).launch(cfg) })?;
            shared_ms += t.stop("shared_accum", stream.raw())?.elapsed_ms;
        }
    }

    // Download output
    let d2h_start = CudaEventTimer::start(stream.raw())?;
    let output = stream.copy_to_vec(&d_out)?;
    let d2h_ms = d2h_start.stop("official_d2h", stream.raw())?.elapsed_ms;

    let total_ms = total_timer.stop("official_total", stream.raw())?.elapsed_ms;

    let cache_counters = if let Some(ref c) = cache {
        (c.hit_count, c.miss_count, c.eviction_count)
    } else { (0, 0, 0) };

    // Compute shared expert FP8 byte counts
    let shared_fp8_weight_bytes = if shared_expert.is_some() {
        2 * (intermediate_size * hidden_size) + out_dim * intermediate_size
    } else { 0 };
    let shared_fp8_scale_bytes = if shared_expert.is_some() {
        2 * (intermediate_size * hidden_size / 128) + out_dim * intermediate_size / 128
    } else { 0 };
    let shared_fp8_total = shared_fp8_weight_bytes + shared_fp8_scale_bytes;

    let routed_fp4_loaded = weight_bytes_loaded + scale_bytes_loaded;
    let routed_fp4_reused = weight_bytes_reused + scale_bytes_reused;
    let total_logical = selected_working_set_bytes + shared_fp8_total;
    let total_loaded = routed_fp4_loaded + shared_fp8_total; // shared always loaded (no cache)
    let total_reused = routed_fp4_reused;

    Ok((output, MoeTelemetry {
        selected_expert_count: selected_experts.len(),
        logical_expert_bytes_requested: selected_working_set_bytes,
        actual_expert_bytes_loaded: routed_fp4_loaded,
        resident_cache_bytes_reused: routed_fp4_reused,
        weight_bytes_loaded, weight_bytes_reused,
        scale_bytes_loaded, scale_bytes_reused,
        resident_cache_resident_bytes: cache.as_ref().map_or(0, |c| c.resident_bytes),
        resident_cache_capacity_bytes: cache.as_ref().map_or(0, |c| c.capacity_bytes),
        resident_cache_hit_count: cache_counters.0,
        resident_cache_miss_count: cache_counters.1,
        resident_cache_eviction_count: cache_counters.2,
        dequantized_scratch_bytes: 0,
        h2d_ms,
        gate_up_qgemv_ms: gate_up_ms,
        activation_ms: act_ms,
        down_qgemv_ms: down_ms,
        accum_ms,
        unaccounted_ms: 0.0,
        total_ms: total_ms - d2h_ms,
        shared_expert_ms: shared_ms,
        bytes_by_tensor_kind: Default::default(),
        bytes_per_expert: single_expert_bytes,
        selected_working_set_bytes,
        routed_fp4_bytes_loaded: routed_fp4_loaded,
        routed_fp4_bytes_reused: routed_fp4_reused,
        shared_fp8_bytes_loaded: shared_fp8_total,
        shared_fp8_bytes_reused: 0,
        total_logical_bytes: total_logical,
        total_loaded_bytes: total_loaded,
        total_reused_bytes: total_reused,
        ..Default::default()
    }))
}

pub fn selected_moe_cpu_native_fp4(
    experts: &[DeepSeekFp4ExpertWeights],
    selected_experts: &[(usize, f32)],
    x: &[f32],
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; out_dim];
    let shape_gate_up = QGemvShape::new(QuantFormat::DeepSeekFp4E2M1, intermediate, hidden);
    let shape_down = QGemvShape::new(QuantFormat::DeepSeekFp4E2M1, out_dim, intermediate);

    for &(e, weight) in selected_experts {
        if e >= experts.len() {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate expert index CPU native fp4",
                format!("expert index {} out of bounds (len={})", e, experts.len()),
                file!(),
                line!(),
                module_path!(),
            ));
        }
        let expert = &experts[e];

        let gate_phys = [intermediate, hidden / 2];
        let gate_log = [intermediate, hidden];
        let gate_f32 = objeta_parser::deepseek::decode_deepseek_fp4_to_f32(
            &expert.gate_weight,
            &expert.gate_scale,
            &gate_phys,
            &gate_log,
            32,
        );
        let gate = dense_gemv_cpu(&gate_f32, x, shape_gate_up)?;

        let up_phys = [intermediate, hidden / 2];
        let up_log = [intermediate, hidden];
        let up_f32 = objeta_parser::deepseek::decode_deepseek_fp4_to_f32(
            &expert.up_weight,
            &expert.up_scale,
            &up_phys,
            &up_log,
            32,
        );
        let up = dense_gemv_cpu(&up_f32, x, shape_gate_up)?;

        let mut act = vec![0.0f32; intermediate];
        for i in 0..intermediate {
            let g = gate[i];
            let silu = g / (1.0f32 + (-g).exp());
            act[i] = silu * up[i];
        }

        let down_phys = [out_dim, intermediate / 2];
        let down_log = [out_dim, intermediate];
        let down_f32 = objeta_parser::deepseek::decode_deepseek_fp4_to_f32(
            &expert.down_weight,
            &expert.down_scale,
            &down_phys,
            &down_log,
            32,
        );
        let down = dense_gemv_cpu(&down_f32, &act, shape_down)?;

        for i in 0..out_dim {
            out[i] += weight * down[i];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CudaBackendBuilder;
    use crate::quant::{q4_quantize_matrix_cpu, fp4_quantize_matrix_cpu, compare_outputs};

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

    fn run_moe_test_case(
        hidden: usize,
        intermediate: usize,
        out_dim: usize,
        num_selected: usize,
        seed: u64,
    ) -> Result<()> {
        let backend = CudaBackendBuilder::new().stream_count(1).build()?;
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0)?;

        let num_experts = 8;
        let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate, hidden);
        let shape_down = QGemvShape::new(QuantFormat::Q4_0, out_dim, intermediate);

        let mut experts = Vec::with_capacity(num_experts);
        let mut experts_fp32 = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w_gate_raw = seeded_f32s(intermediate * hidden, seed ^ (e as u64) ^ 0x1111);
            let w_up_raw = seeded_f32s(intermediate * hidden, seed ^ (e as u64) ^ 0x2222);
            let w_down_raw = seeded_f32s(out_dim * intermediate, seed ^ (e as u64) ^ 0x3333);

            let w_gate = q4_quantize_matrix_cpu(&w_gate_raw, shape_gate_up)?;
            let w_up = q4_quantize_matrix_cpu(&w_up_raw, shape_gate_up)?;
            let w_down = q4_quantize_matrix_cpu(&w_down_raw, shape_down)?;

            experts.push(ExpertWeights { w_gate, w_up, w_down });
            experts_fp32.push(ExpertWeightsFp32 {
                w_gate: w_gate_raw,
                w_up: w_up_raw,
                w_down: w_down_raw,
            });
        }

        let x = seeded_f32s(hidden, seed ^ 0xDEADBEEF);

        // Deterministic selection based on seed
        let mut selected_experts = Vec::new();
        let weights = vec![0.4f32, 0.3f32, 0.2f32, 0.1f32];
        for idx in 0..num_selected {
            let expert_idx = ((seed + idx as u64) % num_experts as u64) as usize;
            let weight = weights[idx % weights.len()];
            selected_experts.push((expert_idx, weight));
        }

        // Run CPU reference (quantized)
        let ref_out = selected_moe_cpu(&experts, &selected_experts, &x, hidden, intermediate, out_dim)?;

        // Run CPU reference (fp32)
        let ref_fp32 = selected_moe_cpu_fp32(&experts_fp32, &selected_experts, &x, hidden, intermediate, out_dim)?;

        // Run CUDA executor
        let (cuda_out, telemetry) = moe_executor.execute_selected_moe_cuda(
            &quant,
            stream,
            &experts,
            &selected_experts,
            &x,
            hidden,
            intermediate,
            out_dim,
            0,
            None,
        )?;

        // Compare outputs
        let diff = compare_outputs(&ref_out, &cuda_out)?;
        println!(
            "MoE Test (Quant-vs-Quant): hidden={}, intermediate={}, out={}, selected={}. Seed={}. Cosine Sim: {:.6}, L2 Err: {:.6}, Max Abs Err: {:.6}",
            hidden, intermediate, out_dim, num_selected, seed, diff.cosine_similarity, diff.relative_l2_error, diff.max_abs_error
        );

        assert!(
            diff.cosine_similarity >= 0.9999,
            "Cosine similarity too low: {:.6}",
            diff.cosine_similarity
        );
        assert!(
            diff.relative_l2_error <= 1.0e-4,
            "Relative L2 error too high: {:.6}",
            diff.relative_l2_error
        );
        assert!(
            diff.max_abs_error <= 1.0e-2,
            "Max absolute error too high: {:.6}",
            diff.max_abs_error
        );

        // Compare outputs (quant vs fp32)
        let diff_fp32 = compare_outputs(&ref_fp32, &cuda_out)?;
        println!(
            "MoE Test (Quant-vs-Fp32): Cosine Sim: {:.6}, L2 Err: {:.6}, Max Abs Err: {:.6}",
            diff_fp32.cosine_similarity, diff_fp32.relative_l2_error, diff_fp32.max_abs_error
        );

        assert!(
            diff_fp32.cosine_similarity >= 0.98,
            "Quant vs Fp32 Cosine similarity too low: {:.6}",
            diff_fp32.cosine_similarity
        );

        // Verify telemetry requirements
        assert_eq!(telemetry.selected_expert_count, num_selected);
        assert_eq!(telemetry.dequantized_scratch_bytes, 0);
        assert_eq!(telemetry.resident_cache_bytes_reused, 0);
        assert_eq!(
            telemetry.logical_expert_bytes_requested,
            telemetry.actual_expert_bytes_loaded
        );
        assert_eq!(
            telemetry.logical_expert_bytes_requested,
            telemetry.actual_expert_bytes_loaded + telemetry.resident_cache_bytes_reused
        );

        Ok(())
    }

    #[test]
    fn test_moe_correctness_sweeps() -> Result<()> {
        let seeds = [0, 1, 2, 3, 4, 123];

        for &seed in &seeds {
            // Case 1: hidden=256, intermediate=512, out=256, selected_experts=1
            run_moe_test_case(256, 512, 256, 1, seed)?;

            // Case 2: hidden=256, intermediate=512, out=256, selected_experts=2
            run_moe_test_case(256, 512, 256, 2, seed)?;

            // Case 3: hidden=1024, intermediate=2048, out=1024, selected_experts=4
            run_moe_test_case(1024, 2048, 1024, 4, seed)?;
        }

        Ok(())
    }

    #[test]
    fn test_cuda_expert_cache_behavior() -> Result<()> {
        let backend = CudaBackendBuilder::new().stream_count(1).build()?;
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0)?;

        let hidden = 256;
        let intermediate = 512;
        let out_dim = 256;
        let num_experts = 4;
        let seed = 42;

        let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate, hidden);
        let shape_down = QGemvShape::new(QuantFormat::Q4_0, out_dim, intermediate);

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w_gate_raw = seeded_f32s(intermediate * hidden, seed ^ (e as u64) ^ 0x1111);
            let w_up_raw = seeded_f32s(intermediate * hidden, seed ^ (e as u64) ^ 0x2222);
            let w_down_raw = seeded_f32s(out_dim * intermediate, seed ^ (e as u64) ^ 0x3333);

            let w_gate = q4_quantize_matrix_cpu(&w_gate_raw, shape_gate_up)?;
            let w_up = q4_quantize_matrix_cpu(&w_up_raw, shape_gate_up)?;
            let w_down = q4_quantize_matrix_cpu(&w_down_raw, shape_down)?;

            experts.push(ExpertWeights { w_gate, w_up, w_down });
        }

        let x = seeded_f32s(hidden, seed ^ 0xDEADBEEF);
        
        let expert_bytes = experts[0].w_gate.len() + experts[0].w_up.len() + experts[0].w_down.len();

        // 1. Cache disabled: loaded == logical, reused == 0
        {
            let selected_experts = vec![(0, 0.5f32), (1, 0.5f32)];
            let (_cuda_out, telemetry) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &selected_experts,
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                None,
            )?;
            assert_eq!(telemetry.resident_cache_bytes_reused, 0);
            assert_eq!(telemetry.actual_expert_bytes_loaded, telemetry.logical_expert_bytes_requested);
            assert_eq!(telemetry.resident_cache_capacity_bytes, 0);
            assert_eq!(telemetry.resident_cache_resident_bytes, 0);
            assert_eq!(
                telemetry.logical_expert_bytes_requested,
                telemetry.actual_expert_bytes_loaded + telemetry.resident_cache_bytes_reused
            );
        }

        // 2. Cache enabled, same experts repeated: second call has reused > 0
        {
            let mut cache = CudaExpertCache::new(1000 * 1024); // 1 MB capacity, fits multiple experts
            let selected_experts = vec![(0, 0.5f32), (1, 0.5f32)];
            
            // First run: Cache warm up (misses)
            let (_cuda_out1, telemetry1) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &selected_experts,
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            assert_eq!(telemetry1.resident_cache_bytes_reused, 0);
            assert_eq!(telemetry1.actual_expert_bytes_loaded, telemetry1.logical_expert_bytes_requested);
            assert_eq!(telemetry1.resident_cache_hit_count, 0);
            assert_eq!(telemetry1.resident_cache_miss_count, 6); // 2 experts * 3 tensors
            assert_eq!(telemetry1.resident_cache_eviction_count, 0);
            assert_eq!(
                telemetry1.logical_expert_bytes_requested,
                telemetry1.actual_expert_bytes_loaded + telemetry1.resident_cache_bytes_reused
            );

            // Second run: Cache hit
            let (_cuda_out2, telemetry2) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &selected_experts,
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            assert_eq!(telemetry2.resident_cache_bytes_reused, telemetry2.logical_expert_bytes_requested);
            assert_eq!(telemetry2.actual_expert_bytes_loaded, 0);
            assert_eq!(telemetry2.resident_cache_hit_count, 6); // 6 hits
            assert_eq!(telemetry2.resident_cache_miss_count, 6); // cumulative misses still 6
            assert_eq!(telemetry2.resident_cache_eviction_count, 0);
            assert_eq!(
                telemetry2.logical_expert_bytes_requested,
                telemetry2.actual_expert_bytes_loaded + telemetry2.resident_cache_bytes_reused
            );
        }

        // 3. Capacity fits all tensors: no eviction after warm run
        {
            let mut cache = CudaExpertCache::new(4 * expert_bytes); // Fits 4 experts easily
            let selected_experts = vec![(0, 0.5f32), (1, 0.5f32), (2, 0.5f32), (3, 0.5f32)];
            let (_cuda_out, telemetry) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &selected_experts,
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            assert_eq!(telemetry.resident_cache_eviction_count, 0);
            assert_eq!(telemetry.resident_cache_resident_bytes, 4 * expert_bytes);
            assert_eq!(
                telemetry.logical_expert_bytes_requested,
                telemetry.actual_expert_bytes_loaded + telemetry.resident_cache_bytes_reused
            );
        }

        // 4. Capacity fits only one expert: eviction occurs
        {
            let mut cache = CudaExpertCache::new(expert_bytes); 
            
            // First run: execute with expert 0 (misses, fits)
            let (_cuda_out1, telemetry1) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(0, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            assert_eq!(telemetry1.resident_cache_eviction_count, 0);
            assert_eq!(telemetry1.resident_cache_resident_bytes, expert_bytes);
            
            // Second run: execute with expert 1 (misses, evicts expert 0)
            let (_cuda_out2, telemetry2) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(1, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            assert!(telemetry2.resident_cache_eviction_count > 0);
            assert_eq!(telemetry2.resident_cache_resident_bytes, expert_bytes);
            
            // Verify expert 0 is evicted
            let (_cuda_out3, telemetry3) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(0, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            assert_eq!(telemetry3.actual_expert_bytes_loaded, expert_bytes);
            assert_eq!(telemetry3.resident_cache_bytes_reused, 0);
        }

        // 5. LRU touch updates eviction order
        {
            let mut cache = CudaExpertCache::new(2 * expert_bytes);
            
            let _ = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(0, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            let _ = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(1, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            
            // Touch expert 0
            let _ = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(0, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            
            // expert 1 should be evicted when expert 2 is loaded
            let _ = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(2, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            
            // Access expert 0 again: hit
            let (_, tel_0) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(0, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            assert_eq!(tel_0.resident_cache_bytes_reused, expert_bytes);
            
            // Access expert 1 again: miss
            let (_, tel_1) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &experts,
                &[(1, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;
            assert_eq!(tel_1.resident_cache_bytes_reused, 0);
            assert_eq!(tel_1.actual_expert_bytes_loaded, expert_bytes);
        }

        Ok(())
    }

    #[test]
    fn test_oversized_tensor_bypass() -> Result<()> {
        let backend = CudaBackendBuilder::new().stream_count(1).build()?;
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0)?;

        let hidden = 256;
        let intermediate = 512;
        let out_dim = 256;
        let seed = 42;

        let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate, hidden);
        let shape_down = QGemvShape::new(QuantFormat::Q4_0, out_dim, intermediate);

        let w_gate_raw = seeded_f32s(intermediate * hidden, seed ^ 0x1111);
        let w_up_raw = seeded_f32s(intermediate * hidden, seed ^ 0x2222);
        let w_down_raw = seeded_f32s(out_dim * intermediate, seed ^ 0x3333);

        let w_gate = q4_quantize_matrix_cpu(&w_gate_raw, shape_gate_up)?;
        let w_up = q4_quantize_matrix_cpu(&w_up_raw, shape_gate_up)?;
        let w_down = q4_quantize_matrix_cpu(&w_down_raw, shape_down)?;

        let single_expert = ExpertWeights { w_gate, w_up, w_down };
        let gate_len = single_expert.w_gate.len();

        let x = seeded_f32s(hidden, seed ^ 0xDEADBEEF);

        // Capacity is smaller than gate_len
        // So gate tensor is oversized and should be bypassed.
        let mut cache = CudaExpertCache::new(gate_len - 1);

        let (_, telemetry) = moe_executor.execute_selected_moe_cuda(
            &quant,
            stream,
            &[single_expert],
            &[(0, 1.0f32)],
            &x,
            hidden,
            intermediate,
            out_dim,
            0,
            Some(&mut cache),
        )?;

        // Gate, Up, Down should all be bypassed since gate_len - 1 is smaller than each tensor size.
        assert_eq!(telemetry.resident_cache_insert_attempt_count, 3);
        assert_eq!(telemetry.resident_cache_insert_accept_count, 0);
        assert_eq!(telemetry.resident_cache_insert_bypass_count, 3);
        assert_eq!(telemetry.resident_cache_oversized_tensor_bypass_count, 3);
        assert_eq!(telemetry.resident_cache_oversized_expert_bypass_count, 0);
        assert_eq!(telemetry.resident_cache_resident_bytes, 0);
        assert_eq!(telemetry.actual_expert_bytes_loaded, telemetry.logical_expert_bytes_requested);
        assert_eq!(telemetry.resident_cache_bytes_reused, 0);

        Ok(())
    }

    #[test]
    fn test_oversized_expert_bypass() -> Result<()> {
        let backend = CudaBackendBuilder::new().stream_count(1).build()?;
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0)?;

        let hidden = 256;
        let intermediate = 512;
        let out_dim = 256;
        let seed = 42;

        let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate, hidden);
        let shape_down = QGemvShape::new(QuantFormat::Q4_0, out_dim, intermediate);

        let w_gate_raw = seeded_f32s(intermediate * hidden, seed ^ 0x1111);
        let w_up_raw = seeded_f32s(intermediate * hidden, seed ^ 0x2222);
        let w_down_raw = seeded_f32s(out_dim * intermediate, seed ^ 0x3333);

        let w_gate = q4_quantize_matrix_cpu(&w_gate_raw, shape_gate_up)?;
        let w_up = q4_quantize_matrix_cpu(&w_up_raw, shape_gate_up)?;
        let w_down = q4_quantize_matrix_cpu(&w_down_raw, shape_down)?;

        let single_expert = ExpertWeights { w_gate, w_up, w_down };
        let expert_bytes = single_expert.w_gate.len() + single_expert.w_up.len() + single_expert.w_down.len();

        let x = seeded_f32s(hidden, seed ^ 0xDEADBEEF);

        // Case A: bypass_oversized_experts = true
        // Capacity is larger than single tensor but smaller than whole expert
        {
            let mut cache = CudaExpertCache::new(expert_bytes - 100);
            cache.bypass_oversized_experts = true;

            let (_, telemetry) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &[single_expert.clone()],
                &[(0, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;

            assert_eq!(telemetry.resident_cache_insert_attempt_count, 3);
            assert_eq!(telemetry.resident_cache_insert_accept_count, 0);
            assert_eq!(telemetry.resident_cache_insert_bypass_count, 3);
            assert_eq!(telemetry.resident_cache_oversized_tensor_bypass_count, 0);
            assert_eq!(telemetry.resident_cache_oversized_expert_bypass_count, 1);
            assert_eq!(telemetry.resident_cache_self_eviction_risk_count, 0);
            assert_eq!(telemetry.resident_cache_resident_bytes, 0);
            assert_eq!(telemetry.actual_expert_bytes_loaded, telemetry.logical_expert_bytes_requested);
            assert_eq!(telemetry.resident_cache_bytes_reused, 0);
        }

        // Case B: bypass_oversized_experts = false
        // Capacity is larger than single tensor but smaller than whole expert
        // Should flag self_eviction_risk_count
        {
            let mut cache = CudaExpertCache::new(expert_bytes - 100);
            cache.bypass_oversized_experts = false;

            let (_, telemetry) = moe_executor.execute_selected_moe_cuda(
                &quant,
                stream,
                &[single_expert.clone()],
                &[(0, 1.0f32)],
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                Some(&mut cache),
            )?;

            assert_eq!(telemetry.resident_cache_self_eviction_risk_count, 1);
            assert_eq!(telemetry.resident_cache_oversized_expert_bypass_count, 0);
            assert!(telemetry.resident_cache_insert_accept_count > 0);
            assert_eq!(
                telemetry.logical_expert_bytes_requested,
                telemetry.actual_expert_bytes_loaded + telemetry.resident_cache_bytes_reused
            );
        }

        Ok(())
    }

    fn split_interleaved_fp4(interleaved: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
        let blocks_per_row = cols / 32;
        let mut weights = Vec::with_capacity(rows * blocks_per_row * 16);
        let mut scales = Vec::with_capacity(rows * blocks_per_row);
        for r in 0..rows {
            let row_start = r * blocks_per_row * 17;
            for b in 0..blocks_per_row {
                let block_start = row_start + b * 17;
                scales.push(interleaved[block_start]);
                weights.extend_from_slice(&interleaved[block_start + 1..block_start + 17]);
            }
        }
        (weights, scales)
    }

    #[test]
    fn test_native_fp4_moe_correctness() -> Result<()> {
        let backend = CudaBackendBuilder::new().stream_count(1).build()?;
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0)?;

        let hidden = 256;
        let intermediate = 512;
        let out_dim = 256;
        let num_experts = 4;
        let seed = 42;

        let shape_gate_up = QGemvShape::new(QuantFormat::DeepSeekFp4E2M1, intermediate, hidden);
        let shape_down = QGemvShape::new(QuantFormat::DeepSeekFp4E2M1, out_dim, intermediate);

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w_gate_raw = seeded_f32s(intermediate * hidden, seed ^ (e as u64) ^ 0x1111);
            let w_up_raw = seeded_f32s(intermediate * hidden, seed ^ (e as u64) ^ 0x2222);
            let w_down_raw = seeded_f32s(out_dim * intermediate, seed ^ (e as u64) ^ 0x3333);

            let w_gate_interleaved = fp4_quantize_matrix_cpu(&w_gate_raw, shape_gate_up)?;
            let w_up_interleaved = fp4_quantize_matrix_cpu(&w_up_raw, shape_gate_up)?;
            let w_down_interleaved = fp4_quantize_matrix_cpu(&w_down_raw, shape_down)?;

            let (gate_weight, gate_scale) = split_interleaved_fp4(&w_gate_interleaved, intermediate, hidden);
            let (up_weight, up_scale) = split_interleaved_fp4(&w_up_interleaved, intermediate, hidden);
            let (down_weight, down_scale) = split_interleaved_fp4(&w_down_interleaved, out_dim, intermediate);

            experts.push(DeepSeekFp4ExpertWeights {
                gate_weight,
                gate_scale,
                up_weight,
                up_scale,
                down_weight,
                down_scale,
            });
        }

        let x = seeded_f32s(hidden, seed ^ 0xDEADBEEF);
        let selected_experts = vec![(0, 0.4f32), (2, 0.6f32)];

        // Run CPU reference (native fp4)
        let ref_out = selected_moe_cpu_native_fp4(&experts, &selected_experts, &x, hidden, intermediate, out_dim)?;

        // Run CUDA executor (native fp4)
        quant.compile_format(QuantFormat::DeepSeekFp4E2M1)?;
        moe_executor.compile()?;

        let (cuda_out, telemetry) = moe_executor.execute_selected_moe_native_fp4_cuda(
            &quant,
            stream,
            &experts,
            &selected_experts,
            &x,
            hidden,
            intermediate,
            out_dim,
            0,
            None,
        )?;

        // Compare outputs
        let diff = compare_outputs(&ref_out, &cuda_out)?;
        println!(
            "FP4 MoE Test: Cosine Sim: {:.6}, L2 Err: {:.6}, Max Abs Err: {:.6}",
            diff.cosine_similarity, diff.relative_l2_error, diff.max_abs_error
        );

        assert!(
            diff.cosine_similarity >= 0.9999,
            "FP4 Cosine similarity too low: {:.6}",
            diff.cosine_similarity
        );

        // Verify telemetry
        assert_eq!(telemetry.selected_expert_count, 2);
        assert_eq!(telemetry.dequantized_scratch_bytes, 0);
        assert_eq!(telemetry.resident_cache_bytes_reused, 0);
        assert_eq!(
            telemetry.logical_expert_bytes_requested,
            telemetry.actual_expert_bytes_loaded
        );

        Ok(())
    }

    #[test]
    fn test_native_fp4_expert_cache() -> Result<()> {
        let backend = CudaBackendBuilder::new().stream_count(1).build()?;
        let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
        let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
        let stream = backend.stream_pool().stream(0)?;

        let hidden = 256;
        let intermediate = 512;
        let out_dim = 256;
        let num_experts = 4;
        let seed = 42;

        let shape_gate_up = QGemvShape::new(QuantFormat::DeepSeekFp4E2M1, intermediate, hidden);
        let shape_down = QGemvShape::new(QuantFormat::DeepSeekFp4E2M1, out_dim, intermediate);

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w_gate_raw = seeded_f32s(intermediate * hidden, seed ^ (e as u64) ^ 0x1111);
            let w_up_raw = seeded_f32s(intermediate * hidden, seed ^ (e as u64) ^ 0x2222);
            let w_down_raw = seeded_f32s(out_dim * intermediate, seed ^ (e as u64) ^ 0x3333);

            let w_gate_interleaved = fp4_quantize_matrix_cpu(&w_gate_raw, shape_gate_up)?;
            let w_up_interleaved = fp4_quantize_matrix_cpu(&w_up_raw, shape_gate_up)?;
            let w_down_interleaved = fp4_quantize_matrix_cpu(&w_down_raw, shape_down)?;

            let (gate_weight, gate_scale) = split_interleaved_fp4(&w_gate_interleaved, intermediate, hidden);
            let (up_weight, up_scale) = split_interleaved_fp4(&w_up_interleaved, intermediate, hidden);
            let (down_weight, down_scale) = split_interleaved_fp4(&w_down_interleaved, out_dim, intermediate);

            experts.push(DeepSeekFp4ExpertWeights {
                gate_weight,
                gate_scale,
                up_weight,
                up_scale,
                down_weight,
                down_scale,
            });
        }

        let x = seeded_f32s(hidden, seed ^ 0xDEADBEEF);
        quant.compile_format(QuantFormat::DeepSeekFp4E2M1)?;
        moe_executor.compile()?;

        let expert_bytes = (intermediate * hidden / 2) + (intermediate * hidden / 32)
            + (intermediate * hidden / 2) + (intermediate * hidden / 32)
            + (out_dim * intermediate / 2) + (out_dim * intermediate / 32);

        assert_eq!(expert_bytes, 208896);

        // Capacity covers exactly 2 experts
        let mut cache = CudaExpertCache::new(expert_bytes * 2);

        // First run: expert 0 selected. Miss.
        let (_, tel_0) = moe_executor.execute_selected_moe_native_fp4_cuda(
            &quant,
            stream,
            &experts,
            &[(0, 1.0f32)],
            &x,
            hidden,
            intermediate,
            out_dim,
            0,
            Some(&mut cache),
        )?;
        assert_eq!(tel_0.resident_cache_bytes_reused, 0);
        assert_eq!(tel_0.actual_expert_bytes_loaded, expert_bytes);
        assert_eq!(cache.resident_bytes, expert_bytes);

        // Second run: expert 0 selected. Hit.
        let (_, tel_1) = moe_executor.execute_selected_moe_native_fp4_cuda(
            &quant,
            stream,
            &experts,
            &[(0, 1.0f32)],
            &x,
            hidden,
            intermediate,
            out_dim,
            0,
            Some(&mut cache),
        )?;
        assert_eq!(tel_1.resident_cache_bytes_reused, expert_bytes);
        assert_eq!(tel_1.actual_expert_bytes_loaded, 0);
        assert_eq!(cache.resident_bytes, expert_bytes);

        // Third run: expert 1 selected. Miss. Resident size grows to 2 * expert_bytes
        let (_, tel_2) = moe_executor.execute_selected_moe_native_fp4_cuda(
            &quant,
            stream,
            &experts,
            &[(1, 1.0f32)],
            &x,
            hidden,
            intermediate,
            out_dim,
            0,
            Some(&mut cache),
        )?;
        assert_eq!(tel_2.resident_cache_bytes_reused, 0);
        assert_eq!(tel_2.actual_expert_bytes_loaded, expert_bytes);
        assert_eq!(cache.resident_bytes, expert_bytes * 2);

        // Fourth run: expert 2 selected. Evicts expert 0 (since capacity is expert_bytes * 2)
        let (_, tel_3) = moe_executor.execute_selected_moe_native_fp4_cuda(
            &quant,
            stream,
            &experts,
            &[(2, 1.0f32)],
            &x,
            hidden,
            intermediate,
            out_dim,
            0,
            Some(&mut cache),
        )?;
        assert_eq!(tel_3.resident_cache_bytes_reused, 0);
        assert_eq!(tel_3.actual_expert_bytes_loaded, expert_bytes);
        assert_eq!(cache.resident_bytes, expert_bytes * 2);

        // Access expert 0 again: should miss (since it was evicted)
        let (_, tel_4) = moe_executor.execute_selected_moe_native_fp4_cuda(
            &quant,
            stream,
            &experts,
            &[(0, 1.0f32)],
            &x,
            hidden,
            intermediate,
            out_dim,
            0,
            Some(&mut cache),
        )?;
        assert_eq!(tel_4.resident_cache_bytes_reused, 0);
        assert_eq!(tel_4.actual_expert_bytes_loaded, expert_bytes);

        Ok(())
    }
}

