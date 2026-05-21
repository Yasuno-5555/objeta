use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{CudaContext, CudaModule, CudaFunction, LaunchConfig, PushKernelArg};

use crate::{cuda_map_err, CudaError, CudaErrorKind, Result, DeviceBuffer};
use crate::context::CudaDeviceInfo;
use crate::quant::{QuantBackend, QuantFormat, QGemvShape, gemv_cpu, dense_gemv_cpu};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExpertTensorKind {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExpertCacheKey {
    pub layer_id: usize,
    pub expert_id: usize,
    pub tensor_kind: ExpertTensorKind,
    pub quant_format: QuantFormat,
}

#[derive(Debug)]
pub struct CudaExpertCache {
    pub capacity_bytes: usize,
    pub resident_bytes: usize,
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
    map: std::collections::HashMap<ExpertCacheKey, (Arc<DeviceBuffer<u8>>, usize)>,
    order: Vec<ExpertCacheKey>,
}

impl CudaExpertCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
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

    pub fn evict_lru(&mut self) -> Option<(ExpertCacheKey, Arc<DeviceBuffer<u8>>)> {
        if self.order.is_empty() {
            return None;
        }
        let lru_key = self.order.remove(0);
        if let Some((buf, size)) = self.map.remove(&lru_key) {
            self.resident_bytes -= size;
            self.eviction_count += 1;
            Some((lru_key, buf))
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.resident_bytes = 0;
    }

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
    pub bytes_by_tensor_kind: BytesByTensorKind,
    pub bytes_per_expert: usize,
    pub selected_working_set_bytes: usize,
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
            bytes_by_tensor_kind: BytesByTensorKind {
                gate: gate_bytes,
                up: up_bytes,
                down: down_bytes,
            },
            bytes_per_expert: single_expert_bytes,
            selected_working_set_bytes,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CudaBackendBuilder;
    use crate::quant::{q4_quantize_matrix_cpu, compare_outputs};

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
}
