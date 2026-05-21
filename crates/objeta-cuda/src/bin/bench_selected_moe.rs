use objeta_cuda::{
    compare_outputs, q4_quantize_matrix_cpu, selected_moe_cpu, selected_moe_cpu_fp32,
    CudaBackendBuilder, ExpertWeights, ExpertWeightsFp32, MoeExecutor, MoeTelemetry,
    QGemvNumericsSuite, QGemvShape, QuantBackend, QuantFormat, Result, CudaExpertCache,
    BytesByTensorKind, CudaError, CudaErrorKind,
};
use std::path::Path;

const SYNTHETIC_DEFAULT_SOURCE: &str = "synthetic";
const SYNTHETIC_FROM_SANITY_REPORT_SOURCE: &str = "synthetic_from_deepseek_sanity_report";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SanityReportShape {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub top_k: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoeStats {
    pub avg: f32,
    pub min: f32,
    pub max: f32,
    pub p50: f32,
    pub p95: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoeStatsTelemetry {
    pub h2d_ms: MoeStats,
    pub gate_up_qgemv_ms: MoeStats,
    pub activation_ms: MoeStats,
    pub down_qgemv_ms: MoeStats,
    pub accum_ms: MoeStats,
    pub unaccounted_ms: MoeStats,
    pub total_ms: MoeStats,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoeReport {
    pub source: &'static str,
    pub backend: &'static str,
    pub format: QuantFormat,
    pub hidden: usize,
    pub intermediate: usize,
    pub out: usize,
    pub selected_experts: usize,
    pub logical_expert_bytes_requested: usize,
    pub actual_expert_bytes_loaded: usize,
    pub resident_cache_bytes_reused: usize,
    pub dequantized_scratch_bytes: usize,
    pub resident_cache_capacity_bytes: usize,
    pub resident_cache_resident_bytes: usize,
    pub resident_cache_hit_count: usize,
    pub resident_cache_miss_count: usize,
    pub resident_cache_eviction_count: usize,
    pub cache_insert_attempt_count: usize,
    pub cache_insert_accept_count: usize,
    pub cache_insert_bypass_count: usize,
    pub oversized_tensor_bypass_count: usize,
    pub oversized_expert_bypass_count: usize,
    pub self_eviction_risk_count: usize,
    pub bytes_by_tensor_kind: BytesByTensorKind,
    pub bytes_per_expert: usize,
    pub selected_working_set_bytes: usize,
    pub cache_hit_rate: f32,
    pub telemetry: MoeTelemetry,
    pub numerics: QGemvNumericsSuite,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoeStatsReport {
    pub source: &'static str,
    pub backend: &'static str,
    pub format: QuantFormat,
    pub hidden: usize,
    pub intermediate: usize,
    pub out: usize,
    pub selected_experts: usize,
    pub logical_expert_bytes_requested: usize,
    pub actual_expert_bytes_loaded: usize,
    pub resident_cache_bytes_reused: usize,
    pub dequantized_scratch_bytes: usize,
    pub resident_cache_capacity_bytes: usize,
    pub resident_cache_resident_bytes: usize,
    pub resident_cache_hit_count: usize,
    pub resident_cache_miss_count: usize,
    pub resident_cache_eviction_count: usize,
    pub cache_insert_attempt_count: usize,
    pub cache_insert_accept_count: usize,
    pub cache_insert_bypass_count: usize,
    pub oversized_tensor_bypass_count: usize,
    pub oversized_expert_bypass_count: usize,
    pub self_eviction_risk_count: usize,
    pub bytes_by_tensor_kind: BytesByTensorKind,
    pub bytes_per_expert: usize,
    pub selected_working_set_bytes: usize,
    pub cache_hit_rate: f32,
    pub telemetry: MoeStatsTelemetry,
    pub numerics: QGemvNumericsSuite,
}

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

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_quant_format(args: &[String]) -> Result<QuantFormat> {
    let format = parse_flag(args, "--format").unwrap_or_else(|| "q4".to_string());
    match format.as_str() {
        "q4" | "q4_0" => Ok(QuantFormat::Q4_0),
        other => Err(CudaError::new(
            CudaErrorKind::Unsupported,
            "parse bench_selected_moe --format",
            format!(
                "unsupported format '{other}' for selected MoE benchmark; current selected MoE path is Q4-only"
            ),
            file!(),
            line!(),
            module_path!(),
        )),
    }
}

fn read_sanity_report_shape(path: &Path) -> Result<SanityReportShape> {
    let content = std::fs::read_to_string(path).map_err(|err| {
        CudaError::new(
            CudaErrorKind::Io,
            format!("read sanity report {}", path.display()),
            err.to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    let report: SanityReportShape = serde_json::from_str(&content).map_err(|err| {
        CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("parse sanity report {}", path.display()),
            err.to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    if report.hidden_size == 0 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate sanity report {}", path.display()),
            "hidden_size must be non-zero".to_string(),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    if report.intermediate_size == 0 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate sanity report {}", path.display()),
            "intermediate_size must be non-zero".to_string(),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    if report.top_k == 0 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("validate sanity report {}", path.display()),
            "top_k must be non-zero".to_string(),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    Ok(report)
}

fn percentile(sorted_values: &[f32], p: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted_values.len() - 1) as f32).round() as usize;
    sorted_values[idx]
}

fn compute_stats(mut values: Vec<f32>) -> MoeStats {
    if values.is_empty() {
        return MoeStats {
            avg: 0.0,
            min: 0.0,
            max: 0.0,
            p50: 0.0,
            p95: 0.0,
        };
    }
    let sum: f32 = values.iter().sum();
    let avg = sum / values.len() as f32;
    let min = *values
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);
    let max = *values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = percentile(&values, 0.50);
    let p95 = percentile(&values, 0.95);

    MoeStats {
        avg,
        min,
        max,
        p50,
        p95,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark_for_shape(
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
    num_selected: usize,
    seed: u64,
    warmup: usize,
    iters: usize,
    repeat: usize,
    cache_bytes: usize,
    bypass_oversized_experts: bool,
    quant: &QuantBackend,
    moe_executor: &MoeExecutor,
    stream: &objeta_cuda::CudaStreamHandle,
) -> Result<(MoeStatsTelemetry, QGemvNumericsSuite, MoeTelemetry)> {
    let num_experts = num_selected.max(8);
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

    let mut cache = if cache_bytes > 0 {
        let mut c = CudaExpertCache::new(cache_bytes);
        c.bypass_oversized_experts = bypass_oversized_experts;
        Some(c)
    } else {
        None
    };

    // Warmup iterations
    for _ in 0..warmup {
        for _ in 0..repeat {
            let _ = moe_executor.execute_selected_moe_cuda(
                quant,
                stream,
                &experts,
                &selected_experts,
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                cache.as_mut(),
            )?;
        }
    }

    if let Some(ref mut c) = cache {
        c.reset_counters();
    }

    // Timed iterations
    let mut h2d_vals = Vec::with_capacity(iters);
    let mut gate_up_qgemv_vals = Vec::with_capacity(iters);
    let mut activation_vals = Vec::with_capacity(iters);
    let mut down_qgemv_vals = Vec::with_capacity(iters);
    let mut accum_vals = Vec::with_capacity(iters);
    let mut unaccounted_vals = Vec::with_capacity(iters);
    let mut total_vals = Vec::with_capacity(iters);
    let mut actual = Vec::new();
    let mut final_telemetry = MoeTelemetry::default();

    for i in 0..iters {
        let mut h2d_ms = 0.0;
        let mut gate_up_qgemv_ms = 0.0;
        let mut activation_ms = 0.0;
        let mut down_qgemv_ms = 0.0;
        let mut accum_ms = 0.0;
        let mut unaccounted_ms = 0.0;
        let mut total_ms = 0.0;
        let mut logical_bytes = 0;
        let mut loaded_bytes = 0;
        let mut reused_bytes = 0;
        let mut final_y = Vec::new();
        let mut last_tel = MoeTelemetry::default();

        for _ in 0..repeat {
            let (y, tel) = moe_executor.execute_selected_moe_cuda(
                quant,
                stream,
                &experts,
                &selected_experts,
                &x,
                hidden,
                intermediate,
                out_dim,
                0,
                cache.as_mut(),
            )?;
            h2d_ms += tel.h2d_ms;
            gate_up_qgemv_ms += tel.gate_up_qgemv_ms;
            activation_ms += tel.activation_ms;
            down_qgemv_ms += tel.down_qgemv_ms;
            accum_ms += tel.accum_ms;
            unaccounted_ms += tel.unaccounted_ms;
            total_ms += tel.total_ms;
            logical_bytes += tel.logical_expert_bytes_requested;
            loaded_bytes += tel.actual_expert_bytes_loaded;
            reused_bytes += tel.resident_cache_bytes_reused;
            final_y = y;
            last_tel = tel;
        }

        h2d_vals.push(h2d_ms);
        gate_up_qgemv_vals.push(gate_up_qgemv_ms);
        activation_vals.push(activation_ms);
        down_qgemv_vals.push(down_qgemv_ms);
        accum_vals.push(accum_ms);
        unaccounted_vals.push(unaccounted_ms);
        total_vals.push(total_ms);

        if i == 0 {
            actual = final_y;
            final_telemetry = last_tel;
            final_telemetry.h2d_ms = h2d_ms;
            final_telemetry.gate_up_qgemv_ms = gate_up_qgemv_ms;
            final_telemetry.activation_ms = activation_ms;
            final_telemetry.down_qgemv_ms = down_qgemv_ms;
            final_telemetry.accum_ms = accum_ms;
            final_telemetry.unaccounted_ms = unaccounted_ms;
            final_telemetry.total_ms = total_ms;
            final_telemetry.logical_expert_bytes_requested = logical_bytes;
            final_telemetry.actual_expert_bytes_loaded = loaded_bytes;
            final_telemetry.resident_cache_bytes_reused = reused_bytes;
        }
    }

    let cuda_vs_cpu_quant = compare_outputs(&ref_out, &actual)?;
    let quant_vs_fp32 = compare_outputs(&ref_fp32, &actual)?;

    let numerics = QGemvNumericsSuite {
        cosine_similarity: cuda_vs_cpu_quant.cosine_similarity,
        relative_l2_error: cuda_vs_cpu_quant.relative_l2_error,
        max_abs_error: cuda_vs_cpu_quant.max_abs_error,
        cuda_vs_cpu_quant,
        quant_vs_fp32,
    };

    let h2d_stats = compute_stats(h2d_vals);
    let gate_up_qgemv_stats = compute_stats(gate_up_qgemv_vals);
    let activation_stats = compute_stats(activation_vals);
    let down_qgemv_stats = compute_stats(down_qgemv_vals);
    let accum_stats = compute_stats(accum_vals);
    let unaccounted_stats = compute_stats(unaccounted_vals);
    let total_stats = compute_stats(total_vals);

    let telemetry_stats = MoeStatsTelemetry {
        h2d_ms: h2d_stats,
        gate_up_qgemv_ms: gate_up_qgemv_stats,
        activation_ms: activation_stats,
        down_qgemv_ms: down_qgemv_stats,
        accum_ms: accum_stats,
        unaccounted_ms: unaccounted_stats,
        total_ms: total_stats,
    };

    // Assert invariant explicitly
    assert_eq!(
        final_telemetry.logical_expert_bytes_requested,
        final_telemetry.actual_expert_bytes_loaded + final_telemetry.resident_cache_bytes_reused,
        "MoE Byte invariant violated!"
    );

    Ok((telemetry_stats, numerics, final_telemetry))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let requested_matrix = has_flag(&args, "--matrix");
    let sanity_report_path = parse_flag(&args, "--from-sanity-report");
    let is_matrix = requested_matrix && sanity_report_path.is_none();
    let iters = parse_flag(&args, "--iters")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(if is_matrix { 10 } else { 1 });
    let warmup = parse_flag(&args, "--warmup")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(if is_matrix { 5 } else { 0 });

    let format = parse_quant_format(&args)?;
    let (source, hidden, intermediate, out, selected_experts) =
        if let Some(path) = sanity_report_path.as_ref() {
            let shape = read_sanity_report_shape(Path::new(path))?;
            (
                SYNTHETIC_FROM_SANITY_REPORT_SOURCE,
                shape.hidden_size,
                shape.intermediate_size,
                shape.hidden_size,
                shape.top_k,
            )
        } else {
            (
                SYNTHETIC_DEFAULT_SOURCE,
                parse_flag(&args, "--hidden")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(256),
                parse_flag(&args, "--intermediate")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(512),
                parse_flag(&args, "--out")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(256),
                parse_flag(&args, "--experts")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(2),
            )
        };
    let seed = parse_flag(&args, "--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(123);
    let cache_bytes = parse_flag(&args, "--cache-bytes")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let repeat = parse_flag(&args, "--repeat")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);

    let backend = CudaBackendBuilder::new().stream_count(1).build()?;
    let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
    let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
    let stream = backend.stream_pool().stream(0)?;

    // Compilation happens outside timed iterations
    quant.compile_format(format)?;
    moe_executor.compile()?;

    let bypass_oversized_experts = has_flag(&args, "--bypass-oversized-experts");

    if is_matrix {
        let matrix_shapes = &[
            // (hidden, intermediate, out, selected_experts)
            (256, 512, 256, 1),
            (256, 512, 256, 2),
            (1024, 2048, 1024, 4),
            (4096, 11008, 4096, 8),
        ];

        for &(h, i, o, se) in matrix_shapes {
            let (telemetry_stats, numerics, raw_telemetry) = run_benchmark_for_shape(
                h,
                i,
                o,
                se,
                seed,
                warmup,
                iters,
                repeat,
                cache_bytes,
                bypass_oversized_experts,
                &quant,
                &moe_executor,
                stream,
            )?;

            let total_cache_ops = raw_telemetry.resident_cache_hit_count + raw_telemetry.resident_cache_miss_count;
            let cache_hit_rate = if total_cache_ops > 0 {
                raw_telemetry.resident_cache_hit_count as f32 / total_cache_ops as f32
            } else {
                0.0
            };

            let report = MoeStatsReport {
                source,
                backend: "cuda",
                format,
                hidden: h,
                intermediate: i,
                out: o,
                selected_experts: se,
                logical_expert_bytes_requested: raw_telemetry.logical_expert_bytes_requested,
                actual_expert_bytes_loaded: raw_telemetry.actual_expert_bytes_loaded,
                resident_cache_bytes_reused: raw_telemetry.resident_cache_bytes_reused,
                dequantized_scratch_bytes: raw_telemetry.dequantized_scratch_bytes,
                resident_cache_capacity_bytes: raw_telemetry.resident_cache_capacity_bytes,
                resident_cache_resident_bytes: raw_telemetry.resident_cache_resident_bytes,
                resident_cache_hit_count: raw_telemetry.resident_cache_hit_count,
                resident_cache_miss_count: raw_telemetry.resident_cache_miss_count,
                resident_cache_eviction_count: raw_telemetry.resident_cache_eviction_count,
                cache_insert_attempt_count: raw_telemetry.resident_cache_insert_attempt_count,
                cache_insert_accept_count: raw_telemetry.resident_cache_insert_accept_count,
                cache_insert_bypass_count: raw_telemetry.resident_cache_insert_bypass_count,
                oversized_tensor_bypass_count: raw_telemetry.resident_cache_oversized_tensor_bypass_count,
                oversized_expert_bypass_count: raw_telemetry.resident_cache_oversized_expert_bypass_count,
                self_eviction_risk_count: raw_telemetry.resident_cache_self_eviction_risk_count,
                bytes_by_tensor_kind: raw_telemetry.bytes_by_tensor_kind,
                bytes_per_expert: raw_telemetry.bytes_per_expert,
                selected_working_set_bytes: raw_telemetry.selected_working_set_bytes,
                cache_hit_rate,
                telemetry: telemetry_stats,
                numerics,
            };
            println!("{}", serde_json::to_string(&report).unwrap());
        }
    } else {
        // Single shape run
        if iters == 1 && warmup == 0 && parse_flag(&args, "--iters").is_none() && parse_flag(&args, "--warmup").is_none() {
            // Run exactly once, print single-run MoeReport
            let (_, numerics, raw_telemetry) = run_benchmark_for_shape(
                hidden,
                intermediate,
                out,
                selected_experts,
                seed,
                0,
                1,
                repeat,
                cache_bytes,
                bypass_oversized_experts,
                &quant,
                &moe_executor,
                stream,
            )?;
            let total_cache_ops = raw_telemetry.resident_cache_hit_count + raw_telemetry.resident_cache_miss_count;
            let cache_hit_rate = if total_cache_ops > 0 {
                raw_telemetry.resident_cache_hit_count as f32 / total_cache_ops as f32
            } else {
                0.0
            };
            let report = MoeReport {
                source,
                backend: "cuda",
                format,
                hidden,
                intermediate,
                out,
                selected_experts,
                logical_expert_bytes_requested: raw_telemetry.logical_expert_bytes_requested,
                actual_expert_bytes_loaded: raw_telemetry.actual_expert_bytes_loaded,
                resident_cache_bytes_reused: raw_telemetry.resident_cache_bytes_reused,
                dequantized_scratch_bytes: raw_telemetry.dequantized_scratch_bytes,
                resident_cache_capacity_bytes: raw_telemetry.resident_cache_capacity_bytes,
                resident_cache_resident_bytes: raw_telemetry.resident_cache_resident_bytes,
                resident_cache_hit_count: raw_telemetry.resident_cache_hit_count,
                resident_cache_miss_count: raw_telemetry.resident_cache_miss_count,
                resident_cache_eviction_count: raw_telemetry.resident_cache_eviction_count,
                cache_insert_attempt_count: raw_telemetry.resident_cache_insert_attempt_count,
                cache_insert_accept_count: raw_telemetry.resident_cache_insert_accept_count,
                cache_insert_bypass_count: raw_telemetry.resident_cache_insert_bypass_count,
                oversized_tensor_bypass_count: raw_telemetry.resident_cache_oversized_tensor_bypass_count,
                oversized_expert_bypass_count: raw_telemetry.resident_cache_oversized_expert_bypass_count,
                self_eviction_risk_count: raw_telemetry.resident_cache_self_eviction_risk_count,
                bytes_by_tensor_kind: raw_telemetry.bytes_by_tensor_kind,
                bytes_per_expert: raw_telemetry.bytes_per_expert,
                selected_working_set_bytes: raw_telemetry.selected_working_set_bytes,
                cache_hit_rate,
                telemetry: raw_telemetry,
                numerics,
            };
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else {
            // Stats run
            let (telemetry_stats, numerics, raw_telemetry) = run_benchmark_for_shape(
                hidden,
                intermediate,
                out,
                selected_experts,
                seed,
                warmup,
                iters,
                repeat,
                cache_bytes,
                bypass_oversized_experts,
                &quant,
                &moe_executor,
                stream,
            )?;
            let total_cache_ops = raw_telemetry.resident_cache_hit_count + raw_telemetry.resident_cache_miss_count;
            let cache_hit_rate = if total_cache_ops > 0 {
                raw_telemetry.resident_cache_hit_count as f32 / total_cache_ops as f32
            } else {
                0.0
            };
            let report = MoeStatsReport {
                source,
                backend: "cuda",
                format,
                hidden,
                intermediate,
                out,
                selected_experts,
                logical_expert_bytes_requested: raw_telemetry.logical_expert_bytes_requested,
                actual_expert_bytes_loaded: raw_telemetry.actual_expert_bytes_loaded,
                resident_cache_bytes_reused: raw_telemetry.resident_cache_bytes_reused,
                dequantized_scratch_bytes: raw_telemetry.dequantized_scratch_bytes,
                resident_cache_capacity_bytes: raw_telemetry.resident_cache_capacity_bytes,
                resident_cache_resident_bytes: raw_telemetry.resident_cache_resident_bytes,
                resident_cache_hit_count: raw_telemetry.resident_cache_hit_count,
                resident_cache_miss_count: raw_telemetry.resident_cache_miss_count,
                resident_cache_eviction_count: raw_telemetry.resident_cache_eviction_count,
                cache_insert_attempt_count: raw_telemetry.resident_cache_insert_attempt_count,
                cache_insert_accept_count: raw_telemetry.resident_cache_insert_accept_count,
                cache_insert_bypass_count: raw_telemetry.resident_cache_insert_bypass_count,
                oversized_tensor_bypass_count: raw_telemetry.resident_cache_oversized_tensor_bypass_count,
                oversized_expert_bypass_count: raw_telemetry.resident_cache_oversized_expert_bypass_count,
                self_eviction_risk_count: raw_telemetry.resident_cache_self_eviction_risk_count,
                bytes_by_tensor_kind: raw_telemetry.bytes_by_tensor_kind,
                bytes_per_expert: raw_telemetry.bytes_per_expert,
                selected_working_set_bytes: raw_telemetry.selected_working_set_bytes,
                cache_hit_rate,
                telemetry: telemetry_stats,
                numerics,
            };
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_q4_format_default_and_aliases() -> Result<()> {
        assert_eq!(parse_quant_format(&["bench".into()])?, QuantFormat::Q4_0);
        assert_eq!(
            parse_quant_format(&["bench".into(), "--format".into(), "q4".into()])?,
            QuantFormat::Q4_0
        );
        assert_eq!(
            parse_quant_format(&["bench".into(), "--format".into(), "q4_0".into()])?,
            QuantFormat::Q4_0
        );
        Ok(())
    }

    #[test]
    fn reject_non_q4_selected_moe_format() {
        let err = parse_quant_format(&["bench".into(), "--format".into(), "q5".into()])
            .unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::Unsupported);
        assert!(err.source_message.contains("Q4-only"));
    }

    #[test]
    fn read_sanity_report_shape_extracts_dimensions() -> Result<()> {
        let dir = std::env::temp_dir().join("objeta_bench_selected_moe_sanity_shape");
        std::fs::create_dir_all(&dir).map_err(|err| {
            CudaError::new(
                CudaErrorKind::Io,
                format!("create test dir {}", dir.display()),
                err.to_string(),
                file!(),
                line!(),
                module_path!(),
            )
        })?;
        let path = dir.join("sanity_report.json");
        let payload = serde_json::json!({
            "layout_kind": "explicit_experts",
            "num_layers": 61,
            "num_experts": 256,
            "top_k": 8,
            "hidden_size": 7168,
            "intermediate_size": 2048,
            "tensor_counts": { "router": 61, "expert": 0, "shared_expert": 0, "other": 0 },
            "working_set": {
                "single_expert_bytes": 1,
                "current_layer_selected_bytes": 8,
                "prefetch_window_2_layers_bytes": 16,
                "prefetch_window_4_layers_bytes": 32,
                "full_pass_selected_bytes": 64
            },
            "cache_fit": { "1GB": true, "2GB": true, "4GB": true, "8GB": true },
            "warnings": [],
            "compatible_with_objeta_cuda_moe": true
        });
        std::fs::write(&path, serde_json::to_string_pretty(&payload).unwrap()).map_err(|err| {
            CudaError::new(
                CudaErrorKind::Io,
                format!("write test sanity report {}", path.display()),
                err.to_string(),
                file!(),
                line!(),
                module_path!(),
            )
        })?;

        let shape = read_sanity_report_shape(&path)?;
        assert_eq!(shape.hidden_size, 7168);
        assert_eq!(shape.intermediate_size, 2048);
        assert_eq!(shape.top_k, 8);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
        Ok(())
    }

    #[test]
    fn reject_zero_top_k_in_sanity_report() {
        let dir = std::env::temp_dir().join("objeta_bench_selected_moe_sanity_invalid");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sanity_report.json");
        let payload = serde_json::json!({
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "top_k": 0
        });
        std::fs::write(&path, serde_json::to_string_pretty(&payload).unwrap()).unwrap();

        let err = read_sanity_report_shape(&path).unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::InvalidInput);
        assert!(err.source_message.contains("top_k"));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
