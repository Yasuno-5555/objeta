use objeta_cuda::{
    compare_outputs, dense_gemv_cpu, gemv_cpu, CudaBackendBuilder,
    QGemvNumericsSuite, QGemvReport, QGemvShape, QuantBackend, QuantFormat, Result,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QGemvStats {
    pub avg: f32,
    pub min: f32,
    pub max: f32,
    pub p50: f32,
    pub p95: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QGemvStatsTelemetry {
    pub h2d_ms: QGemvStats,
    pub kernel_ms: QGemvStats,
    pub d2h_ms: QGemvStats,
    pub unaccounted_ms: QGemvStats,
    pub total_ms: QGemvStats,
    pub bytes_read: usize,
    pub effective_gbps: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QGemvStatsReport {
    pub backend: &'static str,
    pub kernel: &'static str,
    pub format: QuantFormat,
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    pub block_bytes: usize,
    pub telemetry: QGemvStatsTelemetry,
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

fn percentile(sorted_values: &[f32], p: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted_values.len() - 1) as f32).round() as usize;
    sorted_values[idx]
}

fn compute_stats(mut values: Vec<f32>) -> QGemvStats {
    if values.is_empty() {
        return QGemvStats {
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

    QGemvStats {
        avg,
        min,
        max,
        p50,
        p95,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark_for_shape(
    format: QuantFormat,
    rows: usize,
    cols: usize,
    seed: u64,
    warmup: usize,
    iters: usize,
    quant: &QuantBackend,
    stream: &objeta_cuda::CudaStreamHandle,
) -> Result<(QGemvStatsTelemetry, QGemvNumericsSuite)> {
    let shape = QGemvShape::new(format, rows, cols);
    let matrix = seeded_f32s(rows * cols, seed ^ 0xABCD_EF01_2345_6789);
    let x = seeded_f32s(cols, seed ^ 0x1020_3040_5060_7080);

    let qweights = quant.quantize_matrix(format, &matrix, shape)?;
    let reference = gemv_cpu(format, &qweights, &x, shape)?;

    // Warmup iterations
    for _ in 0..warmup {
        let _ = quant.gemv(format, stream, &qweights, &x, shape)?;
    }

    // Timed iterations
    let mut h2d_vals = Vec::with_capacity(iters);
    let mut kernel_vals = Vec::with_capacity(iters);
    let mut d2h_vals = Vec::with_capacity(iters);
    let mut unaccounted_vals = Vec::with_capacity(iters);
    let mut total_vals = Vec::with_capacity(iters);
    let mut actual = Vec::new();

    for i in 0..iters {
        let (y, tel) = quant.gemv(format, stream, &qweights, &x, shape)?;
        h2d_vals.push(tel.h2d_ms);
        kernel_vals.push(tel.kernel_ms);
        d2h_vals.push(tel.d2h_ms);
        unaccounted_vals.push(tel.unaccounted_ms);
        total_vals.push(tel.total_ms);
        if i == 0 {
            actual = y;
        }
    }

    let cuda_vs_cpu_quant = compare_outputs(&reference, &actual)?;
    let dense_ref = dense_gemv_cpu(&matrix, &x, shape)?;
    let quant_vs_fp32 = compare_outputs(&dense_ref, &actual)?;

    let numerics = QGemvNumericsSuite {
        cosine_similarity: cuda_vs_cpu_quant.cosine_similarity,
        relative_l2_error: cuda_vs_cpu_quant.relative_l2_error,
        max_abs_error: cuda_vs_cpu_quant.max_abs_error,
        cuda_vs_cpu_quant,
        quant_vs_fp32,
    };

    let h2d_stats = compute_stats(h2d_vals);
    let kernel_stats = compute_stats(kernel_vals);
    let d2h_stats = compute_stats(d2h_vals);
    let unaccounted_stats = compute_stats(unaccounted_vals);
    let total_stats = compute_stats(total_vals);

    let bytes_read = shape.quantized_matrix_bytes() + shape.cols * std::mem::size_of::<f32>();
    let avg_kernel_s = kernel_stats.avg / 1000.0;
    let effective_gbps = if avg_kernel_s > 0.0 {
        bytes_read as f32 / avg_kernel_s / 1.0e9
    } else {
        0.0
    };

    let telemetry_stats = QGemvStatsTelemetry {
        h2d_ms: h2d_stats,
        kernel_ms: kernel_stats,
        d2h_ms: d2h_stats,
        unaccounted_ms: unaccounted_stats,
        total_ms: total_stats,
        bytes_read,
        effective_gbps,
    };

    Ok((telemetry_stats, numerics))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let format_str = parse_flag(&args, "--format").unwrap_or_else(|| "q4".to_string());
    let format = match format_str.as_str() {
        "q4" | "q4_0" => QuantFormat::Q4_0,
        "q5" | "q5_0" => QuantFormat::Q5_0,
        "iq3" | "iq3_0" => QuantFormat::IQ3_0,
        "fp4" | "fp4_e2m1" | "deepseek_fp4" | "deepseek_fp4_e2m1" => QuantFormat::DeepSeekFp4E2M1,
        other => {
            eprintln!("unsupported --format {other}; expected q4, q5, iq3, or fp4");
            std::process::exit(2);
        }
    };

    let is_matrix = has_flag(&args, "--matrix");
    let iters = parse_flag(&args, "--iters")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(if is_matrix { 10 } else { 1 });
    let warmup = parse_flag(&args, "--warmup")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(if is_matrix { 5 } else { 0 });

    let rows = parse_flag(&args, "--rows")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(11008);
    let cols = parse_flag(&args, "--cols")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096);
    let seed = parse_flag(&args, "--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(123);

    let backend = CudaBackendBuilder::new().stream_count(2).build()?;
    let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
    let stream = backend.stream_pool().stream(0)?;

    // Ensure kernel compilation happens outside timed iterations
    quant.compile_format(format)?;

    if is_matrix {
        let formats = &[
            QuantFormat::Q4_0,
            QuantFormat::Q5_0,
            QuantFormat::IQ3_0,
            QuantFormat::DeepSeekFp4E2M1,
        ];
        let matrix_shapes = &[
            (4096, 4096),
            (11008, 4096),
            (14336, 4096),
            (4096, 6144),
            (14336, 6144),
            (4096, 8192),
            (14336, 8192),
        ];

        for &fmt in formats {
            quant.compile_format(fmt)?;
            for &(r, c) in matrix_shapes {
                let (telemetry, numerics) = run_benchmark_for_shape(
                    fmt,
                    r,
                    c,
                    seed,
                    warmup,
                    iters,
                    &quant,
                    stream,
                )?;
                let shape = QGemvShape::new(fmt, r, c);
                let report = QGemvStatsReport {
                    backend: "cuda",
                    kernel: fmt.kernel_name(),
                    format: fmt,
                    rows: r,
                    cols: c,
                    block_size: shape.block_size,
                    block_bytes: shape.block_bytes,
                    telemetry,
                    numerics,
                };
                println!("{}", serde_json::to_string(&report).unwrap());
            }
        }
    } else {
        // If iters == 1 and warmup == 0, and not explicitly asked for multiple runs, print single-run schema.
        if iters == 1 && warmup == 0 && parse_flag(&args, "--iters").is_none() && parse_flag(&args, "--warmup").is_none() {
            let shape = QGemvShape::new(format, rows, cols);
            let matrix = seeded_f32s(rows * cols, seed ^ 0xABCD_EF01_2345_6789);
            let x = seeded_f32s(cols, seed ^ 0x1020_3040_5060_7080);
            let qweights = quant.quantize_matrix(format, &matrix, shape)?;
            let reference = gemv_cpu(format, &qweights, &x, shape)?;
            let (actual, telemetry) = quant.gemv(format, stream, &qweights, &x, shape)?;
            let cuda_vs_cpu_quant = compare_outputs(&reference, &actual)?;
            let dense_ref = dense_gemv_cpu(&matrix, &x, shape)?;
            let quant_vs_fp32 = compare_outputs(&dense_ref, &actual)?;

            let numerics = QGemvNumericsSuite {
                cosine_similarity: cuda_vs_cpu_quant.cosine_similarity,
                relative_l2_error: cuda_vs_cpu_quant.relative_l2_error,
                max_abs_error: cuda_vs_cpu_quant.max_abs_error,
                cuda_vs_cpu_quant,
                quant_vs_fp32,
            };

            let report = QGemvReport {
                backend: "cuda",
                kernel: format.kernel_name(),
                format,
                rows,
                cols,
                block_size: shape.block_size,
                block_bytes: shape.block_bytes,
                telemetry,
                numerics,
            };
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else {
            let (telemetry, numerics) = run_benchmark_for_shape(
                format,
                rows,
                cols,
                seed,
                warmup,
                iters,
                &quant,
                stream,
            )?;
            let shape = QGemvShape::new(format, rows, cols);
            let report = QGemvStatsReport {
                backend: "cuda",
                kernel: format.kernel_name(),
                format,
                rows,
                cols,
                block_size: shape.block_size,
                block_bytes: shape.block_bytes,
                telemetry,
                numerics,
            };
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
    }

    Ok(())
}
