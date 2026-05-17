//! Experiment 1: Rotation Kernel Fidelity — definitive measurement.
//!
//! Measures whether the FFN output Δ = FFN(x) can be approximated by a
//! low-rank rotation: Δ ≈ U_k @ Σ_k @ V_k^T @ x
//!
//! Key metric: cos(Δ_full, Δ_rot) across rank sweep.
//! Success condition: cos > 0.97 at some achievable rank k.

use nalgebra::DMatrix;
use objeta_parser::ModelWeights;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

const TINYLLAMA: &str = "/Users/yasuno/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sweep layers 0, 7, 14, 21 to cover all zones
    let target_layers = [0, 7, 14, 21];
    let hidden_dim = 2048;
    let ffn_dim = 5632;
    let n_inputs = 200;

    println!("=== Rotation Kernel Fidelity: Multi-Layer Sweep ===");
    println!("Hidden dim: {}, FFN dim: {}, Samples: {}", hidden_dim, ffn_dim, n_inputs);
    println!();

    let weights = ModelWeights::open(TINYLLAMA)?;

    for &layer in &target_layers {
        let (_, _, gate) = weights.get_matrix(
            &format!("model.layers.{}.mlp.gate_proj.weight", layer))?;
        let (_, _, up) = weights.get_matrix(
            &format!("model.layers.{}.mlp.up_proj.weight", layer))?;
        let (_, _, down) = weights.get_matrix(
            &format!("model.layers.{}.mlp.down_proj.weight", layer))?;

        println!("{}", "=".repeat(70));
        println!("Layer L{}", layer);
        println!("{}", "=".repeat(70));

        // Generate inputs
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0 / (hidden_dim as f64).sqrt()).unwrap();
        let inputs: Vec<Vec<f64>> = (0..n_inputs)
            .map(|_| {
                let mut x: Vec<f64> = (0..hidden_dim).map(|_| normal.sample(&mut rng)).collect();
                let n: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
                for v in x.iter_mut() { *v /= n.max(1e-12); }
                x
            })
            .collect();

        // Compute Δ_full for each input
        let deltas: Vec<Vec<f64>> = inputs.iter().map(|x| {
            let mut g = vec![0.0f64; ffn_dim];
            let mut u = vec![0.0f64; ffn_dim];
            for i in 0..ffn_dim {
                let mut sg = 0.0; let mut su = 0.0;
                for j in 0..hidden_dim {
                    sg += gate[i * hidden_dim + j] as f64 * x[j];
                    su += up[i * hidden_dim + j] as f64 * x[j];
                }
                g[i] = sg; u[i] = su;
            }
            let mut h = vec![0.0f64; ffn_dim];
            for i in 0..ffn_dim { h[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i]; }
            let mut delta = vec![0.0f64; hidden_dim];
            for i in 0..hidden_dim {
                let mut s = 0.0;
                for j in 0..ffn_dim { s += down[i * ffn_dim + j] as f64 * h[j]; }
                delta[i] = s;
            }
            delta
        }).collect();

        // SVD of Δ matrix to get output basis U
        let d = hidden_dim;
        let n = n_inputs;
        let mut delta_mat = DMatrix::zeros(d, n);
        for (j, dvec) in deltas.iter().enumerate() {
            for i in 0..d { delta_mat[(i, j)] = dvec[i]; }
        }

        let t0 = Instant::now();
        let delta_svd = delta_mat.clone().svd(true, false);
        let ds = delta_svd.singular_values;
        let u_delta = delta_svd.u.unwrap();
        let svd_time = t0.elapsed();

        let eff_rank = ds.iter().copied().sum::<f64>().powi(2)
            / ds.iter().map(|&v| v*v).sum::<f64>().max(1e-12);
        let sv_ratio = ds[0] / ds.get(1).copied().unwrap_or(1.0).max(1e-8);
        let n_sv = ds.len().min(5);
        println!("SVD: {:.1}s, eff_rank={:.1}, σ₁/σ₂={:.1}, top-{}σ: {:?}",
                 svd_time.as_secs_f64(), eff_rank, sv_ratio, n_sv,
                 &ds.as_slice()[..n_sv].iter().map(|&v| format!("{:.4}", v)).collect::<Vec<_>>());

        // ── METHOD 1: Projection fidelity ──
        println!();
        println!("  Method 1: Projection fidelity ||U_k @ U_k^T @ Δ|| / ||Δ||");
        println!("  {:<6} {:>10} {:>10} {:>10} {:>10}",
                 "k", "mean_cos", "min_cos", "σ²_captured", "FLOP_save");
        println!("  {}", "-".repeat(50));

        let ks = [4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256];
        let mut best_k = None;

        for &k in &ks {
            let real_k = k.min(n);
            let mut cosines = Vec::with_capacity(n);

            for dvec in deltas.iter() {
                let mut utd = vec![0.0f64; real_k];
                for ki in 0..real_k {
                    let mut s = 0.0;
                    for i in 0..d { s += u_delta[(i, ki)] * dvec[i]; }
                    utd[ki] = s;
                }
                let mut proj_norm_sq = 0.0f64;
                for ki in 0..real_k { proj_norm_sq += utd[ki] * utd[ki]; }
                let dn_sq: f64 = dvec.iter().map(|v| v*v).sum();
                let cos = (proj_norm_sq / dn_sq.max(1e-12)).sqrt().min(1.0);
                cosines.push(cos);
            }

            let mean_cos = cosines.iter().sum::<f64>() / cosines.len() as f64;
            let min_cos = cosines.iter().cloned().fold(f64::INFINITY, f64::min);
            let var_captured = if real_k < ds.len() {
                ds.as_slice()[..real_k].iter().map(|&v| v*v).sum::<f64>()
                    / ds.iter().map(|&v| v*v).sum::<f64>().max(1e-12)
            } else { 1.0 };

            let full_flops = 3.0 * hidden_dim as f64 * ffn_dim as f64; // gate+up+down
            let rot_flops = 2.0 * hidden_dim as f64 * real_k as f64; // V^T@x + U@z
            let flop_save = (1.0 - rot_flops / full_flops) * 100.0;

            let flag = if mean_cos > 0.97 { " ★" } else if mean_cos > 0.90 { " ◆" } else { "" };
            println!("  {:<6} {:>10.4} {:>10.4} {:>10.3} {:>9.0}%{}",
                     real_k, mean_cos, min_cos, var_captured, flop_save, flag);

            if mean_cos > 0.97 && best_k.is_none() {
                best_k = Some((real_k, mean_cos, flop_save));
            }
        }

        // ── METHOD 2: Direct low-rank Δ approximation ──
        println!();
        println!("  Method 2: Direct Δ ≈ U_k @ Σ_k @ V^T @ x (learned V)");
        let best_proj_k = best_k.map(|(k, _, _)| k).unwrap_or(128);
        let k = best_proj_k.min(n);

        // Build X matrix (d × n) and C = U_k^T @ Δ (k × n)
        let mut x_mat = DMatrix::zeros(d, n);
        let mut c_mat = DMatrix::zeros(k, n);
        for (j, (x, delta)) in inputs.iter().zip(deltas.iter()).enumerate() {
            for i in 0..d {
                x_mat[(i, j)] = x[i];
                if j < n {
                    let mut s = 0.0;
                    for ki in 0..k { /* computed below */ }
                }
            }
            for ki in 0..k {
                let mut s = 0.0;
                for i in 0..d { s += u_delta[(i, ki)] * delta[i]; }
                c_mat[(ki, j)] = s;
            }
        }

        // V^T = Σ^{-1} @ C @ X^T @ (X @ X^T + λI)^{-1}
        let lambda = 1e-2f64;
        let xtx: DMatrix<f64> = &x_mat * &x_mat.transpose();
        let mut xtx_reg = xtx.clone();
        for i in 0..d { xtx_reg[(i, i)] += lambda; }
        let xtx_inv = xtx_reg.cholesky().expect("pos def").inverse();

        let mut sinv_c = DMatrix::zeros(k, n);
        for ki in 0..k {
            for j in 0..n {
                sinv_c[(ki, j)] = c_mat[(ki, j)] / ds[ki].max(1e-8);
            }
        }
        let vt: DMatrix<f64> = &sinv_c * &x_mat.transpose() * &xtx_inv; // (k × d)

        println!("  {:<6} {:>10} {:>10} {:>10}",
                 "k", "mean_cos", "min_cos", "FLOP_save");
        println!("  {}", "-".repeat(36));

        for &test_k in &[4, 8, 16, 24, 32, 48, 64] {
            let tk = test_k.min(k);
            let mut cosines = Vec::new();

            for (j, (x, delta)) in inputs.iter().zip(deltas.iter()).enumerate() {
                let mut z = vec![0.0f64; tk];
                for ki in 0..tk {
                    let mut s = 0.0;
                    for i in 0..d { s += vt[(ki, i)] * x[i]; }
                    z[ki] = s * ds[ki];
                }
                let mut rot = vec![0.0f64; d];
                for i in 0..d {
                    let mut s = 0.0;
                    for ki in 0..tk { s += u_delta[(i, ki)] * z[ki]; }
                    rot[i] = s;
                }
                let dn: f64 = delta.iter().map(|v| v*v).sum::<f64>().sqrt();
                let rn: f64 = rot.iter().map(|v| v*v).sum::<f64>().sqrt();
                let cos = if dn > 1e-12 && rn > 1e-12 {
                    (delta.iter().zip(rot.iter()).map(|(a,b)| a*b).sum::<f64>() / (dn * rn)).clamp(-1.0, 1.0)
                } else { 0.0 };
                cosines.push(cos);
            }

            let mean_cos = cosines.iter().sum::<f64>() / cosines.len() as f64;
            let min_cos = cosines.iter().cloned().fold(f64::INFINITY, f64::min);
            let rot_flops = 2.0 * d as f64 * tk as f64;
            let full_flops = 3.0 * d as f64 * ffn_dim as f64;
            let flop_save = (1.0 - rot_flops / full_flops) * 100.0;
            let flag = if mean_cos > 0.97 { " ★" } else if mean_cos > 0.90 { " ◆" } else { "" };
            println!("  {:<6} {:>10.4} {:>10.4} {:>9.0}%{}",
                     tk, mean_cos, min_cos, flop_save, flag);
        }

        // Summary
        println!();
        if let Some((k, cos, save)) = best_k {
            println!("  ✓ L{}: cos>0.97 at k={} (cos={:.4}, {:.0}% FLOPs saved)", layer, k, cos, save);
        } else {
            let best = ks.iter().filter(|&&k| k <= n).last().unwrap();
            println!("  ✗ L{}: cos>0.97 NOT reached up to k={}", layer, best);
        }
        println!();
    }

    println!("{}", "=".repeat(70));
    println!("CONCLUSION");
    println!("{}", "=".repeat(70));
    Ok(())
}
