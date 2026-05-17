//! objeta-moe — MoE Routing Compiler for Qwen3.6-35B-A3B.
//!
//! Analyzes router weights to generate expert execution plans:
//!   - Expert occupancy histograms → hot/warm/cold tiering
//!   - Transition matrices → prefetch scheduling
//!   - Routing entropy maps → bridge layer detection
//!   - Occupancy skew → static tiering confidence
//!
//! Output: execution_plan.json

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ── Constants ─────────────────────────────────────────────────────────────

const N_LAYERS: usize = 40;
const HIDDEN_DIM: usize = 2048;
const N_EXPERTS: usize = 256;
const TOP_K: usize = 8;

// ── Router Weights ────────────────────────────────────────────────────────

/// Load router weight matrices from binary files.
/// Format: layer_{l}_router.bin = 256 × 2048 fp32 = 2,097,152 bytes
pub fn load_routers(bin_dir: &Path) -> Vec<Option<Vec<f32>>> {
    (0..N_LAYERS)
        .map(|l| {
            let path = bin_dir.join(format!("layer_{}_router.bin", l));
            if !path.exists() {
                return None;
            }
            let bytes = std::fs::read(&path).ok()?;
            if bytes.len() != N_EXPERTS * HIDDEN_DIM * 4 {
                return None;
            }
            let mut weights = vec![0.0f32; N_EXPERTS * HIDDEN_DIM];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                weights[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Some(weights)
        })
        .collect()
}

// ── Routing Analysis ──────────────────────────────────────────────────────

/// Result of routing analysis for all layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingAnalysis {
    /// Per-layer expert occupancy: (n_layers, n_experts)
    pub occupancy: Vec<Vec<f32>>,
    /// Per-layer routing entropy
    pub entropy_mean: Vec<f32>,
    pub entropy_std: Vec<f32>,
    /// Per-layer occupancy skew (max/mean)
    pub occupancy_skew: Vec<f32>,
    /// Transition matrices: (n_layers-1, n_experts, n_experts)
    /// transitions[l][src][dst] = P(expert_dst at l+1 | expert_src at l)
    pub transitions: Vec<Vec<Vec<f32>>>,
    /// Bridge layers detected
    pub bridge_layers: Vec<BridgeInfo>,
    /// Top-8 experts per layer by occupancy
    pub top_experts: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeInfo {
    pub layer: usize,
    pub score: f32,
    pub entropy: f32,
    pub reasons: Vec<String>,
}

/// Run the full routing analysis.
pub fn analyze_routers(routers: &[Option<Vec<f32>>], n_samples: usize) -> RoutingAnalysis {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0 / (HIDDEN_DIM as f64).sqrt()).unwrap();

    // Generate synthetic hidden states
    let inputs: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            let pos_frac = (i % 30) as f32 / 30.0;
            let mut h = vec![0.0f32; HIDDEN_DIM];
            for v in h.iter_mut() {
                *v = normal.sample(&mut rng) as f32;
            }
            // Position-dependent bias
            for v in h.iter_mut() {
                *v += pos_frac * 0.1;
            }
            // L2 normalize
            let n: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in h.iter_mut() {
                *v /= n.max(1e-8);
            }
            h
        })
        .collect();

    // Per-layer accumulation
    let mut occupancy = vec![vec![0.0f32; N_EXPERTS]; N_LAYERS];
    let mut entropy_sum = vec![0.0f32; N_LAYERS];
    let mut entropy_sq_sum = vec![0.0f32; N_LAYERS];
    let mut top_expert_counts = vec![vec![0u32; N_EXPERTS]; N_LAYERS];
    let mut transitions =
        vec![vec![vec![0.0f32; N_EXPERTS]; N_EXPERTS]; N_LAYERS - 1];

    for input in &inputs {
        let mut prev_top1 = 0usize;

        for l in 0..N_LAYERS {
            let router = match &routers[l] {
                Some(r) => r,
                None => continue,
            };

            // Router forward: logits = W @ h
            let mut logits = vec![0.0f32; N_EXPERTS];
            for e in 0..N_EXPERTS {
                let mut s = 0.0;
                let row = &router[e * HIDDEN_DIM..(e + 1) * HIDDEN_DIM];
                for (j, &h_j) in input.iter().enumerate() {
                    s += row[j] * h_j;
                }
                logits[e] = s;
            }

            // Softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; N_EXPERTS];
            let mut sum_exp = 0.0f32;
            for e in 0..N_EXPERTS {
                let p = (logits[e] - max_logit).exp();
                probs[e] = p;
                sum_exp += p;
            }
            for e in 0..N_EXPERTS {
                probs[e] /= sum_exp.max(1e-12);
            }

            // Entropy
            let entropy: f32 = -probs
                .iter()
                .map(|&p| if p > 1e-12 { p * p.ln() } else { 0.0 })
                .sum::<f32>()
                / 2.302585; // ln(10) → approximate, use natural log
            let entropy_nat: f32 = -probs
                .iter()
                .map(|&p| if p > 1e-12 { p * p.ln() } else { 0.0 })
                .sum::<f32>();
            entropy_sum[l] += entropy_nat;
            entropy_sq_sum[l] += entropy_nat * entropy_nat;

            // Occupancy: accumulate probabilities
            for e in 0..N_EXPERTS {
                occupancy[l][e] += probs[e];
            }

            // Top-1 expert
            let top1 = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            top_expert_counts[l][top1] += 1;

            // Transition from previous layer
            if l > 0 {
                transitions[l - 1][prev_top1][top1] += 1.0;
            }
            prev_top1 = top1;
        }
    }

    // Normalize occupancy
    for l in 0..N_LAYERS {
        let total: f32 = occupancy[l].iter().sum();
        if total > 0.0 {
            for e in 0..N_EXPERTS {
                occupancy[l][e] /= total;
            }
        }
    }

    // Entropy stats
    let entropy_mean: Vec<f32> = (0..N_LAYERS)
        .map(|l| entropy_sum[l] / n_samples as f32)
        .collect();
    let entropy_std: Vec<f32> = (0..N_LAYERS)
        .map(|l| {
            let mean = entropy_mean[l];
            let var = entropy_sq_sum[l] / n_samples as f32 - mean * mean;
            var.max(0.0).sqrt()
        })
        .collect();

    // Occupancy skew
    let occupancy_skew: Vec<f32> = (0..N_LAYERS)
        .map(|l| {
            let mean = occupancy[l].iter().sum::<f32>() / N_EXPERTS as f32;
            let max = occupancy[l].iter().cloned().fold(0.0f32, f32::max);
            if mean > 0.0 { max / mean } else { 0.0 }
        })
        .collect();

    // Normalize transitions
    for l in 0..(N_LAYERS - 1) {
        for src in 0..N_EXPERTS {
            let row_sum: f32 = transitions[l][src].iter().sum();
            if row_sum > 0.0 {
                for dst in 0..N_EXPERTS {
                    transitions[l][src][dst] /= row_sum;
                }
            }
        }
    }

    // Top experts per layer
    let top_experts: Vec<Vec<usize>> = (0..N_LAYERS)
        .map(|l| {
            let mut indexed: Vec<(usize, f32)> =
                occupancy[l].iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.into_iter().take(TOP_K).map(|(i, _)| i).collect()
        })
        .collect();

    // Bridge layer detection
    let bridge_layers = detect_bridges(&occupancy, &transitions, &entropy_mean);

    RoutingAnalysis {
        occupancy,
        entropy_mean,
        entropy_std,
        occupancy_skew,
        transitions,
        bridge_layers,
        top_experts,
    }
}

// ── Bridge Detection ──────────────────────────────────────────────────────

fn detect_bridges(
    occupancy: &[Vec<f32>],
    transitions: &[Vec<Vec<f32>>],
    entropy: &[f32],
) -> Vec<BridgeInfo> {
    let mut bridges = Vec::new();

    for l in 1..(N_LAYERS - 1) {
        let mut score = 0.0f32;
        let mut reasons = Vec::new();

        // 1. Entropy spike (>20% increase)
        if entropy[l] > entropy[l - 1] * 1.2 {
            score += 1.0;
            reasons.push(format!(
                "entropy_spike({:.2}→{:.2})",
                entropy[l - 1], entropy[l]
            ));
        }

        // 2. Transition entropy increase (>30% vs previous)
        let trans_entropy: f32 = transitions[l - 1]
            .iter()
            .map(|row| {
                -row.iter()
                    .map(|&p| if p > 1e-12 { p * p.ln() } else { 0.0 })
                    .sum::<f32>()
            })
            .sum::<f32>()
            / N_EXPERTS as f32;

        let prev_trans_entropy = if l >= 2 {
            transitions[l - 2]
                .iter()
                .map(|row| {
                    -row.iter()
                        .map(|&p| if p > 1e-12 { p * p.ln() } else { 0.0 })
                        .sum::<f32>()
                })
                .sum::<f32>()
                / N_EXPERTS as f32
        } else {
            trans_entropy
        };

        if trans_entropy > prev_trans_entropy * 1.3 {
            score += 1.0;
            reasons.push(format!(
                "trans_entropy_spike({:.2}→{:.2})",
                prev_trans_entropy, trans_entropy
            ));
        }

        // 3. Occupancy correlation drop
        let mut occ_corr = 0.0f32;
        let mean_prev: f32 = occupancy[l - 1].iter().sum::<f32>() / N_EXPERTS as f32;
        let mean_curr: f32 = occupancy[l].iter().sum::<f32>() / N_EXPERTS as f32;
        let mut cov = 0.0;
        let mut var_prev = 0.0;
        let mut var_curr = 0.0;
        for e in 0..N_EXPERTS {
            let dp = occupancy[l - 1][e] - mean_prev;
            let dc = occupancy[l][e] - mean_curr;
            cov += dp * dc;
            var_prev += dp * dp;
            var_curr += dc * dc;
        }
        if var_prev > 1e-12 && var_curr > 1e-12 {
            occ_corr = cov / (var_prev.sqrt() * var_curr.sqrt());
        }
        if occ_corr < 0.7 {
            score += 1.0;
            reasons.push(format!("occ_corr_drop({:.3})", occ_corr));
        }

        if score >= 2.0 {
            bridges.push(BridgeInfo {
                layer: l,
                score,
                entropy: entropy[l],
                reasons,
            });
        }
    }

    bridges
}

// ── Execution Plan Generation ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub model: String,
    pub n_layers: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub hot_experts: HashMap<String, Vec<usize>>,
    pub warm_experts: HashMap<String, Vec<usize>>,
    pub cold_experts: HashMap<String, Vec<usize>>,
    pub prefetch_schedule: HashMap<String, HashMap<String, Vec<usize>>>,
    pub bridge_layers: Vec<usize>,
    pub bridge_details: Vec<BridgeInfo>,
    pub routing_entropy_mean: HashMap<String, f32>,
    pub occupancy_skew: HashMap<String, f32>,
}

pub fn generate_execution_plan(analysis: &RoutingAnalysis) -> ExecutionPlan {
    let mut hot_experts = HashMap::new();
    let mut warm_experts = HashMap::new();
    let mut cold_experts = HashMap::new();

    for l in 0..N_LAYERS {
        let mut indexed: Vec<(usize, f32)> =
            analysis.occupancy[l].iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let hot: Vec<usize> = indexed.iter().take(8).map(|(i, _)| *i).collect();
        let warm: Vec<usize> = indexed.iter().skip(8).take(16).map(|(i, _)| *i).collect();
        let cold: Vec<usize> = indexed.iter().skip(24).map(|(i, _)| *i).collect();

        hot_experts.insert(l.to_string(), hot);
        warm_experts.insert(l.to_string(), warm);
        cold_experts.insert(l.to_string(), cold);
    }

    // Prefetch: for each layer, top-3 next experts per source expert
    let mut prefetch_schedule = HashMap::new();
    for l in 0..(N_LAYERS - 1) {
        let mut layer_schedule = HashMap::new();
        for src in 0..N_EXPERTS {
            let mut indexed: Vec<(usize, f32)> = analysis.transitions[l][src]
                .iter()
                .copied()
                .enumerate()
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top3: Vec<usize> = indexed.into_iter().take(3).map(|(i, _)| i).collect();
            layer_schedule.insert(src.to_string(), top3);
        }
        prefetch_schedule.insert(l.to_string(), layer_schedule);
    }

    let routing_entropy_mean: HashMap<String, f32> = (0..N_LAYERS)
        .map(|l| (l.to_string(), analysis.entropy_mean[l]))
        .collect();

    let occupancy_skew: HashMap<String, f32> = (0..N_LAYERS)
        .map(|l| (l.to_string(), analysis.occupancy_skew[l]))
        .collect();

    ExecutionPlan {
        model: "Qwen3.6-35B-A3B".into(),
        n_layers: N_LAYERS,
        n_experts: N_EXPERTS,
        top_k: TOP_K,
        hot_experts,
        warm_experts,
        cold_experts,
        prefetch_schedule,
        bridge_layers: analysis.bridge_layers.iter().map(|b| b.layer).collect(),
        bridge_details: analysis.bridge_layers.clone(),
        routing_entropy_mean,
        occupancy_skew,
    }
}

// ── CLI entry point ───────────────────────────────────────────────────────

/// Run the full MoE analysis pipeline and write execution_plan.json.
pub fn run(bin_dir: &Path, output_path: &Path, n_samples: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading router weights from {}...", bin_dir.display());
    let routers = load_routers(bin_dir);
    let n_loaded = routers.iter().filter(|r| r.is_some()).count();
    println!("  Loaded {}/{} router matrices", n_loaded, N_LAYERS);

    if n_loaded == 0 {
        return Err("No router weights found".into());
    }

    println!("Running routing analysis ({} samples)...", n_samples);
    let analysis = analyze_routers(&routers, n_samples);

    println!("Generating execution plan...");
    let plan = generate_execution_plan(&analysis);

    // Report
    println!();
    println!("Top-8 experts for key layers:");
    for &l in &[0, 2, 5, 10, 20, 30, 39] {
        println!("  L{:>2}: {:?}", l, &analysis.top_experts[l]);
    }

    println!();
    println!("Occupancy skew:");
    for &l in &[0, 2, 5, 10, 20, 30, 39] {
        println!("  L{:>2}: {:.1}x", l, analysis.occupancy_skew[l]);
    }

    println!();
    println!("Routing entropy:");
    for &l in &[0, 2, 5, 10, 20, 30, 39] {
        println!(
            "  L{:>2}: H={:.3} ± {:.3}",
            l, analysis.entropy_mean[l], analysis.entropy_std[l]
        );
    }

    println!();
    println!("Bridge layers: {}", analysis.bridge_layers.len());
    for b in &analysis.bridge_layers {
        println!("  L{}: score={:.1} reasons={:?}", b.layer, b.score, b.reasons);
    }

    let json = serde_json::to_string_pretty(&plan)?;
    std::fs::write(output_path, &json)?;
    println!("\nExecution plan saved: {}", output_path.display());

    Ok(())
}
