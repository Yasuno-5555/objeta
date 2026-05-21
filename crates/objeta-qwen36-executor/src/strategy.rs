//! Strategy integration — reads objeta strategy JSON, requantizes weights.
//!
//! Pipeline:
//!   strategy.json (in bin_dir) → load on init → requantize weights
//!
//! Weights are stored as f16 (u16). Requantization simulates lower precision
//! by rounding values to N-bit quantization levels.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExpertPolicyConfig {
    #[serde(rename = "exact")]
    Exact,
    #[serde(rename = "lko_aware")]
    LkoAware,
    #[serde(rename = "top_p")]
    TopP {
        p: f32,
        min_experts: usize,
        max_experts: usize,
    },
    #[serde(rename = "contribution")]
    Contribution {
        threshold: f32,
        min_experts: usize,
        max_experts: usize,
        #[serde(default = "default_ema_beta")]
        ema_beta: f32,
    },
    #[serde(rename = "adaptive_entropy")]
    AdaptiveEntropy {
        low_entropy_p: f32,
        mid_entropy_p: f32,
        high_entropy_p: f32,
        repetition_p: f32,
        low_entropy_threshold: f32,
        mid_entropy_threshold: f32,
        min_experts: usize,
        max_experts: usize,
    },
}

impl Default for ExpertPolicyConfig {
    fn default() -> Self {
        Self::Exact
    }
}

pub fn parse_expert_policy_json(json: &str) -> Result<ExpertPolicyConfig, serde_json::Error> {
    match serde_json::from_str::<ExpertPolicyConfig>(json) {
        Ok(policy) => Ok(policy),
        Err(first_err) => {
            let mut value: serde_json::Value = serde_json::from_str(json)?;
            if let Some(obj) = value.as_object_mut() {
                if !obj.contains_key("type") {
                    if let Some(kind) = obj.remove("kind") {
                        obj.insert("type".to_string(), kind);
                    }
                }
            }
            serde_json::from_value::<ExpertPolicyConfig>(value).map_err(|_| first_err)
        }
    }
}

fn default_ema_beta() -> f32 {
    0.95
}

/// Mirror of objeta_core::RuntimeStrategy — minimal fields for executor.
#[derive(Debug, Deserialize)]
pub struct StrategyConfig {
    pub family: String,
    pub dominance: String,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub steering_layers: Vec<usize>,
    pub executor_config: ExecutorBits,
}

#[derive(Debug, Deserialize)]
pub struct ExecutorBits {
    pub ffn_bits: Vec<u8>,
    pub attn_qo_bits: Vec<u8>,
    pub attn_kv_bits: Vec<u8>,
    #[serde(default)]
    pub estimated_tok_per_sec: f64,
    #[serde(default)]
    pub estimated_vram_gb: f64,
    #[serde(default)]
    pub estimated_ppl_delta: f64,
}

/// Load strategy from JSON file. Returns None if file not found.
pub fn load_strategy(bin_dir: &Path) -> Option<StrategyConfig> {
    let path = bin_dir.join("strategy.json");
    if !path.exists() {
        return None;
    }
    let data = std::fs::read_to_string(&path).ok()?;
    let cfg: StrategyConfig = serde_json::from_str(&data).ok()?;
    eprintln!(
        "[objeta] strategy loaded: family={}, dominance={}, confidence={:.0}%",
        cfg.family,
        cfg.dominance,
        cfg.confidence * 100.0
    );
    Some(cfg)
}

/// Re-quantize f16 weight vector to target_bits precision.
/// Values are rounded to 2^target_bits uniform levels.
/// Returns new Vec<u16> with quantized values stored as f16.
pub fn requantize_f16(weights: &[u16], target_bits: u8) -> Vec<u16> {
    if target_bits >= 16 || weights.is_empty() {
        return weights.to_vec();
    }
    let n_levels = (1u32 << target_bits) as f32;

    // Find global min/max for uniform quantization
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    for &h in weights {
        let v = f16_to_f32(h);
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    let span = max_v - min_v;
    if span < 1e-10 {
        return weights.to_vec();
    }
    let scale = span / (n_levels - 1.0);

    weights
        .iter()
        .map(|&h| {
            let v = f16_to_f32(h);
            let q = ((v - min_v) / scale).round().clamp(0.0, n_levels - 1.0);
            let dq = q * scale + min_v;
            f32_to_f16(dq)
        })
        .collect()
}

// ── f16 ↔ f32 ──────────────────────────────────────────────────────────

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x3FF) as u32;
    match exp {
        0 => f32::from_bits(sign | mant << 13),
        31 => f32::from_bits(sign | 0x7F800000 | (mant << 13)),
        e => {
            let exp_f32 = (e as i32 - 15 + 127) as u32;
            f32::from_bits(sign | (exp_f32 << 23) | (mant << 13))
        }
    }
}

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        if mant == 0 {
            sign as u16
        } else {
            (sign as u32 | (mant >> 1)) as u16
        }
    } else if exp >= 31 {
        (sign as u32 | 0x7C00 | mant) as u16
    } else {
        (sign as u32 | ((exp as u32) << 10) | mant) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_requantize_q4() {
        let w: Vec<u16> = (0..256).map(|i| f32_to_f16(i as f32 / 256.0)).collect();
        let result = requantize_f16(&w, 4);
        assert_eq!(result.len(), 256);
        // Check values are quantized to 16 levels
        let mut unique = std::collections::BTreeSet::new();
        for &h in &result {
            unique.insert((f16_to_f32(h) * 1000.0) as i32);
        }
        assert!(
            unique.len() <= 16,
            "q4 should have ≤16 unique values, got {}",
            unique.len()
        );
    }

    #[test]
    fn test_load_strategy_missing() {
        assert!(load_strategy(Path::new("/nonexistent")).is_none());
    }

    #[test]
    fn test_parse_expert_policy_kind_alias() {
        let policy = parse_expert_policy_json(r#"{"kind":"lko_aware"}"#).unwrap();
        assert!(matches!(policy, ExpertPolicyConfig::LkoAware));
    }
}
