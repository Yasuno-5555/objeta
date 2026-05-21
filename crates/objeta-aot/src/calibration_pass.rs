use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs;
use std::path::Path;

use crate::types::*;

/// Load and analyze calibration traces. Returns raw stats for further processing.
pub fn run(calib_path: &Path) -> Result<CalibrationStats, Box<dyn Error>> {
    let events = load_calibration_events(calib_path)?;
    let mut usage: BTreeMap<ExpertKey, ExpertUsageStats> = BTreeMap::new();
    let mut layer_event_counts: BTreeMap<u32, u64> = BTreeMap::new();
    let mut pair_counts: BTreeMap<(u32, u32, u32), u64> = BTreeMap::new();
    let mut total_events = 0u64;

    for event in &events {
        let count = event.selected_experts.len().min(event.selected_weights.len());
        if count == 0 {
            continue;
        }
        total_events += 1;
        *layer_event_counts.entry(event.layer).or_default() += 1;

        for i in 0..count {
            let key = ExpertKey {
                layer: event.layer,
                expert: event.selected_experts[i],
            };
            let weight = event.selected_weights[i] as f64;
            let stats = usage.entry(key).or_default();
            stats.selected_count += 1;
            stats.sum_gate_weight += weight;
            stats.max_gate_weight = stats.max_gate_weight.max(weight);
        }

        let mut uniq = event.selected_experts[..count].to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        for i in 0..uniq.len() {
            for j in (i + 1)..uniq.len() {
                *pair_counts.entry((event.layer, uniq[i], uniq[j])).or_default() += 1;
            }
        }
    }

    // Build importance using existing 2-term formula (for backward compat with CalibrationStats)
    let importance = compute_importance_2term(&usage, &layer_event_counts);

    // Build co-residency
    let mut coresidency_pairs = Vec::new();
    for ((layer, expert_a, expert_b), co_count) in &pair_counts {
        let total = *layer_event_counts.get(layer).unwrap_or(&1) as f64;
        let co_score = *co_count as f64 / total;
        coresidency_pairs.push(ExpertCoresidencyPair {
            layer: *layer,
            expert_a: *expert_a,
            expert_b: *expert_b,
            co_count: *co_count,
            co_score,
        });
    }
    coresidency_pairs.sort_by(|a, b| {
        a.layer
            .cmp(&b.layer)
            .then_with(|| b.co_count.cmp(&a.co_count))
            .then_with(|| a.expert_a.cmp(&b.expert_a))
            .then_with(|| a.expert_b.cmp(&b.expert_b))
    });

    Ok(CalibrationStats {
        importance,
        coresidency: ExpertCoresidency {
            schema_version: 1,
            pairs: coresidency_pairs,
        },
        layer_event_counts,
        total_events,
    })
}

/// 2-term importance formula (existing compile behavior):
/// importance = 0.70 * norm_frequency + 0.30 * norm_avg_gate_weight
fn compute_importance_2term(
    usage: &BTreeMap<ExpertKey, ExpertUsageStats>,
    layer_event_counts: &BTreeMap<u32, u64>,
) -> ExpertImportance {
    compute_importance_inner(usage, layer_event_counts, 0.70, 0.30, 0.0)
}

/// 3-term importance formula (spec):
/// importance = 0.50 * norm_frequency + 0.30 * norm_avg_gate_weight + 0.20 * norm_max_gate_weight
pub fn compute_importance_3term(calib: &CalibrationStats) -> ExpertImportance {
    // Reconstruct usage stats from the existing importance entries
    // This is a re-computation from raw data stored in CalibrationStats
    let mut usage: BTreeMap<ExpertKey, ExpertUsageStats> = BTreeMap::new();
    for e in &calib.importance.experts {
        let key = ExpertKey {
            layer: e.layer,
            expert: e.expert,
        };
        usage.insert(
            key,
            ExpertUsageStats {
                selected_count: e.selected_count,
                sum_gate_weight: e.avg_gate_weight * e.selected_count as f64,
                max_gate_weight: e.max_gate_weight,
            },
        );
    }
    compute_importance_inner(&usage, &calib.layer_event_counts, 0.50, 0.30, 0.20)
}

fn compute_importance_inner(
    usage: &BTreeMap<ExpertKey, ExpertUsageStats>,
    layer_event_counts: &BTreeMap<u32, u64>,
    w_freq: f64,
    w_avg_gate: f64,
    w_max_gate: f64,
) -> ExpertImportance {
    let mut per_layer_entries: BTreeMap<u32, Vec<ExpertImportanceEntry>> = BTreeMap::new();
    let mut layer_max_frequency: HashMap<u32, f64> = HashMap::new();
    let mut layer_max_avg_gate: HashMap<u32, f64> = HashMap::new();
    let mut layer_max_max_gate: HashMap<u32, f64> = HashMap::new();

    for (key, stats) in usage {
        let total_events = *layer_event_counts.get(&key.layer).unwrap_or(&1) as f64;
        let frequency = stats.selected_count as f64 / total_events;
        let avg_gate_weight = if stats.selected_count > 0 {
            stats.sum_gate_weight / stats.selected_count as f64
        } else {
            0.0
        };

        layer_max_frequency
            .entry(key.layer)
            .and_modify(|v| *v = v.max(frequency))
            .or_insert(frequency);
        layer_max_avg_gate
            .entry(key.layer)
            .and_modify(|v| *v = v.max(avg_gate_weight))
            .or_insert(avg_gate_weight);
        layer_max_max_gate
            .entry(key.layer)
            .and_modify(|v| *v = v.max(stats.max_gate_weight))
            .or_insert(stats.max_gate_weight);

        per_layer_entries
            .entry(key.layer)
            .or_default()
            .push(ExpertImportanceEntry {
                layer: key.layer,
                expert: key.expert,
                selected_count: stats.selected_count,
                frequency,
                avg_gate_weight,
                max_gate_weight: stats.max_gate_weight,
                importance: 0.0,
                tier: ExpertTier::Cold,
                recommended_format: "q4".to_string(),
                eviction_priority: 1.0,
            });
    }

    let mut importance_entries = Vec::new();
    for (layer, mut entries) in per_layer_entries {
        let max_freq = layer_max_frequency
            .get(&layer)
            .copied()
            .unwrap_or(1.0)
            .max(1e-12);
        let max_avg_gate = layer_max_avg_gate
            .get(&layer)
            .copied()
            .unwrap_or(1.0)
            .max(1e-12);
        let max_max_gate = layer_max_max_gate
            .get(&layer)
            .copied()
            .unwrap_or(1.0)
            .max(1e-12);

        for entry in &mut entries {
            let norm_frequency = entry.frequency / max_freq;
            let norm_avg_gate_weight = entry.avg_gate_weight / max_avg_gate;
            let norm_max_gate_weight = entry.max_gate_weight / max_max_gate;
            entry.importance = w_freq * norm_frequency
                + w_avg_gate * norm_avg_gate_weight
                + w_max_gate * norm_max_gate_weight;
        }
        entries.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.expert.cmp(&b.expert))
        });

        let n = entries.len();
        let hot_cutoff = ((n as f64) * 0.10).ceil() as usize;
        let warm_cutoff = ((n as f64) * 0.40).ceil() as usize;
        for (idx, entry) in entries.iter_mut().enumerate() {
            entry.tier = if idx < hot_cutoff.max((n > 0) as usize) {
                ExpertTier::Hot
            } else if idx < warm_cutoff.max(hot_cutoff.max((n > 0) as usize)) {
                ExpertTier::Warm
            } else {
                ExpertTier::Cold
            };
            entry.eviction_priority = (1.0 - entry.importance).clamp(0.0, 1.0);
        }
        importance_entries.extend(entries);
    }

    ExpertImportance {
        schema_version: 1,
        experts: importance_entries,
    }
}

pub fn load_calibration_events(
    calib_path: &Path,
) -> Result<Vec<CalibrationTraceEvent>, Box<dyn Error>> {
    let text = fs::read_to_string(calib_path)?;
    if let Ok(env) = serde_json::from_str::<MoeStatsEnvelope>(&text) {
        if !env.moe_io_events.is_empty() {
            return Ok(env
                .moe_io_events
                .into_iter()
                .filter(|e| !e.selected_experts.is_empty() && !e.selected_weights.is_empty())
                .map(|e| CalibrationTraceEvent {
                    token_id: None,
                    layer: e.layer_id,
                    selected_experts: e.selected_experts,
                    selected_weights: e.selected_weights,
                })
                .collect());
        }
    }

    let mut events = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event: CalibrationTraceEvent = serde_json::from_str(line).map_err(|err| {
            format!(
                "failed to parse calibration event at {}:{}: {}",
                calib_path.display(),
                line_no + 1,
                err
            )
        })?;
        events.push(event);
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn importance_uses_three_term_formula() {
        let root = std::env::temp_dir().join(format!("objeta_aot_3term_{}", crate::util::now_rfc3339ish()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("calib_trace.jsonl");
        std::fs::write(
            &path,
            [
                r#"{"token_id":0,"layer":5,"selected_experts":[1,2],"selected_weights":[0.50,0.10]}"#,
                r#"{"token_id":1,"layer":5,"selected_experts":[1,3],"selected_weights":[0.40,0.20]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let calib_stats = run(&path).unwrap();
        let importance_3 = compute_importance_3term(&calib_stats);

        // Expert 1: freq=2/2=1.0, avg_gate=(0.50+0.40)/2=0.45, max_gate=0.50
        // With 3-term: 0.50*1.0 + 0.30*(0.45/0.45) + 0.20*(0.50/0.50) = 0.50 + 0.30 + 0.20 = 1.0
        let e1 = importance_3.experts.iter().find(|e| e.layer == 5 && e.expert == 1).unwrap();
        assert!((e1.importance - 1.0).abs() < 1e-6, "expected ~1.0, got {}", e1.importance);

        let _ = std::fs::remove_dir_all(root);
    }
}
