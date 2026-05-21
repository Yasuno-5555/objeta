use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExpertPageKey {
    pub layer_id: usize,
    pub expert_id: usize,
    pub precision: u8, // 4 = Q4
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExpertPageMeta {
    pub bytes: usize,
    pub last_used_token: usize,
    pub use_count: usize,
    pub ema_gate: f32,
    pub load_count: usize,
}

impl ExpertPageMeta {
    pub fn eviction_score(&self) -> f32 {
        self.last_used_token as f32
    }
}

#[derive(Clone)]
pub struct ExpertPage {
    pub gate_up_bytes: Arc<[u8]>,
    pub down_bytes: Arc<[u8]>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertResidencyDebugTiming {
    pub cache_hit_lookup_sec: f64,
    pub cache_miss_load_sec: f64,
    pub cache_eviction_sec: f64,
    pub cache_insert_sec: f64,
    pub cache_page_clone_sec: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertTier {
    Hot,
    Warm,
    Cold,
}

impl ExpertTier {
    fn retention_rank(self) -> u8 {
        match self {
            ExpertTier::Cold => 0,
            ExpertTier::Warm => 2,
            ExpertTier::Hot => 3,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExpertPriority {
    pub layer_idx: usize,
    pub expert_id: usize,
    pub eviction_priority: f64,
    pub tier: ExpertTier,
    pub importance: f64,
    pub selected_count: u64,
    pub avg_gate_weight: f64,
}

pub struct ExpertResidencyManager {
    pub capacity_bytes: u64,
    pub resident: HashMap<ExpertPageKey, ExpertPage>,
    pub metadata: HashMap<ExpertPageKey, ExpertPageMeta>,
    pub pinned: HashSet<ExpertPageKey>,
    pub group_pinned: HashSet<ExpertPageKey>,
    pub token_used: HashSet<ExpertPageKey>,
    pub token_window_active: bool,
    pub current_token_id: Option<usize>,
    pub soft_capacity_factor: f32,
    // Statistics
    pub logical_expert_bytes_requested: u64,
    /// actual_expert_bytes_loaded = resident miss 時に mmap/SSD-backed source から resident Arc<[u8]> へコピーしたQ4 bytes
    pub actual_expert_bytes_loaded: u64,
    pub resident_cache_bytes_reused: u64,
    pub direct_cold_load_count: u64,
    pub resident_hit_count: u64,
    pub resident_miss_count: u64,
    pub eviction_count: u64,
    pub token_window_peak_resident_bytes: u64,
    pub total_token_window_peak_resident_bytes: u64,
    pub completed_token_windows: u64,
    pub eviction_count_during_token: u64,
    pub eviction_count_at_token_end: u64,
    pub importance_eviction_enabled: bool,
    pub evicted_hot_count: u64,
    pub evicted_warm_count: u64,
    pub evicted_cold_count: u64,
    pub evicted_unknown_count: u64,
    pub expert_eviction_policy: String,
    // Group pre-resolve telemetry
    pub residency_group_size: usize,
    pub group_preresolve_wall_ms: f64,
    pub group_pinned_bytes: u64,
    pub group_preloaded_expert_count: u64,
    pub group_cache_miss_count: u64,
    // Selective and budgeted telemetry
    pub group_preresolve_skipped_by_budget: u64,
    pub group_preresolve_requested_bytes: u64,
    pub group_preresolve_loaded_bytes: u64,
    pub group_preresolve_hit_rate: f64,
    pub expert_priorities: HashMap<ExpertPageKey, ExpertPriority>,
}

impl ExpertResidencyManager {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            resident: HashMap::new(),
            metadata: HashMap::new(),
            pinned: HashSet::new(),
            group_pinned: HashSet::new(),
            token_used: HashSet::new(),
            token_window_active: false,
            current_token_id: None,
            soft_capacity_factor: 1.10,
            logical_expert_bytes_requested: 0,
            actual_expert_bytes_loaded: 0,
            resident_cache_bytes_reused: 0,
            direct_cold_load_count: 0,
            resident_hit_count: 0,
            resident_miss_count: 0,
            eviction_count: 0,
            token_window_peak_resident_bytes: 0,
            total_token_window_peak_resident_bytes: 0,
            completed_token_windows: 0,
            eviction_count_during_token: 0,
            eviction_count_at_token_end: 0,
            importance_eviction_enabled: false,
            evicted_hot_count: 0,
            evicted_warm_count: 0,
            evicted_cold_count: 0,
            evicted_unknown_count: 0,
            expert_eviction_policy: "lru".to_string(),
            residency_group_size: 1,
            group_preresolve_wall_ms: 0.0,
            group_pinned_bytes: 0,
            group_preloaded_expert_count: 0,
            group_cache_miss_count: 0,
            group_preresolve_skipped_by_budget: 0,
            group_preresolve_requested_bytes: 0,
            group_preresolve_loaded_bytes: 0,
            group_preresolve_hit_rate: 0.0,
            expert_priorities: HashMap::new(),
        }
    }

    pub fn load_expert_priorities(&mut self, priorities: Vec<ExpertPriority>) {
        self.expert_priorities = priorities
            .into_iter()
            .map(|priority| {
                (
                    ExpertPageKey {
                        layer_id: priority.layer_idx,
                        expert_id: priority.expert_id,
                        precision: 4,
                    },
                    priority,
                )
            })
            .collect();
        self.importance_eviction_enabled = !self.expert_priorities.is_empty();
        self.expert_eviction_policy = if self.importance_eviction_enabled {
            "importance_lru".to_string()
        } else {
            "lru".to_string()
        };
    }

    fn classify_tier_for_key(&self, key: &ExpertPageKey) -> Option<ExpertTier> {
        self.expert_priorities.get(key).map(|p| p.tier)
    }

    fn eviction_candidate_priority(
        &self,
        key: &ExpertPageKey,
        meta: &ExpertPageMeta,
    ) -> (u8, u64, usize, usize) {
        if let Some(priority) = self.expert_priorities.get(key) {
            (
                priority.tier.retention_rank(),
                (priority.importance * 1_000_000.0) as u64,
                meta.last_used_token,
                key.expert_id,
            )
        } else {
            (1, 0, meta.last_used_token, key.expert_id)
        }
    }

    fn record_evicted_tier(&mut self, key: &ExpertPageKey) {
        match self.classify_tier_for_key(key) {
            Some(ExpertTier::Hot) => self.evicted_hot_count += 1,
            Some(ExpertTier::Warm) => self.evicted_warm_count += 1,
            Some(ExpertTier::Cold) => self.evicted_cold_count += 1,
            None => self.evicted_unknown_count += 1,
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.metadata.values().map(|meta| meta.bytes as u64).sum()
    }

    pub fn is_bypass(&self) -> bool {
        self.capacity_bytes == 0
    }

    fn effective_capacity_bytes(&self) -> u64 {
        if self.token_window_active {
            ((self.capacity_bytes as f64) * self.soft_capacity_factor as f64).ceil() as u64
        } else {
            self.capacity_bytes
        }
    }

    fn update_peak_resident_bytes(&mut self) {
        if self.token_window_active {
            self.token_window_peak_resident_bytes = self
                .token_window_peak_resident_bytes
                .max(self.resident_bytes());
        }
    }

    pub fn begin_token_residency(&mut self, token_id: usize) {
        self.token_window_active = true;
        self.current_token_id = Some(token_id);
        self.pinned.clear();
        self.group_pinned.clear();
        self.token_used.clear();
        self.token_window_peak_resident_bytes = self.resident_bytes();
    }

    pub fn end_token_residency(&mut self) {
        if !self.token_window_active {
            return;
        }
        self.token_window_active = false;
        self.current_token_id = None;
        self.pinned.clear();
        self.group_pinned.clear();
        self.token_used.clear();
        let prev_evictions = self.eviction_count;
        self.evict_until_target(0, self.capacity_bytes, false);
        self.eviction_count_at_token_end += self.eviction_count - prev_evictions;
        self.total_token_window_peak_resident_bytes += self.token_window_peak_resident_bytes;
        self.completed_token_windows += 1;
    }

    pub fn pinned_resident_bytes(&self) -> u64 {
        self.pinned
            .iter()
            .filter_map(|k| self.metadata.get(k))
            .map(|meta| meta.bytes as u64)
            .sum()
    }

    fn evict_until_target(&mut self, needed_bytes: u64, capacity_limit: u64, respect_pins: bool) {
        if self.is_bypass() {
            return;
        }
        while self.resident_bytes() + needed_bytes > capacity_limit {
            let min_key: Option<ExpertPageKey> = if self.importance_eviction_enabled {
                self.metadata
                    .iter()
                    .filter(|(key, _)| !(respect_pins && self.pinned.contains(key)))
                    .min_by(|(ka, ma), (kb, mb)| {
                        self.eviction_candidate_priority(ka, ma)
                            .cmp(&self.eviction_candidate_priority(kb, mb))
                    })
                    .map(|(key, _)| *key)
            } else {
                self.metadata
                    .iter()
                    .filter(|(key, _)| !(respect_pins && self.pinned.contains(key)))
                    .min_by(|(_, ma), (_, mb)| {
                        ma.last_used_token
                            .cmp(&mb.last_used_token)
                            .then_with(|| ma.use_count.cmp(&mb.use_count))
                    })
                    .map(|(key, _)| *key)
            };
            if let Some(k) = min_key {
                self.record_evicted_tier(&k);
                self.resident.remove(&k);
                self.metadata.remove(&k);
                self.eviction_count += 1;
                if self.token_window_active {
                    self.eviction_count_during_token += 1;
                }
            } else {
                break;
            }
        }
    }

    pub fn evict_until_fit(&mut self, needed_bytes: u64) {
        self.evict_until_target(needed_bytes, self.effective_capacity_bytes(), true);
    }

    /// Pre-resolve predicted experts for a group of layers.
    /// Loads + pins expert pages into the resident cache so that subsequent
    /// `ensure_resident` calls hit warm. Does not change model math —
    /// the router still selects exact experts at execution time.
    ///
    /// Returns (preloaded_count, miss_count).
    pub fn pre_resolve_group(
        &mut self,
        group_layers: &[(usize, &[usize])], // (layer_idx, predicted_expert_ids)
        gate_up_mmaps: &[&[u8]],
        down_mmaps: &[&[u8]],
        token_id: usize,
        expert_total_bytes: usize,
    ) -> (u64, u64) {
        if self.is_bypass() {
            return (0, 0);
        }
        self.group_pinned.clear();
        let mut preloaded = 0u64;
        let mut misses = 0u64;
        let expert_total_bytes_u64 = expert_total_bytes as u64;

        // Parse pre-resolve max bytes budget
        let max_bytes = std::env::var("OBJETA_GROUP_PRERESOLVE_MAX_BYTES")
            .ok()
            .and_then(|v| {
                let s = v.trim().to_lowercase();
                if s.ends_with("gb") || s.ends_with("g") {
                    let val: f64 = s
                        .trim_end_matches("gb")
                        .trim_end_matches("g")
                        .trim()
                        .parse()
                        .ok()?;
                    Some((val * 1024.0 * 1024.0 * 1024.0) as u64)
                } else if s.ends_with("mb") || s.ends_with("m") {
                    let val: f64 = s
                        .trim_end_matches("mb")
                        .trim_end_matches("m")
                        .trim()
                        .parse()
                        .ok()?;
                    Some((val * 1024.0 * 1024.0) as u64)
                } else if s.ends_with("kb") || s.ends_with("k") {
                    let val: f64 = s
                        .trim_end_matches("kb")
                        .trim_end_matches("k")
                        .trim()
                        .parse()
                        .ok()?;
                    Some((val * 1024.0) as u64)
                } else {
                    s.parse().ok()
                }
            })
            .unwrap_or(256 * 1024 * 1024); // default 256MB

        let mut budget_exceeded = false;

        for (idx, (layer_idx, expert_ids)) in group_layers.iter().enumerate() {
            let gu_slice = gate_up_mmaps.get(idx).copied().unwrap_or(&[]);
            let d_slice = down_mmaps.get(idx).copied().unwrap_or(&[]);
            for &eid in *expert_ids {
                self.group_preresolve_requested_bytes += expert_total_bytes_u64;

                if budget_exceeded {
                    self.group_preresolve_skipped_by_budget += 1;
                    continue;
                }

                // Check budget
                let current_group_pinned_bytes =
                    (self.group_pinned.len() + 1) as u64 * expert_total_bytes_u64;
                if current_group_pinned_bytes > max_bytes {
                    budget_exceeded = true;
                    self.group_preresolve_skipped_by_budget += 1;
                    continue;
                }

                let key = ExpertPageKey {
                    layer_id: *layer_idx,
                    expert_id: eid,
                    precision: 4,
                };
                if self.resident.contains_key(&key) {
                    // Already resident — just pin
                    self.pinned.insert(key);
                    self.group_pinned.insert(key);
                    preloaded += 1;
                    continue;
                }
                misses += 1;
                // Load + pin
                let gu_off = eid * crate::moe_dispatch::GU_EXPERT_BYTES;
                let d_off = eid * crate::moe_dispatch::D_EXPERT_BYTES;
                let page = ExpertPage {
                    gate_up_bytes: Arc::from(
                        &gu_slice[gu_off..gu_off + crate::moe_dispatch::GU_EXPERT_BYTES],
                    ),
                    down_bytes: Arc::from(
                        &d_slice[d_off..d_off + crate::moe_dispatch::D_EXPERT_BYTES],
                    ),
                };
                let evict_start = std::time::Instant::now();
                self.evict_until_fit(expert_total_bytes_u64);
                let _evict_sec = evict_start.elapsed().as_secs_f64();

                self.resident.insert(key, page.clone());
                self.metadata.insert(
                    key,
                    ExpertPageMeta {
                        bytes: expert_total_bytes,
                        last_used_token: token_id,
                        use_count: 1,
                        ema_gate: 0.0,
                        load_count: 1,
                    },
                );
                self.pinned.insert(key);
                self.group_pinned.insert(key);
                self.actual_expert_bytes_loaded += expert_total_bytes_u64;
                self.group_preresolve_loaded_bytes += expert_total_bytes_u64;
                preloaded += 1;
            }
        }
        self.group_preloaded_expert_count += preloaded;
        self.group_cache_miss_count += misses;
        self.group_pinned_bytes = self.group_pinned_bytes.max(self.pinned_resident_bytes());

        if self.group_preloaded_expert_count > 0 {
            let hits = self
                .group_preloaded_expert_count
                .saturating_sub(self.group_cache_miss_count);
            self.group_preresolve_hit_rate = hits as f64 / self.group_preloaded_expert_count as f64;
        } else {
            self.group_preresolve_hit_rate = 0.0;
        }

        (preloaded, misses)
    }

    /// Unpin group pages while keeping token-window pins.
    /// Token-scoped residency window remains active.
    pub fn unpin_group(&mut self) {
        for key in &self.group_pinned {
            if !self.token_used.contains(key) {
                self.pinned.remove(key);
            }
        }
        self.group_pinned.clear();
        self.update_peak_resident_bytes();
    }

    pub fn ensure_resident<F>(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        precision: u8,
        bytes: usize,
        current_token: usize,
        ema_gate: f32,
        load_fn: F,
    ) -> ExpertPage
    where
        F: FnOnce() -> ExpertPage,
    {
        self.ensure_resident_profiled(
            layer_id,
            expert_id,
            precision,
            bytes,
            current_token,
            ema_gate,
            load_fn,
        )
        .0
    }

    pub fn ensure_resident_profiled<F>(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        precision: u8,
        bytes: usize,
        current_token: usize,
        ema_gate: f32,
        load_fn: F,
    ) -> (ExpertPage, ExpertResidencyDebugTiming)
    where
        F: FnOnce() -> ExpertPage,
    {
        let key = ExpertPageKey {
            layer_id,
            expert_id,
            precision,
        };
        let mut timing = ExpertResidencyDebugTiming::default();

        self.logical_expert_bytes_requested += bytes as u64;

        if self.is_bypass() {
            self.resident_miss_count += 1;
            self.direct_cold_load_count += 1;
            let load_start = Instant::now();
            let page = load_fn();
            timing.cache_miss_load_sec = load_start.elapsed().as_secs_f64();
            self.actual_expert_bytes_loaded += bytes as u64;
            return (page, timing);
        }

        let lookup_start = Instant::now();
        if let Some(page_ref) = self.resident.get(&key) {
            timing.cache_hit_lookup_sec = lookup_start.elapsed().as_secs_f64();
            // Hit!
            self.resident_hit_count += 1;
            self.resident_cache_bytes_reused += bytes as u64;

            if let Some(meta) = self.metadata.get_mut(&key) {
                meta.last_used_token = current_token;
                meta.use_count += 1;
                meta.ema_gate = ema_gate;
            }
            let clone_start = Instant::now();
            let page = page_ref.clone();
            timing.cache_page_clone_sec = clone_start.elapsed().as_secs_f64();
            if self.token_window_active {
                self.pinned.insert(key);
                self.token_used.insert(key);
                self.update_peak_resident_bytes();
            }
            (page, timing)
        } else {
            timing.cache_hit_lookup_sec = lookup_start.elapsed().as_secs_f64();
            // Miss!
            self.resident_miss_count += 1;
            self.direct_cold_load_count += 1;

            let evict_start = Instant::now();
            self.evict_until_fit(bytes as u64);
            timing.cache_eviction_sec = evict_start.elapsed().as_secs_f64();

            let load_start = Instant::now();
            let entry = load_fn();
            timing.cache_miss_load_sec = load_start.elapsed().as_secs_f64();
            self.actual_expert_bytes_loaded += bytes as u64;

            let insert_start = Instant::now();
            self.resident.insert(key, entry.clone());
            self.metadata.insert(
                key,
                ExpertPageMeta {
                    bytes,
                    last_used_token: current_token,
                    use_count: 1,
                    ema_gate,
                    load_count: 1,
                },
            );
            timing.cache_insert_sec = insert_start.elapsed().as_secs_f64();
            if self.token_window_active {
                self.pinned.insert(key);
                self.token_used.insert(key);
                self.update_peak_resident_bytes();
            }

            (entry, timing)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn test_bypass() {
        let mut manager = ExpertResidencyManager::new(0);
        assert!(manager.is_bypass());
        assert_eq!(manager.resident_bytes(), 0);

        let page = manager.ensure_resident(0, 0, 4, 100, 1, 0.5, || ExpertPage {
            gate_up_bytes: Arc::from(vec![1, 2, 3]),
            down_bytes: Arc::from(vec![4, 5]),
        });
        assert_eq!(page.gate_up_bytes.len(), 3);
        assert_eq!(manager.resident_bytes(), 0);
        assert_eq!(manager.logical_expert_bytes_requested, 100);
        assert_eq!(manager.actual_expert_bytes_loaded, 100);
        assert_eq!(manager.resident_hit_count, 0);
        assert_eq!(manager.resident_miss_count, 1);
    }

    #[test]
    fn test_hit_miss_eviction() {
        // Capacity for 2 experts of size 100 bytes each
        let mut manager = ExpertResidencyManager::new(200);
        assert!(!manager.is_bypass());

        // 1st expert load: Miss
        let _page1 = manager.ensure_resident(0, 1, 4, 100, 1, 0.8, || ExpertPage {
            gate_up_bytes: Arc::from(vec![1]),
            down_bytes: Arc::from(vec![2]),
        });
        assert_eq!(manager.resident_bytes(), 100);
        assert_eq!(manager.resident_hit_count, 0);
        assert_eq!(manager.resident_miss_count, 1);
        assert_eq!(manager.actual_expert_bytes_loaded, 100);

        // 2nd expert load: Miss
        let _page2 = manager.ensure_resident(0, 2, 4, 100, 2, 0.9, || ExpertPage {
            gate_up_bytes: Arc::from(vec![3]),
            down_bytes: Arc::from(vec![4]),
        });
        assert_eq!(manager.resident_bytes(), 200);
        assert_eq!(manager.resident_hit_count, 0);
        assert_eq!(manager.resident_miss_count, 2);

        // 1st expert Hit (last_used_token updated to 3)
        let _page1_hit = manager.ensure_resident(0, 1, 4, 100, 3, 0.8, || unreachable!());
        assert_eq!(manager.resident_hit_count, 1);
        assert_eq!(manager.resident_miss_count, 2);
        assert_eq!(manager.resident_bytes(), 200);

        // 3rd expert load: Miss -> causes eviction of 2nd expert (last_used_token = 2, while 1st is 3)
        let _page3 = manager.ensure_resident(0, 3, 4, 100, 4, 0.7, || ExpertPage {
            gate_up_bytes: Arc::from(vec![5]),
            down_bytes: Arc::from(vec![6]),
        });
        assert_eq!(manager.resident_bytes(), 200);
        assert_eq!(manager.resident_hit_count, 1);
        assert_eq!(manager.resident_miss_count, 3);
        assert_eq!(manager.eviction_count, 1);

        // Verify that 2nd expert is indeed evicted and 1st and 3rd are resident
        let key1 = ExpertPageKey {
            layer_id: 0,
            expert_id: 1,
            precision: 4,
        };
        let key2 = ExpertPageKey {
            layer_id: 0,
            expert_id: 2,
            precision: 4,
        };
        let key3 = ExpertPageKey {
            layer_id: 0,
            expert_id: 3,
            precision: 4,
        };
        assert!(manager.resident.contains_key(&key1));
        assert!(!manager.resident.contains_key(&key2));
        assert!(manager.resident.contains_key(&key3));
    }

    #[test]
    fn test_token_window_pins_and_soft_limit() {
        let mut manager = ExpertResidencyManager::new(290);
        manager.soft_capacity_factor = 1.10;
        manager.begin_token_residency(7);

        let _page1 = manager.ensure_resident(0, 1, 4, 100, 7, 0.8, || ExpertPage {
            gate_up_bytes: Arc::from(vec![1]),
            down_bytes: Arc::from(vec![2]),
        });
        let _page2 = manager.ensure_resident(0, 2, 4, 100, 7, 0.7, || ExpertPage {
            gate_up_bytes: Arc::from(vec![3]),
            down_bytes: Arc::from(vec![4]),
        });
        let _page3 = manager.ensure_resident(0, 3, 4, 100, 7, 0.6, || ExpertPage {
            gate_up_bytes: Arc::from(vec![5]),
            down_bytes: Arc::from(vec![6]),
        });

        assert!(manager.resident_bytes() > manager.capacity_bytes);
        assert!(manager.resident_bytes() <= manager.effective_capacity_bytes());
        assert_eq!(manager.eviction_count_during_token, 0);

        manager.end_token_residency();
        assert!(manager.resident_bytes() <= manager.capacity_bytes);
        assert!(manager.eviction_count_at_token_end >= 1);
    }

    #[test]
    fn test_group_pin_and_unpin() {
        let _guard = env_lock().lock().unwrap();
        let mut manager = ExpertResidencyManager::new(300);
        manager.soft_capacity_factor = 1.5;
        manager.begin_token_residency(42);
        std::env::remove_var("OBJETA_GROUP_PRERESOLVE_MAX_BYTES");

        // We mock expert total bytes = 100.
        // For pre-resolve we need some source bytes representing experts.
        // Each expert has size GU_EXPERT_BYTES + D_EXPERT_BYTES.
        let mock_gu = vec![0u8; 6 * crate::moe_dispatch::GU_EXPERT_BYTES];
        let mock_d = vec![0u8; 6 * crate::moe_dispatch::D_EXPERT_BYTES];
        let gu_slice = &mock_gu[..];
        let d_slice = &mock_d[..];

        // 1. Pre-resolve group: layers [0, 1] with predicted experts
        let pred0 = vec![1, 2];
        let pred1 = vec![3];
        let group_layers = vec![(0, pred0.as_slice()), (1, pred1.as_slice())];

        let (preloaded, misses) = manager.pre_resolve_group(
            &group_layers,
            &[gu_slice, gu_slice],
            &[d_slice, d_slice],
            42,
            100,
        );
        assert_eq!(preloaded, 3);
        assert_eq!(misses, 3);

        let k0_1 = ExpertPageKey {
            layer_id: 0,
            expert_id: 1,
            precision: 4,
        };
        let k0_2 = ExpertPageKey {
            layer_id: 0,
            expert_id: 2,
            precision: 4,
        };
        let k1_3 = ExpertPageKey {
            layer_id: 1,
            expert_id: 3,
            precision: 4,
        };

        // All 3 should be pinned and group_pinned
        assert!(manager.pinned.contains(&k0_1));
        assert!(manager.pinned.contains(&k0_2));
        assert!(manager.pinned.contains(&k1_3));
        assert!(manager.group_pinned.contains(&k0_1));
        assert!(manager.group_pinned.contains(&k0_2));
        assert!(manager.group_pinned.contains(&k1_3));

        // 2. Execution phase of the group
        // Expert (0, 1) is actually used
        let _p0_1 = manager.ensure_resident(0, 1, 4, 100, 42, 0.9, || unreachable!());
        // A non-predicted expert (0, 5) is also used
        let _p0_5 = manager.ensure_resident(0, 5, 4, 100, 42, 0.9, || ExpertPage {
            gate_up_bytes: Arc::from(vec![0]),
            down_bytes: Arc::from(vec![0]),
        });

        let k0_5 = ExpertPageKey {
            layer_id: 0,
            expert_id: 5,
            precision: 4,
        };

        // 3. Unpin group
        manager.unpin_group();

        // (0, 1) was predicted AND used -> should remain pinned
        assert!(manager.pinned.contains(&k0_1));
        // (0, 5) was not predicted but used -> should remain pinned
        assert!(manager.pinned.contains(&k0_5));
        // (0, 2) was predicted but NOT used -> should be unpinned
        assert!(!manager.pinned.contains(&k0_2));
        // (1, 3) was predicted but NOT used -> should be unpinned
        assert!(!manager.pinned.contains(&k1_3));

        // group_pinned should be cleared
        assert!(manager.group_pinned.is_empty());

        // 4. End token residency
        manager.end_token_residency();
        // Eviction should have run down to hard capacity (300)
        assert!(manager.resident_bytes() <= 300);
        assert!(!manager.pinned.contains(&k0_1));
        assert!(!manager.pinned.contains(&k0_5));
        std::env::remove_var("OBJETA_GROUP_PRERESOLVE_MAX_BYTES");
    }

    #[test]
    fn test_group_preresolve_budget_guard() {
        let _guard = env_lock().lock().unwrap();
        // Budget = 250 bytes, expert_total_bytes = 100 bytes.
        // Capacity big enough to hold many experts to isolate budget from capacity eviction.
        let mut manager = ExpertResidencyManager::new(100_000);
        manager.begin_token_residency(1);

        // Reset new telemetry fields before the test
        manager.group_preresolve_skipped_by_budget = 0;
        manager.group_preresolve_requested_bytes = 0;
        manager.group_preresolve_loaded_bytes = 0;
        manager.group_preresolve_hit_rate = 0.0;
        manager.group_preloaded_expert_count = 0;
        manager.group_cache_miss_count = 0;

        // Budget: allow at most 2 experts (200 bytes). Third expert would push to 300 > 250.
        // We set OBJETA_GROUP_PRERESOLVE_MAX_BYTES to 250 via mock.
        // Instead, we test the budget logic by providing enough experts to exceed budget.
        // expert_total_bytes=100, budget check: (group_pinned.len() + 1) * 100 > budget
        // With budget=250: expert 1 => 100 <= 250 OK, expert 2 => 200 <= 250 OK,
        //                  expert 3 => 300 > 250 SKIP.
        // We will set the env var for this test.
        std::env::set_var("OBJETA_GROUP_PRERESOLVE_MAX_BYTES", "250");

        let mock_gu = vec![0u8; 6 * crate::moe_dispatch::GU_EXPERT_BYTES];
        let mock_d = vec![0u8; 6 * crate::moe_dispatch::D_EXPERT_BYTES];
        let gu_slice = &mock_gu[..];
        let d_slice = &mock_d[..];

        // 3 experts across 2 layers: experts [0, 1] in layer 0, expert [2] in layer 1.
        // With expert_total_bytes=100 and budget=250: experts 0 and 1 fit, expert 2 is skipped.
        let pred0 = vec![0usize, 1usize];
        let pred1 = vec![2usize];
        let group_layers: Vec<(usize, &[usize])> =
            vec![(0, pred0.as_slice()), (1, pred1.as_slice())];

        let (preloaded, misses) = manager.pre_resolve_group(
            &group_layers,
            &[gu_slice, gu_slice],
            &[d_slice, d_slice],
            1,
            100, // expert_total_bytes
        );

        // Clean up env var to not affect other tests
        std::env::remove_var("OBJETA_GROUP_PRERESOLVE_MAX_BYTES");

        // 2 experts fit under budget (200 <= 250), 1 skipped (300 > 250)
        assert_eq!(preloaded, 2, "Expected 2 experts preloaded within budget");
        assert_eq!(misses, 2, "Expected 2 cache misses (cold load)");
        assert_eq!(
            manager.group_preresolve_skipped_by_budget, 1,
            "Expected 1 expert skipped by budget"
        );

        // requested = 3 * 100 = 300
        assert_eq!(
            manager.group_preresolve_requested_bytes, 300,
            "Expected 300 requested bytes"
        );
        // loaded = 2 * 100 = 200 (only the 2 that fit in budget)
        assert_eq!(
            manager.group_preresolve_loaded_bytes, 200,
            "Expected 200 loaded bytes"
        );

        // hit_rate: 0 hits out of 2 preloaded (all were cold misses)
        assert_eq!(
            manager.group_preresolve_hit_rate, 0.0,
            "Expected 0.0 hit rate on cold load"
        );

        manager.end_token_residency();
    }

    fn insert_resident(
        manager: &mut ExpertResidencyManager,
        layer_id: usize,
        expert_id: usize,
        bytes: usize,
        last_used_token: usize,
    ) -> ExpertPageKey {
        let key = ExpertPageKey {
            layer_id,
            expert_id,
            precision: 4,
        };
        manager.resident.insert(
            key,
            ExpertPage {
                gate_up_bytes: Arc::from(vec![0u8; 1]),
                down_bytes: Arc::from(vec![0u8; 1]),
            },
        );
        manager.metadata.insert(
            key,
            ExpertPageMeta {
                bytes,
                last_used_token,
                use_count: 1,
                ema_gate: 0.0,
                load_count: 1,
            },
        );
        key
    }

    #[test]
    fn cold_recent_page_evicted_before_hot_old_page() {
        let mut manager = ExpertResidencyManager::new(200);
        let hot = insert_resident(&mut manager, 0, 1, 100, 1);
        let cold = insert_resident(&mut manager, 0, 2, 100, 99);
        manager.load_expert_priorities(vec![
            ExpertPriority {
                layer_idx: 0,
                expert_id: 1,
                eviction_priority: 0.9,
                tier: ExpertTier::Hot,
                importance: 0.95,
                selected_count: 10,
                avg_gate_weight: 0.3,
            },
            ExpertPriority {
                layer_idx: 0,
                expert_id: 2,
                eviction_priority: 0.1,
                tier: ExpertTier::Cold,
                importance: 0.05,
                selected_count: 1,
                avg_gate_weight: 0.1,
            },
        ]);
        manager.evict_until_fit(100);
        assert!(manager.resident.contains_key(&hot));
        assert!(!manager.resident.contains_key(&cold));
        assert_eq!(manager.evicted_cold_count, 1);
    }

    #[test]
    fn warm_evicted_before_hot_when_both_fit_candidates() {
        let mut manager = ExpertResidencyManager::new(200);
        let hot = insert_resident(&mut manager, 0, 1, 100, 1);
        let warm = insert_resident(&mut manager, 0, 2, 100, 2);
        manager.load_expert_priorities(vec![
            ExpertPriority {
                layer_idx: 0,
                expert_id: 1,
                eviction_priority: 0.9,
                tier: ExpertTier::Hot,
                importance: 0.95,
                selected_count: 10,
                avg_gate_weight: 0.3,
            },
            ExpertPriority {
                layer_idx: 0,
                expert_id: 2,
                eviction_priority: 0.5,
                tier: ExpertTier::Warm,
                importance: 0.40,
                selected_count: 5,
                avg_gate_weight: 0.2,
            },
        ]);
        manager.evict_until_fit(100);
        assert!(manager.resident.contains_key(&hot));
        assert!(!manager.resident.contains_key(&warm));
        assert_eq!(manager.evicted_warm_count, 1);
    }

    #[test]
    fn fallback_pure_lru_when_no_priorities_loaded() {
        let mut manager = ExpertResidencyManager::new(200);
        let old = insert_resident(&mut manager, 0, 1, 100, 1);
        let recent = insert_resident(&mut manager, 0, 2, 100, 5);
        manager.evict_until_fit(100);
        assert!(!manager.resident.contains_key(&old));
        assert!(manager.resident.contains_key(&recent));
        assert_eq!(manager.evicted_unknown_count, 1);
    }

    #[test]
    fn unknown_evicted_after_cold_but_before_warm_and_hot() {
        let mut manager = ExpertResidencyManager::new(300);
        let cold = insert_resident(&mut manager, 0, 1, 100, 10);
        let unknown = insert_resident(&mut manager, 0, 2, 100, 1);
        let warm = insert_resident(&mut manager, 0, 3, 100, 0);
        manager.load_expert_priorities(vec![
            ExpertPriority {
                layer_idx: 0,
                expert_id: 1,
                eviction_priority: 0.1,
                tier: ExpertTier::Cold,
                importance: 0.05,
                selected_count: 1,
                avg_gate_weight: 0.1,
            },
            ExpertPriority {
                layer_idx: 0,
                expert_id: 3,
                eviction_priority: 0.5,
                tier: ExpertTier::Warm,
                importance: 0.5,
                selected_count: 3,
                avg_gate_weight: 0.2,
            },
        ]);
        manager.evict_until_fit(100);
        assert!(!manager.resident.contains_key(&cold));
        assert!(manager.resident.contains_key(&unknown));
        assert!(manager.resident.contains_key(&warm));
        manager.evict_until_fit(101);
        assert!(!manager.resident.contains_key(&unknown));
        assert!(manager.resident.contains_key(&warm));
        assert_eq!(manager.evicted_cold_count, 1);
        assert_eq!(manager.evicted_unknown_count, 1);
    }
}
