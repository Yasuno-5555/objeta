use crate::qwen36_forward::{Qwen36Runner, RuntimeConfigSource, EXPERT_TOTAL_BYTES};

impl Qwen36Runner {
    pub fn begin_token_residency(&mut self, token_id: usize) {
        self.active_token_id = Some(token_id);
        let token_id_i32 = token_id as i32;
        let model_predicted_this_token =
            self.last_predicted_token_id >= 0 && self.last_predicted_token_id == token_id_i32;
        if self.decode_started || model_predicted_this_token {
            self.decode_started = true;
            self.decode_token_count = self.decode_token_count.saturating_add(1);
            self.current_governor_phase = if self.decode_token_count <= 4 {
                crate::runtime_governor::GovernorPhase::DecodeWarmup
            } else {
                crate::runtime_governor::GovernorPhase::DecodeSteady
            };
        } else {
            self.current_governor_phase = crate::runtime_governor::GovernorPhase::Prefill;
        }
        self.expert_residency_manager
            .begin_token_residency(token_id);
    }

    pub fn init_page_cache(&mut self, capacity_bytes: u64) {
        let existing_priorities: Vec<_> = self
            .expert_residency_manager
            .expert_priorities
            .values()
            .cloned()
            .collect();
        self.expert_residency_manager =
            crate::expert_cache::ExpertResidencyManager::new(capacity_bytes);
        if !existing_priorities.is_empty() {
            self.expert_residency_manager
                .load_expert_priorities(existing_priorities);
        }
        self.expert_cache_size = (capacity_bytes / EXPERT_TOTAL_BYTES) as usize;
        self.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
    }

    pub fn update_cache_capacity(&mut self, new_capacity_bytes: u64) {
        self.expert_residency_manager.capacity_bytes = new_capacity_bytes;
        self.expert_cache_size = (new_capacity_bytes / EXPERT_TOTAL_BYTES) as usize;
    }
}
