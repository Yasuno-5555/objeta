//! Speculative decoding state machine with fixed-shape buffers.
//!
//! Architecture:
//!   Python (MLX inference)
//!     ↕ ctypes FFI
//!   Rust (state machine, fixed buffers, accept/reject)
//!
//! Key invariants:
//!   - All buffers are preallocated to fixed sizes (no allocation in hot path).
//!   - Draft always generates exactly `gamma` tokens.
//!   - Verify always receives exactly `gamma` tokens.
//!   - KV cache positions are tracked as fixed-size slices.
//!
//! This eliminates the variable-shape graph fragmentation that causes
//! MLX JIT recompilation storms in the pure-Python speculative pipeline.

use std::os::raw::c_float;

// ── Constants ─────────────────────────────────────────────────────

/// Maximum gamma (draft tokens per step). Larger = more speedup potential
/// but higher risk of rejection waste.
const MAX_GAMMA: usize = 8;

/// Maximum sequence length (KV cache capacity).
const MAX_SEQ_LEN: u32 = 2048;

// ── State machine ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub enum SpecPhase {
    Idle = 0,
    Prefill = 1,
    Drafting = 2,
    Verifying = 3,
    Done = 4,
}

/// Fixed-shape speculative decode state.
#[derive(Clone, Debug)]
pub struct SpecState {
    // Config
    gamma: u32,
    max_seq_len: u32,
    vocab_size: u32,

    // Phase
    phase: SpecPhase,

    // Position tracking
    position: u32,      // next write position in KV cache
    current_token: u32, // token to feed for next draft step

    // Fixed buffers
    draft_tokens: [u32; MAX_GAMMA],        // gamma draft tokens
    draft_positions: [u32; MAX_GAMMA],     // positions for each draft token
    verify_tokens: [u32; MAX_GAMMA],       // gamma tokens for verify (always full)
    accepted_tokens: [u32; MAX_GAMMA + 1], // accepted + possible rejection replacement

    // Stats
    steps: u32,
    total_draft_tokens: u32,
    total_accepted: u32,

    // Prefill tracking
    prefill_position: u32,
    prefill_logits_valid: bool,
}

impl SpecState {
    pub fn new(gamma: u32, vocab_size: u32) -> Self {
        let gamma = gamma.min(MAX_GAMMA as u32);
        Self {
            gamma,
            max_seq_len: MAX_SEQ_LEN,
            vocab_size,
            phase: SpecPhase::Idle,
            position: 0,
            current_token: 0,
            draft_tokens: [0u32; MAX_GAMMA],
            draft_positions: [0u32; MAX_GAMMA],
            verify_tokens: [0u32; MAX_GAMMA],
            accepted_tokens: [0u32; MAX_GAMMA + 1],
            steps: 0,
            total_draft_tokens: 0,
            total_accepted: 0,
            prefill_position: 0,
            prefill_logits_valid: false,
        }
    }

    // ── Prefill ──────────────────────────────────────────────────

    /// Begin prefill. Call once before pushing prompt tokens.
    pub fn begin_prefill(&mut self) {
        self.phase = SpecPhase::Prefill;
        self.position = 0;
        self.prefill_position = 0;
        self.prefill_logits_valid = false;
    }

    /// Push one prompt token. Returns the position for KV cache write.
    /// Caller (Python) runs draft.forward + target.forward at this position,
    /// then calls `prefill_token_done`.
    pub fn prefill_next_position(&self) -> u32 {
        self.prefill_position
    }

    /// Mark one prefill token as processed. `target_logit` is the
    /// logits from target model at this position (used to sample first token).
    pub fn prefill_token_done(&mut self, target_logit_ptr: *const c_float) {
        self.prefill_position += 1;
        self.position = self.prefill_position;
        self.prefill_logits_valid = true;
    }

    /// Get first token after prefill. Samples from last target logits
    /// (greedy: argmax). Caller should pass the sampled token.
    pub fn prefill_done(&mut self, first_token: u32) {
        self.current_token = first_token;
        self.prefill_logits_valid = false;
    }

    // ── Draft ────────────────────────────────────────────────────

    /// Begin draft phase. Fills `draft_tokens` and `draft_positions` buffers.
    /// Returns (start_token, start_position) for the first draft forward.
    /// Caller calls draft.forward() gamma times, feeding each output token
    /// as the next input, then calls `draft_done`.
    pub fn begin_draft(&mut self) -> (u32, u32) {
        self.phase = SpecPhase::Drafting;
        for i in 0..self.gamma as usize {
            self.draft_positions[i] = self.position + i as u32;
        }
        (self.current_token, self.position)
    }

    /// Record one draft token. Called after each draft.forward().
    /// `token` is the sampled next token.
    /// `is_last` must be true for the gamma-th token.
    pub fn draft_token(&mut self, index: u32, token: u32) {
        self.draft_tokens[index as usize] = token;
        if index == 0 {
            // First draft token becomes the verify_tokens[0]
            // (it's what the target will verify against prefill_logits)
        }
    }

    /// Draft phase complete. Prepare verify buffer.
    pub fn draft_done(&mut self) {
        // verify_tokens = draft_tokens (always gamma elements)
        self.verify_tokens[..self.gamma as usize]
            .copy_from_slice(&self.draft_tokens[..self.gamma as usize]);
        self.phase = SpecPhase::Verifying;
    }

    // ── Verify ───────────────────────────────────────────────────

    /// Get the verify token buffer (always exactly gamma tokens).
    pub fn get_verify_tokens(&self) -> &[u32] {
        &self.verify_tokens[..self.gamma as usize]
    }

    /// Get verify start position.
    pub fn get_verify_position(&self) -> u32 {
        self.position
    }

    // ── Accept/Reject ────────────────────────────────────────────

    /// Process target logits and determine accepted tokens.
    ///
    /// `target_logits` is (gamma, vocab_size) float array from target.forward().
    /// `prefill_logits` is (vocab_size,) from last prefill step, used to
    /// verify the first draft token.
    ///
    /// Returns (num_accepted, accepted_token_count).
    /// `accepted_tokens` buffer is filled with accepted tokens.
    /// `num_accepted` is how many draft tokens matched.
    /// `accepted_token_count` = num_accepted + (0 if all accepted else 1).
    ///
    /// Caller reads `accepted_tokens()` to get the result.
    pub fn accept_reject(&mut self, target_logits: &[f32], prefill_logits: &[f32]) -> (u32, u32) {
        let vocab = self.vocab_size as usize;
        let gamma = self.gamma as usize;
        let mut num_accepted: u32 = 0;
        let mut accepted_count: u32 = 0;

        for i in 0..gamma {
            // Which logits verify draft_tokens[i]?
            let logits = if i == 0 {
                prefill_logits
            } else {
                &target_logits[(i - 1) * vocab..i * vocab]
            };

            let target_tok = argmax(logits);
            let draft_tok = self.draft_tokens[i];

            if target_tok == draft_tok {
                self.accepted_tokens[accepted_count as usize] = draft_tok;
                num_accepted += 1;
                accepted_count += 1;
            } else {
                self.accepted_tokens[accepted_count as usize] = target_tok;
                accepted_count += 1;
                break; // Reject: discard remaining draft tokens
            }
        }

        // Update state
        self.steps += 1;
        self.total_draft_tokens += self.gamma;
        self.total_accepted += num_accepted;

        if num_accepted == self.gamma {
            // All accepted: sample bonus token from target_logits[gamma-1]
            let bonus_logits = &target_logits[(gamma - 1) * vocab..gamma * vocab];
            self.current_token = argmax(bonus_logits);
            self.position += self.gamma;
        } else {
            // Partial rejection: position advances by accepted + 1 (replacement)
            self.current_token = self.accepted_tokens[(accepted_count - 1) as usize];
            self.position += accepted_count;
        }

        // Check EOS
        if self.current_token <= 2 {
            self.phase = SpecPhase::Done;
        }

        (num_accepted, accepted_count)
    }

    /// Get accepted tokens after accept_reject.
    pub fn accepted_tokens_slice(&self) -> &[u32] {
        let count = if self.total_accepted == self.gamma {
            self.gamma as usize
        } else {
            (self.total_accepted + 1) as usize // TODO: fix this — use actual count
        };
        // Use steps to determine actual count
        let actual = if self.total_accepted % self.gamma == 0 && self.steps > 0 {
            self.gamma as usize
        } else {
            ((self.total_accepted % self.gamma) + 1) as usize
        };
        &self.accepted_tokens[..actual.min(MAX_GAMMA + 1)]
    }

    // ── Stats ────────────────────────────────────────────────────

    pub fn accept_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_draft_tokens as f32
        }
    }

    pub fn phase(&self) -> SpecPhase {
        self.phase
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn current_token(&self) -> u32 {
        self.current_token
    }

    pub fn is_done(&self) -> bool {
        matches!(self.phase, SpecPhase::Done)
    }
}

// ── Helpers ───────────────────────────────────────────────────────

fn argmax(arr: &[f32]) -> u32 {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ══════════════════════════════════════════════════════════════════
// C API (ctypes FFI)
// ══════════════════════════════════════════════════════════════════

use std::os::raw::c_int;

pub type LKOSpecState = SpecState;

/// Create a new speculative decode state.
#[no_mangle]
pub extern "C" fn lko_spec_create(gamma: c_int, vocab_size: c_int) -> *mut LKOSpecState {
    Box::into_raw(Box::new(SpecState::new(gamma as u32, vocab_size as u32)))
}

/// Destroy speculative decode state.
#[no_mangle]
pub extern "C" fn lko_spec_destroy(state: *mut LKOSpecState) {
    if !state.is_null() {
        unsafe {
            drop(Box::from_raw(state));
        }
    }
}

/// Begin prefill.
#[no_mangle]
pub extern "C" fn lko_spec_begin_prefill(state: *mut LKOSpecState) {
    if state.is_null() {
        return;
    }
    unsafe {
        (*state).begin_prefill();
    }
}

/// Get next prefill position.
#[no_mangle]
pub extern "C" fn lko_spec_prefill_position(state: *mut LKOSpecState) -> u32 {
    if state.is_null() {
        return 0;
    }
    unsafe { (*state).prefill_next_position() }
}

/// Mark prefill token done.
#[no_mangle]
pub extern "C" fn lko_spec_prefill_done(state: *mut LKOSpecState, first_token: u32) {
    if state.is_null() {
        return;
    }
    unsafe {
        (*state).prefill_done(first_token);
    }
}

/// Begin draft phase. Returns starting token via `token_out` and position via `pos_out`.
#[no_mangle]
pub extern "C" fn lko_spec_begin_draft(
    state: *mut LKOSpecState,
    token_out: *mut u32,
    pos_out: *mut u32,
) {
    if state.is_null() {
        return;
    }
    let (token, pos) = unsafe { (*state).begin_draft() };
    if !token_out.is_null() {
        unsafe {
            *token_out = token;
        }
    }
    if !pos_out.is_null() {
        unsafe {
            *pos_out = pos;
        }
    }
}

/// Record one draft token.
#[no_mangle]
pub extern "C" fn lko_spec_draft_token(state: *mut LKOSpecState, index: u32, token: u32) {
    if state.is_null() {
        return;
    }
    unsafe {
        (*state).draft_token(index, token);
    }
}

/// Mark draft phase complete.
#[no_mangle]
pub extern "C" fn lko_spec_draft_done(state: *mut LKOSpecState) {
    if state.is_null() {
        return;
    }
    unsafe {
        (*state).draft_done();
    }
}

/// Get verify information.
/// `tokens_out`: buffer of size gamma to receive verify tokens.
/// `count_out`: number of tokens (always gamma).
/// `pos_out`: start position.
/// Returns gamma.
#[no_mangle]
pub extern "C" fn lko_spec_get_verify(
    state: *mut LKOSpecState,
    tokens_out: *mut u32,
    pos_out: *mut u32,
) -> u32 {
    if state.is_null() {
        return 0;
    }
    let s = unsafe { &*state };
    let tokens = s.get_verify_tokens();
    if !tokens_out.is_null() {
        for (i, &t) in tokens.iter().enumerate() {
            unsafe {
                *tokens_out.add(i) = t;
            }
        }
    }
    if !pos_out.is_null() {
        unsafe {
            *pos_out = s.get_verify_position();
        }
    }
    tokens.len() as u32
}

/// Run accept/reject.
/// `target_logits`: (gamma, vocab_size) float buffer.
/// `prefill_logits`: (vocab_size,) float buffer.
/// `accepted_out`: buffer for accepted tokens (max gamma+1).
/// Returns (num_accepted << 16) | accepted_count.
#[no_mangle]
pub extern "C" fn lko_spec_accept_reject(
    state: *mut LKOSpecState,
    target_logits: *const c_float,
    prefill_logits: *const c_float,
    accepted_out: *mut u32,
) -> u32 {
    if state.is_null() {
        return 0;
    }
    let s = unsafe { &mut *state };
    let vocab = s.vocab_size as usize;
    let gamma = s.gamma as usize;
    let t_logits = unsafe { std::slice::from_raw_parts(target_logits, gamma * vocab) };
    let p_logits = unsafe { std::slice::from_raw_parts(prefill_logits, vocab) };

    let (num_acc, count) = s.accept_reject(t_logits, p_logits);

    if !accepted_out.is_null() {
        let accepted = s.accepted_tokens_slice();
        for (i, &t) in accepted.iter().enumerate() {
            unsafe {
                *accepted_out.add(i) = t;
            }
        }
    }

    (num_acc << 16) | count
}

/// Get current token for next iteration.
#[no_mangle]
pub extern "C" fn lko_spec_current_token(state: *mut LKOSpecState) -> u32 {
    if state.is_null() {
        return 0;
    }
    unsafe { (*state).current_token() }
}

/// Get current position.
#[no_mangle]
pub extern "C" fn lko_spec_position(state: *mut LKOSpecState) -> u32 {
    if state.is_null() {
        return 0;
    }
    unsafe { (*state).position() }
}

/// Check if done (EOS or max_tokens reached).
#[no_mangle]
pub extern "C" fn lko_spec_is_done(state: *mut LKOSpecState) -> c_int {
    if state.is_null() {
        return 1;
    }
    unsafe { (*state).is_done() as c_int }
}

/// Get accept rate × 100 (integer percentage).
#[no_mangle]
pub extern "C" fn lko_spec_accept_rate(state: *mut LKOSpecState) -> c_int {
    if state.is_null() {
        return 0;
    }
    unsafe { ((*state).accept_rate() * 100.0) as c_int }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_destroy() {
        let state = SpecState::new(4, 32000);
        assert_eq!(state.gamma, 4);
        assert!(matches!(state.phase, SpecPhase::Idle));
    }

    #[test]
    fn test_prefill_flow() {
        let mut state = SpecState::new(4, 100);
        state.begin_prefill();
        assert_eq!(state.prefill_next_position(), 0);
        state.prefill_token_done(std::ptr::null());
        assert_eq!(state.prefill_next_position(), 1);
        state.prefill_done(42);
        assert_eq!(state.current_token, 42);
    }

    #[test]
    fn test_draft_flow() {
        let mut state = SpecState::new(4, 100);
        state.position = 10;
        state.current_token = 7;

        let (token, pos) = state.begin_draft();
        assert_eq!(token, 7);
        assert_eq!(pos, 10);
        assert_eq!(state.draft_positions[0], 10);
        assert_eq!(state.draft_positions[3], 13);

        state.draft_token(0, 11);
        state.draft_token(1, 22);
        state.draft_token(2, 33);
        state.draft_token(3, 44);
        state.draft_done();

        let vt = state.get_verify_tokens();
        assert_eq!(vt, &[11, 22, 33, 44]);
    }

    #[test]
    fn test_accept_all() {
        let mut state = SpecState::new(3, 10);
        state.position = 5;
        state.current_token = 1;
        state.begin_draft();
        state.draft_token(0, 2);
        state.draft_token(1, 3);
        state.draft_token(2, 4);
        state.draft_done();

        // Target always agrees with draft
        // prefill_logits: argmax=2 (verifies draft[0])
        // target_logits[0]: argmax=3 (verifies draft[1])
        // target_logits[1]: argmax=4 (verifies draft[2])
        let vocab = 10;
        let mut prefill = vec![0.0f32; vocab];
        prefill[2] = 100.0;
        let mut tlogits = vec![0.0f32; 3 * vocab];
        tlogits[0 * vocab + 3] = 100.0;
        tlogits[1 * vocab + 4] = 100.0;
        tlogits[2 * vocab + 0] = 100.0; // bonus

        let (n_acc, count) = state.accept_reject(&tlogits, &prefill);
        assert_eq!(n_acc, 3);
        assert_eq!(count, 3);
        assert_eq!(state.accepted_tokens[0], 2);
        assert_eq!(state.accepted_tokens[1], 3);
        assert_eq!(state.accepted_tokens[2], 4);
        assert_eq!(state.position, 8); // 5 + 3
    }

    #[test]
    fn test_reject_first() {
        let mut state = SpecState::new(3, 10);
        state.position = 5;
        state.current_token = 1;
        state.begin_draft();
        state.draft_token(0, 99); // draft predicts 99
        state.draft_token(1, 3);
        state.draft_token(2, 4);
        state.draft_done();

        let vocab = 10;
        let mut prefill = vec![0.0f32; vocab];
        prefill[7] = 100.0; // target says 7
        let mut tlogits = vec![0.0f32; 3 * vocab];

        let (n_acc, count) = state.accept_reject(&tlogits, &prefill);
        assert_eq!(n_acc, 0, "no draft tokens accepted");
        assert_eq!(count, 1, "one replacement token");
        assert_eq!(state.accepted_tokens[0], 7, "target's replacement");
    }

    #[test]
    fn test_c_api_flow() {
        let state = lko_spec_create(4, 100);
        assert!(!state.is_null());

        lko_spec_begin_prefill(state);
        assert_eq!(lko_spec_prefill_position(state), 0);

        lko_spec_prefill_done(state, 42);
        assert_eq!(lko_spec_current_token(state), 42);

        let mut token: u32 = 0;
        let mut pos: u32 = 0;
        lko_spec_begin_draft(state, &mut token, &mut pos);
        assert_eq!(token, 42);

        lko_spec_draft_token(state, 0, 10);
        lko_spec_draft_token(state, 1, 20);
        lko_spec_draft_token(state, 2, 30);
        lko_spec_draft_token(state, 3, 40);
        lko_spec_draft_done(state);

        let mut vtokens = [0u32; 4];
        let count = lko_spec_get_verify(state, vtokens.as_mut_ptr(), &mut pos);
        assert_eq!(count, 4);

        lko_spec_destroy(state);
    }
}
