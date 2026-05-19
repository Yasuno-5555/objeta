//! KV Cache layouts for Qwen3.6 executor.

pub struct KvCacheStats {
    pub allocated_bytes: usize,
}

pub trait KvCache: Send + Sync {
    fn append(&mut self, layer: usize, position: usize, k: &[f32], v: &[f32]);
    fn reset(&mut self);
    fn stats(&self) -> KvCacheStats;
    fn get_k(&self, layer: usize, kv_head: usize, token_idx: usize) -> &[f32];
    fn get_v(&self, layer: usize, kv_head: usize, token_idx: usize) -> &[f32];
    fn as_mut_slices(&mut self, _layer: usize) -> Option<(&mut [f32], &mut [f32])> {
        None
    }
}

pub struct KvCacheLegacy {
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
}

impl KvCacheLegacy {
    pub fn new(n_layers: usize, n_kv: usize, max_seq: usize, head_dim: usize) -> Self {
        let size = n_kv * max_seq * head_dim;
        Self {
            n_layers,
            n_kv,
            max_seq,
            head_dim,
            kv_k: vec![vec![0.0f32; size]; n_layers],
            kv_v: vec![vec![0.0f32; size]; n_layers],
        }
    }
}

impl KvCache for KvCacheLegacy {
    fn append(&mut self, layer: usize, position: usize, k: &[f32], v: &[f32]) {
        for h in 0..self.n_kv {
            let k_off = h * self.max_seq * self.head_dim + position * self.head_dim;
            let v_off = h * self.max_seq * self.head_dim + position * self.head_dim;
            for d in 0..self.head_dim {
                self.kv_k[layer][k_off + d] = k[h * self.head_dim + d];
                self.kv_v[layer][v_off + d] = v[h * self.head_dim + d];
            }
        }
    }

    fn reset(&mut self) {
        for layer in 0..self.n_layers {
            self.kv_k[layer].fill(0.0);
            self.kv_v[layer].fill(0.0);
        }
    }

    fn stats(&self) -> KvCacheStats {
        let allocated_bytes = self.n_layers * self.n_kv * self.max_seq * self.head_dim * 4 * 2;
        KvCacheStats { allocated_bytes }
    }

    fn get_k(&self, layer: usize, kv_head: usize, token_idx: usize) -> &[f32] {
        let start = (kv_head * self.max_seq + token_idx) * self.head_dim;
        &self.kv_k[layer][start..start + self.head_dim]
    }

    fn get_v(&self, layer: usize, kv_head: usize, token_idx: usize) -> &[f32] {
        let start = (kv_head * self.max_seq + token_idx) * self.head_dim;
        &self.kv_v[layer][start..start + self.head_dim]
    }

    fn as_mut_slices(&mut self, layer: usize) -> Option<(&mut [f32], &mut [f32])> {
        Some((&mut self.kv_k[layer], &mut self.kv_v[layer]))
    }
}

pub struct KvCacheTokenMajor {
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
}

impl KvCacheTokenMajor {
    pub fn new(n_layers: usize, n_kv: usize, max_seq: usize, head_dim: usize) -> Self {
        let size = max_seq * n_kv * head_dim;
        Self {
            n_layers,
            n_kv,
            max_seq,
            head_dim,
            kv_k: vec![vec![0.0f32; size]; n_layers],
            kv_v: vec![vec![0.0f32; size]; n_layers],
        }
    }
}

impl KvCache for KvCacheTokenMajor {
    fn append(&mut self, layer: usize, position: usize, k: &[f32], v: &[f32]) {
        let base = position * self.n_kv * self.head_dim;
        for h in 0..self.n_kv {
            let offset = base + h * self.head_dim;
            for d in 0..self.head_dim {
                self.kv_k[layer][offset + d] = k[h * self.head_dim + d];
                self.kv_v[layer][offset + d] = v[h * self.head_dim + d];
            }
        }
    }

    fn reset(&mut self) {
        for layer in 0..self.n_layers {
            self.kv_k[layer].fill(0.0);
            self.kv_v[layer].fill(0.0);
        }
    }

    fn stats(&self) -> KvCacheStats {
        let allocated_bytes = self.n_layers * self.n_kv * self.max_seq * self.head_dim * 4 * 2;
        KvCacheStats { allocated_bytes }
    }

    fn get_k(&self, layer: usize, kv_head: usize, token_idx: usize) -> &[f32] {
        let start = (token_idx * self.n_kv + kv_head) * self.head_dim;
        &self.kv_k[layer][start..start + self.head_dim]
    }

    fn get_v(&self, layer: usize, kv_head: usize, token_idx: usize) -> &[f32] {
        let start = (token_idx * self.n_kv + kv_head) * self.head_dim;
        &self.kv_v[layer][start..start + self.head_dim]
    }
}

pub static mut KV_LAYOUT: i32 = 0; // 0 = Legacy, 1 = TokenMajor

#[no_mangle]
pub unsafe extern "C" fn lko_runner_set_kv_layout(layout: i32) -> i32 {
    KV_LAYOUT = layout;
    0
}
