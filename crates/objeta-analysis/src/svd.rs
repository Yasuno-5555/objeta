//! Minimal SVD utilities — exact for small matrices, randomized for large ones.
//!
//! We don't pull in a full LAPACK dependency. For the effective rank computation
//! we need singular values; for the direction analysis we need top singular vectors.
//!
//! Strategy:
//! - Full SVD via nalgebra for matrices ≤ 2048×2048
//! - Randomized SVD (Halko-Martinsson-Tropp) for larger matrices
//! - Power iteration for top singular vector of a single matrix

use nalgebra::{DMatrix, DVector};
use rand::Rng;

// ── Effective Rank ───────────────────────────────────────────────────────

/// Effective rank of a pre-built nalgebra matrix (full SVD, exact).
pub fn effective_rank_full(mat: &DMatrix<f64>) -> f64 {
    let svd = mat.clone().svd(true, true);
    let s = svd.singular_values;
    let sum_s: f64 = s.iter().sum();
    let sum_s2: f64 = s.iter().map(|v| v * v).sum();
    if sum_s2 < 1e-12 {
        return 1.0;
    }
    (sum_s * sum_s) / sum_s2
}

/// Randomized effective rank for stacked [gate; up] matrices.
///
/// Samples with k+p random Gaussian vectors, computes action of [G;U] and
/// [G;U]^T, builds a (k+p)×(k+p) matrix whose singular values approximate
/// the top-(k+p) singular values of the full matrix.
pub fn effective_rank_randomized(
    gate: &[f32],
    up: &[f32],
    m_half: usize,   // ffn_dim (= rows of gate, rows of up)
    n: usize,        // hidden_dim (= cols)
    k: usize,        // target rank for approximation
) -> f64 {
    let total_m = 2 * m_half;
    let p = 8; // oversampling

    // Step 1: Generate random matrix Omega of size n × (k+p)
    let mut rng = rand::thread_rng();
    let mut omega = vec![0.0f64; n * (k + p)];
    for v in omega.iter_mut() {
        *v = rng.gen_range(-1.0f64..1.0);
    }

    // Step 2: Y = A @ Omega  (total_m × (k+p))
    let mut y = vec![0.0f64; total_m * (k + p)];
    for col in 0..(k + p) {
        let x = &omega[col * n..(col + 1) * n];
        let y_col = &mut y[col * total_m..(col + 1) * total_m];
        // Upper half: gate @ x
        for i in 0..m_half {
            let mut s = 0.0;
            for j in 0..n {
                s += gate[i * n + j] as f64 * x[j];
            }
            y_col[i] = s;
        }
        // Lower half: up @ x
        for i in 0..m_half {
            let mut s = 0.0;
            for j in 0..n {
                s += up[i * n + j] as f64 * x[j];
            }
            y_col[m_half + i] = s;
        }
    }

    // Step 3: QR decomposition of Y (using modified Gram-Schmidt for simplicity)
    let (q, _) = mgs_qr(&y, total_m, k + p);

    // Step 4: B = Q^T @ A (size (k+p) × n)
    // B = A^T @ Q first, then transpose. Actually compute:
    // For each column of Q (size total_m), compute Q_col in A's row space
    let mut b = vec![0.0f64; (k + p) * n];
    for col in 0..(k + p) {
        let q_col = &q[col * total_m..(col + 1) * total_m];
        let b_row = &mut b[col * n..(col + 1) * n];
        // b_row = A^T @ q_col
        // A = [gate; up], so A^T = [gate^T, up^T]
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..m_half {
                s += gate[i * n + j] as f64 * q_col[i];
                s += up[i * n + j] as f64 * q_col[m_half + i];
            }
            b_row[j] = s;
        }
    }

    // Step 5: SVD of B (small, (k+p) × n)
    if k + p <= n {
        let b_mat = DMatrix::from_vec(k + p, n, b);
        let svd = b_mat.svd(true, true);
        let s = svd.singular_values;
        let sum_s: f64 = s.iter().sum();
        let sum_s2: f64 = s.iter().map(|v| v * v).sum();

        // The singular values of B are approx the top-(k+p) singular values of A.
        // We also know trace(A^T A) = sum of all squared singular values.
        // Compute full Frobenius norm of A for the complete denominator.
        let mut full_sq = 0.0f64;
        for i in 0..m_half {
            for j in 0..n {
                full_sq += (gate[i * n + j] as f64).powi(2);
                full_sq += (up[i * n + j] as f64).powi(2);
            }
        }

        // Effective rank using known total sum of squares
        let approx_sum_s2 = sum_s2.min(full_sq);
        if approx_sum_s2 < 1e-12 {
            return 1.0;
        }
        (sum_s * sum_s) / full_sq
    } else {
        // Fallback: just use the QR approximation
        let sum_s: f64 = (0..n).map(|i| {
            let mut col_norm = 0.0;
            for j in 0..(k + p) {
                col_norm += b[j * n + i].powi(2);
            }
            col_norm.sqrt()
        }).sum();

        let mut full_sq = 0.0f64;
        for i in 0..m_half {
            for j in 0..n {
                full_sq += (gate[i * n + j] as f64).powi(2);
                full_sq += (up[i * n + j] as f64).powi(2);
            }
        }
        if full_sq < 1e-12 {
            return 1.0;
        }
        (sum_s * sum_s) / full_sq
    }
}

// ── Randomized SVD for top-k singular vectors ─────────────────────────────

/// Randomized SVD for the stacked [gate; up] matrix.
/// Returns (U, S, Vt) where A ≈ U @ diag(S) @ Vt.
pub fn randomized_svd_stacked(
    gate: &[f32],
    up: &[f32],
    m_half: usize,
    n: usize,
    k: usize,
) -> Option<(DMatrix<f64>, DVector<f64>, DMatrix<f64>)> {
    let total_m = 2 * m_half;
    let p = 8;
    let mut rng = rand::thread_rng();

    // Step 1: Random matrix Omega (n × (k+p))
    let mut omega = vec![0.0f64; n * (k + p)];
    for v in omega.iter_mut() {
        *v = rng.gen_range(-1.0f64..1.0);
    }

    // Step 2: Y = A @ Omega
    let mut y = vec![0.0f64; total_m * (k + p)];
    for col in 0..(k + p) {
        let x = &omega[col * n..(col + 1) * n];
        let y_col = &mut y[col * total_m..(col + 1) * total_m];
        for i in 0..m_half {
            let mut s = 0.0;
            for j in 0..n { s += gate[i * n + j] as f64 * x[j]; }
            y_col[i] = s;
        }
        for i in 0..m_half {
            let mut s = 0.0;
            for j in 0..n { s += up[i * n + j] as f64 * x[j]; }
            y_col[m_half + i] = s;
        }
    }

    // Step 3: Q = orth(Y) via MGS
    let (q, _) = mgs_qr(&y, total_m, k + p);

    // Step 4: B = Q^T @ A (size (k+p) × n)
    let mut b = vec![0.0f64; (k + p) * n];
    for col in 0..(k + p) {
        let q_col = &q[col * total_m..(col + 1) * total_m];
        let b_row = &mut b[col * n..(col + 1) * n];
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..m_half {
                s += gate[i * n + j] as f64 * q_col[i];
                s += up[i * n + j] as f64 * q_col[m_half + i];
            }
            b_row[j] = s;
        }
    }

    // Step 5: Full SVD of B
    let b_mat = DMatrix::from_vec(k + p, n, b);
    let svd = b_mat.svd(true, true);
    let s = svd.singular_values;
    let vb = svd.v_t?; // (n × n) or (n × (k+p))

    // Truncate to top-k
    let top_k = k.min(s.len());
    let s_top = DVector::from(s.as_slice()[..top_k].to_vec());
    let vt_top: DMatrix<f64> = vb.rows(0, top_k).into();

    // U_top = Q @ U_B[:, :top_k]
    let u_b = svd.u?; // ((k+p) × (k+p))
    let u_top = DMatrix::from_vec(total_m, top_k,
        (0..total_m * top_k).map(|idx| {
            let row = idx / top_k;
            let col = idx % top_k;
            (0..(k + p)).map(|i| q[i * total_m + row] * u_b[(i, col)]).sum()
        }).collect()
    );

    Some((u_top, s_top, vt_top))
}

// ── Power Iteration ──────────────────────────────────────────────────────

/// Top right singular vector of a (rows × cols) matrix via power iteration.
/// Returns the dominant right singular vector (size cols).
pub fn power_iteration(mat: &[f32], rows: usize, cols: usize) -> Option<DVector<f64>> {
    let mut rng = rand::thread_rng();
    let mut v = DVector::from((0..cols).map(|_| rng.gen_range(-1.0f64..1.0)).collect::<Vec<f64>>());

    for _ in 0..10 {
        // v = A^T @ A @ v
        // Step 1: u = A @ v
        let mut u = vec![0.0f64; rows];
        for i in 0..rows {
            let mut s = 0.0;
            for j in 0..cols {
                s += mat[i * cols + j] as f64 * v[j];
            }
            u[i] = s;
        }
        // Step 2: v_new = A^T @ u
        let mut v_new = vec![0.0f64; cols];
        for j in 0..cols {
            let mut s = 0.0;
            for i in 0..rows {
                s += mat[i * cols + j] as f64 * u[i];
            }
            v_new[j] = s;
        }
        // Normalize
        let n: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n < 1e-12 {
            return None;
        }
        for (j, val) in v_new.iter().enumerate() {
            v[j] = val / n;
        }
    }

    Some(v)
}

// ── Modified Gram-Schmidt QR ─────────────────────────────────────────────

/// In-place MGS QR. Input Y is (rows × cols) stored column-major.
/// Returns (Q, R) where Q is column-major (rows × cols).
fn mgs_qr(y: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0f64; rows * cols];
    let mut r = vec![0.0f64; cols * cols];

    // Copy Y into Q
    q.copy_from_slice(y);

    for j in 0..cols {
        // Compute norm of column j
        let mut nrm = 0.0;
        for i in 0..rows {
            nrm += q[i * cols + j].powi(2);
        }
        nrm = nrm.sqrt();

        if nrm > 1e-12 {
            r[j * cols + j] = nrm;
            for i in 0..rows {
                q[i * cols + j] /= nrm;
            }
        } else {
            r[j * cols + j] = 0.0;
        }

        // Subtract projection from remaining columns
        for k in (j + 1)..cols {
            let mut dot = 0.0;
            for i in 0..rows {
                dot += q[i * cols + j] * q[i * cols + k];
            }
            r[j * cols + k] = dot;
            for i in 0..rows {
                q[i * cols + k] -= dot * q[i * cols + j];
            }
        }
    }

    (q, r)
}

// ── Full SVD for rotation extraction ──────────────────────────────────────

/// Compute the top-k SVD of the down projection matrix.
/// Returns (U_k, Σ_k, V_k) where down ≈ U_k @ diag(Σ_k) @ V_k^T.
///
/// U: (hidden_dim × k) — output directions
/// V: (ffn_dim × k) — which intermediate neurons matter
/// Σ: (k,) — scaling factors
pub fn svd_down_projection(
    down: &[f32],
    ffn_dim: usize,
    hidden_dim: usize,
    k: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    // down is (hidden_dim × ffn_dim) row-major
    let m = hidden_dim;
    let n = ffn_dim;

    // Use randomized SVD for large matrices
    let (u_mat, s_vec, vt_mat) = randomized_svd_single(down, m, n, k)?;

    // Extract as flat vectors
    let u: Vec<f32> = u_mat.iter().map(|&v| v as f32).collect();
    let sv: Vec<f32> = s_vec.iter().map(|&v| v as f32).collect();
    let v: Vec<f32> = vt_mat.transpose().iter().map(|&v| v as f32).collect();

    Some((u, sv, v))
}

/// Randomized SVD for a single matrix A (m × n, row-major).
fn randomized_svd_single(
    data: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Option<(DMatrix<f64>, DVector<f64>, DMatrix<f64>)> {
    let p = 8;
    let effective_k = (k + p).min(n).min(m);
    let mut rng = rand::thread_rng();

    // Step 1: Random matrix Omega (n × effective_k)
    let mut omega = vec![0.0f64; n * effective_k];
    for v in omega.iter_mut() {
        *v = rng.gen_range(-1.0f64..1.0);
    }

    // Step 2: Y = A @ Omega (m × effective_k)
    let mut y = vec![0.0f64; m * effective_k];
    for col in 0..effective_k {
        let x = &omega[col * n..(col + 1) * n];
        let y_col = &mut y[col * m..(col + 1) * m];
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..n {
                s += data[i * n + j] as f64 * x[j];
            }
            y_col[i] = s;
        }
    }

    // Step 3: QR of Y
    let (q, _) = mgs_qr(&y, m, effective_k);

    // Step 4: B = Q^T @ A (effective_k × n)
    let mut b = vec![0.0f64; effective_k * n];
    for i in 0..effective_k {
        let q_col = &q[i * m..(i + 1) * m];
        let b_row = &mut b[i * n..(i + 1) * n];
        for j in 0..n {
            let mut s = 0.0;
            for r in 0..m {
                s += q_col[r] * data[r * n + j] as f64;
            }
            b_row[j] = s;
        }
    }

    // Step 5: SVD of B
    let b_mat = DMatrix::from_vec(effective_k, n, b);
    let svd = b_mat.clone().svd(true, true);

    let s_full = svd.singular_values;
    let vb = svd.v_t?;

    let top_k = k.min(s_full.len());
    let s_top = DVector::from(s_full.as_slice()[..top_k].to_vec());
    let vt_top: DMatrix<f64> = vb.rows(0, top_k).into();

    // U_top = Q @ U_B
    let u_b = svd.u?;
    let u_top = DMatrix::from_vec(m, top_k,
        (0..m * top_k).map(|idx| {
            let r = idx / top_k;
            let c = idx % top_k;
            (0..effective_k).map(|i| q[i * m + r] * u_b[(i, c)]).sum()
        }).collect()
    );

    Some((u_top, s_top, vt_top))
}
