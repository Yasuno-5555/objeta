//! objeta-metal — Metal Shader Library for MoE Runtime.
//!
//! Kernels (in kernels/metal/):
//!   1. q4_expert_gemv   — Dequantize q4 weights + GEMV on GPU
//!   2. multi_expert_gemv — Parallel dispatch across multiple experts
//!   3. router_forward   — Router logits + softmax + top-k
//!   4. fused_residual_norm — Accumulate + RMSNorm
//!   5. shared_expert_forward — Shared expert FFN
//!
//! Build: xcrun -sdk macosx metal -c kernels/metal/*.metal -o objeta.metallib

/// Paths to Metal shader sources.
pub const KERNEL_DIR: &str = "kernels/metal";

/// Kernel function names.
pub const KERNEL_Q4_EXPERT_GEMV: &str = "q4_expert_gemv";
pub const KERNEL_MULTI_EXPERT_GEMV: &str = "multi_expert_gemv";
pub const KERNEL_ROUTER_LOGITS: &str = "router_logits";
pub const KERNEL_ROUTER_SOFTMAX_TOPK: &str = "router_softmax_topk";
pub const KERNEL_FUSED_RESIDUAL_NORM: &str = "fused_residual_norm";
pub const KERNEL_SHARED_EXPERT_FORWARD: &str = "shared_expert_forward";

/// Generate the build command for compiling Metal shaders.
pub fn metal_build_command(output: &str) -> String {
    format!(
        "xcrun -sdk macosx metal -c {0}/q4_expert_gemv.metal {0}/router_forward.metal {0}/fused_ops.metal -o {1}",
        KERNEL_DIR, output
    )
}

/// Emit all Metal shader sources to a directory.
pub fn emit_kernels(out_dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let kernels = [
        ("q4_expert_gemv.metal", include_str!("../../../kernels/metal/q4_expert_gemv.metal")),
        ("router_forward.metal", include_str!("../../../kernels/metal/router_forward.metal")),
        ("fused_ops.metal", include_str!("../../../kernels/metal/fused_ops.metal")),
    ];
    for (name, source) in kernels {
        std::fs::write(out_dir.join(name), source)?;
    }
    Ok(())
}
