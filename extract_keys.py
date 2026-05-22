import sys, re, json
raw = sys.stdin.read()
# Remove everything before the first '{', including ANSI escapes
start = raw.find('{')
end = raw.rfind('}')
if start >= 0 and end > start:
    # Remove ANSI escape sequences from warnings
    cleaned = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw[start:end+1])
    data = json.loads(cleaned)
    keys = [
        "source", "layer_id", "shared_expert_included", "shared_expert_weight_bytes",
        "routed_only_output_norm", "shared_only_output_norm", "full_moe_output_norm",
        "full_minus_routed_norm", "shared_merge_residual_l2",
        "official_arithmetic_cuda_parity",
        "routed_fp4_bytes_loaded", "routed_fp4_bytes_reused",
        "shared_fp8_bytes_loaded", "shared_fp8_bytes_reused",
        "total_logical_bytes", "total_loaded_bytes", "total_reused_bytes",
        "total_ms"
    ]
    for k in keys:
        v = data.get(k, "MISSING")
        if k in ("cuda_shared_vs_cpu_shared", "cuda_full_moe_vs_cpu_full_moe") and v:
            print(f"  {k}: cosine={v['cosine_similarity']:.2f} rel_l2={v['relative_l2_error']:.2e} max_abs={v['max_abs_error']}")
        else:
            print(f"  {k}: {v}")
    print()
    # Check the new parity fields
    for ck in ("cuda_shared_vs_cpu_shared", "cuda_full_moe_vs_cpu_full_moe"):
        cv = data.get(ck)
        if cv:
            print(f"  {ck}: cosine={cv['cosine_similarity']:.10f} rel_l2={cv['relative_l2_error']:.10e} max_abs={cv['max_abs_error']}")
else:
    print("No JSON found")
