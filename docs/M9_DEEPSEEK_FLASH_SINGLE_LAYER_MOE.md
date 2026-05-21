# M9 Design: Real DeepSeek V4 Flash Single-Layer MoE Proof

## Goal

Run one real DeepSeek V4 Flash MoE layer end to end without loading the full checkpoint into RAM.

Scope:

- one selected layer only
- router tensor for that layer
- expert tensors for that layer only
- CPU reference
- CUDA selected-expert execution
- output comparison
- telemetry

Out of scope:

- full forward
- generation
- attention
- full-model loading
- guessing packed expert slicing

## Layer Selection

Input should require an explicit layer id:

- `--layer <L>`

Execution must stop if:

- `L >= num_layers`
- the selected layer has no router tensor
- the selected layer has no expert tensors

The layer id should be resolved only from parser metadata already written by:

- `deepseek_v4_flash_layout.json`
- `deepseek_v4_flash_expert_layout.json`
- `deepseek_v4_flash_router_layout.json`
- `deepseek_v4_flash_tensor_index.json`

## Router Tensor Identification

Router tensor selection should use `RouterLayout.routers` entries filtered by:

- `layer_id == Some(L)`

Execution must stop if:

- zero router tensors match
- more than one router tensor matches and no tie-break rule is explicit
- router shape is incompatible with `num_experts` and `hidden_size`

Expected router shape should be validated conservatively against metadata, for example:

- `[num_experts, hidden_size]`

If the parser metadata is ambiguous, execution must refuse to continue.

## Expert Tensor Identification

Expert tensor selection should use `ExpertLayout.tensors` filtered by:

- `layer_id == Some(L)`
- `tensor_kind in { "gate", "up", "down", "gate_up" }`

Shared expert tensors may be discovered but should not be silently mixed into the first proof unless the execution path explicitly supports them.

### Explicit Experts Layout

For `layout_kind == "explicit_experts"`:

- group by `expert_id`
- require per-selected-expert presence of:
  - `gate` and `up` and `down`, or
  - a supported fused `gate_up` plus `down`

Execution must stop if any selected expert is missing required tensors.

### Packed Experts Layout

For `layout_kind == "packed_experts"`:

- parser metadata must provide enough slicing information to isolate one expert for one layer
- if that slicing metadata is absent, the executor must refuse to run

Hard rule:

- do not infer per-expert offsets from shape alone
- do not divide bytes evenly and pretend that is execution metadata
- do not guess packed ordering

## Required Packed-Expert Slicing Metadata

Real packed execution needs explicit metadata per packed tensor:

- source tensor name
- source file
- source dtype
- tensor shape
- layer id
- packing axis for expert dimension
- stride or slice rule for each expert
- byte offset base
- byte span per expert slice, or element span plus dtype

A future parser output may need a new structure such as:

- `packed_slice_layout.json`

without this, m9 must fail clearly.

## Reading Only Required Tensor Payloads

Use `ModelWeights` from `objeta-parser` to open the model directory lazily.

Allowed behavior:

- open the relevant safetensors shard(s)
- read only the router tensor
- read only the selected experts' gate/up/down tensors

Disallowed behavior:

- preloading all tensors into RAM
- expanding the full checkpoint into host buffers

For explicit experts:

- look up each required tensor name in `deepseek_v4_flash_tensor_index.json`
- fetch just that tensor through `ModelWeights`

For packed experts:

- only proceed if explicit slice metadata exists
- otherwise emit a diagnostic and stop

## Source Dtype Handling

Real payload loading should accept source dtypes only when explicitly supported.

Initial acceptable dtypes:

- `BF16`
- `F16`
- `F32`

Conversion options:

- CPU reference path:
  - convert source tensor to `fp32`
- CUDA selected MoE path:
  - either quantize loaded tensor to Objeta `Q4_0`
  - or run an explicit unquantized reference path if added later

Execution must stop if:

- dtype is unsupported
- tensor shape does not match expected matrix dimensions
- loaded element count does not match metadata

## Hidden Input Vector

The first proof does not require real upstream activations.

Allowed initial inputs:

- deterministic synthetic hidden vector from seed
- optionally a saved fixture vector from file if explicitly provided

Default:

- seeded synthetic `fp32` vector of length `hidden_size`

Execution must stop if:

- hidden vector length does not equal `hidden_size`

## Router Execution

The router should be run on CPU first for the proof.

Steps:

1. load router tensor for layer `L`
2. compute router logits from hidden vector
3. select `top_k`
4. compute selected weights

The router result should then drive both:

- CPU MoE reference
- CUDA selected MoE path

This keeps the selected experts identical across both paths.

## CPU Reference

CPU reference should:

1. load only selected experts for layer `L`
2. convert weights to `fp32`
3. compute:
   - `gate = W_gate @ x`
   - `up = W_up @ x`
   - `act = silu(gate) * up`
   - `down = W_down @ act`
   - accumulate with router weights

If `gate_up` fused storage exists, CPU reference may split or handle it explicitly only if metadata is sufficient.

## CUDA Selected MoE

Initial real CUDA path should reuse the current selected-expert executor structure:

1. quantize selected expert tensors into Objeta `Q4_0`
2. build `ExpertWeights`
3. call `execute_selected_moe_cuda`

This keeps the first real payload proof aligned with the existing m8 synthetic path.

## Output Comparison

Report both:

- `cuda_vs_cpu_quant`
- `quant_vs_fp32`

Minimum output checks:

- cosine similarity
- relative L2 error
- max absolute error

Router checks should include:

- selected expert ids
- selected weights
- optional top-k overlap if alternate router implementations are compared

## Telemetry

Emit:

- layer id
- source model path
- source tensor names used
- selected expert ids
- selected expert weights
- source dtype per tensor kind
- quant format used
- logical expert bytes requested
- actual expert bytes loaded
- resident cache bytes reused
- resident cache resident bytes
- bytes per expert
- bytes by tensor kind
- selected working set bytes
- cache hit/miss/eviction counters
- oversized bypass counters
- self-eviction risk count
- router time
- tensor load time
- quantization time
- H2D time
- gate/up GEMV time
- activation time
- down GEMV time
- accumulation time
- total time
- numerical comparisons

The existing byte invariant must still hold:

- `logical_expert_bytes_requested = actual_expert_bytes_loaded + resident_cache_bytes_reused`

## Failure Cases That Must Stop Execution

- missing parser JSON files
- missing `tensor_index` entries for required tensors
- missing router tensor for selected layer
- missing expert tensors for selected experts
- unsupported `layout_kind`
- packed expert layout without explicit slicing metadata
- unsupported source dtype
- shape mismatch against `hidden_size`, `intermediate_size`, `num_experts`, or `top_k`
- zero or invalid `top_k`
- duplicate or ambiguous router tensor match
- duplicate or ambiguous expert tensor match for a required slot
- requested layer outside valid range

## Recommended Implementation Order

1. explicit-expert single-layer proof only
2. deterministic hidden input
3. CPU router + CPU MoE reference
4. CUDA selected MoE through existing Q4 path
5. telemetry and diagnostics polish
6. packed-expert support only after explicit slicing metadata exists
