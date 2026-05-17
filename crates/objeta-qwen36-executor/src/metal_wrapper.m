// Metal wrapper with persistent buffer reuse.
// Compiles with: xcrun clang -c metal_wrapper.m -o metal_wrapper.o -framework Metal -framework Foundation

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;
    // Persistent buffers
    id<MTLBuffer> x_buf, y_buf, w_buf;
    size_t x_cap, y_cap, w_cap;
    // Cached pipelines
    id<MTLComputePipelineState> pipe_fp16_gemv, pipe_q4_gemv, pipe_multi_expert;
    id<MTLComputePipelineState> pipe_gqa, pipe_gqa_oproj;
    // GQA persistent buffers (per-layer weights + reusable caches)
    id<MTLBuffer> gqa_w, gqa_wo, gqa_k, gqa_v, gqa_h, gqa_cos, gqa_sin, gqa_out;
    size_t gqa_w_cap, gqa_wo_cap, gqa_kv_cap;
} MetalGpu;

// ── Init / Destroy ────────────────────────────────────────────────────────

MetalGpu* metal_init(const char* metallib_path) {
    MetalGpu* gpu = (MetalGpu*)calloc(1, sizeof(MetalGpu));
    if (!gpu) return NULL;
    gpu->device = MTLCreateSystemDefaultDevice();
    if (!gpu->device) { free(gpu); return NULL; }

    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
    NSError* err = nil;
    gpu->library = [gpu->device newLibraryWithURL:url error:&err];
    if (!gpu->library) { free(gpu); return NULL; }
    gpu->queue = [gpu->device newCommandQueue];
    return gpu;
}

void metal_destroy(MetalGpu* gpu) {
    if (gpu) {
        [gpu->x_buf release];
        [gpu->y_buf release];
        [gpu->w_buf release];
        [gpu->gqa_w release]; [gpu->gqa_wo release];
        [gpu->gqa_k release]; [gpu->gqa_v release];
        [gpu->gqa_h release]; [gpu->gqa_cos release]; [gpu->gqa_sin release];
        [gpu->gqa_out release];
        [gpu->pipe_fp16_gemv release]; [gpu->pipe_q4_gemv release];
        [gpu->pipe_multi_expert release];
        [gpu->pipe_gqa release]; [gpu->pipe_gqa_oproj release];
        free(gpu);
    }
}

// ── Buffer management ─────────────────────────────────────────────────────

static id<MTLBuffer> ensure_buf(id<MTLDevice> dev, id<MTLBuffer>* buf, size_t* cap, size_t needed) {
    if (*buf && *cap >= needed) return *buf;
    if (*buf) { [*buf release]; *buf = nil; }
    *buf = [dev newBufferWithLength:needed options:MTLResourceStorageModeShared];
    if (*buf) [*buf retain];
    *cap = needed;
    return *buf;
}

// ── Pipeline cache ────────────────────────────────────────────────────────

static id<MTLComputePipelineState> get_pipe(MetalGpu* gpu, id<MTLComputePipelineState>* cache, const char* name) {
    if (*cache) return *cache;
    id<MTLFunction> fn = [gpu->library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) return nil;
    *cache = [gpu->device newComputePipelineStateWithFunction:fn error:nil];
    [*cache retain];
    return *cache;
}

// ── Dispatch helper ───────────────────────────────────────────────────────

static int dispatch_kernel(
    MetalGpu* gpu, id<MTLComputePipelineState>* pipe_cache, const char* fn_name,
    const void* w_data, size_t w_bytes,
    const float* x, size_t k,
    float* y, size_t m,
    const uint32_t* dims, size_t dims_bytes)
{
    if (!gpu) return -1;

    id<MTLComputePipelineState> pipe = get_pipe(gpu, pipe_cache, fn_name);
    if (!pipe) return -3;

    id<MTLBuffer> wb = ensure_buf(gpu->device, &gpu->w_buf, &gpu->w_cap, w_bytes);
    id<MTLBuffer> xb = ensure_buf(gpu->device, &gpu->x_buf, &gpu->x_cap, k * sizeof(float));
    id<MTLBuffer> yb = ensure_buf(gpu->device, &gpu->y_buf, &gpu->y_cap, m * sizeof(float));

    memcpy([wb contents], w_data, w_bytes);
    memcpy([xb contents], x, k * sizeof(float));

    id<MTLCommandBuffer> cmd = [gpu->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:0 atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:yb offset:0 atIndex:2];
    if (dims_bytes > 0) [enc setBytes:dims length:dims_bytes atIndex:3];
    [enc dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, [yb contents], m * sizeof(float));
    return (int)m;
}

// ── Public API ────────────────────────────────────────────────────────────

int metal_fp16_gemv(MetalGpu* gpu, const uint16_t* w, size_t w_bytes,
                    const float* x, size_t k, float* y, size_t m) {
    uint32_t d[2] = {(uint32_t)m, (uint32_t)k};
    return dispatch_kernel(gpu, &gpu->pipe_fp16_gemv, "fp16_gemv", w, w_bytes, x, k, y, m, d, 8);
}

int metal_expert_gemv(MetalGpu* gpu, const uint8_t* q4, size_t q4_len,
                      const float* x, size_t k, float* y, size_t m, size_t n_blocks) {
    uint32_t d[4] = {(uint32_t)m, (uint32_t)k, (uint32_t)n_blocks, 0};
    return dispatch_kernel(gpu, &gpu->pipe_q4_gemv, "q4_expert_gemv", q4, q4_len, x, k, y, m, d, 16);
}

int metal_multi_expert_gemv(MetalGpu* gpu,
    const uint8_t* all_q4, size_t q4_len,
    const uint32_t* expert_offsets, size_t n_offsets,
    const float* x, size_t k, float* y,
    const uint32_t* output_offsets, uint32_t n_experts, size_t total_elems)
{
    if (!gpu) return -1;

    id<MTLBuffer> qb = ensure_buf(gpu->device, &gpu->w_buf,  &gpu->w_cap, q4_len);
    id<MTLBuffer> xb = ensure_buf(gpu->device, &gpu->x_buf,  &gpu->x_cap, k * sizeof(float));
    id<MTLBuffer> yb = ensure_buf(gpu->device, &gpu->y_buf,  &gpu->y_cap, total_elems * sizeof(float));
    id<MTLBuffer> ob = [gpu->device newBufferWithBytes:expert_offsets length:n_offsets * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob_out = [gpu->device newBufferWithBytes:output_offsets length:n_experts * 4 options:MTLResourceStorageModeShared];

    memcpy([qb contents], all_q4, q4_len);
    memcpy([xb contents], x, k * sizeof(float));

    id<MTLComputePipelineState> pipe = get_pipe(gpu, &gpu->pipe_multi_expert, "multi_expert_gemv");
    if (!pipe) return -2;

    id<MTLCommandBuffer> cmd = [gpu->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:qb offset:0 atIndex:0];
    [enc setBuffer:ob offset:0 atIndex:1];
    [enc setBuffer:xb offset:0 atIndex:2];
    [enc setBuffer:yb offset:0 atIndex:3];
    [enc setBuffer:ob_out offset:0 atIndex:4];
    [enc setBytes:&n_experts length:4 atIndex:5];

    uint32_t total_rows = 0;
    for (uint32_t e = 0; e < n_experts; e++) total_rows += expert_offsets[e * 4];
    [enc dispatchThreads:MTLSizeMake(total_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, [yb contents], total_elems * sizeof(float));
    return (int)total_elems;
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused GQA Attention (per-layer weight loading + dispatch + O-proj)
// ═══════════════════════════════════════════════════════════════════════════

// ── Preload immutable resources (RoPE tables, once) ─────────────────────

int metal_gqa_init(MetalGpu* gpu,
                   const float* rope_cos, const float* rope_sin, uint32_t max_seq) {
    if (!gpu) return -1;
    size_t rope_bytes = max_seq * 128 * sizeof(float);
    if (!gpu->gqa_cos) {
        gpu->gqa_cos = [gpu->device newBufferWithBytes:rope_cos length:rope_bytes options:MTLResourceStorageModeShared];
        gpu->gqa_sin = [gpu->device newBufferWithBytes:rope_sin length:rope_bytes options:MTLResourceStorageModeShared];
        [gpu->gqa_cos retain]; [gpu->gqa_sin retain];
    }
    if (!gpu->gqa_h) {
        gpu->gqa_h = [gpu->device newBufferWithLength:2048*sizeof(float) options:MTLResourceStorageModeShared];
        [gpu->gqa_h retain];
    }
    if (!gpu->gqa_out) {
        gpu->gqa_out = [gpu->device newBufferWithLength:4096*sizeof(float) options:MTLResourceStorageModeShared];
        [gpu->gqa_out retain];
    }
    return 0;
}

// ── Load per-layer GQA weights into persistent buffers ──────────────────

int metal_gqa_load_weights(MetalGpu* gpu,
                           const uint16_t* w_qkv, size_t w_qkv_bytes,
                           const uint16_t* w_o, size_t w_o_bytes) {
    if (!gpu) return -1;
    // W_qkv: resize buffer if needed, then memcpy
    if (!gpu->gqa_w || gpu->gqa_w_cap < w_qkv_bytes) {
        if (gpu->gqa_w) { [gpu->gqa_w release]; gpu->gqa_w = nil; }
        gpu->gqa_w = [gpu->device newBufferWithBytes:w_qkv length:w_qkv_bytes options:MTLResourceStorageModeShared];
        [gpu->gqa_w retain];
        gpu->gqa_w_cap = w_qkv_bytes;
    } else {
        memcpy([gpu->gqa_w contents], w_qkv, w_qkv_bytes);
    }
    // W_o: same pattern
    if (!gpu->gqa_wo || gpu->gqa_wo_cap < w_o_bytes) {
        if (gpu->gqa_wo) { [gpu->gqa_wo release]; gpu->gqa_wo = nil; }
        gpu->gqa_wo = [gpu->device newBufferWithBytes:w_o length:w_o_bytes options:MTLResourceStorageModeShared];
        [gpu->gqa_wo retain];
        gpu->gqa_wo_cap = w_o_bytes;
    } else {
        memcpy([gpu->gqa_wo contents], w_o, w_o_bytes);
    }
    return 0;
}

// ── Dispatch fused GQA (QKV + RoPE + attention + Q-gate) ──────────────────

int metal_fused_gqa(
    MetalGpu* gpu,
    const float* h, uint32_t pos, uint32_t seq_len, uint32_t max_seq,
    float* k_cache, float* v_cache, size_t kv_bytes,
    float* attn_out)
{
    if (!gpu) return -1;

    // Ensure KV cache buffers (lazy alloc, resized if needed)
    if (!gpu->gqa_k || gpu->gqa_kv_cap < kv_bytes) {
        if (gpu->gqa_k) { [gpu->gqa_k release]; [gpu->gqa_v release]; }
        gpu->gqa_k = [gpu->device newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
        gpu->gqa_v = [gpu->device newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
        [gpu->gqa_k retain]; [gpu->gqa_v retain];
        gpu->gqa_kv_cap = kv_bytes;
    }
    // Copy input data to persistent buffers
    memcpy([gpu->gqa_h contents], h, 2048*sizeof(float));
    memcpy([gpu->gqa_k contents], k_cache, kv_bytes);
    memcpy([gpu->gqa_v contents], v_cache, kv_bytes);

    // Pipeline (cached)
    if (!gpu->pipe_gqa) {
        id<MTLFunction> fn = [gpu->library newFunctionWithName:@"fused_gqa"];
        if (!fn) return -2;
        gpu->pipe_gqa = [gpu->device newComputePipelineStateWithFunction:fn error:nil];
        [gpu->pipe_gqa retain];
    }

    id<MTLCommandBuffer> cmd = [gpu->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gpu->pipe_gqa];
    [enc setBuffer:gpu->gqa_w    offset:0 atIndex:0];  // f16 W_qkv
    [enc setBuffer:gpu->gqa_h    offset:0 atIndex:1];  // h
    [enc setBuffer:gpu->gqa_k    offset:0 atIndex:2];  // k_cache
    [enc setBuffer:gpu->gqa_v    offset:0 atIndex:3];  // v_cache
    [enc setBuffer:gpu->gqa_cos  offset:0 atIndex:4];  // RoPE cos
    [enc setBuffer:gpu->gqa_sin  offset:0 atIndex:5];  // RoPE sin
    [enc setBytes:&pos     length:4 atIndex:6];
    [enc setBytes:&seq_len length:4 atIndex:7];
    [enc setBytes:&max_seq length:4 atIndex:8];
    [enc setBuffer:gpu->gqa_out  offset:0 atIndex:9];  // attn_out
    [enc dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(k_cache, [gpu->gqa_k contents], kv_bytes);
    memcpy(v_cache, [gpu->gqa_v contents], kv_bytes);
    memcpy(attn_out, [gpu->gqa_out contents], 4096*sizeof(float));
    return 4096;
}

// ── Dispatch GQA O-proj ─────────────────────────────────────────────────

int metal_gqa_oproj(MetalGpu* gpu,
                    const uint16_t* w_o, size_t w_o_bytes,
                    const float* attn_out,
                    float* output, uint32_t m, uint32_t k) {
    if (!gpu) return -1;
    // Load W_o into persistent buffer
    metal_gqa_load_weights(gpu, NULL, 0, w_o, w_o_bytes);

    // Pipeline (cached)
    if (!gpu->pipe_gqa_oproj) {
        id<MTLFunction> fn = [gpu->library newFunctionWithName:@"gqa_oproj_f16"];
        if (!fn) return -2;
        gpu->pipe_gqa_oproj = [gpu->device newComputePipelineStateWithFunction:fn error:nil];
        [gpu->pipe_gqa_oproj retain];
    }

    // Load attn_out into persistent buffer
    memcpy([gpu->gqa_out contents], attn_out, k * sizeof(float));

    id<MTLBuffer> out_buf = ensure_buf(gpu->device, &gpu->y_buf, &gpu->y_cap, m * sizeof(float));

    id<MTLCommandBuffer> cmd = [gpu->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gpu->pipe_gqa_oproj];
    [enc setBuffer:gpu->gqa_wo   offset:0 atIndex:0];
    [enc setBuffer:gpu->gqa_out  offset:0 atIndex:1];
    [enc setBuffer:out_buf       offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(output, [out_buf contents], m * sizeof(float));
    return (int)m;
}
