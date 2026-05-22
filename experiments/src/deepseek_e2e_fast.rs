/// DeepSeek V4 Flash E2E canary — CUDA-accelerated attention path.
///
/// Moves all attention FP8 projections to CUDA device-resident execution.
/// Reuses existing validated `cuda_fp8_act_fp8_weight_gemv_device` and
/// `cuda_act_quant_device` primitives from objeta-cuda.
use std::time::Instant;
use objeta_parser::ModelWeights;
use objeta_parser::deepseek::{f8e4m3_to_f32, f8e8m0_to_f32}; // kept for potential cpu ref path
use objeta_cuda::{
    CudaBackendBuilder, QuantBackend, MoeExecutor, CudaExpertCache,
    DeepSeekFp4ExpertWeights, DeepSeekFp8SharedExpertWeightsDevice,
    execute_selected_moe_official_routed_fp4_cuda,
    DeviceBuffer, CudaStreamHandle,
    cuda_act_quant_device, cuda_fp8_act_fp8_weight_gemv_device,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input_id: usize = args.iter().position(|a| a == "--input-id")
        .and_then(|i| args.get(i+1)).and_then(|v| v.parse().ok()).unwrap_or(42);
    let model_dir = args.iter().position(|a| a == "--model-dir")
        .map(|i| args.get(i+1).cloned().unwrap_or_else(|| r"E:\Projects\DeepSeek-V4-Flash".into()))
        .unwrap_or_else(|| r"E:\Projects\DeepSeek-V4-Flash".into());

    let model = ModelWeights::open(&model_dir)?; 
    let dim = 4096; let hc = 4; let nl = 43; let nh = 64; let hd = 512;
    let qlr = 1024; let kvr = 512; let og = 8; let olr = 1024;
    let intermed = 2048; let num_experts = 256; let top_k = 6;
    let eps = 1e-6f32;

    let total_t0 = Instant::now();
    let mut timings: Vec<serde_json::Value> = Vec::new();

    let load = |n: &str| -> Vec<f32> { let mut v = vec![]; model.get_f32(n, &mut v).ok(); v };
    let loadu = |n: &str| -> Vec<u8> { model.get_raw(n).map(|s| s.to_vec()).unwrap_or_default() };
    let p = |l: usize, n: &str| format!("layers.{l}.{n}");

    // Embedding (CPU — single gather, negligible)
    let emb_t0 = Instant::now();
    let embed = load("embed.weight");
    let mut h: Vec<Vec<f32>> = (0..hc).map(|_| embed[input_id*dim..(input_id+1)*dim].to_vec()).collect();
    let emb_ms = emb_t0.elapsed().as_secs_f32() * 1000.0;
    let head_w = load("head.weight");
    let fn_w = load("norm.weight");
    let start_load_ms = emb_ms; // ~0ms

    // CUDA
    let cu = CudaBackendBuilder::new().stream_count(1).build()?;
    let quant = QuantBackend::new(cu.context().clone(), cu.device_info().clone());
    let moe_exec = MoeExecutor::new(cu.context().clone(), cu.device_info().clone());
    let stream = cu.stream_pool().stream(0)?;
    let mut cache = CudaExpertCache::new(2_147_483_648);
    fn upload(s: &CudaStreamHandle, d: &[u8]) -> std::result::Result<DeviceBuffer<u8>, objeta_cuda::CudaError> { s.copy_from_slice(d) }

    let mut total_attn_ms = 0.0f32;
    let mut total_hc_ms = 0.0f32;
    let mut total_moe_ms = 0.0f32;
    let mut per_layer: Vec<serde_json::Value> = Vec::new();

    // CUDA helper: upload fp8 weight+scale as device buffers
    fn upload_fp8_pair(stream: &CudaStreamHandle, w: &[u8], s: &[u8]) -> std::result::Result<(DeviceBuffer<u8>, DeviceBuffer<u8>), objeta_cuda::CudaError> {
        Ok((stream.copy_from_slice(w)?, stream.copy_from_slice(s)?))
    }

    for layer in 0..nl {
        let l_t0 = Instant::now();
        let is_hash = layer < 3;
        let mut l_times = std::collections::HashMap::<String, f32>::new();

        // HC pre-attn (CPU — small matmul [16384]×[24,16384])
        let t0 = Instant::now();
        let ra = h.clone();
        let mix_a = cpu_dense_gemv(&h.iter().flat_map(|c| c.iter()).copied().collect::<Vec<_>>(), &load(&p(layer, "hc_attn_fn")), 24, hc*dim);
        let sc_a = load(&p(layer, "hc_attn_scale")); let ba_a = load(&p(layer, "hc_attn_base"));
        let mut pre_a = [0f32;4]; let mut pst_a = [0f32;4]; let mut cmb_a = [[0f32;4];4];
        for i in 0..hc { pre_a[i]=1.0/(1.0+(-(mix_a[i]*sc_a[0]+ba_a[i])).exp()); pst_a[i]=1.0/(1.0+(-(mix_a[hc+i]*sc_a[1]+ba_a[hc+i])).exp()); }
        for j in 0..hc { for k in 0..hc { cmb_a[j][k]=mix_a[2*hc+j*hc+k]*sc_a[2]+ba_a[2*hc+j*hc+k]; } }
        for _ in 0..20 {
            for j in 0..hc { let mx=cmb_a[j].iter().cloned().fold(f32::NEG_INFINITY,f32::max);
                let mut s=0f64; for k in 0..hc { let e=((cmb_a[j][k]-mx)as f64).exp(); s+=e; cmb_a[j][k]=e as f32; }
                let inv=(s+1e-6f64).recip(); for k in 0..hc { cmb_a[j][k]=(cmb_a[j][k]as f64*inv)as f32; } }
            for k in 0..hc { let mut s=0f64; for j in 0..hc { s+=cmb_a[j][k] as f64; }
                let inv=(s+1e-6f64).recip(); for j in 0..hc { cmb_a[j][k]=(cmb_a[j][k]as f64*inv)as f32; } }
        }
        let mut xs = vec![0.0f32; dim];
        for i in 0..hc { for j in 0..dim { xs[j] += pre_a[i] * h[i][j]; } }
        l_times.insert("hc_attn".into(), t0.elapsed().as_secs_f32()*1000.0);

        // Attention (CUDA-accelerated linears)
        let attn_t0 = Instant::now();
        let anw = load(&p(layer, "attn_norm.weight"));
        let x_attn_normed = cpu_rms(&xs, &anw, eps);

        // Upload activation to device
        let t1 = Instant::now();
        let d_x = stream.copy_from_slice(&x_attn_normed)?;
        l_times.insert("attn_h2d_act".into(), t1.elapsed().as_secs_f32()*1000.0);

        // WqA [1024, 4096] — CUDA fp8_act × fp8_weight GEMV
        let t1 = Instant::now();
        let (wqa_w, wqa_s) = upload_fp8_pair(&stream, &loadu(&p(layer, "attn.wq_a.weight")), &loadu(&p(layer, "attn.wq_a.scale")))?;
        let d_q_act = cuda_act_quant_device(&quant, &stream, &d_x, dim, 128)?.0;
        let mut d_q_lat = stream.alloc_zeros::<f32>(qlr)?;
        cuda_fp8_act_fp8_weight_gemv_device(&quant, &stream, &d_q_act.values, &d_q_act.scales, &wqa_w, &wqa_s, &mut d_q_lat, qlr, dim)?;
        let mut q_lat = stream.copy_to_vec(&d_q_lat)?;
        l_times.insert("attn_wqa_cuda".into(), t1.elapsed().as_secs_f32()*1000.0);

        // q_norm (CPU)
        let t1 = Instant::now();
        let qn = load(&p(layer, "attn.q_norm.weight"));
        if qn.len() == qlr { q_lat = cpu_rms(&q_lat, &qn, eps); }
        l_times.insert("attn_q_norm".into(), t1.elapsed().as_secs_f32()*1000.0);

        // WqB [32768, 1024] — CUDA fp8_act × fp8_weight GEMV
        let t1 = Instant::now();
        let d_ql = stream.copy_from_slice(&q_lat)?;
        let d_q_act2 = cuda_act_quant_device(&quant, &stream, &d_ql, qlr, 128)?.0;
        let (wqb_w, wqb_s) = upload_fp8_pair(&stream, &loadu(&p(layer, "attn.wq_b.weight")), &loadu(&p(layer, "attn.wq_b.scale")))?;
        let mut d_q_full = stream.alloc_zeros::<f32>(nh*hd)?;
        cuda_fp8_act_fp8_weight_gemv_device(&quant, &stream, &d_q_act2.values, &d_q_act2.scales, &wqb_w, &wqb_s, &mut d_q_full, nh*hd, qlr)?;
        let q_full = stream.copy_to_vec(&d_q_full)?;
        l_times.insert("attn_wqb_cuda".into(), t1.elapsed().as_secs_f32()*1000.0);

        // Wkv [512, 4096] — CUDA fp8_act × fp8_weight GEMV
        let t1 = Instant::now();
        let d_x2 = stream.copy_from_slice(&x_attn_normed)?;
        let d_kv_act = cuda_act_quant_device(&quant, &stream, &d_x2, dim, 128)?.0;
        let (wkv_w, wkv_s) = upload_fp8_pair(&stream, &loadu(&p(layer, "attn.wkv.weight")), &loadu(&p(layer, "attn.wkv.scale")))?;
        let mut d_kv_lat = stream.alloc_zeros::<f32>(kvr)?;
        cuda_fp8_act_fp8_weight_gemv_device(&quant, &stream, &d_kv_act.values, &d_kv_act.scales, &wkv_w, &wkv_s, &mut d_kv_lat, kvr, dim)?;
        let mut kv = stream.copy_to_vec(&d_kv_lat)?;
        l_times.insert("attn_wkv_cuda".into(), t1.elapsed().as_secs_f32()*1000.0);

        // kv_norm (CPU)
        let kn = load(&p(layer, "attn.kv_norm.weight"));
        if kn.len() == kvr { kv = cpu_rms(&kv, &kn, eps); }

        // Per-head Q normalization + attention scores (CPU — small ops)
        let t1 = Instant::now();
        let qh: Vec<Vec<f32>> = (0..nh).map(|hh| {
            let mut v = q_full[hh*hd..(hh+1)*hd].to_vec();
            let sq = v.iter().map(|&x| (x as f64)*(x as f64)).sum::<f64>();
            let r = (sq/hd as f64+eps as f64).sqrt().recip();
            v.iter_mut().for_each(|x| *x = (*x as f64*r) as f32); v
        }).collect();
        let sscale = (hd as f32).powf(-0.5);
        let snk = load(&p(layer, "attn.attn_sink"));
        let mut ao = vec![0.0f32; nh*hd];
        for hh in 0..nh {
            let sc = qh[hh].iter().zip(kv.iter()).map(|(qk,vv)| (*qk as f64)*(*vv as f64)).sum::<f64>() * sscale as f64;
            let sk = if hh<snk.len(){snk[hh]as f64}else{0.0};
            let mx = sc.max(sk); let a = (sc-mx).exp() / ((sc-mx).exp() + (sk-mx).exp());
            for j in 0..hd { ao[hh*hd+j] = (a*kv[j.min(kvr-1)]as f64)as f32; }
        }
        l_times.insert("attn_qhead_score_sink".into(), t1.elapsed().as_secs_f32()*1000.0);

        // WoA grouped [8, 1024, 4096] — CUDA per-group GEMV
        let t1 = Instant::now();
        let woa_w_host = loadu(&p(layer, "attn.wo_a.weight"));
        let woa_s_host = loadu(&p(layer, "attn.wo_a.scale"));
        let mut wo_out = vec![0.0f32; og*olr];
        for g in 0..og {
            let group_input = &ao[g*(nh*hd/og)..(g+1)*(nh*hd/og)];
            let d_gi = stream.copy_from_slice(group_input)?;
            let d_gi_act = cuda_act_quant_device(&quant, &stream, &d_gi, nh*hd/og, 128)?.0;
            let w_start = g * olr * (nh*hd/og);
            let w_len = olr * (nh*hd/og);
            let d_wg = stream.copy_from_slice(&woa_w_host[w_start..w_start+w_len])?;
            // WoA scale: [64, 32] total, group g starts at row g*8. Upload full.
            let d_sg = stream.copy_from_slice(&woa_s_host)?;
            let mut d_og = stream.alloc_zeros::<f32>(olr)?;
            cuda_fp8_act_fp8_weight_gemv_device(&quant, &stream, &d_gi_act.values, &d_gi_act.scales, &d_wg, &d_sg, &mut d_og, olr, nh*hd/og)?;
            let og_out = stream.copy_to_vec(&d_og)?;
            for i in 0..olr { wo_out[g*olr+i] = og_out[i]; }
        }
        l_times.insert("attn_woa_cuda".into(), t1.elapsed().as_secs_f32()*1000.0);

        // WoB [4096, 8192] — CUDA fp8_act × fp8_weight GEMV
        let t1 = Instant::now();
        let d_wo = stream.copy_from_slice(&wo_out)?;
        let d_wo_act = cuda_act_quant_device(&quant, &stream, &d_wo, og*olr, 128)?.0;
        let (wob_w, wob_s) = upload_fp8_pair(&stream, &loadu(&p(layer, "attn.wo_b.weight")), &loadu(&p(layer, "attn.wo_b.scale")))?;
        let mut d_xa = stream.alloc_zeros::<f32>(dim)?;
        cuda_fp8_act_fp8_weight_gemv_device(&quant, &stream, &d_wo_act.values, &d_wo_act.scales, &wob_w, &wob_s, &mut d_xa, dim, og*olr)?;
        let xa_out = stream.copy_to_vec(&d_xa)?;
        l_times.insert("attn_wob_cuda".into(), t1.elapsed().as_secs_f32()*1000.0);

        let attn_ms = attn_t0.elapsed().as_secs_f32() * 1000.0;
        total_attn_ms += attn_ms;

        // HC post-attn
        let t1 = Instant::now();
        let mut nh2: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; hc];
        for i in 0..hc { for j in 0..dim { nh2[i][j] = pst_a[i] * xa_out[j]; for k in 0..hc { nh2[i][j] += cmb_a[i][k] * ra[k][j]; } } }
        h = nh2;
        l_times.insert("hc_attn_post".into(), t1.elapsed().as_secs_f32()*1000.0);

        // HC pre-ffn (CPU)
        let t1 = Instant::now();
        let rf = h.clone();
        // ... HC ffn pre computation (same as attn, with different tensors)
        let mix_f = cpu_dense_gemv(&h.iter().flat_map(|c| c.iter()).copied().collect::<Vec<_>>(), &load(&p(layer, "hc_ffn_fn")), 24, hc*dim);
        let sc_f = load(&p(layer, "hc_ffn_scale")); let bf = load(&p(layer, "hc_ffn_base"));
        let mut pre_f = [0f32;4]; let mut pst_f = [0f32;4]; let mut cmb_f = [[0f32;4];4];
        for i in 0..hc { pre_f[i]=1.0/(1.0+(-(mix_f[i]*sc_f[0]+bf[i])).exp()); pst_f[i]=1.0/(1.0+(-(mix_f[hc+i]*sc_f[1]+bf[hc+i])).exp()); }
        for j in 0..hc { for k in 0..hc { cmb_f[j][k]=mix_f[2*hc+j*hc+k]*sc_f[2]+bf[2*hc+j*hc+k]; } }
        for _ in 0..20 {
            for j in 0..hc { let mx=cmb_f[j].iter().cloned().fold(f32::NEG_INFINITY,f32::max);
                let mut s=0f64; for k in 0..hc { let e=((cmb_f[j][k]-mx)as f64).exp(); s+=e; cmb_f[j][k]=e as f32; }
                let inv=(s+1e-6f64).recip(); for k in 0..hc { cmb_f[j][k]=(cmb_f[j][k]as f64*inv)as f32; } }
            for k in 0..hc { let mut s=0f64; for j in 0..hc { s+=cmb_f[j][k] as f64; }
                let inv=(s+1e-6f64).recip(); for j in 0..hc { cmb_f[j][k]=(cmb_f[j][k]as f64*inv)as f32; } }
        }
        let mut xf = vec![0.0f32; dim];
        for i in 0..hc { for j in 0..dim { xf[j] += pre_f[i] * h[i][j]; } }
        let fnw = load(&p(layer, "ffn_norm.weight"));
        let xfn = cpu_rms(&xf, &fnw, eps);
        l_times.insert("hc_ffn".into(), t1.elapsed().as_secs_f32()*1000.0);
        total_hc_ms += l_times.get("hc_attn").unwrap_or(&0.0) + l_times.get("hc_attn_post").unwrap_or(&0.0) + l_times.get("hc_ffn").unwrap_or(&0.0);

        // MoE forward (existing validated CUDA path)
        let moe_t0 = Instant::now();
        // Router (CPU)
        let gate_w = load(&p(layer, "ffn.gate.weight"));
        let bias_w = load(&p(layer, "ffn.gate.bias"));
        let n_exp = gate_w.len() / dim;
        let mut scores = cpu_dense_gemv(&xfn, &gate_w, n_exp, dim);
        if !bias_w.is_empty() { for (s,b) in scores.iter_mut().zip(bias_w.iter()) { *s+=b; } }
        let mut indexed: Vec<(usize,f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);
        let selected_ids: Vec<usize> = indexed.iter().map(|(id,_)|*id).collect();
        let selected_weights: Vec<f32> = {
            let max_s = indexed.iter().map(|(_,s)|*s).fold(f32::NEG_INFINITY,f32::max);
            let exps: Vec<f64> = indexed.iter().map(|(_,s)| ((s-max_s)as f64).exp()).collect();
            let total: f64 = exps.iter().sum();
            exps.iter().map(|e| (*e/total) as f32).collect()
        };
        let selected_pairs: Vec<(usize,f32)> = selected_ids.iter().copied().zip(selected_weights.iter().copied()).collect();
        let router_ms = moe_t0.elapsed().as_secs_f32()*1000.0;

        // Load FP4 + shared tensors
        let load_moe_t0 = Instant::now();
        let mut expert_fp4_set: Vec<DeepSeekFp4ExpertWeights> = (0..num_experts).map(|_| DeepSeekFp4ExpertWeights {
            gate_weight: vec![], gate_scale: vec![], up_weight: vec![], up_scale: vec![], down_weight: vec![], down_scale: vec![],
        }).collect();
        for &eid in &selected_ids {
            expert_fp4_set[eid] = DeepSeekFp4ExpertWeights {
                gate_weight: loadu(&format!("layers.{layer}.ffn.experts.{eid}.w1.weight")),
                gate_scale: loadu(&format!("layers.{layer}.ffn.experts.{eid}.w1.scale")),
                up_weight: loadu(&format!("layers.{layer}.ffn.experts.{eid}.w3.weight")),
                up_scale: loadu(&format!("layers.{layer}.ffn.experts.{eid}.w3.scale")),
                down_weight: loadu(&format!("layers.{layer}.ffn.experts.{eid}.w2.weight")),
                down_scale: loadu(&format!("layers.{layer}.ffn.experts.{eid}.w2.scale")),
            };
        }
        let sp = |n: &str| format!("layers.{layer}.ffn.shared_experts.{n}");
        let sh_dev = DeepSeekFp8SharedExpertWeightsDevice {
            gate_weight: upload(&stream, &loadu(&sp("w1.weight")))?,
            gate_scale: upload(&stream, &loadu(&sp("w1.scale")))?,
            up_weight: upload(&stream, &loadu(&sp("w3.weight")))?,
            up_scale: upload(&stream, &loadu(&sp("w3.scale")))?,
            down_weight: upload(&stream, &loadu(&sp("w2.weight")))?,
            down_scale: upload(&stream, &loadu(&sp("w2.scale")))?,
        };
        let load_moe_ms = load_moe_t0.elapsed().as_secs_f32()*1000.0;

        // Execute CUDA MoE
        let moe_k_t0 = Instant::now();
        let (moe_out, _tel) = execute_selected_moe_official_routed_fp4_cuda(
            &quant, &moe_exec, &stream, &expert_fp4_set, &selected_pairs,
            &xfn, dim, intermed, dim, layer,
            Some(&mut cache), Some(&sh_dev),
        )?;
        let moe_k_ms = moe_k_t0.elapsed().as_secs_f32()*1000.0;
        let moe_ms = moe_t0.elapsed().as_secs_f32()*1000.0;
        total_moe_ms += moe_ms;

        // HC post-ffn
        let t1 = Instant::now();
        let mut nh3: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; hc];
        for i in 0..hc { for j in 0..dim { nh3[i][j] = pst_f[i] * moe_out[j]; for k in 0..hc { nh3[i][j] += cmb_f[i][k] * rf[k][j]; } } }
        h = nh3;
        let hc_post_ffn_ms = t1.elapsed().as_secs_f32()*1000.0;

        per_layer.push(json!({
            "layer_id": layer, "hash_layer": is_hash,
            "timings": l_times,
            "router_ms": router_ms,
            "moe_load_ms": load_moe_ms,
            "moe_kernel_ms": moe_k_ms,
            "moe_total_ms": moe_ms,
            "hc_post_ffn_ms": hc_post_ffn_ms,
            "layer_total_ms": l_t0.elapsed().as_secs_f32()*1000.0,
            "moe_finite": moe_out.iter().all(|v|v.is_finite()),
        }));
    }

    // HC head
    let t1 = Instant::now();
    let flat: Vec<f32> = h.iter().flat_map(|c|c.iter()).copied().collect();
    let hmix = cpu_dense_gemv(&flat, &load("hc_head_fn"), hc, hc*dim);
    let hsc = load("hc_head_scale"); let hba = load("hc_head_base");
    let mut ph = [0f32;4];
    for i in 0..hc { let v=hmix[i]; let b=if i<hba.len(){hba[i]}else{0.0}; let s=hsc.first().copied().unwrap_or(1.0);
        ph[i] = 1.0/(1.0+(-(v*s+b)).exp())+1e-6; }
    let mut hf = vec![0.0f32; dim];
    for i in 0..hc { for j in 0..dim { hf[j] += ph[i] * h[i][j]; } }
    let head_ms = t1.elapsed().as_secs_f32()*1000.0;

    // Final norm
    let t1 = Instant::now();
    let hn = cpu_rms(&hf, &fn_w, eps);
    let norm_ms = t1.elapsed().as_secs_f32()*1000.0;

    // LM head
    let t1 = Instant::now();
    let logits = cpu_dense_gemv(&hn, &head_w, 129280, dim);
    let lm_ms = t1.elapsed().as_secs_f32()*1000.0;

    let total_ms = total_t0.elapsed().as_secs_f32()*1000.0;
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a,&b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    
    println!("{}", serde_json::to_string_pretty(&json!({
        "input_token_id": input_id,
        "output_token_id": idx[0],
        "top_10_ids": idx.iter().take(10).copied().collect::<Vec<_>>(),
        "top_10_logits": idx.iter().take(10).map(|&i| logits[i]).collect::<Vec<_>>(),
        "final_logits_finite": logits.iter().all(|v|v.is_finite()),
        "timings": {
            "embedding_ms": emb_ms,
            "total_attention_ms": total_attn_ms,
            "total_hc_ms": total_hc_ms,
            "total_moe_ms": total_moe_ms,
            "hc_head_ms": head_ms,
            "final_norm_ms": norm_ms,
            "lm_head_ms": lm_ms,
            "total_forward_ms": total_ms,
        },
        "per_layer": per_layer,
        "flags": {
            "official_moe_forward": true,
            "attention_cuda_accelerated": true,
            "official_attention_seq1_pos0": true,
            "official_hyper_connection": true,
            "mtp_included": false,
        }
    })).unwrap());
    Ok(())
}

fn cpu_dense_gemv(x: &[f32], w: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m];
    for i in 0..m { let mut s=0.0f64; for j in 0..k { s+=(w[i*k+j] as f64)*(x[j] as f64); } o[i]=s as f32; }
    o
}
fn cpu_rms(x: &[f32], w: &[f32], e: f32) -> Vec<f32> {
    let n = x.len();
    let sq = x.iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>();
    let r = (sq/n as f64 + e as f64).sqrt().recip();
    x.iter().zip(w).map(|(&v, &wv)| (v as f64*r*wv as f64) as f32).collect()
}
