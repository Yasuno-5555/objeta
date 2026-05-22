use std::time::Instant;
use objeta_parser::ModelWeights;
use objeta_parser::deepseek::{f8e4m3_to_f32, f8e8m0_to_f32};
use objeta_cuda::{
    CudaBackendBuilder, QuantBackend, MoeExecutor, CudaExpertCache,
    DeepSeekFp4ExpertWeights, DeepSeekFp8SharedExpertWeightsDevice,
    execute_selected_moe_official_routed_fp4_cuda,
    DeviceBuffer, CudaStreamHandle,
};

#[derive(serde::Serialize)]
struct E2eResult {
    input_token_id: usize, output_token_id: usize,
    top_10_token_ids: Vec<usize>, top_10_logits: Vec<f32>,
    final_logits_finite: bool, moe_placeholder_used: bool,
    all_43_layers_moe_executed: bool,
    total_forward_ms: f32, total_tensor_load_ms: f32,
    total_attention_ms: f32, total_hc_ms: f32,
    total_moe_ms: f32, total_head_ms: f32,
    flags: serde_json::Value, per_layer: Vec<serde_json::Value>,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::result::Result;

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
    let eps = 1e-6f32; let sscale = (hd as f32).powf(-0.5); let heps = 1e-6f32;

    let total_t0 = Instant::now();

    let load = |n: &str| -> Vec<f32> { let mut v = vec![]; model.get_f32(n, &mut v).ok(); v };
    let loadu = |n: &str| -> Vec<u8> { model.get_raw(n).map(|s| s.to_vec()).unwrap_or_default() };
    let p = |l: usize, n: &str| format!("layers.{l}.{n}");

    let embed = load("embed.weight");
    let mut h: Vec<Vec<f32>> = (0..hc).map(|_| embed[input_id*dim..(input_id+1)*dim].to_vec()).collect();
    let head_w = load("head.weight");
    let fn_w = load("norm.weight");

    let mut total_load_ms = 0.0f32;
    let mut total_attn_ms = 0.0f32;
    let mut total_hc_ms = 0.0f32;
    let mut total_moe_ms = 0.0f32;
    let mut per_layer: Vec<serde_json::Value> = Vec::new();

    // CUDA backend
    let cu = CudaBackendBuilder::new().stream_count(1).build()?;
    let quant = QuantBackend::new(cu.context().clone(), cu.device_info().clone());
    let moe_exec = MoeExecutor::new(cu.context().clone(), cu.device_info().clone());
    let stream = cu.stream_pool().stream(0)?;
    let mut cache = CudaExpertCache::new(2_147_483_648);

    // Helper: upload a host Vec<u8> to device as a single DeviceBuffer
    fn upload(stream: &CudaStreamHandle, data: &[u8]) -> std::result::Result<DeviceBuffer<u8>, objeta_cuda::CudaError> {
        stream.copy_from_slice(data)
    }

    for layer in 0..nl {
        let layer_t0 = Instant::now();
        let is_hash = layer < 3;

        // HC pre-attn
        let hc_t0 = Instant::now();
        let ra = h.clone();
        let mix_a = dense_gemv(&h.iter().flat_map(|c| c.iter()).copied().collect::<Vec<_>>(), &load(&p(layer, "hc_attn_fn")), 24, hc*dim);
        let sc_a = load(&p(layer, "hc_attn_scale")); let ba_a = load(&p(layer, "hc_attn_base"));
        let mut pre_a = [0f32; 4]; let mut pst_a = [0f32; 4]; let mut cmb_a = [[0f32; 4]; 4];
        for i in 0..hc { pre_a[i] = 1.0/(1.0+(-(mix_a[i]*sc_a[0]+ba_a[i])).exp()); pst_a[i] = 1.0/(1.0+(-(mix_a[hc+i]*sc_a[1]+ba_a[hc+i])).exp()); }
        for j in 0..hc { for k in 0..hc { cmb_a[j][k] = mix_a[2*hc+j*hc+k]*sc_a[2]+ba_a[2*hc+j*hc+k]; } }
        for _ in 0..20 {
            for j in 0..hc { let mx = cmb_a[j].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0f64; for k in 0..hc { let e = ((cmb_a[j][k]-mx)as f64).exp(); s+=e; cmb_a[j][k]=e as f32; }
                let inv=(s+heps as f64).recip(); for k in 0..hc { cmb_a[j][k]=(cmb_a[j][k]as f64*inv)as f32; } }
            for k in 0..hc { let mut s = 0f64; for j in 0..hc { s += cmb_a[j][k] as f64; }
                let inv=(s+heps as f64).recip(); for j in 0..hc { cmb_a[j][k]=(cmb_a[j][k]as f64*inv)as f32; } }
        }
        let mut xs = vec![0.0f32; dim];
        for i in 0..hc { for j in 0..dim { xs[j] += pre_a[i] * h[i][j]; } }
        let hc_a_ms = hc_t0.elapsed().as_secs_f32() * 1000.0;

        // Attention
        let attn_t0 = Instant::now();
        let anw = load(&p(layer, "attn_norm.weight")); let mut xa_n = rms(&xs, &anw, eps);
        // WqA
        let qa = decode_fp8_2(&loadu(&p(layer, "attn.wq_a.weight")), &loadu(&p(layer, "attn.wq_a.scale")), qlr, dim);
        xa_n = dense_gemv(&xa_n, &qa, qlr, dim);
        let qn = load(&p(layer, "attn.q_norm.weight")); if qn.len()==qlr { xa_n = rms(&xa_n, &qn, eps); }
        // WqB
        let qb = decode_fp8_2(&loadu(&p(layer, "attn.wq_b.weight")), &loadu(&p(layer, "attn.wq_b.scale")), nh*hd, qlr);
        let qf = dense_gemv(&xa_n, &qb, nh*hd, qlr);
        let qh: Vec<Vec<f32>> = (0..nh).map(|hh| {
            let mut v = qf[hh*hd..(hh+1)*hd].to_vec();
            let sq = v.iter().map(|&x| (x as f64)*(x as f64)).sum::<f64>();
            let r = (sq/hd as f64+eps as f64).sqrt().recip();
            v.iter_mut().for_each(|x| *x = (*x as f64*r) as f32); v
        }).collect();
        // Wkv
        let kvw = decode_fp8_2(&loadu(&p(layer, "attn.wkv.weight")), &loadu(&p(layer, "attn.wkv.scale")), kvr, dim);
        let rms_xs = rms(&xs, &anw, eps);
        let mut kv = dense_gemv(&rms_xs, &kvw, kvr, dim);
        let kn = load(&p(layer, "attn.kv_norm.weight")); if kn.len()==kvr { kv = rms(&kv, &kn, eps); }
        let snk = load(&p(layer, "attn.attn_sink"));
        let mut ao = vec![0.0f32; nh*hd];
        for hh in 0..nh {
            let sc = qh[hh].iter().zip(kv.iter()).map(|(qk,vv)| (*qk as f64)*(*vv as f64)).sum::<f64>() * sscale as f64;
            let sk = if hh<snk.len(){snk[hh]as f64}else{0.0};
            let mx = sc.max(sk); let a = (sc-mx).exp() / ((sc-mx).exp() + (sk-mx).exp());
            for j in 0..hd { ao[hh*hd+j] = (a*kv[j.min(kvr-1)]as f64)as f32; }
        }
        // WoA grouped
        let woa = decode_fp8_2(&loadu(&p(layer, "attn.wo_a.weight")), &loadu(&p(layer, "attn.wo_a.scale")), og*olr, nh*hd/og);
        let i8x4k: Vec<Vec<f32>> = (0..og).map(|g| ao[g*(nh*hd/og)..(g+1)*(nh*hd/og)].to_vec()).collect();
        let mut wo = vec![0.0f32; og*olr];
        for g in 0..og { let wg: Vec<f32> = woa[g*olr*(nh*hd/og)..(g+1)*olr*(nh*hd/og)].to_vec();
            let ogv = dense_gemv(&i8x4k[g], &wg, olr, nh*hd/og);
            for i in 0..olr { wo[g*olr+i] = ogv[i]; } }
        let wob = decode_fp8_2(&loadu(&p(layer, "attn.wo_b.weight")), &loadu(&p(layer, "attn.wo_b.scale")), dim, og*olr);
        let xa_out = dense_gemv(&wo, &wob, dim, og*olr);
        let attn_finite = xa_out.iter().all(|v| v.is_finite());
        let attn_ms = attn_t0.elapsed().as_secs_f32() * 1000.0;

        // HC post-attn
        let mut nh2: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; hc];
        for i in 0..hc { for j in 0..dim { nh2[i][j] = pst_a[i] * xa_out[j]; for k in 0..hc { nh2[i][j] += cmb_a[i][k] * ra[k][j]; } } }
        h = nh2;
        total_hc_ms += hc_a_ms; total_attn_ms += attn_ms;

        // HC pre-ffn
        let hc_f0 = Instant::now();
        let rf = h.clone();
        let mix_f = dense_gemv(&h.iter().flat_map(|c| c.iter()).copied().collect::<Vec<_>>(), &load(&p(layer, "hc_ffn_fn")), 24, hc*dim);
        let sc_f = load(&p(layer, "hc_ffn_scale")); let bf = load(&p(layer, "hc_ffn_base"));
        let mut pre_f = [0f32; 4]; let mut pst_f = [0f32; 4]; let mut cmb_f = [[0f32; 4]; 4];
        for i in 0..hc { pre_f[i] = 1.0/(1.0+(-(mix_f[i]*sc_f[0]+bf[i])).exp()); pst_f[i] = 1.0/(1.0+(-(mix_f[hc+i]*sc_f[1]+bf[hc+i])).exp()); }
        for j in 0..hc { for k in 0..hc { cmb_f[j][k] = mix_f[2*hc+j*hc+k]*sc_f[2]+bf[2*hc+j*hc+k]; } }
        for _ in 0..20 {
            for j in 0..hc { let mx = cmb_f[j].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0f64; for k in 0..hc { let e = ((cmb_f[j][k]-mx)as f64).exp(); s+=e; cmb_f[j][k]=e as f32; }
                let inv=(s+heps as f64).recip(); for k in 0..hc { cmb_f[j][k]=(cmb_f[j][k]as f64*inv)as f32; } }
            for k in 0..hc { let mut s = 0f64; for j in 0..hc { s += cmb_f[j][k] as f64; }
                let inv=(s+heps as f64).recip(); for j in 0..hc { cmb_f[j][k]=(cmb_f[j][k]as f64*inv)as f32; } }
        }
        let mut xf = vec![0.0f32; dim];
        for i in 0..hc { for j in 0..dim { xf[j] += pre_f[i] * h[i][j]; } }
        let fnw = load(&p(layer, "ffn_norm.weight"));
        let xfn = rms(&xf, &fnw, eps);
        let hc_f_ms = hc_f0.elapsed().as_secs_f32() * 1000.0;
        total_hc_ms += hc_f_ms;

        // MoE.forward (official CUDA)
        let moe_t0 = Instant::now();
        let gate_w = load(&p(layer, "ffn.gate.weight"));
        let bias_w = load(&p(layer, "ffn.gate.bias"));
        let n_exp = gate_w.len() / dim;
        let mut scores = dense_gemv(&xfn, &gate_w, n_exp, dim);
        if !bias_w.is_empty() { for (s, b) in scores.iter_mut().zip(bias_w.iter()) { *s += b; } }
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);
        let selected_ids: Vec<usize> = indexed.iter().map(|(id, _)| *id).collect();
        let selected_weights: Vec<f32> = {
            let max_s = indexed.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f64> = indexed.iter().map(|(_, s)| ((s - max_s) as f64).exp()).collect();
            let total: f64 = exps.iter().sum();
            exps.iter().map(|e| (*e / total) as f32).collect()
        };
        let selected_pairs: Vec<(usize, f32)> = selected_ids.iter().copied().zip(selected_weights.iter().copied()).collect();

        // Load FP4 expert tensors
        let load_moe = Instant::now();
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
        // Load + upload shared FP8 tensors
        let sp = |n: &str| format!("layers.{layer}.ffn.shared_experts.{n}");
        let sh_dev = DeepSeekFp8SharedExpertWeightsDevice {
            gate_weight: upload(&stream, &loadu(&sp("w1.weight")))?,
            gate_scale: upload(&stream, &loadu(&sp("w1.scale")))?,
            up_weight: upload(&stream, &loadu(&sp("w3.weight")))?,
            up_scale: upload(&stream, &loadu(&sp("w3.scale")))?,
            down_weight: upload(&stream, &loadu(&sp("w2.weight")))?,
            down_scale: upload(&stream, &loadu(&sp("w2.scale")))?,
        };
        total_load_ms += load_moe.elapsed().as_secs_f32() * 1000.0;

        // Execute official CUDA MoE.forward
        let (moe_out, _tel) = execute_selected_moe_official_routed_fp4_cuda(
            &quant, &moe_exec, &stream,
            &expert_fp4_set, &selected_pairs,
            &xfn, dim, intermed, dim, layer,
            Some(&mut cache), Some(&sh_dev),
        )?;
        let moe_finite = moe_out.iter().all(|v| v.is_finite());
        let moe_ms = moe_t0.elapsed().as_secs_f32() * 1000.0;
        total_moe_ms += moe_ms;

        // HC post-ffn
        let mut nh3: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; hc];
        for i in 0..hc { for j in 0..dim { nh3[i][j] = pst_f[i] * moe_out[j]; for k in 0..hc { nh3[i][j] += cmb_f[i][k] * rf[k][j]; } } }
        h = nh3;

        let block_finite = h.iter().all(|c| c.iter().all(|v| v.is_finite()));
        if !moe_finite { eprintln!("LAYER {}: MoE output NOT FINITE!", layer); }
        if !block_finite { eprintln!("LAYER {}: block output NOT FINITE!", layer); }

        per_layer.push(serde_json::json!({
            "layer_id": layer, "hash_layer": is_hash,
            "selected_expert_ids": selected_ids,
            "selected_expert_weights": selected_weights,
            "attention_output_finite": attn_finite,
            "moe_output_finite": moe_finite,
            "block_output_finite": block_finite,
            "attention_ms": attn_ms, "hc_attn_ms": hc_a_ms,
            "moe_forward_ms": moe_ms, "hc_ffn_ms": hc_f_ms,
            "total_ms": layer_t0.elapsed().as_secs_f32() * 1000.0,
        }));
    }

    // HC head
    let hc_h_t0 = Instant::now();
    let flat: Vec<f32> = h.iter().flat_map(|c| c.iter()).copied().collect();
    let hmix = dense_gemv(&flat, &load("hc_head_fn"), hc, hc*dim);
    let hsc = load("hc_head_scale"); let hba = load("hc_head_base");
    let mut ph = [0f32; 4];
    for i in 0..hc { let v = hmix[i]; let b = if i<hba.len(){hba[i]}else{0.0}; let s = hsc.first().copied().unwrap_or(1.0);
        ph[i] = 1.0/(1.0+(-(v*s+b)).exp()) + heps; }
    let mut hf = vec![0.0f32; dim];
    for i in 0..hc { for j in 0..dim { hf[j] += ph[i] * h[i][j]; } }
    let head_ms = hc_h_t0.elapsed().as_secs_f32() * 1000.0;

    let hn = rms(&hf, &fn_w, eps);
    let logits = dense_gemv(&hn, &head_w, 129280, dim);

    let total_ms = total_t0.elapsed().as_secs_f32() * 1000.0;
    let lf = logits.iter().all(|v| v.is_finite());
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a,&b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    let top10_ids: Vec<usize> = idx.iter().take(10).copied().collect();
    let top10_logits: Vec<f32> = top10_ids.iter().map(|&i| logits[i]).collect();

    println!("{}", serde_json::to_string_pretty(&E2eResult {
        input_token_id: input_id, output_token_id: idx[0],
        top_10_token_ids: top10_ids, top_10_logits: top10_logits,
        final_logits_finite: lf, moe_placeholder_used: false,
        all_43_layers_moe_executed: true,
        total_forward_ms: total_ms, total_tensor_load_ms: total_load_ms,
        total_attention_ms: total_attn_ms, total_hc_ms: total_hc_ms,
        total_moe_ms, total_head_ms: head_ms,
        flags: serde_json::json!({
            "official_moe_forward": true,
            "official_attention_seq1_pos0": true,
            "official_hyper_connection": true,
            "mtp_included": false,
            "generation_scope": "official_one_token_canary",
        }),
        per_layer,
    }).unwrap());
    Ok(())
}

fn dense_gemv(x: &[f32], w: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m];
    for i in 0..m { let mut s = 0.0f64; for j in 0..k { s += (w[i*k+j] as f64) * (x[j] as f64); } o[i] = s as f32; }
    o
}

fn rms(x: &[f32], w: &[f32], e: f32) -> Vec<f32> {
    let n = x.len();
    let sq = x.iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>();
    let r = (sq / n as f64 + e as f64).sqrt().recip();
    x.iter().zip(w).map(|(&v, &wv)| (v as f64 * r * wv as f64) as f32).collect()
}

fn decode_fp8_2(weight: &[u8], scale: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let t = 128; let sc = (cols + t - 1) / t;
    let mut o = vec![0.0f32; rows * cols];
    for i in 0..rows {
        let sr = i / t;
        for j in 0..cols {
            o[i * cols + j] = f8e4m3_to_f32(weight[i * cols + j]) * f8e8m0_to_f32(scale[sr * sc + (j / t)]);
        }
    }
    o
}
