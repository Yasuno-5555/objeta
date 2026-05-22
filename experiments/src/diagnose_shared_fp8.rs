use objeta_parser::deepseek::*;
use objeta_parser::ModelWeights;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new(r"E:\Projects\DeepSeek-V4-Flash");
    let model = ModelWeights::open(model_dir)?;
    
    let layer = 27;
    let hidden_size = 4096;
    let intermediate_size = 2048;
    
    let prefix = format!("layers.{}.ffn.shared_experts", layer);
    let w1 = model.get_raw(&format!("{}.w1.weight", prefix))?.to_vec();
    let w1s = model.get_raw(&format!("{}.w1.scale", prefix))?.to_vec();
    
    // Random hidden state (same as benchmark)
    let mut state: u64 = 42;
    let mut hidden = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((state >> 32) as u32) | 1;
        let unit = (bits as f32) / (u32::MAX as f32);
        hidden[i] = unit * 2.0 - 1.0;
    }
    
    let (act_fp8, act_s) = cpu_act_quant(&hidden, 128);
    
    // Check act_fp8 for NaN/inf patterns
    let mut fp8_exp15 = 0u32;
    for (i, &b) in act_fp8.iter().enumerate() {
        let exp = (b >> 3) & 0x0F;
        if exp == 15 { fp8_exp15 += 1; if fp8_exp15 <= 5 { println!("act_fp8[{i}] raw=0x{b:02X} exp=15"); } }
    }
    println!("act_fp8 exp=15 count: {fp8_exp15} / {len}", len=act_fp8.len());
    
    // Check w1 for NaN/inf patterns
    let mut w1_exp15 = 0u32;
    for (i, &b) in w1.iter().enumerate() {
        let exp = (b >> 3) & 0x0F;
        if exp == 15 { w1_exp15 += 1; if w1_exp15 <= 5 { println!("w1[{i}] raw=0x{b:02X} exp=15"); } }
    }
    println!("w1 exp=15 count: {w1_exp15} / {len}", len=w1.len());
    
    // Now trace the first NaN-producing gate element
    // Gate row 1 produced NaN at index 1
    let row = 1;
    let mut sum: f64 = 0.0;
    for col in 0..hidden_size {
        let act_v = f8e4m3_to_f32(act_fp8[col]);
        let act_s = f8e8m0_to_f32(act_s[col / 128]);
        let wt_v = f8e4m3_to_f32(w1[row * hidden_size + col]);
        let wt_scale_col = col / 128;
        let wt_s = f8e8m0_to_f32(w1s[(row / 128) * 32 + wt_scale_col]);
        
        let term = (act_v as f64) * (act_s as f64) * (wt_v as f64) * (wt_s as f64);
        if term.is_nan() || term.is_infinite() || sum.is_nan() || sum.is_infinite() {
            println!("FIRST NON-FINITE at col={col} row={row}: act_v={act_v} act_s={act_s} wt_v={wt_v} wt_s={wt_s} term={term} sum_before={sum}");
            break;
        }
        sum += term;
    }
    println!("Row {row} total sum = {sum}");
    
    Ok(())
}
