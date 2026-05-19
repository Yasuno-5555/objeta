import sys
import os
import json
import time
import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union

# Set workspace paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import logic from qwen36_full_rust
import experiments.qwen36_full_rust as qwen_runner
from experiments.qwen36_full_rust import lib, HDIM, rust_step_with_entropy, sample, get_moe_stats, get_page_cache_stats, diff_stats, phase_summary_from_snapshots, analyze_output

# Load tokenizer locally in the daemon
from transformers import AutoTokenizer
snap = sorted(os.listdir(
    "/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
tok = AutoTokenizer.from_pretrained(
    f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}")

app = FastAPI(title="OpenAI-compatible Qwen3.6 Rust Executor Server")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 15
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    strategy: Optional[str] = "safe"

# Keep track of active strategy parameters
current_strategy = None

def apply_strategy(strategy_name_or_path: str):
    global current_strategy
    if current_strategy == strategy_name_or_path:
        return {} # Already applied
        
    strat_name = strategy_name_or_path.lower()
    presets = ["safe", "fast", "turbo", "debug"]
    if strat_name in presets:
        strat_path = f"configs/{strat_name}.json"
    else:
        strat_path = strategy_name_or_path
        
    if not os.path.exists(strat_path):
        raise HTTPException(status_code=400, detail=f"Strategy config file '{strat_path}' not found.")
        
    print(f"Applying strategy config from {strat_path}")
    with open(strat_path, "r") as f:
        strategy_dict = json.load(f)
        
    # Apply to Rust executor
    lib.lko_moe_init_page_cache(strategy_dict["expert_cache_mb"] * 1024 * 1024)
    lib.lko_moe_reset_page_cache_stats()
    lib.lko_runner_set_fusion_ratio(strategy_dict["fusion_ratio"])
    lib.lko_runner_set_moe_on_deltanet(strategy_dict["moe_on_deltanet"])
    lib.lko_runner_set_moe_top_p(strategy_dict["moe_top_p"])
    lib.lko_runner_set_moe_prune_mode(0 if strategy_dict["moe_prune_mode"] == "top_p" else 1)
    lib.lko_runner_set_moe_contrib_threshold(strategy_dict["moe_contrib_threshold"])
    
    current_strategy = strategy_name_or_path
    return strategy_dict

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.temperature != 0.0:
        raise HTTPException(status_code=400, detail="Only temperature=0.0 is currently supported.")
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not currently supported.")
        
    # Apply strategy
    strategy_dict = apply_strategy(request.strategy)
    
    # Process messages with chat template
    msgs = [{"role": m.role, "content": m.content} for m in request.messages]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat)
    
    # Run generation
    lib.lko_runner_reset_moe_stats()
    
    gen, prefill_entropies, decode_entropies, phase_stats, metrics, total_decode_wall_s = qwen_runner.generate(
        ids, max_tokens=request.max_tokens, temperature=request.temperature, top_k=0, tok_decoder=tok
    )
    text = tok.decode(gen, skip_special_tokens=True)
    
    # Write run artifacts in the runs/ folder
    run_dir = f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, "strategy.json"), "w") as f:
        json.dump(strategy_dict, f, indent=2)
    with open(os.path.join(run_dir, "output.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(run_dir, "metrics.jsonl"), "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
            
    first_garbage, first_repetition = analyze_output(gen, text)
    pc_stats = get_page_cache_stats()
    moe_stats = phase_stats["decode_total"]
    
    summary_dict = {
        "tok_per_sec": len(gen) / total_decode_wall_s if total_decode_wall_s > 0 else 0.0,
        "avg_entropy": sum(decode_entropies) / len(decode_entropies) if decode_entropies else 0.0,
        "first_repetition": first_repetition,
        "first_garbage": first_garbage,
        "cache_stats": {
            "capacity_bytes": pc_stats.get("cache_capacity_bytes", 0) if pc_stats else 0,
            "resident_bytes": pc_stats.get("cache_resident_bytes", 0) if pc_stats else 0,
            "warm_hit_count": pc_stats.get("warm_hit_count", 0) if pc_stats else 0,
            "cold_load_count": pc_stats.get("cold_load_count", 0) if pc_stats else 0,
            "eviction_count": pc_stats.get("eviction_count", 0) if pc_stats else 0,
            "bytes_loaded_actual": pc_stats.get("bytes_loaded_actual", 0) if pc_stats else 0,
            "warm_hit_rate": pc_stats.get("warm_hit_rate", 0.0) if pc_stats else 0.0,
        },
        "wall_timing": {
            "total_decode_wall_s": total_decode_wall_s,
        }
    }
    
    if moe_stats:
        s = moe_stats["summary"]
        fs = moe_stats.get("forward_summary", {})
        summary_dict["wall_timing"].update({
            "avg_forward_wall_ms": fs.get("avg_forward_wall_ms", 0.0),
            "avg_moe_wall_ms_per_token": fs.get("avg_moe_wall_ms_per_token", 0.0),
            "avg_moe_router_wall_ms": s.get("avg_router_wall_ms", 0.0),
            "avg_moe_select_wall_ms": s.get("avg_select_wall_ms", 0.0),
            "avg_moe_load_wall_ms": s.get("avg_load_wall_ms", 0.0),
            "avg_moe_exec_wall_ms": s.get("avg_exec_wall_ms", 0.0),
            "avg_moe_accumulate_wall_ms": s.get("avg_accumulate_wall_ms", 0.0)
        })
        
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)
        
    # OpenAI response format
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(ids),
            "completion_tokens": len(gen),
            "total_tokens": len(ids) + len(gen)
        },
        "tok_per_sec": summary_dict["tok_per_sec"],
        "summary": summary_dict
    }

if __name__ == "__main__":
    import uvicorn
    # Pre-apply default strategy safe
    apply_strategy("safe")
    uvicorn.run(app, host="0.0.0.0", port=8080)
