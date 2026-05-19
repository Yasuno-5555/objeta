#!/usr/bin/env python3
"""Shared Golden Registry Management for Qwen3.6 Correctness Oracles."""
import os
import json
import hashlib
import subprocess
from pathlib import Path

def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

def calculate_file_sha256(filepath):
    if not os.path.exists(filepath):
        return "missing"
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "error"

def get_model_integrity_hashes(tok_dir, bin_dir):
    tok_path = Path(tok_dir)
    bin_path = Path(bin_dir)
    return {
        "tokenizer_hash": calculate_file_sha256(tok_path / "tokenizer.json"),
        "config_hash": calculate_file_sha256(tok_path / "config.json"),
        "lm_head_hash": calculate_file_sha256(bin_path / "lm_head.bin"),
        "embed_hash": calculate_file_sha256(bin_path / "embed_tokens.bin"),
        "weight_manifest_hash": calculate_file_sha256(bin_path / "manifest.json")
    }

def get_registry_path():
    p = Path("runs/oracles/registry.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_registry():
    path = get_registry_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_registry(registry):
    path = get_registry_path()
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)

def register_golden(
    golden_name,
    model_id,
    strategy_name,
    prompt,
    prompt_hash,
    tokenizer_hash,
    weight_manifest_hash,
    file_path,
    git_commit,
    prompt_mode=None,
    model_input_text=None,
    tokenizer_ids=None,
):
    registry = load_registry()
    registry[golden_name] = {
        "model_id": model_id,
        "strategy": strategy_name,
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "git_commit": git_commit,
        "tokenizer_hash": tokenizer_hash,
        "weight_manifest_hash": weight_manifest_hash,
        "file_path": str(file_path)
    }
    if prompt_mode is not None:
        registry[golden_name]["prompt_mode"] = prompt_mode
    if model_input_text is not None:
        registry[golden_name]["model_input_text"] = model_input_text
    if tokenizer_ids is not None:
        registry[golden_name]["tokenizer_ids"] = tokenizer_ids
    save_registry(registry)
    print(f"Registered '{golden_name}' in golden registry.")

def lookup_golden(golden_name):
    registry = load_registry()
    return registry.get(golden_name)
