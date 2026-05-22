"""Download DeepSeek V4 Flash model from HuggingFace."""
import os
import sys
import time
from huggingface_hub import snapshot_download, HfApi

model_id = "deepseek-ai/DeepSeek-V4-Flash"
local_dir = r"E:\Projects\DeepSeek-V4-Flash"

os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {model_id} to {local_dir}...")
print(f"This model is ~160GB, will take significant time.")
sys.stdout.flush()

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.md", "*.txt", "*.pdf"],
    )
    print("Download complete!")
except KeyboardInterrupt:
    print("\nInterrupted. Partial download can be resumed.")
    sys.exit(1)
except Exception as e:
    print(f"Error during download: {e}", file=sys.stderr)
    sys.exit(1)

# List files
print(f"\nFiles in {local_dir}:")
for f in sorted(os.listdir(local_dir)):
    fpath = os.path.join(local_dir, f)
    if os.path.isfile(fpath):
        sz = os.path.getsize(fpath)
        if sz > 1e9:
            print(f"  {f}: {sz/1e9:.2f} GB")
        elif sz > 1e6:
            print(f"  {f}: {sz/1e6:.1f} MB")
        else:
            print(f"  {f}: {sz} bytes")
