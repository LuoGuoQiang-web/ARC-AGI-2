"""Stage 0: environment + SFT model introspection on the real Kaggle GPU image.

Goal is to learn, from the artifact itself, the exact I/O format the
`sorokin/qwen3_4b_grids15_sft139` (Apache-2.0) checkpoint expects, and to measure
model load time and forward-pass throughput on the actual accelerator.

Nothing here assumes a prompt format; it dumps the ground truth (config,
tokenizer special tokens, chat template, added vocab) and then times the stack.
"""
import json
import os
import sys
import time

print("=== stage0: env + model introspection ===", flush=True)
print("python", sys.version.split()[0])

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f}GiB")

for pkg in ("transformers", "peft", "accelerate", "datasets", "tokenizers"):
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"{pkg}: MISSING ({type(exc).__name__})")

MODEL_DIR = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
print()
print("=== model directory ===")
print("exists:", os.path.isdir(MODEL_DIR))
if os.path.isdir(MODEL_DIR):
    for root, _dirs, files in os.walk(MODEL_DIR):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            print(f"  {os.path.relpath(fp, MODEL_DIR):<48} {os.path.getsize(fp) / 1e6:8.2f} MB")

for name in ("config.json", "generation_config.json", "tokenizer_config.json",
             "special_tokens_map.json"):
    fp = os.path.join(MODEL_DIR, name)
    if os.path.exists(fp):
        print()
        print(f"--- {name} ---")
        raw = open(fp, encoding="utf-8").read()
        try:
            obj = json.loads(raw)
            print(json.dumps(obj, indent=1)[:5000])
        except Exception:
            print(raw[:5000])

print()
print("=== tokenizer ===")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
print("vocab_size(len(tok)):", len(tok))
print("model_max_length:", tok.model_max_length)
added = tok.get_added_vocab()
print("added_vocab size:", len(added))
for key, idx in list(added.items())[:120]:
    print("   ", repr(key), idx)
ct = getattr(tok, "chat_template", None)
print("chat_template present:", bool(ct))
if ct:
    print(ct[:2500])

print()
print("=== model load + throughput ===")
from transformers import AutoModelForCausalLM

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
)
print("load_seconds:", round(time.time() - t0, 1))
model.eval()
print("params_B:", round(sum(p.numel() for p in model.parameters()) / 1e9, 3))

# forward-pass timing at a realistic sequence length
for seq in (512, 1024, 2048):
    ids = torch.randint(0, 1000, (1, seq), device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        model(input_ids=ids)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  forward seq={seq:<5} {dt * 1000:8.1f} ms   ({seq / dt:8.1f} tok/s)")

print()
print("=== torch.cuda.mem summary ===")
free, total = torch.cuda.mem_get_info()
print(f"free {free / 2**30:.1f} GiB / total {total / 2**30:.1f} GiB")
print("stage0 complete", flush=True)
