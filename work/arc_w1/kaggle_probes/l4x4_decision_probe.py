"""Decide whether ARC-AGI-2 can be run on L4x4 instead of a single T4.

Why this matters more than any model tweak: the 2026-04-07 host announcement says ARC-AGI-2
"now has access to Kaggle's pool of powerful new L4x4 machines! These machines offer 96GB of GPU
memory enabling submissions with much larger models." We have been fighting a 14.56 GiB T4 the
whole time, and our binding constraint -- LoRA TTT failing to allocate a 1.77 GiB activation
buffer, so that 86 of 102 tasks never adapted -- is purely a memory limit.

If L4x4 is available, the trade is: 4x the quota burn (the forum reports 12 h of L4x4 consuming
48 GPU-hours against a 30 h weekly allowance), in exchange for ~24 GiB on a single device instead
of 14.56, and 4 devices we can shard across instead of one.

SECURITY: never print environment *values* here. A previous probe leaked
KAGGLE_DATA_PROXY_TOKEN / KAGGLE_USER_SECRETS_TOKEN by dumping os.environ, which cost four
kernels. Only key names and value lengths are printed below.
"""

import os
import subprocess
import sys

print("=" * 78)
print("nvidia-smi")
print("=" * 78)
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
         "--format=csv"],
        capture_output=True, text=True, timeout=120)
    print(out.stdout.strip() or out.stderr.strip()[:500])
except Exception as exc:
    print("nvidia-smi failed:", type(exc).__name__, exc)

print()
try:
    import torch
    print("torch              :", torch.__version__)
    print("cuda available     :", torch.cuda.is_available())
    print("device count       :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free_b, total_b = torch.cuda.mem_get_info(i)
        print(f"  device {i}: {p.name}  cap={p.major}.{p.minor}  "
              f"{total_b / 2**30:.2f} GiB total, {free_b / 2**30:.2f} GiB free")
    print("bf16 supported     :", torch.cuda.is_bf16_supported())
except Exception as exc:
    print("torch probe failed:", type(exc).__name__, exc)

print()
print("=" * 78)
print("competition data")
print("=" * 78)
for d in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
          "/kaggle/input/arc-prize-2026-arc-agi-2"):
    if os.path.isdir(d):
        files = sorted(os.listdir(d))
        print(f"FOUND {d}")
        for f in files:
            print(f"   {f}")
        break
else:
    print("competition data NOT found under either path")
    for base in ("/kaggle/input",):
        if os.path.isdir(base):
            print(f"  {base} contains:", sorted(os.listdir(base))[:10])

print()
print("=" * 78)
print("model mount")
print("=" * 78)
for base in ("/kaggle/input",):
    if not os.path.isdir(base):
        continue
    for entry in sorted(os.listdir(base)):
        p = os.path.join(base, entry)
        if os.path.isdir(p) and "qwen" in entry.lower():
            print("FOUND", p)
            for root, _dirs, fs in os.walk(p):
                for f in fs[:8]:
                    print("   ", os.path.join(root, f).replace(p, "."))
                break

print()
print("=" * 78)
print("environment (KEY NAMES AND LENGTHS ONLY -- never values)")
print("=" * 78)
for k in sorted(os.environ):
    print(f"  {k} len={len(os.environ[k])}")

print()
print("=" * 78)
print("verdict inputs")
print("=" * 78)
try:
    import torch
    n = torch.cuda.device_count()
    if n >= 4:
        print(f"L4x4 PRESENT: {n} devices. Shard with CUDA_VISIBLE_DEVICES + "
              f"--num-shards {n} and merge with tools/merge_shards.py.")
    elif n == 1:
        p = torch.cuda.get_device_properties(0)
        print(f"SINGLE device: {p.name}. Total VRAM {p.total_memory / 2**30:.2f} GiB.")
    else:
        print(f"unexpected device count: {n}")
except Exception as exc:
    print("verdict failed:", type(exc).__name__, exc)
print("python", sys.version.split()[0])
