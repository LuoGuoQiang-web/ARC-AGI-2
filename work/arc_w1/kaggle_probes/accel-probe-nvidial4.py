import os, sys, json
print("=== accel probe ===")
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available())
    n = torch.cuda.device_count()
    print("device_count", n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU{i}: name={p.name!r} sm_{p.major}{p.minor} vram={p.total_memory/2**30:.1f}GiB")
    if n:
        # tf32/fp32 matmul smoke test
        a = torch.randn(1024, 1024, device="cuda")
        r = (a @ a).sum().item()
        print("fp32 matmul ok:", round(r, 2))
        # bf16 support test (L4 native, T4 falls back)
        try:
            b = a.to(torch.bfloat16)
            rb = (b @ b).float().sum().item()
            print("bf16 matmul ok:", round(rb, 2))
        except Exception as exc:
            print("bf16 matmul FAILED:", type(exc).__name__, exc)
except Exception as exc:
    print("torch import FAILED:", type(exc).__name__, exc)

env = {k: v for k, v in os.environ.items()
       if any(t in k.upper() for t in ("ACCEL", "MACHINE", "GPU", "KAGGLE", "CUDA_VISIBLE"))}
print("env:", json.dumps(env, indent=1)[:1500])
