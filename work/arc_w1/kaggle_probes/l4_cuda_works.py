"""Does a real CUDA op actually run on the L4 with this image's PyTorch?

The solver refused to load its model on an L4 with:

    GPU 0 = NVIDIA L4 is sm_89, but this PyTorch build only supports
    sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120

That message comes from our own `assert_gpu_compatible()`, which tests set membership in
`torch.cuda.get_arch_list()`. PyTorch's *own* check is a range check -- it warns only when the
capability falls outside [minimum, maximum], and it reported (7.0)-(12.0) on this image. sm_89
sits inside that range.

The two checks disagree, so one of them is wrong, and the answer decides whether the whole L4
path is usable -- which is the difference between 14.56 GiB on one device and 88 GiB on four.
A public notebook at LB 33.89 reports machine_shape=NvidiaL4, which suggests our guard is the
wrong one, but that is inference, not evidence. This probe runs actual CUDA work: an allocation,
a matmul, a backward pass, and a bf16 matmul, escalating until something fails for real.
"""

import torch

print("torch            :", torch.__version__)
print("cuda available   :", torch.cuda.is_available())
print("arch list        :", torch.cuda.get_arch_list())
print("device count     :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f"  device {i}: {torch.cuda.get_device_name(i)}  capability {cap[0]}.{cap[1]}")

dev = torch.device("cuda:0")
print()
print("=" * 78)
print("escalating real work on cuda:0")
print("=" * 78)

# 1. allocation + fill
try:
    a = torch.ones(1024, 1024, device=dev)
    print("1 alloc+fill     : OK  sum =", float(a.sum()))
except Exception as exc:
    print("1 alloc+fill     : FAIL", type(exc).__name__, str(exc)[:300])
    raise SystemExit(1)

# 2. fp32 matmul
try:
    c = a @ a
    torch.cuda.synchronize()
    print("2 fp32 matmul    : OK  value =", float(c[0, 0]))
except Exception as exc:
    print("2 fp32 matmul    : FAIL", type(exc).__name__, str(exc)[:300])
    raise SystemExit(1)

# 3. backward (this is what TTT actually needs)
try:
    x = torch.randn(512, 512, device=dev, requires_grad=True)
    loss = (x @ x).pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    print("3 autograd       : OK  grad norm =", float(x.grad.norm()))
except Exception as exc:
    print("3 autograd       : FAIL", type(exc).__name__, str(exc)[:300])
    raise SystemExit(1)

# 4. bf16 matmul (the precision the solver trains in)
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = (a @ a).to(torch.bfloat16)
        e = d @ d
    torch.cuda.synchronize()
    print("4 bf16 autocast  : OK  dtype =", e.dtype)
except Exception as exc:
    print("4 bf16 autocast  : FAIL", type(exc).__name__, str(exc)[:300])

# 5. a long-sequence forward, which is the shape TTT actually hits
try:
    seq = torch.randint(0, 16, (1, 8192), device=dev)
    emb = torch.nn.Embedding(16, 2560).to(dev)
    h = emb(seq)
    torch.cuda.synchronize()
    print("5 8192-token fwd : OK  shape =", tuple(h.shape))
except Exception as exc:
    print("5 8192-token fwd : FAIL", type(exc).__name__, str(exc)[:300])

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
print("If checks 1-3 passed, the L4 executes CUDA fine and assert_gpu_compatible()'s")
print("set-membership test is a FALSE POSITIVE that must be relaxed to a range check.")
