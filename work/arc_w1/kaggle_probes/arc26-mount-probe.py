"""CPU probe: what is actually mounted under /kaggle/input for this kernel?

We pushed `competition_sources: ["arc-prize-2026-arc-agi-2"]` but the solver hit
`--data-dir does not exist: /kaggle/input/arc-prize-2026-arc-agi-2`. This walks the
mount tree so we can see the real path instead of guessing.

CPU-only on purpose: consumes no GPU quota.
"""
import json
import os

print("=== /kaggle/input listing ===")
for root in ("/kaggle/input", "/kaggle/input/models"):
    print(f"\n--- {root} exists={os.path.isdir(root)}")
    if not os.path.isdir(root):
        continue
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        kind = "DIR " if os.path.isdir(full) else "FILE"
        size = "" if os.path.isdir(full) else f"{os.path.getsize(full) / 1e6:9.2f} MB"
        print(f"  {kind} {entry:<52} {size}")
        if os.path.isdir(full):
            try:
                for sub in sorted(os.listdir(full))[:12]:
                    subfull = os.path.join(full, sub)
                    skind = "DIR " if os.path.isdir(subfull) else "FILE"
                    ssize = "" if os.path.isdir(subfull) else \
                        f"{os.path.getsize(subfull) / 1e6:9.2f} MB"
                    print(f"        {skind} {sub:<44} {ssize}")
            except Exception as exc:
                print(f"        (unreadable: {exc})")

print("\n=== /kaggle/working ===")
print(os.path.isdir("/kaggle/working"))

print("\n=== environment hints ===")
for k in sorted(os.environ):
    if any(t in k.upper() for t in ("KAGGLE", "COMPETITION", "DATASET", "INPUT")):
        print(f"  {k} = {os.environ[k][:160]}")

# If a hidden-test-style file exists anywhere, report its shape.
print("\n=== searching for arc json files (depth <= 4) ===")
found = []
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root[len("/kaggle/input"):].count(os.sep)
    if depth >= 4:
        dirs[:] = []
        continue
    for fn in files:
        if fn.endswith(".json") and "arc" in fn.lower():
            fp = os.path.join(root, fn)
            found.append((fp, os.path.getsize(fp)))
for fp, sz in sorted(found)[:40]:
    print(f"  {fp}  {sz / 1e6:.2f} MB")
print("total arc json files found:", len(found))

print("\n=== competition mount check ===")
for cand in ("/kaggle/input/arc-prize-2026-arc-agi-2",
             "/kaggle/input/arc-prize-2026-agi-2"):
    print(f"  {cand} exists={os.path.isdir(cand)}")
print("probe complete")
