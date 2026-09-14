import os, subprocess, sys, glob, json
def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout
    except Exception as e: return f'ERR {e}'
print('=== GPU ===')
print(sh('nvidia-smi --query-gpu=name,memory.total --format=csv'))
print('GPU count:', sh('nvidia-smi --list-gpus | wc -l').strip())
print('=== python/torch ===')
print(sys.version.split()[0])
try:
    import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count(), '| bf16', torch.cuda.is_bf16_supported())
except Exception as e: print('torch ERR', e)
for m in ['transformers','unsloth','peft','trl','bitsandbytes','datasets']:
    try:
        mod=__import__(m); print(f'{m}: {getattr(mod,"__version__","?")}')
    except Exception as e: print(f'{m}: MISSING')
print('=== /kaggle/input 树（3 层）===')
print(sh('find /kaggle/input -maxdepth 3 | head -40'))
print('=== 模型目录 ===')
base='/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139'
print('exists:', os.path.isdir(base))
print(sh(f'find {base} -maxdepth 3 | head -30') if os.path.isdir(base) else 'NOT FOUND')
print('=== 竞赛数据 ===')
c='/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
print('exists:', os.path.isdir(c))
print(sh(f'ls -la {c}') if os.path.isdir(c) else 'NOT FOUND')
if os.path.isdir(c):
    p=os.path.join(c,'arc-agi_test_challenges.json')
    if os.path.exists(p):
        d=json.load(open(p)); print('test_challenges 题数:', len(d))
        import hashlib; print('md5:', hashlib.md5(open(p,'rb').read()).hexdigest())
