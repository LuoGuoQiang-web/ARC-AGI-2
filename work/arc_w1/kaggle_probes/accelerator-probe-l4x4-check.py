import subprocess, sys, json
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'],capture_output=True,text=True).stdout)
print('python', sys.version.split()[0])
try:
    import torch; print('torch', torch.__version__, 'bf16', torch.cuda.is_bf16_supported())
except Exception as e:
    print('torch missing:', e)
try:
    import unsloth; print('unsloth OK')
except Exception as e:
    print('unsloth missing:', type(e).__name__)
# 写一个最小的合法提交，确保输出通路可用
import os
d = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
print('competition dir exists:', os.path.isdir(d), os.listdir(d)[:6] if os.path.isdir(d) else None)
