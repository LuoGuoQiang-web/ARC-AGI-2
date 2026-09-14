# kaggle_probes —— 历史探测脚本留档

这些是另一台电脑在 2026-09-13 推送到 Kaggle 的**一次性探测脚本**的源码，为了留档而保存在这里。
它们对应的 Kaggle kernel 中，有 4 个**已被删除**（因为运行日志里打印了环境变量 token，见上级
`README.md` §7）。

| 文件 | 目的 | Kaggle kernel 状态 |
|---|---|---|
| `arc26-mount-probe.py` | 查 `/kaggle/input` 与 `/kaggle/working` 的挂载情况 | 🗑 已删除（token 泄露 4 处） |
| `accel-probe-nvidial4x4.py` | 探测 NvidiaL4x4 加速器 | 🗑 已删除（2 处） |
| `accel-probe-nvidial4.py` | 探测 NvidiaL4 加速器 | 🗑 已删除（2 处） |
| `accel-probe-nvidiateslat4.py` | 探测 NvidiaTeslaT4 加速器 | 🗑 已删除（2 处） |
| `accelerator-probe-l4x4-check.py` | L4x4 可用性复核 | ✅ 保留 |
| `nvarc-env-probe-shape-err.py` | NVARC 环境 + 形状错误复现 | ✅ 保留 |
| `nvarc-env-probe-competition-model-l4x4.py` | NVARC 环境 + 竞赛数据/模型挂载（**实测到 240 题**） | ✅ 保留 |
| `arc26-stage0-model-introspect.py` | 环境 + SFT 模型自省（T4×2 / torch 2.10+cu128） | ✅ 保留 |
| `arc26-stage1-alphabet-and-cost.py` | 16-token 字母表 + 序列化 + 吞吐/反向成本 | ✅ 保留 |

## ⚠️ 运行这些脚本前必读

`arc26-mount-probe.py` 等脚本里有 `print(os.environ)` 或等价的全量环境输出 —— **这正是 token
泄露的原因**。若要重新运行任何探测：

```python
# ❌ 不要这样做
print(os.environ)

# ✅ 这样做：只打印键名与值长度
for k in sorted(os.environ):
    print(k, "len=", len(os.environ[k]))
```

推送到 Kaggle 前确认 `is_private=True`（`tools/kpush.py` 默认即为私有）。
