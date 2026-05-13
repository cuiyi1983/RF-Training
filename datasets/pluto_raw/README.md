# Pluto 实采数据

用于推理验证的 Pluto SDR 采集数据。**禁止用于模型训练。**

## scan_20260508 — 0508 测试数据集

**采集脚本**：`pluto_iq_collector_targeted.py`
**脚本路径（frps 服务器）**：`/tmp/pluto_iq_collector_targeted.py`
**脚本修改时间**：2026-05-05 13:42

**采集参数**（来自脚本配置）：
| 参数 | 值 |
|---|---|
| 采样率 | **60 MHz** |
| 增益 | **20 dB** |
| 中心频点 | 5760 / 5775 / 5800 / 5825 / 5850 MHz |
| Burst/轮 | 500 次 |
| 轮数 | 3 轮 |
| Buffer Size | 2048 |
| 总采样数/文件 | 3,072,000（= 3轮 × 500 bursts × 2048） |

> 注：文件名 `3.1Msps` 意为约 3.1M 总采样数，非采样率

**内容**：
| 目录 | 标签 | 样本数 |
|---|---|---|
| `noise/` | 无无人机 | 5 |
| `drone/` | 有无人机 | 5 |

**NPZ 文件格式**：
```python
{
    "iq": complex64,           # IQ 数据
    "center_freq": float,      # 中心频率 Hz
    "sampling_rate": float,    # 采样率 Hz（60e6）
    "gain": float,             # 增益 dB（20）
    "timestamp": float,         # 时间戳
    "freq_name": str,          # 频点名 如 "SCAN_5760"
    "num_samples": int,        # 采样点数（3072000）
}
```

## 数据目录结构

```
pluto_raw/
└── scan_20260508/
    ├── noise/         ← 5个噪声样本
    │   ├── SCAN_5760_3.1Msps.npz
    │   ├── SCAN_5775_3.1Msps.npz
    │   ├── SCAN_5800_3.1Msps.npz
    │   ├── SCAN_5825_3.1Msps.npz
    │   └── SCAN_5850_3.1Msps.npz
    ├── drone/         ← 5个无人机样本
    │   ├── SCAN_5760_3.1Msps.npz
    │   ├── SCAN_5775_3.1Msps.npz
    │   ├── SCAN_5800_3.1Msps.npz
    │   ├── SCAN_5825_3.1Msps.npz
    │   └── SCAN_5850_3.1Msps.npz
    └── pluto_iq_collector_targeted.py  ← 采集脚本（2026-05-05）
```
