# Pluto 实采数据

用于推理验证的 Pluto SDR 采集数据。**禁止用于模型训练。**

## scan_20260508 — 0508 测试数据集

**来源**：2026-05-08 Pluto 实采数据，采集参数：
- 采样率：60 MHz
- 中心频率：5805 MHz
- 带宽：56 MHz
- 增益：20 dB

**内容**：
| 目录 | 标签 | 样本数 |
|---|---|---|
| `noise/` | 无无人机 | 5 |
| `drone/` | 有无人机 | 5 |

**文件命名**：`SCAN_<freq>_3.1Msps.npz`

**使用说明**：对应 `validate_rar_correct.py` 推理脚本，用于 STFT 二分类模型验证。

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
    └── drone/         ← 5个无人机样本
        ├── SCAN_5760_3.1Msps.npz
        ├── SCAN_5775_3.1Msps.npz
        ├── SCAN_5800_3.1Msps.npz
        ├── SCAN_5825_3.1Msps.npz
        └── SCAN_5850_3.1Msps.npz
```
