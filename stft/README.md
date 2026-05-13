# STFT 预处理模块

## 参数配置

| 参数 | 值 | 说明 |
|---|---|---|
| 采样率 | 60 MHz | Pluto 上限 |
| FFT 点数 | 1024 | 平衡频率/时间分辨率 |
| hop | 512 | 50%重叠 |
| window | Hamming | 与RFUAV官方一致 |
| return_onesided | False | 完整频谱 |
| 幅度变换 | log1p | 与训练数据一致 |
| 归一化 | minmax01 | 归一化到[0,1] |
| 目标shape | (1024, 1170) | 频谱图尺寸 |

## 主要脚本

- `iq_to_spectrogram.py` — IQ 原始数据 → STFT 频谱图
- `preprocessing_utils.py` — 降采样、归一化等工具
