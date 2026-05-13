# STFT 预处理

## Pluto 预处理（新增）

`pluto_preprocess.py` — Pluto IQ 数据转 RFUAV 格式频谱图

**核心逻辑**：将 Pluto 采集的 60 MHz IQ 数据，通过与 RFUAV 官方一致的 STFT 参数转换为频谱图，供下游训练流程使用。

### 参数对照表

| 参数 | RFUAV 官方（USRP）| Pluto 适配 |
|---|---|---|
| 采样率 fs | 100 MHz | **60 MHz** |
| nperseg | 1024 | 1024 |
| window | hamming | hamming |
| return_onesided | false | false |
| fftshift | true | true |
| 幅度类型 | 10·log10(\|Z\|) | 10·log10(\|Z\|) |

### 使用方式

```python
from pluto_preprocess import PlutoPreprocessor

proc = PlutoPreprocessor(config='../configs/stage1/pluto_stft.yaml')
spec = proc.process_iq(iq_data)  # -> (640, 640) spectrogram
```

## RFUAV 官方参考

`rfuav_reference.py` — RFUAV 官方 STFT 实现参考（来自 `graphic/RawDataProcessor.py`）
