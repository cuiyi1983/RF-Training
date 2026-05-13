# 数据集说明

本目录仅记录数据集路径和清单，**不包含实际数据文件**。

## 数据集清单

### official_rfuav/ — RFUAV 官方原始 IQ 数据

**存放位置（HAI）**：`/mnt/data/rfuav/official_rfuav/`

包含 DJI 和 non-DJI 共 20 种机型的原始 IQ 数据（USRP X310 采集）。

| 机型分类 | 目录 | 数量 |
|---|---|---|
| DJI 机型 | `DJI/` | 6 |
| Non-DJI 机型 | `non_DJI/` | 14 |

**采集参数**：
- 设备：USRP X310
- 采样率：100 MHz
- 中心频率：5760 MHz
- 带宽：100 MHz
- 数据格式：Complex Float（.iq 二进制）

详见各 `manifest.json`

### dataset_v5_final/ — 预处理 STFT 频谱图

**存放位置（HAI）**：`/mnt/data/rfuav/dataset_v5_final/`

从 `official_rfuav/` 降采样至 60MHz 后生成，适配 Pluto 设备。

- 频谱图 shape：(1024, 1170) float32
- 训练集 drone：4574 样本
- 训练集 noise：1603 样本
- 验证集 drone：1900 样本
- 验证集 noise：600 样本

### pluto_raw/ — Pluto 实采数据

用于推理验证，采集参数：
- 设备：ADALM PLUTO
- 采样率：60 MHz
- 中心频率：5805 MHz
- 带宽：56 MHz
- 增益：20 dB

## 数据使用规范

- **禁止**将 Pluto 实采数据用于模型训练
- 仅使用 `official_rfuav/` 进行训练
- 推理验证可使用 `pluto_raw/`
