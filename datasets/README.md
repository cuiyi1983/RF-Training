# 数据集说明

本目录仅记录数据集路径和清单，**不包含实际数据文件**。

> ⚠️ **重要说明**：HAI 上的 `official_rfuav/`（133GB / 20 机型）**不是 RFUAV 官方全集**。
> RFUAV 官方全集约 **1.3 TB**，包含 **37 种无人机**原始 IQ 数据。
> HAI 上只有 kitofrank 当时整理的一个子集。

## 官方数据集来源

| 渠道 | 内容 | 地址 | 状态 |
|---|---|---|---|
| **GitHub** | 代码 + processed spectrogram | https://github.com/kitoweeknd/RFUAV | ✅ 可访问 |
| **HuggingFace** | 预处理 spectrogram 图片（1710×1710 JPG，35类）| https://huggingface.co/datasets/kitofrank/RFUAV | ✅ 可访问 |
| **原始 IQ 文件（1.3TB）** | USRP 原始 IQ 数据，37 种机型 | 需联系作者获取 | ⚠️ kito 已退出，链接可能已失效 |
| **Roboflow** | 部分数据 | https://universe.roboflow.com/rui-shi/drone-signal-detect-few-shot | ❌ 需确认 |

> **关键区分**：
> - GitHub/HuggingFace 上的数据是**预处理好的频谱图 JPG 图片**（不是原始 IQ）
> - **原始 .iq 文件**需单独获取，论文中提到"raw data will be free to use after paper accept"

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
