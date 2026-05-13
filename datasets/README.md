# 数据集说明

本目录仅记录数据集路径和清单，**不包含实际数据文件**。

## HAI 存储概览

HAI 存储路径：`/mnt/data/rfuav/`

| 目录 | 用途 | 大小 |
|---|---|---|
| `official_rfuav/` | **RFUAV 官方原始 IQ 数据**（训练用）| ~100 GB |
| `goorm_dronerf/` | Goorm DroneRF 补充数据 | ~18 GB |
| `pluto_raw/` | Pluto 实采数据（推理验证用）| ~50 MB |

> ⚠️ **重要说明**：
> - HAI 上的 `official_rfuav/`（~100 GB / **7 机型**）**不是 RFUAV 官方全集**
> - RFUAV 官方全集约 **1.3 TB**，包含 **37 种无人机**原始 IQ 数据
> - HAI 上只有 kitofrank 当时整理的部分机型
> - **Pluto 实采数据禁止用于模型训练**

---

## official_rfuav/ — RFUAV 官方原始 IQ 数据

**存放位置（HAI）**：`/mnt/data/rfuav/official_rfuav/`

### 机型清单

| 机型 | 大小 | 频段 | 来源 |
|---|---|---|---|
| DJI AVATA2 | 32 GB | **5.8 GHz** | 原有 |
| DJI FPV COMBO | 22 GB | **5.8 GHz** | 原有 |
| DJI MAVIC3 PRO | 14 GB | **5.8 GHz** | 原有 |
| DAUTEL EVO NANO | 9 GB | **5.8 GHz** | HF 下载新增 |
| DJI MINI3.1 | 13 GB | **2.4 GHz** | 原有 |
| DJI MINI4 PRO | 7.5 GB | **2.4 GHz** | 原有 |
| DEVENTION DEVO | 4.5 GB | **2.4 GHz** | HF 下载新增 |

**总计：~100 GB，7 个机型**

### 频段分布

| 频段 | 机型数 | 机型 |
|---|---|---|
| **5.8 GHz** | 4 | DJI AVATA2、DJI FPV COMBO、DJI MAVIC3 PRO、DAUTEL EVO NANO |
| **2.4 GHz** | 3 | DJI MINI3.1、DJI MINI4 PRO、DEVENTION DEVO |

### 采集参数

| 参数 | 值 |
|---|---|
| 设备 | USRP X310 |
| 采样率 | 100 MHz |
| 数据格式 | Complex Float（.iq 二进制）|
| 中心频率 | 随机型变化（见上表）|

---

## 数据来源

| 渠道 | 内容 | 地址 | 状态 |
|---|---|---|---|
| **GitHub** | 代码 + processed spectrogram | https://github.com/kitoweeknd/RFUAV | ✅ 可访问 |
| **HuggingFace** | 预处理 spectrogram 图片（1710×1710 JPG，35类）| https://huggingface.co/datasets/kitofrank/RFUAV | ✅ 可访问 |
| **HuggingFace HF下载** | RFUAV 原始 .rar（部分机型）| /mnt/data/rfuav/rfuav_hf/ | ✅ 已归档 |
| **原始 IQ 文件（1.3TB）** | USRP 原始 IQ 数据，37 种机型 | 需联系作者获取 | ⚠️ 联系 xulu@zstu.edu.cn |

> **关键区分**：
> - GitHub/HuggingFace 上的数据是**预处理好的频谱图 JPG 图片**（不是原始 IQ）
> - **原始 .iq 文件**从 HuggingFace 下载的 .rar 包中解压获取
> - RFUAV 官方全集（1.3TB）需邮件联系作者 Lu Xu（xulu@zstu.edu.cn）

---

## pluto_raw/ — Pluto 实采数据

**存放位置（HAI）**：`/mnt/data/rfuav/pluto_raw/`

用于推理验证，**禁止用于模型训练**。

---

## 其他数据目录

| 目录 | 用途 | 大小 |
|---|---|---|
| `goorm_dronerf/` | Goorm DroneRF 补充数据（AR drone、Bepop、Phantom）| ~18 GB |
| `dataset_v5/` | 早期训练数据集（已废弃）| ~39 GB |
| `dataset_v5_final/` | 最终训练数据集（已废弃）| ~37 GB |
| `dji_subset/` | DJI 子集数据（历史）| ~28 GB |

> ⚠️ 已废弃目录仅供参考，不用于当前训练流程。

---

## 数据使用规范

- **禁止**将 Pluto 实采数据（`pluto_raw/`）用于模型训练
- 仅使用 `official_rfuav/` 进行训练
- 推理验证可使用 `pluto_raw/`
- 不同频段（5.8GHz / 2.4GHz）可混合训练，学的是跳频信号特征
