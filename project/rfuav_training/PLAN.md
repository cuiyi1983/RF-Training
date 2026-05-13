# RFUAV 训练计划

> 版本 v1.0 — 基于两阶段检测架构（Stage1 YOLO + Stage2 ResNet）

---

## 一、方案架构

### 1.1 两阶段检测器（TwoStagesDetector）

```
实时推理流程：

Pluto 采集 IQ（60 MHz）
      ↓
STFT 生成频谱图
      ↓
[Stage 1: YOLOv5 检测器]
  输入：频谱图 (640×640)
  输出：所有无人机区域（bbox），数量不限
      ↓
[Stage 2: ResNet152 分类器]
  输入：每个检测框裁剪出的区域
  输出：该区域是什么机型（7 选 1）
      ↓
最终结果：[机型A @频率X, 机型B @频率Y, ...]
```

### 1.2 核心优势

| 特性 | 说明 |
|---|---|
| **多无人机** | YOLO 每个检测框独立输出，数量不限 ✅ |
| **频率定位** | 检测框对应频谱图位置，可推算信号频率 |
| **与论文一致** | 完全对齐 RFUAV 官方的 TwoStagesDetector 架构 |
| **模块化** | Stage1/Stage2 可独立迭代 |

---

## 二、当前数据集

### 2.1 HAI RFUAV 数据（7 机型，~100 GB）

**存放位置**：`/mnt/data/rfuav/official_rfuav/`

| 机型 | 大小 | 频段 | 中心频率 |
|---|---|---|---|
| DJI AVATA2 | 32 GB | **5.8 GHz** | 5760 MHz |
| DJI FPV COMBO | 22 GB | **5.8 GHz** | 5760 MHz |
| DJI MAVIC3 PRO | 14 GB | **5.8 GHz** | 5800 MHz |
| DAUTEL EVO NANO | 9 GB | **5.8 GHz** | 5770 MHz |
| DJI MINI3.1 | 13 GB | **2.4 GHz** | 2470 MHz |
| DJI MINI4 PRO | 7.5 GB | **2.4 GHz** | 2450 MHz |
| DEVENTION DEVO | 4.5 GB | **2.4 GHz** | 2440 MHz |

**总计：~100 GB，7 个机型**

### 2.2 频段分布

| 频段 | 机型数 | 覆盖范围 |
|---|---|---|
| **5.8 GHz** | 4 | AVATA2、FPV COMBO、MAVIC3 PRO、DAUTEL EVO NANO |
| **2.4 GHz** | 3 | MINI3.1、MINI4 PRO、DEVENTION DEVO |

### 2.3 采集参数

| 参数 | 值 |
|---|---|
| 设备 | USRP X310 |
| 采样率 | 100 MHz |
| 数据格式 | Complex Float（.iq 二进制）|

---

## 三、预处理方案

### 3.1 核心问题

Pluto 采样率（60 MHz）低于 RFUAV 原始采集（100 MHz），需要在训练前对 RFUAV 数据进行降采样预处理。

### 3.2 方案 B：仅降采样，不做带宽截断

```
RFUAV 原始 IQ（100 MHz）
    ↓
[预处理 Stage 0]
  ① 低通滤波（截止 27 MHz，防止混叠）
  ② 降采样（100 MHz → 60 MHz）
    ↓
60 MHz 等效 IQ
    ↓
STFT 生成频谱图（fs=60 MHz）
```

**为什么不截断带宽**：
- Pluto 推理时是多个频点轮询（5760/5775/5800/5825/5850 MHz）
- 每次只看 56 MHz 宽的窗口
- 降采样后全带宽自然覆盖 30 MHz，两侧会自动衰减
- 推理时多频点轮询可以覆盖不同频率区间

### 3.3 降采样代码

```python
from scipy.signal import butter, filtfilt

def downsample_iq(iq_data, orig_fs=100e6, target_fs=60e6):
    # ① 低通滤波（截止 27 MHz，避免混叠）
    cutoff = 0.9 * (target_fs / 2)  # 27 MHz
    b, a = butter(5, cutoff / (orig_fs / 2), btype='low')
    filtered = filtfilt(b, a, iq_data)
    # ② 降采样（100→60，用 scipy.resample_poly 更精确）
    from scipy.signal import resample_poly
    downsampled = resample_poly(filtered, up=3, down=5)
    return downsampled
```

### 3.4 STFT 参数（与 RFUAV 官方一致，仅 fs 变化）

| 参数 | RFUAV 官方 |  Pluto 训练 |
|---|---|---|
| fs | 100 MHz | **60 MHz** |
| nperseg | 1024 | 1024 |
| hop | 512 | 512 |
| window | hamming | hamming |
| return_onesided | False | False |
| fftshift | True | True |
| 幅度 | 10·log10(\|Zxx\|) | 10·log10(\|Zxx\|) |
| resize | — | 640×640（YOLO 输入）|

---

## 四、训练流程

### Stage 1 — YOLOv5 检测模型

**任务**：二分类——有无人机 / 噪声，输出检测框

```
Step 1: 数据获取
  - HAI: /mnt/data/rfuav/official_rfuav/
  - 7 个机型全部使用

Step 2: 预处理
  - 降采样：100 MHz → 60 MHz
  - STFT 生成频谱图（fs=60 MHz）
  - YOLO 格式标注（bbox）

Step 3: 数据集构建
  - drone 类：7 个机型所有频谱图
  - noise 类：噪声数据（可从 RFUAV 元数据生成，或额外采集）
  - 划分：train / val

Step 4: 模型训练
  - 骨架：YOLOv5s（Ultralytics）
  - 输入：640×640
  - 输出：bbox + 类别（drone/noise）
  - 预训练：COCO 预训练权重

Step 5: 验证
  - RFUAV 留出 20% 作为验证集
  - Pluto 实测（scan_20260508）
```

### Stage 2 — ResNet152 分类模型

**任务**：多分类——7 个机型 + 噪声，识别具体型号

```
Step 1: 数据准备
  - Stage1 生成的所有检测框区域
  - 标注：每个区域对应的真实机型

Step 2: 数据集构建
  - 8 类：7 个机型 + 噪声
  - 划分：train / val

Step 3: 模型训练
  - 骨架：ResNet152（ImageNet 预训练）
  - 输入：检测框区域 resize 到 224×224
  - 输出：8 类分类

Step 4: 验证
  - RFUAV 留出 20% 作为验证集
```

### 推理 Pipeline

```
Pluto 采集 IQ
      ↓
STFT 生成频谱图
      ↓
[YOLOv5 Stage1]
  → 检测所有无人机区域（bbox）
      ↓
[ResNet152 Stage2]
  → 每个 bbox 裁剪 → 分类 → 机型
      ↓
结果：[AVATA2 @ 5760MHz, FPV COMBO @ 5800MHz]
```

---

## 五、配置参数汇总

### 5.1 预处理配置

```yaml
preprocess:
  orig_sampling_rate: 100e6    # RFUAV 原始采样率
  target_sampling_rate: 60e6  # Pluto 采样率
  filter_cutoff: 27e6          # 低通滤波截止频率
  filter_order: 5             # 滤波器阶数
  downsample_ratio: 5/3        # 降采样比（100→60）
```

### 5.2 STFT 配置

```yaml
stft:
  fs: 60e6                    # 降采样后采样率
  nperseg: 1024               # 与 RFUAV 官方一致
  noverlap: 512               # 50% 重叠
  window: hamming             # 与 RFUAV 官方一致
  return_onesided: false      # 与 RFUAV 官方一致
  fftshift: true               # 与 RFUAV 官方一致
  amp_type: dB                 # 10 * log10(|Zxx|)
  resize: [640, 640]           # YOLO 输入尺寸
```

### 5.3 Stage1 YOLO 配置

```yaml
stage1:
  model: yolov5s
  input_size: [640, 640]
  pretrained: coco
  classes: 2                   # drone, noise
  epochs: 100
  batch_size: 16
```

### 5.4 Stage2 ResNet 配置

```yaml
stage2:
  model: resnet152
  input_size: [224, 224]
  pretrained: imagenet
  num_classes: 8               # 7 机型 + 噪声
  epochs: 100
  batch_size: 32
```

---

## 六、已确认决策

| 项目 | 决策 | 日期 |
|---|---|---|
| 架构 | 两阶段（Stage1 YOLO + Stage2 ResNet）| 2026-05-13 |
| Stage1 模型 | YOLOv5s | 2026-05-13 |
| Stage2 模型 | ResNet152 | 2026-05-13 |
| 多无人机支持 | 必须 ✅（YOLO 天然支持）| 2026-05-13 |
| 预处理 | 仅降采样（100→60 MHz），不截断带宽 | 2026-05-13 |
| STFT 参数 | nperseg=1024，与 RFUAV 官方一致 | 2026-05-13 |
| 训练策略 | 分步训，串起来用（路径 B）| 2026-05-13 |
| 数据集 | 7 机型，~100 GB | 2026-05-13 |

---

## 七、待讨论/待确认

| # | 问题 | 状态 |
|---|---|---|
| 1 | Stage1 训练 epoch / batch size 具体数值 | 待定 |
| 2 | Stage2 训练 epoch / batch size 具体数值 | 待定 |
| 3 | noise 类数据来源（RFUAV 已有，还是需要额外生成？）| 待讨论 |
| 4 | 推理 burst 配置（burst=100 vs burst=500）| 待崔老板决策 |
| 5 | 5.8GHz 和 2.4GHz 数据是否混合训练 | 待讨论 |
| 6 | 是否需要联系作者获取 RFUAV 全集（37 机型）| 待决策 |

---

## 八、数据使用规范

- **禁止**将 Pluto 实采数据（`pluto_raw/`）用于模型训练
- 仅使用 `official_rfuav/` 进行训练
- 不同频段（5.8GHz / 2.4GHz）可混合训练，学的是跳频信号特征
- HAI 上其他历史数据（`dataset_v5/`、`dji_subset/` 等）已废弃，不用于当前流程
