# RFUAV 训练计划

> 草案 v0.2 — 严格遵循 RFUAV 官方流程，仅针对 Pluto 能力增加预处理适配

---

## 一、RFUAV 官方完整流程（Reference）

### 1.1 数据预处理流程

```
原始 IQ 文件（.iq, Complex Float）
    ↓
STFT 生成频谱图
    - fs = 100 MHz（USRP X310）
    - nperseg = 1024（或 2048）
    - window = hamming
    - return_onesided = False
    - fftshift(axes=0)
    - 幅度：10 * log10(|Zxx|)
    - 输出：JPG/PNG 频谱图（1710×1710 @ 300 DPI）
    ↓
频谱图数据集（train/val 按类别目录组织）
```

**RFUAV 官方 STFT 实现**（`graphic/RawDataProcessor.py`）：
```python
# 核心 STFT 函数
f, t, Zxx = stft(data, fs=100e6,
                  return_onesided=False,
                  window=windows.hamming(stft_point=1024),
                  nperseg=1024)
Zxx = np.fft.fftshift(Zxx, axes=0)
aug = 10 * np.log10(np.abs(Zxx))  # dB 幅度
```

### 1.2 训练流程

#### Stage 1 — 检测模型（YOLOv5 / YOLOv8）
```
频谱图 JPG（RGB）
    ↓
YOLO 检测模型（检测无人机区域）
    ↓
检测结果 + 裁剪区域
```

#### Stage 2 — 分类模型（ResNet50 等）
```
裁剪区域（或完整频谱图）
    ↓
ImageNet 预训练模型（ResNet50/ViT/MobileNet 等）
    ↓
分类输出（35 类 / 37 类）
```

**RFUAV 支持的模型列表**（`trainer.py`）：
- ResNet18/34/50/101/152
- ViT (vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14)
- Swin Transformer (swin_v2_t/s/b)
- MobileNet V3 / V2
- EfficientNet B0-B7
- DenseNet121/161/169/201
- VGG11/13/16/19 (及其 BN 版本)
- ConvNeXt Tiny/Small/Base/Large
- AlexNet, GoogLeNet, ShuffleNet, ResNeXt, Wide ResNet, MNASNet, SqueezeNet, Inception

### 1.3 推理流程

```
原始 IQ 文件 或 频谱图 JPG
    ↓
Stage 1: YOLO 检测（可选）
    ↓
Stage 2: 分类模型
    ↓
输出：机型标签 + 置信度
```

---

## 二、Pluto 硬件约束与适配

### 2.1 硬件能力差距

| 参数 | RFUAV 官方（USRP X310）| Pluto SDR | 差距 |
|---|---|---|---|
| 采样率 | 100 MHz | **60 MHz** | -40% |
| 带宽 | 100 MHz | **56 MHz** | -44% |
| 中心频率 | 5760 MHz | 5805 MHz（推荐）| +45 MHz |
| 频段覆盖 | 全频段 | 600 MHz - 5.9 GHz | 受限 |

### 2.2 Pluto 预处理阶段（新增）

**目标**：将 Pluto 采集的 IQ 数据预处理为与 RFUAV 官方频谱图格式兼容的输入

```
Pluto IQ 数据（60 MHz 采样）
    ↓
[新增] 降采样适配（保持 60 MHz，不降采样）
    ↓
[新增] 带宽截断（56 MHz → 保持）
    ↓
STFT 生成频谱图
    - fs = 60 MHz（Pluto）
    - nperseg = 1024
    - window = hamming
    - return_onesided = False
    - fftshift(axes=0)
    - 幅度：10 * log10(|Zxx|)  ← 与 RFUAV 一致
    - resize 到标准尺寸
    ↓
兼容频谱图
    ↓
RFUAV 训练流程（YOLO / 分类模型）
```

**关键适配点**：
1. **采样率**：60 MHz（不降采样，保持 Pluto 原始采样率）
2. **带宽**：56 MHz（Pluto 实际带宽上限）
3. **STFT 参数**：与 RFUAV 官方一致（hamming, 1024, return_onesided=False, dB幅度）
4. **输出尺寸**：resize 到 YOLO 输入尺寸（640×640）或分类模型输入尺寸

---

## 三、训练配置

### 3.1 Pluto 预处理参数（新增 `stft/pluto_preprocess.py`）

```yaml
# project/rfuav_training/configs/stage1/pluto_stft.yaml

# ========== Pluto 硬件参数 ==========
pluto:
  sampling_rate: 60e6      # 60 MHz（固定）
  bandwidth: 56e6           # 56 MHz
  center_freq: 5805e6       # 5805 MHz（可配置）
  gain: 20                  # dB（20 dB 已有验证数据）

# ========== STFT 参数（与 RFUAV 官方一致）==========
stft:
  fs: 60e6                  # Pluto 采样率
  nperseg: 1024             # 与 RFUAV 官方一致
  hop: 512                  # 50% 重叠
  window: hamming           # 与 RFUAV 官方一致
  return_onesided: false    # 与 RFUAV 官方一致
  fftshift: true            # 与 RFUAV 官方一致
  amp_type: dB              # 10 * log10(|Zxx|)，与 RFUAV 官方一致

# ========== 预处理输出 ==========
preprocess:
  target_size: [640, 640]  # YOLO 输入尺寸
  # 或 [224, 224]          # 分类模型输入尺寸
  normalize: minmax         # 0-1 归一化
```

### 3.2 训练参数（继承 RFUAV 官方）

```yaml
# project/rfuav_training/configs/stage1/train.yaml

# ========== 数据集路径 ==========
data:
  train: /mnt/data/rfuav/dataset_v5_final/train/
  val:   /mnt/data/rfuav/dataset_v5_final/val/

# ========== 模型配置 ==========
model:
  name: resnet50            # 默认 ResNet50（ImageNet 预训练）
  num_classes: 2            # Stage1 二分类：drone / noise
  pretrained: true

# ========== 训练超参数 ==========
train:
  batch_size: 8
  image_size: 640           # YOLO 模式
  # image_size: 224         # 分类模式
  epochs: 150
  lr: 0.00001
  optimizer: adam
  device: cuda
  save_path: ./outputs/stage1/
```

---

## 四、训练流程（按阶段）

### Stage 1 — 二分类（无人机 vs 噪声）

```
Step 1: Pluto 预处理
  - 输入：Pluto IQ 数据（60 MHz）
  - 处理：STFT（60 MHz, 1024, hamming, dB幅度）
  - 输出：频谱图 JPG

Step 2: 数据集构建
  - 使用 RFUAV 官方数据集（H AI 子集 20 机型）
  - 降采样至 60 MHz（模拟 Pluto 采样条件）
  - 划分：train / val

Step 3: 模型训练
  - 采用 RFUAV 官方 ResNet50 训练流程
  - ImageNet 预训练权重
  - 二分类输出（drone / noise）

Step 4: Pluto 实测验证
  - 使用 scan_20260508 数据（5 噪声 + 5 无人机）
  - 评估检测准确率
```

### Stage 2 — 机型识别（可选，待确认）

```
Step 5: Stage 2 数据准备
  - RFUAV 官方 20 机型 / 35 机型分类

Step 6: YOLO 检测模型训练（如采用两阶段方案）
  - 继承 RFUAV 官方 YOLOv5 流程

Step 7: 分类模型训练
  - 继承 RFUAV 官方 ResNet50 流程
  - 多分类输出（20 / 35 类）
```

---

## 五、目录结构

```
project/rfuav_training/
├── stft/
│   ├── pluto_preprocess.py   ← Pluto 预处理脚本（新增）
│   └── rfuav_reference.py     ← RFUAV 官方 STFT 参考实现
├── configs/
│   ├── stage1/
│   │   ├── pluto_stft.yaml   ← Pluto STFT 配置（新增）
│   │   └── train.yaml        ← 训练配置
│   └── stage2/
│       └── train.yaml
├── datasets/                 ← 软链接或清单文件
├── stage1/
│   └── train.py             ← Stage1 训练入口
├── stage2/
│   └── train.py             ← Stage2 训练入口
└── outputs/                ← 训练输出
```

---

## 六、里程碑

```
M1: Pluto 预处理验证
  → Pluto IQ → RFUAV 格式频谱图，验证 STFT 参数正确性
  → 用 scan_20260508 数据验证

M2: Stage 1 模型训练
  → ResNet50 二分类，基于 RFUAV 降采样数据集
  → HAI V100 训练

M3: Stage 1 Pluto 实测
  → scan_20260508 验证
  → 对比现有 stage1_model_v2.onnx 指标

M4: Stage 2 规划（如需要）
  → 确认 YOLO 两阶段 or 纯分类方案
  → 机型数据集准备
```

---

## 七、待确认事项

1. ✅ Pluto 预处理 STFT 参数是否与 RFUAV 官方一致（fs=60MHz, nperseg=1024, hamming, dB幅度）
2. ✅ Stage 1 用 ResNet50（RFUAV 官方 baseline）还是 YOLOv5s？
3. ⚠️ Stage 2 是否需要？（崔老板确认）
4. ⚠️ RFUAV 官方全集（1.3TB / 37 机型）是否获取？
5. ⚠️ HAI 子集 20 机型是否足够训练？
