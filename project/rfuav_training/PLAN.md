# RFUAV 训练计划

> 草案 v0.3 — 严格遵循 RFUAV 官方流程，**对 RFUAV 官方数据集进行降采样/带宽截断预处理**以适配 Pluto 硬件能力

---

## 一、RFUAV 官方完整流程（Reference）

### 1.1 数据预处理流程

```
原始 IQ 文件（USRP X310, .iq, Complex Float）
    ↓
STFT 生成频谱图
    - fs = 100 MHz
    - nperseg = 1024（或 2048）
    - window = hamming
    - return_onesided = False
    - fftshift(axes=0)
    - 幅度：10 * log10(|Zxx|)
    - 输出：JPG/PNG 频谱图
    ↓
频谱图数据集（train/val 按类别目录组织）
```

**RFUAV 官方 STFT 实现**（`graphic/RawDataProcessor.py`）：
```python
f, t, Zxx = stft(data, fs=100e6,
                  return_onesided=False,
                  window=windows.hamming(stft_point=1024),
                  nperseg=1024)
Zxx = np.fft.fftshift(Zxx, axes=0)
aug = 10 * np.log10(np.abs(Zxx))  # dB 幅度
```

### 1.2 训练流程

#### Stage 1 — 检测模型（YOLOv5 / YOLOv8）
#### Stage 2 — 分类模型（ResNet50 等）

---

## 二、Pluto 硬件约束分析

### 2.1 硬件能力对比

| 参数 | RFUAV 官方（USRP X310）| Pluto SDR | 差距 |
|---|---|---|---|
| 采样率 | **100 MHz** | 60 MHz | -40% |
| 带宽 | **100 MHz** | 56 MHz | -44% |
| 中心频率 | 5760 MHz | 5805 MHz（推荐）| +45 MHz |
| 频段覆盖 | 全频段 | 600 MHz - 5.9 GHz | 受限 |

### 2.2 预处理目标

**核心问题**：Pluto 的采样率和带宽均低于 RFUAV 官方采集条件，需要在训练阶段就对 RFUAV 官方数据集进行预处理，模拟 Pluto 的观测条件。

```
RFUAV 官方原始 IQ（100 MHz）
    ↓
[预处理] 降采样 + 带宽截断 → 适配 Pluto 条件
    ↓
STFT 生成频谱图（60 MHz）
    ↓
训练流程（与 RFUAV 官方一致）
```

---

## 三、预处理方案分析

### 3.1 降采样（100 MHz → 60 MHz）

**实现方式**：先低通滤波，再降采样

```python
from scipy.signal import butter, filtfilt

def downsample_iq(iq_data, orig_fs=100e6, target_fs=60e6):
    # 设计低通滤波器（奈奎斯特频率 = target_fs/2 = 30 MHz）
    cutoff = 0.9 * (target_fs / 2)  # 27 MHz，避免频谱边缘突变
    b, a = butter(5, cutoff / (orig_fs / 2), btype='low')
    filtered = filtfilt(b, a, iq_data)  # 零相位滤波
    # 计算降采样因子
    ratio = int(orig_fs / target_fs)
    downsampled = filtered[::ratio]  # 降采样
    return downsampled
```

### 3.2 带宽截断（100 MHz → 56 MHz）

**实现方式**：频域截断

```python
def bandwidth_limit(iq_data, orig_fs=100e6, target_bw=56e6):
    # FFT → 截断中心 56MHz → IFFT
    N = len(iq_data)
    freq_bins = np.fft.fftfreq(N, d=1/orig_fs)
    
    # 中心带宽截断
    bw_bins = int(target_bw / orig_fs * N)
    spectrum = np.fft.fft(iq_data)
    spectrum = np.fft.fftshift(spectrum)
    # 保留中心 56MHz，两侧截断
    center = N // 2
    spectrum[center-bw_bins//2:center+bw_bins//2] = spectrum[center-bw_bins//2:center+bw_bins//2]
    spectrum[:center-bw_bins//2] = 0
    spectrum[center+bw_bins//2:] = 0
    spectrum = np.fft.fftshift(spectrum)
    return np.fft.ifft(spectrum).astype(np.complex64)
```

### 3.3 问题与可行方案

#### 问题 1：频率分辨率下降

| 指标 | 100 MHz（官方）| 60 MHz（Pluto）| 影响 |
|---|---|---|---|
| 频率分辨率 | 97.6 kHz（N=1024）| 58.6 kHz（N=1024）| 降低 40% |
| 相邻跳频频点区分 | 清晰 | 可能重叠 | 跳频检测能力下降 |

**可行方案**：
- 保持 nperseg=1024 不变（与 RFUAV 官方一致），接受分辨率下降
- 备注：分辨率下降对二分类任务（drone vs noise）影响较小

#### 问题 2：DJI 跳频信号带宽覆盖不足

| 指标 | 值 |
|---|---|
| DJI 5.8GHz 跳频总带宽 | ~125 MHz（5725-5850 MHz）|
| Pluto 带宽 | 56 MHz |
| 覆盖率 | ~45%（只能覆盖部分跳频范围）|

**可行方案**：
- 中心频率设为 5805 MHz，覆盖 DJI 跳频中心区域（5775-5830 MHz）
- 训练数据中仅使用 5805 MHz 附近的 56MHz 频段数据
- 推理时 Pluto 也使用相同中心频率

#### 问题 3：混叠风险

- 降采样前未滤除高于 30MHz（奈奎斯特频率）的频率成分，可能导致高频信号混叠到低频
- **必须先低通滤波再降采样**

#### 问题 4：训练数据信息损失

- 降采样 + 带宽截断会丢失原始 RFUAV 信号中部分高频特征
- **方案**：在训练时对每个样本做不同程度的降采样/带宽裁剪，增强模型鲁棒性（数据增强）

---

## 四、预处理配置

```yaml
# project/rfuav_training/configs/stage1/preprocess.yaml

# ========== RFUAV 官方参数 ==========
rfuav:
  orig_sampling_rate: 100e6    # RFUAV 官方采样率
  orig_bandwidth: 100e6         # RFUAV 官方带宽

# ========== Pluto 硬件限制 ==========
pluto:
  target_sampling_rate: 60e6    # Pluto 采样率
  target_bandwidth: 56e6         # Pluto 带宽
  center_freq: 5805e6            # 中心频率

# ========== 预处理参数 ==========
preprocess:
  filter_cutoff: 27e6            # 低通滤波器截止频率（0.9 × 30 MHz）
  filter_order: 5                # 滤波器阶数
  downsample_ratio: 5/3          # 降采样比（100→60 = 5/3）

# ========== STFT 参数（与 RFUAV 官方一致，仅 fs 变化）==========
stft:
  fs: 60e6                       # 降采样后采样率
  nperseg: 1024                  # 与 RFUAV 官方一致
  hop: 512                       # 50% 重叠
  window: hamming                # 与 RFUAV 官方一致
  return_onesided: false          # 与 RFUAV 官方一致
  fftshift: true                 # 与 RFUAV 官方一致
  amp_type: dB                   # 10 * log10(|Zxx|)，与 RFUAV 官方一致

# ========== 数据增强（增强鲁棒性）==========
augmentation:
  random_bandwidth_crop: true    # 随机带宽裁剪（模拟不同观测窗口）
  random_noise_injection: false   # 噪声注入（可选）
```

---

## 五、训练流程

### Stage 1 — 二分类（无人机 vs 噪声）

```
Step 1: RFUAV 原始数据获取
  - HAI: /mnt/data/rfuav/official_rfuav/
  - 采样率: 100 MHz, 带宽: 100 MHz

Step 2: 预处理
  - 降采样: 100 MHz → 60 MHz（低通滤波 + 降采样）
  - 带宽截断: 100 MHz → 56 MHz
  - 中心频率: 5805 MHz（对准 DJI 跳频中心）
  - 输出: 预处理后的 IQ 数据

Step 3: STFT 生成频谱图
  - fs=60MHz, nperseg=1024, hamming, dB幅度
  - resize 到 (640, 640)

Step 4: 数据集构建
  - RFUAV 20 机型 → drone 类
  - 噪声数据 → noise 类
  - 划分: train / val

Step 5: 模型训练
  - ResNet50（ImageNet 预训练）
  - 二分类输出

Step 6: Pluto 实测验证
  - scan_20260508（5 噪声 + 5 无人机）
```

---

## 六、待讨论问题

1. **降采样方法**：低通滤波 + 降采样，还是直接降采样（容忍混叠）？
2. **带宽截断策略**：固定截断中心 56MHz，还是随机带宽裁剪作为数据增强？
3. **nperseg 是否调整**：是否需要增大 nperseg 以补偿频率分辨率下降？（nperseg 翻倍 = 分辨率翻倍）
4. **训练数据子集**：是否仅使用 5805 MHz 附近的 RFUAV 数据，而非全量 100MHz 数据？
