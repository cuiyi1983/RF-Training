# RFUAV 训练详细流程指南

> 版本 v1.0 — 可完整复刻的执行手册

---

## 数据格式规范（必须先理解）

### 原始 IQ 文件格式
- **格式**：SignalHound IQ 二进制，`complex64`（实部+虚部交替）
- **采样率**：100 MHz（来自 USRP X310）
- **文件大小**：每 100M samples ≈ 763 MB（每 sample 8 bytes）
- **文件名规则**：每个文件包含 1 秒数据（1M samples × 8 bytes）
  ```
  pack1_0-1s.iq   → 0-1 秒
  pack1_1-2s.iq   → 1-2 秒
  ...
  ```
- **XML 元数据**：每个 pack 目录有一个 `.xml` 文件，记录 center_freq、sample_rate 等

### 目录结构

```
/mnt/data/rfuav/official_rfuav/
├── DAUTEL EVO NANO/
│   ├── pack1_0-1s.iq     ← 扁平结构，无子目录
│   ├── pack1_1-2s.iq
│   └── pack1.xml
│
├── DJI AVATA2/
│   └── DJI AVATA2/
│       ├── VTSBW=10/
│       │   └── pack1_0-1s.iq   ← 有子目录，按带宽分组
│       ├── VTSBW=20/
│       │   └── pack2_0-1s.iq
│       └── VTSBW=60/
│           └── pack3_0-1s.iq
│
├── DJI MINI3.1/
│   └── DJI MINI3/
│       ├── VTSBW=10/
│       │   └── pack1_0-1s.iq
│       └── VTSBW=20/
│           └── pack2_0-1s.iq
```

### XML 元数据格式（重要）

```xml
<?xml version="1.0" encoding="UTF-8"?>
<SignalHoundIQFile Version="1.0">
    <DeviceType>USRPX310</DeviceType>
    <Drone>DJI AVATA2</Drone>          ← 机型名称
    <SerialNumber>00004</SerialNumber>
    <CenterFrequency>5765000000.000</CenterFrequency>  ← 中心频率（Hz）
    <SampleRate>100000000</SampleRate>   ← 采样率（Hz）
    <IFBandwidth>100000000</IFBandwidth> ← 带宽（Hz）
    <SampleCount>100000000</SampleCount>  ← 总样本数
</SignalHoundIQFile>
```

---

## Stage 0: 预处理（降采样）

**目标**：将 RFUAV 原始 IQ（100 MHz）降采样到 60 MHz，适配 Pluto 推理条件。

**输入**：`/mnt/data/rfuav/official_rfuav/*/.../*.iq`
**输出**：`/mnt/data/rfuav/rfuav_training/preprocessed/` 下的 `.npy` 文件

### 预处理脚本：`scripts/stage0_preprocess.py`

```python
#!/usr/bin/env python3
"""
Stage 0: RFUAV IQ 数据降采样预处理
将 100MHz IQ 数据降采样到 60MHz
"""
import os
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

# ========== 配置 ==========
SRC_BASE = "/mnt/data/rfuav/official_rfuav"
OUT_BASE = "/mnt/data/rfuav/rfuav_training/preprocessed"
ORIG_FS = 100e6      # RFUAV 原始采样率
TARGET_FS = 60e6      # Pluto 采样率
GCD = np.gcd(int(ORIG_FS), int(TARGET_FS))
UP = int(TARGET_FS // GCD)    # 3
DOWN = int(ORIG_FS // GCD)    # 5
CUTOFF = 27e6                  # 低通截止频率 27 MHz

def butter_lowpass(cutoff, fs, order=5):
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def preprocess_iq_file(iq_path, out_path):
    """降采样一个 IQ 文件"""
    if os.path.exists(out_path):
        return True  # 跳过已处理
    
    # 读取 IQ（complex64）
    try:
        data = np.fromfile(iq_path, dtype=np.complex64)
    except Exception as e:
        print(f"  读取失败 {iq_path}: {e}")
        return False
    
    # ① 低通滤波
    b, a = butter_lowpass(CUTOFF, ORIG_FS, order=5)
    filtered = filtfilt(b, a, data)
    
    # ② 降采样（100→60 MHz）
    downsampled = resample_poly(filtered, UP, DOWN)
    
    # ③ 功率归一化
    orig_power = np.mean(np.abs(data)**2)
    ds_power = np.mean(np.abs(downsampled)**2)
    if ds_power > 0:
        scale = np.sqrt(orig_power / ds_power)
        downsampled = downsampled * scale
    
    # 保存为 numpy
    np.save(out_path, downsampled.astype(np.complex64))
    return True

def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    
    # 遍历所有 .iq 文件
    count = 0
    for root, dirs, files in os.walk(SRC_BASE):
        for fn in files:
            if not fn.endswith('.iq'):
                continue
            iq_path = os.path.join(root, fn)
            
            # 计算输出路径：保持目录结构
            rel = os.path.relpath(iq_path, SRC_BASE)
            rel_no_ext = rel.replace('.iq', '')
            out_path = os.path.join(OUT_BASE, rel_no_ext + '.npy')
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            preprocess_iq_file(iq_path, out_path)
            count += 1
            if count % 50 == 0:
                print(f"  已处理 {count} 个文件")
    
    print(f"总计处理 {count} 个 IQ 文件")
    print(f"输出目录: {OUT_BASE}")

if __name__ == '__main__':
    main()
```

### 执行命令

```bash
cd /mnt/data/rfuav/rfuav_training/
python3 scripts/stage0_preprocess.py
```

### 预期输出
- 处理时间：~100GB 数据约需 1-2 小时
- 输出：`.npy` 文件，保存在 `/mnt/data/rfuav/rfuav_training/preprocessed/`

---

## Stage 1: STFT 频谱图生成

**目标**：将降采样后的 IQ 数据转换为频谱图（YOLO 训练用）。

**输入**：`/mnt/data/rfuav/rfuav_training/preprocessed/*.npy`
**输出**：频谱图图片 + YOLO 标签文件

### STFT 参数

| 参数 | 值 | 说明 |
|---|---|---|
| fs | 60 MHz | 降采样后采样率 |
| nperseg | 1024 | 窗长 |
| noverlap | 512 | 50% 重叠 |
| window | hamming | 与 RFUAV 官方一致 |
| nfft | 1024 | FFT 点数 |
| return_onesided | False | 全频谱 |
| fftshift | True | 中心零频移位 |
| 幅度 | 10·log10(\|Zxx\|) | dB 归一化 |
| resize | 640×640 | YOLO 输入尺寸 |

### 频谱图生成脚本：`scripts/stage1_generate_spectrograms.py`

```python
#!/usr/bin/env python3
"""
Stage 1: 生成 STFT 频谱图
输出 YOLO 格式数据集（images/ + labels/）
"""
import os
import numpy as np
from scipy.signal import stft, windows
from PIL import Image

# ========== 配置 ==========
SRC_PREPROCESSED = "/mnt/data/rfuav/rfuav_training/preprocessed"
OUT_BASE = "/mnt/data/rfuav/rfuav_training/yolo_dataset"

# STFT 参数
FS = 60e6
NFFT = 1024
NOVLAP = 512
WIN = windows.hamming(NFFT)

# YOLO 输入尺寸
TARGET_SIZE = (640, 640)

# ========== 关键：确定噪声数据来源 ==========
# RFUAV 数据中，每个机型的部分文件是纯噪声（无无人机信号）
# 从 XML 元数据中读取 ReferenceSNRLevel 或通过人工标注确定
# 以下为简化策略：使用 RFUAV 自带的噪声采样文件

NOISE_SAMPLES_DIR = "/mnt/data/rfuav/rfuav_training/noise_samples"

def load_noise_samples():
    """加载噪声采样文件列表"""
    noise_files = []
    if os.path.exists(NOISE_SAMPLES_DIR):
        for fn in os.listdir(NOISE_SAMPLES_DIR):
            if fn.endswith('.npy'):
                noise_files.append(os.path.join(NOISE_SAMPLES_DIR, fn))
    return noise_files

def iq_to_spectrogram(iq_data):
    """IQ → dB 归一化频谱图"""
    # STFT
    f, t, Zxx = stft(
        iq_data, fs=FS,
        window=WIN,
        nperseg=NFFT,
        noverlap=NOVLAP,
        nfft=NFFT,
        boundary=None,
        padded=False
    )
    
    # fftshift（将零频移到中心）
    Zxx = np.fft.fftshift(Zxx, axes=0)
    
    # dB 幅度
    mag_dB = 10 * np.log10(np.abs(Zxx) + 1e-10)
    
    # Min-Max 归一化到 [0, 1]
    v_min = mag_dB.min()
    v_max = mag_dB.max()
    if v_max > v_min:
        spec = (mag_dB - v_min) / (v_max - v_min)
    else:
        spec = np.zeros_like(mag_dB)
    
    return spec.astype(np.float32)

def spectrogram_to_image(spec, target_size=TARGET_SIZE):
    """频谱图 → 640×640 RGB 图片"""
    # [0,1] → [0,255]
    img = (spec * 255).astype(np.uint8)
    # 灰度图
    pil = Image.fromarray(img, mode='L')
    # 3 通道（YOLO 需要 RGB）
    pil = pil.convert('RGB')
    # resize
    pil = pil.resize(target_size, Image.BILINEAR)
    return pil

def generate_labels(spec_shape, center_freq, sample_rate):
    """
    生成 YOLO 标签
    
    RFUAV 的跳频信号特点：
    - 信号能量集中在特定频率区间
    - 用信号检测算法找到活跃区域 → 生成 bbox
    
    简化策略：使用固定 bbox（覆盖 DJI 跳频典型范围）
    - DJI 5.8GHz: 5725-5850 MHz → 占带宽 ~125 MHz
    - Pluto 60MHz 带宽: 5760-5820 MHz（中心 5790 MHz）
    - bbox 覆盖整个可用带宽
    """
    h, w = spec_shape  # 频谱图尺寸
    
    # 固定 bbox：覆盖整个频谱图高度的 80%
    # center_x=0.5（水平方向中心）, center_y=0.5（垂直方向中心）
    # width=1.0（整个宽度）, height=0.8（80% 高度）
    bbox = [0.5, 0.5, 1.0, 0.8]
    
    # class_id: 0=无人机, 1=噪声（Stage1 二分类）
    label = "0 " + " ".join([f"{v:.6f}" for v in bbox])
    return label

def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    
    splits = {'train': 0.8, 'val': 0.2}
    
    for split_name, split_ratio in splits.items():
        img_dir = os.path.join(OUT_BASE, 'images', split_name)
        lbl_dir = os.path.join(OUT_BASE, 'labels', split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
    
    # 遍历所有预处理后的 IQ 文件
    all_files = []
    for root, dirs, files in os.walk(SRC_PREPROCESSED):
        for fn in files:
            if fn.endswith('.npy'):
                all_files.append(os.path.join(root, fn))
    
    print(f"找到 {len(all_files)} 个预处理文件")
    
    # 划分 train/val
    np.random.seed(42)
    indices = np.random.permutation(len(all_files))
    n_train = int(len(indices) * 0.8)
    
    train_files = [all_files[i] for i in indices[:n_train]]
    val_files = [all_files[i] for i in indices[n_train:]]
    
    for split_name, files in [('train', train_files), ('val', val_files)]:
        print(f"\n=== {split_name} ({len(files)} 文件) ===")
        img_dir = os.path.join(OUT_BASE, 'images', split_name)
        lbl_dir = os.path.join(OUT_BASE, 'labels', split_name)
        
        for i, fpath in enumerate(files):
            try:
                iq = np.load(fpath)
                spec = iq_to_spectrogram(iq)
                img = spectrogram_to_image(spec)
                
                # 输出文件名
                base = os.path.splitext(os.path.basename(fpath))[0]
                img_path = os.path.join(img_dir, f"{base}.jpg")
                lbl_path = os.path.join(lbl_dir, f"{base}.txt")
                
                # 保存图片
                img.save(img_path, quality=95)
                
                # 生成标签（简化：全用固定 bbox）
                label = generate_labels(spec.shape, 0, 0)
                with open(lbl_path, 'w') as f:
                    f.write(label + '\n')
                
                if (i+1) % 200 == 0:
                    print(f"  {split_name}: {i+1}/{len(files)}")
                    
            except Exception as e:
                print(f"  错误 {fpath}: {e}")
    
    print(f"\n完成！数据集保存在: {OUT_BASE}")

if __name__ == '__main__':
    main()
```

---

## Stage 2: YOLO 模型训练

**目标**：训练 YOLOv5s 检测模型，检测频谱图中的无人机信号区域。

### 训练脚本：`scripts/stage2_train_yolo.py`

```python
#!/usr/bin/env python3
"""
Stage 2: YOLOv5s 训练
"""
import os
import subprocess

YOLO_REPO = "/mnt/data/rfuav/rfuav_training/yolov5"
DATA_YAML = "/mnt/data/rfuav/rfuav_training/yolo_dataset.yaml"
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640

def create_dataset_yaml():
    """生成 YOLO dataset.yaml"""
    content = f"""
path: /mnt/data/rfuav/rfuav_training/yolo_dataset
train: images/train
val: images/val

nc: 2
names: ['drone', 'noise']
"""
    with open(DATA_YAML, 'w') as f:
        f.write(content.strip())
    print(f"Dataset YAML 已生成: {DATA_YAML}")

def main():
    create_dataset_yaml()
    
    cmd = [
        'python3', os.path.join(YOLO_REPO, 'train.py'),
        '--img', str(IMG_SIZE),
        '--batch', str(BATCH_SIZE),
        '--epochs', str(EPOCHS),
        '--data', DATA_YAML,
        '--weights', 'yolov5s.pt',    # COCO 预训练权重
        '--name', 'rfuav_stage1',
        '--project', '/mnt/data/rfuav/rfuav_training/runs',
        '--device', '0',               # GPU
    ]
    
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == '__main__':
    exit(main())
```

### 执行命令

```bash
# 克隆 YOLOv5（如果尚未克隆）
cd /mnt/data/rfuav/rfuav_training/
git clone https://github.com/ultralytics/yolov5.git

# 安装依赖
pip install -r yolov5/requirements.txt

# 启动训练
python3 scripts/stage2_train_yolo.py
```

### 模型输出
```
/mnt/data/rfuav/rfuav_training/runs/train/rfuav_stage1/weights/best.pt
```

---

## Stage 3: Stage2 分类数据准备

**目标**：将 Stage1 的检测结果（检测框区域）裁剪出来，构建 Stage2 分类数据集。

### 数据准备脚本：`scripts/stage3_prepare_classify.py`

```python
#!/usr/bin/env python3
"""
Stage 3: 准备 Stage2 分类数据集
对每个频谱图应用 Stage1 YOLO 检测，提取检测框区域作为分类输入
"""
import os
import numpy as np
from PIL import Image
import torch
import sys

# 添加 YOLOv5 路径
sys.path.insert(0, '/mnt/data/rfuav/rfuav_training/yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

YOLO_WEIGHT = "/mnt/data/rfuav/rfuav_training/runs/train/rfuav_stage1/weights/best.pt"
SRC_IMAGES = "/mnt/data/rfuav/rfuav_training/yolo_dataset/images"
OUT_BASE = "/mnt/data/rfuav/rfuav_training/classify_dataset"

# Stage2 类别映射（7 机型 + 噪声 = 8 类）
# 从目录名推断机型（需要在生成频谱图时记录原始机型信息）
# 简化：用文件名中的目录结构记录

def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    
    # 加载 Stage1 模型
    device = torch.device('0')
    model = DetectMultiBackend(YOLO_WEIGHT, device=device)
    model.eval()
    
    splits = ['train', 'val']
    for split in splits:
        img_dir = os.path.join(SRC_IMAGES, split)
        for fn in os.listdir(img_dir):
            if not fn.endswith('.jpg'):
                continue
            
            img_path = os.path.join(img_dir, fn)
            
            # YOLO 推理
            img0 = Image.open(img_path).convert('RGB')
            img = np.array(img0.resize((640, 640))).transpose(2,0,1)
            img_t = torch.from_numpy(img).unsqueeze(0).float() / 255.0
            img_t = img_t.to(device)
            
            pred = model(img_t)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
            
            # 提取检测框区域
            if pred[0] is not None and len(pred[0]) > 0:
                for i, det in enumerate(pred[0]):
                    # det: [x1,y1,x2,y2,conf,cls]
                    x1,y1,x2,y2 = map(int, det[:4].cpu().numpy())
                    crop = img0.crop((x1,y1,x2,y2))
                    crop = crop.resize((224, 224))  # ResNet 输入尺寸
                    
                    # 保存（文件名编码原始机型信息）
                    out_path = os.path.join(OUT_BASE, split, fn.replace('.jpg', f'_crop{i}.jpg'))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    crop.save(out_path)

if __name__ == '__main__':
    main()
```

---

## Stage 4: ResNet152 分类模型训练

### 训练脚本：`scripts/stage4_train_classifier.py`

```python
#!/usr/bin/env python3
"""
Stage 4: ResNet152 分类模型训练
8 类分类（7 机型 + 噪声）
"""
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# ========== 配置 ==========
DATA_DIR = "/mnt/data/rfuav/rfuav_training/classify_dataset"
NUM_CLASSES = 8
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
IMG_SIZE = 224

# ========== 数据集类 ==========
class SpectrogramDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(os.path.join(root, split, 'images'))
            if f.endswith('.jpg')
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.split, 'images', self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 标签从文件名或目录结构获取
        # 这里需要实现标签解析逻辑
        label = 0  # placeholder
        
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    train_loader = DataLoader(
        SpectrogramDataset(DATA_DIR, 'train', transform_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        SpectrogramDataset(DATA_DIR, 'val', transform_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # 加载 ResNet152（预训练）
    model = models.resnet152(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        # Train
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), '/mnt/data/rfuav/rfuav_training/resnet152_best.pt')
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Acc={acc:.2f}%")
    
    print(f"训练完成，最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
```

---

## 推理 Pipeline

```
Pluto 采集 IQ（60MHz, burst=100）
      ↓
STFT 频谱图（640×640）
      ↓
[YOLOv5 Stage1] → 检测框列表
      ↓
每个检测框裁剪 → resize 224×224
      ↓
[ResNet152 Stage2] → 机型分类
      ↓
[机型, 置信度, 频率位置]
```

---

## 完整执行顺序

```bash
# ===== 环境准备 =====
cd /mnt/data/rfuav/rfuav_training/
pip install numpy scipy pillow torch torchvision ultralytics

# ===== Stage 0: 降采样 =====
python3 scripts/stage0_preprocess.py

# ===== Stage 1: STFT 频谱图生成 =====
python3 scripts/stage1_generate_spectrograms.py

# ===== Stage 2: YOLO 训练 =====
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
python3 scripts/stage2_train_yolo.py

# ===== Stage 3: 分类数据准备 =====
python3 scripts/stage3_prepare_classify.py

# ===== Stage 4: ResNet 训练 =====
python3 scripts/stage4_train_classifier.py
```

---

## 关键决策点（待确认）

| # | 问题 | 影响 |
|---|---|---|
| 1 | **Noise 数据来源**：RFUAV 数据中噪声文件如何识别？ | 影响 Stage1 数据集构建 |
| 2 | **Bbox 策略**：用固定 bbox 还是信号检测算法自动生成？ | 影响标签质量和模型泛化 |
| 3 | **Stage1 训练 epoch**：默认 100 是否足够？ | 影响训练时间和精度 |
| 4 | **Stage2 训练数据**：分类标签如何获取？ | 当前方案依赖文件名编码，需要实现解析逻辑 |
