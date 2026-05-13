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

### 频谱图生成脚本：使用已有实现

**已有正确的实现**：`/mnt/data/rfuav/prepare_dataset_v5_final.py`

该脚本已完整实现：
- 降采样（100→60 MHz）
- 按 10ms 窗口切片
- 正确的数据划分（80/20 train/val）

**使用方法**：
```bash
cd /mnt/data/rfuav
python3 prepare_dataset_v5_final.py
```

**输出**：`/mnt/data/rfuav/dataset_v5_final/`
```
dataset_v5_final/
├── train/
│   └── drone/    (4574 samples)
└── val/
    └── drone/    (1900 samples)
```

**数据划分依据**：
- **Drone 类**：来自 `official_rfuav/` 下的全部 7 个机型（AVATA2、FPV COMBO、MAVIC3 PRO、DAUTEL EVO NANO、MINI3.1、MINI4 PRO、DEVENTION DEVO）

> ⚠️ **重要更新（2026-05-13）**：Stage1 **不使用 Noise 类**。
> 
> **原因**：non_dji 信号（如 Futaba/FRSKY 遥控器）有自己的跳频特征，与"真正纯噪声"完全不同。把 non_dji 当 noise 训练可能导致模型学到"non_dji 特征 = noise"，反而干扰 drone 识别。
> 
> **正确做法**：Stage1 只用 Drone 数据训练。推理时，有检测框 = 有无人机；无检测框 = 无无人机。

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

def generate_labels(spec_db, spec_shape):
    """
    基于信号检测算法动态生成 YOLO bbox
    
    流程：能量阈值分割 → 连通域分析 → 外接矩形
    输入：spec_db: STFT 幅度（dB格式），spec_shape: (H, W)
    输出：YOLO 格式标签字符串，或 None（检测不到信号）
    """
    import numpy as np
    from scipy import ndimage
    
    H, W = spec_shape
    
    # ① 噪声底估计（按频率轴方向，列统计）
    noise_floor = np.percentile(spec_db, 15, axis=1, keepdims=True)  # shape: (H, 1)
    
    # ② 能量阈值分割：高于噪声底 6dB 的点标记为信号
    threshold = noise_floor + 6.0
    mask = (spec_db > threshold).astype(np.uint8)
    
    # ③ 形态学清理：开运算去噪点 + 闭运算填空洞
    struct = np.ones((3, 3), dtype=np.uint8)
    mask = ndimage.binary_opening(mask, structure=struct)
    mask = ndimage.binary_closing(mask, structure=struct)
    
    # ④ 连通域分析
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None  # 检测不到信号，不生成标签
    
    # ⑤ 取最大连通域的外接矩形
    areas = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_region_id = areas.argmax() + 1  # area 数组索引从 0，但 labeled 从 1 开始
    
    # 标记最大连通域
    largest_mask = (labeled == largest_region_id)
    
    # 计算外接矩形（行=频率方向，列=时间方向）
    rows = np.any(largest_mask, axis=1)  # 每行是否有信号点
    cols = np.any(largest_mask, axis=0)  # 每列是否有信号点
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # ⑥ 外扩 10% margin
    margin_h = int(H * 0.10)
    margin_w = int(W * 0.10)
    rmin = max(0, rmin - margin_h)
    rmax = min(H - 1, rmax + margin_h)
    cmin = max(0, cmin - margin_w)
    cmax = min(W - 1, cmax + margin_w)
    
    # ⑦ 转换为 YOLO 归一化格式 (cx, cy, w, h)
    cx = (cmin + cmax) / 2.0 / W
    cy = (rmin + rmax) / 2.0 / H
    w = (cmax - cmin) / 1.0 / W
    h = (rmax - rmin) / 1.0 / H
    
    # class_id: 0=drone
    bbox = [cx, cy, w, h]
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
                spec_db = iq_to_spectrogram(iq)  # 返回 dB 格式频谱图
                img = spectrogram_to_image(spec_db)
                
                # 从 IQ 文件路径提取机型名（路径结构：.../official_rfuav/{机型名}/.../{base}.npy）
                rel_path = os.path.relpath(fpath, SRC_PREPROCESSED)
                parts = rel_path.split(os.sep)  # 按路径分隔
                model_name = parts[0] if len(parts) > 1 else 'unknown'
                # 文件名规范：{机型名}__{原始base}.jpg（例如 DJI_AVATA2__pack1_0-1s.jpg）
                original_base = os.path.splitext(os.path.basename(fpath))[0]
                safe_model_name = model_name.replace(' ', '_')
                out_base = f"{safe_model_name}__{original_base}"
                
                img_path = os.path.join(img_dir, f"{out_base}.jpg")
                lbl_path = os.path.join(lbl_dir, f"{out_base}.txt")
                
                # 保存图片
                img.save(img_path, quality=95)
                
                # 动态 bbox 标签（基于信号检测）
                label = generate_labels(spec_db, spec_db.shape)  # spec_db 是 dB 格式
                if label is not None:
                    with open(lbl_path, 'w') as f:
                        f.write(label + '\n')
                # else: 检测不到信号，不生成标签文件（该样本不入训练集）
                
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
EPOCHS = 300
PATIENCE = 20              # 早停：连续20个epoch无提升则停止
BATCH_SIZE = 16
IMG_SIZE = 640

def create_dataset_yaml():
    """生成 YOLO dataset.yaml"""
    content = f"""
path: /mnt/data/rfuav/rfuav_training/yolo_dataset
train: images/train
val: images/val

nc: 1
names: ['drone']  # Stage1: 只检测 drone，推理时无检测=无无人机
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
        '--patience', str(PATIENCE),          # 早停
        '--save_period', '10',
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

**标签来源**：从 IQ 文件的目录路径提取机型名。

```
示例 IQ 路径：
/mnt/data/rfuav/official_rfuav/DJI AVATA2/DJI AVATA2/VTSBW=20/pack2_0-1s.iq
                                                ↑
                           从路径第2层目录名提取机型：DJI AVATA2

分类标签（7 类）：
  0: DJI AVATA2
  1: DJI FPV COMBO
  2: DJI MAVIC3 PRO
  3: DJI MINI3.1
  4: DJI MINI4 PRO
  5: DAUTEL EVO NANO
  6: DEVENTION DEVO
```

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
NUM_CLASSES = 7  # 7个机型分类

# 机型标签映射
CLASS_NAMES = [
    'DJI AVATA2',
    'DJI FPV COMBO',
    'DJI MAVIC3 PRO',
    'DJI MINI3.1',
    'DJI MINI4 PRO',
    'DAUTEL EVO NANO',
    'DEVENTION DEVO',
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

EPOCHS = 100  # ResNet152 通常收敛较快
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
        
        # 标签从文件名解析机型名
        # Stage1 频谱图文件命名规范：{机型名}__{原IQ文件名}.jpg
        # 例如：DJI_AVATA2__pack1_0-1s.jpg
        # Stage3 直接从文件名提取机型名
        filename = os.path.splitext(self.images[idx])[0]  # 去掉 .jpg
        model_name = filename.split('__')[0]             # 取 __ 前面的机型名
        # 机型名还原为空格（生成时将空格替换为下划线）
        model_name = model_name.replace('_', ' ')
        label = CLASS_MAP.get(model_name, 0)
        
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
Pluto 采集 IQ（60MHz, burst=200）
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

## 关键决策点

| # | 问题 | 状态 | 决策 |
|---|---|---|---|
| 1 | **Noise 数据来源**：来自 `non_dji/` 目录 | ✅ 已解决 | **废弃，不用于 Stage1** |
| 2 | **Bbox 策略**：动态 bbox（能量阈值+连通域分析）| ✅ 已解决 | 信号检测算法驱动，外扩10% margin |
| 3 | **Stage1 数据策略**：只训 Drone 还是 Drone + Noise？ | ✅ 已解决 | **只用 Drone，推理时无检测=无无人机** |
| 4 | **Stage1 训练 epoch**：100 是否足够？ | ✅ 已决定 | 100 epochs + patience=20 早停 |
| 5 | **Stage2 训练数据**：分类标签如何获取？ | ✅ 已决定 | 从 IQ 文件路径的目录结构提取机型名 |
