# RF-Training

基于 RFUAV 数据集的无人机检测与识别训练框架。

## 项目概述

采用 RFUAV 官方两阶段方案：
```
Stage 1: YOLOv5s 检测（信号是否存在）
Stage 2: 图像分类器（具体型号识别）
```

## 数据集

| 数据集 | 说明 | 采集设备 |
|---|---|---|
| `official_rfuav/` | RFUAV 官方原始 IQ 数据（DJI + non-DJI，共20种机型）| USRP X310 @ 100MHz |
| `dataset_v5_final/` | 已预处理 STFT 频谱图（降采样至 60MHz，适配 Pluto）| — |
| `pluto_raw/` | Pluto 实采数据（推理验证用）| ADALM PLUTO @ 60MHz |

详见 `datasets/README.md`

## 目录结构

```
RF-Training/
├── datasets/         # 数据集配置（不上传数据文件）
├── stft/             # STFT 预处理模块
├── configs/          # 训练配置文件
├── stage1/           # Stage1 训练（有/无无人机二分类）
├── stage2/           # Stage2 训练（无人机型号识别）
├── models/           # 模型输出
├── utils/            # 公共工具
└── scripts/          # 辅助脚本
```

## 硬件要求

- 训练：GPU（推荐 V100 32GB 或更高）
- 推理验证：PC（Windows + onnxruntime）

## 快速开始

```bash
# 数据预处理
python stft/iq_to_spectrogram.py --input datasets/official_rfuav/DJI/ --output datasets/dataset_v5_final/

# Stage1 训练
python stage1/train.py --config configs/stage1/train.yaml

# Stage2 训练
python stage2/train.py --config configs/stage2/train.yaml
```
