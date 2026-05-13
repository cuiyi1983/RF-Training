# Stage1 — 有/无无人机二分类

## 任务

检测频谱图中是否存在无人机信号。

## 模型

YOLOv8n（Ultralytics）

## 输入

STFT 频谱图，shape (640, 640, 3)

## 输出

二分类：有无人机 / 无无人机

## 训练

```bash
python stage1/train.py --config configs/stage1/train.yaml
```
