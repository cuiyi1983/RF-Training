# Stage2 — 无人机型号识别

## 任务

识别无人机的具体型号。

## 模型

ResNet50（可替换为 EfficientNet/ViT 等）

## 输入

STFT 频谱图，shape (224, 224, 3)

## 输出

20 类无人机型号分类

## 训练

```bash
python stage2/train.py --config configs/stage2/train.yaml
```
