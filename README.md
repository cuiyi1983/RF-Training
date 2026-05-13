# RF-Training

基于 RFUAV 数据集的无人机检测与识别训练框架。

## 目录结构

```
RF-Training/
├── datasets/              # 原始数据集
├── project/               # 训练工作区
│   └── rf_training/      # RFUAV 训练项目
│       ├── configs/      # 训练配置
│       ├── stage1/       # Stage1 训练
│       ├── stage2/       # Stage2 训练
│       ├── stft/         # STFT 预处理
│       ├── utils/
│       ├── scripts/
│       └── models/       # 模型输出
└── release/              # 训练成果发布区
```

## 快速开始

```bash
# Stage1 训练
python project/rf_training/stage1/train.py --config project/rf_training/configs/stage1/train.yaml
```

详见各子目录 README.md
