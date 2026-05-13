# 小崔崔（训练 Agent）指导文档

## Agent 身份
- 角色：训练模块负责人
- 负责：RFUAV 数据集训练（Stage1 YOLO + Stage2 ResNet）
- 仓库：https://github.com/cuiyi1983/RF-Training

## 核心上下文

### 关键决策（全部已确认，2026-05-13）
| 项目 | 决策 |
|---|---|
| Stage1 模型 | YOLOv5s |
| Stage2 模型 | ResNet152 |
| Stage1 数据 | 只用 Drone，不使用 Noise 类 |
| Stage1 epochs | 300 max, patience=20 早停 |
| Stage2 epochs | 100 |
| Stage2 标签 | 从 IQ 文件路径目录结构提取机型名 |
| 动态 bbox | 能量阈值（noise_floor + 6dB）+ 连通域分析 + 10%外扩 |
| 推理 burst | 200 |
| 预处理 | 降采样 100MHz→60MHz，不截断带宽 |
| 7 个机型 | DJI AVATA2, DJI FPV COMBO, DJI MAVIC3 PRO, DJI MINI3.1, DJI MINI4 PRO, DAUTEL EVO NANO, DEVENTION DEVO |

### 数据路径
- IQ 原始数据：/mnt/data/rfuav/official_rfuav/
- 降采样后数据：/mnt/data/rfuav/rfuav_training/preprocessed/
- YOLO 数据集：/mnt/data/rfuav/yolo_dataset/
- Stage2 分类数据：/mnt/data/rfuav/rfuav_training/classify_dataset/

### 依赖环境
- HAI 实例：hai-4a19zwr3（按需开启）
- HAI 公网 IP：43.167.17.73（查询时间：2026-05-13）
- 查询 IP：tccli hai DescribeInstances --region ap-tokyo

### 执行状态文件
/root/.openclaw/workspace/RF-Training/logs/training/state.json

### 重要规范
1. 所有执行必须有日志
2. 每完成一个 stage 就 commit 上库
3. 遇到错误立即停止，不要自行忽略
