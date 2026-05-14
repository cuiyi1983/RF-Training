# 模型版本注册表（RF-Training）

> 记录所有训练产出的模型版本、指标、下载链接

---

## Stage1 YOLO 检测模型

### v2.0 - yolo_stage1_v2（当前最佳）✅

| 项目 | 值 |
|---|---|
| **模型文件** | `models/stage1_yolo_v2/best.onnx` |
| **frps 下载** | http://yoyo-chat.cn:8000/stage1_yolo_v2.onnx |
| **训练框架** | Ultralytics YOLOv5su（pretrained）|
| **训练数据** | RFUAV 7机型，10051 train / 2597 val（降采样100MHz→60MHz）|
| **训练 epoch** | 71（早停，patience=20）|
| **最佳 epoch** | 62 |
| **mAP50** | **0.808** |
| **mAP50-95** | **0.746** |
| **Precision** | **0.944** |
| **Recall** | **0.685** |
| **nc** | 1（仅 drone 类）|
| **input size** | 640×640 |
| **训练时间** | 2026-05-13 22:38 → 2026-05-14 00:11（约1.5h）|
| **导出时间** | 2026-05-14 00:43 |
| **commit** | 见下方 git log |

**训练日志**：`logs/training/stage2.log`
**指标曲线**：`logs/training/results.csv`

---

### 旧版本（已废弃）

#### v1.0 - yolo_stage1（旧数据）
- **路径**：HAI `/mnt/data/rfuav/yolo_stage1/run1/weights/best.pt`
- **mAP50**：约0.75（旧数据，2026-05-05训练）
- **状态**：⚠️ 已废弃，基于旧数据训练

---

## 版本历史

| 版本 | 日期 | mAP50 | 说明 |
|---|---|---|---|
| v1.0 | 2026-05-05 | ~0.75 | 旧数据训练，已废弃 |
| **v2.0** | **2026-05-14** | **0.808** | **RFUAV 7机型，降采样数据，当前最佳** |
