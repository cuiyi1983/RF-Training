#!/usr/bin/env python3
"""
Stage 2: YOLOv5s 训练 - RFUAV Stage1 检测模型
nc=1（仅drone类，无Noise类）
"""
import os
import sys
import subprocess
import json
from datetime import datetime

DATA_YAML = "/mnt/data/rfuav/yolo_dataset/dataset.yaml"
OUTPUT_DIR = "/mnt/data/rfuav/rfuav_training/yolo_stage1_v2"
WORK_LOG = "/mnt/data/rfuav/rfuav_training/logs/training/stage2.log"
STATE_FILE = "/mnt/data/rfuav/rfuav_training/logs/training/state.json"
EPOCHS = 300
PATIENCE = 20
BATCH_SIZE = 16
IMGSZ = 640

os.makedirs(os.path.dirname(WORK_LOG), exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    print(line)
    with open(WORK_LOG, "a") as f:
        f.write(line + "\n")

def update_state_running(epochs_trained=0, best_map=None):
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except:
        state = {"stages": {}}
    state.setdefault("stages", {})["stage2_yolo"] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "best_map": best_map,
        "epochs_trained": epochs_trained,
        "errors": None
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def main():
    log("=" * 60)
    log("Stage2 YOLO Training 开始")
    log("=" * 60)
    log("DATA_YAML: {}".format(DATA_YAML))
    log("OUTPUT_DIR: {}".format(OUTPUT_DIR))
    log("EPOCHS: {} (patience={})".format(EPOCHS, PATIENCE))
    log("BATCH_SIZE: {}".format(BATCH_SIZE))

    train_dir = os.path.join(os.path.dirname(DATA_YAML), "images/train")
    val_dir = os.path.join(os.path.dirname(DATA_YAML), "images/val")
    train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    val_count = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
    log("Train images: {}, Val images: {}".format(train_count, val_count))

    if train_count == 0:
        log("ERROR: No training images!")
        sys.exit(1)

    update_state_running(epochs_trained=0)
    log("开始训练...")

    # 训练代码（避免f-string嵌套引号问题）
    train_code_path = "/mnt/data/rfuav/rfuav_training/stage2_train_inner.py"
    with open(train_code_path, "w") as f:
        f.write("""
import os
from ultralytics import YOLO
import torch
import json

print('CUDA:', torch.cuda.is_available())
print('Device: cuda:0')

model = YOLO('yolov5su.pt')

results = model.train(
    data='{data_yaml}',
    epochs={epochs},
    patience={patience},
    imgsz={imgsz},
    batch={batch_size},
    device='0',
    project='{output_dir}',
    name='run1',
    exist_ok=True,
    pretrained=True,
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    box=0.05,
    cls=0.5,
    kobj=1.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=1.0,
    degrees=0.0,
    save_period=10,
    verbose=True,
    seed=42,
)

best_map = results.results().box.map
print('Best mAP50:', best_map)

# 更新 state
STATE_FILE = '{state_file}'
try:
    with open(STATE_FILE) as f:
        state = json.load(f)
except:
    state = {{'stages': {{}}}}
state.setdefault('stages', {{}})['stage2_yolo'] = {{
    'status': 'completed',
    'started_at': '2026-05-13T17:19:00Z',
    'best_map': float(best_map),
    'epochs_trained': {epochs},
    'errors': 0
}}
with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

# 导出 ONNX
best_path = '{output_dir}/run1/weights/best.pt'
if os.path.exists(best_path):
    print('Exporting to ONNX...')
    model_exp = YOLO(best_path)
    onnx_path = model_exp.export(format='onnx')
    print('ONNX:', onnx_path)
""".format(
            data_yaml=DATA_YAML,
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMGSZ,
            batch_size=BATCH_SIZE,
            output_dir=OUTPUT_DIR,
            state_file=STATE_FILE
        ))

    result = subprocess.run(['/usr/bin/python3', train_code_path])
    if result.returncode != 0:
        log("ERROR: Training failed code={}".format(result.returncode))
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            state["stages"]["stage2_yolo"]["status"] = "failed"
            state["stages"]["stage2_yolo"]["errors"] = result.returncode
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except:
            pass
        sys.exit(result.returncode)

    log("Stage2 完成")

if __name__ == "__main__":
    main()
