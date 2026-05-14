#!/usr/bin/env python3
"""
Stage 3: 分类数据准备 - 为 Stage4 ResNet152 生成 7 机型分类数据集
输入: Stage1 生成的频谱图 + YOLO bbox 标签
输出: classify_dataset/{train,val}/{机型名}/ 目录下的裁剪图
"""
import os
import sys
import json
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

# ========== 配置 ==========
YOLO_IMG_DIR = "/mnt/data/rfuav/yolo_dataset/images"
YOLO_LBL_DIR = "/mnt/data/rfuav/yolo_dataset/labels"
OUT_BASE = "/mnt/data/rfuav/rfuav_training/classify_dataset"
IMG_SIZE = 224  # ResNet152 输入尺寸
TRAIN_RATIO = 0.8

WORK_LOG = "/mnt/data/rfuav/rfuav_training/logs/training/stage3.log"
STATE_FILE = "/mnt/data/rfuav/rfuav_training/logs/training/state.json"

os.makedirs(os.path.dirname(WORK_LOG), exist_ok=True)

def log(msg):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    print(line)
    with open(WORK_LOG, "a") as f:
        f.write(line + "\n")

def update_state(running=True, images_generated=None, errors=None):
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except:
        state = {"stages": {}}
    if running:
        state.setdefault("stages", {})["stage3_classify_data"] = {
            "status": "running",
            "started_at": "2026-05-14T09:10:00Z",
            "images_generated": images_generated,
            "errors": errors
        }
    else:
        state.setdefault("stages", {})["stage3_classify_data"] = {
            "status": "completed",
            "started_at": "2026-05-14T09:10:00Z",
            "completed_at": "2026-05-14T09:30:00Z",
            "images_generated": images_generated,
            "errors": errors
        }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_bbox(label_path):
    """读取YOLO格式bbox列表"""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls, cx, cy, w, h = map(float, parts[:5])
                bboxes.append((cls, cx, cy, w, h))
    return bboxes

def crop_and_resize(img_path, bbox, target_size=IMG_SIZE):
    """根据YOLO bbox裁剪频谱图区域并resize"""
    try:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size  # 640, 640
        cls, cx, cy, w, h = bbox
        x1 = int((cx - w/2) * W)
        y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W)
        y2 = int((cy + h/2) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img.crop((x1, y1, x2, y2))
        crop_resized = crop.resize((target_size, target_size), Image.BILINEAR)
        return crop_resized
    except:
        return None

def extract_model_name(fn):
    """从文件名提取机型名，格式: DJI_AVATA2__pack1_0-1s_w000.jpg"""
    base = os.path.splitext(fn)[0]
    parts = base.split("__")
    if len(parts) >= 1:
        return parts[0].replace("_", " ")
    return "unknown"

def main():
    log("=" * 60)
    log("Stage3: 分类数据准备 开始")
    log("=" * 60)

    update_state(running=True)

    # 收集所有 items（每个item: (img_path, lbl_path, fn)）
    train_items = []
    val_items = []

    for fn in sorted(os.listdir(os.path.join(YOLO_IMG_DIR, "train"))):
        if fn.endswith(".jpg"):
            img_path = os.path.join(YOLO_IMG_DIR, "train", fn)
            lbl_path = os.path.join(YOLO_LBL_DIR, "train", fn.replace(".jpg", ".txt"))
            train_items.append((img_path, lbl_path, fn))

    for fn in sorted(os.listdir(os.path.join(YOLO_IMG_DIR, "val"))):
        if fn.endswith(".jpg"):
            img_path = os.path.join(YOLO_IMG_DIR, "val", fn)
            lbl_path = os.path.join(YOLO_LBL_DIR, "val", fn.replace(".jpg", ".txt"))
            val_items.append((img_path, lbl_path, fn))

    log("Train images: {}, Val images: {}".format(len(train_items), len(val_items)))

    # 清理输出目录
    if os.path.exists(OUT_BASE):
        shutil.rmtree(OUT_BASE)

    total_generated = 0
    class_counts = {}
    errors = 0

    # 处理 train（80%）
    np.random.seed(42)
    train_indices = np.random.permutation(len(train_items))
    n_train = int(len(train_indices) * TRAIN_RATIO)
    train_used = sorted(train_indices[:n_train])  # 取前80%，排序保证确定性

    log("处理 train ({}/{})...".format(len(train_used), len(train_items)))
    for i in tqdm(train_used, desc="train"):
        img_path, lbl_path, fn = train_items[i]
        model_name = extract_model_name(fn)
        out_dir = os.path.join(OUT_BASE, "train", model_name)
        os.makedirs(out_dir, exist_ok=True)
        bboxes = load_bbox(lbl_path)
        if not bboxes:
            continue
        crop = crop_and_resize(img_path, bboxes[0])
        if crop is None:
            errors += 1
            continue
        out_path = os.path.join(out_dir, fn)
        crop.save(out_path, quality=95)
        total_generated += 1
        class_counts[model_name] = class_counts.get(model_name, 0) + 1

    # 处理 val（20%）
    val_indices = np.random.permutation(len(val_items))
    n_val = int(len(val_indices) * TRAIN_RATIO)
    val_used = sorted(val_indices[n_val:])  # val中后20%作为验证集

    log("处理 val ({}/{})...".format(len(val_used), len(val_items)))
    for i in tqdm(val_used, desc="val"):
        img_path, lbl_path, fn = val_items[i]
        model_name = extract_model_name(fn)
        out_dir = os.path.join(OUT_BASE, "val", model_name)
        os.makedirs(out_dir, exist_ok=True)
        bboxes = load_bbox(lbl_path)
        if not bboxes:
            continue
        crop = crop_and_resize(img_path, bboxes[0])
        if crop is None:
            errors += 1
            continue
        out_path = os.path.join(out_dir, fn)
        crop.save(out_path, quality=95)
        total_generated += 1
        class_counts[model_name] = class_counts.get(model_name, 0) + 1

    log("完成！生成 {} 张分类图片".format(total_generated))
    log("类别分布: {}".format(class_counts))
    log("错误数: {}".format(errors))

    update_state(running=False, images_generated=total_generated, errors=errors)
    log("Stage3 完成")

if __name__ == "__main__":
    main()
