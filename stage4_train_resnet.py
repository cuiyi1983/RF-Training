#!/usr/bin/env python3
"""
Stage 4: ResNet152 分类训练 - RFUAV 7机型分类
输入: Stage3 生成的 classify_dataset
输出: ResNet152 分类权重 + ONNX
"""
import os
import sys
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ========== 配置 ==========
DATA_DIR = "/mnt/data/rfuav/rfuav_training/classify_dataset"
OUT_DIR = "/mnt/data/rfuav/rfuav_training/resnet152_v1"
WORK_LOG = "/mnt/data/rfuav/rfuav_training/logs/training/stage4.log"
STATE_FILE = "/mnt/data/rfuav/rfuav_training/logs/training/state.json"
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
NUM_WORKERS = 4
IMG_SIZE = 224

# 7个机型
CLASS_NAMES = [
    "DAUTEL EVO NANO",
    "DEVENTION DEVO",
    "DJI AVATA2",
    "DJI FPV COMBO",
    "DJI MAVIC3 PRO",
    "DJI MINI3.1",
    "DJI MINI4 PRO",
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

os.makedirs(os.path.dirname(WORK_LOG), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    print(line)
    with open(WORK_LOG, "a") as f:
        f.write(line + "\n")

def update_state(running=True, best_acc=None, epochs_trained=None):
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except:
        state = {"stages": {}}
    if running:
        state.setdefault("stages", {})["stage4_resnet"] = {
            "status": "running",
            "started_at": "2026-05-14T09:30:00Z",
            "best_acc": best_acc,
            "epochs_trained": epochs_trained,
        }
    else:
        state.setdefault("stages", {})["stage4_resnet"] = {
            "status": "completed",
            "started_at": "2026-05-14T09:30:00Z",
            "completed_at": datetime.now().isoformat(),
            "best_acc": best_acc,
            "epochs_trained": epochs_trained,
            "errors": 0
        }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

class SpectrogramDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []
        
        for model_name in CLASS_NAMES:
            dir_path = os.path.join(DATA_DIR, split, model_name)
            if not os.path.exists(dir_path):
                continue
            label = CLASS_MAP[model_name]
            for fn in os.listdir(dir_path):
                if fn.endswith(".jpg"):
                    self.samples.append(os.path.join(dir_path, fn))
                    self.labels.append(label)
        
        log("{}: {} samples".format(split, len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def main():
    log("=" * 60)
    log("Stage4 ResNet152 训练开始")
    log("=" * 60)
    log("Epochs: {}, Batch: {}, LR: {}".format(EPOCHS, BATCH_SIZE, LR))
    log("Classes: {}".format(CLASS_NAMES))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log("Device: {}".format(device))

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = SpectrogramDataset("train", train_transform)
    val_dataset = SpectrogramDataset("val", val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    log("Train batches: {}, Val batches: {}".format(len(train_loader), len(val_loader)))

    # 模型
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    best_acc = 0.0
    best_acc_epoch = 0
    update_state(running=True, best_acc=0.0, epochs_trained=0)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        scheduler.step()

        # Val
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        log("Epoch {}/{}: train_acc={:.2f}%, val_acc={:.2f}%, lr={:.6f}".format(
            epoch + 1, EPOCHS, train_acc, val_acc, scheduler.get_last_lr()[0]))

        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
            log("  -> New best! saved best.pt")
            update_state(running=True, best_acc=best_acc, epochs_trained=epoch+1)

    log("Training complete! Best val_acc: {:.2f}% at epoch {}".format(best_acc, best_acc_epoch))
    update_state(running=False, best_acc=best_acc, epochs_trained=EPOCHS)

    # 导出 ONNX
    best_path = os.path.join(OUT_DIR, "best.pt")
    if os.path.exists(best_path):
        log("Exporting to ONNX...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        torch.onnx.export(model, dummy, os.path.join(OUT_DIR, "best.onnx"),
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        log("ONNX exported: {}/best.onnx".format(OUT_DIR))

    log("Stage4 完成")

if __name__ == "__main__":
    main()
