import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

import random
import numpy as np

# ----------------------------------------------------------
# 設定裝置
# ----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"



# ============================================================
# 1. Build EfficientNet Model
# ============================================================

def build_model(num_classes=2):
    model = models.efficientnet_b5(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ============================================================
# 2. Training Function
# ============================================================

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


def set_seed(seed=60):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Set random seed = {seed}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# 3. Main Training Function
# ============================================================

def main():
    set_seed(60)
    print("start")
    g = torch.Generator()
    g.manual_seed(60)

    # ----------------------------------------------------------
    # 路徑設定（依你給的結構）
    # ----------------------------------------------------------
    train_dir = "./data/train"             # 你要確保資料夾格式正確
    save_path = "./model_weight/1st_test_b5.pth"

    # 確保儲存權重的資料夾存在，否則會報錯
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ----------------------------------------------------------
    # check 裝置
    # ----------------------------------------------------------
    
    print("Using device:", device)

    # ----------------------------------------------------------
    # Data Augmentation（必要，提升分數）
    # ----------------------------------------------------------
    train_tf = transforms.Compose([
        transforms.Resize((456, 456)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((456, 456)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ----------------------------------------------------------
    # 建立 Dataset（自動讀取 class0, class1）
    # ----------------------------------------------------------
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset   = datasets.ImageFolder(train_dir, transform=val_tf)

    # (建議：真正作業可切分 train/val，但作業沒強制)
    # 這裡我們全用 train train，模型會更強。

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    # ----------------------------------------------------------
    # 建立模型 + optimizer
    # ----------------------------------------------------------
    model = build_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"   Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f}")
        print(f"   Val   Loss: {val_loss:.5f} | Val   Acc: {val_acc:.5f}")
        print("---------------------------------------------------")

        # ----------------------------------------------------------
        # Save final model weight
        # ----------------------------------------------------------
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()


"""
conda create -n test_for_pretrain python=3.12 -y
conda activate test_for_pretrain


conda deactivate


nvidia-smi ===> cuda version 12.2
# CUDA 11.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118


conda env export > environment.yml
conda env create -f environment.yml
"""