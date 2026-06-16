import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from sklearn.model_selection import StratifiedShuffleSplit

import random
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(num_classes=2):
    model = models.efficientnet_b6(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)


    for imgs, labels in pbar:
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

        pbar.set_postfix({
            "loss": f"{total_loss/total:.5f}",
            "acc": f"{correct/total:.5f}"
        })

    return total_loss / total, correct / total


def eval_one_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss/total:.4f}",
                "acc": f"{correct/total:.4f}"
            })

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
    seed = 60
    set_seed(seed)
    print("start")
    g = torch.Generator()
    g.manual_seed(seed)

    train_dir = "./data/train"
    save_path = "./model_weight/2nd_test_b6.pth"

    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    
    print("Using device:", device)

    batch_size = 8
    num_workers = 4

    num_epochs = 10

    model = build_model(num_classes=2).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    train_tf = transforms.Compose([
        transforms.Resize((528, 528)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((528, 528)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


    dataset_full = datasets.ImageFolder(train_dir)

    targets = np.array(dataset_full.targets)
   

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.01,
        random_state=seed
    )

    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

    train_dataset = torch.utils.data.Subset(dataset_full, train_idx)
    val_dataset   = torch.utils.data.Subset(dataset_full, val_idx)

    dataset_full.transform = None
    train_dataset.dataset.transform = train_tf
    val_dataset.dataset.transform   = val_tf

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)


    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.5f}")
        print(f"   Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.5f}")
        print("---------------------------------------------------")

        
        if (val_acc > best_val_acc):
            best_val_acc = val_acc
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