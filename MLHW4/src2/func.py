import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models

from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import random
import numpy as np
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.fnames = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        return self.transform(img), fname

class PseudoDataset(Dataset): #PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
        
def check_model_size(model):
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1e6
    
    #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"========================================")
    print(f"Total Params (M): {total_params_m:.2f} M")
    print(f"========================================")


def build_model(model, num_classes=2):

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    check_model_size(model)

    return model

def load_weight(model, weight_path):
    
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    return model

def train_one_epoch(model, dataloader, optimizer, criterion):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")

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

    pbar = tqdm(dataloader, desc="Evaluating")

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

def pred_data(pbar, model, threshold):

    result = []

    with torch.no_grad():
        for imgs, fnames in pbar:
            imgs = imgs.to(device)

            outputs = model(imgs)

            # ---------------------------------------------------------
            # label change
            # ori: g->0, r->1
            # competition: r->0, g->1
            # ---------------------------------------------------------

            # class0 prob
            probs = F.softmax(outputs, dim=1)[:, 0]

            # if class0 prob > threshold, pred as 1
            preds = (probs > threshold).long()

            
            for fname, pred in zip(fnames, preds.cpu().tolist()):
                result.append([fname[:-4], pred])

    return result

def set_transform(img_resize):

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        #transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD
        ),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD
        ),
    ])


    return train_tf, test_tf

