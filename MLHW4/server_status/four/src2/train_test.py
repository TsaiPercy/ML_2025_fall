import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import datasets, models

from sklearn.model_selection import StratifiedShuffleSplit

import random
import numpy as np
from tqdm import tqdm

# my append py
import func


device = "cuda" if torch.cuda.is_available() else "cpu"


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

def generate_pseudo_labels(model, unlabel_loader, threshold=0.95):
    model.eval()
    pseudo_imgs = []
    pseudo_labels = []

    pbar = tqdm(unlabel_loader, desc="Pseudo")

    with torch.no_grad():
        for imgs, fnames in pbar:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(dim=1)

            mask = conf > threshold

            for i in range(len(imgs)):
                if mask[i]:
                    pseudo_imgs.append(imgs[i].cpu())
                    pseudo_labels.append(pred[i].item())

    return pseudo_imgs, pseudo_labels


def training(train_dir,
                unlabel_dir,
                save_path,
                batch_size,
                num_workers,
                n_epochs,
                lr,
                weight_decay,
                num_classes,
                model,
                img_resize,
                n_splits,
                test_size,
                seed,
                g,
                use_unlabel=False
            ):
    # ==============================================================
    # --------------------------------------------------------------
    # model setting

    model = func.build_model(model=model, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)


    # ==============================================================
    # --------------------------------------------------------------
    # load data
    
    full_dataset = datasets.ImageFolder(train_dir)
    full_idx = np.array(full_dataset.targets)
   

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=seed
    )

    train_idx, val_idx = next(splitter.split(np.zeros(len(full_idx)), full_idx))

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(full_dataset, val_idx)

    
    train_tf, val_tf = func.set_transform(img_resize=img_resize)

    full_dataset.transform = None
    train_dataset.dataset.transform = train_tf
    val_dataset.dataset.transform   = val_tf

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)



    unlabel_dataset = func.ImageDataset(unlabel_dir, train_tf)
    unlabel_loader  = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                                

    # ==============================================================
    # --------------------------------------------------------------
    # train start

    best_val_acc = 0.0

    for epoch in range(n_epochs):

        
        if use_unlabel:
            if epoch >= 4 and epoch <= 6:  #n_epochs // 2:
            
            
                pseudo_imgs, pseudo_labels = generate_pseudo_labels(model, unlabel_loader, threshold=0.90)
                print(f"append {len(pseudo_imgs)} pseudo")

                if len(pseudo_imgs):

                    pseudo_dataset = func.PseudoDataset(pseudo_imgs, pseudo_labels)


                    combined_train_dataset = ConcatDataset([train_dataset, pseudo_dataset])

                    train_loader = DataLoader(combined_train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            worker_init_fn=seed_worker,
                                            generator=g)





        train_loss, train_acc = func.train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = func.eval_one_epoch(model, val_loader, criterion)
        scheduler.step()

        
        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.5f}")
        print(f"   Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.5f}")
        print("---------------------------------------------------")

        
        if (val_acc > best_val_acc):
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

    # ==============================================================



def main():
    seed = 60
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    print("start")
    print("Using device:", device)

    # --------------------------------------------------------------
    # path setting

    train_dir = "./data/train"
    unlabel_dir = "./data/unlabeled"
    save_path = "./model_weight/3rd_test_b5.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ==============================================================
    # --------------------------------------------------------------
    # hyper setting

    batch_size = 16
    num_workers = 4
    n_epochs = 10

    lr = 1e-4
    weight_decay=1e-5
    num_classes=2


    # model b5
    model = models.efficientnet_b5(weights="IMAGENET1K_V1")

    # b5 = img_resize=(456, 456)
    # b6 = img_resize=(528, 528)
    img_resize=(456, 456)

    # data split
    n_splits=1
    test_size=0.01



    training(
        train_dir=train_dir,
        unlabel_dir=unlabel_dir,
        save_path=save_path,
        batch_size=batch_size,
        num_workers=num_workers,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
        num_classes=num_classes,
        model=model,
        img_resize=img_resize,
        n_splits=n_splits,
        test_size=test_size,
        seed=seed,
        g=g,
        use_unlabel=False
    )
    

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

pip freeze > requirements.txt

tmux
ctrl +b, then d
tmux attach
"""