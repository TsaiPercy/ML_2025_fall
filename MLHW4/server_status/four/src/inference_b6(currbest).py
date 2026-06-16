import os
import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def check_model_size(model):
    
    total_params = sum(p.numel() for p in model.parameters())
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"========================================")
    print(f"模型參數檢查：")
    print(f"總參數數量 (Total Params): {total_params:,}")
    print(f"可訓練參數 (Trainable Params): {trainable_params:,}")
    print(f"----------------------------------------")
    
    total_params_m = total_params / 1e6
    print(f"總參數量 (M): {total_params_m:.2f} M")
    
    print(f"========================================")



def build_model(num_classes=2):

    model = models.efficientnet_b6(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    check_model_size(model)
    return model



def load_weight(model, weight_path):
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model





class TestImageDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.test_dir = test_dir
        self.fnames = sorted(os.listdir(test_dir))
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.test_dir, fname)).convert("RGB")
        return self.transform(img), fname


def generate_submission(test_dir, 
                        weight_path, 
                        output_csv="./submission.csv",
                        batch_size=16):
    
    model = build_model()
    model = load_weight(model, weight_path)

    model.eval()

    # Same transform as training
    transform = transforms.Compose([
        transforms.Resize((528, 528)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = TestImageDataset(test_dir, transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    rows = []
    with torch.no_grad():

        pbar = tqdm(loader, desc="Inference")

        for imgs, fnames in pbar:
            imgs = imgs.to(device)

            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)[:, 1]   # class1 prob
            preds = (probs > 0.5).long()

            # ---------------------------------------------------------
            # label change
            # ori: g->0, r->1
            # competition: r->0, g->1
            # ---------------------------------------------------------
            preds = 1 - preds

            for fname, pred in zip(fnames, preds.cpu().tolist()):
                rows.append([fname[:-4], pred])


    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    print("inference start")
    generate_submission(
        test_dir="./data/test",
        weight_path="./model_weight/2nd_test_b6.pth",
        output_csv="./submission/2_b6.csv",
        batch_size=8
    )
