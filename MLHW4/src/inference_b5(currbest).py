import os
import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"


def check_model_size(model):
    # 計算所有參數的總數 (包含 weight 和 bias)
    total_params = sum(p.numel() for p in model.parameters())
    
    # 計算「可訓練」的參數 (有些層如果被凍結就不算在內)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"========================================")
    print(f"模型參數檢查：")
    print(f"總參數數量 (Total Params): {total_params:,}")
    print(f"可訓練參數 (Trainable Params): {trainable_params:,}")
    print(f"----------------------------------------")
    
    # 轉換成 Million (M) 單位
    total_params_m = total_params / 1e6
    print(f"總參數量 (M): {total_params_m:.2f} M")
    
    print(f"========================================")


# ---------------------------------------------------------
# Build EfficientNet Model
# ---------------------------------------------------------
def build_model(num_classes=2):

    model = models.efficientnet_b5(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    check_model_size(model)
    return model


# ---------------------------------------------------------
# Load Saved Weights
# ---------------------------------------------------------
def load_weight(model, weight_path):
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# Predict single image
# ---------------------------------------------------------
def predict_image(model, img):
    transform = transforms.Compose([
        transforms.Resize((456, 456)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)[0, 1].item()
        pred = int(prob > 0.5)

    return pred, prob


# ---------------------------------------------------------
# Main: Generate submission.csv
# ---------------------------------------------------------

def generate_submission(test_dir, weight_path, output_csv="./submission.csv"):
    
    model = build_model()
    model = load_weight(model, weight_path)

    file_list = sorted(os.listdir(test_dir))

    rows = []
    for fname in file_list:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")

        pred, prob = predict_image(model, img)

        # ---------------------------------------------------------
        # label change
        # ori: g->0, r->1
        # competition: r->0, g->1
        # ---------------------------------------------------------
        if pred == 0:
            pred = 1
        else:
            pred = 0

        rows.append([fname[:-4], pred])

    # Write submission.csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    print("inference start")
    generate_submission(
        test_dir="./data/test",
        weight_path="./model_weight/1st_test_b5.pth",
        output_csv="./submission/1_b5.csv"
    )
