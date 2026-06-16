import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# my append py
import func

device = "cuda" if torch.cuda.is_available() else "cpu"



def inference(test_dir,
                model,
                weight_path,
                img_resize,
                batch_size=16):
    
    # --------------------------------------------------------------
    # model load

    model = func.build_model(model)
    model = func.load_weight(model, weight_path)
    model.eval()

    # ==============================================================
    # --------------------------------------------------------------
    # load data

    # Same transform as training
    _, test_tf = func.set_transform(img_resize=img_resize)

    test_dataset = func.ImageDataset(test_dir, test_tf)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # ==============================================================
    # --------------------------------------------------------------
    # inference start
    result = []

    pbar = tqdm(test_loader, desc="Inference")

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

            # if class0 prob > 0.5, pred as 1
            preds = (probs > 0.5).long()

            for fname, pred in zip(fnames, preds.cpu().tolist()):
                result.append([fname[:-4], pred])

    return result
    
    


def write_csv(result, 
                output_csv="./submission/result.csv",
                ):

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(result)

    print(f"Saved: {output_csv}")




def main():
    print("inference start")

    test_dir="./data/test"
    weight_path="./model_weight/5th_test_b5_pseudo.pth"

    output_csv="./submission/5_b5_pseudo.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # b5 = img_resize=(456, 456)
    # b6 = img_resize=(528, 528)
    img_resize=(456, 456)
    batch_size=8
    model = models.efficientnet_b5(weights="IMAGENET1K_V1")

    result = inference(test_dir=test_dir,
                        model=model,
                        weight_path=weight_path,
                        img_resize=img_resize,
                        batch_size=batch_size
                        )

    write_csv(
        result=result,
        output_csv=output_csv,
    )
    


if __name__ == "__main__":
    main()
    
