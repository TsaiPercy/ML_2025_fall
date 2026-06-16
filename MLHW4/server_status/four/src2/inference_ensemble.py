import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader

import numpy as np
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
    print("inference ensemble start")

    test_dir="./data/test"
    
    output_csv="./submission/ensemble_b5_b5_pseudo_b6.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    batch_size=8

    # b5 = img_resize=(456, 456)
    # b6 = img_resize=(528, 528)

    model_list = []
    weight_path_list = []
    img_resize_list = []

    # model 1
    model_list.append(models.efficientnet_b5(weights="IMAGENET1K_V1"))
    weight_path_list.append("./best/1st_test_b5.pth")
    img_resize_list.append((456, 456))

    # model 2
    model_list.append(models.efficientnet_b5(weights="IMAGENET1K_V1"))
    weight_path_list.append("./best/5th_test_b5_pseudo.pth")
    img_resize_list.append((456, 456))

    # model 3
    model_list.append(models.efficientnet_b6(weights="IMAGENET1K_V1"))
    weight_path_list.append("./best/2nd_test_b6.pth")
    img_resize_list.append((528, 528))


    all_results = []
    for i in range(len(model_list)):
        model = model_list[i]
        weight_path = weight_path_list[i]
        img_resize = img_resize_list[i]

        all_results.append(inference(test_dir=test_dir,
                                        model=model,
                                        weight_path=weight_path,
                                        img_resize=img_resize,
                                        batch_size=batch_size
                                        ))
    
    # (N)
    fname_list = [f for f, _ in all_results[0]]

    # (num_models, N)
    pred_np = np.array([[p for _, p in r] for r in all_results])  


    half_vote = len(all_results) / 2
  
    final_preds = (pred_np.sum(axis=0) > half_vote).tolist()
    final_result = [[fname, int(pred)] for fname, pred in zip(fname_list, final_preds)]

    
    write_csv(
        result=final_result,
        output_csv=output_csv,
    )
    


if __name__ == "__main__":
    main()
    
