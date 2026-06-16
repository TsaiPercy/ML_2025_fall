import func
from torchvision import models

if __name__ == "__main__":
    model = models.efficientnet_b7(weights="IMAGENET1K_V1")
    func.build_model(model, num_classes=2)