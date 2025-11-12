import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import sys
from pathlib import Path
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT))

from config import IMG_SIZE, NORM_MEAN, NORM_STD

def build_model(device):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    
    in_features = model.classifier[0].in_features
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)

    model = model.to(device)
    return model

def get_data_transforms(use_augmentation: bool = False):
    if use_augmentation:
        return T.Compose([
            T.Resize(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(NORM_MEAN, NORM_STD),
        ])
    else:
        return T.Compose([
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(NORM_MEAN, NORM_STD),
        ])