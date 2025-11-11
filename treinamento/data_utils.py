import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT))

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from model_utils import get_data_transforms
import config as cfg

class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = str(img_dir)
        self.tf = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]
        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        lbl = torch.tensor(row['label'], dtype=torch.float)
        
        if self.tf:
            img = self.tf(img)
            
        return img, lbl

def prepare_data(config):
    df = pd.read_csv(cfg.CSV_PATH)
    df = df[df['dx'].isin(cfg.CLASSES_UPLOAD)]
    df['label'] = (df['dx'] == 'mel').astype(int)

    train_df, temp_df = train_test_split(df, stratify=df['label'], test_size=0.3, random_state=config['SEED'])
    val_df, test_df = train_test_split(temp_df, stratify=temp_df['label'], test_size=0.5, random_state=config['SEED'])

    tf_train = get_data_transforms(use_augmentation=True)
    tf_test = get_data_transforms(use_augmentation=False)

    train_ds = SkinDataset(train_df, config['DATA_DIR'], tf_train)
    val_ds = SkinDataset(val_df, config['DATA_DIR'], tf_test)
    test_ds = SkinDataset(test_df, config['DATA_DIR'], tf_test)

    counts = train_df['label'].value_counts()
    
    if 0 not in counts: counts[0] = 0
    if 1 not in counts: counts[1] = 0
        
    if counts[0] == 0 or counts[1] == 0:
        pos_weight_value = 1.0
        sampler = None
    else:
        class_weights = 1. / torch.tensor([counts[0], counts[1]], dtype=torch.float)
        sample_weights = torch.tensor([class_weights[label] for label in train_df['label']])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        pos_weight_value = counts[0] / counts[1]

    train_dl = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], sampler=sampler, num_workers=config['NUM_WORKERS'], pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True)
    
    return train_dl, val_dl, test_dl, pos_weight_value