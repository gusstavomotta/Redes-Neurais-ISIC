import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

import config as cfg

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    
    for imgs, lbls in tqdm(dataloader, desc="Treinando"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, device, threshold):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Avaliando"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs).squeeze(1)
            preds = (torch.sigmoid(logits) > threshold).float()
            
            y_true.extend(lbls.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            
    return f1_score(y_true, y_pred, zero_division=0)

def run_training(model, train_dl, val_dl, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=cfg.DEVICE_TREINO_EVAL))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    best_f1 = 0
    patience_counter = 0
    
    for _ in range(cfg.EPOCHS):
        train_one_epoch(model, train_dl, criterion, optimizer, cfg.DEVICE_TREINO_EVAL)
        val_f1 = evaluate(model, val_dl, cfg.DEVICE_TREINO_EVAL, cfg.THRESHOLD)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), cfg.MODEL_WEIGHTS_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                break