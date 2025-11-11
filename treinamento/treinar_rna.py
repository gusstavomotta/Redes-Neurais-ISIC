import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT))

import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score

from gerar_relatorio_treinamento import generate_final_report
from model_utils import build_model
from data_utils import prepare_data

import config as cfg 

def get_config_dict():
    return {
        "DATA_DIR": cfg.DATA_DIR,
        "CSV_PATH": cfg.CSV_PATH,
        "DEVICE": cfg.DEVICE_TREINO_EVAL,
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "LR": cfg.LR,
        "EPOCHS": cfg.EPOCHS,
        "THRESHOLD": cfg.THRESHOLD,
        "SEED": cfg.SEED,
        "NUM_WORKERS": cfg.NUM_WORKERS,
        "PATIENCE": cfg.PATIENCE
    }

def train_one_epoch(model, dataloader, criterion, optimizer, device, threshold):
    model.train()
    total_loss, total_acc = 0, 0
    
    for imgs, lbls in tqdm(dataloader, desc="Treinando"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > threshold).float()
        total_acc += (preds == lbls).float().mean().item()
        
    return total_loss / len(dataloader), total_acc / len(dataloader)

def evaluate(model, dataloader, criterion, device, threshold):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Avaliando"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            y_true.extend(lbls.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return avg_loss, f1, recall

def run_training(config, model, train_dl, val_dl, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=config['DEVICE']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])

    best_f1_mel = 0
    patience_counter = 0
    
    for epoch in range(config['EPOCHS']):
        train_loss, _ = train_one_epoch(model, train_dl, criterion, optimizer, config['DEVICE'], config['THRESHOLD'])
        val_loss, val_f1, val_recall = evaluate(model, val_dl, criterion, config['DEVICE'], config['THRESHOLD'])

        if val_f1 > best_f1_mel:
            best_f1_mel = val_f1
            torch.save(model.state_dict(), cfg.MODEL_WEIGHTS_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['PATIENCE']:
                break
    return

if __name__ == '__main__':
    config = get_config_dict() 
    
    random.seed(config['SEED'])
    np.random.seed(config['SEED'])
    torch.manual_seed(config['SEED'])
    torch.cuda.manual_seed_all(config['SEED'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    train_dl, val_dl, test_dl, pos_weight = prepare_data(config)
    
    model = build_model(config['DEVICE'])
    
    run_training(config, model, train_dl, val_dl, pos_weight)

    generate_final_report(config, model, test_dl)
    
    print("Processo concluído.")