import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm


class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
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

def get_config():
    return {
        "DATA_DIR": 'data',
        "CSV_PATH": 'data/HAM10000_metadata.csv',
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "BATCH_SIZE": 16,
        "LR": 1e-4,
        "EPOCHS": 25,
        "THRESHOLD": 0.5,
        "SEED": 42,
        "NUM_WORKERS": 4,
        "PATIENCE": 5
    }

def prepare_data(config):
    df = pd.read_csv(config['CSV_PATH'])
    df = df[df['dx'].isin(['mel', 'nv'])]
    df['label'] = (df['dx'] == 'mel').astype(int)

    train_df, temp_df = train_test_split(df, stratify=df['label'], test_size=0.3, random_state=config['SEED'])
    val_df, test_df = train_test_split(temp_df, stratify=temp_df['label'], test_size=0.5, random_state=config['SEED'])

    print("Distribuição de classes no treino (ANTES do balanceamento):")
    print(train_df['label'].value_counts())
    
    tf_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tf_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = SkinDataset(train_df, config['DATA_DIR'], tf_train)
    val_ds = SkinDataset(val_df, config['DATA_DIR'], tf_test)
    test_ds = SkinDataset(test_df, config['DATA_DIR'], tf_test)

    counts = train_df['label'].value_counts()
    class_weights = 1. / torch.tensor([counts[0], counts[1]], dtype=torch.float)
    sample_weights = torch.tensor([class_weights[label] for label in train_df['label']])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], sampler=sampler, num_workers=config['NUM_WORKERS'], pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True)
    
    pos_weight_value = counts[0] / counts[1]
    
    return train_dl, val_dl, test_dl, pos_weight_value

def build_model(device):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, threshold):
    model.train()
    total_loss = 0
    total_acc = 0
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
        
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc

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
    print(f"Peso para a classe positiva (Melanoma) na loss: {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=config['DEVICE']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])

    best_f1_mel = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_recall': [], 'val_f1': []
    }

    for epoch in range(config['EPOCHS']):
        print(f"\n--- Época {epoch+1}/{config['EPOCHS']} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, config['DEVICE'], config['THRESHOLD'])
        history['train_loss'].append(train_loss)
        print(f"Resultado Treino: Perda={train_loss:.4f} | Acurácia={train_acc:.4f}")
        
        val_loss, val_f1, val_recall = evaluate(model, val_dl, criterion, config['DEVICE'], config['THRESHOLD'])
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_recall'].append(val_recall)
        print(f"Resultado Validação: Perda={val_loss:.4f} | Recall MEL={val_recall:.4f} | F1 MEL={val_f1:.4f}")

        if val_f1 > best_f1_mel:
            best_f1_mel = val_f1
            torch.save(model.state_dict(), "resultados/best_model.pt")
            print(f"Novo melhor modelo salvo com F1-Score: {best_f1_mel:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['PATIENCE']:
                print("\nParando por early stopping: paciência esgotada.")
                break
    
    return history

def generate_final_report(config, model, test_dl):
    print("\n--- Iniciando Avaliação Final no Conjunto de Teste ---")
    model.load_state_dict(torch.load("resultados/best_model.pt"))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_dl, desc="Teste Final"):
            imgs = imgs.to(config['DEVICE'])
            logits = model(imgs).squeeze(1)
            y_prob.extend(torch.sigmoid(logits).cpu().tolist())
            y_true.extend(lbls.tolist())
    
    print("\n--- Avaliação com Diferentes Thresholds ---")
    
    thresholds_to_test = np.arange(0.25, 0.76, 0.05) 
    precision_list, recall_list, f1_list = [], [], []

    for thresh in thresholds_to_test:
        y_pred = [1 if p > thresh else 0 for p in y_prob]
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        print(f"Threshold={thresh:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_to_test, precision_list, marker='o', label='Precision')
    plt.plot(thresholds_to_test, recall_list, marker='o', label='Recall')
    plt.plot(thresholds_to_test, f1_list, marker='o', label='F1 Score')
    
    plt.axvline(config['THRESHOLD'], color='red', linestyle='--', label=f"Threshold Configurado ({config['THRESHOLD']})")
    
    plt.xlabel("Threshold")
    plt.ylabel("Valor da Métrica")
    plt.title("Métricas no Conjunto de Teste por Threshold")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("resultados/avaliacao_thresholds.png", dpi=300)
    plt.show()

    print(f"\n--- Relatório Final com Threshold={config['THRESHOLD']} ---")
    y_pred_final = [1 if p > config['THRESHOLD'] else 0 for p in y_prob]
    
    report = classification_report(y_true, y_pred_final, target_names=['Nevo (nv)', 'Melanoma (mel)'])
    roc_auc = roc_auc_score(y_true, y_prob)
    print("\n===== Relatório Final no Conjunto de Teste =====")
    print(report)
    print(f"AUC da Curva ROC: {roc_auc:.4f}")

    with open("resultados/relatorio_teste.txt", "w") as f:
        f.write(f"===== Relatório Final no Conjunto de Teste (Threshold={config['THRESHOLD']}) =====\n")
        f.write(report)
        f.write(f"\nAUC da Curva ROC: {roc_auc:.4f}\n")
        
    cm = confusion_matrix(y_true, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nevo", "Melanoma"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Matriz de Confusão (Threshold={config['THRESHOLD']})")
    plt.tight_layout()
    plt.savefig("resultados/matriz_confusao.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    
    config = get_config() 
    print(config['THRESHOLD'])
    random.seed(config['SEED'])
    np.random.seed(config['SEED'])
    torch.manual_seed(config['SEED'])
    torch.cuda.manual_seed_all(config['SEED'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs("resultados", exist_ok=True)
    print(f"Usando dispositivo: {config['DEVICE']}")

    train_dl, val_dl, test_dl, pos_weight = prepare_data(config)
    
    model = build_model(config['DEVICE'])
    
    history = run_training(config, model, train_dl, val_dl, pos_weight)

    generate_final_report(config, model, test_dl)
