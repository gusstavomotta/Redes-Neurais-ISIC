import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm
from torch.utils.data import ConcatDataset 

DATA_DIR = 'HAM10000_images_part_1'
CSV_PATH = 'HAM10000_metadata.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE, LR, EPOCHS = 16, 1e-4, 25

# Parâmetros de oversampling e threshold
# oversampling 2 está equilibrado entre o número de melanomas e nevos no conjunto de treino
THRESHOLD_TREINAMENTO = 0.35
FATOR_OVERSAMPLING = 2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv(CSV_PATH)
df = df[df['dx'].isin(['mel','nv'])]
df['label'] = (df['dx'] == 'mel').astype(int)

train, temp = train_test_split(df, stratify=df['label'], test_size=0.3, random_state=42)
val, test = train_test_split(temp, stratify=temp['label'], test_size=0.5, random_state=42)

mel_train = train[train['label'] == 1]
train = pd.concat([train] + [mel_train]*FATOR_OVERSAMPLING).sample(frac=1, random_state=42).reset_index(drop=True)

print("Distribuição de classes no treino (split tradicional):")
print(train['label'].value_counts())

class SkinDataset(Dataset):

    def __init__(self, df, img_dir, transforms=None):
        self.df, self.img_dir, self.tf = df.reset_index(drop=True), img_dir, transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]
        img = Image.open(os.path.join(self.img_dir, row['image_id'] + '.jpg')).convert('RGB')
        lbl = row['label']
        if self.tf: img = self.tf(img)
        return img, torch.tensor(lbl, dtype=torch.float)

tf_train = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
tf_test = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

tf_train_padrao = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tf_melanoma_extra = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.8, 1.2)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


train_ds_padrao = SkinDataset(train, DATA_DIR, tf_train_padrao)
mel_train_extra = pd.concat([mel_train]*FATOR_OVERSAMPLING).reset_index(drop=True)
train_ds_mel_aug = SkinDataset(mel_train_extra, DATA_DIR, tf_melanoma_extra)
train_ds = ConcatDataset([train_ds_padrao, train_ds_mel_aug])

val_ds = SkinDataset(val, DATA_DIR, tf_test)
test_ds = SkinDataset(test, DATA_DIR, tf_test)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, BATCH_SIZE)
test_dl = DataLoader(test_ds, BATCH_SIZE)


model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

weights = torch.tensor([(len(train_ds)-train['label'].sum())/len(train_ds),
                        train['label'].sum()/len(train_ds)]).to(DEVICE)


criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(DEVICE))
opt = torch.optim.Adam(model.parameters(), LR)

best_f1_mel = 0
patience = 5
counter = 0

train_loss_history, val_loss_history = [], []
val_recall_mel_history, val_f1_history = [], []

for ep in range(EPOCHS):
    model.train()
    lh, acc = 0, 0
    for imgs, lbls in tqdm(train_dl, desc=f"Epoch {ep+1}"):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        opt.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, lbls)
        loss.backward()
        opt.step()
        lh += loss.item()
        preds = (torch.sigmoid(logits) > THRESHOLD_TREINAMENTO).float()
        acc += (preds == lbls).float().mean().item()

    train_loss = lh / len(train_dl)
    train_loss_history.append(train_loss)
    print(f"[Época {ep+1}] Erro de Treinamento: {train_loss:.4f} | Acurácia: {acc/len(train_dl):.4f}")

    # Validação
    model.eval()
    val_loss, val_acc = 0, 0
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            val_loss += loss.item()

            prob = torch.sigmoid(logits)
            pred = (prob > THRESHOLD_TREINAMENTO).float()

            y_true += lbls.tolist()
            y_pred += pred.tolist()
            y_prob += prob.tolist()

            val_acc += (pred == lbls).float().mean().item()

    val_loss /= len(val_dl)
    val_acc /= len(val_dl)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    val_loss_history.append(val_loss)
    val_recall_mel_history.append(rec)
    val_f1_history.append(f1)

    print(f"[Época {ep+1}] Erro de Validação: {val_loss:.4f} | Acurácia: {val_acc:.4f} | Recall MEL: {rec:.4f} | F1 MEL: {f1:.4f}")

    if f1 > best_f1_mel:
        best_f1_mel = f1
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Parando early: paciência esgotada.")
            break

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

y_true, y_prob = [], []
with torch.no_grad():
    for imgs, lbls in test_dl:
        imgs = imgs.to(DEVICE)
        logits = model(imgs).squeeze(1)
        y_prob += torch.sigmoid(logits).cpu().tolist()
        y_true += lbls.tolist()

print("Avaliação com diferentes thresholds:")
thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
precision_list, recall_list, f1_list = [], [], []

for thresh in thresholds:
    y_pred = [1 if p > thresh else 0 for p in y_prob]
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    print(f"Threshold={thresh:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}")

plt.figure(figsize=(8,5))
plt.plot(thresholds, precision_list, marker='o', label='Precision')
plt.plot(thresholds, recall_list, marker='o', label='Recall')
plt.plot(thresholds, f1_list, marker='o', label='F1 Score')
plt.axvline(THRESHOLD_TREINAMENTO, color='gray', linestyle='--', label='Threshold Treinamento')
plt.xlabel("Threshold")
plt.ylabel("Valor")
plt.title("Métricas no Conjunto de Teste por Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("avaliacao_thresholds.png", dpi=300)
plt.show()

y_pred_final = [1 if p > THRESHOLD_TREINAMENTO else 0 for p in y_prob]

print("\n===== Relatório Final no Conjunto de Teste =====")
print(classification_report(y_true, y_pred_final, target_names=['Nevo (nv)', 'Melanoma (mel)']))

roc_auc = roc_auc_score(y_true, y_prob)
print(f"AUC da Curva ROC: {roc_auc:.4f}")

cm = confusion_matrix(y_true, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nevo", "Melanoma"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Matriz de Confusão - Conjunto de Teste")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8,5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.plot(val_recall_mel_history, label='Recall MEL')
plt.plot(val_f1_history, label='F1 Score')
plt.xlabel("Época")
plt.ylabel("Valor")
plt.title("Histórico de Treinamento")
plt.legend()
plt.tight_layout()
plt.savefig("historico_treinamento.png", dpi=300)
plt.show()

with open("relatorio_teste.txt", "w") as f:
    f.write("===== Relatório Final no Conjunto de Teste =====\n")
    f.write(classification_report(y_true, y_pred_final, target_names=['Nevo (nv)', 'Melanoma (mel)']))
    f.write(f"\nAUC da Curva ROC: {roc_auc:.4f}\n")


'''best_f1_mel = 0  # substitui best_recall
patience = 5
counter = 0

train_loss_history, val_loss_history = [], []
val_recall_mel_history, val_f1_history = [], []

for ep in range(EPOCHS):
    model.train()
    lh, acc = 0, 0
    for imgs, lbls in tqdm(train_dl, desc=f"Epoch {ep+1}"):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        opt.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, lbls)
        loss.backward()
        opt.step()
        lh += loss.item()
        preds = (torch.sigmoid(logits) > THRESHOLD_TREINAMENTO).float()
        acc += (preds == lbls).float().mean().item()

    train_loss = lh / len(train_dl)
    train_loss_history.append(train_loss)
    print(f"[Época {ep+1}] Erro de Treinamento: {train_loss:.4f} | Acurácia: {acc/len(train_dl):.4f}")

    # Validação
    model.eval()
    val_loss, val_acc = 0, 0
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            val_loss += loss.item()

            prob = torch.sigmoid(logits)
            pred = (prob > THRESHOLD_TREINAMENTO).float()

            y_true += lbls.tolist()
            y_pred += pred.tolist()
            y_prob += prob.tolist()

            val_acc += (pred == lbls).float().mean().item()

    val_loss /= len(val_dl)
    val_acc /= len(val_dl)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    val_loss_history.append(val_loss)
    val_recall_mel_history.append(rec)
    val_f1_history.append(f1)

    print(f"[Época {ep+1}] Erro de Validação: {val_loss:.4f} | Acurácia: {val_acc:.4f} | Recall MEL: {rec:.4f} | F1 MEL: {f1:.4f}")

    # Salvar modelo se F1 MEL for o melhor até agora
    if f1 > best_f1_mel:
        best_f1_mel = f1
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Parando early: paciência esgotada.")
            break'''


'''
best_recall = 0
patience = 5
counter = 0

train_loss_history, val_loss_history = [], []
val_recall_mel_history, val_f1_history = [], []

for ep in range(EPOCHS):
    model.train()
    lh, acc = 0, 0
    for imgs, lbls in tqdm(train_dl, desc=f"Epoch {ep+1}"):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        opt.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, lbls)
        loss.backward()
        opt.step()
        lh += loss.item()
        preds = (torch.sigmoid(logits) > THRESHOLD_TREINAMENTO).float()
        acc += (preds == lbls).float().mean().item()

    train_loss = lh / len(train_dl)
    train_loss_history.append(train_loss)
    print(f"[Época {ep+1}] Erro de Treinamento: {train_loss:.4f} | Acurácia: {acc/len(train_dl):.4f}")

    # Validação
    model.eval()
    val_loss, val_acc = 0, 0
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            val_loss += loss.item()

            prob = torch.sigmoid(logits)
            pred = (prob > THRESHOLD_TREINAMENTO).float()

            y_true += lbls.tolist()
            y_pred += pred.tolist()
            y_prob += prob.tolist()

            val_acc += (pred == lbls).float().mean().item()

    val_loss /= len(val_dl)
    val_acc /= len(val_dl)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    val_loss_history.append(val_loss)
    val_recall_mel_history.append(rec)
    val_f1_history.append(f1)

    print(f"[Época {ep+1}] Erro de Validação: {val_loss:.4f} | Acurácia: {val_acc:.4f} | Recall MEL: {rec:.4f} | F1: {f1:.4f}")

    # Salvar modelo se melhorou recall
    if rec > best_recall:
        best_recall = rec
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Parando early: paciência esgotada.")
            break'''