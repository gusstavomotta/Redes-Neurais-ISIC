
import torchvision.transforms as T, torchvision.models as models 
import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

#Define o tamanho do batch, taxa de aprendizado e número de épocas
#verifica se a GPU está disponível e define o dispositivo
DATA_DIR = 'HAM10000_images_part_1'
CSV_PATH = 'HAM10000_metadata.csv'
BATCH_SIZE, LR, EPOCHS = 16, 1e-4, 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Acessar o csv com o dados e filtra apenas nevos e melanomas
#cria a coluna label, onde 1 é melanoma e 0 é nevo
df = pd.read_csv(CSV_PATH)
df = df[df['dx'].isin(['mel','nv'])]
df['label'] = (df['dx'] == 'mel').astype(int)

#Separa os dados em treino, validação e teste
train, temp = train_test_split(df, stratify=df['label'], test_size=0.3, random_state=42)
val, test = train_test_split(temp, stratify=temp['label'], test_size=0.5, random_state=42)

# Cria um conjunto de treino mais balanceado usando oversampling
# Repete os melanomas para balancear com os nevos
# PODE GERAR ALGUM PROBLEMA USAR MUITO OVERSAMPLING
mel_train = train[train['label'] == 1]
train = pd.concat([train, mel_train, mel_train, mel_train, mel_train]).sample(frac=1, random_state=42).reset_index(drop=True)

# Contar quantos melanomas vieram para o treino
print("Distribuição de classes no treino (split tradicional):")
print(train['label'].value_counts())


#Define a classe SkinDataset para carregar as imagens
#Pega as imagens e retorna imagem com rótulo no getitem
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

#Normaliza os conjuntos de treino, validação e teste
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

#Cria os datasets e dataloaders para os conjuntos de treino, validação e teste
train_ds = SkinDataset(train, DATA_DIR, tf_train)
val_ds = SkinDataset(val, DATA_DIR, tf_test)
test_ds = SkinDataset(test, DATA_DIR, tf_test)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, BATCH_SIZE)
test_dl = DataLoader(test_ds, BATCH_SIZE)

#Cria o modelo ResNet50 pré-treinado
#Substitui a última camada totalmente conectada para saída binária
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

#Define a função de perda e o otimizador
#Usa BCEWithLogitsLoss com pesos para lidar com o desbalanceamento de classes
weights = torch.tensor([(len(train_ds)-train['label'].sum())/len(train_ds),
                        train['label'].sum()/len(train_ds)]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1:2])
opt = torch.optim.Adam(model.parameters(), LR)

best_val_loss = float('inf')
patience = 3
counter = 0
# realiza o loop de treinamento e validação 
# limita o threshold de predição para 0.35
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
        preds = (torch.sigmoid(logits)>0.35).float()
        acc += (preds==lbls).float().mean().item()
    print(f"[Época {ep+1}] Erro de Treinamento: {lh/len(train_dl):.4f} | Acurácia de Treinamento: {acc/len(train_dl):.4f}")

    model.eval()
    with torch.no_grad():
        val_loss, val_acc = 0,0
        for imgs, lbls in val_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            val_loss += loss.item()
            preds = (torch.sigmoid(logits)>0.35).float()
            val_acc += (preds==lbls).float().mean().item()
    print(f"[Época {ep+1}] Erro de Validação:   {val_loss/len(val_dl):.4f} | Acurácia de Validação:   {val_acc/len(val_dl):.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            break

print(f"Melhor erro de validação atingido: {best_val_loss / len(val_dl):.4f}")

# Avaliação do model usando o conjunto de teste
model.eval()
y_true, y_prob = [], []
with torch.no_grad():
    for imgs, lbls in test_dl:
        imgs = imgs.to(DEVICE)
        logits = model(imgs).squeeze(1)
        y_prob += torch.sigmoid(logits).cpu().tolist()
        y_true += lbls.tolist()

print("Avaliação com diferentes thresholds:")
for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    y_pred = [1 if p > thresh else 0 for p in y_prob]
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Threshold={thresh:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}")


print("\n===== Relatório Final no Conjunto de Teste =====")
print(classification_report(y_true, y_pred, target_names=['Nevo (nv)', 'Melanoma (mel)']))

roc_auc = roc_auc_score(y_true, y_prob)
print(f"AUC da Curva ROC: {roc_auc:.4f}")

# Gera a matriz de confusão
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nevo", "Melanoma"])
disp.plot(cmap='Blues', values_format='d')  # mostra números inteiros
plt.title("Matriz de Confusão - Conjunto de Teste")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.tight_layout()

plt.savefig("matriz_confusao.png", dpi=300, bbox_inches='tight')
plt.show()
