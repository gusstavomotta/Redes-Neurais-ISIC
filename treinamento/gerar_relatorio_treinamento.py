import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm

from config import (
    MODEL_WEIGHTS_PATH, THRESHOLDS_PLOT_PATH, 
    REPORT_PATH, CONFUSION_MATRIX_PATH, CLASSES_NOMES, THRESHOLD
)

def generate_final_report(config, model, test_dl):
    
    threshold = config.get('THRESHOLD', THRESHOLD) 
    device = config.get('DEVICE', torch.device('cpu'))

    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERRO: Não foi possível carregar o modelo de {MODEL_WEIGHTS_PATH} para gerar o relatório.")
        return
            
    model.to(device)
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_dl, desc="Gerando Relatório"):
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            y_prob.extend(torch.sigmoid(logits).cpu().tolist())
            y_true.extend(lbls.tolist())
    
    thresholds_to_test = np.arange(0.25, 0.76, 0.05) 
    precision_list, recall_list, f1_list = [], [], []

    for thresh in thresholds_to_test:
        y_pred = [1 if p > thresh else 0 for p in y_prob]
        precision_list.append(precision_score(y_true, y_pred, zero_division=0))
        recall_list.append(recall_score(y_true, y_pred, zero_division=0))
        f1_list.append(f1_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_to_test, precision_list, marker='o', label='Precision')
    plt.plot(thresholds_to_test, recall_list, marker='o', label='Recall')
    plt.plot(thresholds_to_test, f1_list, marker='o', label='F1 Score')
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold Configurado ({threshold})")
    plt.xlabel("Threshold")
    plt.ylabel("Valor da Métrica")
    plt.title("Métricas no Conjunto de Teste por Threshold")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(THRESHOLDS_PLOT_PATH, dpi=300)

    y_pred_final = [1 if p > threshold else 0 for p in y_prob]
    
    target_names = [nome.split(" ")[0] for nome in CLASSES_NOMES]
    report = classification_report(y_true, y_pred_final, target_names=target_names)
    roc_auc = roc_auc_score(y_true, y_prob)

    with open(REPORT_PATH, "w") as f:
        f.write(f"===== Relatório Final no Conjunto de Teste (Threshold={threshold}) =====\n")
        f.write(report)
        f.write(f"\nAUC da Curva ROC: {roc_auc:.4f}\n")
        
    cm = confusion_matrix(y_true, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Matriz de Confusão (Threshold={threshold})")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)