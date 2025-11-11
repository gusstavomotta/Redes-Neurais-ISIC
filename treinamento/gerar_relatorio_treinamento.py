
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm

def generate_final_report(config, model, test_dl):
    model.load_state_dict(torch.load("treinamento/resultados/best_model.pt"))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_dl, desc="Teste Final"):
            imgs = imgs.to(config['DEVICE'])
            logits = model(imgs).squeeze(1)
            y_prob.extend(torch.sigmoid(logits).cpu().tolist())
            y_true.extend(lbls.tolist())
    
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
    plt.savefig("treinamento/resultados/avaliacao_thresholds.png", dpi=300)

    y_pred_final = [1 if p > config['THRESHOLD'] else 0 for p in y_prob]
    
    report = classification_report(y_true, y_pred_final, target_names=['Nevo (nv)', 'Melanoma (mel)'])
    roc_auc = roc_auc_score(y_true, y_prob)

    with open("treinamento/resultados/relatorio_teste.txt", "w") as f:
        f.write(f"===== Relatório Final no Conjunto de Teste (Threshold={config['THRESHOLD']}) =====\n")
        f.write(report)
        f.write(f"\nAUC da Curva ROC: {roc_auc:.4f}\n")
        
    cm = confusion_matrix(y_true, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nevo", "Melanoma"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Matriz de Confusão (Threshold={config['THRESHOLD']})")
    plt.tight_layout()
    plt.savefig("treinamento/resultados/matriz_confusao.png", dpi=300)