import torch
from PIL import Image
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT))

from config import (
    MODEL_WEIGHTS_PATH, IMAGE_TO_TEST, DEVICE_TREINO_EVAL, 
    THRESHOLD, CLASSES_MAP
)
from treinamento.model_utils import build_model, get_data_transforms

DEVICE = DEVICE_TREINO_EVAL
data_transforms = get_data_transforms(use_augmentation=False)

def test_single_image():
    if not os.path.exists(IMAGE_TO_TEST):
        print(f"ERRO: Imagem não encontrada em: {IMAGE_TO_TEST}")
        return

    model = build_model(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERRO: Pesos do modelo não encontrados em: {MODEL_WEIGHTS_PATH}")
        return
    
    model.eval()

    try:
        img = Image.open(IMAGE_TO_TEST).convert("RGB")
        input_tensor = data_transforms(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"ERRO ao processar a imagem: {e}")
        return

    with torch.no_grad(): 
        output_logit = model(input_tensor)
        probabilidade = torch.sigmoid(output_logit)

    prob_valor = probabilidade.item()
    
    pred_label = 1 if prob_valor > THRESHOLD else 0
    classificacao = CLASSES_MAP[pred_label]
    prob_percentual = prob_valor * 100

    print(f"\n=== RESULTADO PARA: {os.path.basename(IMAGE_TO_TEST)} ===")
    print(f"Classificação: {classificacao}")
    print(f"Probabilidade de Melanoma: {prob_percentual:.2f}%")
    print("========================================\n")

if __name__ == '__main__':
    test_single_image()