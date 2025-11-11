import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from PIL import Image
import os

MODEL_WEIGHTS_PATH = "treinamento/resultados/best_model.pt" 
IMAGE_TO_TEST = "treinamento/imagem_avaliacao/image.png" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(device):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) 
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    return model

data_transforms = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(),         
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def test_single_image():
    if not os.path.exists(IMAGE_TO_TEST):
        print(f"ERRO: Imagem não encontrada em: {IMAGE_TO_TEST}")
        return

    model = build_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
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
    threshold = 0.5
    
    classificacao = "MELANOMA (Positivo)" if prob_valor > threshold else "NEVO (Negativo)"
    prob_percentual = prob_valor * 100

    print(f"\n=== RESULTADO PARA: {os.path.basename(IMAGE_TO_TEST)} ===")
    print(f"Classificação: {classificacao}")
    print(f"Probabilidade de Melanoma: {prob_percentual:.2f}%")
    print("========================================\n")

if __name__ == '__main__':
    test_single_image()