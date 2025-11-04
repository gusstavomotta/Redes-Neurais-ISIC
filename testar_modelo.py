import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from PIL import Image
import os

MODEL_WEIGHTS_PATH = "resultados/best_model.pt" 
IMAGE_TO_TEST = "foto_real_teste.jpg" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

def build_model(device):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) 
    model.fc = nn.Linear(model.fc.in_features, 1) #
    model = model.to(device)
    return model


data_transforms = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(),         
    T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])
print("Transformações de teste carregadas.")

def test_single_image():
    if not os.path.exists(IMAGE_TO_TEST):
        print(f"ERRO: Imagem não encontrada em: {IMAGE_TO_TEST}")
        print("Por favor, coloque sua foto real com este nome na pasta do script.")
        return

    model = build_model(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    
    model.eval()
    print(f"Modelo {MODEL_WEIGHTS_PATH} carregado e em modo de avaliação.")

    try:
        img = Image.open(IMAGE_TO_TEST).convert("RGB")
        input_tensor = data_transforms(img)
        
        input_tensor = input_tensor.unsqueeze(0) 
        input_tensor = input_tensor.to(DEVICE)
        print(f"Imagem {IMAGE_TO_TEST} carregada e processada.")

    except Exception as e:
        print(f"ERRO ao carregar ou processar a imagem: {e}")
        return

    with torch.no_grad(): 
        output_logit = model(input_tensor)
 
        probabilidade = torch.sigmoid(output_logit)

    prob_valor = probabilidade.item()
    
    print("   RESULTADO DA INFERÊNCIA (PYTHON)")
    print("="*30)
    print(f"Saída (Logit): {output_logit.item():.4f}")
    print(f"Probabilidade: {prob_valor:.4f}")

if __name__ == '__main__':
    test_single_image()