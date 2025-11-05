import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from PIL import Image
import os

MODEL_WEIGHTS_PATH = "resultados/best_model.pt" 
IMAGE_TO_TEST = "imagem_avaliacao/image.png" 
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
    logit_valor = output_logit.item()
    
    threshold = 0.5
    if prob_valor > threshold:
        classificacao = "MELANOMA (Positivo)"
    else:
        classificacao = "NEVO (Não-Melanoma / Negativo)"
        
    prob_percentual = prob_valor * 100

    print("RESULTADO DA INFERÊNCIA")
    print(f"Imagem Testada:   {IMAGE_TO_TEST}")
    print("-" * 40)
    print(f"Logit (Saída Bruta):      {logit_valor:.4f}")
    print(f"Probabilidade (Melanoma): {prob_percentual:.2f}% (Valor: {prob_valor:.6f})")
    print("\nClassificação (com Threshold 0.5):")
    print(f">>> {classificacao}")
    print("========================================")

if __name__ == '__main__':
    test_single_image()