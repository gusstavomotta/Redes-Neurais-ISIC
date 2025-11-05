import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from PIL import Image
import os
import io 
import json

from flask import Flask, request, jsonify

print("Iniciando o servidor...")
MODEL_WEIGHTS_PATH = "resultados/best_model.pt" 
DEVICE = torch.device('cpu') 
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

model = build_model(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print(f"Modelo {MODEL_WEIGHTS_PATH} carregado. Servidor pronto!")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"erro": "Nenhum arquivo de imagem enviado"}), 400
    
    file = request.files['image']
    
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        input_tensor = data_transforms(img)
        input_tensor = input_tensor.unsqueeze(0) 
        input_tensor = input_tensor.to(DEVICE)
        print("Imagem recebida e processada com sucesso.")

    except Exception as e:
        return jsonify({"erro": f"Erro ao processar a imagem: {e}"}), 400

    try:
        with torch.no_grad(): 
            output_logit = model(input_tensor)
            probabilidade = torch.sigmoid(output_logit)

        prob_valor = probabilidade.item()
        
        print(f"Inferência bem-sucedida, retornando: {prob_valor:.4f}")

        return jsonify({
            "probabilidade": prob_valor,
            "logit": output_logit.item()
        })
        
    except Exception as e:
        return jsonify({"erro": f"Erro durante a inferência: {e}"}), 500

if __name__ == '__main__':

    app.run(debug=True, host="0.0.0.0", port=5000)