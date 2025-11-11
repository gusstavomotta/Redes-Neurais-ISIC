import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from PIL import Image
import os
import io
import csv
import uuid
from filelock import FileLock
from flask import Flask, request, jsonify

MODEL_WEIGHTS_PATH = "treinamento/resultados/best_model.pt"
DATA_DIR = "treinamento/data"
CSV_PATH = "treinamento/data/HAM10000_metadata.csv"
CSV_LOCK_PATH = "treinamento/data/HAM10000_metadata.csv.lock"

DEVICE = torch.device('cpu')

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

model = build_model(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
model.eval()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"erro": "Nenhum arquivo de imagem enviado"}), 400

    file = request.files['image']

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = data_transforms(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output_logit = model(input_tensor)
            probabilidade = torch.sigmoid(output_logit).item()

        classificacao = "Melanoma (Positivo)" if probabilidade > 0.5 else "Nevo (Não-Melanoma)"
        prob_percentual = f"{probabilidade * 100:.2f}%"

        return jsonify({
            "probabilidade": probabilidade,
            "porcentagem": prob_percentual,
            "classificacao": classificacao,
            "logit": output_logit.item()
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400
    if 'classification' not in request.form:
        return jsonify({"erro": "Nenhuma classificação enviada"}), 400

    label = request.form['classification']
    if label not in ['mel', 'nv']:
        return jsonify({"erro": "Classificação inválida. Use 'mel' ou 'nv'"}), 400

    try:
        file = request.files['image']

        unique_id = uuid.uuid4().hex[:8]
        image_id = f"CONTRIB_{unique_id}"
        lesion_id = f"LESION_{unique_id}"

        filename = f"{image_id}.jpg"
        save_path = os.path.join(DATA_DIR, filename)

        img = Image.open(file.stream).convert("RGB")
        img.save(save_path, "JPEG", quality=95)

        new_row = [lesion_id, image_id, label, 'user_contribution', '', 'unknown', 'unknown']

        lock = FileLock(CSV_LOCK_PATH, timeout=10)
        with lock:
            with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(new_row)

        return jsonify({
            "status": "sucesso",
            "mensagem": f"Imagem salva com sucesso!"
        }), 200

    except Exception as e:
        if 'save_path' in locals() and os.path.exists(save_path):
                os.remove(save_path)
        return jsonify({"erro": f"Falha ao salvar contribuição: {str(e)}"}), 500

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)