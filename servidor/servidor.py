import torch
from PIL import Image
import os
import io
import csv
import uuid
from filelock import FileLock
from flask import Flask, request, jsonify
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT))

import config as cfg
from treinamento.model_utils import build_model, get_data_transforms

data_transforms = get_data_transforms(use_augmentation=False)

print(f"Carregando modelo em {cfg.DEVICE_SERVIDOR}...")
model = build_model(cfg.DEVICE_SERVIDOR)
try:
    model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH, map_location=cfg.DEVICE_SERVIDOR))
    model.eval()
    print("Modelo carregado. Servidor pronto.")
except FileNotFoundError:
    print(f"!!! ERRO FATAL: Não foi possível carregar o modelo de {cfg.MODEL_WEIGHTS_PATH} !!!")
    model = None 

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"erro": "Modelo não foi carregado com sucesso."}), 503

    if 'image' not in request.files:
        return jsonify({"erro": "Nenhum arquivo de imagem enviado"}), 400

    file = request.files['image']

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = data_transforms(img).unsqueeze(0).to(cfg.DEVICE_SERVIDOR)

        with torch.no_grad():
            output_logit = model(input_tensor)
            probabilidade = torch.sigmoid(output_logit).item()

        pred_label = 1 if probabilidade > cfg.THRESHOLD else 0
        classificacao = cfg.CLASSES_MAP[pred_label]
        prob_percentual = f"{probabilidade * 100:.2f}%"

        return jsonify({
            "probabilidade": probabilidade,
            "porcentagem": prob_percentual,
            "classificacao": classificacao,
            "logit": output_logit.item()
        })

    except Exception as e:
        return jsonify({"erro": f"Erro durante a predição: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400
    if 'classification' not in request.form:
        return jsonify({"erro": "Nenhuma classificação enviada"}), 400

    label = request.form['classification']
    if label not in cfg.CLASSES_UPLOAD: 
        return jsonify({"erro": f"Classificação inválida. Use {cfg.CLASSES_UPLOAD}"}), 400

    save_path = None
    try:
        file = request.files['image']
        unique_id = uuid.uuid4().hex[:8]
        image_id = f"CONTRIB_{unique_id}"
        lesion_id = f"LESION_{unique_id}"
        filename = f"{image_id}.jpg"
        save_path = os.path.join(cfg.DATA_DIR, filename) 

        img = Image.open(file.stream).convert("RGB")
        img.save(save_path, "JPEG", quality=95)

        new_row = [lesion_id, image_id, label, 'user_contribution', '', 'unknown', 'unknown']

        lock = FileLock(cfg.CSV_LOCK_PATH, timeout=10) 
        with lock:
            with open(cfg.CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(new_row)

        return jsonify({"status": "sucesso", "mensagem": "Imagem salva com sucesso!"}), 200

    except Exception as e:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"erro": f"Falha ao salvar contribuição: {str(e)}"}), 500


if __name__ == '__main__':
    os.makedirs(cfg.DATA_DIR, exist_ok=True) 
    app.run(debug=True, host="0.0.0.0", port=5000)