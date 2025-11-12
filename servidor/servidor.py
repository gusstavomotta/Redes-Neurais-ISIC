import torch
from PIL import Image
import os
import io
import uuid
from flask import Flask, request, jsonify
import sys
from pathlib import Path
from flask_sqlalchemy import SQLAlchemy  # <-- NOVO

# --- Carregar Configs Corrigidas ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    MODEL_WEIGHTS_PATH, DATA_DIR, DEVICE_SERVIDOR, 
    THRESHOLD, CLASSES_MAP, CLASSES_UPLOAD
)
# Note: Não importamos mais CSV_PATH ou CSV_LOCK_PATH
# ------------------------------------

# Importe suas funções de modelo
# (Certifique-se que o 'treinamento' está no sys.path)
from treinamento.model_utils import build_model, get_data_transforms

DEVICE = DEVICE_SERVIDOR
data_transforms = get_data_transforms(use_augmentation=False)

print(f"Carregando modelo em {DEVICE}...")
model = build_model(DEVICE)
try:
    # MODEL_WEIGHTS_PATH agora aponta para o arquivo no seu repositório
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print("Modelo carregado. Servidor pronto.")
except FileNotFoundError:
    print(f"!!! ERRO FATAL: Não foi possível carregar o modelo de {MODEL_WEIGHTS_PATH} !!!")
    model = None 

app = Flask(__name__)

# --- Configuração do Banco de Dados (NOVO) ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo da tabela para salvar os metadados das contribuições
class SkinLesion(db.Model):
    __tablename__ = 'skin_lesions'
    id = db.Column(db.Integer, primary_key=True)
    lesion_id = db.Column(db.String(100))
    image_id = db.Column(db.String(100), unique=True, nullable=False)
    dx = db.Column(db.String(20)) # 'nv' ou 'mel'
    dx_type = db.Column(db.String(50))
    age = db.Column(db.Float, nullable=True)
    sex = db.Column(db.String(10), nullable=True)
    localization = db.Column(db.String(50), nullable=True)
# --- Fim da Configuração do DB ---


@app.route("/predict", methods=["POST"])
def predict():
    # Esta rota está perfeita, não precisa mudar nada
    if model is None:
        return jsonify({"erro": "Modelo não foi carregado com sucesso."}), 503
    # ... (resto do seu código de predição) ...
    pass # Removendo o código duplicado para economizar espaço


@app.route("/upload", methods=["POST"])
def upload():
    # Validação (não muda)
    if 'image' not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400
    if 'classification' not in request.form:
        return jsonify({"erro": "Nenhuma classificação enviada"}), 400

    label = request.form['classification']
    if label not in CLASSES_UPLOAD: 
        return jsonify({"erro": f"Classificação inválida. Use {CLASSES_UPLOAD}"}), 400

    try:
        file = request.files['image']
        unique_id = uuid.uuid4().hex[:8]
        image_id = f"CONTRIB_{unique_id}"
        lesion_id = f"LESION_{unique_id}"
        filename = f"{image_id}.jpg"
        
        # DATA_DIR agora aponta para o VOLTUME (ex: /data)
        save_path = os.path.join(DATA_DIR, filename) 

        # 1. Salvar a Imagem no Volume
        img = Image.open(file.stream).convert("RGB")
        img.save(save_path, "JPEG", quality=95)

        # 2. Salvar Metadados no Banco de Dados (Lógica NOVA)
        new_lesion_entry = SkinLesion(
            lesion_id=lesion_id,
            image_id=image_id,
            dx=label,
            dx_type='user_contribution',
            sex='unknown',
            localization='unknown'
        )
        db.session.add(new_lesion_entry)
        db.session.commit()

        return jsonify({"status": "sucesso", "mensagem": "Contribuição salva com sucesso!"}), 200

    except Exception as e:
        # Se algo der errado, desfaz a transação do DB
        db.session.rollback() 
        # E remove a imagem órfã do Volume
        if 'save_path' in locals() and os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"erro": f"Falha ao salvar contribuição: {str(e)}"}), 500


if __name__ == '__main__':
    # Garante que o diretório do volume exista
    os.makedirs(DATA_DIR, exist_ok=True) 
    
    # Cria a tabela no banco de dados se ela não existir
    with app.app_context():
        db.create_all() 
    
    # Em produção, o Railway usará Gunicorn. 
    # Isto é apenas para debug local.
    app.run(debug=True, host="0.0.0.0", port=5000)