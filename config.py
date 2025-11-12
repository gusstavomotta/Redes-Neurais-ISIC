# Em config.py
import torch
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent 
RESULTS_DIR = BASE_DIR / "treinamento" / "resultados"
MODEL_WEIGHTS_PATH = RESULTS_DIR / "best_model.pt"

VOLUME_MOUNT_PATH = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/data')
DATA_DIR = Path(VOLUME_MOUNT_PATH) 

DEVICE_TREINO_EVAL = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
DEVICE_SERVIDOR = torch.device('cpu') 
THRESHOLD = 0.5

CLASSES_MAP = {0: "Nevo (Negativo)", 1: "MELANOMA (Positivo)"}
CLASSES_NOMES = [CLASSES_MAP[0], CLASSES_MAP[1]]
CLASSES_UPLOAD = ['nv', 'mel']

# --- Configurações do Modelo e Imagem (ADICIONE ISTO) ---
IMG_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]