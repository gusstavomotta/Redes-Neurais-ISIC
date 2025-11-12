# Em config.py
import torch
from pathlib import Path
import os

# __file__ é /app/config.py
# .parent é /app (a raiz)
BASE_DIR = Path(__file__).resolve().parent 

# O modelo está no seu repositório Git
RESULTS_DIR = BASE_DIR / "treinamento" / "resultados"
MODEL_WEIGHTS_PATH = RESULTS_DIR / "best_model.pt"

# O Volume para onde vão os novos uploads
VOLUME_MOUNT_PATH = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/data')
DATA_DIR = Path(VOLUME_MOUNT_PATH) 

# Configurações do Servidor e Modelo
DEVICE_TREINO_EVAL = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
DEVICE_SERVIDOR = torch.device('cpu') 
THRESHOLD = 0.5

# Rótulos
CLASSES_MAP = {0: "Nevo (Negativo)", 1: "MELANOMA (Positivo)"}
CLASSES_NOMES = [CLASSES_MAP[0], CLASSES_MAP[1]]
CLASSES_UPLOAD = ['nv', 'mel']