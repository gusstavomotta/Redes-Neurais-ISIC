# Em config.py
import torch
from pathlib import Path
import os

# --- CAMINHOS DO PROJETO (NO REPOSITÓRIO GIT) ---
# Define o BASE_DIR como a pasta onde o 'config.py' está
BASE_DIR = Path(__file__).resolve().parent 

# O modelo está no seu repositório Git
RESULTS_DIR = BASE_DIR / "treinamento" / "resultados"
MODEL_WEIGHTS_PATH = RESULTS_DIR / "best_model.pt"

# --- CAMINHOS DO VOLUME (PARA NOVOS UPLOADS) ---
# Pega o caminho do volume do Railway, ou usa '/data' como padrão
VOLUME_MOUNT_PATH = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/data')
# Este é o único lugar onde salvaremos NOVOS dados
DATA_DIR = Path(VOLUME_MOUNT_PATH) 

# --- CONFIGS DO SERVIDOR E MODELO ---
DEVICE_TREINO_EVAL = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
DEVICE_SERVIDOR = torch.device('cpu') 
THRESHOLD = 0.5

# Rótulos
CLASSES_MAP = {0: "Nevo (Negativo)", 1: "MELANOMA (Positivo)"}
CLASSES_NOMES = [CLASSES_MAP[0], CLASSES_MAP[1]]
CLASSES_UPLOAD = ['nv', 'mel']

# (Remova todas as outras configs como BATCH_SIZE, LR, etc. 
# O servidor não precisa delas, apenas o script de treino local.)