import torch
from pathlib import Path
import os # <-- Importante

# --- CAMINHOS BASE (PARA O SERVIDOR RAILWAY) ---
# BASE_DIR é a raiz do projeto (onde o config.py está)
BASE_DIR = Path(__file__).resolve().parent 

# --- CAMINHO DO MODELO (DENTRO DO REPOSITÓRIO GIT) ---
# O servidor carrega o modelo do seu repositório.
RESULTS_DIR = BASE_DIR / "treinamento" / "resultados"
MODEL_WEIGHTS_PATH = RESULTS_DIR / "best_model.pt"

# --- CAMINHO DO VOLUME (PARA ONDE OS UPLOADS VÃO) ---
# O servidor salva as novas imagens de upload aqui.
VOLUME_MOUNT_PATH = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/data')
DATA_DIR = Path(VOLUME_MOUNT_PATH) 

# --- CONFIGURAÇÕES DO SERVIDOR ---
DEVICE_SERVIDOR = torch.device('cpu') 
THRESHOLD = 0.5

# --- CONFIGURAÇÕES DE IMAGEM (Requeridas pelo model_utils.py) ---
# O servidor precisa disso para transformar as imagens de upload.
IMG_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# --- RÓTULOS ---
CLASSES_MAP = {0: "Nevo (Negativo)", 1: "MELANOMA (Positivo)"}
CLASSES_NOMES = [CLASSES_MAP[0], CLASSES_MAP[1]]
CLASSES_UPLOAD = ['nv', 'mel']

# --- NOTA SOBRE OUTRAS VARIÁVEIS ---
# Todas as outras variáveis do seu config original (BATCH_SIZE, LR, EPOCHS,
# CSV_PATH, DEVICE_TREINO_EVAL, etc.) não são necessárias 
# para o servidor de produção e podem ser removidas com segurança 
# deste arquivo.
#
# Elas são usadas apenas pelo seu script 'treinar.py' local.