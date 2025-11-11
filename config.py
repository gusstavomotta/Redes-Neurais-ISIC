import torch
from pathlib import Path

SEED = 42
THRESHOLD = 0.5
DEVICE_TREINO_EVAL = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
DEVICE_SERVIDOR = torch.device('cpu') 

BASE_DIR = Path(__file__).resolve().parent 

# Dados de Treinamento
DATA_DIR = BASE_DIR / "treinamento" / "data"
CSV_PATH = DATA_DIR / "HAM10000_metadata.csv"
CSV_LOCK_PATH = DATA_DIR / "HAM10000_metadata.csv.lock"
ZIP_FILE_NAME = BASE_DIR / "archive.zip"
CSV_IN_ZIP = 'HAM10000_metadata.csv'
IMG_FOLDER_1_IN_ZIP = 'HAM10000_images_part_1/'
IMG_FOLDER_2_IN_ZIP = 'HAM10000_images_part_2/'

# Resultados do Treinamento
RESULTS_DIR = BASE_DIR / "treinamento" / "resultados"
MODEL_WEIGHTS_PATH = RESULTS_DIR / "best_model.pt"
REPORT_PATH = RESULTS_DIR / "relatorio_teste.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "matriz_confusao.png"
THRESHOLDS_PLOT_PATH = RESULTS_DIR / "avaliacao_thresholds.png"

# Avaliação
IMAGE_TO_TEST = BASE_DIR / "avaliacao" / "image.png" 

# Hiperparâmetros de Treinamento 
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 25
NUM_WORKERS = 8
PATIENCE = 5

# Configurações do Modelo e Imagem
IMG_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Rótulos
CLASSES_MAP = {0: "Nevo (Negativo)", 1: "MELANOMA (Positivo)"}
CLASSES_NOMES = [CLASSES_MAP[0], CLASSES_MAP[1]]
CLASSES_UPLOAD = ['nv', 'mel']