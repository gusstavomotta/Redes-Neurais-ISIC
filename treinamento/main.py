import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
import random
import numpy as np
import torch

import config as cfg
from setup import run_setup
from data_utils import prepare_data
from model_utils import build_model
from train import run_training
from gerar_relatorio_treinamento import generate_final_report


def setup_seeds():
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    setup_seeds()
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    run_setup()
    train_dl, val_dl, test_dl, pos_weight = prepare_data()
    model = build_model(cfg.DEVICE_TREINO_EVAL)
    run_training(model, train_dl, val_dl, pos_weight)
    generate_final_report(model, test_dl)

    print("\nProcesso concluido.")


if __name__ == '__main__':
    main()