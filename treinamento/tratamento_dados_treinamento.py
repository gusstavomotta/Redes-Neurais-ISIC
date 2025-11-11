# C:\dev\Redes-Neurais-ISIC\treinamento\tratamento_dados_treinamento.py
# (VERSÃO LIMPA)

import os
import zipfile
import pandas as pd
import shutil
from tqdm import tqdm
import time
import sys
from pathlib import Path

# --- Hack para adicionar a raiz do projeto ao sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT))
# --- Fim do Hack ---

import config 

def run_setup():
    start_time = time.time()
    
    if not os.path.exists(config.ZIP_FILE_NAME):
        print(f"ERRO: Arquivo '{config.ZIP_FILE_NAME}' não encontrado.")
        return
    if os.path.exists(config.DATA_DIR):
        # Mantive este AVISO pois ele impede a execução
        print(f"AVISO: A pasta '{config.DATA_DIR}' já existe. Setup não será executado.")
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)

    try:
        with zipfile.ZipFile(config.ZIP_FILE_NAME, 'r') as zip_ref:
            
            with zip_ref.open(config.CSV_IN_ZIP) as csv_file:
                df = pd.read_csv(csv_file)
            
            clean_df = df[df['dx'].isin(config.CLASSES_UPLOAD)].copy() 
            clean_df.to_csv(config.CSV_PATH, index=False)
            
            image_ids_to_keep = set(clean_df['image_id'] + '.jpg')
            all_files_in_zip = zip_ref.namelist()
            
            files_to_extract = [
                fp for fp in all_files_in_zip
                if (fp.startswith(config.IMG_FOLDER_1_IN_ZIP) or 
                    fp.startswith(config.IMG_FOLDER_2_IN_ZIP))
                and os.path.basename(fp) in image_ids_to_keep
            ]

            # A barra do tqdm ainda vai aparecer, o que é bom.
            for file_path in tqdm(files_to_extract, desc="Extraindo imagens"):
                image_name = os.path.basename(file_path)
                dest_path = os.path.join(config.DATA_DIR, image_name)
                with zip_ref.open(file_path) as source_file, \
                     open(dest_path, 'wb') as dest_file:
                    shutil.copyfileobj(source_file, dest_file)

    except Exception as e:
        print(f"\nERRO DURANTE O SETUP: {e}")
        if os.path.exists(config.DATA_DIR):
            shutil.rmtree(config.DATA_DIR)
        return

    end_time = time.time()
    print("\n" + "="*40)
    print(f"SETUP CONCLUÍDO (em {end_time - start_time:.2f}s)")
    print(f"Pasta final '{config.DATA_DIR}' criada.")
    print(f"{len(clean_df)} imagens unificadas.")
    print("="*40)

if __name__ == '__main__':
    run_setup()