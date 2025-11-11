import os
import zipfile
import pandas as pd
import shutil
from tqdm import tqdm
import time

ZIP_FILE_NAME = 'archive.zip'

FINAL_DIR = 'treinamento/data'
FINAL_CSV_PATH = os.path.join(FINAL_DIR, 'HAM10000_metadata.csv')

CSV_IN_ZIP = 'HAM10000_metadata.csv'
IMG_FOLDER_1_IN_ZIP = 'HAM10000_images_part_1/'
IMG_FOLDER_2_IN_ZIP = 'HAM10000_images_part_2/'

def run_setup():
    start_time = time.time()
    
    if not os.path.exists(ZIP_FILE_NAME):
        print(f"ERRO: Arquivo '{ZIP_FILE_NAME}' não encontrado na raiz.")
        return

    if os.path.exists(FINAL_DIR):
        print(f"AVISO: A pasta '{FINAL_DIR}' já existe. O setup já foi feito.")
        print("Para recomeçar, apague a pasta 'treinamento/data' manualmente.")
        return

    print(f"Iniciando a extração e limpeza do '{ZIP_FILE_NAME}'...")
    os.makedirs(FINAL_DIR, exist_ok=True)

    try:
        with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
            
            print(f"Extraindo e filtrando {CSV_IN_ZIP}...")
            with zip_ref.open(CSV_IN_ZIP) as csv_file:
                df = pd.read_csv(csv_file)
            
            clean_df = df[df['dx'].isin(['mel', 'nv'])].copy()
            
            clean_df.to_csv(FINAL_CSV_PATH, index=False)
            print(f"CSV limpo salvo em: {FINAL_CSV_PATH}")

            print(f"Extraindo e unificando {len(clean_df)} imagens para '{FINAL_DIR}'...")
            
            image_ids_to_keep = set(clean_df['image_id'] + '.jpg')
            
            all_files_in_zip = zip_ref.namelist()
            
            files_to_extract = []
            for file_path in all_files_in_zip:
                if (file_path.startswith(IMG_FOLDER_1_IN_ZIP) or file_path.startswith(IMG_FOLDER_2_IN_ZIP)):
                    if os.path.basename(file_path) in image_ids_to_keep:
                        files_to_extract.append(file_path)

            for file_path in tqdm(files_to_extract, desc="Extraindo imagens"):
                image_name = os.path.basename(file_path)
                dest_path = os.path.join(FINAL_DIR, image_name)

                with zip_ref.open(file_path) as source_file:
                    with open(dest_path, 'wb') as dest_file:
                        shutil.copyfileobj(source_file, dest_file)

    except Exception as e:
        print(f"\nERRO DURANTE O PROCESSO: {e}")
        print("Limpando a pasta de destino...")
        if os.path.exists(FINAL_DIR):
            shutil.rmtree(FINAL_DIR)
        return

    end_time = time.time()
    print("\n" + "="*40)
    print(f"SETUP CONCLUÍDO (em {end_time - start_time:.2f}s)")
    print(f"Pasta final '{FINAL_DIR}' criada com sucesso.")
    print(f"{len(clean_df)} imagens ('mel' e 'nv') unificadas.")
    print(f"CSV limpo salvo como '{FINAL_CSV_PATH}'.")
    print("="*40)

if __name__ == '__main__':
    run_setup()