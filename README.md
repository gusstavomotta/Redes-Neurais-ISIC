🧠 Classificação de Lesões de Pele com Redes Neurais (HAM10000)

Este projeto utiliza Redes Neurais Convolucionais (CNNs) para classificar imagens de lesões de pele como benignas ou malignas, com base no dataset HAM10000.
O modelo foi desenvolvido em PyTorch, com suporte a GPU (CUDA) para acelerar o treinamento.

📋 Requisitos

Python 3.11: https://www.python.org/downloads/release/python-3110/

⚙️ Instalação
1️⃣ Clonar o repositório

Você pode clonar o projeto de duas formas:
gh repo clone gusstavomotta/Redes-Neurais-ISIC

ou

git clone https://github.com/gusstavomotta/Redes-Neurais-ISIC.git
cd Redes-Neurais-ISIC

2️⃣ Criar o ambiente virtual
python -m venv venv

Ativar o ambiente:

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

3️⃣ Instalar as dependências

Atualize o pip:
pip install --upgrade pip

🧩 Instalar PyTorch com suporte a CUDA (recomendado se possuir GPU NVIDIA)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

🔸 Caso não possua GPU, instale a versão CPU:
pip install torch torchvision torchaudio

📦 Instalar demais dependências do projeto

Para instalar as depenências, execute:
pip install -r requirements.txt

📂 Dataset
O dataset utilizado é o HAM10000, disponível no Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download

Após o download:

Crie uma pasta chamada data na raiz do projeto.
Coloque dentro dela as imagens das pastas HAM10000_images_part_1 e HAM10000_images_part_2.
Coloque também o arquivo HAM10000_metadata.csv dentro dessa mesma pasta data.

A estrutura deve ficar assim:

Redes-Neurais-ISIC/
├── data/
│   ├── Todas as imagens
│   └── HAM10000_metadata.csv
├── src/
├── models/
├── venv/
└── ...
