Como rodar o projeto: 

1. Instalar o Python 3.11
Você precisa especificamente da versão 3.11 do Python. Baixe e instale a partir deste link: https://www.python.org/downloads/

2. Baixar o Dataset (HAM10000)
O projeto usa o dataset "Skin Cancer MNIST: HAM10000". Baixe os arquivos neste link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download

3. Preparar a Pasta de Dados
Na pasta principal do seu projeto (onde está seu código), crie uma nova pasta chamada "data".
Pegue todos os arquivos que você baixou do Kaggle (todas as imagens .jpg e o arquivo "HAM10000_metadata.csv") e mova-os para dentro desta pasta "data" que você acabou de criar.

4. Configurar o Ambiente Virtual
   
Abra o seu terminal ou prompt de comando na pasta raiz do seu projeto.
Digite o comando para criar um ambiente virtual: python3.11 -m venv venv

Agora, ative esse ambiente:
Se estiver no Windows: .\venv\Scripts\activate
Se estiver no Linux ou macOS: source venv/bin/activate

(Você saberá que funcionou se vir "(venv)" no início da linha do seu terminal).

5. Instalar as Dependências (Bibliotecas)
Com o ambiente ainda ativo, rode o comando para instalar tudo o que está no arquivo requirements.txt: pip install -r requirements.txt

6. ATENÇÃO: Se for usar a GPU (Placa de Vídeo)
Se você quiser treinar usando sua GPU (NVIDIA/CUDA), você precisa de uma versão especial do PyTorch.
Vá ao site oficial do PyTorch: https://pytorch.org/get-started/locally/

Na página, selecione as opções corretas para o seu sistema (Ex: Stable, Windows, Pip, Python, a sua versão do CUDA).
O site vai gerar um comando de instalação (algo como "pip install torch...").

Copie esse comando e rode-o no seu terminal (com o ambiente "venv" ativo). Isso vai instalar a versão correta do PyTorch para sua GPU.
