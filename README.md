# 🧠 Treinamento de Rede Neural para Classificação de Câncer de Pele (HAM10000)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Badge" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11 Badge" />
</p>

Este projeto implementa um programa para o treinamento de uma Rede Neural Artificial (RNA) utilizando a biblioteca **PyTorch**. O objetivo é classificar imagens do dataset **Skin Cancer MNIST: HAM10000**, auxiliando na detecção e classificação de lesões de pele.

---

## 📋 Pré-requisitos

Para rodar este projeto, você precisará ter o **Python 3.11** instalado em sua máquina e o dataset devidamente configurado.

### 🐍 1. Instalação do Python 3.11

O projeto requer especificamente a versão **Python 3.11**.

| Versão Necessária | Link para Download |
| :--- | :--- |
| **Python 3.11** | [Python Downloads (Página Oficial)](https://www.python.org/downloads/) |

> **Dica de Instalação:** Ao executar o instalador, **marque a caixa "Add Python to PATH"** para que você possa usar os comandos `python` e `pip` diretamente no terminal.

### 📥 2. Download e Organização do Dataset

O projeto utiliza o **Skin Cancer MNIST: HAM10000**.

1.  Acesse o link para download:
    [**https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download**](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download)
2.  Faça o download e descompacte o conteúdo.
3.  Crie uma pasta chamada **`data`** na **raiz** do projeto (no mesmo nível deste `README.md`).
4.  Mova **todas as imagens** e o arquivo **`HAM10000_metadata.csv`** para dentro da pasta **`data`**.

#### 📁 Estrutura de Pastas Esperada

```text
seu_projeto/
├── data/
│   ├── ISIC_0024306.jpg
│   ├── ... (todas as imagens)
│   └── HAM10000_metadata.csv 
├── requirements.txt
├── README.md
└── seu_script_principal.py

