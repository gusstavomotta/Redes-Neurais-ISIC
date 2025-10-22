import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights # Importamos ResNet50_Weights por clareza, mas usaremos 'None'
from torch.utils.mobile_optimizer import optimize_for_mobile
import os

MODEL_WEIGHTS_PATH = "resultados/best_model.pt" 
OUTPUT_DIR = "modelo_mobile_ptl" 
PTL_MODEL_PATH = os.path.join(OUTPUT_DIR, "model_mobile.ptl")

class SkinModel(nn.Module):
    def __init__(self):
        super(SkinModel, self).__init__()
        self.backbone = resnet50(weights=None) 
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

def build_model(device=torch.device('cpu')):
    model = SkinModel().to(device)
    return model

def run_export():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    DEVICE = torch.device('cpu') 
    model = build_model(DEVICE)

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"ERRO: Arquivo de pesos não encontrado em: {MODEL_WEIGHTS_PATH}")
        return

    try:

        state = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        
        if missing or unexpected:
            print(f"ATENÇÃO AO CARREGAR PESOS:")
            print(f"Camadas faltando: {missing}")
            print(f"Camadas inesperadas: {unexpected}")
        else:
            print("✅ Pesos carregados com sucesso no 'backbone' do modelo.")

    except Exception as e:
        print(f"ERRO ao carregar pesos: {e}")
        return
        
    model.eval() 
    DUMMY_INPUT = torch.randn(1, 3, 224, 224, device=DEVICE) 

    try:
        print("Iniciando o 'trace' do modelo...")
        traced_model = torch.jit.trace(model, DUMMY_INPUT)
        
        print("Otimizando o modelo para mobile...")
        optimized_model = optimize_for_mobile(traced_model)

        print("Salvando modelo .ptl...")
        optimized_model._save_for_lite_interpreter(PTL_MODEL_PATH)
        
        print(f"\nSUCESSO! Modelo PyTorch Mobile (.ptl) salvo em: {PTL_MODEL_PATH}")

    except Exception as e:
        print(f"ERRO FATAL na exportação para TorchScript: {e}")
        return

if __name__ == '__main__':
    run_export()