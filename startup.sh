#!/bin/bash

# 1. Tenta inicializar o banco de dados usando o caminho do módulo correto
echo "Iniciando o banco de dados (db.create_all())..."

python << END
try:
    print("Tentando importar 'servidor.servidor'...")
    # Caminho corrigido:
    from servidor.servidor import app, db
    print("Importação bem-sucedida.")
    
    with app.app_context():
        print("Executando db.create_all()...")
        db.create_all()
    print("Banco de dados pronto.")

except Exception as e:
    print("--- !!! ERRO FATAL DURANTE A INICIALIZAÇÃO !!! ---")
    print(f"O ERRO REAL É: {e}")
    import traceback
    traceback.print_exc()
    print("-------------------------------------------------")
    exit(1) # Sai com erro
END

# Captura o status de saída do script python
STATUS=$?

# Se o script python falhou (status != 0), o container para.
if [ $STATUS -ne 0 ]; then
    echo "!!! Falha no script de inicialização. O deploy vai parar."
    exit $STATUS
fi

# 2. Se tudo deu certo, inicia o Gunicorn com o caminho do módulo correto
echo "Inicialização concluída. Iniciando Gunicorn..."
# Caminho corrigido:
gunicorn servidor.servidor:app --bind 0.0.0.0:$PORT