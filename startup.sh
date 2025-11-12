#!/bin/bash

# Este comando é seguro, ele só cria a tabela se ela não existir.
echo "Iniciando o banco de dados (db.create_all())..."
python -c "from servidor import app, db; with app.app_context(): db.create_all()"

# Depois que o banco estiver pronto, inicie o servidor web.
echo "Iniciando Gunicorn..."
gunicorn servidor:app --bind 0.0.0.0:$PORT