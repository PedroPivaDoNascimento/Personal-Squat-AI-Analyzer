# Arquivo: main.py

import os
import sys

# O diretório 'src' deve ser adicionado ao sys.path para importações absolutas.
# O diretório atual é o 'ProgramaAgachamento/'.
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_streamlit_app():
    """Chama o Streamlit para rodar o arquivo de fluxo principal."""
    print("Iniciando a aplicação Streamlit...")
    
    # O arquivo principal está agora em src/ui/app_flow.py (ou src/gui/app_flow.py, dependendo da sua preferência)
    # Vou manter o caminho original por enquanto: src/gui/main_app.py
    os.system("streamlit run src/gui/main_app.py") 

if __name__ == "__main__":
    run_streamlit_app()