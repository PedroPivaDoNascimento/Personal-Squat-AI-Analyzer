import os
import sys

# O diretório 'src' deve ser adicionado ao sys.path para importações absolutas
# O diretório atual é o 'ProgramaAgachamento/'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_streamlit_app():
    """Chama o Streamlit para rodar o arquivo de fluxo principal."""
    print("Iniciando a aplicação Streamlit...")
    
    os.system("streamlit run src/gui/main_app.py") 

if __name__ == "__main__":
    run_streamlit_app()