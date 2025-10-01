import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..')) 

if src_dir not in sys.path:
    sys.path.append(src_dir)


# Importa a função principal da análise sagital
from sagittal_analysis import show_sagittal_analysis

# --- Gerenciamento de Estado do Streamlit ---
if 'page' not in st.session_state:
    st.session_state.page = 'selection'

# Função para resetar o estado (usada internamente na lógica de navegação)
def navigate_to_selection():
    st.session_state.page = 'selection'
    st.rerun()

# Função da Nova Tela Inicial
def show_selection_page():
    """Exibe a tela inicial com as opções de análise."""
    st.title('Programa de Análise de Agachamento')
    st.write('Por favor, selecione o tipo de agachamento que você deseja analisar:')
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Agachamento Sagital Direito (SLS)', width='stretch'):
            st.session_state.page = 'sagittal'
            st.rerun()
            
    with col2:
        if st.button('Agachamento Frontal Direito (SLS)', width='stretch'):
            st.info("Funcionalidade de Agachamento Frontal ainda não implementada. Por favor, selecione a análise Sagital.")

# --- Lógica Principal da Aplicação ---
if __name__ == "__main__":
    if st.session_state.page == 'selection':
        show_selection_page()
    elif st.session_state.page == 'sagittal':
        show_sagittal_analysis()