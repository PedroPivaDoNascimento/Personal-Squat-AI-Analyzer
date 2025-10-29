import streamlit as st
import os
import sys
import warnings
import google.protobuf.symbol_database as symbol_database 

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf.symbol_database"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..')) 

if src_dir not in sys.path:
    sys.path.append(src_dir)


from frontal_analysis import show_frontal_analysis
from saggital.right import show_sagittal_right_analysis
from saggital.left import show_sagittal_left_analysis

if 'page' not in st.session_state:
    st.session_state.page = 'selection'

def navigate_to_selection():
    st.session_state.page = 'selection'
    st.rerun()

def show_selection_page():
    """Exibe a tela inicial com as opções de análise."""
    st.title('Programa de Análise de Agachamento')
    st.write('Por favor, selecione o tipo de agachamento que você deseja analisar:')
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button('Agachamento Sagital Direito (SLS)', width='stretch'):
            st.session_state.page = 'sagittal_right'
            st.rerun()
            
    with col2:
        if st.button('Agachamento Frontal Direito (SLS)', width='stretch'):
            st.session_state.page = 'frontal'
            st.rerun()
    with col3:
        if st.button('Agachamento Sagital Esquerdo (SLS)', width='stretch'):
            st.session_state.page = 'sagittal_left'
            st.rerun()

if __name__ == "__main__":
    if st.session_state.page == 'selection':
        show_selection_page()
    elif st.session_state.page == 'sagittal_right':
        show_sagittal_right_analysis()
    elif st.session_state.page == 'frontal':
        show_frontal_analysis()
    elif st.session_state.page == 'sagittal_left':
        show_sagittal_left_analysis()