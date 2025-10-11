import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..')) 

if src_dir not in sys.path:
    sys.path.append(src_dir)


from sagittal_analysis import show_sagittal_analysis
from frontal_analysis import show_frontal_analysis

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

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Agachamento Sagital Direito (SLS)', width='stretch'):
            st.session_state.page = 'sagittal'
            st.rerun()
            
    with col2:
        if st.button('Agachamento Frontal Direito (SLS)', width='stretch'):
            st.session_state.page = 'frontal'
            st.rerun()

if __name__ == "__main__":
    if st.session_state.page == 'selection':
        show_selection_page()
    elif st.session_state.page == 'sagittal':
        show_sagittal_analysis()
    elif st.session_state.page == 'frontal':
        show_frontal_analysis()