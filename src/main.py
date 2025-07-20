# main.py (Finalizado para a modularização das classes)
import streamlit as st
import pandas as pd # Usado para exibir DataFrames no Streamlit
import os # Usado para manusear o arquivo temporário

# --- Importações das suas classes e utilitários separados ---
from classes.personal_ai import PersonalAI # Importa a classe PersonalAI do novo arquivo
from ultils.feedback_messages import feedback_messages # Importa o dicionário de feedback do novo arquivo

# O 'model_path' deve ser o nome exato do seu arquivo de modelo do MediaPipe.
model_path = 'models/pose_landmarker_full.task'

# --- REMOVA A CLASSE PersonalAI AQUI ---
# Remova todo o bloco da classe PersonalAI que estava aqui antes.

# --- REMOVA O DICIONÁRIO feedback_messages AQUI ---
# Remova todo o bloco do dicionário feedback_messages que estava aqui antes.


# --- Streamlit UI ---
st.title('Análise Sagital Direita - Refatoração Finalizada: Modularização')
name_input = st.text_input('Nome da pessoa')
uploaded = st.file_uploader('Envie o vídeo (Sagital Direita)', type=['mp4', 'avi', 'mov'])

st.write('### Configurações de Visualização')
col_display1, col_display2 = st.columns(2)
with col_display1:
    draw = st.checkbox('Desenhar Marcadores Corporais', value=True)
with col_display2:
    display = st.checkbox('Mostrar Vídeo Durante Análise', value=False)

st.write('### Parâmetros de Avaliação do Exercício')
col_param1, col_param2 = st.columns(2)
with col_param1:
    descent_th = st.slider('Sensibilidade da Descida (Repetição)', 0.01, 0.10, 0.05, 0.005, format='%.3f', help="Percentual de movimento da orelha para baixo para iniciar a contagem da repetição.")
    trunk_err_th = st.slider('Tolerância de Desvio - Tronco (Duração Permitida)', 1, 20, 5, 1, help="Número de instantes que o tronco pode estar desalinhado antes de ser considerado um erro na repetição.")
    head_err_th = st.slider('Tolerância de Desvio - Cabeça (Duração Permitida)', 1, 20, 5, 1, help="Número de instantes que a cabeça pode estar desalinhada antes de ser considerado um erro na repetição.")
with col_param2:
    ascent_return_th = st.slider('Tolerância de Retorno na Subida (Repetição)', 0.005, 0.05, 0.02, 0.005, format='%.3f', help="Percentual de proximidade da posição inicial da orelha para finalizar a contagem da repetição.")
    knee_err_th = st.slider('Tolerância de Desvio - Joelho (Duração Permitida)', 1, 20, 5, 1, help="Número de instantes que o joelho pode estar desalinhado antes de ser considerado um erro na repetição.")
    foot_err_th = st.slider('Tolerância de Desvio - Calcanhar (Duração Permitida)', 1, 20, 5, 1, help="Número de instantes que o calcanhar pode estar levantado antes de ser considerado um erro na repetição.")

if uploaded and name_input:
    ext = os.path.splitext(uploaded.name)[1]
    temp_path = f'temp_sag_dir{ext}'

    with open(temp_path, 'wb') as f:
        f.write(uploaded.getbuffer())
    st.info('Analisando vídeo...')
    
    ai = PersonalAI(
        temp_path, name_input, model_path,
        descent_threshold=descent_th,
        ascent_return_threshold=ascent_return_th,
        trunk_error_threshold=trunk_err_th,
        knee_error_threshold=knee_err_th,
        head_error_threshold=head_err_th,
        foot_error_threshold=foot_err_th
    )
    ai.process_video(draw, display)
    st.success('Análise concluída!')
    
    st.markdown(f"""
    ---
    ## Resultados da Análise para: **{name_input}**
    ---
    """)

    st.write(f'### Resumo das Repetições Detectadas: {ai.squat_analyzer.repetitions_detected}')
    
    if ai.squat_analyzer.repetitions_detected > 0: 
        st.write('### Contagem de Desvios por Repetição') 
        error_data = {
            'Repetição': [f'Rep {i+1}' for i in range(len(ai.squat_analyzer.trunk_error_history))],
            'Tronco': ai.squat_analyzer.trunk_error_history,
            'Joelho': ai.squat_analyzer.knee_error_history,
            'Cabeça': ai.squat_analyzer.head_error_history,
            'Calcanhar': ai.squat_analyzer.foot_error_history
        }
        df_errors = pd.DataFrame(error_data).set_index('Repetição')
        
        st.bar_chart(df_errors, use_container_width=True, height=400)
        st.markdown("""
        **Eixo Y:** Duração do Desvio (em Instantes).
        *Ao passar o mouse sobre as barras, a indicação 'value' refere-se a essa duração em instantes.*
        """)

        st.write('### Detalhes por Repetição')
        for i in range(len(ai.squat_analyzer.reps['trunk'])):
            st.markdown(f"#### Repetição {i+1} (Finalizada em {ai.squat_analyzer.repetition_timestamps[i]:.2f} segundos)")
            
            trunk_status = "DESVIO ❌" if ai.squat_analyzer.reps['trunk'][i] == 1 else "OK ✅" 
            knee_status = "DESVIO ❌" if ai.squat_analyzer.reps['knee'][i] == 1 else "OK ✅" 
            head_status = "DESVIO ❌" if ai.squat_analyzer.reps['head'][i] == 1 else "OK ✅" 
            heel_status = "DESVIO ❌" if ai.squat_analyzer.reps['heel'][i] == 1 else "OK ✅" 

            st.markdown(f"- **Tronco:** {trunk_status} ({ai.squat_analyzer.trunk_error_history[i]} instantes)") 
            st.markdown(f"- **Joelho:** {knee_status} ({ai.squat_analyzer.knee_error_history[i]} instantes)") 
            st.markdown(f"- **Cabeça:** {head_status} ({ai.squat_analyzer.head_error_history[i]} instantes)") 
            st.markdown(f"- **Calcanhar:** {heel_status} ({ai.squat_analyzer.foot_error_history[i]} instantes)") 

            st.write("---") 
            st.write("**Feedback para esta repetição:**")
            feedback_given = False
            if ai.squat_analyzer.reps['trunk'][i] == 1:
                st.info(f"💡 {feedback_messages['trunk_error']}")
                feedback_given = True
            if ai.squat_analyzer.reps['knee'][i] == 1:
                st.info(f"💡 {feedback_messages['knee_error']}")
                feedback_given = True
            if ai.squat_analyzer.reps['head'][i] == 1:
                st.info(f"💡 {feedback_messages['head_error']}")
                feedback_given = True
            if ai.squat_analyzer.reps['heel'][i] == 1:
                st.info(f"💡 {feedback_messages['heel_error']}")
                feedback_given = True
            
            if not feedback_given:
                st.success("✅ **Ótima execução!** Continue assim.")
            st.write("---") 

    else:
        st.write('Nenhuma repetição foi detectada com os parâmetros atuais. Por favor, verifique se o movimento de agachamento foi completo ou ajuste os parâmetros de sensibilidade.')

    st.write('### Detalhe da Análise Ponto a Ponto (Momentos de Desvio)') 
    st.dataframe(ai.head_df)
    st.dataframe(ai.trunk_df)
    st.dataframe(ai.heel_df)
    st.dataframe(ai.knee_df)

    st.write(f'### Tempo total de análise do vídeo: {ai.execution_time:.2f} segundos')