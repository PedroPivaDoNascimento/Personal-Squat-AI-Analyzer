import streamlit as st
import pandas as pd
import os

from classes.personal_ai.saggital_personal_ai import SagittalAI as PersonalAI
from ultils.feedback_messages import feedback_messages
from classes.excel.squat_report_excel_writer.sagittal_report_excel_writer import SagittalReportExcelWriter
from classes.excel.set_folders import SetFolders


MODEL_PATH = 'models/pose_landmarker_full.task'

def show_sagittal_right_analysis(): 
    """
    Função principal que monta a interface e gerencia o fluxo da análise sagital.
    """
        
    st.title('Análise Sagital Direito - Agachamento')
    name_input = st.text_input('Nome da pessoa')
    user_height_cm = st.number_input("Sua Altura em centímetros", min_value=100, max_value=250, value=170)
    
    uploaded_file_data = st.file_uploader('Envie o vídeo (Sagital Direita)', type=['mp4', 'avi', 'mov'])
    
    st.write('### Parâmetros de Avaliação do Exercício')
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        descent_th = st.slider('Sensibilidade da Descida (Repetição)', 0.01, 0.10, 0.05, 0.005, format='%.3f', help="Percentual de movimento da orelha para baixo para iniciar a contagem da repetição.")
        trunk_err_th = st.slider('Tolerância de Desvio - Tronco (Duração Permitida)', 1, 150, 23, 1, help="Número de instantes que o tronco pode estar desalinhado antes de ser considerado um erro na repetição.")
        head_err_th = st.slider('Tolerância de Desvio - Cabeça (Duração Permitida)', 1, 150, 2, 1, help="Número de instantes que a cabeça pode estar desalinhada antes de ser considerado um erro na repetição.")
    with col_param2:
        ascent_return_th = st.slider('Tolerância de Retorno na Subida (Repetição)', 0.005, 0.05, 0.02, 0.005, format='%.3f', help="Percentual de proximidade da posição inicial da orelha para finalizar a contagem da repetição.")
        knee_err_th = st.slider('Tolerância de Desvio - Joelho (Duração Permitida)', 1, 150, 6, 1, help="Número de instantes que o joelho pode estar desalinhado antes de ser considerado um erro na repetição.")
        foot_err_th = st.slider('Tolerância de Desvio - Calcanhar (Duração Permitida)', 1, 150, 8, 1, help="Número de instantes que o calcanhar pode estar levantado antes de ser considerado um erro na repetição.")

    params = {
        'descent_threshold': descent_th,
        'ascent_return_threshold': ascent_return_th,
        'trunk_error_threshold': trunk_err_th,
        'knee_error_threshold': knee_err_th,
        'head_error_threshold': head_err_th,
        'foot_error_threshold': foot_err_th
    }
    
    if uploaded_file_data and name_input and user_height_cm:
        set_folders = SetFolders(person_name=name_input, plane_folder_name="sagital", side="direito")
        set_folders.create_folders()

        ai_instance = process_and_analyze_video(uploaded_file_data, name_input, user_height_cm, params)
        
        display_overall_summary(ai_instance.squat_analyzer, name_input)
        
        if ai_instance.squat_analyzer.repetitions_detected > 0:
            display_detailed_charts(ai_instance.squat_analyzer)
            display_repetition_details_and_feedback(ai_instance.squat_analyzer)
            display_data_frames(ai_instance)
        else:
            display_no_repetitions_found_message()


def process_and_analyze_video(uploaded_file, name_input, user_height_cm, params):
    """Salva o vídeo, inicializa a IA, processa e gera o relatório."""
    ext = os.path.splitext(uploaded_file.name)[1]
    temp_path = f'temp_sag_dir{ext}'

    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.info('Analisando vídeo...')

    ai = PersonalAI(
        temp_path, name_input, "right", user_height_cm, MODEL_PATH,
        **params 
    )
    ai.process_video(True, True) 
    st.success('Análise concluída!')

    excel_writer = SagittalReportExcelWriter(name_input, ai.squat_analyzer, "direito")
    excel_writer.generate_report()     
    os.remove(temp_path)
    return ai


def display_overall_summary(ai_analyzer, name):
    st.markdown(f"""
    ---
    ## Resultados da Análise para: **{name}**
    ---
    """)
    st.write(f'### Resumo das Repetições Detectadas: {ai_analyzer.repetitions_detected}')

def display_detailed_charts(ai_analyzer):
    st.write('### Análise Detalhada de Desvios por Repetição')
    st.markdown("""
    Abaixo, você encontrará um gráfico de barras para cada uma das repetições analisadas.
    Cada gráfico mostra a quantidade de desvios para diferentes partes do corpo.
    """)

    for i in range(len(ai_analyzer.trunk_error_history)):
        if ai_analyzer.trunk_error_history[i] is not None:
            st.write(f'#### Repetição {i+1}')
            
            rep_error_data = {
                'Parte do Corpo': ['Tronco', 'Joelho', 'Cabeça', 'Calcanhar'],
                'Contagem de Erros': [
                    ai_analyzer.trunk_error_history[i], 
                    ai_analyzer.knee_error_history[i],   
                    ai_analyzer.head_error_history[i],   
                    ai_analyzer.foot_error_history[i]    
                ]
            }
            
            df_rep_errors = pd.DataFrame(rep_error_data).set_index('Parte do Corpo')
            
            st.bar_chart(df_rep_errors, width='stretch', height=300) 
            st.markdown("---")
        else:
            st.write(f'#### Repetição {i+1}: Não Detectada')
            st.info(f"Não há dados completos para a Repetição {i+1}. O agachamento pode não ter sido concluído ou detectado.")
            st.markdown("---")

def display_repetition_details_and_feedback(ai_analyzer):
    st.write('### Detalhes por Repetição')
    for i in range(len(ai_analyzer.reps['trunk'])):
        if ai_analyzer.reps['trunk'][i] is not None and ai_analyzer.repetition_timestamps[i] is not None:
            st.markdown(f"#### Repetição {i+1} (Finalizada em {ai_analyzer.repetition_timestamps[i]:.2f} segundos)")
            
            trunk_status = "DESVIO ❌" if ai_analyzer.reps['trunk'][i] == 1 else "OK ✅" 
            knee_status = "DESVIO ❌" if ai_analyzer.reps['knee'][i] == 1 else "OK ✅" 
            head_status = "DESVIO ❌" if ai_analyzer.reps['head'][i] == 1 else "OK ✅" 
            heel_status = "DESVIO ❌" if ai_analyzer.reps['heel'][i] == 1 else "OK ✅" 

            st.markdown(f"- **Tronco:** {trunk_status} ({ai_analyzer.trunk_error_history[i]} instantes)") 
            st.markdown(f"- **Joelho:** {knee_status} ({ai_analyzer.knee_error_history[i]} instantes)") 
            st.markdown(f"- **Cabeça:** {head_status} ({ai_analyzer.head_error_history[i]} instantes)") 
            st.markdown(f"- **Calcanhar:** {heel_status} ({ai_analyzer.foot_error_history[i]} instantes)") 

            st.write("---")
            st.write("**Feedback para esta repetição:**")
            feedback_given = False
            if ai_analyzer.reps['trunk'][i] == 1:
                st.info(f"💡 {feedback_messages['trunk_error']}")
                feedback_given = True
            if ai_analyzer.reps['knee'][i] == 1:
                st.info(f"💡 {feedback_messages['knee_error']}")
                feedback_given = True
            if ai_analyzer.reps['head'][i] == 1:
                st.info(f"💡 {feedback_messages['head_error']}")
                feedback_given = True
            if ai_analyzer.reps['heel'][i] == 1:
                st.info(f"💡 {feedback_messages['heel_error']}")
                feedback_given = True
            
            if not feedback_given:
                st.success("✅ **Ótima execução!** Continue assim.")
            st.write("---")
        else:
            st.markdown(f"#### Repetição {i+1}: Detalhes Indisponíveis")
            st.info(f"Detalhes para a Repetição {i+1} não estão disponíveis, pois ela não foi detectada ou concluída.")
            st.write("---")

def display_no_repetitions_found_message():
    st.write('Nenhuma repetição foi detectada com os parâmetros atuais. Por favor, verifique se o movimento de agachamento foi completo ou ajuste os parâmetros de sensibilidade.')

def display_data_frames(ai):
    st.write('### Detalhe da Análise Ponto a Ponto (Momentos de Desvio)')

    dataframes_to_display = {
        "Desvios da Cabeça": ai.head_df,
        "Desvios do Tronco": ai.trunk_df,
        "Desvios do Calcanhar": ai.heel_df,
        "Desvios do Joelho": ai.knee_df,
        "Pontos de Interseção do Tronco": ai.squat_analyzer.trunk_intersections_df
    }

    for title, df in dataframes_to_display.items():
        if not df.empty:
            st.write(f'#### {title}')
            
            df_display = df.copy()         
            time_column_name = 'Tempo (ms)'
    
            df_display[time_column_name] = (df_display[time_column_name] / 1000).round(2)
            df_display.rename(columns={time_column_name: 'Tempo (s)'}, inplace=True)
    
            st.dataframe(df_display, width='stretch')
        else:
            st.write(f'#### {title}')
            st.info(f"Nenhum desvio registado para {title.lower()}.")
        st.markdown("---")