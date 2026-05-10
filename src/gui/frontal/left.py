import streamlit as st
import pandas as pd
import os

from classes.personal_ai.frontal_personal_ai import FrontalAI as PersonalAI
from classes.excel.squat_report_excel_writer.frontal_report_excel_writer import FrontalReportExcelWriter
from classes.excel.set_folders import SetFolders
from ultils.feedback_messages import feedback_messages

MODEL_PATH = 'models/pose_landmarker_full.task'


def show_frontal_left_analysis(): 
    """
    Função principal que monta a interface e gerencia o fluxo da análise frontal.
    """
        
    st.write("Marque as repetições que você quer salvar os dados no arquivo:")
    # Criando os checkboxes e armazenando o estado (True/False)
    c1 = st.checkbox("Salvar repetição 1")
    c2 = st.checkbox("Salvar repetição 2")
    c3 = st.checkbox("Salvar repetição 3")

    #c1 = True
    #c2 = True
    #c3 = True

    # Criando o vetor baseado no que foi marcado
    opcoes_marcadas = []

    if c1: opcoes_marcadas.append(1)
    if c2: opcoes_marcadas.append(2)
    if c3: opcoes_marcadas.append(3)    
    st.title('Análise Frontal Esquerdo - Agachamento')
    name_input = st.text_input('Nome da pessoa')
    
    uploaded_file_data = st.file_uploader('Envie o vídeo (Frontal Esquerdo)', type=['mp4', 'avi', 'mov'])
        
    st.write('### Parâmetros de Avaliação do Exercício')
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        descent_th = st.slider('Sensibilidade da Descida (Repetição)', 0.01, 0.10, 0.05, 0.005, format='%.3f', help="Percentual de movimento da orelha para baixo para iniciar a contagem da repetição.")
        hip_err_th = st.slider('Tolerância de Desvio - Quadril (Duração Permitida)', 1, 150, 1, 1, help="Número de instantes que o quadril pode estar desalinhado antes de ser considerado um erro na repetição.")
    with col_param2:
        ascent_return_th = st.slider('Tolerância de Retorno na Subida (Repetição)', 0.005, 0.05, 0.02, 0.005, format='%.3f', help="Percentual de proximidade da posição inicial da orelha para finalizar a contagem da repetição.")
        knee_valgus_th = st.slider('Tolerância de Desvio - Joelho (Valgo/Varo)', 1, 150, 5, 1, help="Número de instantes que o joelho pode estar em valgo ou varo antes de ser considerado um erro na repetição.")
        foot_pronation_th = st.slider('Tolerância de Desvio - Pé (Pronação)', 1, 150, 7 , 1, help="Número de instantes que o pé pode estar pronado antes de ser considerado um erro na repetição.")

    params = {
        'descent_threshold': descent_th,
        'ascent_return_threshold': ascent_return_th,
        'hip_error_threshold': hip_err_th,  
        'knee_valgus_error_threshold': knee_valgus_th,     
        'foot_pronation_error_threshold': foot_pronation_th 
    }
    
    # 2. Processamento e Exibição de Resultados
    if uploaded_file_data and name_input:
        set_folders = SetFolders(person_name=name_input, plane_folder_name="frontal", side="esquerdo")
        set_folders.create_folders()

        ai_instance = process_and_analyze_video(uploaded_file_data, name_input, params, opcoes_marcadas)
        
        display_overall_summary(ai_instance.squat_analyzer, name_input)
        
        if ai_instance.squat_analyzer.repetitions_detected > 0:
            display_detailed_charts(ai_instance.squat_analyzer)
            display_repetition_details_and_feedback(ai_instance.squat_analyzer)
            display_data_frames(ai_instance)
        else:
            display_no_repetitions_found_message()


def process_and_analyze_video(uploaded_file, name_input, params, opcoes_marcadas):
    """Salva o vídeo, inicializa a IA, processa e gera o relatório."""
    ext = os.path.splitext(uploaded_file.name)[1]
    temp_path = f'temp_frontal_dir{ext}' 

    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.info('Analisando vídeo...')

    ai = PersonalAI(
        file_name=temp_path, 
        name_pessoa=name_input, 
        model_path=MODEL_PATH,
        **params, side="left",
        options_marcadas=opcoes_marcadas
    )
    ai.process_video(True, True) 
    st.success('Análise concluída!')

    excel_writer = FrontalReportExcelWriter(name_input, ai.squat_analyzer, "esquerdo")
    excel_writer.generate_report()
    
    os.remove(temp_path)
    return ai


def display_overall_summary(ai_analyzer, name):
    st.markdown(f"""
    ---
    ## Resultados da Análise Frontal para: **{name}**
    ---
    """)
    st.write(f'### Resumo das Repetições Detectadas: {ai_analyzer.repetitions_detected}')

def display_detailed_charts(ai_analyzer):
    st.write('### Análise Detalhada de Desvios por Repetição (Contagem de Instantes)')
    st.markdown("""
    Abaixo, você encontrará um gráfico de barras para cada uma das repetições analisadas.
    Cada barra representa o número de *frames* em que o desvio foi detectado.
    """)

    # Usa os novos históricos de contagem para o gráfico
    for i in range(len(ai_analyzer.hip_error_history)):
        if ai_analyzer.hip_error_history[i] is not None: 
            st.write(f'#### Repetição {i+1}')
            
            rep_error_data = {
                'Parte do Corpo': ['Quadril (Inclinação)', 'Joelho (Valgo/Varo)', 'Pé (Pronação)'],
                'Contagem de Erros': [
                    ai_analyzer.hip_error_history[i], 
                    ai_analyzer.knee_valgus_error_history[i],   
                    ai_analyzer.foot_pronation_error_history[i],   
                ]
            }
            
            df_rep_errors = pd.DataFrame(rep_error_data).set_index('Parte do Corpo')
            
            st.bar_chart(df_rep_errors, width='stretch', height=300) 
            st.markdown("---")
        else:
            st.write(f'#### Repetição {i+1}: Não Detectada')
            st.info(f"Não há dados completos para a Repetição {i+1}.")
            st.markdown("---")


def display_repetition_details_and_feedback(ai_analyzer):
    st.write('### Detalhes por Repetição')
    for i in range(len(ai_analyzer.reps['hip'])):
        
        # Recupera o status booleano (0 ou 1)
        hip_rep_status = ai_analyzer.reps['hip'][i] 
        knee_rep_status = ai_analyzer.reps['knee_valgus'][i] 
        foot_rep_status = ai_analyzer.reps['foot_pronation'][i] 
        
        # Recupera a contagem de instantes (o que o usuário queria de volta)
        hip_count = ai_analyzer.hip_error_history[i]
        knee_valgus_count = ai_analyzer.knee_valgus_error_history[i]
        foot_pronation_count = ai_analyzer.foot_pronation_error_history[i]

        if hip_rep_status is not None and ai_analyzer.repetition_timestamps[i] is not None:
            st.markdown(f"#### Repetição {i+1} (Finalizada em {ai_analyzer.repetition_timestamps[i]:.2f} segundos)")
            
            hip_status = "DESVIO ❌" if hip_rep_status == 1 else "OK ✅" 
            knee_valgus_status = "DESVIO ❌" if knee_rep_status == 1 else "OK ✅" 
            foot_pronation_status = "DESVIO ❌" if foot_rep_status == 1 else "OK ✅" 

            # Display com a Contagem de Instantes
            st.markdown(f"- **Inclinação do Quadril:** {hip_status} **({hip_count} instantes)**") 
            st.markdown(f"- **Valgo/Varo de Joelho:** {knee_valgus_status} **({knee_valgus_count} instantes)**") 
            st.markdown(f"- **Pronação do Pé:** {foot_pronation_status} **({foot_pronation_count} instantes)**") 

            st.write("---")
            st.write("**Feedback para esta repetição:**")
            feedback_given = False
            
            # Feedback baseado no status (1)
            if hip_rep_status == 1:
                st.info(f"💡 Desvio no Quadril: **{feedback_messages.get('hip_error', 'Verifique a estabilidade lateral do quadril.')}**")
                feedback_given = True
            if knee_rep_status == 1:
                st.info(f"💡 Desvio no Joelho: **{feedback_messages.get('knee_valgus_error', 'Valgo ou Varo detectado. Fortaleça abdutores.')}**")
                feedback_given = True
            if foot_rep_status == 1:
                st.info(f"💡 Desvio no Pé: **{feedback_messages.get('foot_pronation_error', 'Pronação excessiva. Fortaleça a musculatura intrínseca do pé.')}**")
                feedback_given = True
            
            if not feedback_given:
                st.success("✅ **Ótima execução!** Continue assim.")
            st.write("---")
        else:
            st.markdown(f"#### Repetição {i+1}: Detalhes Indisponíveis")
            st.info(f"Detalhes para a Repetição {i+1} não estão disponíveis.")
            st.write("---")
            
def display_no_repetitions_found_message():
    st.write('Nenhuma repetição foi detectada com os parâmetros atuais. Por favor, verifique se o movimento de agachamento foi completo ou ajuste os parâmetros de sensibilidade.')

def display_data_frames(ai):
    st.write('### Detalhe da Análise Ponto a Ponto (Momentos de Desvio)')

    # DataFrames atualizados para o Plano Frontal
    dataframes_to_display = {
        "Desvios de Inclinação do Quadril": ai.hip_tilt_df,
        "Desvios de Valgo/Varo de Joelho": ai.knee_valgus_df,
        "Desvios de Pronação do Pé": ai.foot_pronation_df
    }

    for title, df in dataframes_to_display.items():
        if not df.empty:
            st.write(f'#### {title}')
            
            df_display = df.copy()         
            time_column_name = 'Tempo (ms)'
    
            df_display[time_column_name] = (df_display[time_column_name] / 1000).round(2)
            st.dataframe(df_display)