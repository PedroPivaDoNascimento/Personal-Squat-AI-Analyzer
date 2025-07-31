import streamlit as st
import pandas as pd
import os
import cv2

from classes.personal_ai import PersonalAI
from ultils.feedback_messages import feedback_messages
from classes.squat_report_excel_writer import SquatReportExcelWriter

MODEL_PATH = 'models/pose_landmarker_full.task'

def setup_app_ui(): 
    """
    Configura a interface do usuário do Streamlit, incluindo título,
    campos de entrada e sliders de parâmetros.
    Retorna os valores dos parâmetros e o arquivo de vídeo enviado.
    """

    st.title('Análise Sagital Direita - Agachamento')
    name_input = st.text_input('Nome da pessoa')
    uploaded_file = st.file_uploader('Envie o vídeo (Sagital Direita)', type=['mp4', 'avi', 'mov'])

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

    params = {
        'descent_threshold': descent_th,
        'ascent_return_threshold': ascent_return_th,
        'trunk_error_threshold': trunk_err_th,
        'knee_error_threshold': knee_err_th,
        'head_error_threshold': head_err_th,
        'foot_error_threshold': foot_err_th
    }
    return name_input, uploaded_file, params

def process_and_analyze_video(uploaded_file, name_input, params):
    """
    Salva o vídeo temporariamente, inicializa a IA e processa o vídeo.
    Retorna a instância do PersonalAI após a análise.
    """
    # Extrai a extensão do arquivo original
    ext = os.path.splitext(uploaded_file.name)[1]
    # Cria um nome de arquivo temporário
    temp_path = f'temp_sag_dir{ext}'

    # Salva o conteúdo do arquivo enviado em disco
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.info('Analisando vídeo...')

    # --- Cálculo da duração total do vídeo em segundos ---
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_video_duration_seconds = frame_count / fps if fps > 0 else 0
    cap.release()
    # --- FIM DO CÁLCULO ---

    # Inicializa a classe PersonalAI com os parâmetros do usuário
    ai = PersonalAI(
        temp_path, name_input, MODEL_PATH,
        **params # Desempacota o dicionário de parâmetros
    )
    # Processa o vídeo. draw=True e display=True são para visualização durante o processo.
    ai.process_video(True, True) 
    st.success('Análise concluída!')

    excel_writer = SquatReportExcelWriter(name_input, ai.squat_analyzer)
    excel_writer.generate_report(total_video_duration_seconds)     
    # Limpa o arquivo temporário após o processamento
    os.remove(temp_path)
    return ai

def display_overall_summary(ai_analyzer, name):
    """
    Exibe um resumo geral das repetições detectadas.
    """
    st.markdown(f"""
    ---
    ## Resultados da Análise para: **{name}**
    ---
    """)
    st.write(f'### Resumo das Repetições Detectadas: {ai_analyzer.repetitions_detected}')

def display_detailed_charts(ai_analyzer):
    """
    Gera e exibe gráficos de barras individuais para cada repetição detectada,
    mostrando a contagem de desvios por parte do corpo.
    """
    st.write('### Análise Detalhada de Desvios por Repetição')
    st.markdown("""
    Abaixo, você encontrará um gráfico de barras para cada uma das repetições analisadas.
    Cada gráfico mostra a quantidade de desvios para diferentes partes do corpo.
    """)

    for i in range(len(ai_analyzer.trunk_error_history)):
        if ai_analyzer.trunk_error_history[i] is not None:
            st.write(f'#### Repetição {i+1}') # Título para o gráfico da repetição atual
            
            # Prepara os dados para o gráfico de barras desta repetição específica.
            rep_error_data = {
                'Parte do Corpo': ['Tronco', 'Joelho', 'Cabeça', 'Calcanhar'],
                'Contagem de Erros': [
                    ai_analyzer.trunk_error_history[i],  # Erros de tronco para esta repetição
                    ai_analyzer.knee_error_history[i],   # Erros de joelho para esta repetição
                    ai_analyzer.head_error_history[i],   # Erros de cabeça para esta repetição
                    ai_analyzer.foot_error_history[i]    # Erros de calcanhar para esta repetição
                ]
            }
            
            # Cria um DataFrame Pandas para esta única repetição, definindo 'Parte do Corpo' como índice.
            df_rep_errors = pd.DataFrame(rep_error_data).set_index('Parte do Corpo')
            
            # Exibe o gráfico de barras para a repetição atual.
            st.bar_chart(df_rep_errors, use_container_width=True, height=300)
            st.markdown("---") # Adiciona um separador visual entre os gráficos
        else:
            # Se o slot da repetição for None (não detectada/completa), exibe uma mensagem.
            st.write(f'#### Repetição {i+1}: Não Detectada')
            st.info(f"Não há dados completos para a Repetição {i+1}. O agachamento pode não ter sido concluído ou detectado.")
            st.markdown("---") # Adiciona um separador visual

def display_repetition_details_and_feedback(ai_analyzer):
    """
    Exibe detalhes textuais para cada repetição, incluindo status OK/DESVIO
    e mensagens de feedback específicas.
    """
    st.write('### Detalhes por Repetição')
    # Itera sobre o número de repetições registradas (inclusive as preenchidas com None)
    for i in range(len(ai_analyzer.reps['trunk'])):
        # Verifica se a repetição atual tem dados reais ou é um slot None
        if ai_analyzer.reps['trunk'][i] is not None and ai_analyzer.repetition_timestamps[i] is not None:
            st.markdown(f"#### Repetição {i+1} (Finalizada em {ai_analyzer.repetition_timestamps[i]:.2f} segundos)")
            
            # Determina o status de OK/DESVIO para cada parte do corpo
            trunk_status = "DESVIO ❌" if ai_analyzer.reps['trunk'][i] == 1 else "OK ✅" 
            knee_status = "DESVIO ❌" if ai_analyzer.reps['knee'][i] == 1 else "OK ✅" 
            head_status = "DESVIO ❌" if ai_analyzer.reps['head'][i] == 1 else "OK ✅" 
            heel_status = "DESVIO ❌" if ai_analyzer.reps['heel'][i] == 1 else "OK ✅" 

            # Exibe o status e a contagem de instantes de desvio
            st.markdown(f"- **Tronco:** {trunk_status} ({ai_analyzer.trunk_error_history[i]} instantes)") 
            st.markdown(f"- **Joelho:** {knee_status} ({ai_analyzer.knee_error_history[i]} instantes)") 
            st.markdown(f"- **Cabeça:** {head_status} ({ai_analyzer.head_error_history[i]} instantes)") 
            st.markdown(f"- **Calcanhar:** {heel_status} ({ai_analyzer.foot_error_history[i]} instantes)") 

            st.write("---") # Separador visual
            st.write("**Feedback para esta repetição:**")
            feedback_given = False
            # Exibe mensagens de feedback específicas se houver desvio
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
            
            # Se nenhum desvio foi detectado, exibe mensagem de sucesso
            if not feedback_given:
                st.success("✅ **Ótima execução!** Continue assim.")
            st.write("---") # Separador visual
        else:
            # Mensagem para slots de repetição que são None (não detectados)
            st.markdown(f"#### Repetição {i+1}: Detalhes Indisponíveis")
            st.info(f"Detalhes para a Repetição {i+1} não estão disponíveis, pois ela não foi detectada ou concluída.")
            st.write("---") # Separador visual

def display_no_repetitions_found_message():
    """
    Exibe uma mensagem quando nenhuma repetição completa é detectada.
    """
    st.write('Nenhuma repetição foi detectada com os parâmetros atuais. Por favor, verifique se o movimento de agachamento foi completo ou ajuste os parâmetros de sensibilidade.')

def display_data_frames(ai):
    """
    Exibe DataFrames detalhados de desvios ponto a ponto,
    convertendo a coluna de tempo para segundos.
    Assume que os DataFrames (ai.head_df, ai.trunk_df, etc.)
    possuem uma coluna de tempo que precisa ser convertida.
    """
    st.write('### Detalhe da Análise Ponto a Ponto (Momentos de Desvio)')

    # Lista dos DataFrames a serem exibidos e os seus nomes para o título
    dataframes_to_display = {
        "Desvios da Cabeça": ai.head_df,
        "Desvios do Tronco": ai.trunk_df,
        "Desvios do Calcanhar": ai.heel_df,
        "Desvios do Joelho": ai.knee_df
    }

    # Itera sobre cada DataFrame para processar e exibir
    for title, df in dataframes_to_display.items():
        if not df.empty: # Verifica se o DataFrame não está vazio
            st.write(f'#### {title}')
            
            # Cria uma cópia do DataFrame para evitar modificar o original
            df_display = df.copy()         
            time_column_name = 'Tempo (ms)'
    
            # Converte para segundos e arredonda para 2 casas decimais para melhor legibilidade
            df_display[time_column_name] = (df_display[time_column_name] / 1000).round(2)
            df_display.rename(columns={time_column_name: 'Tempo (s)'}, inplace=True)
    
            st.dataframe(df_display, use_container_width=True)
        else:
            st.write(f'#### {title}')
            st.info(f"Nenhum desvio registado para {title.lower()}.")
        st.markdown("---") # Separador visual entre os DataFrames

if __name__ == "__main__":
    name_input, uploaded_file, params = setup_app_ui()

    #Processa o vídeo se um arquivo for enviado e um nome for fornecido
    if uploaded_file and name_input:
        ai_instance = process_and_analyze_video(uploaded_file, name_input, params)
        
        # Exibir o resumo geral
        display_overall_summary(ai_instance.squat_analyzer, name_input)
        
        # Exibir gráficos detalhados e feedback se houver repetições
        if ai_instance.squat_analyzer.repetitions_detected > 0:
            display_detailed_charts(ai_instance.squat_analyzer)
            display_repetition_details_and_feedback(ai_instance.squat_analyzer)
            display_data_frames(ai_instance)
        else:
            display_no_repetitions_found_message()

