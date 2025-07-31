import pandas as pd
import os
import streamlit as st # Para usar st.success e st.error na interface do Streamlit

class SquatReportExcelWriter:
    def __init__(self, person_name, squat_analyzer_instance):
        """
        Inicializa o gerador de relatórios Excel.

        Args:
            person_name (str): O nome da pessoa para o relatório. Este será o nome do arquivo Excel.
            squat_analyzer_instance (SquatRepetitionAnalyzer): A instância do analisador de agachamento,
                                                               contendo todos os dados de análise (históricos de erros, DataFrames de desvio, etc.).
        """
        self.person_name = person_name
        self.analyzer = squat_analyzer_instance 
        
    def _calculate_errors_per_second(self, error_history_list, total_video_duration_seconds):
        """
        Calcula o número total de erros por segundo para uma parte do corpo.

        Args:
            error_history_list (list): Uma lista contendo o número de erros para cada repetição
                                       (ex: self.analyzer.trunk_error_history). Pode conter None.
            total_video_duration_seconds (float): A duração total do vídeo em segundos.

        Returns:
            float: O número de erros por segundo, arredondado para zero casas decimais.
                   Retorna 0.0 se a duração do vídeo for zero ou se a lista de erros estiver vazia/contiver apenas None.
        """
        # Soma todos os valores na lista de histórico, tratando None como 0.
        total_errors_in_video = sum(val for val in error_history_list if val is not None)

        if total_video_duration_seconds > 0:
            errors_per_second = round(total_errors_in_video / total_video_duration_seconds, 0)
        else:
            errors_per_second = 0.0
        
        return errors_per_second

    def _fill_repetition_data(self, df_report):
        """
        Preenche as colunas 'Repetição 1', 'Repetição 2', 'Repetição 3' e 'Resultado'
        no DataFrame do relatório, usando os dados de self.analyzer.reps.

        Args:
            df_report (pd.DataFrame): O DataFrame do relatório a ser preenchido.
        """
        # Mapeamento das chaves internas do analyzer.reps para os nomes de exibição
        body_part_map_internal_to_display = {
            'Cabeça': 'head',
            'Tronco': 'trunk',
            'Joelho': 'knee',
            'Pé': 'heel'
        }

        # Itera sobre as linhas do DataFrame para preencher os dados de repetição
        for index, row in df_report.iterrows():
            parte_display_name = row['Partes do corpo']
            internal_key = body_part_map_internal_to_display.get(parte_display_name)

            if internal_key and internal_key in self.analyzer.reps:
                # Obtém os status de erro (0 ou 1) para cada repetição da estrutura self.analyzer.reps
                reps_status = self.analyzer.reps.get(internal_key, [])
                
                # Garante que sempre haja 3 valores para as colunas de repetição, preenchendo com 0s se houver menos
                # E trata None como 0 para exibição na planilha e cálculo do resultado.
                padded_reps_status = [(val if val is not None else 0) for val in (reps_status + [None, None, None])[:3]]

                # Preenche as colunas de repetição
                df_report.loc[index, 'Repetição 1'] = padded_reps_status[0]
                df_report.loc[index, 'Repetição 2'] = padded_reps_status[1]
                df_report.loc[index, 'Repetição 3'] = padded_reps_status[2]
                
                # Condição para a coluna 'Resultado': 1 se 2 ou mais repetições tiveram erro, 0 caso contrário.
                # A soma já é segura porque padded_reps_status já tratou None como 0.
                resultado = 1 if sum(padded_reps_status) >= 2 else 0
                df_report.loc[index, 'Resultado'] = resultado
            else:
                print(f"DEBUG: Dados de repetição não encontrados ou chave inválida para '{parte_display_name}'.")


    def generate_report(self, total_video_duration_seconds): 
        """
        Gera o relatório Excel completo com os dados da análise.

        Args:
            total_video_duration_seconds (float): A duração total do vídeo em segundos.
                                                  Essencial para calcular 'Número de Erros por segundo'.
        """
        # 1. Define os cabeçalhos da planilha. Esta variável 'columns' é usada explicitamente.
        columns = [
            'Partes do corpo',
            'Número de Erros por segundo',
            'Repetição 1',
            'Repetição 2',
            'Repetição 3',
            'Resultado'
        ]

        # 2. Define os dados para a coluna 'Partes do corpo'
        body_parts_data = ['Cabeça', 'Tronco', 'Joelho', 'Pé']

        # 3. Prepara um dicionário com os dados iniciais para o DataFrame.
        # Inicializa as colunas numéricas com 0.0 e as de repetição/resultado com None.
        data_for_df = {
            'Partes do corpo': body_parts_data,
            'Número de Erros por segundo': [0.0] * len(body_parts_data), 
            'Repetição 1': [None] * len(body_parts_data),
            'Repetição 2': [None] * len(body_parts_data),
            'Repetição 3': [None] * len(body_parts_data),
            'Resultado': [None] * len(body_parts_data)
        }

        # 4. Cria o DataFrame Pandas usando a lista 'columns' para definir a estrutura.
        df_report = pd.DataFrame(data_for_df, columns=columns)

        # 5. Calcula e preenche a coluna 'Número de Erros por segundo'
        erros_por_segundo_list = []
        
        erros_por_segundo_list.append(self._calculate_errors_per_second(self.analyzer.head_error_history, total_video_duration_seconds))
        erros_por_segundo_list.append(self._calculate_errors_per_second(self.analyzer.trunk_error_history, total_video_duration_seconds))
        erros_por_segundo_list.append(self._calculate_errors_per_second(self.analyzer.knee_error_history, total_video_duration_seconds))
        erros_por_segundo_list.append(self._calculate_errors_per_second(self.analyzer.foot_error_history, total_video_duration_seconds)) 

        df_report['Número de Erros por segundo'] = erros_por_segundo_list

        # ---- Chama a função para preencher os dados de repetição e resultado ---
        self._fill_repetition_data(df_report)

        # 6. Cria a pasta 'planilhas' se ela não existir
        output_folder = 'planilhas'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"DEBUG: Pasta '{output_folder}' criada.")

        # 7. Define o caminho completo do arquivo
        file_path = os.path.join(output_folder, f"{self.person_name}.xlsx")
        print(f"DEBUG: Tentando salvar planilha em: {file_path}")

        # 8. Salva o DataFrame no arquivo Excel
        try:
            df_report.to_excel(file_path, index=False)
            st.success(f"Relatório de análise salvo com sucesso em '{file_path}'!")
        except Exception as e:
            st.error(f"Erro ao salvar o relatório Excel: {e}")
            st.warning("Certifique-se de que o arquivo não está aberto em outro programa e que você tem permissões de escrita.")