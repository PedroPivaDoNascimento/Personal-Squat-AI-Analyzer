import pandas as pd
import os
import streamlit as st

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
        
    def _fill_repetition_data(self, df_report):
        """
        Preenche as colunas 'Repetição 1', 'Repetição 2', 'Repetição 3' e 'Resultado'
        no DataFrame do relatório, usando os dados de self.analyzer.reps.
        Se os dados de uma parte do corpo estiverem ausentes, as células correspondentes serão preenchidas com 0.

        Args:
            df_report (pd.DataFrame): O DataFrame do relatório a ser preenchido.
        """
        # Mapeamento das chaves internas do analyzer.reps para os nomes de exibição
        body_part_map_internal_to_display = {
            'Cabeça': 'head',
            'Tronco': 'trunk',
            'Joelho': 'knee',
            'Pé': 'foot'
        }

        # Itera sobre as linhas do DataFrame para preencher os dados de repetição
        for index, row in df_report.iterrows():
            parte_display_name = row['Partes do corpo']
            internal_key = body_part_map_internal_to_display.get(parte_display_name)

            # Usamos o .get() com um valor padrão de lista vazia [] para garantir que
            # sempre teremos uma lista para trabalhar, mesmo que a chave 'foot' não exista.
            reps_status = self.analyzer.reps.get(internal_key, [])
            
            # Preenche os dados de repetição, substituindo valores ausentes por 0
            padded_reps_status = [(val if val is not None else 0) for val in (reps_status + [None, None, None])[:3]]

            df_report.loc[index, 'Repetição 1'] = padded_reps_status[0]
            df_report.loc[index, 'Repetição 2'] = padded_reps_status[1]
            df_report.loc[index, 'Repetição 3'] = padded_reps_status[2]
            
            # O resultado é calculado com base nos dados preenchidos
            resultado = 1 if sum(padded_reps_status) >= 2 else 0
            df_report.loc[index, 'Resultado'] = resultado


    def generate_report(self): 
        """
        Gera o relatório Excel completo com os dados da análise.
        """
        # 1. Define os cabeçalhos da planilha na ordem CORRETA.
        columns = [
            'Partes do corpo',
            'Número de erros Repetição 01',
            'Número de erros Repetição 02',
            'Número de erros Repetição 03',
            'Repetição 1',
            'Repetição 2',
            'Repetição 3',
            'Resultado'
        ]

        # 2. Define os dados para a coluna 'Partes do corpo'
        body_parts_data = ['Cabeça', 'Tronco', 'Joelho', 'Pé']

        # 3. Prepara um dicionário com os dados iniciais.
        data_for_df = {
            'Partes do corpo': body_parts_data,
            'Número de erros Repetição 01': [None] * len(body_parts_data),
            'Número de erros Repetição 02': [None] * len(body_parts_data),
            'Número de erros Repetição 03': [None] * len(body_parts_data),
            'Repetição 1': [None] * len(body_parts_data),
            'Repetição 2': [None] * len(body_parts_data),
            'Repetição 3': [None] * len(body_parts_data),
            'Resultado': [None] * len(body_parts_data)
        }

        # 4. Cria o DataFrame Pandas
        df_report = pd.DataFrame(data_for_df, columns=columns)

        # 5. Mapeamento das chaves internas do analyzer para os nomes de exibição
        body_part_map_internal_to_display = {
            'Cabeça': 'head_error_history',
            'Tronco': 'trunk_error_history',
            'Joelho': 'knee_error_history',
            'Pé': 'foot_error_history'
        }
        
        # 6. Preenche as colunas de contagem de erros por repetição
        for index, row in df_report.iterrows():
            parte_display_name = row['Partes do corpo']
            internal_key = body_part_map_internal_to_display.get(parte_display_name)

            if internal_key:
                error_counts = getattr(self.analyzer, internal_key, [])
                
                # Preenche os dados de contagem de erros, substituindo valores ausentes por 0
                padded_error_counts = [(val if val is not None else 0) for val in (error_counts + [None, None, None])[:3]]

                df_report.loc[index, 'Número de erros Repetição 01'] = padded_error_counts[0]
                df_report.loc[index, 'Número de erros Repetição 02'] = padded_error_counts[1]
                df_report.loc[index, 'Número de erros Repetição 03'] = padded_error_counts[2]
            else:
                print(f"DEBUG: Dados de histórico de erros não encontrados para '{parte_display_name}'.")

        # 7. Chama a função para preencher os dados de status (0 ou 1) de repetição e resultado.
        self._fill_repetition_data(df_report)

        # 8. Cria a pasta 'planilhas' se ela não existir
        output_folder = 'planilhas'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 9. Define o caminho completo do arquivo
        file_path = os.path.join(output_folder, f"{self.person_name}.xlsx")

        # 10. Salva o DataFrame no arquivo Excel
        try:
            df_report.to_excel(file_path, index=False)
            st.success(f"Relatório de análise salvo com sucesso em '{file_path}'!")
        except Exception as e:
            st.error(f"Erro ao salvar o relatório Excel: {e}")
            st.warning("Certifique-se de que o arquivo não está aberto em outro programa e que você tem permissões de escrita.")
