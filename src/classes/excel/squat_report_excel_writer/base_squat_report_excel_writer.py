# Arquivo: classes/base_squat_report_excel_writer.py

from abc import ABC, abstractmethod
import pandas as pd
import os
import streamlit as st

class BaseSquatReportExcelWriter(ABC):
    def __init__(self, person_name, squat_analyzer_instance, plane_folder_name, side="direito"):
        """
        Inicializa o gerador de relatórios Excel.

        Args:
            person_name (str): O nome da pessoa.
            squat_analyzer_instance (SquatRepetitionAnalyzer): A instância do analisador.
            plane_folder_name (str): O nome da subpasta específica do plano ('sagital' ou 'frontal').
        """
        
        self.side = side
        self.person_name = person_name
        self.analyzer = squat_analyzer_instance 
        self.plane_folder_name = plane_folder_name # Define o nome da pasta a partir da subclasse
        

    @abstractmethod
    def _get_body_parts_data(self):
        """
        Retorna a lista de nomes de partes do corpo específicas do plano para o relatório.
        """
        pass
    
    @abstractmethod
    def _get_internal_map_status(self):
        """
        Retorna o mapeamento de nomes de exibição para as chaves de status (0/1) 
        no self.analyzer.reps.
        """
        pass

    @abstractmethod
    def _get_internal_map_count(self):
        """
        Retorna o mapeamento de nomes de exibição para os atributos de histórico 
        de contagem no self.analyzer.
        """
        pass


    def _fill_repetition_data(self, df_report):
        """
        Preenche as colunas de status de repetição (0 ou 1) e Resultado.
        """
        body_part_map_internal_to_display = self._get_internal_map_status()
        
        for index, row in df_report.iterrows():
            parte_display_name = row['Partes do corpo']
            internal_key = body_part_map_internal_to_display.get(parte_display_name)
            
            if internal_key is None:
                padded_reps_status = [0, 0, 0]
            else:
                reps_status = self.analyzer.reps.get(internal_key, [])
                
                # Preenche e limita a 3 repetições
                padded_reps_status = [(val if val is not None else 0) for val in (reps_status + [None, None, None])[:3]]

            df_report.loc[index, 'Repetição 1'] = padded_reps_status[0]
            df_report.loc[index, 'Repetição 2'] = padded_reps_status[1]
            df_report.loc[index, 'Repetição 3'] = padded_reps_status[2]
            
            resultado = 1 if sum(padded_reps_status) >= 2 else 0
            df_report.loc[index, 'Resultado'] = resultado
            
            
    def _fill_error_count_data(self, df_report):
        """
        Preenche as colunas de contagem de erros por repetição.
        """
        body_part_map_internal_to_display = self._get_internal_map_count()
        
        for index, row in df_report.iterrows():
            parte_display_name = row['Partes do corpo']
            internal_key = body_part_map_internal_to_display.get(parte_display_name)

            if internal_key:
                error_counts = getattr(self.analyzer, internal_key, [])
                
                # Preenche e limita a 3 repetições
                padded_error_counts = [(val if val is not None else 0) for val in (error_counts + [None, None, None])[:3]]

                df_report.loc[index, 'Número de erros Repetição 01'] = padded_error_counts[0]
                df_report.loc[index, 'Número de erros Repetição 02'] = padded_error_counts[1]
                df_report.loc[index, 'Número de erros Repetição 03'] = padded_error_counts[2]
            else:
                # Preenche com 0 se não houver mapeamento
                df_report.loc[index, ['Número de erros Repetição 01', 'Número de erros Repetição 02', 'Número de erros Repetição 03']] = [0, 0, 0]


    def _save_report_to_excel(self, df_report):
        
        """
        Salva o relatório gerado em um arquivo Excel no diretório especificado.
        
        Args:
            df_report (pandas.DataFrame): O DataFrame com os dados do relatório.
        """
        
        output_folder = 'planilhas'
        
        # 1. Caminho da pasta do Plano (planilhas/sagital ou planilhas/frontal)
        plane_output_folder = os.path.join(output_folder, self.plane_folder_name)
        
        # 2. Caminho da pasta do Lado (planilhas/sagital/right ou planilhas/frontal/left)
        # O nome do lado deve ser minúsculo para consistência.
        side_folder_name = self.side.lower() 
        final_output_folder = os.path.join(plane_output_folder, side_folder_name) 
        
        # Define o caminho completo do arquivo
        file_path = os.path.join(final_output_folder, f"{self.person_name}_Relatorio_{self.plane_folder_name}_{side_folder_name}.xlsx") # Adicionei nome do plano e lado ao nome do arquivo para clareza

        try:
            df_report.to_excel(file_path, index=False)
            st.success(f"Relatório de análise salvo com sucesso em '{file_path}'!")
        except Exception as e:
            st.error(f"Erro ao salvar o relatório Excel: {e}")
            st.warning("Certifique-se de que o arquivo não está aberto em outro programa e que você tem permissões de escrita.")

    def generate_report(self): 
        """
        Método Template: Orquestra a geração do relatório.
        """
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

        # Obtém os dados das partes do corpo (Chamada Abstrata)
        body_parts_data = self._get_body_parts_data()

        # Prepara o DataFrame inicial
        data_for_df = {col: [None] * len(body_parts_data) if col != 'Partes do corpo' else body_parts_data for col in columns}
        df_report = pd.DataFrame(data_for_df, columns=columns)
        
        # Preenche as colunas de contagem de erros
        self._fill_error_count_data(df_report)

        # Preenche os dados de status (0 ou 1) e resultado
        self._fill_repetition_data(df_report)

        # Salva o relatório
        self._save_report_to_excel(df_report)