# Arquivo: classes/sagittal_report_excel_writer.py

from .base_squat_report_excel_writer import BaseSquatReportExcelWriter

class SagittalReportExcelWriter(BaseSquatReportExcelWriter):
    """Implementação concreta para gerar relatórios do Plano Sagital."""

    def __init__(self, person_name, squat_analyzer_instance, side):
        # Define o nome da pasta como 'sagital'
        self.side = side
        super().__init__(person_name, squat_analyzer_instance, 'sagital', side=side) 

    def _get_body_parts_data(self):
        # Partes do corpo específicas do plano sagital
        return ['Cabeça', 'Tronco', 'Joelho', 'Pé']

    def _get_internal_map_status(self):
        # Mapeamento do nome de exibição para a chave em self.analyzer.reps
        return {
            'Cabeça': 'head',
            'Tronco': 'trunk',
            'Joelho': 'knee',
            'Pé': 'heel' 
        }

    def _get_internal_map_count(self):
        # Mapeamento do nome de exibição para o atributo de histórico de contagem
        return {
            'Cabeça': 'head_error_history',
            'Tronco': 'trunk_error_history',
            'Joelho': 'knee_error_history',
            'Pé': 'foot_error_history'
        }