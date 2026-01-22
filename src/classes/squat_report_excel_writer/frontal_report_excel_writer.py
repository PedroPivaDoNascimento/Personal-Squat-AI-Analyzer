# Arquivo: classes/frontal_report_excel_writer.py

from .base_squat_report_excel_writer import BaseSquatReportExcelWriter

class FrontalReportExcelWriter(BaseSquatReportExcelWriter):
    """Implementação concreta para gerar relatórios do Plano Frontal."""

    def __init__(self, person_name, squat_analyzer_instance, side="direito"):
        # Define o nome da pasta como 'frontal'
        super().__init__(person_name, squat_analyzer_instance, 'frontal', side=side) # <-- Define a pasta

    def _get_body_parts_data(self):
        # Partes do corpo específicas do plano frontal
        return ['Quadril (Inclinação)', 'Joelho (Valgo/Varo)', 'Pé (Pronação)']

    def _get_internal_map_status(self):
        # Mapeamento do nome de exibição para a chave em self.analyzer.reps
        return {
            'Quadril (Inclinação)': 'hip',
            'Joelho (Valgo/Varo)': 'knee_valgus',
            'Pé (Pronação)': 'foot_pronation' 
        }

    def _get_internal_map_count(self):
        # Mapeamento do nome de exibição para o atributo de histórico de contagem
        return {
            'Quadril (Inclinação)': 'hip_error_history',
            'Joelho (Valgo/Varo)': 'knee_valgus_error_history',
            'Pé (Pronação)': 'foot_pronation_error_history'
        }