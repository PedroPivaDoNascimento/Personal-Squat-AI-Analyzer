"""
Serviço de Análise de Agachamento.

Este módulo encapsula a lógica de negócio da aplicação, mantendo-a separada
das Views (Django) e Templates. Aqui reside toda a lógica que antes estava
misturada com os componentes do Streamlit.
"""

import os
import uuid
from typing import Dict, Any, Optional, List

from django.conf import settings

# Importação das classes de negócio originais (regra de negócio preservada)
import sys
sys.path.insert(0, os.path.join(settings.BASE_DIR, 'src'))

from classes.personal_ai.saggital_personal_ai import SagittalAI
from classes.personal_ai.frontal_personal_ai import FrontalAI
from classes.excel.squat_report_excel_writer.sagittal_report_excel_writer import SagittalReportExcelWriter
from classes.excel.squat_report_excel_writer.frontal_report_excel_writer import FrontalReportExcelWriter
from classes.excel.set_folders import SetFolders
from ultils.feedback_messages import feedback_messages


MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'pose_landmarker_full.task')


class SquatAnalysisService:
    """
    Serviço responsável por orquestrar a análise de agachamento.
    
    Esta classe abstrai toda a lógica de processamento de vídeo e geração
    de relatórios, fornecendo uma interface limpa para as Views do Django.
    """
    
    def __init__(self):
        self.ai_instance = None
        self.session_id = None
    
    def analyze_sagittal(
        self,
        video_path: str,
        person_name: str,
        side: str,
        user_height_cm: int,
        params: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Realiza análise no plano sagital (direito ou esquerdo).
        
        Args:
            video_path: Caminho para o vídeo enviado
            person_name: Nome da pessoa analisada
            side: 'right' ou 'left'
            user_height_cm: Altura da pessoa em centímetros
            params: Dicionário com parâmetros de análise
            
        Returns:
            Dicionário com todos os resultados da análise
        """
        self.session_id = str(uuid.uuid4())[:8]
        
        # Criar estrutura de pastas
        side_label = "direito" if side == "right" else "esquerdo"
        set_folders = SetFolders(
            person_name=person_name,
            plane_folder_name="sagital",
            side=side_label
        )
        set_folders.create_folders()
        
        # Instanciar IA e processar vídeo
        self.ai_instance = SagittalAI(
            file_name=video_path,
            name_pessoa=person_name,
            side=side,
            user_height_cm=user_height_cm,
            model_path=MODEL_PATH,
            **params
        )
        self.ai_instance.process_video(draw=True, display=False)
        
        # Gerar relatório Excel
        excel_writer = SagittalReportExcelWriter(
            person_name,
            self.ai_instance.squat_analyzer,
            side_label
        )
        excel_writer.generate_report()
        
        # Remover arquivo temporário
        os.remove(video_path)
        
        return self._build_result_dict(person_name, is_frontal=False)
    
    def analyze_frontal(
        self,
        video_path: str,
        person_name: str,
        side: str,
        params: Dict[str, float],
        selected_repetitions: List[int]
    ) -> Dict[str, Any]:
        """
        Realiza análise no plano frontal (direito ou esquerdo).
        
        Args:
            video_path: Caminho para o vídeo enviado
            person_name: Nome da pessoa analisada
            side: 'right' ou 'left'
            params: Dicionário com parâmetros de análise
            selected_repetitions: Lista de repetições marcadas para salvar
            
        Returns:
            Dicionário com todos os resultados da análise
        """
        self.session_id = str(uuid.uuid4())[:8]
        
        # Criar estrutura de pastas
        side_label = "direito" if side == "right" else "esquerdo"
        set_folders = SetFolders(
            person_name=person_name,
            plane_folder_name="frontal",
            side=side_label
        )
        set_folders.create_folders()
        
        # Instanciar IA e processar vídeo
        self.ai_instance = FrontalAI(
            file_name=video_path,
            name_pessoa=person_name,
            model_path=MODEL_PATH,
            side=side,
            options_marcadas=selected_repetitions,
            **params
        )
        self.ai_instance.process_video(draw=True, display=False)
        
        # Gerar relatório Excel
        excel_writer = FrontalReportExcelWriter(
            person_name,
            self.ai_instance.squat_analyzer,
            side_label
        )
        excel_writer.generate_report()
        
        # Remover arquivo temporário
        os.remove(video_path)
        
        return self._build_result_dict(person_name, is_frontal=True)
    
    def _build_result_dict(self, person_name: str, is_frontal: bool) -> Dict[str, Any]:
        """
        Constrói um dicionário padronizado com os resultados da análise.
        """
        analyzer = self.ai_instance.squat_analyzer
        
        result = {
            'session_id': self.session_id,
            'person_name': person_name,
            'repetitions_detected': analyzer.repetitions_detected,
            'repetition_timestamps': analyzer.repetition_timestamps,
            'feedback_messages': feedback_messages,
        }
        
        if is_frontal:
            # Plano Frontal
            result['error_history'] = {
                'hip': analyzer.hip_error_history,
                'knee_valgus': analyzer.knee_valgus_error_history,
                'foot_pronation': analyzer.foot_pronation_error_history,
            }
            result['reps_status'] = {
                'hip': analyzer.reps['hip'],
                'knee_valgus': analyzer.reps['knee_valgus'],
                'foot_pronation': analyzer.reps['foot_pronation'],
            }
            result['dataframes'] = {
                'Desvios de Inclinação do Quadril': self.ai_instance.hip_tilt_df,
                'Desvios de Valgo/Varo de Joelho': self.ai_instance.knee_valgus_df,
                'Desvios de Pronação do Pé': self.ai_instance.foot_pronation_df,
            }
        else:
            # Plano Sagital
            result['error_history'] = {
                'trunk': analyzer.trunk_error_history,
                'knee': analyzer.knee_error_history,
                'head': analyzer.head_error_history,
                'heel': analyzer.foot_error_history,
            }
            result['reps_status'] = {
                'trunk': analyzer.reps['trunk'],
                'knee': analyzer.reps['knee'],
                'head': analyzer.reps['head'],
                'heel': analyzer.reps['heel'],
            }
            result['dataframes'] = {
                'Desvios da Cabeça': self.ai_instance.head_df,
                'Desvios do Tronco': self.ai_instance.trunk_df,
                'Desvios do Calcanhar': self.ai_instance.heel_df,
                'Desvios do Joelho': self.ai_instance.knee_df,
                'Pontos de Interseção do Tronco': analyzer.trunk_intersections_df,
            }
        
        return result
    
    @staticmethod
    def get_default_params(analysis_type: str) -> Dict[str, float]:
        """
        Retorna os parâmetros padrão para cada tipo de análise.
        """
        if 'sagittal' in analysis_type:
            return {
                'descent_threshold': 0.05,
                'ascent_return_threshold': 0.02,
                'trunk_error_threshold': 23,
                'knee_error_threshold': 6,
                'head_error_threshold': 2,
                'foot_error_threshold': 8,
            }
        else:  # frontal
            return {
                'descent_threshold': 0.05,
                'ascent_return_threshold': 0.02,
                'hip_error_threshold': 1,
                'knee_valgus_error_threshold': 12 if 'right' in analysis_type else 5,
                'foot_pronation_error_threshold': 7,
            }
