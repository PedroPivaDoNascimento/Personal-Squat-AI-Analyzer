"""
Serviço de análise de agachamento - Camada de Modelo/Service
Encapsula toda a lógica de negócio para processamento de vídeo e ML.
Aplica o princípio de Responsabilidade Única (SOLID).
"""
import os
import sys
import tempfile
import pandas as pd

# Adiciona o diretório raiz ao path para importar as classes existentes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classes.personal_ai.frontal_personal_ai import FrontalAI
from src.classes.personal_ai.saggital_personal_ai import SagittalAI
from src.classes.excel.squat_report_excel_writer.frontal_report_excel_writer import FrontalReportExcelWriter
from src.classes.excel.squat_report_excel_writer.sagittal_report_excel_writer import SagittalReportExcelWriter
from src.classes.excel.set_folders import SetFolders


class SquatAnalysisService:
    """
    Serviço responsável por processar vídeos de agachamento e realizar análises.
    Responsabilidade única: orquestrar o fluxo de análise sem depender de frameworks web.
    """
    
    MODEL_PATH = 'models/pose_landmarker_full.task'
    
    def __init__(self):
        self.ai_instance = None
        self.analysis_result = None
    
    def analyze_frontal(self, video_file, person_name, side, params, selected_reps):
        """
        Realiza análise frontal (direito ou esquerdo).
        
        Args:
            video_file: Arquivo de vídeo enviado
            person_name: Nome da pessoa
            side: 'direito' ou 'esquerdo'
            params: Dicionário com parâmetros de análise
            selected_reps: Lista de repetições selecionadas para salvar
        
        Returns:
            Dicionário com resultados da análise
        """
        # Cria pasta para organização
        set_folders = SetFolders(
            person_name=person_name,
            plane_folder_name="frontal",
            side=side
        )
        set_folders.create_folders()
        
        # Salva vídeo temporariamente
        ext = os.path.splitext(video_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(video_file.read())
            temp_path = tmp_file.name
        
        try:
            # Instancia e processa com a IA
            ai_side = "right" if side == "direito" else "left"
            self.ai_instance = FrontalAI(
                file_name=temp_path,
                name_pessoa=person_name,
                model_path=self.MODEL_PATH,
                side=ai_side,
                options_marcadas=selected_reps,
                **params
            )
            self.ai_instance.process_video(draw=False, display=False)
            
            # Gera relatório Excel
            excel_writer = FrontalReportExcelWriter(
                person_name=person_name,
                squat_analyzer_instance=self.ai_instance.squat_analyzer,
                side=side
            )
            excel_writer.generate_report()
            
            # Extrai resultados
            self.analysis_result = self._extract_frontal_results()
            
        finally:
            # Limpa arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return self.analysis_result
    
    def analyze_sagittal(self, video_file, person_name, side, user_height_cm, params):
        """
        Realiza análise sagital (direito ou esquerdo).
        
        Args:
            video_file: Arquivo de vídeo enviado
            person_name: Nome da pessoa
            side: 'direito' ou 'esquerdo'
            user_height_cm: Altura do usuário em cm
            params: Dicionário com parâmetros de análise
        
        Returns:
            Dicionário com resultados da análise
        """
        # Cria pasta para organização
        set_folders = SetFolders(
            person_name=person_name,
            plane_folder_name="sagital",
            side=side
        )
        set_folders.create_folders()
        
        # Salva vídeo temporariamente
        ext = os.path.splitext(video_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(video_file.read())
            temp_path = tmp_file.name
        
        try:
            # Instancia e processa com a IA
            ai_side = "right" if side == "direito" else "left"
            self.ai_instance = SagittalAI(
                file_name=temp_path,
                name_pessoa=person_name,
                side=ai_side,
                user_height_cm=user_height_cm,
                model_path=self.MODEL_PATH,
                **params
            )
            self.ai_instance.process_video(draw=False, display=False)
            
            # Gera relatório Excel
            excel_writer = SagittalReportExcelWriter(
                person_name=person_name,
                squat_analyzer_instance=self.ai_instance.squat_analyzer,
                side=side
            )
            excel_writer.generate_report()
            
            # Extrai resultados
            self.analysis_result = self._extract_sagittal_results()
            
        finally:
            # Limpa arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return self.analysis_result
    
    def _extract_frontal_results(self):
        """Extrai resultados padronizados da análise frontal."""
        analyzer = self.ai_instance.squat_analyzer
        
        # Prepara dados dos gráficos por repetição
        repetition_charts = []
        for i in range(len(analyzer.hip_error_history)):
            if analyzer.hip_error_history[i] is not None:
                chart_data = {
                    'repetition': i + 1,
                    'hip_count': analyzer.hip_error_history[i],
                    'knee_count': analyzer.knee_valgus_error_history[i],
                    'foot_count': analyzer.foot_pronation_error_history[i]
                }
                repetition_charts.append(chart_data)
            else:
                repetition_charts.append({'repetition': i + 1, 'not_detected': True})
        
        # Prepara detalhes por repetição
        repetition_details = []
        for i in range(len(analyzer.reps['hip'])):
            hip_status = analyzer.reps['hip'][i]
            knee_status = analyzer.reps['knee_valgus'][i]
            foot_status = analyzer.reps['foot_pronation'][i]
            
            if hip_status is not None and analyzer.repetition_timestamps[i] is not None:
                detail = {
                    'repetition': i + 1,
                    'timestamp': analyzer.repetition_timestamps[i],
                    'hip_status': 'DESVIO ❌' if hip_status == 1 else 'OK ✅',
                    'knee_status': 'DESVIO ❌' if knee_status == 1 else 'OK ✅',
                    'foot_status': 'DESVIO ❌' if foot_status == 1 else 'OK ✅',
                    'hip_count': analyzer.hip_error_history[i],
                    'knee_count': analyzer.knee_valgus_error_history[i],
                    'foot_count': analyzer.foot_pronation_error_history[i],
                    'feedback': self._get_frontal_feedback(hip_status, knee_status, foot_status)
                }
                repetition_details.append(detail)
        
        # Prepara DataFrames para exibição
        dataframes = {
            'hip_tilt': self._prepare_dataframe_display(self.ai_instance.hip_tilt_df),
            'knee_valgus': self._prepare_dataframe_display(self.ai_instance.knee_valgus_df),
            'foot_pronation': self._prepare_dataframe_display(self.ai_instance.foot_pronation_df)
        }
        
        return {
            'type': 'frontal',
            'repetitions_detected': analyzer.repetitions_detected,
            'repetition_charts': repetition_charts,
            'repetition_details': repetition_details,
            'dataframes': dataframes
        }
    
    def _extract_sagittal_results(self):
        """Extrai resultados padronizados da análise sagital."""
        analyzer = self.ai_instance.squat_analyzer
        
        # Prepara dados dos gráficos por repetição
        repetition_charts = []
        for i in range(len(analyzer.trunk_error_history)):
            if analyzer.trunk_error_history[i] is not None:
                chart_data = {
                    'repetition': i + 1,
                    'trunk_count': analyzer.trunk_error_history[i],
                    'knee_count': analyzer.knee_error_history[i],
                    'head_count': analyzer.head_error_history[i],
                    'heel_count': analyzer.foot_error_history[i]
                }
                repetition_charts.append(chart_data)
            else:
                repetition_charts.append({'repetition': i + 1, 'not_detected': True})
        
        # Prepara detalhes por repetição
        repetition_details = []
        for i in range(len(analyzer.reps['trunk'])):
            if analyzer.reps['trunk'][i] is not None and analyzer.repetition_timestamps[i] is not None:
                detail = {
                    'repetition': i + 1,
                    'timestamp': analyzer.repetition_timestamps[i],
                    'trunk_status': 'DESVIO ❌' if analyzer.reps['trunk'][i] == 1 else 'OK ✅',
                    'knee_status': 'DESVIO ❌' if analyzer.reps['knee'][i] == 1 else 'OK ✅',
                    'head_status': 'DESVIO ❌' if analyzer.reps['head'][i] == 1 else 'OK ✅',
                    'heel_status': 'DESVIO ❌' if analyzer.reps['heel'][i] == 1 else 'OK ✅',
                    'trunk_count': analyzer.trunk_error_history[i],
                    'knee_count': analyzer.knee_error_history[i],
                    'head_count': analyzer.head_error_history[i],
                    'heel_count': analyzer.foot_error_history[i],
                    'feedback': self._get_sagittal_feedback(
                        analyzer.reps['trunk'][i],
                        analyzer.reps['knee'][i],
                        analyzer.reps['head'][i],
                        analyzer.reps['heel'][i]
                    )
                }
                repetition_details.append(detail)
        
        # Prepara DataFrames para exibição
        dataframes = {
            'head': self._prepare_dataframe_display(self.ai_instance.head_df),
            'trunk': self._prepare_dataframe_display(self.ai_instance.trunk_df),
            'heel': self._prepare_dataframe_display(self.ai_instance.heel_df),
            'knee': self._prepare_dataframe_display(self.ai_instance.knee_df),
            'trunk_intersections': self._prepare_dataframe_display(analyzer.trunk_intersections_df)
        }
        
        return {
            'type': 'sagittal',
            'repetitions_detected': analyzer.repetitions_detected,
            'repetition_charts': repetition_charts,
            'repetition_details': repetition_details,
            'dataframes': dataframes
        }
    
    def _prepare_dataframe_display(self, df):
        """Prepara DataFrame para exibição no template."""
        if df is None or df.empty:
            return None
        
        df_display = df.copy()
        time_column = 'Tempo (ms)'
        if time_column in df_display.columns:
            df_display[time_column] = (df_display[time_column] / 1000).round(2)
            df_display.rename(columns={time_column: 'Tempo (s)'}, inplace=True)
        
        return df_display.to_dict('records')
    
    def _get_frontal_feedback(self, hip_status, knee_status, foot_status):
        """Retorna feedback baseado nos status da análise frontal."""
        from src.ultils.feedback_messages import feedback_messages
        
        feedback_list = []
        if hip_status == 1:
            feedback_list.append(f"💡 Desvio no Quadril: {feedback_messages.get('hip_error', 'Verifique a estabilidade lateral do quadril.')}")
        if knee_status == 1:
            feedback_list.append(f"💡 Desvio no Joelho: {feedback_messages.get('knee_valgus_error', 'Valgo ou Varo detectado. Fortaleça abdutores.')}")
        if foot_status == 1:
            feedback_list.append(f"💡 Desvio no Pé: {feedback_messages.get('foot_pronation_error', 'Pronação excessiva. Fortaleça a musculatura intrínseca do pé.')}")
        
        if not feedback_list:
            return ["✅ **Ótima execução!** Continue assim."]
        return feedback_list
    
    def _get_sagittal_feedback(self, trunk_status, knee_status, head_status, heel_status):
        """Retorna feedback baseado nos status da análise sagital."""
        from src.ultils.feedback_messages import feedback_messages
        
        feedback_list = []
        if trunk_status == 1:
            feedback_list.append(f"💡 {feedback_messages['trunk_error']}")
        if knee_status == 1:
            feedback_list.append(f"💡 {feedback_messages['knee_error']}")
        if head_status == 1:
            feedback_list.append(f"💡 {feedback_messages['head_error']}")
        if heel_status == 1:
            feedback_list.append(f"💡 {feedback_messages['heel_error']}")
        
        if not feedback_list:
            return ["✅ **Ótima execução!** Continue assim."]
        return feedback_list
    
    @staticmethod
    def get_excel_file_path(person_name, analysis_type, side):
        """
        Retorna o caminho do arquivo Excel gerado para uma análise específica.
        
        Args:
            person_name: Nome da pessoa
            analysis_type: 'frontal' ou 'sagittal'
            side: 'direito' ou 'esquerdo'
        
        Returns:
            Caminho completo do arquivo Excel
        """
        # Mapeia o tipo de análise para o nome da pasta
        plane_folder = 'frontal' if analysis_type == 'frontal' else 'sagital'
        side_lower = side.lower()
        
        # Constrói o caminho base
        output_folder = 'planilhas'
        plane_output_folder = os.path.join(output_folder, plane_folder)
        
        # Para análise frontal, há uma subpasta adicional 'dados_pe'
        if analysis_type == 'frontal':
            final_output_folder = os.path.join(plane_output_folder, side_lower)
        else:
            final_output_folder = os.path.join(plane_output_folder, side_lower)
        
        # Nome do arquivo segue o padrão: {person_name}_Relatorio_{plane}_{side}.xlsx
        file_name = f"{person_name}_Relatorio_{plane_folder}_{side_lower}.xlsx"
        file_path = os.path.join(final_output_folder, file_name)
        
        return file_path
