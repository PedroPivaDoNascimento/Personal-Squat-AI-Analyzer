from abc import ABC, abstractmethod

from classes.excel.foot_data_excel_writer import FootDataExcelWriter
from ...images.video_processor import VideoProcessor
from classes.excel.pixels_excel_writer import PixelsDataExcelWriter

class BaseFrontal(ABC):
    
    def __init__(self, descent_threshold=0.05, ascent_return_threshold=0.02, 
                 hip_error_threshold=5, knee_valgus_error_threshold=5, 
                 foot_pronation_error_threshold=5, side="", person_name="", options_marcadas=[]):
        
        self.DESCENT_THRESHOLD = descent_threshold
        self.ASCENT_RETURN_THRESHOLD = ascent_return_threshold
        self.options_marcadas = options_marcadas
        
        # Thresholds para Contagem de Erros (mantidos como parâmetros)
        self.HIP_ERROR_THRESHOLD = hip_error_threshold
        self.KNEE_VALGUS_ERROR_THRESHOLD = knee_valgus_error_threshold
        self.FOOT_PRONATION_ERROR_THRESHOLD = foot_pronation_error_threshold

        self.side = side
        self.person_name = person_name
        
        # Tolerâncias Angulares e de Deslocamento (FIXADAS INTERNAMENTE)
        self.HIP_ANGLE_TOLERANCE =  180.25       # Limite de inclinação do quadril  180.25 
        self.KNEE_ANGLE_MIN = 170.0              # Limite inferior do ângulo H-K-A (170°)
        self.Y_SHIFT_TOLERANCE = 0           # Valor que o ponto entre o dedão e o calanhar deve subir 0.002
        
        
        # Históricos para Detecção de Repetição
        self.ear_y_inicial = None
        self.ear_y_history = [] 
        self.initial_midpoint_y = None
        
        self.heel_y_inicial = None          # Y inicial do Calcanhar
        self.big_toe_y_inicial = None       # Y inicial do Dedão
        self.heel_y_history = [] 
        self.big_toe_y_history = [] 
        
        self.repetitions_detected = 0
        self.current_phase = 'inicial'
        self.min_y_in_rep = None

        # Contadores Consecutivos (para erros temporários)
        self.consecutive_hip_error_counter = 0
        self.consecutive_knee_valgus_error_counter = 0
        self.consecutive_foot_pronation_error_counter = 0

        # Contadores Totais de Erro (para o relatório final da repetição)
        self.total_hip_error_counter = 0
        self.total_knee_valgus_error_counter = 0
        self.total_foot_pronation_error_counter = 0
        
        self.hip_error_history = []
        self.knee_valgus_error_history = []
        self.foot_pronation_error_history = []
        
        # Resultados por Repetição (status booleano 0/1)
        self.reps = {'hip': [], 'knee_valgus': [], 'foot_pronation': []}
        self.repetition_timestamps = []
        self.foot_repeat_data = []

        self.video_processor = VideoProcessor("pe_esquerdo")

        self.history_whites_pixels_with_frame = {}
        self.pixel_excel_writer = PixelsDataExcelWriter(person_name=self.person_name, plane_folder_name="frontal", side=self.side)


    
    @abstractmethod
    def create_dictionary_landmarks(self, lm_obj):
        pass
    
    @abstractmethod
    def _get_foot_data(self, dict_lm):
        pass

    

    def process_frame_landmarks(self, landmarks_obj, timestamp_ms, frame, frame_number):
        """
        Processa os landmarks de uma frame e retorna os status de erro de quadril, joelho valgo e pronacao do pé.
        
        Parameters:
        landmarks_obj (Landmark): Landmarks da frame
        timestamp_ms (int): Timestamp da frame em milissegundos
        frame: Frame atual
        frame_number (int): Número do frame atual
        
        Returns:
        hip (int): Status de erro de quadril (0 - sem erro, 1 - com erro)
        kn_valgus (int): Status de erro de joelho valgo (0 - sem erro, 1 - com erro)
        foot_pronation (int): Status de erro de pronacao do pé (0 - sem erro, 1 - com erro)
        """

        hip = kn_valgus = foot_pronation = 0
        
        dict_lm =  self.create_dictionary_landmarks(landmarks_obj)

        if dict_lm == {}:
            return hip, kn_valgus, foot_pronation
        
        # Pego os dados do pé e coloco no vetor onde tem os dados da repetição de pé
        foot_data = self._get_foot_data(dict_lm)
        self.foot_repeat_data.append(foot_data)
        
        self._detect_repetition_phase(dict_lm, timestamp_ms)
        hip, kn_valgus, foot_pronation = self._check_errors_frontal(dict_lm, timestamp_ms, frame, frame_number)
        
        return hip, kn_valgus, foot_pronation


    @abstractmethod
    def _detect_repetition_phase(self, dict_lm, ts):
        pass

    @abstractmethod
    def _check_hip_tilt_error(self, dict_lm, timestamp_ms):
        pass

    @abstractmethod
    def _check_knee_valgus_error(self, dict_lm, timestamp_ms):
       pass

    @abstractmethod
    def _check_foot_pronation_error(self, dict_lm, timestamp_ms, frame):
        pass

    @abstractmethod
    def _get_center_point_foot(self, dict_lm):
        pass

    def _check_errors_frontal(self, dict_lm, timestamp_ms, frame, frame_number):
        hip_status = kn_valgus_status = foot_pronation_status = 0

        if self.current_phase in ['descendo', 'subindo']:
            
            hip_status = self._check_hip_tilt_error(dict_lm, timestamp_ms)
            kn_valgus_status = self._check_knee_valgus_error(dict_lm, timestamp_ms)
            foot_pronation_status = self._check_foot_pronation_error(dict_lm, timestamp_ms, frame, frame_number)
        
        return hip_status, kn_valgus_status, foot_pronation_status

    def _reset_consecutive_counters(self):
        """Reseta apenas os contadores de frames CONSECUTIVOS."""
        self.consecutive_hip_error_counter = 0
        self.consecutive_knee_valgus_error_counter = 0
        self.consecutive_foot_pronation_error_counter = 0


    def _complete_repetition(self, current_ts):        
        if self.repetitions_detected < 3:
            
            hip_rep_result = 1 if self.total_hip_error_counter > 0 else 0
            knee_rep_result = 1 if self.total_knee_valgus_error_counter > 0 else 0
            foot_rep_result = 1 if self.total_foot_pronation_error_counter > 0 else 0
            
            self.reps['hip'].append(hip_rep_result)
            self.reps['knee_valgus'].append(knee_rep_result)
            self.reps['foot_pronation'].append(foot_rep_result)
            
            self.hip_error_history.append(self.total_hip_error_counter)
            self.knee_valgus_error_history.append(self.total_knee_valgus_error_counter)
            self.foot_pronation_error_history.append(self.total_foot_pronation_error_counter)
            
            self.repetitions_detected += 1
            self.repetition_timestamps.append(current_ts / 1000)
            
            self.total_hip_error_counter = 0
            self.total_knee_valgus_error_counter = 0
            self.total_foot_pronation_error_counter = 0
            # Os consecutivos serão resetados no 'inicial'

            if self.repetitions_detected in self.options_marcadas:
                foot_data_excel_writer = FootDataExcelWriter(self.repetitions_detected, self.foot_repeat_data, self.person_name, "frontal", self.side)
                foot_data_excel_writer.write_raw_foot_data()
                foot_data_excel_writer.write_statistic_foot_data()

            self.foot_repeat_data = []




  
    def finalize_analysis(self, current_ts):
        """
        Finaliza a análise de repetições detectadas.

        Se o voluntário fez menos de 2 repetições, preenche os slots restantes com 0.
        Se a 3ª repetição começou mas não terminou (voluntário parou no meio), completa a repetição.

        Args:
            current_ts (int): O timestamp atual em milissegundos.
        """
        num_detected = self.repetitions_detected

        if num_detected == 2:
            self._complete_repetition(current_ts=current_ts)
            return

        reps_to_fill = 3 - num_detected
        
        for _ in range(reps_to_fill):
            for key in self.reps.keys():
                self.reps[key].append(-1)
            
            self._fill_error_histories(value=-1)
            self.repetition_timestamps.append(None)

        self.pixel_excel_writer.write_num_pixels_data(self.history_whites_pixels_with_frame)
        

        

    def _fill_error_histories(self, value):
        """
        Preenche as históricos de erros com um valor específico.
        
        Args:
            value (int): O valor a ser preenchido nasæ históricos de erros.
        """
        histories = [
            self.hip_error_history, 
            self.knee_valgus_error_history, 
            self.foot_pronation_error_history
        ]
        for hist in histories:
            hist.append(value)