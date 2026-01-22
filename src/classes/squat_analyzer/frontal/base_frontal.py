from ...vector_calculator import VectorCalculator 
from abc import ABC, abstractmethod

# TODO Terminar os calculos de cada parte, testar eles

class BaseFrontal(ABC):
    
    def __init__(self, descent_threshold=0.05, ascent_return_threshold=0.02, 
                 hip_error_threshold=5, knee_valgus_error_threshold=5, 
                 foot_pronation_error_threshold=5):
        
        self.DESCENT_THRESHOLD = descent_threshold
        self.ASCENT_RETURN_THRESHOLD = ascent_return_threshold
        
        # Thresholds para Contagem de Erros (mantidos como parâmetros)
        self.HIP_ERROR_THRESHOLD = hip_error_threshold
        self.KNEE_VALGUS_ERROR_THRESHOLD = knee_valgus_error_threshold
        self.FOOT_PRONATION_ERROR_THRESHOLD = foot_pronation_error_threshold
        
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
    
    @abstractmethod
    def create_dictionary_landmarks(self, lm_obj):
        pass
    
    def process_frame_landmarks(self, landmarks_obj, timestamp_ms):
        hip = kn_valgus = foot_pronation = 0
        
        dict_lm =  self.create_dictionary_landmarks(landmarks_obj)

        if dict_lm == {}:
            return hip, kn_valgus, foot_pronation
        
        self._detect_repetition_phase(dict_lm, timestamp_ms)
        hip, kn_valgus, foot_pronation = self._check_errors_frontal(dict_lm, timestamp_ms)
        
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
    def _check_foot_pronation_error(self, dict_lm, timestamp_ms):
        pass

    def _check_errors_frontal(self, dict_lm, timestamp_ms):
        hip_status = kn_valgus_status = foot_pronation_status = 0

        if self.current_phase in ['descendo', 'subindo']:
            
            hip_status = self._check_hip_tilt_error(dict_lm, timestamp_ms)
            kn_valgus_status = self._check_knee_valgus_error(dict_lm, timestamp_ms)
            foot_pronation_status = self._check_foot_pronation_error(dict_lm, timestamp_ms)
        
        return hip_status, kn_valgus_status, foot_pronation_status

    def _reset_consecutive_counters(self):
        """Reseta apenas os contadores de frames CONSECUTIVOS."""
        self.consecutive_hip_error_counter = 0
        self.consecutive_knee_valgus_error_counter = 0
        self.consecutive_foot_pronation_error_counter = 0

    def _complete_repetition(self, current_ts):
        """Registra os erros de uma repetição completa e prepara para a próxima."""
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
  
    def finalize_analysis(self):
        """Garante que a estrutura de 3 repetições esteja completa."""
        num_detected = self.repetitions_detected
        if num_detected < 3:
            for i in range(num_detected, 3):
                for key in ['hip', 'knee_valgus', 'foot_pronation']:
                    self.reps[key].append(0)
                
                # Preenche os novos históricos de contagem com 0
                self.hip_error_history.append(0)
                self.knee_valgus_error_history.append(0)
                self.foot_pronation_error_history.append(0)
                    
                self.repetition_timestamps.append(None)