import math
import numpy as np
import pandas as pd
# Importação relativa
from ..vector_calculator import VectorCalculator 

class SquatRepetitionAnalyzerFrontal:
    # O construtor agora só aceita thresholds de contagem
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
        self.HIP_ANGLE_TOLERANCE = 5.0           # Limite de inclinação do quadril (5°)
        self.KNEE_ANGLE_MIN = 160.0              # Limite inferior do ângulo H-K-A (160°)
        self.FOOT_SHIFT_TOLERANCE = 0.9          # Limite de colapso lateral (0.9 = 10% de redução na distância A-T)

        # Históricos para Detecção de Repetição
        self.ear_y_inicial = None
        self.ear_y_history = []
        
        # Históricos para Calibração de Pronação
        self.ankle_x_inicial = None
        self.big_toe_x_inicial = None
        self.ankle_x_history = []
        self.big_toe_x_history = []

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
        
        # NOVOS: Históricos para Contagem Total de Erros por Repetição
        self.hip_error_history = []
        self.knee_valgus_error_history = []
        self.foot_pronation_error_history = []
        
        # Resultados por Repetição (status booleano 0/1)
        self.reps = {'hip': [], 'knee_valgus': [], 'foot_pronation': []}
        self.repetition_timestamps = []
        
        # DataFrames de Detalhe
        self.knee_valgus_df = pd.DataFrame(columns=["Tempo (ms)", "Ângulo Joelho"])
        self.hip_tilt_df = pd.DataFrame(columns=["Tempo (ms)", "Ângulo Quadril"])

    
    def _angle_to_horizontal(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1 
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        angle_deg = angle_deg % 360
        return min(abs(angle_deg), abs(180 - angle_deg))


    def create_dictionary_landmarks(self, lm_obj):
        try:
            return {
                'right_hip_x': lm_obj[24].x, 'right_hip_y': lm_obj[24].y,
                'left_hip_x': lm_obj[23].x, 'left_hip_y': lm_obj[23].y,
                'right_knee_x': lm_obj[26].x, 'right_knee_y': lm_obj[26].y,
                'right_ankle_x': lm_obj[28].x, 'right_ankle_y': lm_obj[28].y,
                'right_big_toe_x': lm_obj[32].x, 'right_big_toe_y': lm_obj[32].y,
                'right_ear_y': lm_obj[7].y, 
            }
        except Exception:
            return {}

    
    def process_frame_landmarks(self, landmarks_obj, timestamp_ms):
        hip = kn_valgus = foot_pronation = 0
        
        dict_lm =  self.create_dictionary_landmarks(landmarks_obj)

        if dict_lm == {}:
            return hip, kn_valgus, foot_pronation
        
        self._detect_repetition_phase(dict_lm, timestamp_ms)
        hip, kn_valgus, foot_pronation = self._check_errors_frontal(dict_lm, timestamp_ms)
        
        return hip, kn_valgus, foot_pronation

    def _detect_repetition_phase(self, dict_lm, ts):
        ear_y = dict_lm.get('right_ear_y')
        ankle_x = dict_lm.get('right_ankle_x')
        big_toe_x = dict_lm.get('right_big_toe_x')
        
        if ear_y is None or ankle_x is None or big_toe_x is None:
            return

        if self.ear_y_inicial is None:
            if len(self.ear_y_history) >= 10:
                self.ear_y_inicial = np.mean(self.ear_y_history[-10:])
                self.ankle_x_inicial = np.mean(self.ankle_x_history[-10:])
                self.big_toe_x_inicial = np.mean(self.big_toe_x_history[-10:])
            else:
                self.ear_y_history.append(ear_y)
                self.ankle_x_history.append(ankle_x)
                self.big_toe_x_history.append(big_toe_x)
                return
        
        self.ear_y_history.append(ear_y)

        if self.current_phase == 'inicial':
            self._reset_consecutive_counters() # Reseta apenas os consecutivos no início
            if ear_y > self.ear_y_inicial * (1 + self.DESCENT_THRESHOLD):
                self.current_phase = 'descendo'
                self.min_y_in_rep = ear_y
                
        elif self.current_phase == 'descendo':
            if ear_y > self.min_y_in_rep:
                self.min_y_in_rep = ear_y
            
            if ear_y < self.min_y_in_rep * 0.98: 
                self.current_phase = 'subindo'
                
        elif self.current_phase == 'subindo':
            if ear_y <= self.ear_y_inicial * (1 + self.ASCENT_RETURN_THRESHOLD):
                self.current_phase = 'final'
                self._complete_repetition(ts)
                
                if self.repetitions_detected < 3:
                    self.current_phase = 'inicial'
                    self.min_y_in_rep = None

    def _check_hip_tilt_error(self, dict_lm, timestamp_ms):
        hip_status = 0
        try:
            x1, y1 = dict_lm['left_hip_x'], dict_lm['left_hip_y']
            x2, y2 = dict_lm['right_hip_x'], dict_lm['right_hip_y']
            angle_deg = self._angle_to_horizontal(x1, y1, x2, y2)
            
            if angle_deg > self.HIP_ANGLE_TOLERANCE:
                self.consecutive_hip_error_counter += 1
                hip_status = 1
                self.hip_tilt_df.loc[len(self.hip_tilt_df)] = [timestamp_ms, angle_deg]
            else:
                self.consecutive_hip_error_counter = 0

            if self.consecutive_hip_error_counter >= self.HIP_ERROR_THRESHOLD:
                self.total_hip_error_counter += 1
                
        except Exception as e:
            self.consecutive_hip_error_counter = 0
            
        return hip_status


    def _check_knee_valgus_error(self, dict_lm, timestamp_ms):
        kn_valgus_status = 0
        try:
            p1 = (dict_lm['right_hip_x'], dict_lm['right_hip_y'])
            p2 = (dict_lm['right_knee_x'], dict_lm['right_knee_y'])
            p3 = (dict_lm['right_ankle_x'], dict_lm['right_ankle_y'])
            
            # Requer VectorCalculator.calculate_angle_3p
            angle_hka = VectorCalculator.calculate_angle_3p(p1, p2, p3)
            
            if angle_hka < self.KNEE_ANGLE_MIN:
                self.consecutive_knee_valgus_error_counter += 1
                kn_valgus_status = 1
                self.knee_valgus_df.loc[len(self.knee_valgus_df)] = [timestamp_ms, angle_hka]
            else:
                self.consecutive_knee_valgus_error_counter = 0

            if self.consecutive_knee_valgus_error_counter >= self.KNEE_VALGUS_ERROR_THRESHOLD:
                self.total_knee_valgus_error_counter += 1
                
        except Exception as e:
            self.consecutive_knee_valgus_error_counter = 0
            
        return kn_valgus_status

    def _check_foot_pronation_error(self, dict_lm, timestamp_ms):
        foot_pronation_status = 0
        try:
            ankle_x = dict_lm['right_ankle_x']
            big_toe_x = dict_lm['right_big_toe_x']

            initial_lateral_distance = abs(self.big_toe_x_inicial - self.ankle_x_inicial)
            current_lateral_distance = abs(big_toe_x - ankle_x)

            if current_lateral_distance < initial_lateral_distance * self.FOOT_SHIFT_TOLERANCE:
                self.consecutive_foot_pronation_error_counter += 1
                foot_pronation_status = 1
            else:
                self.consecutive_foot_pronation_error_counter = 0

            if self.consecutive_foot_pronation_error_counter >= self.FOOT_PRONATION_ERROR_THRESHOLD:
                self.total_foot_pronation_error_counter += 1
                
        except Exception as e:
            self.consecutive_foot_pronation_error_counter = 0
            
        return foot_pronation_status


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
            
            # 1. Armazena o status booleano (0 ou 1)
            hip_rep_result = 1 if self.total_hip_error_counter > 0 else 0
            knee_rep_result = 1 if self.total_knee_valgus_error_counter > 0 else 0
            foot_rep_result = 1 if self.total_foot_pronation_error_counter > 0 else 0
            
            self.reps['hip'].append(hip_rep_result)
            self.reps['knee_valgus'].append(knee_rep_result)
            self.reps['foot_pronation'].append(foot_rep_result)
            
            # 2. Armazena a contagem TOTAL de instantes para o display
            self.hip_error_history.append(self.total_hip_error_counter)
            self.knee_valgus_error_history.append(self.total_knee_valgus_error_counter)
            self.foot_pronation_error_history.append(self.total_foot_pronation_error_counter)
            
            self.repetitions_detected += 1
            self.repetition_timestamps.append(current_ts / 1000)
            
            # 3. Reseta os contadores totais para a nova repetição
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