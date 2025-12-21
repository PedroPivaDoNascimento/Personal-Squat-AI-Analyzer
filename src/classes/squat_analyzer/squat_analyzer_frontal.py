import numpy as np
import math
from ..vector_calculator import VectorCalculator 

# TODO Terminar os calculos de cada parte, testar eles

class SquatRepetitionAnalyzerFrontal:
    
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
        self.HIP_ANGLE_TOLERANCE = 2.8       # Variação máxima em graus que o quadril pode ter   < 2.8°   >1.75  Antes 2.8 
        self.HIP_ANGLE_TOLERANCE_INVERSE = -2.1   # Variação máxima em graus para o outro quadrante Antes era -2.1
        self.KNEE_ANGLE_MIN = 170.0          # Limite inferior do ângulo H-K-A (170°)
        self.Y_SHIFT_TOLERANCE = 0           # Valor que o ponto entre o dedão e o calanhar deve subir 0.002
        
        # Variaveis usadas no cálculo do quadril
        self.left_hip_x_initial = None
        self.left_hip_y_initial = None
        self.right_hip_x_initial = None
        self.right_hip_y_initial = None

        self.left_hip_x_history = []
        self.left_hip_y_history = []
        self.right_hip_x_history = []
        self.right_hip_y_history = []

        self.initial_angle_hip = None

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
    
    
    def create_dictionary_landmarks(self, lm_obj):
        try:
            return {
                'right_hip_x': lm_obj[24].x, 'right_hip_y': lm_obj[24].y,
                'left_hip_x': lm_obj[23].x, 'left_hip_y': lm_obj[23].y,
                'right_knee_x': lm_obj[26].x, 'right_knee_y': lm_obj[26].y,
                'right_ankle_x': lm_obj[28].x, 'right_ankle_y': lm_obj[28].y,
                'right_big_toe_x': lm_obj[32].x, 'right_big_toe_y': lm_obj[32].y,
                'right_ear_y': lm_obj[7].y,
                'right_heel_y': lm_obj[30].y,
            }
        except Exception as e:
            print(f"Erro ao criar dicionário de landmarks: {e}")
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
        ear_y = dict_lm['right_ear_y']
        heel_y = dict_lm['right_heel_y']
        big_toe_y = dict_lm['right_big_toe_y']

        left_hip_x, left_hip_y = dict_lm['left_hip_x'], dict_lm['left_hip_y']
        right_hip_x, right_hip_y = dict_lm['right_hip_x'], dict_lm['right_hip_y']

        # Atualizando a checagem com os novos pontos Y
        if ear_y is None or heel_y is None or big_toe_y is None:
            return

        if self.ear_y_inicial is None:
            # Se ainda não calibrou (está nos primeiros frames)
            
            if len(self.ear_y_history) >= 10:
                self.ear_y_inicial = np.mean(self.ear_y_history[-10:])

                self.heel_y_inicial = np.mean(self.heel_y_history[-10:])
                self.big_toe_y_inicial = np.mean(self.big_toe_y_history[-10:])
                self.initial_midpoint_y = (self.heel_y_inicial + self.big_toe_y_inicial) / 2

                self.left_hip_x_initial = np.mean(self.left_hip_x_history[-10:])
                self.left_hip_y_initial = np.mean(self.left_hip_y_history[-10:])
                self.right_hip_x_initial = np.mean(self.right_hip_x_history[-10:])
                self.right_hip_y_initial = np.mean(self.right_hip_y_history[-10:])
                self.initial_angle_hip = VectorCalculator.angle_to_horizontal(left_hip_x, left_hip_y, right_hip_x, right_hip_y)
                     
            else:
                self.ear_y_history.append(ear_y)

                self.heel_y_history.append(heel_y) 
                self.big_toe_y_history.append(big_toe_y)

                self.left_hip_x_history.append(left_hip_x)
                self.left_hip_y_history.append(left_hip_y)
                self.right_hip_x_history.append(right_hip_x)
                self.right_hip_y_history.append(right_hip_y)
                return
        
        self.ear_y_history.append(ear_y)

        if self.current_phase == 'inicial':
            self._reset_consecutive_counters() 
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

            angle_deg = VectorCalculator.angle_to_horizontal(x1, y1, x2, y2)
            
            angulo_variou = angle_deg > (self.initial_angle_hip + self.HIP_ANGLE_TOLERANCE)
            angulo_variou_inverso = (angle_deg - self.initial_angle_hip < self.HIP_ANGLE_TOLERANCE_INVERSE)
            angulou_passou_180 = angle_deg > 180

            #if (self.repetitions_detected != 1):
            print(f"Angulo do quadril: {angle_deg:.2f}° atual, angulo do quadril inicial: {self.initial_angle_hip:.2f}°, variação do angulo: {angle_deg - self.initial_angle_hip:.2f}°, no segundo: {timestamp_ms/1000:.2f}")


            if (angulo_variou or angulou_passou_180 or angulo_variou_inverso):
                self.consecutive_hip_error_counter += 1
                hip_status = 1


            else:
                self.consecutive_hip_error_counter = 0

            if self.consecutive_hip_error_counter >= self.HIP_ERROR_THRESHOLD:
                self.total_hip_error_counter += 1
                self.consecutive_hip_error_counter = 0
        except Exception as e:
            print(f"Erro ao calcular inclinação do quadril: {e}")
            self.consecutive_hip_error_counter = 0
            
        return hip_status

    def _check_knee_valgus_error(self, dict_lm, timestamp_ms):
        kn_valgus_status = 0
        try:
            # Extrai as 6 coordenadas (x,y) de p1, p2 e p3
            x1 = dict_lm['right_hip_x']
            y1 = dict_lm['right_hip_y']
            x2 = dict_lm['right_knee_x']
            y2 = dict_lm['right_knee_y']
            x3 = dict_lm['right_ankle_x']
            y3 = dict_lm['right_ankle_y']
            
            angle_hka = VectorCalculator.calculate_angle_3p(x1, y1, x2, y2, x3, y3)
            
            if angle_hka < self.KNEE_ANGLE_MIN and angle_hka < 0:
                #print(f"{angle_hka:.2f}")
                self.consecutive_knee_valgus_error_counter += 1
                kn_valgus_status = 1
            else:
                self.consecutive_knee_valgus_error_counter = 0

            if self.consecutive_knee_valgus_error_counter >= self.KNEE_VALGUS_ERROR_THRESHOLD:
                self.total_knee_valgus_error_counter += 1
                self.consecutive_knee_valgus_error_counter = 0
                #print(f"Angulo atual é de {angle_hka:.2f} e ocorreu no segundo {timestamp_ms/1000:.2f}")


                
        except Exception as e:
            print(f"Erro ao calcular valgo de joelho: {e}")
            self.consecutive_knee_valgus_error_counter = 0
            
        return kn_valgus_status

    def _check_foot_pronation_error(self, dict_lm, timestamp_ms):
        foot_pronation_status = 0
    
    
        if self.initial_midpoint_y is None:
            return 0 
        
        try:
            heel_y = dict_lm['right_heel_y']
            big_toe_y = dict_lm['right_big_toe_y']
            
            current_midpoint_y = (heel_y + big_toe_y) / 2
            
    
            if current_midpoint_y < self.initial_midpoint_y - self.Y_SHIFT_TOLERANCE:
                self.consecutive_foot_pronation_error_counter += 1
                foot_pronation_status = 1
                #print(f"ERRO DE PRONAÇÃO (Midpoint Y): Ponto médio atual: {current_midpoint_y:.4f} (Limite: {self.initial_midpoint_y + self.Y_SHIFT_TOLERANCE:.4f}), ocorreu no segundo: {timestamp_ms/1000:.2f}")


            else:
                self.consecutive_foot_pronation_error_counter = 0

            if self.consecutive_foot_pronation_error_counter >= self.FOOT_PRONATION_ERROR_THRESHOLD:
                self.total_foot_pronation_error_counter += 1
                self.consecutive_foot_pronation_error_counter = 0


                
        except KeyError as e:
            print(f"Erro: O ponto anatômico {e} não foi encontrado no dicionário (KeyError).")
            self.consecutive_foot_pronation_error_counter = 0
        except Exception as e:
            print(f"Erro inesperado ao calcular pronação do pé por ponto médio Y: {e}")
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