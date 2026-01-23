import numpy as np

from ...vector_calculator import VectorCalculator
from .base_frontal import BaseFrontal


# TODO Terminar os calculos de cada parte, testar eles

class LeftFrontal(BaseFrontal):
    
    def create_dictionary_landmarks(self, lm_obj):
        try:
            return {
                'right_hip_x': lm_obj[24].x, 'right_hip_y': lm_obj[24].y,
                'left_hip_x': lm_obj[23].x, 'left_hip_y': lm_obj[23].y,
                'left_knee_x': lm_obj[25].x, 'left_knee_y': lm_obj[25].y,
                'left_ankle_x': lm_obj[27].x, 'left_ankle_y': lm_obj[27].y,
                'left_big_toe_x': lm_obj[31].x, 'left_big_toe_y': lm_obj[31].y,
                'left_ear_y': lm_obj[7].y, 
                'left_heel_y': lm_obj[29].y,
            }
        except Exception as e:
            print(f"Erro ao criar dicionário de landmarks: {e}")
            return {}

    def _detect_repetition_phase(self, dict_lm, ts):
        ear_y = dict_lm['left_ear_y']
        heel_y = dict_lm['left_heel_y']
        big_toe_y = dict_lm['left_big_toe_y']

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
                                
            else:
                self.ear_y_history.append(ear_y)
                self.heel_y_history.append(heel_y) 
                self.big_toe_y_history.append(big_toe_y)
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
            
            #if (timestamp_ms/1000 > 3 and timestamp_ms/1000 <= 5) or (timestamp_ms/1000 > 7 and timestamp_ms/1000 <= 9) or (timestamp_ms/1000 > 10 and timestamp_ms/1000 <= 12):
            #    print(f"Angulo atual quadil é de {angle_deg:.2f} e ocorreu no segundo {timestamp_ms/1000:.2f} limite = {self.HIP_ANGLE_TOLERANCE - 0.5:.2f}")
            
            if angle_deg < self.HIP_ANGLE_TOLERANCE - 0.5:
                #if (self.repetitions_detected != 1):
                #    print(f"Repetição {(self.repetitions_detected+1)}: Ângulo atual quadril é de {angle_deg:.2f} e ocorreu esse erro no segundo {timestamp_ms/1000:.2f}. Limite = {self.HIP_ANGLE_TOLERANCE - 0.5:.2f}")
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
            x1 = dict_lm['left_hip_x']
            y1 = dict_lm['left_hip_y']
            x2 = dict_lm['left_knee_x']
            y2 = dict_lm['left_knee_y']
            x3 = dict_lm['left_ankle_x']
            y3 = dict_lm['left_ankle_y']
            
            angle_hka = VectorCalculator.calculate_angle_3p(x1, y1, x2, y2, x3, y3)
            
        
            print(f"Repetição {(self.repetitions_detected+1)}: Angulo atual joelho é de {angle_hka:.2f} e ocorreu no segundo {timestamp_ms/1000:.2f}, limite é de {self.KNEE_ANGLE_MIN}")

            if angle_hka > 0:
                #if (self.repetitions_detected == 0):
                #    print(f"Angulo atual joelho é de {angle_hka:.2f} e ocorreu no segundo {timestamp_ms/1000:.2f}")
                self.consecutive_knee_valgus_error_counter += 1
                kn_valgus_status = 1
            else:
                self.consecutive_knee_valgus_error_counter = 0

            if self.consecutive_knee_valgus_error_counter >= self.KNEE_VALGUS_ERROR_THRESHOLD:
                self.total_knee_valgus_error_counter += 1
                self.consecutive_knee_valgus_error_counter = 0


                
        except Exception as e:
            print(f"Erro ao calcular valgo de joelho: {e}")
            self.consecutive_knee_valgus_error_counter = 0
            
        return kn_valgus_status

    def _check_foot_pronation_error(self, dict_lm, timestamp_ms):
        foot_pronation_status = 0
    
    
        if self.initial_midpoint_y is None:
            return 0 
        
        try:
            heel_y = dict_lm['left_heel_y']
            big_toe_y = dict_lm['left_big_toe_y']
            
            current_midpoint_y = (heel_y + big_toe_y) / 2
            
            #print(f"Calcanhar Ponto médio atual: {current_midpoint_y:.4f} (Limite: {self.initial_midpoint_y + self.Y_SHIFT_TOLERANCE:.4f}), ocorreu no segundo: {timestamp_ms/1000:.2f}")

    
            if current_midpoint_y < self.initial_midpoint_y - self.Y_SHIFT_TOLERANCE:
                self.consecutive_foot_pronation_error_counter += 1
                foot_pronation_status = 1

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
  