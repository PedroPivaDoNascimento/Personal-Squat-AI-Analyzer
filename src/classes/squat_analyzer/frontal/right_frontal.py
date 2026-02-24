import numpy as np
import joblib
import os

from ...vector_calculator import VectorCalculator
from .base_frontal import BaseFrontal
from classes.excel.foot_data_excel_writer import FootDataExcelWriter
class RightFrontal(BaseFrontal):
    
    def create_dictionary_landmarks(self, lm_obj):
        try:
            return {
                'right_hip_x': lm_obj[24].x, 'right_hip_y': lm_obj[24].y,
                'left_hip_x': lm_obj[23].x, 'left_hip_y': lm_obj[23].y,
                'right_knee_x': lm_obj[26].x, 'right_knee_y': lm_obj[26].y,
                'right_ankle_x': lm_obj[28].x, 'right_ankle_y': lm_obj[28].y,
                'right_big_toe_x': lm_obj[32].x, 'right_big_toe_y': lm_obj[32].y,
                'right_ear_y': lm_obj[7].y,
                'right_heel_y': lm_obj[30].y,    'right_heel_x': lm_obj[30].x
            }
        except Exception as e:
            print(f"Erro ao criar dicionário de landmarks: {e}")
            return {}

    def _detect_repetition_phase(self, dict_lm, ts):
        ear_y = dict_lm['right_ear_y']
        heel_y = dict_lm['right_heel_y']
        big_toe_y = dict_lm['right_big_toe_y']

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

    def _get_foot_data(self, dict_lm):
        """
        Retorna um dicionário com as seguintes chaves:
        - right_ankle_x: coordenada x do tornozelo direito
        - right_ankle_y: coordenada y do tornozelo direito
        - right_big_toe_x: coordenada x do dedo do pé direito
        - right_big_toe_y: coordenada y do dedo do pé direito
        - right_heel_x: coordenada x do calcanhar direito
        - right_heel_y: coordenada y do calcanhar direito
        """

        return {
           'ankle_x': dict_lm['right_ankle_x'],
           'ankle_y': dict_lm['right_ankle_y'],
           'big_toe_x': dict_lm['right_big_toe_x'],
           'big_toe_y': dict_lm['right_big_toe_y'],
           'heel_x': dict_lm['right_heel_x'],
           'heel_y': dict_lm['right_heel_y']
        }
        
    def _check_hip_tilt_error(self, dict_lm, timestamp_ms):
            hip_status = 0
            try:
                x1, y1 = dict_lm['left_hip_x'], dict_lm['left_hip_y']
                x2, y2 = dict_lm['right_hip_x'], dict_lm['right_hip_y']

                angle_deg = VectorCalculator.angle_to_horizontal(x1, y1, x2, y2)
                

                if angle_deg > self.HIP_ANGLE_TOLERANCE:
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
            
            if angle_hka < 0:
                self.consecutive_knee_valgus_error_counter += 1
                kn_valgus_status = 1
                
                # Verificando o tamanho do contador entre os intervalos de 5 e 12
                if self.consecutive_knee_valgus_error_counter >= 5 and self.consecutive_knee_valgus_error_counter <= 12:
                    print(f" Repetição {(self.repetitions_detected+1)}: ocorreu no segundo {timestamp_ms/1000:.2f} e o valor atual do contador é de {self.consecutive_knee_valgus_error_counter}")

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

    def _check_foot_pronation_error(self):
        foot_data_excel_writer = FootDataExcelWriter(self.repetitions_detected, self.foot_repeat_data, self.person_name, "frontal", self.side)
        static_data = foot_data_excel_writer.convert_data_to_statistic_pandas()
        X = static_data.iloc[:, 2:].values

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../../../../models/modelo_pe_frontal_direito.pkl")
        model = joblib.load(model_path)

        y_pred = model.predict(X)
        foot_pronation_status = y_pred[0]
        
        return foot_pronation_status
        
  