import numpy as np
import os
import joblib
import cv2 as cv

from ...vector_calculator import VectorCalculator
from .base_frontal import BaseFrontal



class LeftFrontal(BaseFrontal):
    
    def __init__(self, options_marcadas, person_name, side, descent_threshold=0.05, ascent_return_threshold=0.02, hip_error_threshold=5, knee_valgus_error_threshold=5,
                 foot_pronation_error_threshold=5):
        self.history_whites_pixels = []
        self.history_whites_pixels_with_frame = {}
        self.sequential_frame_counter = 0  # Contador sequencial para frames válidos (1, 2, 3...)
        super().__init__(options_marcadas=options_marcadas, person_name=person_name, side=side, descent_threshold=descent_threshold, 
                         ascent_return_threshold=ascent_return_threshold, hip_error_threshold=hip_error_threshold, knee_valgus_error_threshold=knee_valgus_error_threshold,
                         foot_pronation_error_threshold=foot_pronation_error_threshold)

    def create_dictionary_landmarks(self, lm_obj):
        """
        Cria um dicionário com as seguintes chaves:
        - right_hip_x: coordenada x do quadril do pé direito
        - right_hip_y: coordenada y do quadril do pé direito
        - left_hip_x: coordenada x do quadril do pé esquerdo
        - left_hip_y: coordenada y do quadril do pé esquerdo
        - left_knee_x: coordenada x do joelho esquerdo
        - left_knee_y: coordenada y do joelho esquerdo
        - left_ankle_x: coordenada x do tornozelo esquerdo
        - left_ankle_y: coordenada y do tornozelo esquerdo
        - left_big_toe_x: coordenada x do dedo do pé esquerdo
        - left_big_toe_y: coordenada y do dedo do pé esquerdo
        - left_ear_y: coordenada y da orelha esquerda
        - left_heel_y: coordenada y do calcanhar esquerdo
        - left_heel_x: coordenada x do calcanhar esquerdo
        """
        try:
            return {
                'right_hip_x': lm_obj[24].x, 'right_hip_y': lm_obj[24].y,
                'left_hip_x': lm_obj[23].x, 'left_hip_y': lm_obj[23].y,
                'left_knee_x': lm_obj[25].x, 'left_knee_y': lm_obj[25].y,
                'left_ankle_x': lm_obj[27].x, 'left_ankle_y': lm_obj[27].y,
                'left_big_toe_x': lm_obj[31].x, 'left_big_toe_y': lm_obj[31].y,
                'left_ear_y': lm_obj[7].y, 
                'left_heel_y': lm_obj[29].y,     'left_heel_x': lm_obj[29].x
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

    def _get_foot_data(self, dict_lm):
        """
        Retorna um dicionário com as seguintes chaves:
        - left_ankle_x: coordenada x do tornozelo esquerdo
        - left_ankle_y: coordenada y do tornozelo esquerdo
        - left_big_toe_x: coordenada x do dedo do pé esquerdo
        - left_big_toe_y: coordenada y do dedo do pé esquerdo
        - left_heel_x: coordenada x do calcanhar esquerdo
        - left_heel_y: coordenada y do calcanhar esquerdo
        """
        
        return {
            'ankle_x': dict_lm['left_ankle_x'],
            'ankle_y': dict_lm['left_ankle_y'],
            'big_toe_x': dict_lm['left_big_toe_x'],
            'big_toe_y': dict_lm['left_big_toe_y'],
            'heel_x': dict_lm['left_heel_x'],
            'heel_y': dict_lm['left_heel_y']
        }
    
    def _get_center_point_foot(self, dict_lm):
        mean_x = np.mean([dict_lm['left_ankle_x'], dict_lm['left_big_toe_x'], dict_lm['left_heel_x']]) 
        mean_y = np.mean([dict_lm['left_ankle_y'], dict_lm['left_big_toe_y'], dict_lm['left_heel_y']]) 
        return mean_x, mean_y

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
            
        
            #print(f"Repetição {(self.repetitions_detected+1)}: Angulo atual joelho é de {angle_hka:.2f} e ocorreu no segundo {timestamp_ms/1000:.2f}, limite é de {self.KNEE_ANGLE_MIN}")

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

    def _should_sample_data(self, timestamp_ms, interval_ms=500):
        """
        Garante a captura do PRIMEIRO frame encontrado (seja ele qual for)
        e mantém o intervalo de 500ms a partir dele.
        """
        # Se a variável não existir, criamos como None
        if not hasattr(self, 'last_save_ts'):
            self.last_save_ts = None 

        # É o primeiríssimo frame que o código encontra
        if self.last_save_ts is None:
            self.last_save_ts = timestamp_ms
            return True

        # Já salvamos algo, agora conferimos se passou o intervalo
        if timestamp_ms >= self.last_save_ts + interval_ms:
            self.last_save_ts = timestamp_ms
            return True
        
        return False

    def _check_foot_pronation_error(self, dict_lm, timestamp_ms, frame, frame_number):
        """Método responsável por cálculo a pronação do pé

        Args:
            dict_lm (Dict[str, Any]): Dicionário com os landmarks
            timestamp_ms (float): Timestamp em milissegundos
            frame (int): Frame
            frame_number (int): Número do frame atual

        Returns:
            int: Status da pronação do pé
        """
        # Extrair os dados brutos
        mean_x, mean_y = self._get_center_point_foot(dict_lm)
        cut_frame = self.video_processor.crop_roi(frame, mean_x, mean_y)
        num_withed_pixels = self.video_processor.count_white_pixels(cut_frame)

        # Setar o status da pronação com a antiga variação, verificação usada para caso o frame não esteja no intervalo de 500 ms
        foot_pronation_status = getattr(self, "current_foot_status", 0)

        # Verificar se o timestamp deve ser salvo
        if self._should_sample_data(timestamp_ms, interval_ms=500):
            
            # Incrementa o contador sequencial para obter o próximo número de frame ordinal (1, 2, 3...)
            self.sequential_frame_counter += 1
            
            # Ações de gravação - salva usando o contador sequencial como chave (frame1, frame2, frame3...)
            self.history_whites_pixels_with_frame[self.sequential_frame_counter] = num_withed_pixels
            self.history_whites_pixels.append(num_withed_pixels)

            # Se não houver histórico suficiente para comparar, encerramos a análise deste frame
            if len(self.history_whites_pixels) <= 1:
                return foot_pronation_status

            # Ações de cálculo matemático
            relative_increase_white_pixels = 0
            try:
                ultimo = num_withed_pixels
                penultimo = self.history_whites_pixels[-2]
                relative_increase_white_pixels = (ultimo - penultimo) / penultimo
            except ZeroDivisionError:
                pass

            # Ações de atualização de contadores de erro
            if 0.6 < relative_increase_white_pixels < 3:
                self.consecutive_foot_pronation_error_counter += 1
                foot_pronation_status = 1
            else:   
                self.consecutive_foot_pronation_error_counter = 0

            if self.consecutive_foot_pronation_error_counter >= 1:
                self.total_foot_pronation_error_counter += 1
                self.consecutive_foot_pronation_error_counter = 0
            
            # Guardamos o status atualizado para os frames "vazios" seguintes
            self.current_foot_status = foot_pronation_status

        # Retorna o status (se estiver fora do IF de 500ms, ele retorna o último valor calculado)
        return foot_pronation_status