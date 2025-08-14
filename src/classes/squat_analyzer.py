import math
import numpy as np
from mediapipe import solutions

class SquatRepetitionAnalyzer:
    def __init__(self, 
                 descent_threshold=0.05, 
                 ascent_return_threshold=0.02,
                 trunk_error_threshold=5, # O erro só é contado se ocorrer por 5 frames seguidos
                 knee_error_threshold=5,
                 head_error_threshold=5,
                 foot_error_threshold=5):
        
        self.DESCENT_THRESHOLD = descent_threshold 
        self.ASCENT_RETURN_THRESHOLD = ascent_return_threshold 
        self.TRUNK_ERROR_THRESHOLD = trunk_error_threshold
        self.KNEE_ERROR_THRESHOLD = knee_error_threshold
        self.HEAD_ERROR_THRESHOLD = head_error_threshold
        self.FOOT_ERROR_THRESHOLD = foot_error_threshold

        self.ear_y_inicial = None
        self.ear_y_history = []

        self.heel_y_inicial = None
        self.heel_y_history = []

        #As variaveis iniciais do programa, visando evitar os erros das landmarks sairem de um ponto para o outro
        self.ankle_x_inicial = None
        self.ankle_x_history = []
        self.knee_x_inicial = None
        self.knee_x_history = []


        self.repetitions_detected = 0
        self.current_phase = 'inicial'
        self.min_y_in_rep = None

        # Contadores de ERROS CONSECUTIVOS
        self.consecutive_trunk_error_counter = 0
        self.consecutive_knee_error_counter = 0
        self.consecutive_head_error_counter = 0
        self.consecutive_foot_error_counter = 0

        # Contadores de ERROS TOTAIS para a repetição atual
        self.total_trunk_error_counter = 0
        self.total_knee_error_counter = 0
        self.total_head_error_counter = 0
        self.total_foot_error_counter = 0

        # Históricos de ERROS TOTAIS de todas as repetições (para o relatório final)
        self.trunk_error_history = []
        self.knee_error_history = []
        self.head_error_history = []
        self.foot_error_history = []

        # Estrutura para armazenar os resultados de cada repetição (0 ou 1)
        self.reps = {'head': [], 'trunk': [], 'heel': [], 'knee': []}

        # Lista para armazenar os timestamps de finalização de cada repetição
        self.repetition_timestamps = []
        
    def process_frame_landmarks(self, landmarks_obj, timestamp_ms): 
        """ hp: Significa Head Posture (Postura da Cabeça).
            tr: Significa Trunk (Tronco).
            hl: Significa Heel Lift (Elevação do Calcanhar).
            kn: Significa Knee (Joelho)."""
        
        hp = tr = hl = kn = 0 
        
        if not landmarks_obj: 
            print("!!!!!!!!!!Nenhum landmark detectado.!!!!!!!!!")
            return hp, tr, hl, kn 

        ear_y = landmarks_obj[solutions.pose.PoseLandmark.RIGHT_EAR].y
        heel_y = landmarks_obj[solutions.pose.PoseLandmark.RIGHT_HEEL].y 
        
        self._detect_repetition_phase(ear_y, heel_y, timestamp_ms)
        
        hp, tr, hl, kn = self._check_errors(landmarks_obj) 
        
        return hp, tr, hl, kn 

    def _detect_repetition_phase(self, ear_y, heel_y, ts):
        if self.ear_y_inicial is None and self.heel_y_inicial is None and self.knee_x_inicial is None and self.ankle_x_inicial is None: # Se ainda não calibramos a posição inicial
            if len(self.ear_y_history) >= 10 and len(self.heel_y_history) >= 10 and len(self.knee_x_history) >= 10 and len(self.ankle_x_history) >= 10: # Se já coletamos 10 ou mais pontos
                self.ear_y_inicial = np.mean(self.ear_y_history[-10:])
                self.heel_y_inicial = np.mean(self.heel_y_history[-10:])
                self.knee_x_inicial = np.mean(self.knee_x_history[-10:])
                self.ankle_x_inicial = np.mean(self.ankle_x_history[-10:])
            else: # Senão, continua coletando pontos
                self.ear_y_history.append(ear_y)
                self.heel_y_history.append(heel_y)
                self.knee_x_history.append(ear_y)
                self.ankle_x_history.append(heel_y)
                return 
        
        self.ear_y_history.append(ear_y)

        if self.current_phase == 'inicial':
            self._reset_error_counters() 
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

    def create_dictionary_landmarks(self, lm_obj):
        """
        Extrai as coordenadas das landmarks essenciais e as armazena em um dicionário.
        """
        return {
            'right_shoulder_x': lm_obj[12].x,
            'left_shoulder_x': lm_obj[11].x,
            'right_hip_x': lm_obj[24].x,
            'right_knee_x': lm_obj[26].x,
            'right_ankle_x': lm_obj[28].x,
            'right_eye_x': lm_obj[5].x,
            'right_ear_x': lm_obj[7].x,
            'right_big_toe_x': lm_obj[32].x,
            'right_heel_x': lm_obj[30].x,
            'right_shoulder_y': lm_obj[12].y,
            'right_hip_y': lm_obj[24].y,
            'right_knee_y': lm_obj[26].y,
            'right_ankle_y': lm_obj[28].y,
            'right_eye_y': lm_obj[5].y,
            'right_ear_y': lm_obj[7].y,
            'right_big_toe_y': lm_obj[32].y,
            'right_heel_y': lm_obj[30].y,
            'nose_x': lm_obj[0].x,
            'nose_y': lm_obj[0].y
        }

    def position_validation(self, dict_lm, name_body_part):
        """
        Valida a posição de uma parte do corpo não de deslocou muito em relação a posição inicial.
        Retorna True se a posição estiver dentro dos limites aceitáveis, False caso contrário.
        """
        
        if name_body_part == 'ankle':
            if abs(dict_lm['right_ankle_x'] - self.ankle_x_inicial > 20):
                return False
            return True
        elif name_body_part == 'knee':
            if abs(dict_lm['right_knee_x'] - self.knee_x_inicial > 20):
                return False
            return True
            
        
        return True


    def _check_head_posture_error(self, dict_lm):
        """
        Verifica o erro de postura da cabeça comparando a posição horizontal do nariz
        com a linha dos ombros. Esta abordagem é mais robusta para detectar
        a "cabeça para frente" (forward head posture).

        Parâmetros:
        dict_lm: Um dicionário contendo os pontos de referência (landmarks) do corpo.

        Retorna:
        hp_status (int): 1 se um erro de postura da cabeça for detectado, 0 caso contrário.
        """
        hp_status = 0
        try:
            # Obter as coordenadas x dos ombros e do nariz.
            ombro_esquerdo_x = dict_lm['left_shoulder_x']
            ombro_direito_x = dict_lm['right_shoulder_x']
            nariz_x = dict_lm['nose_x']

            # Calcular o ponto médio horizontal entre os ombros.
            ponto_medio_ombros_x = (ombro_esquerdo_x + ombro_direito_x) / 2
            
            # Definir uma tolerância horizontal.
            TOLERANCIA_DESLOCAMENTO_HORIZONTAL = 0.05
            
            # Calcular o deslocamento horizontal do nariz em relação ao ponto médio dos ombros.
            deslocamento_x = abs(nariz_x - ponto_medio_ombros_x)

            if deslocamento_x > TOLERANCIA_DESLOCAMENTO_HORIZONTAL:
                self.consecutive_head_error_counter += 1
                hp_status = 1
            else:
                self.consecutive_head_error_counter = 0

            # Manter a lógica de contadores consecutivos.
            if self.consecutive_head_error_counter >= self.HEAD_ERROR_THRESHOLD:
                self.total_head_error_counter += 1
                self.consecutive_head_error_counter = 0

        except Exception as e:
            print(f"Erro no cálculo alternativo da cabeça: {e}")
            self.consecutive_head_error_counter = 0
            
        return hp_status

    def _check_trunk_flexion_error(self, dict_lm):
        """
        Verifica o erro de excesso de flexão do tronco e atualiza os contadores,
        agora com uma tolerância para evitar falsos positivos.
        
        A tolerância permite uma pequena diferença entre o ângulo do tronco e da tíbia
        sem que isso seja considerado um erro.
        """
        tr_status = 0
        try:
            # Calcula o ângulo do tronco em graus.
            trunk_angle_rad = math.atan2(dict_lm['right_hip_y'] - dict_lm['right_shoulder_y'], dict_lm["right_hip_x"] - dict_lm['right_shoulder_x'])
            trunk_angle_deg = abs(math.degrees(trunk_angle_rad))
            
            # Calcula o ângulo da tíbia em graus.
            tibia_angle_rad = math.atan2(dict_lm['right_ankle_y'] - dict_lm['right_knee_y'], dict_lm['right_ankle_x'] - dict_lm['right_knee_x'])
            tibia_angle_deg = abs(math.degrees(tibia_angle_rad))

            # A condição de erro é se o ângulo do tronco for significativamente menor.
            if trunk_angle_deg < tibia_angle_deg and self.position_validation(dict_lm, 'knee') and self.position_validation(dict_lm, 'ankle'):
                self.consecutive_trunk_error_counter += 1
                tr_status = 1
            else:
                self.consecutive_trunk_error_counter = 0

            # Atualiza os contadores de erro total se o erro for consecutivo.
            if self.consecutive_trunk_error_counter >= self.TRUNK_ERROR_THRESHOLD:
                self.total_trunk_error_counter += 1
                self.consecutive_trunk_error_counter = 0
        except Exception as e:
            print(f"Erro específico no cálculo do tronco: {e}")
            self.consecutive_trunk_error_counter = 0
        return tr_status

    def _check_knee_translation_error(self, dict_lm):
        """
        Verifica o erro de translação excessiva do joelho e atualiza os contadores.
        """
        kn_status = 0
        try:
            foot_length_x = abs(dict_lm['right_big_toe_x'] - dict_lm['right_heel_x'])
            allowed_forward_translation = foot_length_x * 0.30
            
            if dict_lm['right_knee_x'] > dict_lm['right_big_toe_x'] + allowed_forward_translation:    
                self.consecutive_knee_error_counter += 1
                kn_status = 1
            else:
                self.consecutive_knee_error_counter = 0

            if self.consecutive_knee_error_counter >= self.KNEE_ERROR_THRESHOLD:
                self.total_knee_error_counter += 1
                self.consecutive_knee_error_counter = 0
        except Exception as e:
            print(f"Erro específico no cálculo do joelho (translação do pé): {e}")
            self.consecutive_knee_error_counter = 0
        return kn_status
    
    def _check_heel_lift_error(self, dict_lm):
        """
        Verifica o erro de elevação do calcanhar e atualiza os contadores.
        """
        hl_status = 0
        try:
            if (dict_lm["right_heel_y"]) < self.heel_y_inicial: # Verifica se o calcanhar está elevado
                self.consecutive_foot_error_counter += 1
                hl_status = 1
            else:
                self.consecutive_foot_error_counter = 0

            if self.consecutive_foot_error_counter >= self.FOOT_ERROR_THRESHOLD:
                self.total_foot_error_counter += 1
                self.consecutive_foot_error_counter = 0
        except Exception as e:
            print(f"Erro específico no cálculo do calcanhar: {e}")
            self.consecutive_foot_error_counter = 0
        return hl_status

    def _check_errors(self, lm_obj):
        # Inicialização dos Status de Erro para o Frame Atual
        hp_status = tr_status = hl_status = kn_status = 0

        # Somente verifica erros se a fase atual for 'descendo' ou 'subindo'
        if self.current_phase in ['descendo', 'subindo']:
            try:
                dict_lm = self.create_dictionary_landmarks(lm_obj)
            except Exception as e:
                print(f"Erro ao acessar landmarks essenciais para cálculo de erros: {e}. Análise de erros ignorada para este frame.")
                return hp_status, tr_status, hl_status, kn_status
            
            # 1. ERRO DE POSTURA DA CABEÇA
            hp_status = self._check_head_posture_error(dict_lm)
            
            # 2. ERRO DE TRONCO
            tr_status = self._check_trunk_flexion_error(dict_lm)

            # 3. ERRO DE JOELHO
            kn_status = self._check_knee_translation_error(dict_lm)
                    
            # 4. ERRO DE ELEVAÇÃO DO CALCANHAR
            hl_status = self._check_heel_lift_error(dict_lm)
         
        return hp_status, tr_status, hl_status, kn_status

    def _reset_error_counters(self):
        # Reseta os contadores CONSECUTIVOS e TOTAIS para a nova repetição
        self.consecutive_trunk_error_counter = 0
        self.consecutive_knee_error_counter = 0
        self.consecutive_head_error_counter = 0
        self.consecutive_foot_error_counter = 0

        self.total_trunk_error_counter = 0
        self.total_knee_error_counter = 0
        self.total_head_error_counter = 0
        self.total_foot_error_counter = 0

    def _complete_repetition(self, current_ts):
        if self.repetitions_detected < 3: 
            # O resultado da repetição é 1 se o erro ocorreu pelo menos uma vez
            trunk_rep_result = 1 if self.total_trunk_error_counter > 0 else 0
            knee_rep_result = 1 if self.total_knee_error_counter > 0 else 0
            head_rep_result = 1 if self.total_head_error_counter > 0 else 0
            foot_rep_result = 1 if self.total_foot_error_counter > 0 else 0
            
            # Agora, o histórico de erros armazena o número total de vezes que o erro persistente ocorreu
            self.trunk_error_history.append(int(self.total_trunk_error_counter))
            self.knee_error_history.append(int(self.total_knee_error_counter))
            self.head_error_history.append(int(self.total_head_error_counter))
            self.foot_error_history.append(int(self.total_foot_error_counter))
            
            self.reps['trunk'].append(trunk_rep_result)
            self.reps['knee'].append(knee_rep_result)
            self.reps['head'].append(head_rep_result)
            self.reps['heel'].append(foot_rep_result)
            
            self.repetitions_detected += 1
            self.repetition_timestamps.append(current_ts / 1000)
            
    def finalize_analysis(self):  
        if self.repetitions_detected == 0 and self.current_phase != 'inicial':
            print("Nenhuma repetição completa detectada neste vídeo. Preenchendo slots com 0.")
            for i in range(3):
                for key in ['head', 'trunk', 'heel', 'knee']:
                    self.reps[key].append(0)
                self.repetition_timestamps.append(None)
                self.trunk_error_history.append(0)
                self.knee_error_history.append(0)
                self.head_error_history.append(0)
                self.foot_error_history.append(0)
                print(f"  Slot para Repetição {i+1} preenchido com 0.")
        else:
            num_detected = self.repetitions_detected
            if num_detected < 3:
                print(f"{num_detected} repetição(ões) completa(s) detectada(s). Preenchendo slots restantes com 0.")
            
            for i in range(num_detected, 3): 
                for key in ['head', 'trunk', 'heel', 'knee']:
                    self.reps[key].append(0) 
                self.repetition_timestamps.append(None)
                self.trunk_error_history.append(0)
                self.knee_error_history.append(0)
                self.head_error_history.append(0)
                self.foot_error_history.append(0)
                print(f"  Slot para Repetição {i+1} preenchido com 0.")
