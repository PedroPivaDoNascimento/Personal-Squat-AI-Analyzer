import math
import numpy as np
from mediapipe import solutions

class SquatRepetitionAnalyzer:
    def __init__(self, 
                 descent_threshold=0.05, 
                 ascent_return_threshold=0.02,
                 trunk_error_threshold=5,
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
        self.repetitions_detected = 0
        self.current_phase = 'inicial'
        self.min_y_in_rep = None

        # Contadores de erros para cada repetição
        self.trunk_error_counter = 0
        self.knee_error_counter = 0
        self.head_error_counter = 0
        self.foot_error_counter = 0

        # Históricos de erros de todas as repetições
        self.trunk_error_history = []
        self.knee_error_history = []
        self.head_error_history = []
        self.foot_error_history = []

        # Estrutura para armazenar os resultados de cada repetição
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
        
        self._detect_repetition_phase(ear_y, timestamp_ms)
        
        hp, tr, hl, kn = self._check_errors(landmarks_obj) 
        
        return hp, tr, hl, kn 

    def _detect_repetition_phase(self, ear_y, ts):
        if self.ear_y_inicial is None: # Se ainda não calibramos a posição inicial
            if len(self.ear_y_history) >= 10: # Se já coletamos 10 ou mais pontos
                self.ear_y_inicial = np.mean(self.ear_y_history[-10:])  # Calcula a média dos últimos 10 pontos como o ponto inicial
            else: # Senão, continua coletando pontos
                self.ear_y_history.append(ear_y)
                return # Sai da função até ter pontos suficientes para calibrar
        
        self.ear_y_history.append(ear_y) # Adiciona o ear_y do frame atual ao histórico.

        if self.current_phase == 'inicial':
            self._reset_error_counters() # Reseta os contadores de erro para uma nova repetição
            if ear_y > self.ear_y_inicial * (1 + self.DESCENT_THRESHOLD):
                self.current_phase = 'descendo'
                self.min_y_in_rep = ear_y
                
        elif self.current_phase == 'descendo':
            if ear_y > self.min_y_in_rep: 
                self.min_y_in_rep = ear_y
            
            if ear_y < self.min_y_in_rep * 0.98: # Verifica se o usuário está subindo e adiconei um pequeno limite para que o usuário tenha q subir 2% a mais do mínimo para evitar erros 
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
            'right_heel_y': lm_obj[30].y
        }

    def _check_head_posture_error(self, dict_lm):
        """
        Verifica o erro de postura da cabeça (Plano de Frankfurt).
        Retorna 1 se houver erro e incrementa o contador, 0 caso contrário.
        """

        hp_status = 0
        try:
            # Calcula o ângulo da cabeça usando a posição do olho e da orelha
            head_angle_rad = math.atan2(dict_lm['right_ear_y'] - dict_lm['right_eye_y'], dict_lm['right_ear_x'] - dict_lm['right_eye_x'])
            head_angle_deg = math.degrees(head_angle_rad)
            
            # Define a tolerância
            tolerance_deg = 5
            
            # O ângulo deve estar próximo de 0 graus OU próximo de 180/-180 graus.
            is_aligned_horizontal = (
                abs(head_angle_deg) <= tolerance_deg or # Próximo de 0 graus
                abs(abs(head_angle_deg) - 180) <= tolerance_deg # Próximo de 180 ou -180 graus
            )
            
            # Se NÃO estiver alinhado horizontalmente, é um erro.
            if not is_aligned_horizontal:
                self.head_error_counter += 1
                hp_status = 1
        except Exception as e:
            print(f"Erro específico no cálculo da cabeça: {e}")
        return hp_status

    def _check_trunk_flexion_error(self, dict_lm):
        """
        Verifica o erro de excesso de flexão do tronco em relação à tíbia.
        Retorna 1 se houver erro e incrementa o contador, 0 caso contrário.
        """
        tr_status = 0
        try:
            trunk_angle_rad = math.atan2(dict_lm['right_hip_y'] - dict_lm['right_shoulder_y'], dict_lm["right_hip_x"] - dict_lm['right_shoulder_x'])
            trunk_angle_deg = abs(math.degrees(trunk_angle_rad))
            
            tibia_angle_rad = math.atan2(dict_lm['right_ankle_y'] - dict_lm['right_knee_y'], dict_lm['right_ankle_x'] - dict_lm['right_knee_x'])
            tibia_angle_deg = abs(math.degrees(tibia_angle_rad))

            if trunk_angle_deg < tibia_angle_deg:
                self.trunk_error_counter += 1
                tr_status = 1
        except Exception as e:
            print(f"Erro específico no cálculo do tronco: {e}")
        return tr_status

    def _check_knee_translation_error(self, dict_lm):
        """
        Verifica o erro de translação excessiva do joelho em relação ao pé.
        Retorna 1 se houver erro e incrementa o contador, 0 caso contrário.
        """
        kn_status = 0
        try:
            foot_length_x = abs(dict_lm['right_big_toe_x'] - dict_lm['right_heel_x'])
            allowed_forward_translation = foot_length_x * 0.30
            
            if dict_lm['right_knee_x'] > dict_lm['right_big_toe_x'] + allowed_forward_translation:    
                self.knee_error_counter += 1
                kn_status = 1
        except Exception as e:
            print(f"Erro específico no cálculo do joelho (translação do pé): {e}")
        return kn_status
    
    def _check_heel_lift_error(self, dict_lm):
        """
        Verifica o erro de elevação do calcanhar.
        Retorna 1 se houver erro e incrementa o contador, 0 caso contrário.
        """
        hl_status = 0
        try:
            if (dict_lm['right_ankle_y'] - dict_lm["right_heel_y"]) > 0.01:
                self.foot_error_counter += 1
                hl_status = 1
        except Exception as e:
            print(f"Erro específico no cálculo do calcanhar: {e}")
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
            
            # 1. ERRO DE POSTURA DA CABEÇA (HEAD POSTURE - Plano de Frankfurt)
            hp_status = self._check_head_posture_error(dict_lm)

            # 2. ERRO DE TRONCO (TRUNK) - Excesso de Flexão em relação à Tíbia
            tr_status = self._check_trunk_flexion_error(dict_lm)

            # 3. ERRO DE JOELHO (KNEE) - Translação Excessiva em relação ao Pé
            kn_status = self._check_knee_translation_error(dict_lm)
                    
            # 4. ERRO DE ELEVAÇÃO DO CALCANHAR (HEEL LIFT)
            hl_status = self._check_heel_lift_error(dict_lm)
         
        return hp_status, tr_status, hl_status, kn_status

    def _reset_error_counters(self):
        self.trunk_error_counter = 0
        self.knee_error_counter = 0
        self.head_error_counter = 0
        self.foot_error_counter = 0

    def _complete_repetition(self, current_ts):
        if self.repetitions_detected < 3: 
            trunk_rep_result = 1 if self.trunk_error_counter > self.TRUNK_ERROR_THRESHOLD else 0
            knee_rep_result = 1 if self.knee_error_counter > self.KNEE_ERROR_THRESHOLD else 0
            head_rep_result = 1 if self.head_error_counter > self.HEAD_ERROR_THRESHOLD else 0
            foot_rep_result = 1 if self.foot_error_counter > self.FOOT_ERROR_THRESHOLD else 0
            
            self.trunk_error_history.append(int(self.trunk_error_counter))
            self.knee_error_history.append(int(self.knee_error_counter))
            self.head_error_history.append(int(self.head_error_counter))
            self.foot_error_history.append(int(self.foot_error_counter))
            
            self.reps['trunk'].append(trunk_rep_result)
            self.reps['knee'].append(knee_rep_result)
            self.reps['head'].append(head_rep_result)
            self.reps['heel'].append(foot_rep_result)
            
            self.repetitions_detected += 1
            self.repetition_timestamps.append(current_ts / 1000)
            
    def finalize_analysis(self):  
        if self.repetitions_detected == 0 and self.current_phase != 'inicial':
            print("Nenhuma repetição completa detectada neste vídeo. Preenchendo slots com 'None'.")
            # Preenche todos os 3 slots com 'None' se nada foi detectado
            for i in range(3):
                for key in ['head', 'trunk', 'heel', 'knee']:
                    self.reps[key].append(None)
                self.repetition_timestamps.append(None)
                self.trunk_error_history.append(None)
                self.knee_error_history.append(None)
                self.head_error_history.append(None)
                self.foot_error_history.append(None)
                print(f"  Slot para Repetição {i+1} preenchido com 'None'.")
        else:
            # Se houve repetições detectadas, preenche os slots restantes até 3.
            num_detected = self.repetitions_detected
            if num_detected < 3:
                print(f"{num_detected} repetição(ões) completa(s) detectada(s). Preenchendo slots restantes com 'None'.")
            
            for i in range(num_detected, 3): # Começa do número de repetições detectadas
                for key in ['head', 'trunk', 'heel', 'knee']:
                    self.reps[key].append(None) 
                self.repetition_timestamps.append(None)
                self.trunk_error_history.append(None)
                self.knee_error_history.append(None)
                self.head_error_history.append(None)
                self.foot_error_history.append(None)
                print(f"  Slot para Repetição {i+1} preenchido com 'None'.")
    