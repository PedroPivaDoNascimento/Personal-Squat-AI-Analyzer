import math
import numpy as np
from mediapipe import solutions # Necessário para acessar PoseLandmark

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
        
        # Variável para armazenar a diferença inicial da cabeça e do ombro
        self.head_initial_diff = None

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
        
        if self.head_initial_diff is None:
            self.head_initial_diff = landmarks_obj[solutions.pose.PoseLandmark.RIGHT_SHOULDER].y - landmarks_obj[solutions.pose.PoseLandmark.RIGHT_EAR].y

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


    def _check_errors(self, lm_obj):
        hp_status = tr_status = hl_status = kn_status = 0

        if self.current_phase in ['descendo', 'subindo']:
            try:
                # Landmarks para o cálculo da inclinação do tronco
                shoulder_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_SHOULDER].y
                shoulder_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_SHOULDER].x
                hip_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_HIP].y
                hip_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_HIP].x
                
                # Landmarks para o cálculo do alinhamento do joelho (alguns já foram pegos acima)
                knee_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_KNEE].x
                ankle_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_ANKLE].x
                
                # Landmarks para o cálculo da postura da cabeça
                right_ear_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_EAR].y

                # Landmarks para o cálculo da elevação do calcanhar
                heel_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_HEEL].y
                ankle_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_ANKLE].y
                
            except Exception as e:
                print(f"Erro ao acessar landmarks para cálculo de erros: {e}. Análise de erros ignorada para este frame.")
                return hp_status, tr_status, hl_status, kn_status
            
            # ERRO DE TRONCO (TRUNK) - Inclinação
            try:
                # Calcula a inclinação da linha que vai do seu ombro ao quadril.
                angle_rad = math.atan2(hip_y - shoulder_y, hip_x - shoulder_x)
                angle_deg = abs(math.degrees(angle_rad))
                
                if not (70 <= angle_deg <= 110):
                    self.trunk_error_counter += 1 
                    tr_status = 1
            except Exception as e:
                print(f"Erro específico no cálculo do tronco: {e}")
                pass 

            # ERRO DE JOELHO (KNEE) - Alinhamento Horizontal
            try:
                margin_x = 0.03 # Uma pequena margem de tolerância.
                # Verifica se o joelho está alinhado entre o quadril e o tornozelo.
                if not (min(hip_x, ankle_x) - margin_x <= knee_x <= max(hip_x, ankle_x) + margin_x):
                    self.knee_error_counter += 1 
                    kn_status = 1 
            except Exception as e:
                print(f"Erro específico no cálculo do joelho: {e}")
                pass

            # ERRO DE POSTURA DA CABEÇA (HEAD POSTURE)
            try:
                # Calcula a diferença de altura entre o ombro e a orelha neste frame.
                current_head_diff = shoulder_y - right_ear_y
                
                # Verifica se a diferença de altura da cabeça está dentro de um intervalo aceitável.
                if self.head_initial_diff is not None and \
                (current_head_diff < self.head_initial_diff * 0.9 or current_head_diff > self.head_initial_diff * 1.1):
                    self.head_error_counter += 1 
                    hp_status = 1
            except Exception as e:
                print(f"Erro específico no cálculo da cabeça: {e}")
                pass
            
            # ERRO DE ELEVAÇÃO DO CALCANHAR (HEEL LIFT)
            try:
                # Verifica se o calcanhar está levantado em relação ao tornozelo.
                if (ankle_y - heel_y) > 0.01:
                    self.foot_error_counter += 1 
                    hl_status = 1
            except Exception as e:
                print(f"Erro específico no cálculo do calcanhar: {e}")
                pass
        
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