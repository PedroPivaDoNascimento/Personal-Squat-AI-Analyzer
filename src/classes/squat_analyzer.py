import math
import numpy as np

# TODO Remover pontos que não estão sendo mais usados (heel_z, ankle_z, etc)
# TODO Após concertar os limites, trabalhar na organização do código
# TODO Adicionar o novo falso positivo no calcanhar
# TODO Testar nova função de cálculo do tronco

from .vector_calculator import VectorCalculator

class SquatRepetitionAnalyzer:
    def __init__(self, descent_threshold=0.05, ascent_return_threshold=0.02, trunk_error_threshold=5, knee_error_threshold=5, head_error_threshold=5, foot_error_threshold=5,):
        
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

        self.ankle_x_inicial = None
        self.ankle_x_history = []
        self.knee_x_inicial = None
        self.knee_x_history = []
        self.heel_x_inicial = None
        self.heel_x_history = []
        self.showlder_x_inicial = None
        self.showlder_x_history = []
        self.hip_x_inicial = None
        self.hip_x_history = []
        self.heel_z_inicial = None
        self.heel_z_history = []

        self.repetitions_detected = 0
        self.current_phase = 'inicial'
        self.min_y_in_rep = None

        self.consecutive_trunk_error_counter = 0
        self.consecutive_knee_error_counter = 0
        self.consecutive_head_error_counter = 0
        self.consecutive_foot_error_counter = 0

        self.total_trunk_error_counter = 0
        self.total_knee_error_counter = 0
        self.total_head_error_counter = 0
        self.total_foot_error_counter = 0

        self.trunk_error_history = []
        self.knee_error_history = []
        self.head_error_history = []
        self.foot_error_history = []

        self.reps = {'head': [], 'trunk': [], 'heel': [], 'knee': []}

        self.repetition_timestamps = []
        
    def process_frame_landmarks(self, landmarks_obj, timestamp_ms):
        hp = tr = hl = kn = 0
        
        if not landmarks_obj:
            print("!!!!!!!!!!Nenhum landmark detectado.!!!!!!!!!")
            return hp, tr, hl, kn

        dict_lm =  self.create_dictionary_landmarks(landmarks_obj)
        if dict_lm == {}:
            print("Dicionário de landmarks vazio. Análise de erros ignorada para este frame.")
            return hp, tr, hl, kn

        self._detect_repetition_phase(dict_lm, timestamp_ms)
        hp, tr, hl, kn = self._check_errors(dict_lm)
        
        return hp, tr, hl, kn

    def _detect_repetition_phase(self, dict_lm, ts):
        ear_y = dict_lm['right_ear_y']
        heel_y = dict_lm['right_heel_y']
        heel_x = dict_lm['right_heel_x']
        showlder_x = dict_lm['right_shoulder_x']
        hip_x = dict_lm['right_hip_x']
        knee_x = dict_lm['right_knee_x']
        ankle_x = dict_lm['right_ankle_x']
        heel_z = dict_lm['right_heel_z']  # ? Ainda não sei se vou usar essa variável

        
        if self.ear_y_inicial is None:
            if len(self.ear_y_history) >= 10:
                self.ear_y_inicial = np.mean(self.ear_y_history[-10:])
                
                # Calibra o "chão" usando o valor MÁXIMO da posição Y
                # para pegar o ponto mais baixo e estável do calcanhar.

                self.heel_y_inicial = np.mean(self.heel_y_history[-10:])
                
                self.knee_x_inicial = np.mean(self.knee_x_history[-10:])
                self.ankle_x_inicial = np.mean(self.ankle_x_history[-10:])
                self.heel_x_inicial = np.mean(self.heel_x_history[-10:])
                self.showlder_x_inicial = np.mean(self.showlder_x_history[-10:])
                self.hip_x_inicial = np.mean(self.hip_x_history[-10:])
                self.heel_z_inicial = np.mean(self.heel_z_history[-10:])
            else:
                self.ear_y_history.append(ear_y)
                self.heel_y_history.append(heel_y)
                self.knee_x_history.append(knee_x)
                self.ankle_x_history.append(ankle_x)
                self.heel_x_history.append(heel_x)
                self.showlder_x_history.append(showlder_x)
                self.hip_x_history.append(hip_x)
                self.heel_z_history.append(heel_z)
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
        try:
            return {
                'right_shoulder_x': lm_obj[12].x,
                'left_shoulder_x': lm_obj[11].x,
                'right_hip_x': lm_obj[24].x,
                'right_knee_x': lm_obj[26].x,
                'right_ankle_x': lm_obj[28].x,
                'right_eye_x': lm_obj[5].x,
                'left_eye_x': lm_obj[2].x,      
                'right_ear_x': lm_obj[7].x,
                'right_big_toe_x': lm_obj[32].x,
                'right_heel_x': lm_obj[30].x,
                'right_shoulder_y': lm_obj[12].y,
                'left_shoulder_y': lm_obj[11].y,
                'right_hip_y': lm_obj[24].y,
                'right_knee_y': lm_obj[26].y,
                'right_ankle_y': lm_obj[28].y,
                'right_eye_y': lm_obj[5].y,
                'left_eye_y': lm_obj[2].y,      
                'right_ear_y': lm_obj[7].y,
                'right_big_toe_y': lm_obj[32].y,
                'right_heel_y': lm_obj[30].y,
                'nose_x': lm_obj[0].x,
                'nose_y': lm_obj[0].y,
                'right_heel_z': lm_obj[30].z,
            }
        except Exception as e:
            print(f"Erro ao criar dicionário de landmarks: {e}")
            return {}

    def position_validation(self, dict_lm, name_body_part):
        if name_body_part == 'ankle':
            if dict_lm['right_ankle_x'] < self.ankle_x_inicial - 0.1:
                return False
        elif name_body_part == 'knee':
            if dict_lm['right_knee_x'] < self.knee_x_inicial - 0.1:
                return False
        elif name_body_part == 'heel':
            if dict_lm['right_heel_x'] < self.heel_x_inicial - 0.025:   
                return False
        elif name_body_part == 'shoulder':
            if dict_lm['right_shoulder_x'] < self.showlder_x_inicial - 0.1:
                return False
        elif name_body_part == 'hip':
            if dict_lm['right_hip_x'] < self.hip_x_inicial - 0.1:
                return False
        else:
            print("Nome invalido, verifique se vc forneceu o nome correto do ponto")
        return True

    def _check_head_posture_error(self, dict_lm):
        hp_status = 0
        try:
            # Coordenadas dos olhos
            olho_direito_x = dict_lm['right_eye_x']
            olho_direito_y = dict_lm['right_eye_y']
            olho_esquerdo_x = dict_lm.get('left_eye_x', None)
            olho_esquerdo_y = dict_lm.get('left_eye_y', None)

            # Se não tiver left_eye, calcula pelo nariz e ombros (fallback)
            if olho_esquerdo_x is not None and olho_esquerdo_y is not None:
                dx = olho_direito_x - olho_esquerdo_x
                dy = olho_direito_y - olho_esquerdo_y
            else:
                # Fallback: usa ombros
                ombro_esquerdo_x = dict_lm['left_shoulder_x']
                ombro_esquerdo_y = dict_lm.get('left_shoulder_y', 0)
                ombro_direito_x = dict_lm['right_shoulder_x']
                ombro_direito_y = dict_lm.get('right_shoulder_y', 0)
                dx = ombro_direito_x - ombro_esquerdo_x
                dy = ombro_direito_y - ombro_esquerdo_y

            angulo_rad = math.atan2(dy, dx)
            angulo_graus = abs(math.degrees(angulo_rad))

            # O ideal é que a linha dos olhos esteja em 0° (horizontal)
            # Se o ângulo for maior que 5°, considera erro
            TOLERANCIA_ANGULO = 5
            if min(abs(angulo_graus), abs(angulo_graus - 180)) > TOLERANCIA_ANGULO:
                self.consecutive_head_error_counter += 1
                hp_status = 1
            else:
                self.consecutive_head_error_counter = 0

            if self.consecutive_head_error_counter >= self.HEAD_ERROR_THRESHOLD:
                self.total_head_error_counter += 1
                self.consecutive_head_error_counter = 0

        except Exception as e:
            print(f"Erro no cálculo do alinhamento da cabeça: {e}")
            self.consecutive_head_error_counter = 0
        
        return hp_status

# TODO Funão antiga apagar quando a nova iplementação estiver testada e funcionando
#    def _check_trunk_flexion_error(self, dict_lm):
        tr_status = 0
        TOLERANCIA_ANGULO = 14 # Por enquanto isso está definido
        try:
            trunk_angle_rad = math.atan2(dict_lm['right_hip_y'] - dict_lm['right_shoulder_y'], dict_lm["right_hip_x"] - dict_lm['right_shoulder_x'])
            trunk_angle_deg = abs(math.degrees(trunk_angle_rad))
            
            tibia_angle_rad = math.atan2(dict_lm['right_ankle_y'] - dict_lm['right_knee_y'], dict_lm['right_ankle_x'] - dict_lm['right_knee_x'])
            tibia_angle_deg = abs(math.degrees(tibia_angle_rad))
            
            if (self.position_validation(dict_lm, 'knee') is False) or (self.position_validation(dict_lm, 'ankle') is False or self.position_validation(dict_lm, 'hip') is False or self.position_validation(dict_lm, 'shoulder') is False):
                print("Os pontos para o cálculo do TRONCO estão marcados incorretamente.")
            else:
                if (trunk_angle_deg + TOLERANCIA_ANGULO < tibia_angle_deg):
                    self.consecutive_trunk_error_counter += 1
                    tr_status = 1
                else:
                    self.consecutive_trunk_error_counter = 0

            if self.consecutive_trunk_error_counter >= self.TRUNK_ERROR_THRESHOLD:
                self.total_trunk_error_counter += 1
                self.consecutive_trunk_error_counter = 0
        except Exception as e:
            print(f"Erro específico no cálculo do tronco: {e}")
            self.consecutive_trunk_error_counter = 0
        return tr_status

    def _check_trunk_flexion_error(self, dict_lm):
        tr_status = 0
        # Tolerancia < 100
        TOLERANCIA_PONTO_MAXIMO = 90 # ? Continuar testando esse valor
        # Tolerância ?
        TOLERANCIA_PONTO = 1 # ? Continuar testando esse valor

        try:
            right_hip_x = dict_lm['right_hip_x']
            right_hip_y = dict_lm['right_hip_y']
            right_shoulder_x = dict_lm['right_shoulder_x']
            right_shoulder_y = dict_lm['right_shoulder_y']
            right_ankle_x = dict_lm['right_ankle_x']
            right_ankle_y = dict_lm['right_ankle_y']
            right_knee_x = dict_lm['right_knee_x']
            right_knee_y = dict_lm['right_knee_y']

            if (self.position_validation(dict_lm, 'knee') is False) or (self.position_validation(dict_lm, 'ankle') is False or self.position_validation(dict_lm, 'hip') is False or self.position_validation(dict_lm, 'shoulder') is False):
                #print("Os pontos para o cálculo do TRONCO estão marcados incorretamente.")
                return tr_status

            # Pegando a equação da reta do tronco e da tíbia
            line_equation_trunk = VectorCalculator.get_line_equation(right_hip_x, right_hip_y, right_shoulder_x, right_shoulder_y)
            line_equation_tibia = VectorCalculator.get_line_equation(right_ankle_x, right_ankle_y, right_knee_x, right_knee_y)

            # Encontrando o ponto de interseção entre as duas retas
            intersection_point = VectorCalculator.find_line_intersection(line_equation_trunk, line_equation_tibia)
            if intersection_point is None:
                return tr_status  # Retas paralelas, não há interseção
            
            if ((intersection_point[0] < TOLERANCIA_PONTO_MAXIMO) and ((intersection_point[0]- TOLERANCIA_PONTO) > max(right_hip_x, right_shoulder_x) or (intersection_point[0] - TOLERANCIA_PONTO) > max(right_ankle_x, right_knee_x))):
                print(f"O ponto de interseção equivale esta x: {intersection_point[0]:.4f}")
                print(f"A distância entre o ponto de interseção e ponto máximo das retas é de {abs(intersection_point[0] - max(right_hip_x, right_shoulder_x)):.4f} no tronco e {abs(intersection_point[0] - max(right_ankle_x, right_knee_x)):.4f} na tíbia.")
                self.consecutive_trunk_error_counter += 1
                tr_status = 1
            else:
                self.consecutive_trunk_error_counter = 0

            if self.consecutive_trunk_error_counter >= self.TRUNK_ERROR_THRESHOLD:
                self.total_trunk_error_counter += 1
                self.consecutive_trunk_error_counter = 0
        except Exception as e:
            print(f"Erro específico no cálculo do tronco: {e}")
            self.consecutive_trunk_error_counter = 0
        return tr_status
    
    def _check_knee_translation_error(self, dict_lm):
        kn_status = 0
        try:
            foot_length_x = abs(dict_lm['right_big_toe_x'] - dict_lm['right_heel_x'])
            allowed_forward_translation = foot_length_x * 0.30
            
            if (self.position_validation(dict_lm, 'knee') is False) or (self.position_validation(dict_lm, 'ankle') is False):
                #print("Os pontos para o cálculo do JOELHO estão marcados incorretamente.")
                return kn_status
    
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
        hl_status = 0
        try:
            posicao_x_calcanhar = dict_lm["right_heel_x"]
            posicao_y_calcanhar = dict_lm["right_heel_y"]
            posicao_x_tornozelo = dict_lm["right_ankle_x"]

            #print(f"Tornozelo X: {posicao_x_tornozelo:.4f}, Calcanhar X: {posicao_x_calcanhar:.4f}, Calcanhar Y: {posicao_y_calcanhar:.4f}")

            # Verifica se o calcanhar está mais alto que o inicial
            # E se o tornozelo avançou no eixo x em relação à posição inicial
            avancou_tornozelo = posicao_x_tornozelo > self.ankle_x_inicial + 0.1 # ? Esse valor ainda precisa ser ajustado
            # ? Essa função do avancou_calcanhar ainda está sendo testada
            avancou_calcanhar = abs(posicao_x_calcanhar - self.heel_x_inicial) > 0.01

            if (self.position_validation(dict_lm, 'heel') is False and self.position_validation(dict_lm, 'ankle') is False):
                #print("Os pontos para o cálculo do CALCANHAR estão marcados incorretamente.")
                return hl_status

            if posicao_y_calcanhar < self.heel_y_inicial and (avancou_tornozelo or avancou_calcanhar):
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
    
    def _check_errors(self, dict_lm):
        hp_status = tr_status = hl_status = kn_status = 0

        if self.current_phase in ['descendo', 'subindo']:
            
            hp_status = self._check_head_posture_error(dict_lm)
            tr_status = self._check_trunk_flexion_error(dict_lm)
            kn_status = self._check_knee_translation_error(dict_lm)
            hl_status = self._check_heel_lift_error(dict_lm)
        
        return hp_status, tr_status, hl_status, kn_status

    def _reset_error_counters(self):
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
            trunk_rep_result = 1 if self.total_trunk_error_counter > 0 else 0
            knee_rep_result = 1 if self.total_knee_error_counter > 0 else 0
            head_rep_result = 1 if self.total_head_error_counter > 0 else 0
            foot_rep_result = 1 if self.total_foot_error_counter > 0 else 0
            
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
