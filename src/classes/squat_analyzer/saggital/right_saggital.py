import numpy as np
import math

from .base_saggital import BaseSaggital
from ...vector_calculator import VectorCalculator

class RightSaggital(BaseSaggital):
    def _calibrate_and_validate_with_height(self, dict_lm):
        """
        Calcula o fator de escala usando a altura real do usuário e
        as proporções da pessoa no vídeo.
        """
        if self.scale_factor_cm is not None:
            return

        try:
            # Altura da pessoa no vídeo (normalizada)
            normalized_person_height = abs(dict_lm['nose_y'] - dict_lm['right_heel_y'])
            
            # Comprimento da tíbia no vídeo (normalizado)
            normalized_tibia_length = VectorCalculator.calculate_distance(
                dict_lm['right_knee_x'], dict_lm['right_knee_y'],
                dict_lm['right_ankle_x'], dict_lm['right_ankle_y']
            )

            if normalized_person_height == 0 or normalized_tibia_length == 0:
                print("Não foi possível calibrar. Certifique-se de que a pessoa inteira está visível.")
                return

            tibia_proportion_in_video = (normalized_tibia_length / normalized_person_height)
            #print(f"Validação: A tíbia ocupa {tibia_proportion_in_video:.2%} da altura da pessoa no vídeo.")

            self.tibia_length_cm = self.user_height_cm * tibia_proportion_in_video
            
            self.scale_factor_cm = self.tibia_length_cm / normalized_tibia_length

            #print(f"Calibração finalizada. O comprimento da sua tíbia foi estimado em {estimated_real_tibia_cm:.2f} cm.")
            #print(f"Fator de escala: {self.scale_factor_cm:.2f} cm por unidade normalizada.")

        except KeyError as e:
            print(f"Erro: Landmarks necessários para a calibração não foram encontrados: {e}")

    def _detect_repetition_phase(self, dict_lm, ts):    
        ear_y = dict_lm['right_ear_y']
        heel_y = dict_lm['right_heel_y']
        heel_x = dict_lm['right_heel_x']
        showlder_x = dict_lm['right_shoulder_x']
        hip_x = dict_lm['right_hip_x']
        knee_x = dict_lm['right_knee_x']
        knee_y = dict_lm['right_knee_y']
        ankle_x = dict_lm['right_ankle_x']
        ankle_y = dict_lm['right_ankle_y']
        
        if self.ear_y_inicial is None:
            if len(self.ear_y_history) >= 10:
                self.ear_y_inicial = np.mean(self.ear_y_history[-10:])
                
                # Calibra o "chão" usando o valor maximo da posição Y
                # para pegar o ponto mais baixo e estável do calcanhar.

                self.heel_y_inicial = np.max(self.heel_y_history[-10:])
                
                self.knee_x_inicial = np.mean(self.knee_x_history[-10:])
                self.knee_y_inicial = np.mean(self.knee_y_history[-10:])
                self.ankle_x_inicial = np.mean(self.ankle_x_history[-10:])
                self.ankle_y_inicial = np.mean(self.ankle_y_history[-10:])
                self.heel_x_inicial = np.mean(self.heel_x_history[-10:])
                self.showlder_x_inicial = np.mean(self.showlder_x_history[-10:])
                self.hip_x_inicial = np.mean(self.hip_x_history[-10:])

                avg_ankle_y = np.mean(self.ankle_y_history[-10:])
                avg_heel_y = np.mean(self.heel_y_history[-10:])
                self.initial_heel_ankle_distance = VectorCalculator.calculate_distance(
                    self.ankle_x_inicial, avg_ankle_y, self.heel_x_inicial, avg_heel_y
                )


            else:
                self.ear_y_history.append(ear_y)
                self.heel_y_history.append(heel_y)
                self.knee_x_history.append(knee_x)
                self.knee_y_history.append(knee_y)
                self.ankle_x_history.append(ankle_x)
                self.ankle_y_history.append(ankle_y)
                self.heel_x_history.append(heel_x)
                self.showlder_x_history.append(showlder_x)
                self.hip_x_history.append(hip_x)
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
                'right_knee_y': lm_obj[26].y,
                'right_ankle_x': lm_obj[28].x,
                'right_ankle_y': lm_obj[28].y,
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
            }
        except Exception as e:
            print(f"Erro ao criar dicionário de landmarks: {e}")
            return {}
    
    def position_validation(self, dict_lm, name_body_part):
        if name_body_part == 'ankle':
            if dict_lm['right_ankle_x'] < self.ankle_x_inicial - 0.03:
                #print("Tornozelo pra tras")
                return False
        elif name_body_part == 'knee':
            if dict_lm['right_knee_x'] < self.knee_x_inicial - 0.03:
                #print("Joelho pra tras")
                return False
        elif name_body_part == 'heel':
            if dict_lm['right_heel_x'] < self.heel_x_inicial - 0.03:   
                #print("Calcanhar pra tras")
                return False
        else:
            print("Nome invalido, verifique se vc forneceu o nome correto do ponto")
        return True

    def _check_head_posture_error(self, dict_lm, timestamp_ms):
        """
        Verifica a postura da cabeça no plano sagital (vista lateral)
        usando o plano de Frankfurt, checando o ângulo absoluto contra 0° (Horizonte).
        """
        hp_status = 0
        TOLERANCIA_ANGULO = 2.5
        
        try:
            ear_x = dict_lm['right_ear_x']
            ear_y = dict_lm['right_ear_y']
            eye_x = dict_lm['right_eye_x']
            eye_y = dict_lm['right_eye_y']
            
            dx = eye_x - ear_x
            dy = eye_y - ear_y
            
            angulo_rad = math.atan2(dy, dx)
            angulo_graus = math.degrees(angulo_rad)
            

            #if (timestamp_ms/1000 >= 6 and timestamp_ms/1000 < 7):
                #print(f"Calibração cabeça - Frame Tempo: {timestamp_ms/1000:.2f}s, Ângulo: {angulo_graus:.2f}°")

            if angulo_graus > TOLERANCIA_ANGULO:
                self.consecutive_head_error_counter += 1
                hp_status = 1
            else:
                self.consecutive_head_error_counter = 0

            # O restante da sua lógica de contagem
            if self.consecutive_head_error_counter >= self.HEAD_ERROR_THRESHOLD:
                self.total_head_error_counter += 1
                self.consecutive_head_error_counter = 0

        except Exception as e:
            # Mensagens de erro resumidas
            if 'KeyError' in str(e):
                print(f"Erro: Landmarks necessários para o cálculo da cabeça não foram encontrados.")
            else:
                print(f"Erro no cálculo do alinhamento da cabeça: {e}")
                
            self.consecutive_head_error_counter = 0
            
        return hp_status

    
    def _check_trunk_flexion_error(self, dict_lm, timestamp_ms):
        tr_status = 0

        try:
            right_hip_x = dict_lm['right_hip_x']
            right_hip_y = dict_lm['right_hip_y']
            right_shoulder_x = dict_lm['right_shoulder_x']
            right_shoulder_y = dict_lm['right_shoulder_y']
            right_ankle_x = dict_lm['right_ankle_x']
            right_ankle_y = dict_lm['right_ankle_y']
            right_knee_x = dict_lm['right_knee_x']
            right_knee_y = dict_lm['right_knee_y']

            if (self.position_validation(dict_lm, 'knee') is False) or (self.position_validation(dict_lm, 'ankle') is False):
                #print("Os pontos para o cálculo do TRONCO estão marcados incorretamente.")
                return tr_status

            # Pegando a equação da reta do tronco e da tíbia
            line_equation_trunk = VectorCalculator.get_line_equation(right_hip_x, right_hip_y, right_shoulder_x, right_shoulder_y)
            line_equation_tibia = VectorCalculator.get_line_equation(right_ankle_x, right_ankle_y, right_knee_x, right_knee_y)

            # Encontrando o ponto de interseção entre as duas retas
            intersection_point = VectorCalculator.find_line_intersection(line_equation_trunk, line_equation_tibia)
            if intersection_point is None:
                return tr_status  # Retas paralelas, não há interseção
            

            
            limit_of_x = 2
            if (((intersection_point[0] > max(right_hip_x, right_shoulder_x)) or (intersection_point[0] > max(right_ankle_x, right_knee_x))) and (intersection_point[0] < limit_of_x)):
                self.consecutive_trunk_error_counter += 1
                tr_status = 1

                # Adiciona uma nova linha ao DataFrame de interseções, agora com o tempo
                self.trunk_intersections_df.loc[len(self.trunk_intersections_df)] = [
                    timestamp_ms,
                    self.tibia_length_cm,
                    intersection_point[0],
                    intersection_point[1]
                ]

            else:
                self.consecutive_trunk_error_counter = 0

            if self.consecutive_trunk_error_counter >= self.TRUNK_ERROR_THRESHOLD:
                self.total_trunk_error_counter += 1
                self.consecutive_trunk_error_counter = 0
        except Exception as e:
            print(f"Erro específico no cálculo do tronco: {e}")
            self.consecutive_trunk_error_counter = 0
        return tr_status
    
    def _check_knee_translation_error(self, dict_lm, timestamp_ms):
        kn_status = 0
        try:
            foot_length_x = abs(dict_lm['right_big_toe_x'] - dict_lm['right_heel_x'])

            allowed_forward_translation = foot_length_x * 0.19  # Permite até 19% do comprimento do pé como avanço do joelho
            avancou_percentual = (dict_lm['right_knee_x'] - dict_lm['right_big_toe_x']) / foot_length_x
            avancou_exessivamente = (avancou_percentual > 1.0)


            if (self.position_validation(dict_lm, 'knee') is False) or (self.position_validation(dict_lm, 'ankle') is False or (self.position_validation(dict_lm, 'heel') is False) or avancou_exessivamente):
                #print("Os pontos para o cálculo do JOELHO estão marcados incorretamente.")
                return kn_status
    
            if dict_lm['right_knee_x'] > dict_lm['right_big_toe_x'] + allowed_forward_translation:
                # debug de quantos porcentos o joelho avançou em relação ao comprimento do pé
                #print(f"Joelho avançou {avancou_percentual:.2f} do comprimento do pé (Erro) e isso ocorreu no segundo: {timestamp_ms/1000:.2f}")
                self.consecutive_knee_error_counter += 1
                kn_status = 1
            else:
                self.consecutive_knee_error_counter = 0
    
            if self.consecutive_knee_error_counter >= self.KNEE_ERROR_THRESHOLD:
                #print(f"Joelho avançou {avancou_percentual:.2f} do comprimento do pé e isso ocorreu no segundo: {timestamp_ms/1000:.2f}")
                self.total_knee_error_counter += 1
                self.consecutive_knee_error_counter = 0
        except Exception as e:
            print(f"Erro específico no cálculo do joelho (translação do pé): {e}")
            self.consecutive_knee_error_counter = 0
        return kn_status

    def _check_big_toe_lower_heel(self, dict_lm):
        TOLERANCIA_Y = 0.05 # ? testar tolerância

        try:
            if dict_lm['right_big_toe_y'] > (self.heel_y_inicial + TOLERANCIA_Y):
                print("Dedo do pe abaixou")
            return dict_lm['right_big_toe_y'] > (self.heel_y_inicial + TOLERANCIA_Y)
        except KeyError as e:
            print(f"Erro: Landmarks necessários para a verificação do pé não foram encontrados: {e}")
            return False

    def _check_heel_and_ankle_proximity(self, dict_lm):
        """
        Verifica se a distância entre o calcanhar e o tornozelo estão ficando próximos
        """
        # Se a distância inicial ainda não foi calibrada, não prossiga.
        if self.initial_heel_ankle_distance is None:
            return False
            
        try:
            ankle_x = dict_lm['right_ankle_x']
            ankle_y = dict_lm['right_ankle_y']
            heel_x = dict_lm['right_heel_x']
            heel_y = dict_lm['right_heel_y']
            
            # Calcula a distância euclidiana atual entre os dois pontos
            current_distance = VectorCalculator.calculate_distance(ankle_x, ankle_y, heel_x, heel_y)

            # Verifica se a distância atual é menor que a distância inicial
            if current_distance < (self.initial_heel_ankle_distance * 1):
                print("Calcanhar e tornozelo muito proximos")
                return True
            
        except KeyError as e:
            print(f"Erro: Landmarks necessários para a verificação de proximidade não foram encontrados: {e}")
            return False

    def _check_heel_upper_ankle(self, dict_lm):
        try:
            if (dict_lm['right_heel_y'] < self.ankle_y_inicial):
                print("Calcanhar alto até de mais")
            return dict_lm['right_heel_y'] < self.ankle_y_inicial
        except KeyError as e:
            print(f"Erro: Landmarks necessários para a verificação de proximidade não foram encontrados: {e}")
            return False

    def _check_heel_lift_error(self, dict_lm, timestamp_ms):
        hl_status = 0
        LIMITE_SUBIDA_CALCANHAR = 0.0125  # ? Testar esse valor
        try:
            posicao_x_calcanhar = dict_lm["right_heel_x"]
            posicao_y_calcanhar = dict_lm["right_heel_y"]
            posicao_x_tornozelo = dict_lm["right_ankle_x"]

            #print(f"Tornozelo X inicial: {self.ankle_x_inicial:.4f}, Calcanhar X inicial: {self.heel_x_inicial:.4f}, Calcanhar Y inicial: {self.heel_y_inicial:.4f}")
            #print(f"Tornozelo X: {posicao_x_tornozelo:.4f}, Calcanhar X: {posicao_x_calcanhar:.4f}, Calcanhar Y: {posicao_y_calcanhar:.4f}")

            # Verifica se o calcanhar está mais alto que o inicial
            # E se o tornozelo avançou no eixo x em relação à posição inicial
            avancou_tornozelo = abs(posicao_x_tornozelo - self.ankle_x_inicial) > 0.027 # ? Esse valor ainda precisa ser ajustado
            # ? Essa função do avancou_calcanhar ainda está sendo testada
            avancou_calcanhar = abs(posicao_x_calcanhar - self.heel_x_inicial) > 0.0179

            pontos_proximos = self._check_heel_and_ankle_proximity(dict_lm)
            falso_positivo_calcanhar_acima_tornozelo = self._check_heel_upper_ankle(dict_lm)
            falso_positivo_dedo_pe_abaixo_calcanhar = self._check_big_toe_lower_heel(dict_lm)

            if (self.position_validation(dict_lm, 'heel') is False or self.position_validation(dict_lm, 'ankle') is False or pontos_proximos or falso_positivo_calcanhar_acima_tornozelo or falso_positivo_dedo_pe_abaixo_calcanhar):
                return hl_status

            if (posicao_y_calcanhar < self.heel_y_inicial - LIMITE_SUBIDA_CALCANHAR) and (avancou_tornozelo or avancou_calcanhar):
                self.consecutive_foot_error_counter += 1
                hl_status = 1
            else:
                self.consecutive_foot_error_counter = 0
        
            if self.consecutive_foot_error_counter >= self.FOOT_ERROR_THRESHOLD:
                print(f"O calcanhar levantou no segundo {timestamp_ms/1000:.2f}")
                self.total_foot_error_counter += 1
                self.consecutive_foot_error_counter = 0
        except Exception as e:
            print(f"Erro específico no cálculo do calcanhar: {e}")
            self.consecutive_foot_error_counter = 0
        return hl_status