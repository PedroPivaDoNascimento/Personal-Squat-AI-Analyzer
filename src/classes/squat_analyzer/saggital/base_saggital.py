import pandas as pd

# TODO Trabalhar na organização do código e adptar as funções de calculo vetoriais para a classe do vector_calculator
# TODO Ajustar a função do joelho para que ela idendifique melhor os erros


from abc import ABC, abstractmethod

class BaseSaggital(ABC):
    def __init__(self, descent_threshold=0.05, ascent_return_threshold=0.02, trunk_error_threshold=5, knee_error_threshold=5, head_error_threshold=5, foot_error_threshold=5, user_height_cm=170):
        
        self.DESCENT_THRESHOLD = descent_threshold
        self.ASCENT_RETURN_THRESHOLD = ascent_return_threshold
        self.TRUNK_ERROR_THRESHOLD = trunk_error_threshold
        self.KNEE_ERROR_THRESHOLD = knee_error_threshold
        self.HEAD_ERROR_THRESHOLD = head_error_threshold
        self.FOOT_ERROR_THRESHOLD = foot_error_threshold

        self.user_height_cm = user_height_cm
        self.scale_factor_cm = None

        self.ear_y_inicial = None
        self.ear_y_history = []

        self.heel_y_inicial = None
        self.heel_y_history = []

        self.ankle_x_inicial = None
        self.ankle_y_inicial = None
        self.ankle_x_history = []
        self.ankle_y_history = []
        self.knee_x_inicial = None
        self.knee_y_inicial = None
        self.knee_x_history = []
        self.knee_y_history = []
        self.heel_x_inicial = None
        self.heel_x_history = []
        self.showlder_x_inicial = None
        self.showlder_x_history = []
        self.hip_x_inicial = None
        self.hip_x_history = []

        self.tibia_length_cm = None  # Comprimento da tíbia, calculado após a calibração
        self.initial_heel_ankle_distance = None


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

        self.trunk_intersections_df = pd.DataFrame(columns=[
            'Tempo (ms)', 
            'Comprimento Tíbia (cm)', 
            'Ponto Interseção X', 
            'Ponto Interseção Y'
        ])
        
        # As variáveis de suavização self.head_angle_history e self.SMOOTHING_WINDOW_SIZE
        # foram removidas conforme sua solicitação.
    
    @abstractmethod
    def _calibrate_and_validate_with_height(self, dict_lm):
        pass
    
    def process_frame_landmarks(self, landmarks_obj, timestamp_ms):
        hp = tr = hl = kn = 0
        
        dict_lm =  self.create_dictionary_landmarks(landmarks_obj)

        if dict_lm == {}:
            print("Dicionário de landmarks vazio. Análise de erros ignorada para este frame.")
            return hp, tr, hl, kn
        
        self._calibrate_and_validate_with_height(dict_lm)

        self._detect_repetition_phase(dict_lm, timestamp_ms)
        hp, tr, hl, kn = self._check_errors(dict_lm, timestamp_ms)
        
        return hp, tr, hl, kn
    
    @abstractmethod
    def _detect_repetition_phase(self, dict_lm, ts):
        pass

    @abstractmethod
    def create_dictionary_landmarks(self, lm_obj):
        pass

    @abstractmethod
    def position_validation(self, dict_lm, name_body_part):
        pass

    @abstractmethod
    def _check_head_posture_error(self, dict_lm, timestamp_ms):
        pass

    @abstractmethod
    def _check_trunk_flexion_error(self, dict_lm, timestamp_ms):
        pass
    
    @abstractmethod
    def _check_knee_translation_error(self, dict_lm, timestamp_ms):
        pass
    
    @abstractmethod
    def _check_big_toe_lower_heel(self, dict_lm):
        pass

    @abstractmethod
    def _check_heel_and_ankle_proximity(self, dict_lm, timestamp_ms):
        pass

    @abstractmethod
    def _check_heel_upper_ankle(self, dict_lm):
        pass

    @abstractmethod
    def _check_heel_lift_error(self, dict_lm):
        pass
    
    def _check_errors(self, dict_lm, timestamp_ms):
        hp_status = tr_status = hl_status = kn_status = 0

        if self.current_phase in ['descendo', 'subindo']:
            
            hp_status = self._check_head_posture_error(dict_lm, timestamp_ms)
            tr_status = self._check_trunk_flexion_error(dict_lm, timestamp_ms)
            kn_status = self._check_knee_translation_error(dict_lm, timestamp_ms)
            hl_status = self._check_heel_lift_error(dict_lm, timestamp_ms)
        
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