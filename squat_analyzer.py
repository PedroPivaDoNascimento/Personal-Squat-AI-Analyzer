# squat_analyzer.py
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

        self.trunk_error_counter = 0
        self.knee_error_counter = 0
        self.head_error_counter = 0
        self.foot_error_counter = 0

        self.trunk_error_history = []
        self.knee_error_history = []
        self.head_error_history = []
        self.foot_error_history = []

        self.reps = {'head': [], 'trunk': [], 'heel': [], 'knee': []}
        self.repetition_timestamps = []
        
        self.head_initial_diff = None

    def process_frame_landmarks(self, landmarks_obj, timestamp_ms): 
        hp = tr = hl = kn = 0 
        
        if not landmarks_obj: 
            return hp, tr, hl, kn 

        ear_y = landmarks_obj[solutions.pose.PoseLandmark.LEFT_EAR].y 
        
        if self.head_initial_diff is None:
            self.head_initial_diff = landmarks_obj[solutions.pose.PoseLandmark.LEFT_SHOULDER].y - landmarks_obj[solutions.pose.PoseLandmark.LEFT_EAR].y

        self._detect_repetition_phase(ear_y, timestamp_ms)
        
        hp, tr, hl, kn = self._check_errors(landmarks_obj) 
        
        return hp, tr, hl, kn 


    def _detect_repetition_phase(self, ear_y, ts):
        if self.ear_y_inicial is None:
            if len(self.ear_y_history) >= 10: 
                self.ear_y_inicial = np.mean(self.ear_y_history[-10:])
            else:
                self.ear_y_history.append(ear_y)
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


    def _check_errors(self, lm_obj): 
        hp_status = tr_status = hl_status = kn_status = 0 
        if self.current_phase in ['descendo', 'subindo']:
            try:
                shoulder_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_SHOULDER].y
                shoulder_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_SHOULDER].x
                hip_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_HIP].y
                hip_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_HIP].x

                angle_rad = math.atan2(hip_y - shoulder_y, hip_x - shoulder_x)
                angle_deg = abs(math.degrees(angle_rad)) 
                
                if not (70 <= angle_deg <= 110): 
                    self.trunk_error_counter += 1
                    tr_status = 1 
            except Exception:
                pass

            try:
                hip_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_HIP].x
                knee_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_KNEE].x
                ankle_x = lm_obj[solutions.pose.PoseLandmark.RIGHT_ANKLE].x

                margin_x = 0.03 
                
                if not (min(hip_x, ankle_x) - margin_x <= knee_x <= max(hip_x, ankle_x) + margin_x):
                    self.knee_error_counter += 1
                    kn_status = 1 
            except Exception:
                pass

            try:
                current_head_diff = lm_obj[solutions.pose.PoseLandmark.LEFT_SHOULDER].y - lm_obj[solutions.pose.PoseLandmark.LEFT_EAR].y
                
                if self.head_initial_diff is not None and \
                   (current_head_diff < self.head_initial_diff * 0.9 or current_head_diff > self.head_initial_diff * 1.1): 
                    self.head_error_counter += 1
                    hp_status = 1 
            except Exception:
                pass
            
            try:
                heel_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_HEEL].y
                ankle_y = lm_obj[solutions.pose.PoseLandmark.RIGHT_ANKLE].y
                
                if (ankle_y - heel_y) > 0.01: 
                    self.foot_error_counter += 1
                    hl_status = 1 
            except Exception:
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
            

    def finalize_analysis(self, final_ts):
        if self.repetitions_detected == 0 and self.current_phase != 'inicial':
            self._complete_repetition(final_ts)
            self.repetitions_detected = 1 

        for key in ['head', 'trunk', 'heel', 'knee']:
            while len(self.reps[key]) < 3:
                self.reps[key].append(0) 
        while len(self.repetition_timestamps) < 3:
            self.repetition_timestamps.append(0.0)
        while len(self.trunk_error_history) < 3: 
            self.trunk_error_history.append(0)
        while len(self.knee_error_history) < 3:
            self.knee_error_history.append(0)
        while len(self.head_error_history) < 3:
            self.head_error_history.append(0)
        while len(self.foot_error_history) < 3:
            self.foot_error_history.append(0)