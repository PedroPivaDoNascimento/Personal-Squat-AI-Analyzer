import pandas as pd
import cv2
import numpy as np
import queue
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Importar as classes que PersonalAI utiliza
from .pose_detector import PoseDetector
from .squat_analyzer import SquatRepetitionAnalyzer

class PersonalAI:
    def __init__(self, file_name, name_pessoa, model_path,
                 descent_threshold=0.05, ascent_return_threshold=0.02,
                 trunk_error_threshold=5, knee_error_threshold=5,
                 head_error_threshold=5, foot_error_threshold=5):
        
        self.file_name = file_name
        self.name_pessoa = name_pessoa
        self.image_q = queue.Queue()
        
        self.pose_detector = PoseDetector(model_path)
        self.squat_analyzer = SquatRepetitionAnalyzer(
            descent_threshold=descent_threshold,
            ascent_return_threshold=ascent_return_threshold,
            trunk_error_threshold=trunk_error_threshold, 
            knee_error_threshold=knee_error_threshold,   
            head_error_threshold=head_error_threshold,   
            foot_error_threshold=foot_error_threshold    
        )

        # DataFrames para armazenar os dados de cada frame, mostrando se ouve algum desvio ou não
        self.head_df = pd.DataFrame(columns=["Tempo (ms)", "Desvio da Cabeça"])
        self.trunk_df = pd.DataFrame(columns=["Tempo (ms)", "Desvio do Tronco"])
        self.heel_df = pd.DataFrame(columns=["Tempo (ms)", "Elevação do Calcanhar"])
        self.knee_df = pd.DataFrame(columns=["Tempo (ms)", "Desvio do Joelho"])
        
        self.frame = 0

    def draw_landmarks(self, rgb, res):
        out = np.copy(rgb)
        if res.pose_landmarks: 
            for pose_landmark_group in res.pose_landmarks: 
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z)
                    for l in pose_landmark_group 
                ])
                solutions.drawing_utils.draw_landmarks(
                    out, proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )
        return out

    def process_video(self, draw, display):
        cap = cv2.VideoCapture(self.file_name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        ts = 0
        
        try:
            while cap.isOpened():
                #ret é um booleano que indica se o frame ainda está sendo lido ou se o vídeo já acabou e o frame é a imagem capturada
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.frame += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts += 1000 / fps
                
                res = self.pose_detector.detect(rgb)
                
                current_hp, current_tr, current_hl, current_kn = 0, 0, 0, 0 
                if res.pose_landmarks and res.pose_landmarks[0]:
                    current_hp, current_tr, current_hl, current_kn = \
                        self.squat_analyzer.process_frame_landmarks(res.pose_landmarks[0], ts)
                else:
                    current_hp, current_tr, current_hl, current_kn = \
                        self.squat_analyzer.process_frame_landmarks(None, ts)
                    print("Nenhum landmark detectado no frame.")

                # Adiciona os dados ao DataFrame correspondente    
                for df, val in [
                    (self.head_df, current_hp), 
                    (self.trunk_df, current_tr),
                    (self.heel_df, current_hl), 
                    (self.knee_df, current_kn)
                ]:
                    df.loc[len(df)] = [int(ts), val]

                # Desenha os landmarks se necessário
                if draw:
                    frame = self.draw_landmarks(rgb, res)

                # Mostra o frame se necessário    
                if display:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"ATENÇÃO: Ocorreu um erro durante o processamento do vídeo: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_detector.close()
        
        self.squat_analyzer.finalize_analysis()
        
        self.image_q.put((1, 1, 'done')) # Sinaliza que o processamento/fluxo de frames foi concluído.