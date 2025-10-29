import pandas as pd
import cv2

from .base_personal_ai import BaseAI 
from ..squat_analyzer.saggital.right_saggital import RightSaggital
from ..squat_analyzer.saggital.left_saggital import LeftSaggital 

class SagittalAI(BaseAI):
    """
    Classe concreta para análise do agachamento unipodal no Plano Sagital.
    """
    def __init__(self, file_name, name_pessoa, side, user_height_cm ,model_path, descent_threshold=0.05, ascent_return_threshold=0.02, trunk_error_threshold=5, knee_error_threshold=5, head_error_threshold=5, foot_error_threshold=5):
        
        kwargs = {
            'descent_threshold': descent_threshold,
            'ascent_return_threshold': ascent_return_threshold,
            'trunk_error_threshold': trunk_error_threshold, 
            'knee_error_threshold': knee_error_threshold,   
            'head_error_threshold': head_error_threshold,   
            'foot_error_threshold': foot_error_threshold,
        }
        
        super().__init__(file_name, name_pessoa, user_height_cm, model_path, **kwargs)
        
        if (side == "right"):
            self.squat_analyzer = RightSaggital(
                user_height_cm=user_height_cm,
                **kwargs 
            )
        elif (side == "left"):
            self.squat_analyzer = LeftSaggital(
                user_height_cm=user_height_cm,
                **kwargs 
            )
        else:
            print("Erro ao definir o lado")


        
        self.head_df = pd.DataFrame(columns=["Tempo (ms)", "Desvio da Cabeça"])
        self.trunk_df = pd.DataFrame(columns=["Tempo (ms)", "Desvio do Tronco"])
        self.heel_df = pd.DataFrame(columns=["Tempo (ms)", "Elevação do Calcanhar"])
        self.knee_df = pd.DataFrame(columns=["Tempo (ms)", "Desvio do Joelho"])


    def process_video(self, draw, display):
        """
        Implementação concreta: Lógica de processamento de frames para o plano sagital.
        """
        cap = cv2.VideoCapture(self.file_name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        ts = 0
        
        try:
            while cap.isOpened():
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
                
                # Adiciona os dados ao DataFrame 
                for df, val in [
                    (self.head_df, current_hp), 
                    (self.trunk_df, current_tr),
                    (self.heel_df, current_hl), 
                    (self.knee_df, current_kn)
                ]:
                    if df is not None:
                        df.loc[len(df)] = [int(ts), val]

                if draw:
                    frame = self.draw_landmarks(rgb, res)

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
        
        self.image_q.put((1, 1, 'done'))