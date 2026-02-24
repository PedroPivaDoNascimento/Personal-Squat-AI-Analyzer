import pandas as pd
import cv2

from .base_personal_ai import BaseAI 
from ..squat_analyzer.frontal.right_frontal import RightFrontal 
from ..squat_analyzer.frontal.left_frontal import LeftFrontal


class FrontalAI(BaseAI):
    """
    Classe concreta para análise do agachamento unipodal no Plano Frontal.
    """
    def __init__(self, file_name, name_pessoa, model_path, 
                 descent_threshold=0.05, ascent_return_threshold=0.02, 
                 hip_error_threshold=5, knee_valgus_error_threshold=5, 
                 foot_pronation_error_threshold=5, side="right", options_marcadas=[]):
        
        kwargs = {
            'descent_threshold': descent_threshold,
            'ascent_return_threshold': ascent_return_threshold,
            'hip_error_threshold': hip_error_threshold,
            'knee_valgus_error_threshold': knee_valgus_error_threshold,   
            'foot_pronation_error_threshold': foot_pronation_error_threshold,
        }

        self.options_marcadas = options_marcadas
        
        # user_height_cm foi passado como 0 (ou None) para satisfazer a BaseAI
        super().__init__(file_name, name_pessoa, 0, model_path, **kwargs)
        
        if (side == "right"):
            self.squat_analyzer = RightFrontal(**kwargs, side="direito", person_name=name_pessoa, options_marcadas=options_marcadas)
        elif (side == "left"):
            self.squat_analyzer = LeftFrontal(**kwargs, side="esquerdo", person_name=name_pessoa, options_marcadas=options_marcadas)
        else:
            print("Erro ao definir o lado")
            
        
        self.hip_tilt_df = pd.DataFrame(columns=["Tempo (ms)", "Desvio do Quadril"])
        self.knee_valgus_df = pd.DataFrame(columns=["Tempo (ms)", "Valgo de Joelho"])
        self.foot_pronation_df = pd.DataFrame(columns=["Tempo (ms)", "Pronação do Pé"])


    def process_video(self, draw, display):
        """
        Implementação concreta: Lógica de processamento de frames para o plano frontal.
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
                
                current_hip, current_kn_valgus, current_foot_pronation = 0, 0, 0 
                
                if res.pose_landmarks and res.pose_landmarks[0]:
                    current_hip, current_kn_valgus, current_foot_pronation = \
                        self.squat_analyzer.process_frame_landmarks(res.pose_landmarks[0], ts)
                else:
                    self.squat_analyzer.process_frame_landmarks(None, ts)
                
                # Adiciona os dados ao DataFrame (Adaptado para Frontal)
                for df, val in [
                    (self.hip_tilt_df, current_hip), 
                    (self.knee_valgus_df, current_kn_valgus),
                    (self.foot_pronation_df, current_foot_pronation)
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