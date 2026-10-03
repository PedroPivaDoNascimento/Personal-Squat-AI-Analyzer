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


    def _add_dataframe_data(self, ts, hp, tr, hl, kn):
        """
        Adiciona dados nos dataframes correspondentes (Camada View/Data).
        """
        data_map = [
            (self.head_df, hp), 
            (self.trunk_df, tr),
            (self.heel_df, hl), 
            (self.knee_df, kn)
        ]
        for df, val in data_map:
            if df is not None:
                df.loc[len(df)] = [int(ts), val]

    # Resolucao-alvo para downscaling previo (Otimizacao Fase 3.1): frames
    # acima de 720p sao reduzidos antes da inferencia do MediaPipe. Os
    # landmarks sao normalizados [0..1], portanto a geometria angular dos
    # calculos biomecanicos permanece identica, com menos pixels na CPU/RAM.
    MAX_PROCESS_WIDTH = 1280
    MAX_PROCESS_HEIGHT = 720

    def _prepare_frame(self, frame):
        """Redimensiona o frame (downscaling c/ aspect ratio) se exceder 720p."""
        h, w = frame.shape[:2]
        if w <= self.MAX_PROCESS_WIDTH and h <= self.MAX_PROCESS_HEIGHT:
            return frame
        scale = min(self.MAX_PROCESS_WIDTH / w, self.MAX_PROCESS_HEIGHT / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def process_video(self, draw, display):
        cap = cv2.VideoCapture(self.file_name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        ts = 0
        current_hp, current_tr, current_hl, current_kn = 0, 0, 0, 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.frame += 1
                ts += 1000 / fps
                
                # Downscaling previo (Fase 3.1): reduz carga de CPU/RAM em videos pesados
                frame = self._prepare_frame(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.pose_detector.detect(rgb, timestamp_ms=ts)
                
                landmarks = res.pose_landmarks[0] if res.pose_landmarks and res.pose_landmarks[0] else None
                
                # Processa os dados do frame (Camada Controller/Model)
                current_hp, current_tr, current_hl, current_kn = \
                    self.squat_analyzer.process_frame_landmarks(landmarks, ts)
                
                # Alimenta o gráfico em tempo real
                self._add_dataframe_data(ts, current_hp, current_tr, current_hl, current_kn)

                if draw:
                    frame = self.draw_landmarks(rgb, res)
                if display:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # Desalocacao explicita dos buffers pesados do frame atual (Fase 4.2)
                del frame, rgb, res, landmarks

        except Exception as e:
            print(f"ATENÇÃO: Ocorreu um erro durante o processamento do vídeo: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_detector.close()

            self.squat_analyzer.finalize_analysis(current_ts=ts)
            
            num_detected = self.squat_analyzer.repetitions_detected
            for i in range(num_detected, 3):
                ts += 1
                self._add_dataframe_data(ts, 0, 0, 0, 0)
        
        self.image_q.put((1, 1, 'done'))