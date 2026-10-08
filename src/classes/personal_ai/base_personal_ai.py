
import numpy as np
import queue
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from abc import ABC, abstractmethod

from ..pose_detector import PoseDetector 
from ..segment_data_collector import SegmentDataCollector

class BaseAI(ABC):
    """
    Classe base abstrata para a análise de movimento.
    Define a infraestrutura (detector, desenho) e o contrato de processamento.
    """
    def __init__(self, file_name, name_pessoa, user_height_cm, model_path, plane_folder_name=None, **kwargs):
        
        self.user_height_cm = user_height_cm
        self.file_name = file_name
        self.name_pessoa = name_pessoa
        self.image_q = queue.Queue()
        
        self.pose_detector = PoseDetector(model_path)
        self.squat_analyzer = None 
        
        # Coletor das séries temporais dos segmentos biomecânicos
        # (docs/tasks/(3)NEW-DATAXANALYTICS-SHEETS.md - Fase 2). Fica em None
        # quando o plano não é 'frontal'/'sagital', desativando a coleta sem
        # quebrar fluxos legados.
        self.plane_folder_name = plane_folder_name
        self.segment_collector = (
            SegmentDataCollector(plane_folder_name)
            if plane_folder_name in ('frontal', 'sagital') else None
        )

        self.head_df = None
        self.trunk_df = None
        self.heel_df = None
        self.knee_df = None
        self.frame = 0

    def draw_landmarks(self, rgb, res):
        """
        Implementação concreta: Desenha os landmarks (idêntica em todos os planos).
        """
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

    @abstractmethod
    def process_video(self, draw, display):
        """
        Método abstrato: Deve ser implementado pela classe filha.
        """
        pass

    def _collect_segment_frame(self, landmarks):
        """
        Alimenta o coletor de segmentos (dados_do_segmento) com o frame atual.
        Nunca interrompe o pipeline principal: falhas de coleta são apenas
        registradas (docs/tasks/(3)NEW-DATAXANALYTICS-SHEETS.md - tarefa 4.1).
        """
        if self.segment_collector is None or landmarks is None:
            return
        try:
            self.segment_collector.collect_frame(
                landmarks_obj=landmarks,
                timestamp_ms=getattr(self, '_last_ts_ms', 0),
                frame_number=self.frame,
            )
        except Exception as exc:
            print(f"Aviso: falha ao coletar dados de segmento no frame {self.frame}: {exc}")