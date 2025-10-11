
import numpy as np
import queue
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from abc import ABC, abstractmethod

from ..pose_detector import PoseDetector 

class BaseAI(ABC):
    """
    Classe base abstrata para a análise de movimento.
    Define a infraestrutura (detector, desenho) e o contrato de processamento.
    """
    def __init__(self, file_name, name_pessoa, user_height_cm, model_path, **kwargs):
        
        self.user_height_cm = user_height_cm
        self.file_name = file_name
        self.name_pessoa = name_pessoa
        self.image_q = queue.Queue()
        
        self.pose_detector = PoseDetector(model_path)
        self.squat_analyzer = None 
        
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