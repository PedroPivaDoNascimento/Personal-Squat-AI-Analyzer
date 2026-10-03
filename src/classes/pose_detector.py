import mediapipe as mp
from mediapipe.tasks.python import vision


class PoseDetector:
    """
    Encapsula o MediaPipe PoseLandmarker (Tasks API).

    Otimizacoes de CPU/RAM aplicadas (docs/tasks/(2)OPTIMIZE.md - Fase 3.2):
    - num_poses=1: a analise e unipodal com um unico voluntario no quadro;
      limitar a deteccao a 1 pose reduz drasticamente a carga da rede neural.
    - running_mode=VIDEO + timestamp em ms: habilita o rastreamento temporal
      interno (frame-to-frame tracking), evitando reprocessar o frame inteiro
      como uma foto isolada (modo IMAGE) a cada iteracao.
    - Confiancas calibradas: menos re-deteccoes completas por frame quando o
      tracker ja esta confivel, mantendo a taxa de deteccao exigida no PRD.
    """

    def __init__(self, model_path):
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image, timestamp_ms=0):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Realiza a deteccao de pose no modo VIDEO (exige timestamp em ms)
        return self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

    def close(self):
        self._landmarker.close()
