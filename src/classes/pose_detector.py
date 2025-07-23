import mediapipe as mp
from mediapipe.tasks.python import vision

class PoseDetector:
    def __init__(self, model_path):
        self._landmarker = vision.PoseLandmarker.create_from_model_path(model_path)

    def detect(self, image): 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Realiza a detecção de pose
        return self._landmarker.detect(mp_image)

    def close(self):
        self._landmarker.close()