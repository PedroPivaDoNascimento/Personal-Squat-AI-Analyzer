# pose_detector.py
import mediapipe as mp
from mediapipe.tasks import python

class PoseDetector:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = python.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=python.vision.RunningMode.VIDEO
        )
        self._landmarker = python.vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_data, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
        return self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

    def close(self):
        if self._landmarker:
            self._landmarker.close()