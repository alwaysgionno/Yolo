"""Stima della posa con YOLOv8-pose: naso, spalle e polsi di ogni persona inquadrata.

Serve solo al saluto con la mano (`app/wave_detector.py`). I pesi `yolov8n-pose.pt` vengono
scaricati automaticamente da Ultralytics nella cartella `models/` al primo avvio.
"""

from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from .config import Settings
from .models import ArmPose, BodyPose

logger = logging.getLogger(__name__)

POSE_CONFIDENCE = 0.4        # confidenza minima della persona rilevata
KEYPOINT_CONFIDENCE = 0.5    # confidenza minima di un singolo punto del corpo
# Indici dei punti nel formato COCO usato da YOLOv8-pose
NOSE = 0
SHOULDERS = (5, 6)           # sinistra, destra
WRISTS = (9, 10)


class PoseEstimator:
    """Modello YOLOv8-pose. Usato da un solo thread, il worker di inferenza."""

    def __init__(self, settings: Settings) -> None:
        settings.pose_model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Caricamento del modello di posa per il saluto con la mano...")
        self._model = YOLO(str(settings.pose_model_path))
        self._image_size = settings.yolo_image_size
        self._model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=self._image_size, verbose=False)

    def estimate(self, frame: np.ndarray) -> list[BodyPose]:
        """Restituisce le pose delle persone con il naso visibile e almeno un braccio affidabile."""
        results = self._model.predict(frame, conf=POSE_CONFIDENCE, imgsz=self._image_size, verbose=False)
        keypoints = results[0].keypoints
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return []
        poses: list[BodyPose] = []
        for points in keypoints.data.cpu().numpy():  # (17, 3): x, y, confidenza
            if points.shape[1] < 3 or points[NOSE, 2] < KEYPOINT_CONFIDENCE:
                continue
            arms = tuple(
                ArmPose(side, _point(points[shoulder]), _point(points[wrist]))
                for side, (shoulder, wrist) in enumerate(zip(SHOULDERS, WRISTS))
                if points[shoulder, 2] >= KEYPOINT_CONFIDENCE and points[wrist, 2] >= KEYPOINT_CONFIDENCE
            )
            if arms:
                poses.append(BodyPose(_point(points[NOSE]), arms))
        return poses


def _point(keypoint: np.ndarray) -> tuple[float, float]:
    return float(keypoint[0]), float(keypoint[1])
