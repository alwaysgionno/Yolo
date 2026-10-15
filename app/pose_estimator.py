"""Stima della posa con YOLOv8-pose: naso, spalle, gomiti e polsi di ogni persona inquadrata.

Serve solo al saluto con la mano (`app/wave_detector.py`). I pesi `yolov8n-pose.pt` vengono
scaricati automaticamente da Ultralytics nella cartella `models/` al primo avvio.
"""

from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from .config import Settings
from .models import ArmPose, BodyPose, Point

logger = logging.getLogger(__name__)

POSE_CONFIDENCE = 0.35       # confidenza minima della persona rilevata
# Confidenza minima di un singolo punto. Bassa di proposito: durante il saluto il polso e mosso e
# il modello ne e meno sicuro, ma la sua posizione resta buona
KEYPOINT_CONFIDENCE = 0.3
MIN_SHOULDER_WIDTH_PX = 20.0  # spalle piu vicine di cosi: persona di profilo o troppo lontana
# Indici dei punti nel formato COCO usato da YOLOv8-pose
NOSE = 0
EYES_AND_EARS = (1, 2, 3, 4)
SHOULDERS = (5, 6)           # sinistra, destra
ELBOWS = (7, 8)
WRISTS = (9, 10)


class PoseEstimator:
    """Modello YOLOv8-pose. Usato da un solo thread, il worker di inferenza."""

    def __init__(self, settings: Settings) -> None:
        settings.pose_model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Caricamento del modello di posa per il saluto con la mano...")
        self._model = YOLO(str(settings.pose_model_path))
        self._image_size = settings.pose_image_size
        self._model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=self._image_size, verbose=False)

    def estimate(self, frame: np.ndarray) -> list[BodyPose]:
        """Restituisce le pose delle persone con naso e spalle visibili e almeno un braccio affidabile."""
        results = self._model.predict(frame, conf=POSE_CONFIDENCE, imgsz=self._image_size, verbose=False)
        keypoints = results[0].keypoints
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return []
        poses: list[BodyPose] = []
        for points in keypoints.data.cpu().numpy():  # (17, 3): x, y, confidenza
            pose = parse_keypoints(points)
            if pose is not None:
                poses.append(pose)
        return poses


def parse_keypoints(points: np.ndarray) -> BodyPose | None:
    """Converte i 17 punti COCO (x, y, confidenza) di una persona in un `BodyPose`, se utilizzabile."""
    if points.shape[0] < 17 or points.shape[1] < 3:
        return None

    def reliable(index: int) -> Point | None:
        # Ultralytics mette a (0, 0) i punti fuori dall'immagine, a volte con una confidenza alta
        if points[index, 2] < KEYPOINT_CONFIDENCE or (points[index, 0] <= 0 and points[index, 1] <= 0):
            return None
        return float(points[index, 0]), float(points[index, 1])

    # La mano che saluta puo coprire il naso: in quel caso la testa e il centro di occhi e orecchie visibili
    nose = reliable(NOSE)
    if nose is None:
        head = [point for point in (reliable(index) for index in EYES_AND_EARS) if point is not None]
        if head:
            nose = (sum(x for x, _ in head) / len(head), sum(y for _, y in head) / len(head))
    left, right = (reliable(index) for index in SHOULDERS)
    if nose is None or left is None or right is None:
        return None
    shoulder_width = float(np.hypot(left[0] - right[0], left[1] - right[1]))
    if shoulder_width < MIN_SHOULDER_WIDTH_PX:
        return None

    arms: list[ArmPose] = []
    for side, (shoulder, elbow, wrist) in enumerate(zip((left, right), ELBOWS, WRISTS)):
        wrist_point = reliable(wrist)
        if wrist_point is not None:
            arms.append(ArmPose(side, shoulder, wrist_point, reliable(elbow)))
    if not arms:
        return None
    return BodyPose(nose, shoulder_width, tuple(arms))
