"""Pipeline di visione artificiale: rilevamento con YOLO, embedding con InsightFace.

Flusso per ogni fotogramma:
    1. YOLO individua le regioni che contengono un volto (bounding box).
    2. Ogni regione viene ritagliata con un margine e passata a InsightFace (ArcFace),
       che allinea il volto e produce un vettore biometrico a 512 dimensioni.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO

from .config import PROJECT_ROOT, Settings
from .liveness import align_face
from .models import BBox, FaceDetection

logger = logging.getLogger(__name__)

FALLBACK_YOLO_WEIGHTS = "yolov8n.pt"  # modello generico COCO, scaricato da Ultralytics in models/
TARGET_CLASS_ID = 0                   # "volto" nei pesi per volti, "persona" nel modello generico
MIN_REGION_SIDE_PX = 16               # regioni piu piccole non contengono un volto utilizzabile
MIN_CROP_SIDE_PX = 160                # i ritagli piccoli vengono ingranditi prima di InsightFace
CROP_PADDING_RATIO = 0.25             # margine extra attorno al bounding box di YOLO
CROP_EDGE_MARGIN_PX = 2.0             # un volto cosi vicino al bordo del ritaglio e tagliato a meta
DUPLICATE_FACE_IOU = 0.4              # oltre questa sovrapposizione due volti sono lo stesso


class _Crop(NamedTuple):
    """Ritaglio del fotogramma e posizione del suo angolo in alto a sinistra."""

    image: np.ndarray
    offset_x: int
    offset_y: int


class _Face(NamedTuple):
    """Esito dell'estrazione dell'embedding su un singolo ritaglio."""

    vector: np.ndarray  # embedding normalizzato L2
    bbox: BBox          # riquadro del volto nel frame originale, in pixel
    det_score: float    # confidenza del rilevatore facciale di InsightFace
    patch: np.ndarray | None  # volto allineato, usato dalla verifica anti-spoofing


class FaceProcessor:
    """Rilevatore YOLO e riconoscitore InsightFace.

    L'istanza viene usata da un solo thread (il worker di inferenza), perche
    i modelli YOLO e ONNX non sono progettati per essere condivisi tra thread.
    """

    def __init__(self, settings: Settings) -> None:
        weights = settings.yolo_model_path
        self._has_face_weights = weights.exists()
        if self._has_face_weights:
            weights_arg = str(weights)
        else:
            fallback = PROJECT_ROOT / "models" / FALLBACK_YOLO_WEIGHTS
            fallback.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "Pesi YOLO per volti non trovati (%s). Uso il modello generico %s "
                "che rileva le persone: il volto viene poi cercato nel ritaglio.",
                weights,
                FALLBACK_YOLO_WEIGHTS,
            )
            weights_arg = str(fallback)

        logger.info("Caricamento del modello YOLO...")
        self._detector = YOLO(weights_arg)
        self._confidence = settings.yolo_confidence
        self._image_size = settings.yolo_image_size

        logger.info(
            "Caricamento del modello InsightFace '%s' (la prima esecuzione scarica i pesi)...",
            settings.insightface_model,
        )
        self._embedder = FaceAnalysis(
            name=settings.insightface_model,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        det_size = settings.insightface_det_size
        self._embedder.prepare(ctx_id=-1, det_size=(det_size, det_size))  # ctx_id=-1 significa CPU
        self._warm_up()
        logger.info("Modelli di inferenza pronti")

    def _warm_up(self) -> None:
        """Esegue un'inferenza a vuoto per evitare un ritardo sul primo fotogramma reale."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self._detector.predict(blank, imgsz=self._image_size, verbose=False)
        self._embedder.get(blank)

    def process(self, frame: np.ndarray) -> list[FaceDetection]:
        """Rileva i volti nel fotogramma e restituisce le detection con il rispettivo embedding.

        Le regioni in cui InsightFace non trova un volto utilizzabile vengono scartate.
        """
        height, width = frame.shape[:2]
        candidates: list[tuple[_Face, BBox, float]] = []
        for region, confidence in self._detect_regions(frame):
            crop = _crop_with_padding(frame, region)
            if crop is None:
                continue
            faces = self._extract_faces(crop, width, height)
            if self._has_face_weights:
                # La regione e un volto: si tiene quello che InsightFace giudica piu affidabile
                faces = faces[:1]
            # Con il modello generico la regione e una persona intera e puo contenere anche i volti
            # dei vicini: si tengono tutti, cosi nessuno va perso quando le persone si sovrappongono
            candidates.extend((face, region, confidence) for face in faces)

        # Lo stesso volto puo comparire in piu regioni: si tiene la versione piu affidabile
        candidates.sort(key=lambda candidate: candidate[0].det_score, reverse=True)
        detections: list[FaceDetection] = []
        face_boxes: list[BBox] = []
        for face, region, confidence in candidates:
            if any(_iou(face.bbox, other) > DUPLICATE_FACE_IOU for other in face_boxes):
                continue
            face_boxes.append(face.bbox)

            x1, y1, x2, y2 = face.bbox
            detections.append(
                FaceDetection(
                    bbox=region if self._has_face_weights else face.bbox,
                    detector_confidence=confidence,
                    embedding=face.vector,
                    face_width=float(x2 - x1),
                    sharpness=_sharpness(frame[y1:y2, x1:x2]),
                    patch=face.patch,
                )
            )
        return detections

    def _detect_regions(self, frame: np.ndarray) -> list[tuple[BBox, float]]:
        """Esegue YOLO e restituisce i bounding box in pixel con la relativa confidenza."""
        results = self._detector.predict(
            frame,
            conf=self._confidence,
            imgsz=self._image_size,
            classes=[TARGET_CLASS_ID],
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        height, width = frame.shape[:2]
        coords = boxes.xyxy.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        regions: list[tuple[BBox, float]] = []
        for (x1, y1, x2, y2), score in zip(coords, scores):
            bbox: BBox = (
                int(np.clip(x1, 0, width - 1)),
                int(np.clip(y1, 0, height - 1)),
                int(np.clip(x2, 0, width)),
                int(np.clip(y2, 0, height)),
            )
            regions.append((bbox, float(score)))
        return regions

    def _extract_faces(self, crop: _Crop, frame_width: int, frame_height: int) -> list[_Face]:
        """Calcola gli embedding ArcFace dei volti presenti nel ritaglio, dal piu affidabile al meno."""
        image = crop.image
        min_side = min(image.shape[:2])
        scale = max(1.0, MIN_CROP_SIDE_PX / max(min_side, 1))
        if scale > 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        faces: list[_Face] = []
        for found in sorted(self._embedder.get(image), key=lambda face: float(face.det_score), reverse=True):
            if found.embedding is None:
                continue
            raw = np.asarray(found.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(raw))
            if norm == 0.0 or not np.isfinite(norm):
                continue

            # Il riquadro di InsightFace e relativo al ritaglio (eventualmente ingrandito):
            # si riporta alle coordinate del frame originale
            left, top, right, bottom = (float(value) / scale for value in found.bbox[:4])
            if _is_cut_by_crop(left, top, right, bottom, crop, frame_width, frame_height):
                continue  # volto del vicino tagliato dal ritaglio: l'embedding sarebbe inaffidabile
            bbox: BBox = (
                int(np.clip(round(crop.offset_x + left), 0, frame_width - 1)),
                int(np.clip(round(crop.offset_y + top), 0, frame_height - 1)),
                int(np.clip(round(crop.offset_x + right), 0, frame_width)),
                int(np.clip(round(crop.offset_y + bottom), 0, frame_height)),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            # I 5 punti del volto sono nelle coordinate del ritaglio: il volto si allinea da li
            patch = align_face(image, found.kps) if found.kps is not None else None
            faces.append(_Face(raw / norm, bbox, float(found.det_score), patch))
        return faces


def _crop_with_padding(frame: np.ndarray, bbox: BBox, padding: float = CROP_PADDING_RATIO) -> _Crop | None:
    """Ritaglia la regione aggiungendo un margine proporzionale, restando dentro il frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if x2 - x1 < MIN_REGION_SIDE_PX or y2 - y1 < MIN_REGION_SIDE_PX:
        return None
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return _Crop(np.ascontiguousarray(frame[y1:y2, x1:x2]), x1, y1)


def _is_cut_by_crop(
    left: float,
    top: float,
    right: float,
    bottom: float,
    crop: _Crop,
    frame_width: int,
    frame_height: int,
) -> bool:
    """True se il volto tocca un bordo del ritaglio che non coincide con il bordo del fotogramma.

    Le coordinate sono relative al ritaglio. Un volto tagliato dal bordo del fotogramma viene
    invece accettato: e la vista migliore disponibile.
    """
    crop_height, crop_width = crop.image.shape[:2]
    return (
        (left < CROP_EDGE_MARGIN_PX and crop.offset_x > 0)
        or (top < CROP_EDGE_MARGIN_PX and crop.offset_y > 0)
        or (right > crop_width - CROP_EDGE_MARGIN_PX and crop.offset_x + crop_width < frame_width)
        or (bottom > crop_height - CROP_EDGE_MARGIN_PX and crop.offset_y + crop_height < frame_height)
    )


def _iou(first: BBox, second: BBox) -> float:
    """Sovrapposizione tra due riquadri (intersezione diviso unione), tra 0 e 1."""
    inter_w = min(first[2], second[2]) - max(first[0], second[0])
    inter_h = min(first[3], second[3]) - max(first[1], second[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / float(area_first + area_second - intersection)


def _sharpness(image: np.ndarray) -> float:
    """Misura la nitidezza come varianza del Laplaciano: valori bassi indicano immagini sfocate."""
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
