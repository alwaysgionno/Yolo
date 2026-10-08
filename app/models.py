"""Strutture dati condivise tra i moduli dell'applicazione."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

BBox = tuple[int, int, int, int]
"""Bounding box in pixel nel formato (x1, y1, x2, y2)."""


@dataclass(frozen=True)
class Match:
    """Persona riconosciuta, con la similarita coseno rispetto al vettore piu vicino."""

    person_id: int
    full_name: str
    similarity: float  # tra -1 e 1; 1 significa vettori identici


@dataclass
class FaceDetection:
    """Un volto rilevato in un fotogramma e il relativo embedding biometrico."""

    bbox: BBox
    detector_confidence: float
    embedding: np.ndarray | None = None  # vettore L2-normalizzato a 512 dimensioni
    face_width: float = 0.0              # larghezza del volto in pixel (sul frame originale)
    sharpness: float = 0.0               # varianza del Laplaciano sul volto: piu alto = piu nitido
    match: Match | None = None           # valorizzato dal RecognitionService


@dataclass
class FrameResult:
    """Esito dell'inferenza su un fotogramma, pubblicato al thread della UI."""

    frame_id: int
    detections: list[FaceDetection] = field(default_factory=list)
    inference_ms: float = 0.0
    db_ok: bool = True  # False se il database non ha risposto durante il riconoscimento
    created_at: float = field(default_factory=time.monotonic)  # istante di pubblicazione (orologio monotono)
