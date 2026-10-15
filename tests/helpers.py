"""Oggetti di supporto per i test: impostazioni, orologio, volti e archivio simulati."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.config import Settings
from app.person_store import StoreError
from app.models import FaceDetection, Match

EMBEDDING_SIZE = 512


def make_settings(**overrides: object) -> Settings:
    """Impostazioni valide, con la possibilita di sostituire singoli valori."""
    values: dict[str, object] = {
        "mqtt_host": "127.0.0.1",
        "mqtt_port": 1883,
        "mqtt_username": "camera",
        "mqtt_password": "password-solo-per-i-test",
        "camera_index": 0,
        "camera_width": 640,
        "camera_height": 480,
        "yolo_model_path": Path("models/yolov8n-face.pt"),
        "yolo_confidence": 0.5,
        "yolo_image_size": 640,
        "insightface_model": "buffalo_l",
        "insightface_det_size": 320,
        "recognition_threshold": 0.40,
        "continuous_learning_threshold": 0.55,
        "liveness_check": False,  # i test che la riguardano la attivano in modo esplicito
        "liveness_min_parallax": 0.07,
        "wave_greeting": False,  # i test che lo riguardano lo attivano in modo esplicito
        "wave_cooldown_seconds": 30.0,
        "pose_model_path": Path("models/yolov8n-pose.pt"),
        "welcome_back_absence_seconds": 600.0,
        "unknown_snooze_seconds": 60.0,
        "unknown_confirm_seconds": 1.5,
        "enrollment_samples": 5,
        "min_face_width_px": 90,
        "min_sharpness": 60.0,
        "continuous_batch_size": 3,
        "continuous_min_interval_seconds": 600.0,
        "top_k_neighbors": 5,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class FakeClock:
    """Orologio controllato dal test, al posto di time.monotonic."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def face_vector(axis: int, spread: float = 0.3) -> np.ndarray:
    """Embedding normalizzato della stessa persona, con una piccola variazione sull'asse indicato.

    Due vettori con assi diversi hanno similarita circa 0.92: abbastanza vicini da essere
    la stessa persona, abbastanza diversi da non essere campioni duplicati.
    """
    vector = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
    vector[0] = 1.0
    if axis:
        vector[axis] = spread
    return vector / np.linalg.norm(vector)


def other_person_vector() -> np.ndarray:
    """Embedding di una persona diversa (similarita 0 con `face_vector`)."""
    vector = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
    vector[100] = 1.0
    return vector


def make_detection(
    embedding: np.ndarray | None = None,
    *,
    face_width: float = 150.0,
    sharpness: float = 120.0,
    match: Match | None = None,
    live: bool | None = None,
    patch: np.ndarray | None = None,
    photo: np.ndarray | None = None,
) -> FaceDetection:
    """Volto rilevato con qualita sufficiente, salvo diversa indicazione."""
    return FaceDetection(
        live=live,
        patch=patch,
        photo=photo,
        bbox=(100, 100, 250, 250),
        detector_confidence=0.9,
        embedding=face_vector(1) if embedding is None else embedding,
        face_width=face_width,
        sharpness=sharpness,
        match=match,
    )


class FakeStore:
    """Sostituisce PersonStore: tiene i dati in memoria e puo simulare un guasto."""

    def __init__(self) -> None:
        self.failing: set[str] = set()   # nomi dei metodi che devono sollevare StoreError
        self.existing_names: set[str] = set()
        self.nearest: list[Match] = []   # risposta di find_nearest
        self.created: list[tuple[str, list[np.ndarray], list[float]]] = []
        self.photos: list[np.ndarray | None] = []  # foto inviata con ogni richiesta di registrazione
        self.pending: Match | None = None          # risposta di find_pending
        self.greetings: list[tuple[str, float]] = []
        self.continuous: list[tuple[str, np.ndarray, float]] = []

    def _check(self, method: str) -> None:
        if method in self.failing:
            raise StoreError("archivio simulato non disponibile")

    def person_exists(self, full_name: str) -> bool:
        self._check("person_exists")
        return full_name.lower() in self.existing_names

    def create_person(
        self,
        full_name: str,
        embeddings: Sequence[np.ndarray],
        qualities: Sequence[float],
        photo: np.ndarray | None = None,
    ) -> str:
        self._check("create_person")
        self.created.append((full_name, list(embeddings), list(qualities)))
        self.photos.append(photo)
        self.existing_names.add(full_name.lower())
        return f"uid-{len(self.created)}"

    def find_nearest(self, embedding: np.ndarray, limit: int) -> list[Match]:
        self._check("find_nearest")
        return list(self.nearest[:limit])

    def find_pending(self, embedding: np.ndarray) -> Match | None:
        return self.pending

    def record_greeting(self, person_id: str, similarity: float) -> None:
        self._check("record_greeting")
        self.greetings.append((person_id, similarity))

    def add_continuous_embedding(self, person_id: str, embedding: np.ndarray, quality: float) -> None:
        self._check("add_continuous_embedding")
        self.continuous.append((person_id, embedding, quality))
