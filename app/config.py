"""Configurazione centralizzata dell'applicazione.

Tutti i parametri vengono letti dal file `.env` nella radice del progetto
(vedi `.env.example`). Nel codice sorgente non compare nessuna credenziale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = PROJECT_ROOT / ".env"
PLACEHOLDER_PASSWORDS = frozenset({"", "cambia_questa_password_forte"})
MODEL_STRIDE = 32  # YOLO e il rilevatore di InsightFace lavorano su lati multipli di 32


class ConfigurationError(RuntimeError):
    """Sollevata quando una variabile del file .env manca o non e valida."""


def _get_str(name: str, default: str | None = None) -> str:
    """Legge una stringa. Se `default` e None, la variabile e obbligatoria."""
    raw = os.getenv(name, "").strip()
    if raw:
        return raw
    if default is None:
        raise ConfigurationError(f"Variabile obbligatoria mancante nel file .env: {name}")
    return default


def _get_int(name: str, default: int) -> int:
    """Legge un intero, restituendo il valore predefinito se la variabile non e impostata."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} deve essere un numero intero, trovato: {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    """Legge un numero decimale, restituendo il valore predefinito se non impostato."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} deve essere un numero, trovato: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Parametri di esecuzione, caricati una sola volta all'avvio."""

    # Database
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    # Webcam
    camera_index: int
    camera_width: int
    camera_height: int

    # Modelli di intelligenza artificiale
    yolo_model_path: Path
    yolo_confidence: float
    yolo_image_size: int
    insightface_model: str
    insightface_det_size: int

    # Soglie di riconoscimento (similarita coseno, valori tra 0 e 1)
    recognition_threshold: float
    continuous_learning_threshold: float

    # Comportamento applicativo
    welcome_back_absence_seconds: float
    unknown_snooze_seconds: float
    unknown_confirm_seconds: float
    enrollment_samples: int
    min_face_width_px: int
    min_sharpness: float
    continuous_batch_size: int
    continuous_min_interval_seconds: float
    max_continuous_embeddings: int
    top_k_neighbors: int

    @classmethod
    def from_env(cls) -> Settings:
        """Carica il file .env e restituisce le impostazioni validate.

        Raises:
            ConfigurationError: se il file manca, una variabile non e valida
                o le credenziali sono ancora quelle di esempio.
        """
        if not ENV_FILE.exists():
            raise ConfigurationError(
                f"File .env non trovato ({ENV_FILE}). "
                "Copia .env.example in .env e imposta DB_PASSWORD."
            )
        load_dotenv(ENV_FILE)

        yolo_path = Path(_get_str("YOLO_MODEL_PATH", "models/yolov8n-face.pt"))
        if not yolo_path.is_absolute():
            yolo_path = PROJECT_ROOT / yolo_path

        settings = cls(
            db_host=_get_str("DB_HOST", "127.0.0.1"),
            db_port=_get_int("DB_PORT", 5432),
            db_name=_get_str("DB_NAME", "face_access"),
            db_user=_get_str("DB_USER", "face_app"),
            db_password=_get_str("DB_PASSWORD"),
            camera_index=_get_int("CAMERA_INDEX", 0),
            camera_width=_get_int("CAMERA_WIDTH", 1280),
            camera_height=_get_int("CAMERA_HEIGHT", 720),
            yolo_model_path=yolo_path,
            yolo_confidence=_get_float("YOLO_CONFIDENCE", 0.5),
            yolo_image_size=_get_int("YOLO_IMAGE_SIZE", 640),
            insightface_model=_get_str("INSIGHTFACE_MODEL", "buffalo_l"),
            insightface_det_size=_get_int("INSIGHTFACE_DET_SIZE", 320),
            recognition_threshold=_get_float("RECOGNITION_THRESHOLD", 0.40),
            continuous_learning_threshold=_get_float("CONTINUOUS_LEARNING_THRESHOLD", 0.55),
            welcome_back_absence_seconds=_get_float("WELCOME_BACK_ABSENCE_SECONDS", 600.0),
            unknown_snooze_seconds=_get_float("UNKNOWN_SNOOZE_SECONDS", 60.0),
            unknown_confirm_seconds=_get_float("UNKNOWN_CONFIRM_SECONDS", 1.5),
            enrollment_samples=_get_int("ENROLLMENT_SAMPLES", 8),
            min_face_width_px=_get_int("MIN_FACE_WIDTH_PX", 90),
            min_sharpness=_get_float("MIN_SHARPNESS", 60.0),
            continuous_batch_size=_get_int("CONTINUOUS_BATCH_SIZE", 5),
            continuous_min_interval_seconds=_get_float("CONTINUOUS_MIN_INTERVAL_SECONDS", 600.0),
            max_continuous_embeddings=_get_int("MAX_CONTINUOUS_EMBEDDINGS", 20),
            top_k_neighbors=_get_int("TOP_K_NEIGHBORS", 5),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        """Verifica che i valori siano coerenti tra loro."""
        if self.db_password in PLACEHOLDER_PASSWORDS:
            raise ConfigurationError("Imposta una DB_PASSWORD personale nel file .env")

        for label, value in (
            ("YOLO_CONFIDENCE", self.yolo_confidence),
            ("RECOGNITION_THRESHOLD", self.recognition_threshold),
            ("CONTINUOUS_LEARNING_THRESHOLD", self.continuous_learning_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{label} deve essere compreso tra 0 e 1")

        if self.continuous_learning_threshold < self.recognition_threshold:
            raise ConfigurationError(
                "CONTINUOUS_LEARNING_THRESHOLD non puo essere inferiore a RECOGNITION_THRESHOLD"
            )
        if not 5 <= self.enrollment_samples <= 10:
            raise ConfigurationError("ENROLLMENT_SAMPLES deve essere compreso tra 5 e 10")
        if not 1 <= self.db_port <= 65535:
            raise ConfigurationError("DB_PORT deve essere compreso tra 1 e 65535")

        for label, value in (
            ("CAMERA_WIDTH", self.camera_width),
            ("CAMERA_HEIGHT", self.camera_height),
            ("YOLO_IMAGE_SIZE", self.yolo_image_size),
            ("INSIGHTFACE_DET_SIZE", self.insightface_det_size),
            ("MIN_FACE_WIDTH_PX", self.min_face_width_px),
            ("CONTINUOUS_BATCH_SIZE", self.continuous_batch_size),
            ("MAX_CONTINUOUS_EMBEDDINGS", self.max_continuous_embeddings),
            ("TOP_K_NEIGHBORS", self.top_k_neighbors),
        ):
            if value < 1:
                raise ConfigurationError(f"{label} deve essere un intero maggiore di zero")

        for label, value in (
            ("CAMERA_INDEX", self.camera_index),
            ("WELCOME_BACK_ABSENCE_SECONDS", self.welcome_back_absence_seconds),
            ("UNKNOWN_SNOOZE_SECONDS", self.unknown_snooze_seconds),
            ("UNKNOWN_CONFIRM_SECONDS", self.unknown_confirm_seconds),
            ("MIN_SHARPNESS", self.min_sharpness),
            ("CONTINUOUS_MIN_INTERVAL_SECONDS", self.continuous_min_interval_seconds),
        ):
            if value < 0:
                raise ConfigurationError(f"{label} non puo essere negativo")

        for label, value in (
            ("YOLO_IMAGE_SIZE", self.yolo_image_size),
            ("INSIGHTFACE_DET_SIZE", self.insightface_det_size),
        ):
            if value % MODEL_STRIDE != 0:
                raise ConfigurationError(f"{label} deve essere un multiplo di {MODEL_STRIDE}")
