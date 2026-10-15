"""Contratto MQTT con il server (versione 1): topic e formato dei messaggi.

La descrizione completa e nella repository del server (`docs/mqtt-contract.md`), che ha la
propria copia di questo contratto (`ingest/contract.py`): una modifica va fatta in entrambe.
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

TOPIC_ROOT = "faceaccess/v1"

# Telecamera -> server
TOPIC_EVENT_GREETED = f"{TOPIC_ROOT}/events/greeted"
TOPIC_EVENT_ENROLLED = f"{TOPIC_ROOT}/events/enrolled"
TOPIC_EVENT_EMBEDDING = f"{TOPIC_ROOT}/events/embedding"
TOPIC_CAMERA_STATUS = f"{TOPIC_ROOT}/camera/status"

# Server -> telecamera (messaggi retained)
TOPIC_GALLERY_PREFIX = f"{TOPIC_ROOT}/gallery/"
TOPIC_GALLERY_ALL = f"{TOPIC_ROOT}/gallery/#"
TOPIC_GALLERY_INFO = f"{TOPIC_ROOT}/gallery_info"
TOPIC_SERVER_STATUS = f"{TOPIC_ROOT}/server/status"

EMBEDDING_DIMENSION = 512
_EMBEDDING_DTYPE = np.dtype("<f4")  # float32 little-endian
MAX_PHOTO_BYTES = 256 * 1024        # foto della richiesta di registrazione, prima della codifica base64
PHOTO_JPEG_QUALITIES = (90, 80, 70, 60, 50)


class ContractError(ValueError):
    """Messaggio che non rispetta il contratto."""


def encode_photo(image: np.ndarray) -> str | None:
    """Codifica il ritaglio del volto (BGR) come base64 di un JPEG entro `MAX_PHOTO_BYTES`.

    Restituisce None se l'immagine non si puo codificare: la richiesta parte senza foto.
    """
    for quality in PHOTO_JPEG_QUALITIES:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        if len(encoded) <= MAX_PHOTO_BYTES:
            return base64.b64encode(encoded.tobytes()).decode("ascii")
    return None


def encode_embedding(vector: np.ndarray) -> str:
    """Codifica un embedding come base64 di 512 float32 little-endian."""
    if vector.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(f"Embedding con forma {vector.shape}, atteso ({EMBEDDING_DIMENSION},)")
    return base64.b64encode(vector.astype(_EMBEDDING_DTYPE).tobytes()).decode("ascii")


def decode_embedding(text: object) -> np.ndarray:
    """Decodifica un embedding e ne controlla dimensione e valori."""
    if not isinstance(text, str):
        raise ContractError("embedding: attesa una stringa base64")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError("embedding: base64 non valido") from exc
    if len(raw) != EMBEDDING_DIMENSION * _EMBEDDING_DTYPE.itemsize:
        raise ContractError(f"embedding: attesi {EMBEDDING_DIMENSION} valori float32")
    vector = np.frombuffer(raw, dtype=_EMBEDDING_DTYPE).astype(np.float32)
    if not np.all(np.isfinite(vector)):
        raise ContractError("embedding: contiene valori non finiti")
    return vector


def format_timestamp(epoch_seconds: float) -> str:
    """Istante (secondi Unix) in formato ISO 8601 UTC."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def parse_timestamp(text: object) -> float:
    """Legge un istante ISO 8601 con fuso orario e lo restituisce in secondi Unix."""
    if not isinstance(text, str):
        raise ContractError("attesa una data ISO 8601")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContractError("data non valida") from exc
    if moment.tzinfo is None:
        raise ContractError("data senza fuso orario")
    return moment.timestamp()


def encode_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode_json(payload: bytes) -> dict[str, Any]:
    """Legge un messaggio JSON: deve essere un oggetto."""
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError("messaggio JSON non valido") from exc
    if not isinstance(data, dict):
        raise ContractError("il messaggio deve essere un oggetto JSON")
    return data


def encode_status(online: bool) -> bytes:
    return encode_json({"state": "online" if online else "offline"})
