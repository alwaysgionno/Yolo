"""Selezione del volto sconosciuto da registrare.

Senza tracking tra fotogrammi, una persona gia registrata puo risultare "sconosciuta"
per qualche fotogramma (ingresso nell'inquadratura, profilo, immagine mossa). Per questo
la registrazione non viene proposta al primo fotogramma: il selettore sceglie il volto
sconosciuto piu vicino alla webcam e lo conferma solo dopo averlo rivisto, sempre
sconosciuto, in piu risultati di inferenza consecutivi.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .config import Settings
from .models import FaceDetection

UNKNOWN_CONFIRM_RESULTS = 3   # risultati consecutivi con lo stesso volto sconosciuto prima di proporlo
IDENTITY_CONSISTENCY = 0.50   # similarita minima tra due embedding dello stesso volto


def find_same_face(detections: Iterable[FaceDetection], reference: np.ndarray) -> FaceDetection | None:
    """Restituisce il volto piu simile a `reference`, se lo e abbastanza da essere la stessa persona."""
    best: FaceDetection | None = None
    best_similarity = IDENTITY_CONSISTENCY
    for detection in detections:
        if detection.embedding is None:
            continue
        similarity = float(np.dot(detection.embedding, reference))
        if similarity >= best_similarity:
            best, best_similarity = detection, similarity
    return best


class NewFaceSelector:
    """Sceglie il volto nuovo da registrare e ne attende la conferma. Usata solo dal thread della UI."""

    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic) -> None:
        self._settings = settings
        self._clock = clock
        self._anchor: np.ndarray | None = None  # embedding del volto in osservazione
        self._first_seen_at = 0.0
        self._observations = 0

    def reset(self) -> None:
        """Scarta il volto in osservazione: la conferma ripartira da zero."""
        self._anchor = None
        self._observations = 0

    def update(self, detections: Sequence[FaceDetection]) -> FaceDetection | None:
        """Valuta i volti di un nuovo risultato di inferenza (una sola chiamata per risultato).

        Il volto in osservazione deve ricomparire, sempre sconosciuto, per almeno
        `UNKNOWN_CONFIRM_RESULTS` risultati consecutivi e `UNKNOWN_CONFIRM_SECONDS` secondi.

        Returns:
            Il volto da registrare, quando e confermato; altrimenti None.
        """
        now = self._clock()
        tracked = find_same_face(detections, self._anchor) if self._anchor is not None else None
        if tracked is None or not self._is_eligible(tracked):
            # Il volto in osservazione e uscito, si e allontanato o e stato riconosciuto:
            # si riparte dal volto sconosciuto piu vicino alla webcam, se ce n'e uno.
            tracked = max(filter(self._is_eligible, detections), key=lambda d: d.face_width, default=None)
            if tracked is None:
                self.reset()
                return None
            self._anchor = tracked.embedding
            self._first_seen_at = now
            self._observations = 0

        self._observations += 1
        if self._observations < UNKNOWN_CONFIRM_RESULTS:
            return None
        if now - self._first_seen_at < self._settings.unknown_confirm_seconds:
            return None
        self.reset()
        return tracked

    def _is_eligible(self, detection: FaceDetection) -> bool:
        """True se il volto e sconosciuto, ha un embedding ed e abbastanza vicino alla webcam."""
        return (
            detection.match is None
            and detection.embedding is not None
            and detection.face_width >= self._settings.min_face_width_px
        )
