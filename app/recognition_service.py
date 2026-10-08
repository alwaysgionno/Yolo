"""Logica di riconoscimento: soglie, saluti con debouncing e apprendimento continuo.

Il metodo `process` viene eseguito dal thread di inferenza. `register_greeting` puo
essere chiamato anche dal thread della UI, quindi lo stato dei saluti e protetto da lock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from .config import Settings
from .database_manager import DatabaseError, DatabaseManager
from .models import FaceDetection, Match

logger = logging.getLogger(__name__)


class RecognitionService:
    """Associa i volti alle persone registrate e gestisce saluti e aggiornamenti dei profili."""

    def __init__(
        self,
        settings: Settings,
        database: DatabaseManager,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._db = database
        self._clock = clock

        # Stato dei saluti: condiviso tra il thread di inferenza e quello della UI
        self._greeting_lock = threading.Lock()
        self._last_greeted: dict[int, float] = {}

        # Stato dell'apprendimento continuo: usato solo dal thread di inferenza.
        # Ogni buffer contiene coppie (embedding, nitidezza).
        self._learning_buffers: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
        self._last_learning_update: dict[int, float] = {}

    def process(self, detections: list[FaceDetection]) -> list[str]:
        """Identifica ogni volto e restituisce i nomi delle persone da salutare in questo fotogramma.

        Raises:
            DatabaseError: se il database non risponde durante la ricerca.
        """
        greeted: list[str] = []
        for detection in detections:
            if detection.embedding is None:
                continue

            candidates = self._db.find_nearest(detection.embedding, self._settings.top_k_neighbors)
            if not candidates or candidates[0].similarity < self._settings.recognition_threshold:
                continue  # volto sconosciuto: sara la UI a proporre la registrazione

            best = candidates[0]
            detection.match = best
            if self._claim_greeting(best.person_id):
                self._record_access(best)
                greeted.append(best.full_name)
            self._learn_if_confident(detection, best)
        return greeted

    def register_greeting(self, person_id: int) -> None:
        """Segna una persona come gia salutata (ad esempio dopo la registrazione), evitando doppi saluti."""
        with self._greeting_lock:
            self._last_greeted[person_id] = self._clock()

    def _claim_greeting(self, person_id: int) -> bool:
        """Debouncing: concede il saluto solo se e trascorso il cooldown dall'ultimo saluto."""
        now = self._clock()
        with self._greeting_lock:
            last = self._last_greeted.get(person_id)
            if last is not None and now - last < self._settings.greeting_cooldown_seconds:
                return False
            self._last_greeted[person_id] = now
            return True

    def _record_access(self, match: Match) -> None:
        """Registra l'accesso nel database. Un errore DB non blocca il saluto a schermo."""
        try:
            self._db.record_greeting(match.person_id, match.similarity)
        except DatabaseError as exc:
            logger.error("Accesso di %s non registrato nel database: %s", match.full_name, exc)
        logger.info("Saluto: %s (similarita %.2f)", match.full_name, match.similarity)

    def _learn_if_confident(self, detection: FaceDetection, match: Match) -> None:
        """Apprendimento continuo: accumula riconoscimenti sicuri e salva la media in batch."""
        settings = self._settings
        if match.similarity < settings.continuous_learning_threshold:
            return
        if detection.embedding is None or detection.sharpness < settings.min_sharpness:
            return

        # Dentro l'intervallo minimo non si accumula nulla: il prossimo batch sara composto
        # solo da riconoscimenti recenti
        now = self._clock()
        last_update = self._last_learning_update.get(match.person_id)
        if last_update is not None and now - last_update < settings.continuous_min_interval_seconds:
            return

        buffer = self._learning_buffers[match.person_id]
        buffer.append((detection.embedding, detection.sharpness))
        if len(buffer) < settings.continuous_batch_size:
            return

        mean_vector = np.mean(np.stack([embedding for embedding, _ in buffer]), axis=0)
        mean_quality = float(np.mean([sharpness for _, sharpness in buffer]))
        buffer.clear()
        norm = float(np.linalg.norm(mean_vector))
        if norm == 0.0:
            return

        try:
            self._db.add_continuous_embedding(
                match.person_id,
                (mean_vector / norm).astype(np.float32),
                quality=mean_quality,
                max_stored=settings.max_continuous_embeddings,
            )
        except DatabaseError as exc:
            logger.error("Apprendimento continuo non riuscito per %s: %s", match.full_name, exc)
            return
        self._last_learning_update[match.person_id] = now
