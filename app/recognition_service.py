"""Logica di riconoscimento: soglie, verifica anti-spoofing, saluti e apprendimento continuo.

Tutto lo stato del servizio viene usato solo dal thread di inferenza: i saluti
raggiungono la UI attraverso la coda dell'InferenceWorker.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np

from .config import Settings
from .person_store import PersonStore, StoreError
from .greeting_policy import GreetingPolicy
from .liveness import LivenessTracker
from .models import BBox, BodyPose, FaceDetection, Greeting, GreetingKind, Match
from .wave_detector import WaveDetector

logger = logging.getLogger(__name__)


class RecognitionService:
    """Associa i volti alle persone registrate e gestisce saluti e aggiornamenti dei profili."""

    def __init__(
        self,
        settings: Settings,
        store: PersonStore,
        clock: Callable[[], float] = time.monotonic,
        liveness: LivenessTracker | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._clock = clock
        self._greeting_policy = GreetingPolicy(settings.welcome_back_absence_seconds, clock)
        # Verifica che i volti siano tridimensionali (None se disattivata con LIVENESS_CHECK=0)
        if liveness is None and settings.liveness_check:
            liveness = LivenessTracker(
                settings.liveness_min_parallax, settings.min_face_width_px, settings.min_sharpness, clock
            )
        self._liveness = liveness
        # Saluto con la mano (None se disattivato con WAVE_GREETING=0)
        self._waves = WaveDetector(settings.wave_cooldown_seconds, clock) if settings.wave_greeting else None

        # Stato dell'apprendimento continuo. Ogni buffer contiene coppie (embedding, nitidezza).
        self._learning_buffers: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
        self._last_learning_update: dict[str, float] = {}

    def process(self, detections: list[FaceDetection], poses: Sequence[BodyPose] = ()) -> list[Greeting]:
        """Identifica ogni volto e restituisce i saluti dovuti in questo fotogramma.

        Args:
            detections: volti rilevati nel fotogramma.
            poses: pose delle persone inquadrate, per il saluto con la mano (vuoto se disattivato).

        Raises:
            StoreError: se la galleria non e ancora arrivata dal server.
        """
        # Prima si identificano tutti i volti: se la galleria non e disponibile l'errore emerge
        # qui, quando nessun saluto e ancora stato consumato senza poter essere mostrato.
        for detection in detections:
            detection.match = self._identify(detection)

        greetings: list[Greeting] = []
        greetable: list[tuple[BBox, Match]] = []  # chi puo ricevere il saluto con la mano
        for detection in detections:
            match = detection.match
            live = self._liveness is None or self._liveness.observe(detection, match.person_id if match else None)
            if match is None:
                continue  # volto sconosciuto: sara la UI a proporre la registrazione
            if match.pending:
                # Registrazione non ancora approvata dal server: la persona non ha accesso, quindi
                # niente saluto, niente accesso registrato e niente apprendimento continuo
                continue
            if not live:
                # Potrebbe essere una foto: niente saluto, niente accesso registrato e niente apprendimento.
                # Per la politica dei saluti la persona non e stata vista: il saluto arrivera a verifica riuscita.
                continue
            greetable.append((detection.bbox, match))
            kind = self._greeting_policy.observe(match.person_id, match.last_access_age_s)
            if kind is not None:
                self._record_access(match, kind)
                greetings.append(Greeting(kind, match.full_name))
            self._learn_if_confident(detection, match)

        if self._waves is not None:
            for match in self._waves.update(poses, greetable):
                # Solo a schermo: il saluto con la mano non e un accesso e non viene inviato al server
                logger.info("Saluto con la mano: %s", match.full_name)
                greetings.append(Greeting(GreetingKind.WAVE, match.full_name))
        return greetings

    def _identify(self, detection: FaceDetection) -> Match | None:
        """Restituisce la persona registrata piu simile al volto, se supera la soglia di riconoscimento.

        Se nessuna persona approvata la supera, cerca tra le registrazioni in attesa di approvazione:
        il volto risulta allora "in attesa" (`Match.pending`) e non viene riproposto per la registrazione.
        """
        if detection.embedding is None:
            return None
        threshold = self._settings.recognition_threshold
        candidates = self._store.find_nearest(detection.embedding, self._settings.top_k_neighbors)
        if candidates and candidates[0].similarity >= threshold:
            return candidates[0]
        pending = self._store.find_pending(detection.embedding)
        if pending is not None and pending.similarity >= threshold:
            return pending
        return None

    def _record_access(self, match: Match, kind: GreetingKind) -> None:
        """Invia l'accesso al server. Un evento non accodato non blocca il saluto a schermo."""
        try:
            self._store.record_greeting(match.person_id, match.similarity)
        except StoreError as exc:
            logger.error("Accesso di %s non inviato al server: %s", match.full_name, exc)
        logger.info("Saluto %s: %s (similarita %.2f)", kind.name, match.full_name, match.similarity)

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
            self._store.add_continuous_embedding(
                match.person_id,
                (mean_vector / norm).astype(np.float32),
                quality=mean_quality,
            )
        except StoreError as exc:
            logger.error("Apprendimento continuo non riuscito per %s: %s", match.full_name, exc)
            return
        self._last_learning_update[match.person_id] = now
