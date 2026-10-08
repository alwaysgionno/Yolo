"""Registrazione di un nuovo volto.

Flusso (macchina a stati gestita dal thread della UI):

    IDLE -> NAME_INPUT (l'utente digita il nome)
         -> CAPTURING (acquisizione di 5-10 campioni validi)
         -> COMPLETED (salvataggio nel database)   oppure   CANCELLED (annullato o scaduto)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from enum import Enum, auto

import numpy as np

from .config import Settings
from .database_manager import DatabaseError, DatabaseManager
from .models import FaceDetection

logger = logging.getLogger(__name__)

ESC_KEY = 27
ENTER_KEYS = (10, 13)
BACKSPACE_KEYS = (8, 127)
PRINTABLE_KEYS = range(32, 127)

MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 40
MIN_SAMPLE_INTERVAL_S = 0.4
UNKNOWN_CONFIRM_RESULTS = 3    # risultati consecutivi con il volto sconosciuto prima di chiedere il nome
IDENTITY_CONSISTENCY = 0.50    # similarita minima con il primo campione (stessa persona)
DUPLICATE_SIMILARITY = 0.985   # campioni quasi identici: scartati per garantire varieta
NAME_INPUT_TIMEOUT_S = 60.0
CAPTURE_TIMEOUT_S = 30.0


class EnrollmentState(Enum):
    """Stati della registrazione."""

    IDLE = auto()
    NAME_INPUT = auto()
    CAPTURING = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class EnrollmentSession:
    """Gestisce una registrazione alla volta. Usata solo dal thread della UI."""

    def __init__(
        self,
        settings: Settings,
        database: DatabaseManager,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._db = database
        self._clock = clock

        self.state = EnrollmentState.IDLE
        self.name = ""
        self.feedback = ""       # suggerimento mostrato durante la cattura
        self.error = ""          # motivo dell'ultimo errore o annullamento
        self.person_id: int | None = None

        self._samples: list[np.ndarray] = []
        self._qualities: list[float] = []
        self._reference: np.ndarray | None = None
        self._state_entered_at = 0.0
        self._last_sample_at = float("-inf")
        self._snoozed_until = 0.0
        self._unknown_since: float | None = None
        self._unknown_streak = 0

    @property
    def is_active(self) -> bool:
        """True mentre l'utente sta digitando il nome o la cattura e in corso."""
        return self.state in (EnrollmentState.NAME_INPUT, EnrollmentState.CAPTURING)

    @property
    def progress(self) -> tuple[int, int]:
        """Restituisce (campioni acquisiti, campioni richiesti)."""
        return len(self._samples), self._settings.enrollment_samples

    def consider_start(self, detections: Sequence[FaceDetection]) -> bool:
        """Valuta un nuovo risultato di inferenza e, se serve, avvia la registrazione.

        La registrazione parte solo quando un unico volto, sconosciuto e abbastanza grande,
        resta nel fotogramma per piu risultati consecutivi: un singolo fotogramma mosso o
        di profilo non basta a far comparire la richiesta del nome. Va chiamata una volta
        per ogni risultato di inferenza.

        Returns:
            True se la registrazione e stata avviata.
        """
        if self.state is not EnrollmentState.IDLE:
            return False
        now = self._clock()
        if now < self._snoozed_until or not self._is_candidate(detections):
            self._reset_unknown_streak()
            return False

        if self._unknown_since is None:
            self._unknown_since = now
        self._unknown_streak += 1
        if self._unknown_streak < UNKNOWN_CONFIRM_RESULTS:
            return False
        if now - self._unknown_since < self._settings.unknown_confirm_seconds:
            return False
        self.start()
        return True

    def start(self) -> None:
        """Avvia la registrazione chiedendo il nome."""
        self._clear_buffers()
        self._reset_unknown_streak()
        self.name = ""
        self.error = ""
        self.feedback = ""
        self.person_id = None
        self._set_state(EnrollmentState.NAME_INPUT)

    def handle_key(self, key: int) -> None:
        """Gestisce i tasti premuti durante la registrazione (ESC, INVIO, BACKSPACE, caratteri)."""
        if key == ESC_KEY and self.is_active:
            self._cancel("Registrazione annullata", snooze=True)
            return
        if self.state is not EnrollmentState.NAME_INPUT:
            return
        if key in ENTER_KEYS:
            self._confirm_name()
        elif key in BACKSPACE_KEYS:
            self.name = self.name[:-1]
            self.error = ""
        elif key in PRINTABLE_KEYS and len(self.name) < MAX_NAME_LENGTH:
            self.name += chr(key)
            self.error = ""

    def offer_frame(self, detections: Sequence[FaceDetection]) -> None:
        """Valuta i volti di un nuovo risultato di inferenza durante la cattura."""
        if self.state is not EnrollmentState.CAPTURING:
            return
        if not detections:
            self.feedback = "Nessun volto rilevato: guarda la webcam"
        elif len(detections) > 1:
            self.feedback = "Piu volti rilevati: resta solo tu davanti alla webcam"
        else:
            self.offer_detection(detections[0])

    def offer_detection(self, detection: FaceDetection) -> None:
        """Valuta il volto del fotogramma corrente e, se valido, lo aggiunge ai campioni.

        Ogni chiamata e idempotente rispetto al tempo: campioni troppo ravvicinati vengono ignorati.
        """
        if self.state is not EnrollmentState.CAPTURING or detection.embedding is None:
            return
        if self._clock() - self._last_sample_at < MIN_SAMPLE_INTERVAL_S:
            return

        settings = self._settings
        if detection.match is not None:
            # Il volto appartiene a un profilo esistente: salvarlo creerebbe un duplicato
            self.feedback = f"Volto gia registrato come {detection.match.full_name}"
            return
        if detection.face_width < settings.min_face_width_px:
            self.feedback = "Avvicinati alla webcam"
            return
        if detection.sharpness < settings.min_sharpness:
            self.feedback = (
                f"Immagine sfocata ({detection.sharpness:.0f}/{settings.min_sharpness:.0f}): "
                "stai fermo e migliora la luce"
            )
            return

        embedding = detection.embedding
        if self._reference is None:
            self._reference = embedding
        elif float(np.dot(embedding, self._reference)) < IDENTITY_CONSISTENCY:
            self.feedback = "Volto diverso dal primo campione: resta solo tu davanti alla webcam"
            return

        if self._samples and max(float(np.dot(embedding, prev)) for prev in self._samples) > DUPLICATE_SIMILARITY:
            self.feedback = "Muovi un po la testa o cambia l'inclinazione"
            return

        self._samples.append(embedding)
        self._qualities.append(detection.sharpness)
        self._last_sample_at = self._clock()
        self.feedback = "Campione acquisito"
        logger.info("Campione di registrazione %d/%d acquisito", len(self._samples), settings.enrollment_samples)

        if len(self._samples) >= settings.enrollment_samples:
            self._save()

    def tick(self) -> None:
        """Annulla la registrazione se rimane ferma troppo a lungo in uno stato."""
        elapsed = self._clock() - self._state_entered_at
        if self.state is EnrollmentState.NAME_INPUT and elapsed > NAME_INPUT_TIMEOUT_S:
            self._cancel("Tempo scaduto: nessun nome inserito", snooze=True)
        elif self.state is EnrollmentState.CAPTURING and elapsed > CAPTURE_TIMEOUT_S:
            self._cancel("Cattura non completata: riprova", snooze=True)

    def reset(self) -> None:
        """Riporta la sessione allo stato IDLE dopo aver gestito un esito finale."""
        self._clear_buffers()
        self._reset_unknown_streak()
        self._set_state(EnrollmentState.IDLE)

    def _is_candidate(self, detections: Sequence[FaceDetection]) -> bool:
        """True se nel fotogramma c'e un solo volto, sconosciuto e abbastanza grande da registrare."""
        if len(detections) != 1:
            return False
        candidate = detections[0]
        return (
            candidate.match is None
            and candidate.embedding is not None
            and candidate.face_width >= self._settings.min_face_width_px
        )

    def _confirm_name(self) -> None:
        """Valida il nome digitato e passa alla fase di cattura."""
        name = " ".join(self.name.split())
        if len(name) < MIN_NAME_LENGTH:
            self.error = f"Il nome deve avere almeno {MIN_NAME_LENGTH} caratteri"
            return
        try:
            if self._db.person_exists(name):
                self.error = "Nome gia presente nel database"
                return
        except DatabaseError as exc:
            logger.error("Verifica del nome non riuscita: %s", exc)
            self._cancel("Database non disponibile", snooze=True)
            return
        self.name = name
        self._set_state(EnrollmentState.CAPTURING)

    def _save(self) -> None:
        """Salva la persona e i campioni nel database."""
        try:
            self.person_id = self._db.create_person(self.name, self._samples, self._qualities)
        except DatabaseError as exc:
            logger.error("Salvataggio del profilo non riuscito: %s", exc)
            self._cancel("Salvataggio nel database fallito", snooze=True)
            return
        self._set_state(EnrollmentState.COMPLETED)

    def _cancel(self, message: str, snooze: bool) -> None:
        """Annulla la registrazione; con `snooze` il volto non viene riproposto per un po'."""
        self.error = message
        if snooze:
            self._snoozed_until = self._clock() + self._settings.unknown_snooze_seconds
        self._set_state(EnrollmentState.CANCELLED)

    def _set_state(self, state: EnrollmentState) -> None:
        self.state = state
        self._state_entered_at = self._clock()

    def _clear_buffers(self) -> None:
        self._samples.clear()
        self._qualities.clear()
        self._reference = None
        self._last_sample_at = float("-inf")

    def _reset_unknown_streak(self) -> None:
        self._unknown_since = None
        self._unknown_streak = 0
