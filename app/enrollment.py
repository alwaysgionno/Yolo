"""Registrazione di un nuovo volto.

Flusso (macchina a stati gestita dal thread della UI):

    IDLE -> NAME_INPUT (l'utente digita il nome)
         -> CAPTURING (acquisizione di 5-10 campioni validi)
         -> COMPLETED (profilo inviato al server)   oppure   CANCELLED (annullato o scaduto)

La sessione segue un solo volto, quello scelto dal NewFaceSelector: gli altri volti
presenti nel fotogramma vengono ignorati.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from enum import Enum, auto

import numpy as np

from .config import Settings
from .person_store import PersonStore, StoreError
from .models import FaceDetection
from .new_face_selector import NewFaceSelector, find_same_face

logger = logging.getLogger(__name__)

ESC_KEY = 27
ENTER_KEYS = (10, 13)
BACKSPACE_KEYS = (8, 127)
PRINTABLE_KEYS = range(32, 127)

MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 40
MIN_SAMPLE_INTERVAL_S = 0.4
DUPLICATE_SIMILARITY = 0.985     # campioni quasi identici: scartati per garantire varieta
KNOWN_FACE_CONFIRMATIONS = 2     # riconoscimenti di fila che rivelano un volto gia registrato
NAME_INPUT_TIMEOUT_S = 60.0
CANDIDATE_LOST_TIMEOUT_S = 10.0  # senza volto e senza nome digitato la richiesta viene ritirata
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
        store: PersonStore,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._store = store
        self._clock = clock
        self._selector = NewFaceSelector(settings, clock)

        self.state = EnrollmentState.IDLE
        self.name = ""
        self.feedback = ""       # suggerimento mostrato durante la cattura
        self.error = ""          # motivo dell'ultimo errore o annullamento
        self.person_id: str | None = None

        self._samples: list[np.ndarray] = []
        self._qualities: list[float] = []
        self._reference: np.ndarray | None = None  # embedding del volto in registrazione
        self._known_streak = 0
        self._state_entered_at = 0.0
        self._candidate_seen_at = 0.0
        self._last_sample_at = float("-inf")
        self._snoozed_until = 0.0

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

        La registrazione parte quando il NewFaceSelector conferma un volto sconosciuto,
        fuori dal periodo di snooze: un singolo fotogramma mosso o di profilo non basta a
        far comparire la richiesta del nome. Va chiamata una volta per ogni risultato di inferenza.

        Returns:
            True se la registrazione e stata avviata.
        """
        if self.state is not EnrollmentState.IDLE:
            return False
        if self._clock() < self._snoozed_until:
            self._selector.reset()
            return False
        candidate = self._selector.update(detections)
        if candidate is None:
            return False
        self.start(candidate)
        return True

    def start(self, candidate: FaceDetection) -> None:
        """Avvia la registrazione del volto indicato chiedendo il nome."""
        if candidate.embedding is None:
            raise ValueError("Il volto da registrare deve avere un embedding")
        self._clear_buffers()
        self._selector.reset()
        self.name = ""
        self.error = ""
        self.feedback = ""
        self.person_id = None
        self._reference = candidate.embedding
        self._candidate_seen_at = self._clock()
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

    def candidate_in(self, detections: Iterable[FaceDetection]) -> FaceDetection | None:
        """Restituisce, tra i volti indicati, quello in registrazione (None se assente o a sessione inattiva)."""
        if not self.is_active or self._reference is None:
            return None
        return find_same_face(detections, self._reference)

    def offer_frame(self, detections: Sequence[FaceDetection]) -> None:
        """Valuta i volti di un nuovo risultato di inferenza mentre la registrazione e attiva.

        Viene seguito solo il volto selezionato all'avvio. Durante la cattura, se e valido,
        diventa un nuovo campione. Se risulta gia registrato in piu risultati consecutivi,
        la registrazione viene annullata. Va chiamata una volta per ogni risultato di inferenza.
        """
        candidate = self.candidate_in(detections)
        if candidate is None:
            if self.state is EnrollmentState.CAPTURING:
                self.feedback = (
                    "Volto diverso da quello in registrazione: torna davanti alla webcam"
                    if detections
                    else "Nessun volto rilevato: guarda la webcam"
                )
            return

        self._candidate_seen_at = self._clock()
        if candidate.match is not None:
            # Il volto appartiene a un profilo esistente: salvarlo creerebbe un duplicato
            self.feedback = f"Volto già registrato come {candidate.match.full_name}"
            self._known_streak += 1
            if self._known_streak >= KNOWN_FACE_CONFIRMATIONS:
                self._cancel(self.feedback, snooze=True)
            return
        self._known_streak = 0

        if self.state is EnrollmentState.CAPTURING:
            self._add_sample(candidate)

    def tick(self) -> None:
        """Annulla la registrazione se rimane ferma troppo a lungo in uno stato."""
        now = self._clock()
        elapsed = now - self._state_entered_at
        if self.state is EnrollmentState.NAME_INPUT:
            if elapsed > NAME_INPUT_TIMEOUT_S:
                self._cancel("Tempo scaduto: nessun nome inserito", snooze=True)
            elif not self.name and now - self._candidate_seen_at > CANDIDATE_LOST_TIMEOUT_S:
                self._cancel("Registrazione annullata: volto non più presente", snooze=False)
        elif self.state is EnrollmentState.CAPTURING and elapsed > CAPTURE_TIMEOUT_S:
            self._cancel("Cattura non completata: riprova", snooze=True)

    def reset(self) -> None:
        """Riporta la sessione allo stato IDLE dopo aver gestito un esito finale."""
        self._clear_buffers()
        self._selector.reset()
        self._set_state(EnrollmentState.IDLE)

    def _add_sample(self, detection: FaceDetection) -> None:
        """Valuta il volto in registrazione e, se valido, lo aggiunge ai campioni.

        Campioni troppo ravvicinati nel tempo vengono ignorati.
        """
        embedding = detection.embedding
        if embedding is None or self._clock() - self._last_sample_at < MIN_SAMPLE_INTERVAL_S:
            return

        settings = self._settings
        if detection.face_width < settings.min_face_width_px:
            self.feedback = "Avvicinati alla webcam"
            return
        if detection.sharpness < settings.min_sharpness:
            self.feedback = (
                f"Immagine sfocata ({detection.sharpness:.0f}/{settings.min_sharpness:.0f}): "
                "stai fermo e migliora la luce"
            )
            return
        if self._samples and max(float(np.dot(embedding, prev)) for prev in self._samples) > DUPLICATE_SIMILARITY:
            self.feedback = "Muovi un po' la testa o cambia l'inclinazione"
            return

        if not self._samples:
            self._reference = embedding  # il primo campione valido diventa il riferimento dell'identita
        self._samples.append(embedding)
        self._qualities.append(detection.sharpness)
        self._last_sample_at = self._clock()
        self.feedback = "Campione acquisito"
        logger.info("Campione di registrazione %d/%d acquisito", len(self._samples), settings.enrollment_samples)

        if len(self._samples) >= settings.enrollment_samples:
            self._save()

    def _confirm_name(self) -> None:
        """Valida il nome digitato e passa alla fase di cattura."""
        name = " ".join(self.name.split())
        if len(name) < MIN_NAME_LENGTH:
            self.error = f"Il nome deve avere almeno {MIN_NAME_LENGTH} caratteri"
            return
        try:
            if self._store.person_exists(name):
                self.error = "Nome già registrato"
                return
        except StoreError as exc:
            logger.error("Verifica del nome non riuscita: %s", exc)
            self._cancel("Elenco delle persone non disponibile", snooze=True)
            return
        self.name = name
        self._set_state(EnrollmentState.CAPTURING)

    def _save(self) -> None:
        """Registra la persona con i suoi campioni: galleria locale subito, database tramite il server."""
        try:
            self.person_id = self._store.create_person(self.name, self._samples, self._qualities)
        except StoreError as exc:
            logger.error("Salvataggio del profilo non riuscito: %s", exc)
            self._cancel("Invio del profilo al server fallito", snooze=True)
            return
        self._set_state(EnrollmentState.COMPLETED)

    def _cancel(self, message: str, snooze: bool) -> None:
        """Annulla la registrazione; con `snooze` nessun volto viene riproposto per un po'."""
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
        self._known_streak = 0
        self._last_sample_at = float("-inf")
