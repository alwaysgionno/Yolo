"""Galleria locale: copia in memoria delle persone registrate, usata per il riconoscimento.

La fonte di verita e il database del server. Il server pubblica un messaggio MQTT retained
per ogni persona (`gallery/{uid}`) e l'elenco degli uid validi (`gallery_info`): la
telecamera li riceve alla connessione e a ogni modifica. Il confronto coseno avviene qui,
quindi il riconoscimento continua anche quando il server non e raggiungibile.

La galleria vive solo in memoria: alla chiusura dell'applicazione non resta nulla su disco.

Thread: i messaggi arrivano dal thread di rete MQTT, `find_nearest` viene chiamato dal
thread di inferenza, `add_local` e `name_exists` dal thread della UI.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from . import contract
from .contract import EMBEDDING_DIMENSION, ContractError
from .models import Match

logger = logging.getLogger(__name__)


@dataclass
class _Person:
    uid: str
    full_name: str
    last_access: float | None  # secondi Unix dell'ultimo saluto; None se mai salutata
    embeddings: np.ndarray     # forma (n, 512), righe normalizzate; vuoto per le persone non attive


class Gallery:
    """Persone registrate e ricerca del volto piu simile."""

    def __init__(self, wall_clock: Callable[[], float] = time.time) -> None:
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._persons: dict[str, _Person] = {}
        # Persone registrate qui e non ancora confermate dal server: l'elenco del server non le rimuove
        self._pending: set[str] = set()
        self._expected: frozenset[str] | None = None  # uid annunciati dal server
        self._ready = False
        # Matrice di tutti gli embedding e, riga per riga, la persona a cui appartengono.
        # Viene sostituita per intero a ogni modifica: chi cerca legge sempre una coppia coerente.
        self._index: tuple[np.ndarray, list[_Person]] = (_empty_matrix(), [])

    @property
    def ready(self) -> bool:
        """True da quando la galleria e stata ricevuta per intero almeno una volta."""
        return self._ready

    def __len__(self) -> int:
        with self._lock:
            return len(self._persons)

    def apply_person_message(self, person_uid: str, payload: bytes) -> None:
        """Applica un messaggio `gallery/{uid}`: aggiorna la persona, oppure la rimuove se e vuoto.

        Raises:
            ContractError: se il messaggio non rispetta il contratto (la galleria non cambia).
        """
        if not payload:
            with self._lock:
                self._pending.discard(person_uid)
                if self._persons.pop(person_uid, None) is not None:
                    logger.info("Persona rimossa dalla galleria (uid=%s)", person_uid)
                    self._rebuild()
            return

        data = contract.decode_json(payload)
        full_name = data.get("full_name")
        raw_embeddings = data.get("embeddings")
        if not isinstance(full_name, str) or not isinstance(raw_embeddings, list):
            raise ContractError("voce di galleria incompleta")
        last_access = data.get("last_access_at")
        person = _Person(
            uid=person_uid,
            full_name=full_name,
            last_access=contract.parse_timestamp(last_access) if last_access is not None else None,
            embeddings=_normalized_rows([contract.decode_embedding(item) for item in raw_embeddings]),
        )
        with self._lock:
            self._persons[person_uid] = person
            self._pending.discard(person_uid)
            self._rebuild()
            self._update_ready()

    def apply_info_message(self, payload: bytes) -> None:
        """Applica `gallery_info`: l'elenco degli uid validi. Chi non vi compare viene rimosso.

        Raises:
            ContractError: se il messaggio non rispetta il contratto (la galleria non cambia).
        """
        person_uids = contract.decode_json(payload).get("person_uids")
        if not isinstance(person_uids, list) or not all(isinstance(uid, str) for uid in person_uids):
            raise ContractError("gallery_info: atteso l'elenco person_uids")
        with self._lock:
            self._expected = frozenset(person_uids)
            removed = [uid for uid in self._persons if uid not in self._expected and uid not in self._pending]
            for uid in removed:
                del self._persons[uid]
            if removed:
                logger.info("Persone non piu presenti sul server rimosse dalla galleria: %d", len(removed))
                self._rebuild()
            self._update_ready()

    def add_local(self, person_uid: str, full_name: str, embeddings: Sequence[np.ndarray]) -> None:
        """Aggiunge una persona appena registrata, senza attendere la conferma del server."""
        person = _Person(person_uid, full_name, self._wall_clock(), _normalized_rows(embeddings))
        with self._lock:
            self._persons[person_uid] = person
            self._pending.add(person_uid)
            self._rebuild()

    def mark_greeted(self, person_uid: str) -> None:
        """Registra l'istante dell'ultimo saluto, come fara il server quando ricevera l'evento."""
        with self._lock:
            person = self._persons.get(person_uid)
            if person is not None:
                person.last_access = self._wall_clock()

    def name_exists(self, full_name: str) -> bool:
        """Verifica se esiste gia una persona con lo stesso nome (senza distinguere maiuscole)."""
        wanted = full_name.lower()
        with self._lock:
            return any(person.full_name.lower() == wanted for person in self._persons.values())

    def find_nearest(self, embedding: np.ndarray, limit: int) -> list[Match]:
        """Restituisce le `limit` corrispondenze piu vicine, ordinate per similarita coseno decrescente.

        Ogni corrispondenza riporta anche i secondi trascorsi dall'ultimo saluto della persona
        (None se non e mai stata salutata).
        """
        matrix, owners = self._index
        if not owners:
            return []
        norm = float(np.linalg.norm(embedding))
        if norm == 0.0:
            return []
        similarities = matrix @ (embedding.astype(np.float32) / norm)
        now = self._wall_clock()
        matches = []
        for row in np.argsort(-similarities)[:limit]:
            person = owners[row]
            last_access = person.last_access
            matches.append(
                Match(
                    person_id=person.uid,
                    full_name=person.full_name,
                    similarity=float(similarities[row]),
                    last_access_age_s=max(0.0, now - last_access) if last_access is not None else None,
                )
            )
        return matches

    def _rebuild(self) -> None:
        """Ricostruisce la matrice di ricerca. Va chiamato con il lock acquisito."""
        owners: list[_Person] = []
        blocks: list[np.ndarray] = []
        for person in self._persons.values():
            if len(person.embeddings):
                blocks.append(person.embeddings)
                owners.extend([person] * len(person.embeddings))
        self._index = (np.concatenate(blocks) if blocks else _empty_matrix(), owners)

    def _update_ready(self) -> None:
        """La galleria e pronta quando sono arrivate tutte le persone annunciate dal server."""
        if not self._ready and self._expected is not None and self._expected <= self._persons.keys():
            self._ready = True
            logger.info("Galleria ricevuta dal server: %d persone", len(self._persons))


def _empty_matrix() -> np.ndarray:
    return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)


def _normalized_rows(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Impila gli embedding in una matrice con righe di norma 1 (il prodotto scalare e la similarita coseno)."""
    if not embeddings:
        return _empty_matrix()
    matrix = np.stack(embeddings).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
