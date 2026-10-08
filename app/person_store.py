"""Archivio delle persone visto dalla telecamera: letture sulla galleria locale, scritture via MQTT.

Sostituisce l'accesso diretto a PostgreSQL. Le letture (riconoscimento, controllo del nome)
usano la galleria in memoria; le scritture (registrazione, saluto, apprendimento continuo)
diventano eventi pubblicati al server, che li salva nel database.

Viene usato sia dal thread di inferenza sia dal thread della UI.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Sequence

import numpy as np

from . import contract
from .gallery import Gallery
from .models import Match
from .mqtt_link import LinkError, MqttLink

logger = logging.getLogger(__name__)


class StoreError(RuntimeError):
    """L'archivio non puo rispondere: galleria non ancora ricevuta o evento non accodato."""


class PersonStore:
    """Incapsula tutte le letture e le scritture sulle persone registrate."""

    def __init__(self, gallery: Gallery, link: MqttLink, wall_clock: Callable[[], float] = time.time) -> None:
        self._gallery = gallery
        self._link = link
        self._wall_clock = wall_clock

    def _require_gallery(self) -> None:
        # Senza galleria ogni volto sembrerebbe sconosciuto e verrebbe proposto per la registrazione
        if not self._gallery.ready:
            raise StoreError("galleria non ancora ricevuta dal server")

    def person_exists(self, full_name: str) -> bool:
        """Verifica se esiste gia una persona con lo stesso nome (senza distinguere maiuscole)."""
        self._require_gallery()
        return self._gallery.name_exists(full_name)

    def find_nearest(self, embedding: np.ndarray, limit: int) -> list[Match]:
        """Restituisce le `limit` corrispondenze piu vicine, ordinate per similarita decrescente."""
        self._require_gallery()
        return self._gallery.find_nearest(embedding, limit)

    def create_person(
        self,
        full_name: str,
        embeddings: Sequence[np.ndarray],
        qualities: Sequence[float],
    ) -> str:
        """Registra una nuova persona: la pubblica al server e la aggiunge subito alla galleria locale.

        La registrazione vale come primo accesso: la persona appena accolta con "Benvenuto" non
        riceve un secondo saluto al primo riconoscimento. Restituisce l'uid assegnato.
        """
        if not embeddings:
            raise ValueError("Serve almeno un embedding per registrare una persona")
        if len(embeddings) != len(qualities):
            raise ValueError("Ogni embedding deve avere il proprio punteggio di qualita")
        self._require_gallery()
        person_uid = str(uuid.uuid4())
        self._publish(
            contract.TOPIC_EVENT_ENROLLED,
            {
                "person_uid": person_uid,
                "full_name": full_name,
                "samples": [
                    {"embedding": contract.encode_embedding(embedding), "quality": float(quality)}
                    for embedding, quality in zip(embeddings, qualities)
                ],
            },
        )
        self._gallery.add_local(person_uid, full_name, embeddings)
        logger.info("Nuova persona registrata: %s (uid=%s, campioni=%d)", full_name, person_uid, len(embeddings))
        return person_uid

    def record_greeting(self, person_id: str, similarity: float) -> None:
        """Pubblica il saluto e aggiorna l'ultimo accesso nella galleria locale."""
        self._publish(contract.TOPIC_EVENT_GREETED, {"person_uid": person_id, "similarity": float(similarity)})
        self._gallery.mark_greeted(person_id)

    def add_continuous_embedding(self, person_id: str, embedding: np.ndarray, quality: float) -> None:
        """Pubblica un embedding di apprendimento continuo: tornera nella galleria quando il server lo avra salvato."""
        self._publish(
            contract.TOPIC_EVENT_EMBEDDING,
            {"person_uid": person_id, "embedding": contract.encode_embedding(embedding), "quality": float(quality)},
        )
        logger.info("Apprendimento continuo inviato al server (uid=%s)", person_id)

    def _publish(self, topic: str, payload: dict) -> None:
        """Completa l'evento con identificativo e orario e lo affida al collegamento MQTT."""
        event = {
            "event_id": str(uuid.uuid4()),  # il server lo usa per scartare un messaggio ricevuto due volte
            "occurred_at": contract.format_timestamp(self._wall_clock()),
            **payload,
        }
        try:
            self._link.publish_event(topic, event)
        except LinkError as exc:
            raise StoreError(str(exc)) from exc
