"""Regole dei saluti: quando spetta "Benvenuto" e quando "Bentornato".

Il saluto dipende dalla presenza della persona davanti alla webcam, non dal tempo
trascorso dall'ultimo saluto:

- "Benvenuto" spetta solo a chi non e mai stato salutato prima;
- "Bentornato" spetta a chi rientra nel campo visivo dopo un'assenza di almeno
  `absence_seconds`, misurata dall'ultimo avvistamento.

Chi resta davanti alla webcam non viene piu salutato, per quanto a lungo rimanga.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .models import GreetingKind


class GreetingPolicy:
    """Tiene traccia degli avvistamenti e decide il saluto. Usata solo dal thread di inferenza."""

    def __init__(self, absence_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._absence_seconds = absence_seconds
        self._clock = clock
        self._last_seen: dict[int, float] = {}

    def observe(self, person_id: int, last_access_age_s: float | None) -> GreetingKind | None:
        """Registra un avvistamento e restituisce il saluto dovuto, oppure None.

        Args:
            person_id: persona riconosciuta nel fotogramma corrente.
            last_access_age_s: secondi dall'ultimo accesso salvato nel database, None se la
                persona non e mai stata salutata. Conta solo al primo avvistamento dopo
                l'avvio, quando in memoria non c'e ancora uno storico.
        """
        now = self._clock()
        last_seen = self._last_seen.get(person_id)
        self._last_seen[person_id] = now

        if last_seen is not None:
            absence = now - last_seen
        elif last_access_age_s is None:
            return GreetingKind.WELCOME
        else:
            absence = last_access_age_s
        return GreetingKind.WELCOME_BACK if absence >= self._absence_seconds else None
