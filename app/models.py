"""Strutture dati condivise tra i moduli dell'applicazione."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

BBox = tuple[int, int, int, int]
"""Bounding box in pixel nel formato (x1, y1, x2, y2)."""

Point = tuple[float, float]
"""Punto in pixel nel formato (x, y)."""


@dataclass(frozen=True)
class Match:
    """Persona riconosciuta, con la similarita coseno rispetto al vettore piu vicino."""

    person_id: str  # uid della persona: lo stesso usato dal server
    full_name: str
    similarity: float  # tra -1 e 1; 1 significa vettori identici
    last_access_age_s: float | None = None  # secondi dall'ultimo saluto noto; None se mai salutata
    # True se il volto appartiene a una registrazione in attesa dell'approvazione di un amministratore:
    # la persona non va salutata ne usata per l'apprendimento continuo
    pending: bool = False


@dataclass(frozen=True)
class EnrollmentDecision:
    """Esito di una richiesta di registrazione inviata da questa telecamera, deciso sul server."""

    person_id: str
    full_name: str  # nome definitivo: l'amministratore puo averlo corretto
    approved: bool


class GreetingKind(Enum):
    """Tipo di saluto dovuto a una persona riconosciuta."""

    WELCOME = auto()       # "Benvenuto": primo riconoscimento in assoluto
    WELCOME_BACK = auto()  # "Bentornato": rientro dopo un'assenza dal campo visivo
    WAVE = auto()          # "Ciao": la persona ha salutato con la mano (solo a schermo, non inviato al server)


@dataclass(frozen=True)
class Greeting:
    """Saluto deciso dal thread di inferenza e mostrato dal thread della UI."""

    kind: GreetingKind
    full_name: str


@dataclass(frozen=True)
class ArmPose:
    """Un braccio stimato dal modello di posa, usato per riconoscere il saluto con la mano."""

    side: int  # 0 sinistro, 1 destro (dal punto di vista della persona)
    shoulder: Point
    wrist: Point
    elbow: Point | None = None  # None se il gomito non e affidabile


@dataclass(frozen=True)
class BodyPose:
    """Punti del corpo di una persona utili al saluto con la mano. Il naso serve ad associarla al volto."""

    nose: Point
    shoulder_width: float      # distanza tra le spalle in pixel: l'unita di misura dei movimenti
    arms: tuple[ArmPose, ...]  # solo i bracci con spalla e polso affidabili


@dataclass
class FaceDetection:
    """Un volto rilevato in un fotogramma e il relativo embedding biometrico."""

    bbox: BBox
    detector_confidence: float
    embedding: np.ndarray | None = None  # vettore L2-normalizzato a 512 dimensioni
    face_width: float = 0.0              # larghezza del volto in pixel (sul frame originale)
    sharpness: float = 0.0               # varianza del Laplaciano sul volto: piu alto = piu nitido
    match: Match | None = None           # valorizzato dal RecognitionService
    patch: np.ndarray | None = None      # volto allineato in scala di grigi, per la verifica anti-spoofing
    live: bool | None = None             # esito della verifica anti-spoofing; None se non e stata eseguita
    liveness_wait_s: float = 0.0         # da quanti secondi il volto attende di essere verificato
    photo: np.ndarray | None = None      # ritaglio a colori del volto (BGR), inviato con la richiesta di registrazione


@dataclass
class FrameResult:
    """Esito dell'inferenza su un fotogramma, pubblicato al thread della UI."""

    frame_id: int
    detections: list[FaceDetection] = field(default_factory=list)
    inference_ms: float = 0.0
    gallery_ok: bool = True  # False se la galleria non era ancora arrivata dal server: volti non confrontati
    created_at: float = field(default_factory=time.monotonic)  # istante di pubblicazione (orologio monotono)
