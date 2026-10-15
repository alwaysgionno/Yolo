"""Saluto con la mano: riconosce quando una persona alza la mano e la agita a destra e a sinistra.

Il modello di posa (`PoseEstimator`) fornisce per ogni persona naso, spalle, gomiti e polsi. Qui:
    1. le pose vengono seguite da un fotogramma all'altro in base alla posizione del naso. Il gesto
       appartiene alla posa e non al volto: quando la mano passa davanti alla faccia il volto puo non
       essere riconosciuto per qualche fotogramma, ma il gesto non si interrompe;
    2. una posa prende il nome della persona riconosciuta (e verificata) il cui volto contiene il suo
       naso, e lo conserva per qualche secondo;
    3. per ogni braccio alzato si segue la posizione orizzontale del polso rispetto alla spalla, in
       larghezze delle spalle, cosi non dipende dalla distanza dalla webcam;
    4. il saluto e riconosciuto quando il polso cambia direzione abbastanza volte e con un'ampiezza
       sufficiente (`count_swings`).

Tutto lo stato viene usato solo dal thread di inferenza.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .models import ArmPose, BBox, BodyPose, Match

WINDOW_SECONDS = 2.5      # il gesto deve stare in questo intervallo
MAX_GAP_SECONDS = 1.0     # senza il polso per piu di questo tempo, il gesto ricomincia da capo
# Ampiezza minima di un movimento, in larghezze delle spalle (circa 6 cm): il tremolio dei punti stimati
# resta sotto, il polso di chi saluta la supera
MIN_SWING = 0.15
# Movimenti alternati richiesti: andata e ritorno. Con pochi fotogrammi al secondo una parte delle
# inversioni non viene vista, quindi chiederne di piu farebbe perdere molti saluti veri
MIN_SWINGS = 2
# Mano alzata: polso sopra la spalla, oppure avambraccio rivolto verso l'alto (polso sopra il gomito
# di almeno questa frazione della larghezza delle spalle), come quando si saluta con la mano all'altezza del petto
FOREARM_RAISE = 0.35
TRACK_TIMEOUT_SECONDS = 1.0   # una posa non piu vista da questo tempo viene dimenticata
TRACK_MAX_JUMP = 0.75         # spostamento massimo del naso tra due fotogrammi, in larghezze delle spalle
NAME_MEMORY_SECONDS = 3.0     # per quanto una posa ricorda la persona riconosciuta nel suo volto


def count_swings(positions: Sequence[float], min_swing: float = MIN_SWING) -> int:
    """Conta i movimenti alternati (avanti e indietro) piu ampi di `min_swing`.

    Le oscillazioni piu piccole di `min_swing`, come il tremolio dei punti stimati, non contano.
    """
    if not positions:
        return 0
    swings = 0
    direction = 0  # +1 verso destra nell'immagine, -1 verso sinistra, 0 non ancora deciso
    low = high = extreme = positions[0]
    for position in positions[1:]:
        if direction == 0:
            low, high = min(low, position), max(high, position)
            if position - low >= min_swing:
                direction, extreme, swings = 1, position, 1
            elif high - position >= min_swing:
                direction, extreme, swings = -1, position, 1
        elif (position - extreme) * direction > 0:
            extreme = position  # il movimento continua nella stessa direzione
        elif (extreme - position) * direction >= min_swing:
            direction, extreme = -direction, position
            swings += 1
    return swings


def is_raised(arm: ArmPose, shoulder_width: float) -> bool:
    """True se la mano e alzata: polso sopra la spalla o avambraccio verso l'alto (y cresce verso il basso)."""
    if arm.wrist[1] < arm.shoulder[1]:
        return True
    return arm.elbow is not None and arm.elbow[1] - arm.wrist[1] >= FOREARM_RAISE * shoulder_width


def _contains(bbox: BBox, point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


@dataclass
class _Track:
    """Una persona seguita tra un fotogramma e l'altro attraverso la sua posa."""

    nose: tuple[float, float]
    shoulder_width: float
    last_seen: float
    wrists: dict[int, deque[tuple[float, float]]] = field(default_factory=dict)  # lato -> (istante, posizione)
    person: Match | None = None
    person_seen_at: float = 0.0


class WaveDetector:
    """Segue le pose delle persone inquadrate e segnala chi saluta con la mano."""

    def __init__(self, cooldown_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._tracks: list[_Track] = []
        self._last_wave: dict[str, float] = {}

    def update(self, poses: Sequence[BodyPose], people: Sequence[tuple[BBox, Match]]) -> list[Match]:
        """Elabora le pose del fotogramma corrente e restituisce le persone che hanno appena salutato.

        Args:
            poses: pose stimate nel fotogramma.
            people: riquadro del volto e persona, solo per le persone che possono essere salutate
                (riconosciute, approvate e verificate dall'anti-spoofing).
        """
        now = self._clock()
        self._tracks = [track for track in self._tracks if now - track.last_seen <= TRACK_TIMEOUT_SECONDS]
        waved: list[Match] = []
        for track, pose in self._match(poses, now):
            for bbox, match in people:
                if _contains(bbox, pose.nose):
                    track.person, track.person_seen_at = match, now
                    break
            if track.person is not None and now - track.person_seen_at > NAME_MEMORY_SECONDS:
                track.person = None  # il volto non e piu stato riconosciuto: la posa torna anonima
            if self._observe_arms(track, pose, now) and track.person is not None:
                person = track.person
                last_wave = self._last_wave.get(person.person_id)
                if last_wave is None or now - last_wave >= self._cooldown_seconds:
                    if all(other.person_id != person.person_id for other in waved):
                        self._last_wave[person.person_id] = now
                        waved.append(person)
        return waved

    def _match(self, poses: Sequence[BodyPose], now: float) -> list[tuple[_Track, BodyPose]]:
        """Associa ogni posa alla traccia col naso piu vicino; le pose senza traccia ne aprono una nuova."""
        candidates: list[tuple[float, int, int]] = []
        for pose_index, pose in enumerate(poses):
            for track_index, track in enumerate(self._tracks):
                distance = ((pose.nose[0] - track.nose[0]) ** 2 + (pose.nose[1] - track.nose[1]) ** 2) ** 0.5
                if distance <= TRACK_MAX_JUMP * max(pose.shoulder_width, track.shoulder_width):
                    candidates.append((distance, pose_index, track_index))

        pairs: dict[int, _Track] = {}
        used_tracks: set[int] = set()
        for _, pose_index, track_index in sorted(candidates):
            if pose_index in pairs or track_index in used_tracks:
                continue
            pairs[pose_index] = self._tracks[track_index]
            used_tracks.add(track_index)

        matched: list[tuple[_Track, BodyPose]] = []
        for pose_index, pose in enumerate(poses):
            track = pairs.get(pose_index)
            if track is None:
                track = _Track(pose.nose, pose.shoulder_width, now)
                self._tracks.append(track)
            track.nose, track.shoulder_width, track.last_seen = pose.nose, pose.shoulder_width, now
            matched.append((track, pose))
        return matched

    def _observe_arms(self, track: _Track, pose: BodyPose, now: float) -> bool:
        """Aggiorna i polsi della traccia. True se uno dei due bracci ha appena completato il gesto."""
        for arm in pose.arms:
            if not is_raised(arm, pose.shoulder_width):
                track.wrists.pop(arm.side, None)  # mano abbassata: il gesto ricomincia
                continue
            history = track.wrists.setdefault(arm.side, deque())
            if history and now - history[-1][0] > MAX_GAP_SECONDS:
                history.clear()
            history.append((now, (arm.wrist[0] - arm.shoulder[0]) / pose.shoulder_width))
            while history and now - history[0][0] > WINDOW_SECONDS:
                history.popleft()
            if count_swings([position for _, position in history]) >= MIN_SWINGS:
                track.wrists.clear()
                return True
        return False
