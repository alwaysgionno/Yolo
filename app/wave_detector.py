"""Saluto con la mano: riconosce quando una persona alza la mano e la agita a destra e a sinistra.

Il modello di posa (`PoseEstimator`) fornisce per ogni persona naso, spalle e polsi. Qui:
    1. ogni posa viene associata al volto che contiene il suo naso (`attach_arms`);
    2. per ogni persona riconosciuta si segue nel tempo la posizione orizzontale del polso,
       misurata rispetto alla spalla e in larghezze del volto, cosi non dipende dalla distanza
       dalla webcam e non conta gli spostamenti di tutto il corpo;
    3. il saluto e riconosciuto quando, a mano alzata (polso sopra la spalla), il polso cambia
       direzione abbastanza volte e con un'ampiezza sufficiente (`count_swings`).

Tutto lo stato viene usato solo dal thread di inferenza.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence

from .models import ArmPose, BodyPose, FaceDetection

WINDOW_SECONDS = 2.0      # il gesto deve stare in questo intervallo
MAX_GAP_SECONDS = 0.6     # senza il polso per piu di questo tempo, il gesto ricomincia da capo
MIN_SWING = 0.25          # ampiezza minima di un movimento, in larghezze del volto
MIN_SWINGS = 3            # movimenti alternati richiesti: destra, sinistra, destra (o viceversa)
RAISE_MARGIN = 0.0        # il polso deve stare sopra la spalla di almeno questo, in larghezze del volto


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


def attach_arms(detections: Sequence[FaceDetection], poses: Sequence[BodyPose]) -> None:
    """Assegna a ogni volto i bracci della posa il cui naso cade nel riquadro del volto.

    Se piu pose cadono nello stesso volto vince quella col naso piu vicino al centro; ogni posa
    viene assegnata a un solo volto.
    """
    candidates: list[tuple[float, int, int]] = []
    for face_index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bbox
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for pose_index, pose in enumerate(poses):
            nose_x, nose_y = pose.nose
            if x1 <= nose_x <= x2 and y1 <= nose_y <= y2:
                distance = (nose_x - center_x) ** 2 + (nose_y - center_y) ** 2
                candidates.append((distance, face_index, pose_index))

    used_faces: set[int] = set()
    used_poses: set[int] = set()
    for _, face_index, pose_index in sorted(candidates):
        if face_index in used_faces or pose_index in used_poses:
            continue
        used_faces.add(face_index)
        used_poses.add(pose_index)
        detections[face_index].arms = poses[pose_index].arms


class WaveDetector:
    """Segue i polsi delle persone riconosciute e segnala quando salutano con la mano."""

    def __init__(self, cooldown_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        # (persona, braccio) -> posizioni recenti del polso: (istante, posizione orizzontale)
        self._tracks: dict[tuple[str, int], deque[tuple[float, float]]] = {}
        self._last_wave: dict[str, float] = {}

    def observe(self, person_id: str, arms: Sequence[ArmPose], face_width: float) -> bool:
        """Registra i bracci della persona nel fotogramma corrente. True se ha appena salutato."""
        now = self._clock()
        last_wave = self._last_wave.get(person_id)
        if last_wave is not None and now - last_wave < self._cooldown_seconds:
            return False  # ha gia salutato da poco: non si raccolgono movimenti
        if face_width <= 0:
            return False

        for arm in arms:
            key = (person_id, arm.side)
            raised = arm.shoulder[1] - arm.wrist[1] >= RAISE_MARGIN * face_width  # y cresce verso il basso
            if not raised:
                self._tracks.pop(key, None)  # mano abbassata: il gesto ricomincia
                continue
            track = self._tracks.setdefault(key, deque())
            if track and now - track[-1][0] > MAX_GAP_SECONDS:
                track.clear()
            track.append((now, (arm.wrist[0] - arm.shoulder[0]) / face_width))
            while track and now - track[0][0] > WINDOW_SECONDS:
                track.popleft()
            if count_swings([position for _, position in track]) >= MIN_SWINGS:
                self._last_wave[person_id] = now
                for side in (0, 1):
                    self._tracks.pop((person_id, side), None)
                return True
        return False
