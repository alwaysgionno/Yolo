"""Anti-spoofing: verifica che il volto inquadrato sia tridimensionale, e non una foto.

Principio: quando una testa vera gira anche di poco, il naso, le guance e la fronte si
spostano in modo diverso tra loro (parallasse), perche stanno a profondita diverse. Una foto
stampata o mostrata su un telefono e un piano: comunque la si inclini o la si sposti, tutti
i suoi punti seguono una sola trasformazione prospettica (omografia).

Per ogni volto si confrontano due immagini riprese in momenti diversi:
    1. il volto viene ritagliato e allineato (`align_face`) usando i 5 punti di InsightFace;
    2. si cerca l'omografia che sovrappone al meglio la seconda immagine alla prima;
    3. si misura con il flusso ottico quanto movimento resta dopo la sovrapposizione
       (`parallax`): quasi nullo per una foto, evidente per una testa che ha girato.

Si confrontano i pixel e non i punti del volto: i modelli che stimano i punti "immaginano"
la forma di una testa vera anche davanti a una foto inclinata, e verrebbero ingannati.

Limiti: un video della persona riprodotto su uno schermo mostra una testa che gira davvero,
e supera la verifica; lo stesso vale per una foto curvata o una maschera. Le soglie sono
state tarate in simulazione e vanno controllate con la webcam reale (`LIVENESS_MIN_PARALLAX`).

Le immagini di riferimento restano in memoria per pochi secondi e non vengono mai salvate.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .models import FaceDetection

PATCH_SIZE = 160              # lato, in pixel, del volto allineato
# Posizione dei 5 punti (occhi, naso, angoli della bocca) nel volto allineato: e il modello di
# ArcFace per immagini 112x112, rimpicciolito per lasciare un margine attorno al volto
_ARCFACE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]
)
_TEMPLATE = (_ARCFACE_112 - 56.0) * (PATCH_SIZE / 112.0) * 0.85 + PATCH_SIZE / 2.0
EYE_DISTANCE = float(np.linalg.norm(_TEMPLATE[1] - _TEMPLATE[0]))  # unita di misura del parallasse

BLUR_SIGMA = 1.0              # attenua il rumore della webcam prima di ogni confronto
MIN_FEATURES = 30             # sotto questo numero di punti il volto ha troppo poco dettaglio
MIN_TRACKED = 20
MIN_INLIERS = 12
MAX_TRACKING_ERROR_PX = 1.0   # un punto seguito all'andata deve tornare al punto di partenza
ECC_ITERATIONS = 60
ECC_EPSILON = 1e-5
MAX_STRETCH, MIN_STRETCH = 1.6, 0.45   # deformazioni oltre questi limiti: sovrapposizione non credibile
MAX_PERSPECTIVE = 0.004
MIN_VALID_AREA = 0.7          # parte del volto che deve restare coperta dopo la sovrapposizione
PARALLAX_PERCENTILE = 90      # il parallasse riguarda una parte del volto: si guarda la coda alta

CONFIRMATIONS = 2             # confronti sopra soglia richiesti prima di considerare vero il volto
REFERENCE_MAX_AGE_S = 4.0     # senza conferme, il riferimento viene rinnovato (luce e posa cambiano)
MIN_CHECK_INTERVAL_S = 0.25   # un confronto costa qualche decina di millisecondi: non uno per fotogramma
HOLD_SECONDS = 30.0           # un volto verificato resta tale finche non sparisce per questo tempo
SAME_FACE_SIMILARITY = 0.50   # similarita minima tra due embedding dello stesso volto sconosciuto
MAX_UNKNOWN_FACES = 8

_LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}
_ECC_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, ECC_ITERATIONS, ECC_EPSILON)


def _build_face_mask() -> np.ndarray:
    """Ellisse sull'interno del volto: sfondo, capelli e bordi della foto restano fuori dal confronto."""
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    center = (PATCH_SIZE // 2, int(PATCH_SIZE * 0.60))
    cv2.ellipse(mask, center, (int(0.95 * EYE_DISTANCE), int(1.25 * EYE_DISTANCE)), 0, 0, 360, 255, cv2.FILLED)
    return mask


_FACE_MASK = _build_face_mask()
_FACE_AREA = _FACE_MASK > 0
_FULL_PATCH = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
_ERODE_KERNEL = np.ones((9, 9), dtype=np.uint8)
_flow_estimator: cv2.DISOpticalFlow | None = None  # creato al primo uso, dal thread di inferenza


def _similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotazione, scala e traslazione che portano `source` su `target` ai minimi quadrati (matrice 2x3)."""
    count = len(source)
    system = np.zeros((2 * count, 4))
    system[0::2] = np.column_stack([source[:, 0], -source[:, 1], np.ones(count), np.zeros(count)])
    system[1::2] = np.column_stack([source[:, 1], source[:, 0], np.zeros(count), np.ones(count)])
    a, b, shift_x, shift_y = np.linalg.lstsq(system, target.reshape(-1), rcond=None)[0]
    return np.array([[a, -b, shift_x], [b, a, shift_y]])


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Ritaglia il volto in scala di grigi, raddrizzato e a dimensione fissa.

    Args:
        image: immagine BGR o in scala di grigi che contiene il volto.
        landmarks: i 5 punti di InsightFace (occhi, naso, angoli della bocca) nelle coordinate di `image`.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    transform = _similarity_transform(np.asarray(landmarks, dtype=np.float64), _TEMPLATE)
    return cv2.warpAffine(gray, transform, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_LINEAR)


def _correlation(first: np.ndarray, second: np.ndarray, area: np.ndarray) -> float:
    """Correlazione normalizzata tra due immagini nell'area indicata: 1 se coincidono."""
    a = first[area].astype(np.float64)
    b = second[area].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def _register(reference: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    """Omografia che porta le coordinate di `reference` su `current`, oppure None se non e affidabile."""
    points = cv2.goodFeaturesToTrack(reference, maxCorners=250, qualityLevel=0.01, minDistance=4, mask=_FACE_MASK)
    if points is None or len(points) < MIN_FEATURES:
        return None
    moved, found, _ = cv2.calcOpticalFlowPyrLK(reference, current, points, None, **_LK_PARAMS)
    back, found_back, _ = cv2.calcOpticalFlowPyrLK(current, reference, moved, None, **_LK_PARAMS)
    error = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
    good = (found.ravel() == 1) & (found_back.ravel() == 1) & (error < MAX_TRACKING_ERROR_PX)
    if good.sum() < MIN_TRACKED:
        return None

    # Prima una stima robusta con pochi parametri, poi l'omografia rifinita su tutti i pixel del volto:
    # adattata solo ai punti, sarebbe instabile
    affine, inliers = cv2.estimateAffine2D(points[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if affine is None or inliers.sum() < MIN_INLIERS:
        return None
    homography = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)
    try:
        _, homography = cv2.findTransformECC(
            reference, current, homography, cv2.MOTION_HOMOGRAPHY, _ECC_CRITERIA, _FACE_MASK, 5
        )
    except cv2.error:  # l'allineamento non converge: confronto non misurabile
        return None

    stretch = np.linalg.svd(homography[:2, :2], compute_uv=False)
    if stretch[0] > MAX_STRETCH or stretch[1] < MIN_STRETCH:
        return None
    if abs(homography[2, 0]) > MAX_PERSPECTIVE or abs(homography[2, 1]) > MAX_PERSPECTIVE:
        return None
    return homography


def parallax(reference: np.ndarray, current: np.ndarray) -> float | None:
    """Misura quanto `current` si discosta da una vista piana di `reference`.

    Args:
        reference, current: due volti allineati con `align_face`, della stessa persona.

    Returns:
        Il movimento che resta dopo la migliore sovrapposizione prospettica, in frazioni della
        distanza tra gli occhi (circa 0.01 per una foto, oltre 0.06 per una testa girata di una
        decina di gradi). None se il confronto non e misurabile: poco dettaglio, volto troppo
        diverso o sovrapposizione non credibile.
    """
    global _flow_estimator
    reference = cv2.GaussianBlur(reference, (0, 0), BLUR_SIGMA)
    current = cv2.GaussianBlur(current, (0, 0), BLUR_SIGMA)
    homography = _register(reference, current)
    if homography is None:
        return None

    size = (PATCH_SIZE, PATCH_SIZE)
    registered = cv2.warpPerspective(current, homography, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    covered = cv2.warpPerspective(_FULL_PATCH, homography, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    area = _FACE_AREA & (cv2.erode(covered, _ERODE_KERNEL) > 0)
    if area.sum() < MIN_VALID_AREA * _FACE_AREA.sum():
        return None
    # Una sovrapposizione che peggiora la somiglianza e sbagliata: il movimento residuo non direbbe nulla
    if _correlation(reference, registered, area) < _correlation(reference, current, area) - 0.01:
        return None

    if _flow_estimator is None:
        _flow_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = _flow_estimator.calc(reference, registered, None)
    magnitude = np.linalg.norm(flow, axis=2)[area]
    return float(np.percentile(magnitude, PARALLAX_PERCENTILE) / EYE_DISTANCE)


class LivenessCheck:
    """Accumula le prove che un volto e tridimensionale. Una istanza per volto seguito."""

    def __init__(
        self,
        min_parallax: float,
        clock: Callable[[], float] = time.monotonic,
        measure: Callable[[np.ndarray, np.ndarray], float | None] = parallax,
    ) -> None:
        self._min_parallax = min_parallax
        self._clock = clock
        self._measure = measure
        self._reference: np.ndarray | None = None
        self._reference_at = 0.0
        self._last_check_at = float("-inf")
        self._confirmations = 0
        self.verified = False
        self.started_at = clock()

    def observe(self, patch: np.ndarray) -> bool:
        """Valuta una nuova immagine del volto. Restituisce True da quando il volto e verificato."""
        if self.verified:
            return True
        now = self._clock()
        if self._reference is None:
            self._set_reference(patch, now)
            return False
        if now - self._last_check_at < MIN_CHECK_INTERVAL_S:
            return False
        self._last_check_at = now

        score = self._measure(self._reference, patch)
        if score is not None and score >= self._min_parallax:
            self._confirmations += 1
            if self._confirmations >= CONFIRMATIONS:
                self.verified = True
                self._reference = None  # non serve piu: l'immagine non resta in memoria
        elif now - self._reference_at > REFERENCE_MAX_AGE_S:
            # Nessuna prova con questo riferimento: si riparte dall'immagine attuale
            self._set_reference(patch, now)
        return self.verified

    def _set_reference(self, patch: np.ndarray, now: float) -> None:
        self._reference = patch
        self._reference_at = now
        self._confirmations = 0


@dataclass
class _Tracked:
    check: LivenessCheck
    last_seen: float
    embedding: np.ndarray | None = None  # solo per i volti sconosciuti, che non hanno un identificativo


class LivenessTracker:
    """Stato della verifica per ogni volto inquadrato. Usato solo dal thread di inferenza.

    Le persone riconosciute sono seguite per identificativo; i volti sconosciuti vengono
    riconosciuti tra un fotogramma e l'altro confrontando gli embedding.
    """

    def __init__(
        self,
        min_parallax: float,
        min_face_width: float,
        min_sharpness: float,
        clock: Callable[[], float] = time.monotonic,
        measure: Callable[[np.ndarray, np.ndarray], float | None] = parallax,
    ) -> None:
        self._min_parallax = min_parallax
        self._min_face_width = min_face_width
        self._min_sharpness = min_sharpness
        self._clock = clock
        self._measure = measure
        self._known: dict[str, _Tracked] = {}
        self._unknown: list[_Tracked] = []

    def observe(self, detection: FaceDetection, person_id: str | None) -> bool:
        """Aggiorna la verifica del volto e la scrive in `detection.live` e `detection.liveness_wait_s`.

        Args:
            detection: volto rilevato nel fotogramma corrente.
            person_id: persona riconosciuta, oppure None se il volto e sconosciuto.

        Returns:
            True se il volto e stato verificato come tridimensionale.
        """
        now = self._clock()
        self._forget_absent(now)
        tracked = self._find(detection, person_id, now)
        if tracked is None:  # volto sconosciuto senza embedding: non si puo seguire
            detection.live = False
            return False
        tracked.last_seen = now

        usable = (
            detection.patch is not None
            and detection.face_width >= self._min_face_width
            and detection.sharpness >= self._min_sharpness
        )
        if usable and not tracked.check.verified:
            tracked.check.observe(detection.patch)
        detection.live = tracked.check.verified
        detection.liveness_wait_s = 0.0 if detection.live else now - tracked.check.started_at
        return detection.live

    def _find(self, detection: FaceDetection, person_id: str | None, now: float) -> _Tracked | None:
        if person_id is not None:
            tracked = self._known.get(person_id)
            if tracked is None:
                tracked = self._known[person_id] = _Tracked(self._new_check(), now)
            return tracked

        embedding = detection.embedding
        if embedding is None:
            return None
        best: _Tracked | None = None
        best_similarity = SAME_FACE_SIMILARITY
        for candidate in self._unknown:
            similarity = float(np.dot(candidate.embedding, embedding))
            if similarity >= best_similarity:
                best, best_similarity = candidate, similarity
        if best is None:
            best = _Tracked(self._new_check(), now, embedding)
            self._unknown.append(best)
            del self._unknown[:-MAX_UNKNOWN_FACES]  # restano i piu recenti
        return best

    def _new_check(self) -> LivenessCheck:
        return LivenessCheck(self._min_parallax, self._clock, self._measure)

    def _forget_absent(self, now: float) -> None:
        """Chi non si vede da un po' dovra essere verificato di nuovo."""
        for person_id in [key for key, tracked in self._known.items() if now - tracked.last_seen > HOLD_SECONDS]:
            del self._known[person_id]
        self._unknown = [tracked for tracked in self._unknown if now - tracked.last_seen <= HOLD_SECONDS]
