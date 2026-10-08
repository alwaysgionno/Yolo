"""Interfaccia grafica basata su OpenCV: riquadri dei volti, messaggi e pannelli di registrazione.

Tutte le funzioni di questo modulo devono essere eseguite dal thread principale,
perche le finestre OpenCV (imshow, waitKey) non sono thread-safe.

Nota: il font Hershey di OpenCV non supporta le lettere accentate, quindi i testi
a schermo sono scritti senza accenti.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .enrollment import EnrollmentSession, EnrollmentState
from .models import FaceDetection, FrameResult

WINDOW_NAME = "Face Access"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_KNOWN = (80, 200, 80)      # BGR: verde
COLOR_UNKNOWN = (60, 60, 230)    # BGR: rosso
COLOR_ENROLLING = (0, 170, 255)  # BGR: arancione, volto in registrazione
COLOR_UNVERIFIED = (150, 150, 150)
COLOR_TOAST = (40, 120, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_WARNING = (0, 200, 255)
NO_KEY = 255                     # valore restituito da waitKey quando non viene premuto nulla
MIN_FONT_SCALE = 0.35
SCREEN_MARGIN = 10
STATUS_BAR_HEIGHT = 24
TOAST_TOP_Y = 70                 # linea di base del primo messaggio temporaneo
TOAST_SPACING = 66               # distanza verticale tra due messaggi impilati
MAX_VISIBLE_TOASTS = 3


@dataclass
class _Toast:
    """Messaggio temporaneo. La durata decorre da quando diventa visibile."""

    text: str
    seconds: float
    expires_at: float | None = None


class UIController:
    """Disegna l'interfaccia sul fotogramma e gestisce la finestra di OpenCV."""

    def __init__(self, window_width: int, window_height: int, toast_seconds: float = 4.0) -> None:
        self._toast_seconds = toast_seconds
        self._toasts: list[_Toast] = []
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, window_width, window_height)

    def show_toast(self, text: str, seconds: float | None = None) -> None:
        """Accoda un messaggio temporaneo in alto, ad esempio il saluto personalizzato.

        I messaggi non si sovrascrivono: vengono impilati e ognuno resta visibile per tutta la sua durata.
        """
        self._toasts.append(_Toast(text, seconds if seconds is not None else self._toast_seconds))

    def render(
        self,
        frame: np.ndarray,
        result: FrameResult | None,
        enrollment: EnrollmentSession,
        camera_connected: bool,
        server_online: bool,
        ui_fps: float,
        inference_fps: float,
    ) -> np.ndarray:
        """Compone il fotogramma finale con tutti gli overlay e restituisce una copia."""
        canvas = frame.copy()

        if result is not None:
            candidate = enrollment.candidate_in(result.detections)
            for detection in result.detections:
                _draw_detection(canvas, detection, verified=result.gallery_ok, enrolling=detection is candidate)

        if enrollment.state is EnrollmentState.NAME_INPUT:
            _draw_name_prompt(canvas, enrollment)
        elif enrollment.state is EnrollmentState.CAPTURING:
            _draw_capture_progress(canvas, enrollment)

        self._draw_toasts(canvas)
        if not camera_connected:
            _draw_centered_message(canvas, "Webcam non disponibile: riconnessione in corso...")

        gallery_ok = result.gallery_ok if result is not None else True
        _draw_status_line(canvas, ui_fps, inference_fps, gallery_ok, server_online, camera_connected)
        return canvas

    def show(self, canvas: np.ndarray) -> None:
        """Mostra il fotogramma nella finestra."""
        cv2.imshow(WINDOW_NAME, canvas)

    @staticmethod
    def wait_key(delay_ms: int = 1) -> int:
        """Attende un tasto per `delay_ms` millisecondi. Restituisce NO_KEY se nessun tasto e premuto."""
        return cv2.waitKey(delay_ms) & 0xFF

    @staticmethod
    def is_closed() -> bool:
        """True se l'utente ha chiuso la finestra con la X."""
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    @staticmethod
    def destroy() -> None:
        """Chiude tutte le finestre OpenCV."""
        cv2.destroyAllWindows()

    def _draw_toasts(self, canvas: np.ndarray) -> None:
        """Disegna i messaggi attivi, impilati dal piu vecchio; quelli in eccesso attendono il loro turno."""
        now = time.monotonic()
        self._toasts = [toast for toast in self._toasts if toast.expires_at is None or now < toast.expires_at]
        for slot, toast in enumerate(self._toasts[:MAX_VISIBLE_TOASTS]):
            if toast.expires_at is None:
                toast.expires_at = now + toast.seconds
            _draw_toast(canvas, toast.text, TOAST_TOP_Y + slot * TOAST_SPACING)


def _draw_toast(canvas: np.ndarray, text: str, y: int) -> None:
    """Messaggio su sfondo colorato, centrato in orizzontale, con la linea di base a quota `y`."""
    width = canvas.shape[1]
    scale = _fit_font_scale([text], width - 2 * (SCREEN_MARGIN + 20), 1.0)
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, 2)
    x = max(SCREEN_MARGIN + 20, (width - text_w) // 2)
    cv2.rectangle(
        canvas,
        (x - 20, y - text_h - 16),
        (x + text_w + 20, y + baseline + 14),
        COLOR_TOAST,
        cv2.FILLED,
    )
    cv2.putText(canvas, text, (x, y), FONT, scale, COLOR_TEXT, 2, cv2.LINE_AA)


def _fit_font_scale(lines: list[str], max_width: int, font_scale: float, thickness: int = 2) -> float:
    """Riduce la dimensione del font quanto basta perche la riga piu lunga stia in `max_width` pixel."""
    widest = max(cv2.getTextSize(line, FONT, font_scale, thickness)[0][0] for line in lines)
    if widest <= max_width or widest == 0:
        return font_scale
    return max(MIN_FONT_SCALE, font_scale * max_width / widest)


def _draw_detection(canvas: np.ndarray, detection: FaceDetection, verified: bool, enrolling: bool = False) -> None:
    """Disegna il riquadro del volto con nome e similarita (verde) o 'Sconosciuto' (rosso).

    Il volto seguito dalla registrazione in corso (`enrolling`) e mostrato in arancione.
    Se la galleria non e ancora arrivata dal server (`verified` False) il volto non e stato confrontato:
    viene mostrato in grigio, senza indicarlo come sconosciuto.
    """
    x1, y1, x2, y2 = detection.bbox
    if detection.match is not None:
        color = COLOR_KNOWN
        label = f"{detection.match.full_name} {detection.match.similarity:.2f}"
    elif enrolling:
        color = COLOR_ENROLLING
        label = "In registrazione"
    elif not verified:
        color = COLOR_UNVERIFIED
        label = "Non verificato"
    else:
        color = COLOR_UNKNOWN
        label = "Sconosciuto"
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    _draw_label(canvas, label, (x1, max(30, y1)), color)


def _draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int], background: tuple[int, int, int]) -> None:
    """Testo su sfondo colorato, utile per le etichette dei volti."""
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, 0.6, 2)
    cv2.rectangle(canvas, (x, y - text_h - baseline - 8), (x + text_w + 10, y), background, cv2.FILLED)
    cv2.putText(canvas, text, (x + 5, y - baseline - 4), FONT, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)


def _draw_panel(
    canvas: np.ndarray,
    lines: list[str],
    top: int,
    left: int | None = None,
    font_scale: float = 0.7,
    line_height: int = 34,
) -> None:
    """Pannello scuro semitrasparente con alcune righe di testo.

    Il font viene ridotto se il testo non entra nel fotogramma. Con `left` uguale a None
    il pannello e centrato in orizzontale.
    """
    height, width = canvas.shape[:2]
    scale = _fit_font_scale(lines, width - 2 * SCREEN_MARGIN - 30, font_scale)
    line_height = max(14, int(line_height * scale / font_scale))
    text_width = max(cv2.getTextSize(line, FONT, scale, 2)[0][0] for line in lines)
    panel_width = text_width + 30
    panel_height = 20 + line_height * len(lines)

    x = (width - panel_width) // 2 if left is None else left
    x = max(0, min(x, width - panel_width))
    y = max(0, min(top, height - panel_height))
    region = canvas[y:y + panel_height, x:x + panel_width]
    if region.size:
        region[:] = cv2.addWeighted(region, 0.3, np.zeros_like(region), 0.7, 0)
    for index, line in enumerate(lines):
        baseline_y = y + 10 + int(line_height * (index + 0.7))
        cv2.putText(canvas, line, (x + 15, baseline_y), FONT, scale, COLOR_TEXT, 2, cv2.LINE_AA)


def _draw_name_prompt(canvas: np.ndarray, enrollment: EnrollmentSession) -> None:
    """Pannello centrale per l'inserimento del nome del nuovo volto."""
    height = canvas.shape[0]
    lines = [
        "NUOVO VOLTO RILEVATO",
        "Digita il tuo nome e premi INVIO (ESC per annullare)",
        f"Nome: {enrollment.name}_",
    ]
    if enrollment.error:
        lines.append(f"! {enrollment.error}")
    _draw_panel(canvas, lines, top=height // 2 - 90)


def _draw_capture_progress(canvas: np.ndarray, enrollment: EnrollmentSession) -> None:
    """Pannello in basso con lo stato della cattura e una barra di avanzamento."""
    height, width = canvas.shape[:2]
    done, total = enrollment.progress
    lines = [
        f"Registrazione di {enrollment.name}: campione {done}/{total}",
        enrollment.feedback or "Guarda la webcam e muovi leggermente la testa",
        "ESC per annullare",
    ]
    _draw_panel(canvas, lines, top=height - 200, left=20)

    bar_x, bar_y, bar_h = 20, height - STATUS_BAR_HEIGHT - 22, 14
    bar_w = min(400, width - 2 * bar_x)
    filled = int(bar_w * done / total) if total else 0
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 1)
    if filled:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), COLOR_KNOWN, cv2.FILLED)


def _draw_centered_message(canvas: np.ndarray, text: str) -> None:
    """Messaggio di avviso al centro dello schermo (es. webcam disconnessa)."""
    height, width = canvas.shape[:2]
    scale = _fit_font_scale([text], width - 2 * SCREEN_MARGIN, 0.8)
    (text_w, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.putText(canvas, text, ((width - text_w) // 2, height // 2), FONT, scale, COLOR_WARNING, 2, cv2.LINE_AA)


def _draw_status_line(
    canvas: np.ndarray,
    ui_fps: float,
    inference_fps: float,
    gallery_ok: bool,
    server_online: bool,
    camera_connected: bool,
) -> None:
    """Barra di stato in basso con frame rate, stato della galleria, del server e della webcam."""
    height, width = canvas.shape[:2]
    status = (
        f"Video {ui_fps:4.1f} fps | Inferenza {inference_fps:4.1f} fps | "
        f"Galleria: {'OK' if gallery_ok else 'ATTESA'} | Server: {'OK' if server_online else 'OFF'} | "
        f"Webcam: {'OK' if camera_connected else 'OFF'} | Q: esci"
    )
    cv2.rectangle(canvas, (0, height - STATUS_BAR_HEIGHT), (width, height), (0, 0, 0), cv2.FILLED)
    scale = _fit_font_scale([status], width - 2 * SCREEN_MARGIN, 0.5, thickness=1)
    cv2.putText(canvas, status, (SCREEN_MARGIN, height - 8), FONT, scale, COLOR_TEXT, 1, cv2.LINE_AA)
