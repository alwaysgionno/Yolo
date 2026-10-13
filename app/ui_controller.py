"""Interfaccia grafica: riquadri dei volti, saluti, finestra del nome e pannelli di registrazione.

Tutte le funzioni di questo modulo devono essere eseguite dal thread principale,
perche le finestre OpenCV (imshow, waitKey) non sono thread-safe.

Le forme sono disegnate con OpenCV. I testi sono disegnati con Pillow, che usa i font
TrueType del sistema (lettere accentate comprese): ogni scritta viene convertita una sola
volta in una maschera e poi fusa sul fotogramma, cosi il costo per fotogramma resta basso.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .enrollment import EnrollmentSession, EnrollmentState
from .models import BBox, FaceDetection, FrameResult

WINDOW_NAME = "Face Access"
NO_KEY = 255                     # valore restituito da waitKey quando non viene premuto nulla

Color = tuple[int, int, int]     # BGR
COLOR_KNOWN: Color = (153, 211, 52)       # verde
COLOR_UNKNOWN: Color = (113, 113, 248)    # rosso
COLOR_ENROLLING: Color = (36, 191, 251)   # ambra, volto in registrazione
COLOR_UNVERIFIED: Color = (184, 163, 148)
COLOR_ACCENT: Color = (248, 189, 56)      # azzurro
COLOR_WARNING: Color = (36, 191, 251)
COLOR_PANEL: Color = (39, 24, 17)         # sfondo dei pannelli
COLOR_FIELD: Color = (59, 41, 30)         # campo di testo e tasti
COLOR_BORDER: Color = (105, 85, 71)
COLOR_TEXT: Color = (252, 250, 248)
COLOR_MUTED: Color = (184, 163, 148)
COLOR_ON_COLOR: Color = (39, 24, 17)      # testo scuro sopra un colore pieno

# Font provati in ordine: Windows, poi i piu comuni su Linux e macOS
REGULAR_FONTS = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf")
BOLD_FONTS = ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
MIN_FONT_SIZE = 11

REFERENCE_SIZE = (1280, 720)     # dimensioni a cui si riferiscono le misure in pixel di questo modulo
MIN_SCALE, MAX_SCALE = 0.75, 2.0
SCREEN_MARGIN = 16
STATUS_BAR_HEIGHT = 36
PANEL_OPACITY = 0.9
BACKDROP_BRIGHTNESS = 0.4        # luminosita del video dietro la finestra del nome
TOAST_TOP = 22
TOAST_GAP = 10
TOAST_FADE_S = 0.25
MAX_VISIBLE_TOASTS = 3
MASK_OVERSAMPLING = 4            # le maschere sono disegnate ingrandite e poi ridotte: bordi morbidi
THUMBNAIL_SIZE = 160             # lato della miniatura del volto conservata per la finestra del nome
CARET_BLINK_S = 0.5


class ToastKind(Enum):
    """Aspetto di un messaggio temporaneo."""

    GREETING = auto()   # saluto: grande, verde
    INFO = auto()
    WARNING = auto()


TOAST_COLORS = {
    ToastKind.GREETING: COLOR_KNOWN,
    ToastKind.INFO: COLOR_ACCENT,
    ToastKind.WARNING: COLOR_WARNING,
}


@dataclass
class _Toast:
    """Messaggio temporaneo. La durata decorre da quando diventa visibile."""

    text: str
    seconds: float
    kind: ToastKind
    shown_at: float | None = None

    def opacity(self, now: float) -> float:
        """Dissolvenza in entrata e in uscita: 0 invisibile, 1 pieno."""
        if self.shown_at is None:
            return 0.0
        elapsed = now - self.shown_at
        return float(np.clip(min(elapsed, self.seconds - elapsed) / TOAST_FADE_S, 0.0, 1.0))


class UIController:
    """Disegna l'interfaccia sul fotogramma e gestisce la finestra di OpenCV."""

    def __init__(self, window_width: int, window_height: int, toast_seconds: float = 4.0) -> None:
        self._toast_seconds = toast_seconds
        self._toasts: list[_Toast] = []
        self._thumbnail: np.ndarray | None = None  # ultimo ritaglio del volto in registrazione
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, window_width, window_height)

    def show_toast(self, text: str, seconds: float | None = None, kind: ToastKind = ToastKind.INFO) -> None:
        """Accoda un messaggio temporaneo in alto, ad esempio il saluto personalizzato.

        I messaggi non si sovrascrivono: vengono impilati e ognuno resta visibile per tutta la sua durata.
        """
        self._toasts.append(_Toast(text, seconds if seconds is not None else self._toast_seconds, kind))

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
        height, width = canvas.shape[:2]
        scale = float(np.clip(min(width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1]), MIN_SCALE, MAX_SCALE))

        if result is not None:
            candidate = enrollment.candidate_in(result.detections)
            if candidate is not None:
                # Il ritaglio va preso dal fotogramma originale, prima di disegnarci sopra
                thumbnail = _crop_face(frame, candidate.bbox)
                if thumbnail is not None:
                    self._thumbnail = thumbnail
            for detection in result.detections:
                _draw_detection(canvas, detection, result.gallery_ok, detection is candidate, scale)
        if not enrollment.is_active:
            self._thumbnail = None

        if enrollment.state is EnrollmentState.NAME_INPUT:
            _draw_name_dialog(canvas, enrollment, self._thumbnail, scale)
        elif enrollment.state is EnrollmentState.CAPTURING:
            _draw_capture_progress(canvas, enrollment, scale)

        self._draw_toasts(canvas, scale)
        if not camera_connected:
            _draw_camera_notice(canvas, scale)

        gallery_ok = result.gallery_ok if result is not None else True
        _draw_status_bar(canvas, ui_fps, inference_fps, gallery_ok, server_online, camera_connected, scale)
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

    def _draw_toasts(self, canvas: np.ndarray, scale: float) -> None:
        """Disegna i messaggi attivi, impilati dal piu vecchio; quelli in eccesso attendono il loro turno."""
        now = time.monotonic()
        self._toasts = [
            toast for toast in self._toasts if toast.shown_at is None or now < toast.shown_at + toast.seconds
        ]
        top = _px(TOAST_TOP, scale)
        for toast in self._toasts[:MAX_VISIBLE_TOASTS]:
            if toast.shown_at is None:
                toast.shown_at = now
            top += _draw_toast(canvas, toast, top, toast.opacity(now), scale) + _px(TOAST_GAP, scale)


# --- Primitive di disegno ---------------------------------------------------------------------


def _px(value: float, scale: float) -> int:
    """Converte una misura riferita a REFERENCE_SIZE nei pixel del fotogramma corrente."""
    return max(1, round(value * scale))


@lru_cache(maxsize=64)
def _font(size: int, bold: bool) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in BOLD_FONTS if bold else REGULAR_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


@lru_cache(maxsize=512)
def _text_mask(text: str, size: int, bold: bool) -> np.ndarray:
    """Maschera (0-255) della scritta. L'altezza dipende solo dal font, non dalle lettere."""
    font = _font(size, bold)
    image = Image.new("L", (max(1, int(np.ceil(font.getlength(text)))), _text_height(size, bold)), 0)
    ImageDraw.Draw(image).text((0, 0), text, font=font, fill=255)
    return np.asarray(image)


def _text_width(text: str, size: int, bold: bool = False) -> int:
    return _text_mask(text, size, bold).shape[1]


@lru_cache(maxsize=64)
def _text_height(size: int, bold: bool = False) -> int:
    return sum(_font(size, bold).getmetrics())


def _fit_font_size(text: str, size: int, bold: bool, max_width: int) -> int:
    """Riduce la dimensione del font quanto basta perche la scritta stia in `max_width` pixel."""
    width = _text_width(text, size, bold)
    if width <= max_width:
        return size
    return max(MIN_FONT_SIZE, int(size * max_width / width))


@lru_cache(maxsize=128)
def _rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
    """Maschera (0-255) di un rettangolo con gli angoli arrotondati."""
    big_width, big_height = width * MASK_OVERSAMPLING, height * MASK_OVERSAMPLING
    radius = max(0, min(radius * MASK_OVERSAMPLING, (big_width - 1) // 2, (big_height - 1) // 2))
    mask = np.zeros((big_height, big_width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (big_width - 1 - radius, big_height - 1), 255, cv2.FILLED)
    cv2.rectangle(mask, (0, radius), (big_width - 1, big_height - 1 - radius), 255, cv2.FILLED)
    for center_x in (radius, big_width - 1 - radius):
        for center_y in (radius, big_height - 1 - radius):
            cv2.circle(mask, (center_x, center_y), radius, 255, cv2.FILLED)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=16)
def _check_mask(size: int) -> np.ndarray:
    """Maschera (0-255) di un segno di spunta."""
    big_size = size * MASK_OVERSAMPLING
    mask = np.zeros((big_size, big_size), dtype=np.uint8)
    points = (np.array([[0.27, 0.53], [0.44, 0.69], [0.74, 0.34]]) * big_size).astype(np.int32)
    cv2.polylines(mask, [points], False, 255, max(2, big_size // 9), cv2.LINE_AA)
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)


def _blend(
    canvas: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    fill: Color | np.ndarray,
    opacity: float = 1.0,
) -> None:
    """Fonde sul fotogramma un colore, o un'immagine grande quanto la maschera, pesato dalla maschera."""
    height, width = canvas.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + mask.shape[1], width), min(y + mask.shape[0], height)
    if x1 >= x2 or y1 >= y2 or opacity <= 0:
        return
    visible = np.s_[y1 - y:y2 - y, x1 - x:x2 - x]
    weights = mask[visible].astype(np.float32) * (opacity / 255.0)
    region = canvas[y1:y2, x1:x2]
    if isinstance(fill, np.ndarray):
        source = np.ascontiguousarray(fill[visible])
    else:
        source = np.empty_like(region)
        source[:] = fill
    region[:] = cv2.blendLinear(source, region, weights, 1.0 - weights)


def _draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    size: int,
    color: Color = COLOR_TEXT,
    bold: bool = False,
    opacity: float = 1.0,
) -> int:
    """Disegna la scritta con l'angolo in alto a sinistra in (x, y). Restituisce la larghezza."""
    mask = _text_mask(text, size, bold)
    _blend(canvas, mask, x, y, color, opacity)
    return mask.shape[1]


def _draw_box(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    color: Color = COLOR_PANEL,
    opacity: float = PANEL_OPACITY,
    border: Color | None = None,
) -> None:
    """Rettangolo arrotondato semitrasparente, con un bordo opzionale di un pixel."""
    if width < 3 or height < 3:
        return
    if border is not None:
        _blend(canvas, _rounded_mask(width, height, radius), x, y, border, opacity)
        x, y, width, height, radius = x + 1, y + 1, width - 2, height - 2, radius - 1
    _blend(canvas, _rounded_mask(width, height, radius), x, y, color, opacity)


def _draw_badge(
    canvas: np.ndarray,
    x: int,
    y: int,
    diameter: int,
    color: Color,
    symbol: str | None,
    opacity: float = 1.0,
) -> None:
    """Cerchio colorato con un simbolo scuro: un carattere, oppure la spunta se `symbol` e None."""
    _blend(canvas, _rounded_mask(diameter, diameter, diameter // 2), x, y, color, opacity)
    if symbol is None:
        _blend(canvas, _check_mask(diameter), x, y, COLOR_ON_COLOR, opacity)
        return
    size = int(diameter * 0.62)
    _draw_text(
        canvas,
        symbol,
        x + (diameter - _text_width(symbol, size, True)) // 2,
        y + (diameter - _text_height(size, True)) // 2,
        size,
        COLOR_ON_COLOR,
        bold=True,
        opacity=opacity,
    )


def _draw_key_hint(canvas: np.ndarray, key: str, action: str, x: int, y: int, scale: float) -> int:
    """Disegna un tasto seguito dalla sua azione ("Esc  Annulla"). Restituisce la x dove finisce."""
    size = _px(13, scale)
    key_height = _text_height(size, True) + _px(6, scale)
    key_width = _text_width(key, size, True) + _px(16, scale)
    _draw_box(canvas, x, y, key_width, key_height, _px(6, scale), COLOR_FIELD, 1.0, COLOR_BORDER)
    _draw_text(canvas, key, x + _px(8, scale), y + _px(3, scale), size, COLOR_TEXT, bold=True)
    x += key_width + _px(8, scale)
    return x + _draw_text(canvas, action, x, y + _px(3, scale), size, COLOR_MUTED)


def _crop_face(frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
    """Ritaglio quadrato del volto, con un po' di margine, ridotto a THUMBNAIL_SIZE."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    side = int(max(x2 - x1, y2 - y1) * 1.3)
    left, top = max(0, (x1 + x2 - side) // 2), max(0, (y1 + y2 - side) // 2)
    crop = frame[top:min(height, top + side), left:min(width, left + side)]
    side = min(crop.shape[:2])
    if side < 8:
        return None
    offset_y, offset_x = (crop.shape[0] - side) // 2, (crop.shape[1] - side) // 2
    square = crop[offset_y:offset_y + side, offset_x:offset_x + side]
    return cv2.resize(square, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), interpolation=cv2.INTER_AREA)


# --- Elementi dell'interfaccia ----------------------------------------------------------------


def _draw_detection(
    canvas: np.ndarray,
    detection: FaceDetection,
    verified: bool,
    enrolling: bool,
    scale: float,
) -> None:
    """Disegna il riquadro del volto con nome e similarita (verde) o 'Sconosciuto' (rosso).

    Il volto seguito dalla registrazione in corso (`enrolling`) e mostrato in ambra.
    Se la galleria non e ancora arrivata dal server (`verified` False) il volto non e stato confrontato:
    viene mostrato in grigio, senza indicarlo come sconosciuto.
    """
    x1, y1, x2, y2 = detection.bbox
    detail = ""
    if detection.match is not None:
        color = COLOR_KNOWN
        label = detection.match.full_name
        detail = f"{detection.match.similarity:.2f}"
    elif enrolling:
        color = COLOR_ENROLLING
        label = "In registrazione"
    elif not verified:
        color = COLOR_UNVERIFIED
        label = "Non verificato"
    else:
        color = COLOR_UNKNOWN
        label = "Sconosciuto"

    # Riquadro sottile con gli angoli marcati
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    corner = max(8, min(x2 - x1, y2 - y1) // 5)
    thickness = max(2, _px(3, scale))
    for corner_x, corner_y, dir_x, dir_y in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(canvas, (corner_x, corner_y), (corner_x + dir_x * corner, corner_y), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (corner_x, corner_y), (corner_x, corner_y + dir_y * corner), color, thickness, cv2.LINE_AA)

    # Etichetta sopra il riquadro (dentro, se in alto non c'e spazio)
    size = _px(16, scale)
    pad, gap = _px(10, scale), _px(8, scale)
    label_height = _text_height(size, True) + _px(6, scale)
    label_width = 2 * pad + _text_width(label, size, True)
    if detail:
        label_width += gap + _text_width(detail, size)
    x = max(0, min(x1, canvas.shape[1] - label_width))
    y = y1 - label_height - _px(8, scale)
    if y < 0:
        y = y1 + _px(8, scale)
    _draw_box(canvas, x, y, label_width, label_height, label_height // 2, color, 0.95)
    text_y = y + _px(3, scale)
    x += pad + _draw_text(canvas, label, x + pad, text_y, size, COLOR_ON_COLOR, bold=True)
    if detail:
        _draw_text(canvas, detail, x + gap, text_y, size, COLOR_ON_COLOR, opacity=0.75)


def _draw_toast(canvas: np.ndarray, toast: _Toast, top: int, opacity: float, scale: float) -> int:
    """Messaggio in alto, centrato: simbolo colorato e testo. Restituisce l'altezza occupata.

    I saluti sono piu grandi degli altri messaggi. `opacity` segue la dissolvenza, durante
    la quale il messaggio scende di qualche pixel verso la sua posizione.
    """
    greeting = toast.kind is ToastKind.GREETING
    color = TOAST_COLORS[toast.kind]
    pad = _px(22 if greeting else 16, scale)
    badge = _px(44 if greeting else 28, scale)
    gap = _px(16 if greeting else 12, scale)
    height = _px(76 if greeting else 50, scale)
    max_text_width = canvas.shape[1] - 2 * SCREEN_MARGIN - 2 * pad - badge - gap
    size = _fit_font_size(toast.text, _px(32 if greeting else 19, scale), True, max_text_width)
    width = pad + badge + gap + _text_width(toast.text, size, True) + pad + _px(6, scale)

    x = (canvas.shape[1] - width) // 2
    y = top - round((1.0 - opacity) * _px(12, scale))
    _draw_box(canvas, x, y, width, height, height // 2, opacity=PANEL_OPACITY * opacity, border=color)
    symbol = None if greeting else "!" if toast.kind is ToastKind.WARNING else "i"
    _draw_badge(canvas, x + pad, y + (height - badge) // 2, badge, color, symbol, opacity)
    _draw_text(
        canvas,
        toast.text,
        x + pad + badge + gap,
        y + (height - _text_height(size, True)) // 2,
        size,
        bold=True,
        opacity=opacity,
    )
    return height


def _draw_name_dialog(
    canvas: np.ndarray,
    enrollment: EnrollmentSession,
    thumbnail: np.ndarray | None,
    scale: float,
) -> None:
    """Finestra al centro dello schermo per inserire il nome del nuovo volto.

    Il video sullo sfondo viene scurito; la miniatura mostra quale volto sta per essere registrato.
    """
    height, width = canvas.shape[:2]
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=BACKDROP_BRIGHTNESS)

    pad = _px(28, scale)
    caption_size, title_size, body_size = _px(13, scale), _px(28, scale), _px(15, scale)
    input_size, error_size = _px(22, scale), _px(14, scale)
    thumb = _px(88, scale) if thumbnail is not None else 0
    texts_height = _text_height(caption_size, True) + _text_height(title_size, True) + _text_height(body_size)
    header_height = max(thumb, texts_height)
    field_height = _px(56, scale)
    hints_height = _text_height(_px(13, scale), True) + _px(6, scale)
    card_width = min(_px(580, scale), width - 2 * SCREEN_MARGIN)
    card_height = (
        pad + header_height + _px(22, scale) + field_height + _px(8, scale)
        + _text_height(error_size) + _px(14, scale) + hints_height + pad
    )
    card_x = (width - card_width) // 2
    card_y = max(0, (height - _px(STATUS_BAR_HEIGHT, scale) - card_height) // 2)
    inner_x, inner_width = card_x + pad, card_width - 2 * pad
    _draw_box(canvas, card_x, card_y, card_width, card_height, _px(20, scale), opacity=0.97, border=COLOR_BORDER)

    # Intestazione: miniatura del volto e testi
    y = card_y + pad
    text_x = inner_x
    if thumbnail is not None:
        radius = _px(16, scale)
        _draw_box(canvas, inner_x, y, thumb, thumb, radius, COLOR_ENROLLING, 1.0)
        inset = max(2, _px(3, scale))
        picture = cv2.resize(thumbnail, (thumb - 2 * inset, thumb - 2 * inset), interpolation=cv2.INTER_AREA)
        _blend(canvas, _rounded_mask(*picture.shape[1::-1], radius - inset), inner_x + inset, y + inset, picture)
        text_x += thumb + _px(20, scale)
    text_y = y + (header_height - texts_height) // 2
    _draw_text(canvas, "NUOVO VOLTO RILEVATO", text_x, text_y, caption_size, COLOR_ENROLLING, bold=True)
    text_y += _text_height(caption_size, True)
    _draw_text(canvas, "Come ti chiami?", text_x, text_y, title_size, bold=True)
    text_y += _text_height(title_size, True)
    _draw_text(canvas, "Scrivi il tuo nome per creare il profilo", text_x, text_y, body_size, COLOR_MUTED)

    # Campo di testo con cursore lampeggiante
    y += header_height + _px(22, scale)
    border = COLOR_UNKNOWN if enrollment.error else COLOR_ACCENT
    _draw_box(canvas, inner_x, y, inner_width, field_height, _px(12, scale), COLOR_FIELD, 1.0, border)
    field_pad = _px(16, scale)
    text_y = y + (field_height - _text_height(input_size)) // 2
    caret_x = inner_x + field_pad
    if enrollment.name:
        # Un nome piu lungo del campo scorre: resta visibile la parte finale
        mask = _text_mask(enrollment.name, input_size, False)[:, -(inner_width - 2 * field_pad - _px(4, scale)):]
        _blend(canvas, mask, caret_x, text_y, COLOR_TEXT)
        caret_x += mask.shape[1] + _px(2, scale)
    else:
        _draw_text(canvas, "Nome e cognome", caret_x + _px(6, scale), text_y, input_size, COLOR_MUTED, opacity=0.6)
    if int(time.monotonic() / CARET_BLINK_S) % 2 == 0:
        caret_top = y + _px(14, scale)
        cv2.rectangle(
            canvas,
            (caret_x, caret_top),
            (caret_x + max(1, _px(2, scale)) - 1, y + field_height - _px(14, scale)),
            COLOR_ACCENT,
            cv2.FILLED,
        )

    # Errore di validazione (lo spazio e sempre riservato: la finestra non cambia altezza)
    y += field_height + _px(8, scale)
    if enrollment.error:
        size = _fit_font_size(enrollment.error, error_size, False, inner_width)
        _draw_text(canvas, enrollment.error, inner_x, y, size, COLOR_UNKNOWN)

    y += _text_height(error_size) + _px(14, scale)
    x = _draw_key_hint(canvas, "Invio", "Conferma", inner_x, y, scale)
    _draw_key_hint(canvas, "Esc", "Annulla", x + _px(22, scale), y, scale)


def _draw_capture_progress(canvas: np.ndarray, enrollment: EnrollmentSession, scale: float) -> None:
    """Pannello in basso con lo stato della cattura, il suggerimento e una barra di avanzamento."""
    height, width = canvas.shape[:2]
    done, total = enrollment.progress
    pad = _px(22, scale)
    title_size, body_size = _px(20, scale), _px(15, scale)
    bar_height = _px(10, scale)
    hints_height = _text_height(_px(13, scale), True) + _px(6, scale)
    card_width = min(_px(540, scale), width - 2 * SCREEN_MARGIN)
    card_height = (
        pad + _text_height(title_size, True) + _px(12, scale) + bar_height + _px(12, scale)
        + _text_height(body_size) + _px(12, scale) + hints_height + pad
    )
    card_x = (width - card_width) // 2
    card_y = max(0, height - _px(STATUS_BAR_HEIGHT, scale) - _px(18, scale) - card_height)
    inner_x, inner_width = card_x + pad, card_width - 2 * pad
    _draw_box(canvas, card_x, card_y, card_width, card_height, _px(18, scale), border=COLOR_BORDER)

    # Titolo e contatore dei campioni
    y = card_y + pad
    counter = f"{done}/{total}"
    counter_width = _text_width(counter, title_size, True)
    title = f"Registrazione di {enrollment.name}"
    size = _fit_font_size(title, title_size, True, inner_width - counter_width - _px(16, scale))
    _draw_text(canvas, title, inner_x, y + (_text_height(title_size, True) - _text_height(size, True)) // 2, size, bold=True)
    _draw_text(canvas, counter, inner_x + inner_width - counter_width, y, title_size, COLOR_ENROLLING, bold=True)

    y += _text_height(title_size, True) + _px(12, scale)
    _draw_box(canvas, inner_x, y, inner_width, bar_height, bar_height // 2, COLOR_FIELD, 1.0)
    filled = int(inner_width * done / total) if total else 0
    if filled:
        _draw_box(canvas, inner_x, y, max(filled, bar_height), bar_height, bar_height // 2, COLOR_ENROLLING, 1.0)

    y += bar_height + _px(12, scale)
    feedback = enrollment.feedback or "Guarda la webcam e muovi leggermente la testa"
    size = _fit_font_size(feedback, body_size, False, inner_width)
    _draw_text(canvas, feedback, inner_x, y, size, COLOR_MUTED)

    y += _text_height(body_size) + _px(12, scale)
    _draw_key_hint(canvas, "Esc", "Annulla", inner_x, y, scale)


def _draw_camera_notice(canvas: np.ndarray, scale: float) -> None:
    """Avviso al centro dello schermo quando la webcam e scollegata."""
    height, width = canvas.shape[:2]
    title, body = "Webcam non disponibile", "Riconnessione in corso..."
    title_size, body_size = _px(22, scale), _px(15, scale)
    pad, badge, gap = _px(24, scale), _px(44, scale), _px(18, scale)
    texts_height = _text_height(title_size, True) + _text_height(body_size)
    card_width = min(
        2 * pad + badge + gap + max(_text_width(title, title_size, True), _text_width(body, body_size)),
        width - 2 * SCREEN_MARGIN,
    )
    card_height = 2 * pad + max(badge, texts_height)
    x, y = (width - card_width) // 2, (height - card_height) // 2
    _draw_box(canvas, x, y, card_width, card_height, _px(18, scale), border=COLOR_WARNING)
    _draw_badge(canvas, x + pad, y + (card_height - badge) // 2, badge, COLOR_WARNING, "!")
    text_x, text_y = x + pad + badge + gap, y + (card_height - texts_height) // 2
    _draw_text(canvas, title, text_x, text_y, title_size, bold=True)
    _draw_text(canvas, body, text_x, text_y + _text_height(title_size, True), body_size, COLOR_MUTED)


def _draw_status_bar(
    canvas: np.ndarray,
    ui_fps: float,
    inference_fps: float,
    gallery_ok: bool,
    server_online: bool,
    camera_connected: bool,
    scale: float,
) -> None:
    """Barra di stato in basso: galleria, server e webcam a sinistra, frame rate e uscita a destra."""
    height, width = canvas.shape[:2]
    bar_height = _px(STATUS_BAR_HEIGHT, scale)
    top = height - bar_height
    size = _px(14, scale)
    text_y = top + (bar_height - _text_height(size)) // 2
    bar = canvas[top:]
    bar[:] = (bar * (1.0 - PANEL_OPACITY) + np.array(COLOR_PANEL) * PANEL_OPACITY).astype(np.uint8)

    x = SCREEN_MARGIN
    x += _draw_text(canvas, "Face Access", x, text_y, size, bold=True) + _px(22, scale)
    indicators = (
        ("Galleria" if gallery_ok else "Galleria in attesa", COLOR_KNOWN if gallery_ok else COLOR_WARNING),
        ("Server" if server_online else "Server non collegato", COLOR_KNOWN if server_online else COLOR_UNKNOWN),
        ("Webcam" if camera_connected else "Webcam scollegata", COLOR_KNOWN if camera_connected else COLOR_UNKNOWN),
    )
    dot = _px(9, scale)
    for label, color in indicators:
        _blend(canvas, _rounded_mask(dot, dot, dot // 2), x, top + (bar_height - dot) // 2, color)
        x += dot + _px(7, scale)
        x += _draw_text(canvas, label, x, text_y, size, COLOR_TEXT) + _px(20, scale)

    # A destra: tasto di uscita e, se c'e spazio, i frame rate
    key_size = _px(13, scale)
    key_y = top + (bar_height - _text_height(key_size, True) - _px(6, scale)) // 2
    hint_width = _text_width("Q", key_size, True) + _px(24, scale) + _text_width("Esci", key_size)
    right = width - SCREEN_MARGIN - hint_width
    if right > x:
        _draw_key_hint(canvas, "Q", "Esci", right, key_y, scale)
    rates = f"Video {ui_fps:.0f} fps  ·  AI {inference_fps:.1f} fps"
    rates_x = right - _px(22, scale) - _text_width(rates, size)
    if rates_x > x:
        _draw_text(canvas, rates, rates_x, text_y, size, COLOR_MUTED)
