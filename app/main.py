"""Punto d'ingresso dell'applicazione Face Access.

Avvio dalla cartella principale del progetto:
    python -m app.main
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from .camera_stream import CameraStream
from .config import ConfigurationError, Settings
from .database_manager import DatabaseError, DatabaseManager
from .enrollment import EnrollmentSession, EnrollmentState
from .face_processor import FaceProcessor
from .inference_worker import InferenceWorker
from .models import FrameResult
from .recognition_service import RecognitionService
from .ui_controller import NO_KEY, UIController

logger = logging.getLogger("face_access")

QUIT_KEYS = (ord("q"), ord("Q"))
FPS_SMOOTHING = 0.1
GREETING_TOAST_SECONDS = 4.0
NOTICE_TOAST_SECONDS = 2.5
FRAME_WAIT_TIMEOUT_S = 0.05   # attesa massima di un nuovo fotogramma: la UI resta reattiva ai tasti
RESULT_MAX_AGE_S = 3.0        # un risultato di inferenza piu vecchio non descrive piu la scena


class FaceAccessApp:
    """Compone i componenti e gestisce il ciclo principale sul thread della UI."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._database = DatabaseManager(settings)
        try:
            self._camera = CameraStream(settings.camera_index, settings.camera_width, settings.camera_height)
            self._processor = FaceProcessor(settings)
            self._recognizer = RecognitionService(settings, self._database)
            self._worker = InferenceWorker(self._camera, self._processor, self._recognizer)
            self._enrollment = EnrollmentSession(settings, self._database)
            self._ui = UIController(settings.camera_width, settings.camera_height)
        except BaseException:
            self._database.close()
            raise
        self._ui_fps = 0.0
        # Id dell'ultimo fotogramma il cui risultato e stato valutato per la registrazione
        self._last_enrollment_frame_id = -1

    def run(self) -> None:
        """Avvia i thread di servizio e rimane nel ciclo UI finche l'utente non esce."""
        try:
            self._camera.start()
            self._worker.start()
            logger.info("Applicazione avviata. Premi Q per uscire.")
            self._ui_loop()
        finally:
            self._shutdown()

    def _ui_loop(self) -> None:
        """Ciclo UI: disegna l'ultimo frame con l'ultimo risultato e gestisce registrazione e tasti."""
        blank = np.zeros((self._settings.camera_height, self._settings.camera_width, 3), dtype=np.uint8)
        frame: np.ndarray | None = None
        last_frame_id = -1
        last_tick = time.perf_counter()
        smoothed_interval: float | None = None
        while True:
            # L'attesa del fotogramma limita il ciclo alla velocita della webcam e lascia
            # la CPU al thread di inferenza
            packet = self._camera.wait_for_frame(last_frame_id, FRAME_WAIT_TIMEOUT_S)
            if packet is not None:
                last_frame_id, frame = packet
            camera_connected = self._camera.is_connected

            result = self._current_result(camera_connected)
            greeted = self._worker.drain_greetings()
            if greeted:
                self._ui.show_toast(_greeting_text(greeted), GREETING_TOAST_SECONDS)
            self._update_enrollment(result)

            canvas = self._ui.render(
                frame if frame is not None else blank,  # webcam non ancora pronta: schermata nera
                result,
                self._enrollment,
                camera_connected,
                self._ui_fps,
                self._worker.fps,
            )
            self._ui.show(canvas)

            key = self._ui.wait_key(1)
            if self._handle_key(key) or self._ui.is_closed():
                break

            # Si media l'intervallo tra due cicli e non il suo inverso: con intervalli
            # irregolari la media degli fps istantanei risulterebbe piu alta del reale
            now = time.perf_counter()
            interval = now - last_tick
            smoothed_interval = interval if smoothed_interval is None else (
                (1 - FPS_SMOOTHING) * smoothed_interval + FPS_SMOOTHING * interval
            )
            self._ui_fps = 1.0 / max(smoothed_interval, 1e-6)
            last_tick = now

    def _current_result(self, camera_connected: bool) -> FrameResult | None:
        """Restituisce l'ultimo risultato di inferenza, se descrive ancora la scena inquadrata."""
        result = self._worker.latest_result()
        if result is None or not camera_connected:
            return None
        if time.monotonic() - result.created_at > RESULT_MAX_AGE_S:
            return None
        return result

    def _handle_key(self, key: int) -> bool:
        """Gestisce il tasto premuto. Restituisce True se l'utente chiede di uscire."""
        if key == NO_KEY:
            return False
        if self._enrollment.is_active:
            self._enrollment.handle_key(key)
            return False
        return key in QUIT_KEYS

    def _update_enrollment(self, result: FrameResult | None) -> None:
        """Avvia, alimenta e conclude la registrazione in base all'ultimo risultato di inferenza."""
        enrollment = self._enrollment

        # 1. Ogni risultato di inferenza viene valutato una sola volta, e solo se il database ha risposto
        if result is not None and result.db_ok and result.frame_id > self._last_enrollment_frame_id:
            self._last_enrollment_frame_id = result.frame_id
            if enrollment.state is EnrollmentState.IDLE:
                # Un volto sconosciuto e solo nel fotogramma da qualche istante: si propone la registrazione
                if enrollment.consider_start(result.detections):
                    self._ui.show_toast("Nuovo volto rilevato", NOTICE_TOAST_SECONDS)
            elif enrollment.state is EnrollmentState.CAPTURING:
                enrollment.offer_frame(result.detections)

        # 2. Scadenze e esiti finali
        enrollment.tick()
        if enrollment.state is EnrollmentState.COMPLETED:
            self._ui.show_toast(f"Profilo creato. Benvenuto, {enrollment.name}!", GREETING_TOAST_SECONDS)
            if enrollment.person_id is not None:
                self._recognizer.register_greeting(enrollment.person_id)
            # I risultati calcolati sui fotogrammi precedenti al salvataggio vedono ancora la persona
            # come sconosciuta: vengono ignorati, altrimenti la registrazione ripartirebbe subito
            newest_frame_id, _ = self._camera.latest()
            self._last_enrollment_frame_id = max(self._last_enrollment_frame_id, newest_frame_id)
            enrollment.reset()
        elif enrollment.state is EnrollmentState.CANCELLED:
            self._ui.show_toast(enrollment.error or "Registrazione annullata", GREETING_TOAST_SECONDS)
            enrollment.reset()

    def _shutdown(self) -> None:
        """Chiude in ordine thread, finestre e connessioni al database."""
        logger.info("Chiusura in corso...")
        self._worker.stop()
        self._camera.stop()
        self._ui.destroy()
        self._database.close()
        logger.info("Applicazione chiusa")


def _greeting_text(names: list[str]) -> str:
    """Compone il saluto per una o piu persone riconosciute nello stesso momento."""
    unique = list(dict.fromkeys(names))
    if len(unique) == 1:
        return f"Bentornato, {unique[0]}!"
    return f"Bentornati, {', '.join(unique[:-1])} e {unique[-1]}!"


def main() -> int:
    """Carica la configurazione, inizializza i componenti e avvia l'applicazione."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        settings = Settings.from_env()
        app = FaceAccessApp(settings)
    except (ConfigurationError, DatabaseError) as exc:
        logger.error("Avvio non riuscito: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Avvio interrotto dall'utente")
        return 1
    except Exception:
        logger.exception("Errore inatteso durante l'inizializzazione dei modelli AI o della webcam")
        return 1

    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrotto dall'utente")
    except Exception:
        logger.exception("Errore inatteso durante l'esecuzione")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
