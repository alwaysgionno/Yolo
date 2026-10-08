"""Thread di inferenza: esegue YOLO, InsightFace e il riconoscimento in background.

Il thread legge sempre il fotogramma piu recente e salta quelli intermedi: in questo
modo la latenza dell'intelligenza artificiale non si accumula e la UI resta fluida.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from .camera_stream import CameraStream
from .database_manager import DatabaseError
from .face_processor import FaceProcessor
from .models import FrameResult
from .recognition_service import RecognitionService

logger = logging.getLogger(__name__)

PACKET_TIMEOUT_S = 0.2
RETRY_DELAY_S = 0.5
FPS_SMOOTHING = 0.2


class InferenceWorker(threading.Thread):
    """Consuma i fotogrammi della webcam e pubblica l'ultimo risultato disponibile."""

    def __init__(
        self,
        camera: CameraStream,
        processor: FaceProcessor,
        recognizer: RecognitionService,
    ) -> None:
        super().__init__(name="InferenceWorker", daemon=True)
        self._camera = camera
        self._processor = processor
        self._recognizer = recognizer
        self._stop_event = threading.Event()
        self._result_lock = threading.Lock()
        self._latest: FrameResult | None = None
        self._greetings: queue.Queue[str] = queue.Queue()
        self._fps = 0.0
        self._db_ok = True  # esito dell'ultimo accesso al database, usato solo dal thread di inferenza

    @property
    def fps(self) -> float:
        """Frame rate medio dell'inferenza (fotogrammi elaborati al secondo)."""
        return self._fps

    def stop(self) -> None:
        """Richiede l'arresto del thread e ne attende la chiusura."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=5.0)

    def latest_result(self) -> FrameResult | None:
        """Restituisce l'ultimo risultato di inferenza disponibile."""
        with self._result_lock:
            return self._latest

    def drain_greetings(self) -> list[str]:
        """Restituisce tutti i nomi da salutare accumulati dall'ultima chiamata."""
        names: list[str] = []
        while True:
            try:
                names.append(self._greetings.get_nowait())
            except queue.Empty:
                return names

    def run(self) -> None:
        """Ciclo principale: attende un frame nuovo, lo elabora e pubblica il risultato."""
        last_frame_id = -1
        last_tick: float | None = None
        smoothed_interval: float | None = None
        while not self._stop_event.is_set():
            packet = self._camera.wait_for_frame(last_frame_id, PACKET_TIMEOUT_S)
            if packet is None:  # webcam ferma: nessuna inferenza in corso
                self._fps = 0.0
                last_tick = None
                smoothed_interval = None
                continue
            last_frame_id, frame = packet
            try:
                result = self._infer(frame, last_frame_id)
            except Exception:
                logger.exception("Errore durante l'inferenza: riprovo sul fotogramma successivo")
                self._stop_event.wait(RETRY_DELAY_S)
                continue

            with self._result_lock:
                self._latest = result
            # Media dell'intervallo tra due risultati (non dei suoi inversi, che la falserebbero)
            now = time.perf_counter()
            if last_tick is not None:
                interval = now - last_tick
                smoothed_interval = interval if smoothed_interval is None else (
                    (1 - FPS_SMOOTHING) * smoothed_interval + FPS_SMOOTHING * interval
                )
                self._fps = 1.0 / max(smoothed_interval, 1e-6)
            last_tick = now
        logger.info("Thread di inferenza terminato")

    def _infer(self, frame: np.ndarray, frame_id: int) -> FrameResult:
        """Esegue la pipeline completa su un singolo fotogramma."""
        started = time.perf_counter()
        detections = self._processor.process(frame)

        db_ok = True
        try:
            greetings = self._recognizer.process(detections)
        except DatabaseError as exc:
            if self._db_ok:  # si registra solo il cambio di stato, non un errore per fotogramma
                logger.error("Database non disponibile durante il riconoscimento: %s", exc)
            greetings, db_ok = [], False
        if db_ok and not self._db_ok:
            logger.info("Database di nuovo disponibile: riconoscimento ripreso")
        self._db_ok = db_ok

        for name in greetings:
            self._greetings.put(name)
        return FrameResult(
            frame_id=frame_id,
            detections=detections,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            db_ok=db_ok,
        )
