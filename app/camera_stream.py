"""Thread dedicato alla cattura dei fotogrammi dalla webcam.

La cattura gira in un thread separato e conserva solo l'ultimo fotogramma:
i consumatori (inferenza e interfaccia) non accumulano mai ritardo, quindi il
video resta fluido anche quando l'elaborazione AI e piu lenta.
"""

from __future__ import annotations

import logging
import sys
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

READ_RETRY_DELAY_S = 0.02
IS_WINDOWS = sys.platform.startswith("win")


class CameraStream:
    """Legge continuamente la webcam e gestisce la riconnessione automatica."""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        reconnect_delay: float = 2.0,
        max_read_failures: int = 30,
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._reconnect_delay = reconnect_delay
        self._max_read_failures = max_read_failures

        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._connected = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="CameraStream", daemon=True)

    def start(self) -> None:
        """Avvia il thread di cattura."""
        self._thread.start()

    def stop(self) -> None:
        """Ferma il thread di cattura e rilascia la webcam."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    @property
    def is_connected(self) -> bool:
        """True se l'ultima lettura dalla webcam e andata a buon fine."""
        with self._condition:
            return self._connected

    def latest(self) -> tuple[int, np.ndarray | None]:
        """Restituisce (id, frame) dell'ultimo fotogramma disponibile, senza attendere."""
        with self._condition:
            return self._frame_id, self._frame

    def wait_for_frame(self, last_frame_id: int, timeout: float) -> tuple[int, np.ndarray] | None:
        """Attende un fotogramma piu recente di `last_frame_id`.

        Returns:
            Tupla (id, frame), oppure None se scade il timeout.
        """
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._frame is not None and self._frame_id != last_frame_id,
                timeout=timeout,
            )
            if not ready or self._frame is None:
                return None
            return self._frame_id, self._frame

    def _open_capture(self) -> cv2.VideoCapture | None:
        """Apre la webcam provando i backend piu adatti al sistema operativo."""
        # Su Windows DirectShow si apre subito; se non riesce si lascia scegliere a OpenCV
        backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if IS_WINDOWS else (cv2.CAP_ANY,)
        capture: cv2.VideoCapture | None = None
        for backend in backends:
            try:
                candidate = cv2.VideoCapture(self._camera_index, backend)
            except cv2.error:
                continue
            if candidate.isOpened():
                capture = candidate
                break
            candidate.release()
        if capture is None:
            return None

        if IS_WINDOWS:
            # Senza MJPG molte webcam scendono a pochi fps quando si chiede l'alta definizione
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # riduce la latenza del buffer interno
        logger.info(
            "Webcam %d aperta: richiesti %dx%d, ottenuti %dx%d",
            self._camera_index,
            self._width,
            self._height,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return capture

    def _set_connected(self, value: bool) -> None:
        with self._condition:
            self._connected = value

    def _run(self) -> None:
        """Ciclo di cattura: apre la webcam, legge i frame e si riconnette se necessario."""
        capture: cv2.VideoCapture | None = None
        failures = 0
        unavailable_reported = False
        try:
            while not self._stop_event.is_set():
                if capture is None:
                    capture = self._open_capture()
                    if capture is None:
                        self._set_connected(False)
                        if not unavailable_reported:  # un solo avviso, non uno a ogni tentativo
                            logger.warning(
                                "Webcam %d non disponibile: controlla che sia collegata e il valore di "
                                "CAMERA_INDEX nel file .env. Nuovo tentativo ogni %.0f secondi",
                                self._camera_index,
                                self._reconnect_delay,
                            )
                            unavailable_reported = True
                        self._stop_event.wait(self._reconnect_delay)
                        continue
                    unavailable_reported = False

                try:
                    ok, frame = capture.read()
                except cv2.error:
                    ok, frame = False, None

                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    self._set_connected(False)
                    if failures >= self._max_read_failures:
                        logger.warning("Lettura dalla webcam fallita %d volte: riconnessione", failures)
                        capture.release()
                        capture = None
                        failures = 0
                        self._stop_event.wait(self._reconnect_delay)
                    else:
                        self._stop_event.wait(READ_RETRY_DELAY_S)
                    continue

                failures = 0
                with self._condition:
                    self._frame = frame
                    self._frame_id += 1
                    self._connected = True
                    self._condition.notify_all()
        finally:
            if capture is not None:
                capture.release()
            self._set_connected(False)
            logger.info("Thread di cattura webcam terminato")
