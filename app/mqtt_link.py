"""Collegamento MQTT con il server: riceve la galleria e pubblica gli eventi.

Il client paho lavora su un proprio thread di rete e si riconnette da solo. Gli eventi
pubblicati mentre il broker non e raggiungibile restano in coda in memoria (al massimo
`MAX_QUEUED_EVENTS`) e partono alla riconnessione.
"""

from __future__ import annotations

import logging
from typing import Any

import paho.mqtt.client as mqtt

from . import contract
from .config import Settings
from .contract import ContractError
from .gallery import Gallery

logger = logging.getLogger(__name__)

CLIENT_ID = "face-access-camera"
QOS = 1                    # consegna garantita, eventualmente ripetuta: il server scarta i doppioni
KEEPALIVE_S = 30
MAX_QUEUED_EVENTS = 1000
CLOSE_FLUSH_TIMEOUT_S = 3.0


class LinkError(RuntimeError):
    """L'evento non e stato accettato dal client MQTT (coda piena)."""


class MqttLink:
    """Connessione al broker: alimenta la galleria e pubblica gli eventi della telecamera."""

    def __init__(self, settings: Settings, gallery: Gallery) -> None:
        self._settings = settings
        self._gallery = gallery
        self._server_online = False
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID, clean_session=True)
        self._client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
        # Se la telecamera sparisce senza chiudere la connessione, il broker pubblica "offline" al suo posto
        self._client.will_set(contract.TOPIC_CAMERA_STATUS, contract.encode_status(False), qos=QOS, retain=True)
        self._client.max_queued_messages_set(MAX_QUEUED_EVENTS)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    @property
    def server_online(self) -> bool:
        """True se il broker e raggiungibile e il servizio del server che salva gli eventi e attivo."""
        return self._client.is_connected() and self._server_online

    def start(self) -> None:
        """Avvia la connessione in background: non blocca se il broker non e raggiungibile."""
        self._client.connect_async(self._settings.mqtt_host, self._settings.mqtt_port, keepalive=KEEPALIVE_S)
        self._client.loop_start()

    def publish_event(self, topic: str, payload: dict[str, Any]) -> None:
        """Mette in coda un evento per il server.

        Raises:
            LinkError: se la coda e piena (server irraggiungibile da molto tempo).
        """
        info = self._client.publish(topic, contract.encode_json(payload), qos=QOS)
        # MQTT_ERR_NO_CONN non e un errore: il messaggio resta in coda fino alla riconnessione
        if info.rc == mqtt.MQTT_ERR_QUEUE_SIZE:
            raise LinkError("coda degli eventi piena: il server non e raggiungibile")

    def close(self) -> None:
        """Pubblica lo stato "offline", attende brevemente l'invio degli eventi in coda e si disconnette."""
        info = self._client.publish(contract.TOPIC_CAMERA_STATUS, contract.encode_status(False), qos=QOS, retain=True)
        try:
            # I messaggi partono in ordine: quando parte questo, gli eventi precedenti sono gia partiti
            info.wait_for_publish(timeout=CLOSE_FLUSH_TIMEOUT_S)
            flushed = info.is_published()
        except (RuntimeError, ValueError):  # paho li solleva se il broker non e connesso
            flushed = False
        if not flushed:
            logger.warning("Chiusura con il server non raggiungibile: gli eventi ancora in coda vanno persi")
        self._client.disconnect()
        self._client.loop_stop()

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if reason_code.is_failure:
            logger.error("Connessione al broker MQTT rifiutata: %s (controlla MQTT_PASSWORD)", reason_code)
            return
        logger.info("Connesso al broker MQTT (%s:%d)", self._settings.mqtt_host, self._settings.mqtt_port)
        # A ogni iscrizione il broker rimanda i messaggi retained: la galleria si riallinea da sola
        client.subscribe(
            [
                (contract.TOPIC_GALLERY_ALL, QOS),
                (contract.TOPIC_GALLERY_INFO, QOS),
                (contract.TOPIC_SERVER_STATUS, QOS),
            ]
        )
        client.publish(contract.TOPIC_CAMERA_STATUS, contract.encode_status(True), qos=QOS, retain=True)

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        self._server_online = False
        if reason_code.is_failure:  # una chiusura richiesta da close() non e un guasto
            logger.warning("Connessione al broker MQTT persa (%s): nuovo tentativo in corso", reason_code)

    def _on_message(self, client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
        topic, payload = message.topic, message.payload
        try:
            if topic.startswith(contract.TOPIC_GALLERY_PREFIX):
                self._gallery.apply_person_message(topic.removeprefix(contract.TOPIC_GALLERY_PREFIX), payload)
            elif topic == contract.TOPIC_GALLERY_INFO:
                self._gallery.apply_info_message(payload)
            elif topic == contract.TOPIC_SERVER_STATUS:
                self._server_online = contract.decode_json(payload).get("state") == "online"
        except ContractError as exc:
            logger.error("Messaggio scartato su %s: %s", topic, exc)
        except Exception:
            # Un'eccezione qui fermerebbe il thread di rete di paho
            logger.exception("Errore nell'elaborazione del messaggio su %s", topic)
