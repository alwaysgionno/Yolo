"""Test di PersonStore: letture sulla galleria, scritture come eventi MQTT, formato dei messaggi."""

from __future__ import annotations

import base64
import unittest
import uuid
from typing import Any

import numpy as np

from app import contract
from app.contract import ContractError
from app.gallery import Gallery
from app.mqtt_link import LinkError
from app.person_store import PersonStore, StoreError
from tests.helpers import EMBEDDING_SIZE, FakeClock, face_vector

NOW = 1_800_000_000.0
NOW_ISO = "2027-01-15T08:00:00+00:00"


class FakeLink:
    """Sostituisce MqttLink: raccoglie gli eventi pubblicati e puo simulare la coda piena."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.full = False

    def publish_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self.full:
            raise LinkError("coda piena simulata")
        self.events.append((topic, payload))


class PersonStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock(start=NOW)
        self.gallery = Gallery(wall_clock=self.clock)
        self.link = FakeLink()
        self.store = PersonStore(self.gallery, self.link, wall_clock=self.clock)  # type: ignore[arg-type]

    def receive_gallery(self) -> None:
        self.gallery.apply_info_message(contract.encode_json({"person_uids": []}))


class GalleryNotReadyTest(PersonStoreTestCase):
    def test_reads_and_enrollment_wait_for_the_gallery(self) -> None:
        with self.assertRaises(StoreError):
            self.store.find_nearest(face_vector(1), 5)
        with self.assertRaises(StoreError):
            self.store.person_exists("Mario")
        with self.assertRaises(StoreError):
            self.store.create_person("Mario", [face_vector(1)], [90.0])
        self.assertEqual(self.link.events, [])


class EventsTest(PersonStoreTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.receive_gallery()

    def test_enrollment_request_is_published_and_waits_for_approval(self) -> None:
        photo = np.full((320, 320, 3), 128, dtype=np.uint8)
        person_uid = self.store.create_person("Mario", [face_vector(1), face_vector(2)], [90.0, 95.0], photo)

        (topic, event), = self.link.events
        self.assertEqual(topic, contract.TOPIC_EVENT_ENROLLED)
        self.assertEqual(event["person_uid"], person_uid)
        self.assertEqual(event["full_name"], "Mario")
        self.assertEqual(event["occurred_at"], NOW_ISO)
        uuid.UUID(event["event_id"])
        uuid.UUID(person_uid)
        self.assertEqual([sample["quality"] for sample in event["samples"]], [90.0, 95.0])
        np.testing.assert_array_equal(contract.decode_embedding(event["samples"][1]["embedding"]), face_vector(2))

        jpeg = base64.b64decode(event["photo"])
        self.assertTrue(jpeg.startswith(b"\xff\xd8\xff"))
        self.assertLessEqual(len(jpeg), contract.MAX_PHOTO_BYTES)

        # In attesa: il nome e riservato, ma il volto non viene riconosciuto
        self.assertTrue(self.store.person_exists("mario"))
        self.assertEqual(self.store.find_nearest(face_vector(1), 5), [])
        self.assertEqual(self.store.find_pending(face_vector(1)).person_id, person_uid)

    def test_enrollment_without_photo_has_no_photo_field(self) -> None:
        self.store.create_person("Mario", [face_vector(1)], [90.0])
        self.assertNotIn("photo", self.link.events[0][1])

    def test_enrollment_needs_one_quality_per_embedding(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create_person("Mario", [face_vector(1), face_vector(2)], [90.0])
        with self.assertRaises(ValueError):
            self.store.create_person("Mario", [], [])

    def test_enrollment_that_cannot_be_queued_is_not_kept_locally(self) -> None:
        self.link.full = True
        with self.assertRaises(StoreError):
            self.store.create_person("Mario", [face_vector(1)], [90.0])
        self.assertFalse(self.store.person_exists("Mario"))

    def test_greeting_is_published_and_updates_the_last_access(self) -> None:
        person_uid = "11111111-1111-1111-1111-111111111111"
        self.gallery.apply_person_message(
            person_uid,
            contract.encode_json({
                "full_name": "Mario", "last_access_at": NOW_ISO,
                "embeddings": [contract.encode_embedding(face_vector(1))],
            }),
        )
        self.clock.advance(700.0)
        self.assertEqual(self.store.find_nearest(face_vector(1), 1)[0].last_access_age_s, 700.0)

        self.store.record_greeting(person_uid, 0.8)
        topic, event = self.link.events[-1]
        self.assertEqual(topic, contract.TOPIC_EVENT_GREETED)
        self.assertEqual((event["person_uid"], event["similarity"]), (person_uid, 0.8))
        self.assertEqual(self.store.find_nearest(face_vector(1), 1)[0].last_access_age_s, 0.0)

    def test_continuous_embedding_is_published(self) -> None:
        self.store.add_continuous_embedding("uid-mario", face_vector(3), quality=88.0)
        (topic, event), = self.link.events
        self.assertEqual(topic, contract.TOPIC_EVENT_EMBEDDING)
        self.assertEqual((event["person_uid"], event["quality"]), ("uid-mario", 88.0))
        np.testing.assert_array_equal(contract.decode_embedding(event["embedding"]), face_vector(3))

    def test_every_event_has_its_own_id(self) -> None:
        self.store.record_greeting("uid-mario", 0.8)
        self.store.record_greeting("uid-mario", 0.8)
        first, second = (event["event_id"] for _, event in self.link.events)
        self.assertNotEqual(first, second)

    def test_full_queue_becomes_store_error(self) -> None:
        self.link.full = True
        with self.assertRaises(StoreError):
            self.store.record_greeting("uid-mario", 0.8)


class ContractTest(unittest.TestCase):
    def test_embedding_round_trip(self) -> None:
        vector = np.linspace(-1.0, 1.0, EMBEDDING_SIZE, dtype=np.float32)
        text = contract.encode_embedding(vector)
        self.assertEqual(len(text), 2732)  # 512 float32 = 2048 byte in base64
        np.testing.assert_array_equal(contract.decode_embedding(text), vector)

    def test_wrong_embeddings_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            contract.encode_embedding(np.zeros(128, dtype=np.float32))
        not_finite = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
        not_finite[3] = np.nan
        for text in (None, "non base64!", "AAAA", contract.encode_embedding(not_finite)):
            with self.subTest(text=str(text)[:12]):
                with self.assertRaises(ContractError):
                    contract.decode_embedding(text)

    def test_timestamp_round_trip(self) -> None:
        self.assertEqual(contract.format_timestamp(NOW), NOW_ISO)
        self.assertEqual(contract.parse_timestamp(NOW_ISO), NOW)
        self.assertEqual(contract.parse_timestamp("2027-01-15T09:00:00+01:00"), NOW)
        for text in (None, "ieri", "2027-01-15T08:00:00"):  # l'ultima e senza fuso orario
            with self.assertRaises(ContractError):
                contract.parse_timestamp(text)


if __name__ == "__main__":
    unittest.main()
