"""Test di RecognitionService: soglia, saluti e apprendimento continuo."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from app.person_store import StoreError
from app.models import Greeting, GreetingKind, Match
from app.recognition_service import RecognitionService
from tests.helpers import FakeClock, FakeStore, face_vector, make_detection, make_settings

WELCOME_MARIO = Greeting(GreetingKind.WELCOME, "Mario")
WELCOME_BACK_MARIO = Greeting(GreetingKind.WELCOME_BACK, "Mario")


class RecognitionServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.db = FakeStore()
        self.service = RecognitionService(make_settings(), self.db, clock=self.clock)

    def see(
        self,
        similarity: float,
        *,
        axis: int = 1,
        sharpness: float = 120.0,
        last_access_age_s: float | None = None,
    ) -> list[Greeting]:
        """Simula un fotogramma in cui il vicino piu simile e Mario con la similarita indicata.

        `last_access_age_s` e l'eta dell'ultimo saluto noto alla galleria (None: mai salutato).
        """
        self.db.nearest = [Match("uid-mario", "Mario", similarity, last_access_age_s), Match("uid-anna", "Anna", 0.10)]
        return self.service.process([make_detection(face_vector(axis), sharpness=sharpness)])


class ThresholdTest(RecognitionServiceTestCase):
    def test_face_below_threshold_stays_unknown(self) -> None:
        self.db.nearest = [Match("uid-mario", "Mario", 0.39)]
        detection = make_detection()
        self.assertEqual(self.service.process([detection]), [])
        self.assertIsNone(detection.match)
        self.assertEqual(self.db.greetings, [])

    def test_empty_gallery_means_unknown(self) -> None:
        detection = make_detection()
        self.assertEqual(self.service.process([detection]), [])
        self.assertIsNone(detection.match)

    def test_face_at_threshold_is_recognized(self) -> None:
        self.db.nearest = [Match("uid-mario", "Mario", 0.40)]
        detection = make_detection()
        self.assertEqual(self.service.process([detection]), [WELCOME_MARIO])
        self.assertEqual(detection.match, Match("uid-mario", "Mario", 0.40))

    def test_missing_gallery_is_reported_to_the_caller(self) -> None:
        self.db.failing.add("find_nearest")
        with self.assertRaises(StoreError):
            self.service.process([make_detection()])


class GreetingTest(RecognitionServiceTestCase):
    def test_welcome_only_at_the_first_recognition_ever(self) -> None:
        self.assertEqual(self.see(0.45), [WELCOME_MARIO])
        self.clock.advance(1.0)
        self.assertEqual(self.see(0.45), [])
        self.assertEqual(self.db.greetings, [("uid-mario", 0.45)])

    def test_no_greeting_while_the_person_stays_in_view(self) -> None:
        self.see(0.45)
        for _ in range(400):  # 20 minuti davanti alla webcam, un riconoscimento ogni 3 secondi
            self.clock.advance(3.0)
            self.assertEqual(self.see(0.45), [])

    def test_welcome_back_only_after_ten_minutes_out_of_view(self) -> None:
        self.assertEqual(self.see(0.45, last_access_age_s=30.0), [])
        self.clock.advance(599.0)
        self.assertEqual(self.see(0.45), [])  # rivisto: l'assenza si misura da qui, non dall'ultimo saluto
        self.clock.advance(599.0)
        self.assertEqual(self.see(0.45), [])
        self.clock.advance(600.0)
        self.assertEqual(self.see(0.47), [WELCOME_BACK_MARIO])
        self.assertEqual(self.db.greetings, [("uid-mario", 0.47)])

    def test_person_waiting_for_approval_is_marked_but_not_greeted(self) -> None:
        self.db.pending = Match("uid-carla", "Carla", 0.9, pending=True)
        detection = make_detection()
        self.assertEqual(self.service.process([detection]), [])
        self.assertTrue(detection.match.pending)
        self.assertEqual((self.db.greetings, self.db.continuous), ([], []))

    def test_approved_person_wins_over_a_pending_one(self) -> None:
        self.db.nearest = [Match("uid-mario", "Mario", 0.70)]
        self.db.pending = Match("uid-carla", "Carla", 0.9, pending=True)
        detection = make_detection()
        self.assertEqual(self.service.process([detection]), [WELCOME_MARIO])
        self.assertEqual(detection.match.person_id, "uid-mario")

    def test_pending_face_below_threshold_stays_unknown(self) -> None:
        self.db.pending = Match("uid-carla", "Carla", 0.2, pending=True)
        detection = make_detection()
        self.service.process([detection])
        self.assertIsNone(detection.match)

    def test_person_just_approved_is_not_greeted_again(self) -> None:
        # Il server registra la richiesta come primo accesso: poco dopo non c'e un secondo saluto
        self.assertEqual(self.see(0.60, last_access_age_s=0.5), [])
        self.assertEqual(self.db.greetings, [])

    def test_first_sighting_after_restart_relies_on_the_saved_access(self) -> None:
        self.assertEqual(self.see(0.60, last_access_age_s=600.0), [WELCOME_BACK_MARIO])

    def test_same_person_detected_twice_is_greeted_once(self) -> None:
        self.db.nearest = [Match("uid-mario", "Mario", 0.70)]
        self.assertEqual(self.service.process([make_detection(), make_detection()]), [WELCOME_MARIO])

    def test_greeting_is_shown_even_if_the_log_cannot_be_written(self) -> None:
        self.db.failing.add("record_greeting")
        self.assertEqual(self.see(0.45), [WELCOME_MARIO])
        self.assertEqual(self.db.greetings, [])

    def test_failure_in_the_middle_of_a_frame_does_not_lose_greetings(self) -> None:
        mario, anna = [Match("uid-mario", "Mario", 0.70)], [Match("uid-anna", "Anna", 0.70, last_access_age_s=900.0)]
        faces = [make_detection(), make_detection()]

        # Mario viene identificato, poi la ricerca di Anna fallisce: nessun saluto deve andare perso
        with mock.patch.object(self.db, "find_nearest", side_effect=[mario, StoreError("simulato")]):
            with self.assertRaises(StoreError):
                self.service.process(faces)
        self.assertEqual(self.db.greetings, [])

        self.clock.advance(0.3)
        with mock.patch.object(self.db, "find_nearest", side_effect=[mario, anna]):
            greetings = self.service.process(faces)
        self.assertEqual(greetings, [WELCOME_MARIO, Greeting(GreetingKind.WELCOME_BACK, "Anna")])


class ContinuousLearningTest(RecognitionServiceTestCase):
    def see_batch(self, similarity: float, sharpness: float = 90.0) -> None:
        for axis in (1, 2, 3):  # CONTINUOUS_BATCH_SIZE = 3 nelle impostazioni di test
            self.see(similarity, axis=axis, sharpness=sharpness)

    def test_confident_matches_are_averaged_and_saved(self) -> None:
        self.see(0.80, axis=1, sharpness=80.0)
        self.see(0.80, axis=2, sharpness=90.0)
        self.assertEqual(self.db.continuous, [])
        self.see(0.80, axis=3, sharpness=100.0)

        self.assertEqual(len(self.db.continuous), 1)
        person_id, embedding, quality = self.db.continuous[0]
        self.assertEqual(person_id, "uid-mario")
        self.assertAlmostEqual(quality, 90.0, places=3)
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
        for axis in (1, 2, 3):
            self.assertGreater(float(np.dot(embedding, face_vector(axis))), 0.9)

    def test_at_most_one_update_per_interval(self) -> None:
        self.see_batch(0.80)
        self.clock.advance(599.0)
        self.see_batch(0.80)
        self.assertEqual(len(self.db.continuous), 1)
        self.clock.advance(2.0)
        self.see_batch(0.80)
        self.assertEqual(len(self.db.continuous), 2)

    def test_uncertain_or_blurry_matches_are_not_learned(self) -> None:
        self.see_batch(0.50)                  # riconosciuto, ma sotto la soglia di apprendimento (0.55)
        self.see_batch(0.80, sharpness=10.0)  # sicuro, ma sfocato
        self.assertEqual(self.db.continuous, [])

    def test_failed_update_is_retried_with_the_next_batch(self) -> None:
        self.db.failing.add("add_continuous_embedding")
        self.see_batch(0.80)
        self.db.failing.clear()
        self.see_batch(0.80)
        self.assertEqual(len(self.db.continuous), 1)


if __name__ == "__main__":
    unittest.main()
