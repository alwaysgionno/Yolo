"""Test di RecognitionService: soglia, saluti con debouncing e apprendimento continuo."""

from __future__ import annotations

import unittest

import numpy as np

from app.database_manager import DatabaseError
from app.models import Match
from app.recognition_service import RecognitionService
from tests.helpers import FakeClock, FakeDatabase, face_vector, make_detection, make_settings


class RecognitionServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.db = FakeDatabase()
        self.service = RecognitionService(make_settings(), self.db, clock=self.clock)

    def see(self, similarity: float, *, axis: int = 1, sharpness: float = 120.0) -> list[str]:
        """Simula un fotogramma in cui il vicino piu simile e Mario con la similarita indicata."""
        self.db.nearest = [Match(1, "Mario", similarity), Match(2, "Anna", 0.10)]
        return self.service.process([make_detection(face_vector(axis), sharpness=sharpness)])


class ThresholdTest(RecognitionServiceTestCase):
    def test_face_below_threshold_stays_unknown(self) -> None:
        self.db.nearest = [Match(1, "Mario", 0.39)]
        detection = make_detection()
        self.assertEqual(self.service.process([detection]), [])
        self.assertIsNone(detection.match)
        self.assertEqual(self.db.greetings, [])

    def test_empty_database_means_unknown(self) -> None:
        detection = make_detection()
        self.assertEqual(self.service.process([detection]), [])
        self.assertIsNone(detection.match)

    def test_face_at_threshold_is_recognized(self) -> None:
        self.db.nearest = [Match(1, "Mario", 0.40)]
        detection = make_detection()
        self.assertEqual(self.service.process([detection]), ["Mario"])
        self.assertEqual(detection.match, Match(1, "Mario", 0.40))

    def test_database_failure_is_reported_to_the_caller(self) -> None:
        self.db.failing.add("find_nearest")
        with self.assertRaises(DatabaseError):
            self.service.process([make_detection()])


class GreetingTest(RecognitionServiceTestCase):
    def test_one_greeting_per_cooldown(self) -> None:
        self.assertEqual(self.see(0.45), ["Mario"])
        self.clock.advance(299.0)
        self.assertEqual(self.see(0.45), [])
        self.clock.advance(2.0)
        self.assertEqual(self.see(0.47), ["Mario"])
        self.assertEqual(self.db.greetings, [(1, 0.45), (1, 0.47)])

    def test_greeting_registered_by_enrollment_suppresses_the_first_one(self) -> None:
        self.service.register_greeting(1)
        self.assertEqual(self.see(0.60), [])
        self.clock.advance(301.0)
        self.assertEqual(self.see(0.60), ["Mario"])

    def test_greeting_is_shown_even_if_the_log_cannot_be_written(self) -> None:
        self.db.failing.add("record_greeting")
        self.assertEqual(self.see(0.45), ["Mario"])
        self.assertEqual(self.db.greetings, [])


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
        person_id, embedding, quality, max_stored = self.db.continuous[0]
        self.assertEqual(person_id, 1)
        self.assertEqual(max_stored, 20)
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
