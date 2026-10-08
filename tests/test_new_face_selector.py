"""Test di NewFaceSelector: scelta e conferma del volto sconosciuto da registrare."""

from __future__ import annotations

import unittest

import numpy as np

from app.models import FaceDetection, Match
from app.new_face_selector import UNKNOWN_CONFIRM_RESULTS, NewFaceSelector, find_same_face
from tests.helpers import (
    EMBEDDING_SIZE,
    FakeClock,
    face_vector,
    make_detection,
    make_settings,
    other_person_vector,
)

KNOWN = Match("uid-mario", "Anna", 0.7)


class NewFaceSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.selector = NewFaceSelector(make_settings(), clock=self.clock)

    def observe(self, *faces: FaceDetection, seconds: float = 0.6) -> FaceDetection | None:
        """Simula un nuovo risultato di inferenza, poi fa passare del tempo."""
        selected = self.selector.update(list(faces))
        self.clock.advance(seconds)
        return selected

    def test_same_unknown_face_is_confirmed_after_results_and_time(self) -> None:
        views = [make_detection(face_vector(axis)) for axis in (1, 2, 3, 4)]  # stessa persona, pose diverse
        results = [self.observe(view) for view in views]
        self.assertEqual(results[:3], [None, None, None])
        self.assertIs(results[3], views[3])

    def test_fast_inference_still_waits_the_minimum_time(self) -> None:
        for _ in range(UNKNOWN_CONFIRM_RESULTS * 3):  # 9 risultati in 0.45 secondi
            self.assertIsNone(self.observe(make_detection(), seconds=0.05))

    def test_known_face_seen_as_unknown_now_and_then_is_never_proposed(self) -> None:
        for _ in range(10):
            self.assertIsNone(self.observe(make_detection()))
            self.assertIsNone(self.observe(make_detection()))
            self.assertIsNone(self.observe(make_detection(match=KNOWN)))  # riconosciuta: si riparte

    def test_change_of_person_restarts_the_confirmation(self) -> None:
        self.observe(make_detection())
        self.observe(make_detection())
        self.assertIsNone(self.observe(make_detection(other_person_vector())))
        self.assertIsNone(self.observe(make_detection(other_person_vector())))

    def test_face_moving_away_restarts_the_confirmation(self) -> None:
        self.observe(make_detection())
        self.observe(make_detection())
        self.assertIsNone(self.observe(make_detection(face_width=40.0)))
        self.assertIsNone(self.observe(make_detection()))

    def test_closest_unknown_face_wins_and_known_faces_are_ignored(self) -> None:
        known = make_detection(face_vector(1), match=KNOWN, face_width=400.0)
        far = make_detection(other_person_vector(), face_width=100.0)
        third_person = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
        third_person[200] = 1.0
        near = make_detection(third_person, face_width=250.0)

        selected = None
        for _ in range(4):
            selected = self.observe(known, far, near)
        self.assertIs(selected, near)

    def test_observed_face_is_kept_even_if_another_comes_closer(self) -> None:
        first = make_detection(face_vector(1), face_width=120.0)
        self.observe(first)
        newcomer = make_detection(other_person_vector(), face_width=300.0)
        selected = None
        for _ in range(3):
            selected = self.observe(newcomer, first)
        self.assertIs(selected, first)

    def test_confirmation_starts_over_after_a_selection(self) -> None:
        results = [self.observe(make_detection()) for _ in range(5)]
        self.assertIsNotNone(results[3])
        self.assertIsNone(results[4])


class FindSameFaceTest(unittest.TestCase):
    def test_most_similar_face_above_the_identity_threshold(self) -> None:
        other = make_detection(other_person_vector())
        similar = make_detection(face_vector(2))
        identical = make_detection(face_vector(1))
        self.assertIs(find_same_face([other, similar, identical], face_vector(1)), identical)
        self.assertIs(find_same_face([other, similar], face_vector(1)), similar)
        self.assertIsNone(find_same_face([other], face_vector(1)))
        self.assertIsNone(find_same_face([], face_vector(1)))


if __name__ == "__main__":
    unittest.main()
