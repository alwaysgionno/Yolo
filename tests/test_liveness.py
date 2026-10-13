"""Test dell'anti-spoofing: misura del parallasse, accumulo delle prove, effetti su saluti e registrazione."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.enrollment import ENTER_KEYS, EnrollmentSession, EnrollmentState
from app.liveness import (
    CONFIRMATIONS,
    HOLD_SECONDS,
    PATCH_SIZE,
    REFERENCE_MAX_AGE_S,
    LivenessCheck,
    LivenessTracker,
    align_face,
    parallax,
)
from app.models import Greeting, GreetingKind, Match
from app.recognition_service import RecognitionService
from tests.helpers import FakeClock, FakeStore, face_vector, make_detection, make_settings, other_person_vector

MIN_PARALLAX = 0.07
STEP_S = 0.3  # tempo tra due risultati di inferenza: piu dell'intervallo minimo tra due confronti


def pose_patch(pose: float) -> np.ndarray:
    """Volto simulato: l'unico valore che conta e la "posa", letta da `pose_distance`."""
    return np.full((2, 2), pose, dtype=np.float32)


def pose_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """Sostituisce `parallax`: il parallasse cresce con la differenza di posa (0.01 per unita)."""
    return abs(float(current[0, 0]) - float(reference[0, 0])) / 100.0


def textured_face(seed: int = 0) -> np.ndarray:
    """Immagine con dettaglio ovunque, al posto di un volto allineato."""
    noise = np.random.default_rng(seed).normal(0.0, 1.0, (PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.5)
    return cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def tilted(image: np.ndarray, shrink: float, shift: float) -> np.ndarray:
    """La stessa immagine vista come una foto inclinata: una trasformazione prospettica."""
    size = float(PATCH_SIZE)
    corners = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    moved = np.float32([
        [size * shrink + shift, size * 0.04], [size - shift, 0], [size - shift, size], [size * shrink + shift, size * 0.96]
    ])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(corners, moved), (PATCH_SIZE, PATCH_SIZE),
                               borderMode=cv2.BORDER_REFLECT)


def turned(image: np.ndarray, nose_shift_px: float) -> np.ndarray:
    """La stessa immagine con il centro spostato rispetto ai bordi: come un naso che sporge quando la testa gira."""
    grid_y, grid_x = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE].astype(np.float32)
    bump = np.exp(-((grid_x - PATCH_SIZE / 2) ** 2 + (grid_y - PATCH_SIZE * 0.6) ** 2) / (2 * 22.0 ** 2))
    return cv2.remap(image, grid_x - nose_shift_px * bump, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class ParallaxTest(unittest.TestCase):
    def test_same_view_has_no_parallax(self) -> None:
        face = textured_face()
        noisy = np.clip(face.astype(np.float32) + np.random.default_rng(1).normal(0, 4, face.shape), 0, 255)
        score = parallax(face, noisy.astype(np.uint8))
        self.assertIsNotNone(score)
        self.assertLess(score, 0.03)

    def test_tilted_photo_stays_flat(self) -> None:
        face = textured_face()
        for shrink, shift in ((0.10, 0.0), (0.20, 4.0), (0.28, -5.0)):
            with self.subTest(shrink=shrink):
                score = parallax(face, tilted(face, shrink, shift))
                self.assertIsNotNone(score)
                self.assertLess(score, 0.03)

    def test_turning_head_shows_parallax(self) -> None:
        face = textured_face()
        score = parallax(face, turned(face, 12.0))
        self.assertIsNotNone(score)
        self.assertGreater(score, MIN_PARALLAX)

    def test_turning_head_is_not_hidden_by_a_tilt(self) -> None:
        face = textured_face()
        score = parallax(face, tilted(turned(face, 12.0), 0.12, 3.0))
        self.assertIsNotNone(score)
        self.assertGreater(score, MIN_PARALLAX)

    def test_face_without_detail_cannot_be_measured(self) -> None:
        flat = np.full((PATCH_SIZE, PATCH_SIZE), 128, dtype=np.uint8)
        self.assertIsNone(parallax(flat, flat))

    def test_unrelated_images_cannot_be_measured(self) -> None:
        self.assertIsNone(parallax(textured_face(0), textured_face(5)))

    def test_alignment_puts_the_face_in_a_fixed_position(self) -> None:
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        landmarks = np.array([[150, 120], [210, 120], [180, 155], [158, 190], [202, 190]], dtype=np.float32)
        for x, y in landmarks:
            cv2.circle(image, (int(x), int(y)), 4, (255, 255, 255), cv2.FILLED)
        patch = align_face(image, landmarks)
        self.assertEqual(patch.shape, (PATCH_SIZE, PATCH_SIZE))
        # Lo stesso volto, piu grande e ruotato nell'immagine, produce quasi lo stesso ritaglio
        rotation = cv2.getRotationMatrix2D((180, 150), 20, 1.3)
        other = align_face(cv2.warpAffine(image, rotation, (400, 300)), cv2.transform(landmarks[None], rotation)[0])
        bright, other_bright = patch > 100, other > 100
        self.assertGreater((bright & other_bright).sum(), 0.6 * bright.sum())


class LivenessCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.check = LivenessCheck(MIN_PARALLAX, self.clock, pose_distance)

    def show(self, pose: float) -> bool:
        verified = self.check.observe(pose_patch(pose))
        self.clock.advance(STEP_S)
        return verified

    def test_still_face_is_never_verified(self) -> None:
        for _ in range(60):
            self.assertFalse(self.show(0.0))

    def test_small_movements_are_not_enough(self) -> None:
        for pose in (0.0, 3.0, -4.0, 5.0, 0.0, 6.0):
            self.assertFalse(self.show(pose))

    def test_turning_must_be_confirmed(self) -> None:
        self.assertEqual(CONFIRMATIONS, 2)
        self.assertFalse(self.show(0.0))   # riferimento
        self.assertFalse(self.show(10.0))  # prima prova
        self.assertTrue(self.show(12.0))   # seconda prova
        self.assertTrue(self.show(0.0))    # una volta verificato lo resta

    def test_isolated_evidence_expires_with_the_reference(self) -> None:
        self.show(0.0)
        self.assertFalse(self.show(10.0))  # un solo fotogramma anomalo
        self.clock.advance(REFERENCE_MAX_AGE_S)
        self.assertFalse(self.show(0.0))   # nessuna conferma: nuovo riferimento, si riparte da zero
        self.assertFalse(self.show(10.0))
        self.assertTrue(self.show(10.0))

    def test_checks_are_rate_limited(self) -> None:
        calls: list[None] = []

        def counting(reference: np.ndarray, current: np.ndarray) -> float:
            calls.append(None)
            return 0.0

        check = LivenessCheck(MIN_PARALLAX, self.clock, counting)
        for _ in range(10):  # dieci fotogrammi in un decimo di secondo
            check.observe(pose_patch(0.0))
            self.clock.advance(0.01)
        self.assertEqual(len(calls), 1)

    def test_unmeasurable_comparisons_prove_nothing(self) -> None:
        check = LivenessCheck(MIN_PARALLAX, self.clock, lambda reference, current: None)
        for _ in range(10):
            self.assertFalse(check.observe(pose_patch(50.0)))
            self.clock.advance(STEP_S)


class LivenessTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tracker = LivenessTracker(MIN_PARALLAX, 90.0, 60.0, self.clock, pose_distance)

    def show(self, pose: float, person_id: str | None = "uid-mario", **fields: object) -> bool:
        fields.setdefault("patch", pose_patch(pose))
        detection = make_detection(**fields)  # type: ignore[arg-type]
        live = self.tracker.observe(detection, person_id)
        self.assertIs(detection.live, live)
        self.clock.advance(STEP_S)
        return live

    def turn_head(self, person_id: str | None = "uid-mario", **fields: object) -> bool:
        return [self.show(pose, person_id, **fields) for pose in (0.0, 10.0, 12.0)][-1]

    def test_person_is_verified_after_turning_the_head(self) -> None:
        self.assertTrue(self.turn_head())
        self.assertTrue(self.show(0.0))

    def test_each_person_is_verified_separately(self) -> None:
        self.assertTrue(self.turn_head("uid-mario"))
        self.assertFalse(self.show(0.0, "uid-anna"))

    def test_poor_frames_are_not_used(self) -> None:
        self.assertFalse(self.turn_head(sharpness=10.0))
        self.assertFalse(self.turn_head(face_width=50.0))
        self.assertFalse(self.turn_head(patch=None))

    def test_waiting_time_is_reported_until_verified(self) -> None:
        detection = make_detection(patch=pose_patch(0.0))
        self.tracker.observe(detection, "uid-mario")
        self.clock.advance(3.0)
        detection = make_detection(patch=pose_patch(0.0))
        self.tracker.observe(detection, "uid-mario")
        self.assertAlmostEqual(detection.liveness_wait_s, 3.0)

    def test_verification_is_kept_during_short_absences_only(self) -> None:
        self.turn_head()
        self.clock.advance(HOLD_SECONDS - 1.0)
        self.assertTrue(self.show(0.0))
        self.clock.advance(HOLD_SECONDS + 1.0)
        self.assertFalse(self.show(0.0))

    def test_unknown_faces_are_followed_by_their_embedding(self) -> None:
        self.assertTrue(self.turn_head(None, embedding=face_vector(1)))
        self.assertTrue(self.show(0.0, None, embedding=face_vector(2)))        # stessa persona, altra posa
        self.assertFalse(self.show(0.0, None, embedding=other_person_vector()))  # un'altra persona


class GreetingWithLivenessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.db = FakeStore()
        settings = make_settings(liveness_check=True)
        tracker = LivenessTracker(MIN_PARALLAX, 90.0, 60.0, self.clock, pose_distance)
        self.service = RecognitionService(settings, self.db, clock=self.clock, liveness=tracker)

    def see(self, pose: float, axis: int = 1) -> list[Greeting]:
        self.db.nearest = [Match("uid-mario", "Mario", 0.80)]
        self.detection = make_detection(face_vector(axis), patch=pose_patch(pose))
        greetings = self.service.process([self.detection])
        self.clock.advance(STEP_S)
        return greetings

    def test_photo_is_recognised_but_never_greeted(self) -> None:
        for _ in range(40):
            self.assertEqual(self.see(0.0), [])
        self.assertIsNotNone(self.detection.match)
        self.assertIs(self.detection.live, False)
        self.assertEqual(self.db.greetings, [])

    def test_photo_is_never_learned(self) -> None:
        for axis in (1, 2, 3, 1, 2, 3):
            self.see(0.0, axis=axis)
        self.assertEqual(self.db.continuous, [])

    def test_greeting_arrives_when_the_face_is_verified(self) -> None:
        self.assertEqual(self.see(0.0), [])
        self.assertEqual(self.see(10.0), [])
        self.assertEqual(self.see(12.0), [Greeting(GreetingKind.WELCOME, "Mario")])
        self.assertIs(self.detection.live, True)
        self.assertEqual(len(self.db.greetings), 1)
        self.assertEqual(self.see(0.0), [])  # gia salutato

    def test_liveness_is_built_from_the_settings(self) -> None:
        service = RecognitionService(make_settings(liveness_check=True), self.db, clock=self.clock)
        self.db.nearest = [Match("uid-mario", "Mario", 0.80)]
        detection = make_detection()  # nessuna immagine del volto: non verificabile
        self.assertEqual(service.process([detection]), [])
        self.assertIs(detection.live, False)

    def test_without_the_check_faces_are_greeted_at_once(self) -> None:
        service = RecognitionService(make_settings(liveness_check=False), self.db, clock=self.clock)
        self.db.nearest = [Match("uid-mario", "Mario", 0.80)]
        detection = make_detection()
        self.assertEqual(service.process([detection]), [Greeting(GreetingKind.WELCOME, "Mario")])
        self.assertIsNone(detection.live)


class EnrollmentWithLivenessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.db = FakeStore()
        self.session = EnrollmentSession(make_settings(liveness_check=True), self.db, clock=self.clock)
        self.session.start(make_detection(live=False))
        for char in "Mario Rossi":
            self.session.handle_key(ord(char))
        self.session.handle_key(ENTER_KEYS[1])

    def offer(self, count: int, live: bool) -> None:
        for axis in range(1, count + 1):
            self.session.offer_frame([make_detection(face_vector(axis), live=live)])
            self.clock.advance(0.5)

    def test_photo_is_not_saved_even_with_all_the_samples(self) -> None:
        self.offer(5, live=False)
        self.assertEqual(self.session.progress, (5, 5))
        self.assertIs(self.session.state, EnrollmentState.CAPTURING)
        self.assertIn("gira lentamente la testa", self.session.feedback)
        self.assertEqual(self.db.created, [])

        self.clock.advance(30.0)
        self.session.tick()
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertIn("Volto non verificato", self.session.error)
        self.assertEqual(self.db.created, [])

    def test_profile_is_saved_as_soon_as_the_face_is_verified(self) -> None:
        self.offer(5, live=False)
        self.session.offer_frame([make_detection(face_vector(1), live=True)])
        self.assertIs(self.session.state, EnrollmentState.COMPLETED)
        _, embeddings, _ = self.db.created[0]
        self.assertEqual(len(embeddings), 5)  # il fotogramma della verifica non aggiunge un campione

    def test_face_verified_while_typing_the_name_needs_nothing_more(self) -> None:
        session = EnrollmentSession(make_settings(liveness_check=True), self.db, clock=self.clock)
        session.start(make_detection(live=False))
        session.offer_frame([make_detection(live=True)])  # verificato durante l'inserimento del nome
        for char in "Anna":
            session.handle_key(ord(char))
        session.handle_key(ENTER_KEYS[1])
        for axis in range(1, 6):
            session.offer_frame([make_detection(face_vector(axis), live=False)])
            self.clock.advance(0.5)
        self.assertIs(session.state, EnrollmentState.COMPLETED)

    def test_verification_does_not_carry_over_to_the_next_enrollment(self) -> None:
        self.session.offer_frame([make_detection(face_vector(1), live=True)])
        self.session.reset()
        self.session.start(make_detection(live=False))
        for char in "Anna":
            self.session.handle_key(ord(char))
        self.session.handle_key(ENTER_KEYS[1])
        self.offer(5, live=False)
        self.assertIs(self.session.state, EnrollmentState.CAPTURING)


if __name__ == "__main__":
    unittest.main()
