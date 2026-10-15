"""Saluto con la mano: riconoscimento del gesto, associazione al volto e saluto "Ciao"."""

from __future__ import annotations

import unittest

from app.models import ArmPose, BodyPose, Greeting, GreetingKind, Match
from app.recognition_service import RecognitionService
from app.wave_detector import MIN_SWING, WaveDetector, attach_arms, count_swings
from tests.helpers import FakeClock, FakeStore, make_detection, make_settings

FACE_WIDTH = 150.0
SHOULDER = (300.0, 400.0)
STEP_S = 0.1  # un risultato di inferenza ogni 100 ms
PERSON = "uid-mario"


def arm(offset: float, raised: bool = True, side: int = 1) -> ArmPose:
    """Braccio con il polso spostato di `offset` larghezze del volto rispetto alla spalla."""
    wrist_y = SHOULDER[1] - (FACE_WIDTH if raised else -FACE_WIDTH)
    return ArmPose(side, SHOULDER, (SHOULDER[0] + offset * FACE_WIDTH, wrist_y))


# Polso che va avanti e indietro di mezza larghezza del volto: tre movimenti alternati
WAVE = [0.0, 0.25, 0.5, 0.25, 0.0, 0.25, 0.5, 0.25, 0.0]


class CountSwingsTest(unittest.TestCase):
    def test_back_and_forth_movements_are_counted(self) -> None:
        self.assertEqual(count_swings([]), 0)
        self.assertEqual(count_swings([0.0, 0.5]), 1)
        self.assertEqual(count_swings([0.0, 0.5, 0.0]), 2)
        self.assertEqual(count_swings(WAVE), 4)

    def test_small_jitter_does_not_count(self) -> None:
        jitter = [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0]
        self.assertEqual(count_swings(jitter), 0)

    def test_steady_movement_is_one_swing(self) -> None:
        self.assertEqual(count_swings([0.0, 0.2, 0.4, 0.6, 0.8]), 1)

    def test_reversal_must_exceed_the_minimum_swing(self) -> None:
        self.assertEqual(count_swings([0.0, 0.5, 0.5 - MIN_SWING / 2, 0.5]), 1)


class AttachArmsTest(unittest.TestCase):
    def test_pose_goes_to_the_face_containing_its_nose(self) -> None:
        left, right = make_detection(), make_detection()
        left.bbox, right.bbox = (0, 0, 100, 100), (200, 0, 300, 100)
        right_arms, left_arms = (arm(0.0, side=0),), (arm(0.0, side=1),)
        attach_arms([left, right], [BodyPose((250.0, 50.0), right_arms), BodyPose((50.0, 50.0), left_arms)])
        self.assertEqual(left.arms, left_arms)
        self.assertEqual(right.arms, right_arms)

    def test_pose_outside_every_face_is_ignored(self) -> None:
        detection = make_detection()
        attach_arms([detection], [BodyPose((900.0, 900.0), (arm(0.0),))])
        self.assertEqual(detection.arms, ())

    def test_each_pose_goes_to_one_face_only(self) -> None:
        near, far = make_detection(), make_detection()
        near.bbox, far.bbox = (0, 0, 100, 100), (40, 40, 140, 140)
        arms = (arm(0.0),)
        attach_arms([near, far], [BodyPose((50.0, 50.0), arms)])
        self.assertEqual(near.arms, arms)
        self.assertEqual(far.arms, ())


class WaveDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.detector = WaveDetector(cooldown_seconds=30.0, clock=self.clock)

    def play(self, offsets: list[float], raised: bool = True, step: float = STEP_S) -> bool:
        """Mostra il braccio nelle posizioni indicate. True se in qualche momento il saluto e riconosciuto."""
        waved = False
        for offset in offsets:
            waved |= self.detector.observe(PERSON, [arm(offset, raised)], FACE_WIDTH)
            self.clock.advance(step)
        return waved

    def test_raised_hand_moving_back_and_forth_is_a_wave(self) -> None:
        self.assertTrue(self.play(WAVE))

    def test_lowered_hand_is_not_a_wave(self) -> None:
        self.assertFalse(self.play(WAVE, raised=False))

    def test_raised_still_hand_is_not_a_wave(self) -> None:
        self.assertFalse(self.play([0.3] * 20))

    def test_single_sweep_is_not_a_wave(self) -> None:
        self.assertFalse(self.play([0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_slow_movement_outside_the_window_is_not_a_wave(self) -> None:
        # Tre movimenti ci sono, ma non stanno nella finestra di WINDOW_SECONDS
        self.assertFalse(self.play(WAVE, step=0.55))

    def test_losing_the_wrist_for_too_long_restarts_the_gesture(self) -> None:
        self.assertFalse(self.play(WAVE, step=1.0))

    def test_lowering_the_hand_restarts_the_gesture(self) -> None:
        self.assertFalse(self.play([0.0, 0.5]))
        self.assertFalse(self.play([0.5], raised=False))
        self.assertFalse(self.play([0.0, 0.5]))
        self.assertTrue(self.play([0.0, 0.5, 0.0]))

    def test_no_new_wave_during_the_cooldown(self) -> None:
        self.assertTrue(self.play(WAVE))
        self.assertFalse(self.play(WAVE))
        self.clock.advance(30.0)
        self.assertTrue(self.play(WAVE))

    def test_people_are_tracked_separately(self) -> None:
        waved = False
        for offset in WAVE:
            self.assertFalse(self.detector.observe("uid-anna", [arm(0.0)], FACE_WIDTH))
            waved |= self.detector.observe(PERSON, [arm(offset)], FACE_WIDTH)
            self.clock.advance(STEP_S)
        self.assertTrue(waved)


class WaveGreetingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.db = FakeStore()
        self.db.nearest = [Match(PERSON, "Mario", 0.45, last_access_age_s=0.0)]  # appena salutato

    def run_wave(self, **settings: object) -> list[Greeting]:
        service = RecognitionService(make_settings(**settings), self.db, clock=self.clock)
        greetings: list[Greeting] = []
        for offset in WAVE:
            detection = make_detection()
            detection.arms = (arm(offset),)
            greetings.extend(service.process([detection]))
            self.clock.advance(STEP_S)
        return greetings

    def test_recognized_person_waving_gets_ciao(self) -> None:
        self.assertEqual(self.run_wave(wave_greeting=True), [Greeting(GreetingKind.WAVE, "Mario")])

    def test_wave_is_not_sent_to_the_server(self) -> None:
        self.run_wave(wave_greeting=True)
        self.assertEqual(self.db.greetings, [])

    def test_disabled_by_default(self) -> None:
        self.assertEqual(self.run_wave(), [])

    def test_unknown_face_waving_is_ignored(self) -> None:
        self.db.nearest = []
        self.assertEqual(self.run_wave(wave_greeting=True), [])

    def test_pending_person_waving_is_ignored(self) -> None:
        self.db.nearest = []
        self.db.pending = Match(PERSON, "Mario", 0.45, pending=True)
        self.assertEqual(self.run_wave(wave_greeting=True), [])


if __name__ == "__main__":
    unittest.main()
