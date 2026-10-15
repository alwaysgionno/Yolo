"""Saluto con la mano: riconoscimento del gesto, persone seguite tra i fotogrammi e saluto "Ciao"."""

from __future__ import annotations

import unittest

import numpy as np

from app.models import ArmPose, BBox, BodyPose, Greeting, GreetingKind, Match
from app.pose_estimator import parse_keypoints
from app.recognition_service import RecognitionService
from app.wave_detector import MIN_SWING, WaveDetector, count_swings, is_raised
from tests.helpers import FakeClock, FakeStore, make_detection, make_settings

SHOULDER_WIDTH = 200.0
STEP_S = 0.15  # un risultato di inferenza ogni 150 ms, circa 6-7 al secondo
MARIO = Match("uid-mario", "Mario", 0.45, last_access_age_s=0.0)  # appena salutato: niente "Bentornato"
ANNA = Match("uid-anna", "Anna", 0.45, last_access_age_s=0.0)
FACE: BBox = (100, 100, 250, 250)  # lo stesso riquadro di `make_detection`
NOSE = (175.0, 180.0)

# Polso che va avanti e indietro di un quarto della larghezza delle spalle
WAVE = [0.0, 0.25, 0.0, 0.25, 0.0, 0.25]


def arm(offset: float, raised: bool = True, side: int = 1, x: float = NOSE[0]) -> ArmPose:
    """Braccio con il polso spostato di `offset` larghezze delle spalle rispetto alla spalla."""
    shoulder = (x - SHOULDER_WIDTH / 2, NOSE[1] + 120.0)
    wrist_y = shoulder[1] - 100.0 if raised else shoulder[1] + 200.0
    return ArmPose(side, shoulder, (shoulder[0] + offset * SHOULDER_WIDTH, wrist_y))


def body(offset: float, raised: bool = True, nose: tuple[float, float] = NOSE) -> BodyPose:
    return BodyPose(nose, SHOULDER_WIDTH, (arm(offset, raised, x=nose[0]),))


class CountSwingsTest(unittest.TestCase):
    def test_back_and_forth_movements_are_counted(self) -> None:
        self.assertEqual(count_swings([]), 0)
        self.assertEqual(count_swings([0.0, 0.5]), 1)
        self.assertEqual(count_swings([0.0, 0.5, 0.0]), 2)
        self.assertEqual(count_swings([0.0, 0.25, 0.5, 0.25, 0.0, 0.25, 0.5]), 3)

    def test_small_jitter_does_not_count(self) -> None:
        self.assertEqual(count_swings([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0]), 0)

    def test_steady_movement_is_one_swing(self) -> None:
        self.assertEqual(count_swings([0.0, 0.2, 0.4, 0.6, 0.8]), 1)

    def test_reversal_must_exceed_the_minimum_swing(self) -> None:
        self.assertEqual(count_swings([0.0, 0.5, 0.5 - MIN_SWING / 2, 0.5]), 1)


class RaisedHandTest(unittest.TestCase):
    def test_wrist_above_the_shoulder_is_raised(self) -> None:
        self.assertTrue(is_raised(ArmPose(0, (100.0, 300.0), (120.0, 250.0)), SHOULDER_WIDTH))

    def test_forearm_pointing_up_is_raised_even_below_the_shoulder(self) -> None:
        # Mano all'altezza del petto, gomito piu in basso: si saluta cosi anche senza alzare il braccio
        self.assertTrue(is_raised(ArmPose(0, (100.0, 300.0), (120.0, 330.0), elbow=(110.0, 420.0)), SHOULDER_WIDTH))

    def test_bent_horizontal_forearm_is_not_raised(self) -> None:
        # Braccio piegato davanti al corpo, ad esempio per tenere il telefono
        self.assertFalse(is_raised(ArmPose(0, (100.0, 300.0), (160.0, 380.0), elbow=(120.0, 400.0)), SHOULDER_WIDTH))

    def test_hanging_arm_is_not_raised(self) -> None:
        self.assertFalse(is_raised(ArmPose(0, (100.0, 300.0), (105.0, 500.0), elbow=(102.0, 400.0)), SHOULDER_WIDTH))
        self.assertFalse(is_raised(ArmPose(0, (100.0, 300.0), (105.0, 500.0)), SHOULDER_WIDTH))


class ParseKeypointsTest(unittest.TestCase):
    @staticmethod
    def keypoints(confidence: float = 0.9) -> np.ndarray:
        points = np.zeros((17, 3), dtype=np.float32)
        points[:, 2] = confidence
        points[0, :2] = (200, 100)               # naso
        points[1:5, :2] = (200, 90)              # occhi e orecchie
        points[5, :2], points[6, :2] = (280, 200), (120, 200)   # spalle
        points[7, :2], points[8, :2] = (300, 280), (100, 280)   # gomiti
        points[9, :2], points[10, :2] = (310, 150), (90, 360)   # polsi
        return points

    def test_pose_with_both_arms(self) -> None:
        pose = parse_keypoints(self.keypoints())
        self.assertIsNotNone(pose)
        self.assertEqual(pose.nose, (200.0, 100.0))
        self.assertAlmostEqual(pose.shoulder_width, 160.0)
        self.assertEqual([arm.side for arm in pose.arms], [0, 1])
        self.assertEqual(pose.arms[0].elbow, (300.0, 280.0))

    def test_hidden_nose_is_replaced_by_eyes_and_ears(self) -> None:
        points = self.keypoints()
        points[0, 2] = 0.1  # la mano copre il naso
        self.assertEqual(parse_keypoints(points).nose, (200.0, 90.0))

    def test_points_outside_the_image_are_dropped(self) -> None:
        points = self.keypoints()
        points[9, :2] = (0, 0)  # polso fuori dall'immagine, con confidenza alta
        self.assertEqual([arm.side for arm in parse_keypoints(points).arms], [1])

    def test_unreliable_points_are_dropped(self) -> None:
        points = self.keypoints()
        points[9, 2] = 0.1  # polso sinistro incerto
        points[8, 2] = 0.1  # gomito destro incerto
        pose = parse_keypoints(points)
        self.assertEqual([arm.side for arm in pose.arms], [1])
        self.assertIsNone(pose.arms[0].elbow)

    def test_pose_without_shoulders_is_ignored(self) -> None:
        points = self.keypoints()
        points[5, 2] = 0.1
        self.assertIsNone(parse_keypoints(points))


class WaveDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.detector = WaveDetector(cooldown_seconds=30.0, clock=self.clock)

    def play(
        self,
        offsets: list[float],
        raised: bool = True,
        step: float = STEP_S,
        face_visible: bool = True,
    ) -> list[str]:
        """Mostra Mario che muove il braccio. Restituisce i nomi di chi ha salutato."""
        names: list[str] = []
        for offset in offsets:
            people = [(FACE, MARIO)] if face_visible else []
            names += [match.full_name for match in self.detector.update([body(offset, raised)], people)]
            self.clock.advance(step)
        return names

    def test_raised_hand_moving_back_and_forth_is_a_wave(self) -> None:
        self.assertEqual(self.play(WAVE), ["Mario"])

    def test_lowered_hand_is_not_a_wave(self) -> None:
        self.assertEqual(self.play(WAVE, raised=False), [])

    def test_raised_still_hand_is_not_a_wave(self) -> None:
        self.assertEqual(self.play([0.3] * 20), [])

    def test_single_sweep_is_not_a_wave(self) -> None:
        self.assertEqual(self.play([0.0, 0.25, 0.5, 0.75, 1.0]), [])

    def test_slow_movement_outside_the_window_is_not_a_wave(self) -> None:
        self.assertEqual(self.play([0.0, 0.25, 0.0], step=1.3), [])

    def test_lowering_the_hand_restarts_the_gesture(self) -> None:
        self.assertEqual(self.play([0.0, 0.25]), [])
        self.assertEqual(self.play([0.25], raised=False), [])
        self.assertEqual(self.play([0.0, 0.25]), [])
        self.assertEqual(self.play([0.0]), ["Mario"])

    def test_hand_covering_the_face_does_not_interrupt_the_gesture(self) -> None:
        # Il volto e riconosciuto all'inizio, poi la mano lo copre: la posa ricorda chi e
        self.assertEqual(self.play([0.0]), [])
        self.assertEqual(self.play([0.25, 0.0, 0.25], face_visible=False), ["Mario"])

    def test_pose_never_recognized_is_not_greeted(self) -> None:
        self.assertEqual(self.play(WAVE, face_visible=False), [])

    def test_name_is_forgotten_after_a_while(self) -> None:
        self.play([0.0])
        self.assertEqual(self.play([0.0] * 25 + WAVE, face_visible=False), [])

    def test_no_new_wave_during_the_cooldown(self) -> None:
        self.assertEqual(self.play(WAVE), ["Mario"])
        self.assertEqual(self.play(WAVE), [])
        self.clock.advance(30.0)
        self.assertEqual(self.play(WAVE), ["Mario"])

    def test_people_are_followed_separately(self) -> None:
        anna_nose = (600.0, 180.0)
        anna_face: BBox = (520, 100, 680, 250)
        names: list[str] = []
        for offset in WAVE:
            poses = [body(offset), body(0.0, nose=anna_nose)]
            names += [m.full_name for m in self.detector.update(poses, [(FACE, MARIO), (anna_face, ANNA)])]
            self.clock.advance(STEP_S)
        self.assertEqual(names, ["Mario"])


class WaveGreetingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.db = FakeStore()
        self.db.nearest = [MARIO]

    def run_wave(self, **settings: object) -> list[Greeting]:
        service = RecognitionService(make_settings(**settings), self.db, clock=self.clock)
        greetings: list[Greeting] = []
        for offset in WAVE:
            greetings.extend(service.process([make_detection()], [body(offset)]))
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
        self.db.pending = Match("uid-mario", "Mario", 0.45, pending=True)
        self.assertEqual(self.run_wave(wave_greeting=True), [])


if __name__ == "__main__":
    unittest.main()
