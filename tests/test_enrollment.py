"""Test della macchina a stati di registrazione (EnrollmentSession)."""

from __future__ import annotations

import unittest

from app.enrollment import ENTER_KEYS, ESC_KEY, EnrollmentSession, EnrollmentState
from app.models import Match
from tests.helpers import FakeClock, FakeStore, face_vector, make_detection, make_settings, other_person_vector

ENTER = ENTER_KEYS[1]
BACKSPACE = 8


class EnrollmentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.db = FakeStore()
        self.session = EnrollmentSession(make_settings(), self.db, clock=self.clock)

    def type_name(self, name: str) -> None:
        for char in name:
            self.session.handle_key(ord(char))
        self.session.handle_key(ENTER)

    def begin_capture(self, name: str = "Mario Rossi") -> None:
        self.session.start(make_detection())
        self.type_name(name)
        self.assertIs(self.session.state, EnrollmentState.CAPTURING)


class StartTest(EnrollmentTestCase):
    def observe(self, detections: list, seconds: float = 0.6) -> bool:
        """Simula un nuovo risultato di inferenza, poi fa passare del tempo."""
        started = self.session.consider_start(detections)
        self.clock.advance(seconds)
        return started

    def test_unknown_face_must_stay_in_view_before_the_prompt(self) -> None:
        unknown = [make_detection()]
        self.assertFalse(self.observe(unknown))  # 0.0 s
        self.assertFalse(self.observe(unknown))  # 0.6 s
        self.assertFalse(self.observe(unknown))  # 1.2 s: tre risultati, ma meno di 1.5 secondi
        self.assertTrue(self.observe(unknown))   # 1.8 s
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)

    def test_time_alone_is_not_enough(self) -> None:
        unknown = [make_detection()]
        self.assertFalse(self.observe(unknown, seconds=5.0))
        self.assertFalse(self.observe(unknown, seconds=5.0))  # solo due risultati
        self.assertTrue(self.observe(unknown))

    def test_interruption_restarts_the_confirmation(self) -> None:
        unknown = [make_detection()]
        known = [make_detection(match=Match("uid-mario", "Anna", 0.7))]
        someone_else = [make_detection(other_person_vector())]
        interruptions = {"volto riconosciuto": known, "nessun volto": [], "altra persona": someone_else}
        for label, interruption in interruptions.items():
            with self.subTest(label):
                self.session.reset()
                self.observe(unknown)
                self.observe(unknown)
                self.assertFalse(self.observe(interruption))
                self.assertFalse(self.observe(unknown))
                self.assertFalse(self.observe(unknown))
                self.assertFalse(self.observe(unknown))
                self.assertTrue(self.observe(unknown))

    def test_distant_face_is_never_proposed(self) -> None:
        far = [make_detection(face_width=40.0)]
        for _ in range(10):
            self.assertFalse(self.observe(far))
        self.assertIs(self.session.state, EnrollmentState.IDLE)

    def test_known_person_in_view_does_not_block_the_proposal(self) -> None:
        scene = [make_detection(other_person_vector(), match=Match("uid-mario", "Anna", 0.7)), make_detection()]
        results = [self.observe(scene) for _ in range(4)]
        self.assertEqual(results, [False, False, False, True])

    def test_closest_unknown_face_is_selected_and_followed(self) -> None:
        far = make_detection(other_person_vector(), face_width=100.0)
        near = make_detection(face_vector(1), face_width=220.0)
        for _ in range(4):
            self.observe([far, near])
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)
        self.assertIs(self.session.candidate_in([far, near]), near)
        self.assertIsNone(self.session.candidate_in([far]))

    def test_cancel_snoozes_new_proposals(self) -> None:
        unknown = [make_detection()]
        self.session.start(make_detection())
        self.session.handle_key(ESC_KEY)
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertEqual(self.session.error, "Registrazione annullata")
        self.session.reset()

        for _ in range(20):  # 20 x 2.9 s = 58 s: ancora dentro i 60 secondi di snooze
            self.assertFalse(self.observe(unknown, seconds=2.9))
        self.clock.advance(3.0)
        results = [self.observe(unknown) for _ in range(4)]
        self.assertEqual(results, [False, False, False, True])


class NameInputTest(EnrollmentTestCase):
    def test_short_name_is_refused(self) -> None:
        self.session.start(make_detection())
        self.type_name("M")
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)
        self.assertIn("almeno 2 caratteri", self.session.error)

    def test_duplicate_name_is_refused_ignoring_case(self) -> None:
        self.db.existing_names.add("mario rossi")
        self.session.start(make_detection())
        self.type_name("Mario ROSSI")
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)
        self.assertIn("già registrato", self.session.error)

    def test_name_is_cleaned_and_backspace_works(self) -> None:
        self.session.start(make_detection())
        for char in "  Mario   Rossix":
            self.session.handle_key(ord(char))
        self.session.handle_key(BACKSPACE)
        self.session.handle_key(ENTER)
        self.assertIs(self.session.state, EnrollmentState.CAPTURING)
        self.assertEqual(self.session.name, "Mario Rossi")

    def test_name_timeout_cancels(self) -> None:
        self.session.start(make_detection())
        self.session.handle_key(ord("M"))  # con un nome in corso la richiesta resta anche se il volto esce
        self.clock.advance(59.0)
        self.session.tick()
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)
        self.clock.advance(2.0)
        self.session.tick()
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertIn("Tempo scaduto", self.session.error)

    def test_missing_gallery_cancels(self) -> None:
        self.db.failing.add("person_exists")
        self.session.start(make_detection())
        self.type_name("Mario")
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertEqual(self.session.error, "Elenco delle persone non disponibile")


class FollowedFaceTest(EnrollmentTestCase):
    """La richiesta del nome resta legata al volto che l'ha fatta comparire."""

    KNOWN = Match("uid-mario", "Anna", 0.7)

    def test_prompt_is_withdrawn_when_the_face_leaves_before_typing(self) -> None:
        self.session.start(make_detection())
        self.clock.advance(9.0)
        self.session.tick()
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)
        self.clock.advance(2.0)
        self.session.tick()
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertIn("volto non più presente", self.session.error)

        # Nessuno snooze: un volto nuovo puo essere proposto subito
        self.session.reset()
        results = []
        for _ in range(4):
            results.append(self.session.consider_start([make_detection()]))
            self.clock.advance(0.6)
        self.assertEqual(results, [False, False, False, True])

    def test_prompt_stays_while_the_face_is_in_view(self) -> None:
        self.session.start(make_detection())
        for _ in range(25):  # 50 secondi senza digitare nulla, ma con il volto sempre presente
            self.clock.advance(2.0)
            self.session.offer_frame([make_detection(face_vector(2))])
            self.session.tick()
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)

    def test_face_recognised_as_registered_withdraws_the_prompt(self) -> None:
        self.session.start(make_detection())
        self.session.offer_frame([make_detection(match=self.KNOWN)])
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)  # un solo riconoscimento non basta
        self.session.offer_frame([make_detection(match=self.KNOWN)])
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertEqual(self.session.error, "Volto già registrato come Anna")

        self.session.reset()
        self.assertFalse(any(self.session.consider_start([make_detection()]) for _ in range(5)))  # snooze

    def test_isolated_recognition_does_not_cancel(self) -> None:
        self.session.start(make_detection())
        self.session.offer_frame([make_detection(match=self.KNOWN)])
        self.session.offer_frame([make_detection()])
        self.session.offer_frame([make_detection(match=self.KNOWN)])
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)

    def test_recognition_of_another_face_is_irrelevant(self) -> None:
        self.session.start(make_detection())
        for _ in range(5):
            self.session.offer_frame([make_detection(other_person_vector(), match=self.KNOWN), make_detection()])
        self.assertIs(self.session.state, EnrollmentState.NAME_INPUT)


class CaptureTest(EnrollmentTestCase):
    def offer_valid_samples(self, count: int) -> None:
        for axis in range(1, count + 1):
            self.session.offer_frame([make_detection(face_vector(axis), sharpness=100.0 + axis)])
            self.clock.advance(0.5)

    def test_valid_samples_are_saved_in_one_profile(self) -> None:
        self.begin_capture()
        self.offer_valid_samples(4)
        self.assertEqual(self.session.progress, (4, 5))
        self.assertIs(self.session.state, EnrollmentState.CAPTURING)

        self.offer_valid_samples(5)  # i primi quattro sono doppioni scartati, il quinto completa il profilo
        self.assertIs(self.session.state, EnrollmentState.COMPLETED)
        self.assertEqual(self.session.person_id, "uid-1")
        self.assertEqual(len(self.db.created), 1)
        name, embeddings, qualities = self.db.created[0]
        self.assertEqual(name, "Mario Rossi")
        self.assertEqual(len(embeddings), 5)
        self.assertEqual(len(qualities), 5)

    def test_unusable_frames_are_refused_with_a_hint(self) -> None:
        self.begin_capture()
        cases = [
            ([], "Nessun volto"),
            ([make_detection(other_person_vector())], "Volto diverso"),
            ([make_detection(match=Match("uid-anna", "Anna", 0.8))], "già registrato come Anna"),
            ([make_detection(face_width=50.0)], "Avvicinati"),
            ([make_detection(sharpness=10.0)], "sfocata (10/60)"),
        ]
        for detections, expected in cases:
            with self.subTest(expected):
                self.session.offer_frame(detections)
                self.assertIn(expected, self.session.feedback)
                self.assertEqual(self.session.progress[0], 0)

    def test_other_person_and_duplicate_samples_are_refused(self) -> None:
        self.begin_capture()
        self.session.offer_frame([make_detection(face_vector(1))])
        self.clock.advance(0.5)

        self.session.offer_frame([make_detection(other_person_vector())])
        self.assertIn("Volto diverso", self.session.feedback)
        self.session.offer_frame([make_detection(face_vector(1))])
        self.assertIn("Muovi", self.session.feedback)
        self.assertEqual(self.session.progress[0], 1)

    def test_other_faces_in_view_are_never_sampled(self) -> None:
        self.begin_capture()
        bystander = make_detection(other_person_vector(), face_width=300.0)
        for axis in range(1, 6):
            self.session.offer_frame([bystander, make_detection(face_vector(axis))])
            self.clock.advance(0.5)
        self.assertIs(self.session.state, EnrollmentState.COMPLETED)
        _, embeddings, _ = self.db.created[0]
        for embedding in embeddings:
            self.assertLess(float(embedding @ other_person_vector()), 0.1)

    def test_face_recognised_during_capture_cancels_without_saving(self) -> None:
        self.begin_capture()
        self.offer_valid_samples(2)
        known = [make_detection(match=Match("uid-anna", "Anna", 0.8))]
        self.session.offer_frame(known)
        self.assertIs(self.session.state, EnrollmentState.CAPTURING)
        self.assertEqual(self.session.progress[0], 2)  # il fotogramma "riconosciuto" non diventa un campione
        self.session.offer_frame(known)
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertEqual(self.db.created, [])

    def test_samples_too_close_in_time_are_ignored(self) -> None:
        self.begin_capture()
        self.session.offer_frame([make_detection(face_vector(1))])
        self.clock.advance(0.1)
        self.session.offer_frame([make_detection(face_vector(2))])
        self.assertEqual(self.session.progress[0], 1)

    def test_capture_timeout_cancels(self) -> None:
        self.begin_capture()
        self.offer_valid_samples(2)
        self.clock.advance(30.0)
        self.session.tick()
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertEqual(self.db.created, [])

    def test_save_failure_cancels(self) -> None:
        self.db.failing.add("create_person")
        self.begin_capture()
        self.offer_valid_samples(5)
        self.assertIs(self.session.state, EnrollmentState.CANCELLED)
        self.assertIn("Invio del profilo", self.session.error)
        self.assertIsNone(self.session.person_id)


if __name__ == "__main__":
    unittest.main()
