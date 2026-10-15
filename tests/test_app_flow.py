"""Test del coordinamento tra risultati di inferenza e registrazione (FaceAccessApp).

L'applicazione viene costruita senza `__init__`: niente modelli AI, webcam, finestre o server.
"""

from __future__ import annotations

import unittest

from app.enrollment import ENTER_KEYS, EnrollmentSession, EnrollmentState
from app.main import FaceAccessApp, _decision_toast, _greeting_texts
from app.models import EnrollmentDecision, FrameResult, Greeting, GreetingKind, Match
from tests.helpers import FakeClock, FakeStore, face_vector, make_detection, make_settings, other_person_vector


class FakeUI:
    def __init__(self) -> None:
        self.toasts: list[str] = []

    def show_toast(self, text: str, seconds: float | None = None, kind: object = None) -> None:
        self.toasts.append(text)


class FakeCamera:
    def __init__(self) -> None:
        self.frame_id = 0

    def latest(self) -> tuple[int, None]:
        return self.frame_id, None


class EnrollmentFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.db = FakeStore()
        settings = make_settings()
        self.app = FaceAccessApp.__new__(FaceAccessApp)
        self.app._settings = settings
        self.app._enrollment = EnrollmentSession(settings, self.db, clock=self.clock)
        self.app._ui = FakeUI()
        self.app._camera = FakeCamera()
        self.app._last_enrollment_frame_id = -1
        self.frame_id = 0

    @property
    def state(self) -> EnrollmentState:
        return self.app._enrollment.state

    def unknown_result(self, axis: int = 1, *, gallery_ok: bool = True) -> FrameResult:
        """Nuovo risultato di inferenza con un solo volto sconosciuto; la webcam e gia due frame avanti."""
        self.frame_id += 1
        self.app._camera.frame_id = self.frame_id + 2
        return FrameResult(frame_id=self.frame_id, detections=[make_detection(face_vector(axis))], gallery_ok=gallery_ok)

    def feed(self, result: FrameResult | None, seconds: float = 0.6) -> None:
        self.app._update_enrollment(result)
        self.clock.advance(seconds)

    def confirm_unknown_face(self) -> None:
        for _ in range(4):
            self.feed(self.unknown_result())
        self.assertIs(self.state, EnrollmentState.NAME_INPUT)

    def enroll(self, name: str) -> None:
        self.confirm_unknown_face()
        for char in name:
            self.assertFalse(self.app._handle_key(ord(char)))
        self.app._handle_key(ENTER_KEYS[1])
        self.assertIs(self.state, EnrollmentState.CAPTURING)
        for axis in range(1, 6):
            self.feed(self.unknown_result(axis), seconds=0.5)

    def test_complete_registration(self) -> None:
        self.enroll("Mario")
        self.assertIs(self.state, EnrollmentState.IDLE)
        self.assertEqual([name for name, _, _ in self.db.created], ["Mario"])
        self.assertEqual(self.app._ui.toasts, ["Richiesta inviata, Mario: attendi l'approvazione"])

    def test_decisions_of_the_server_have_their_own_message(self) -> None:
        self.assertEqual(
            _decision_toast(EnrollmentDecision("uid-1", "Mario Rossi", approved=True))[0],
            "Registrazione approvata. Benvenuto, Mario Rossi!",
        )
        self.assertEqual(
            _decision_toast(EnrollmentDecision("uid-1", "Mario", approved=False))[0],
            "Registrazione di Mario non approvata",
        )

    def test_face_waiting_for_approval_is_not_proposed_again(self) -> None:
        for _ in range(6):
            result = self.unknown_result()
            result.detections[0].match = Match("uid-1", "Mario", 0.9, pending=True)
            self.feed(result)
        self.assertIs(self.state, EnrollmentState.IDLE)

    def test_registration_starts_with_a_known_person_in_view(self) -> None:
        anna = make_detection(other_person_vector(), match=Match("uid-anna", "Anna", 0.8))
        for _ in range(4):
            result = self.unknown_result()
            result.detections.insert(0, anna)
            self.feed(result)
        self.assertIs(self.state, EnrollmentState.NAME_INPUT)

    def test_prompt_is_withdrawn_when_the_face_turns_out_to_be_registered(self) -> None:
        self.confirm_unknown_face()
        for _ in range(2):  # la stessa persona, ora riconosciuta: era un falso "sconosciuto"
            result = self.unknown_result()
            result.detections[0].match = Match("uid-anna", "Anna", 0.8)
            self.feed(result)
        self.assertIs(self.state, EnrollmentState.IDLE)
        self.assertEqual(self.app._ui.toasts, ["Volto già registrato come Anna"])

    def test_results_older_than_the_save_do_not_restart_the_registration(self) -> None:
        self.enroll("Mario")
        # Risultati calcolati su fotogrammi catturati prima del salvataggio: Mario vi appare sconosciuto
        for _ in range(2):
            self.frame_id += 1
            stale = FrameResult(frame_id=self.frame_id, detections=[make_detection(face_vector(1))])
            for _ in range(5):
                self.feed(stale)
        self.assertIs(self.state, EnrollmentState.IDLE)

        # Un'altra persona sconosciuta, inquadrata dopo il salvataggio, viene proposta normalmente
        self.confirm_unknown_face()

    def test_same_result_counts_once(self) -> None:
        result = self.unknown_result()
        for _ in range(50):  # il ciclo UI rivede lo stesso risultato molte volte
            self.feed(result)
        self.assertIs(self.state, EnrollmentState.IDLE)

    def test_results_without_gallery_are_ignored(self) -> None:
        for _ in range(10):
            self.feed(self.unknown_result(gallery_ok=False))
            self.feed(None)
        self.assertIs(self.state, EnrollmentState.IDLE)

    def test_quit_key_types_a_letter_while_entering_the_name(self) -> None:
        self.assertTrue(self.app._handle_key(ord("q")))
        self.confirm_unknown_face()
        self.assertFalse(self.app._handle_key(ord("q")))
        self.assertEqual(self.app._enrollment.name, "q")


class GreetingTextTest(unittest.TestCase):
    @staticmethod
    def back(*names: str) -> list[Greeting]:
        return [Greeting(GreetingKind.WELCOME_BACK, name) for name in names]

    def test_one_or_more_names(self) -> None:
        self.assertEqual(_greeting_texts([]), [])
        self.assertEqual(_greeting_texts(self.back("Mario")), ["Bentornato, Mario!"])
        self.assertEqual(_greeting_texts(self.back("Mario", "Anna")), ["Bentornati, Mario e Anna!"])
        self.assertEqual(
            _greeting_texts(self.back("Mario", "Anna", "Mario", "Luca")),
            ["Bentornati, Mario, Anna e Luca!"],
        )

    def test_welcome_and_welcome_back_are_separate_messages(self) -> None:
        greetings = [Greeting(GreetingKind.WELCOME, "Carla"), *self.back("Mario", "Anna")]
        self.assertEqual(_greeting_texts(greetings), ["Benvenuto, Carla!", "Bentornati, Mario e Anna!"])
        newcomers = [Greeting(GreetingKind.WELCOME, "Carla"), Greeting(GreetingKind.WELCOME, "Dario")]
        self.assertEqual(_greeting_texts(newcomers), ["Benvenuti, Carla e Dario!"])


if __name__ == "__main__":
    unittest.main()
