"""Regole di "Benvenuto" e "Bentornato"."""

from __future__ import annotations

import unittest

from app.greeting_policy import GreetingPolicy
from app.models import GreetingKind
from tests.helpers import FakeClock

ABSENCE = 600.0
PERSON = 7


class GreetingPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.policy = GreetingPolicy(ABSENCE, self.clock)

    def test_welcome_only_on_first_recognition_ever(self) -> None:
        self.assertIs(self.policy.observe(PERSON, None), GreetingKind.WELCOME)
        self.clock.advance(1.0)
        self.assertIsNone(self.policy.observe(PERSON, None))

    def test_no_greeting_while_person_stays_in_view(self) -> None:
        self.policy.observe(PERSON, None)
        for _ in range(400):  # 20 minuti davanti alla webcam, un avvistamento ogni 3 secondi
            self.clock.advance(3.0)
            self.assertIsNone(self.policy.observe(PERSON, 0.0))

    def test_no_greeting_after_short_absence(self) -> None:
        self.policy.observe(PERSON, None)
        self.clock.advance(ABSENCE - 1.0)
        self.assertIsNone(self.policy.observe(PERSON, 0.0))

    def test_welcome_back_after_ten_minutes_out_of_view(self) -> None:
        self.policy.observe(PERSON, None)
        self.clock.advance(ABSENCE)
        self.assertIs(self.policy.observe(PERSON, 0.0), GreetingKind.WELCOME_BACK)
        self.clock.advance(1.0)
        self.assertIsNone(self.policy.observe(PERSON, 0.0))

    def test_absence_is_measured_from_last_sighting_not_from_last_greeting(self) -> None:
        self.policy.observe(PERSON, None)
        self.clock.advance(ABSENCE - 10.0)
        self.assertIsNone(self.policy.observe(PERSON, 0.0))  # rivista: l'assenza riparte da qui
        self.clock.advance(ABSENCE - 10.0)
        self.assertIsNone(self.policy.observe(PERSON, 0.0))

    def test_first_sighting_after_restart_uses_saved_last_access(self) -> None:
        self.assertIsNone(self.policy.observe(1, ABSENCE - 1.0))
        self.assertIs(self.policy.observe(2, ABSENCE), GreetingKind.WELCOME_BACK)

    def test_people_are_tracked_independently(self) -> None:
        self.policy.observe(1, None)
        self.clock.advance(ABSENCE)
        self.assertIs(self.policy.observe(2, None), GreetingKind.WELCOME)
        self.assertIs(self.policy.observe(1, 0.0), GreetingKind.WELCOME_BACK)


if __name__ == "__main__":
    unittest.main()
