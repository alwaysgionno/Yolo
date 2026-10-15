"""Test della configurazione: lettura del file .env e validazione dei valori."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import config
from app.config import ConfigurationError, Settings
from tests.helpers import make_settings


class ValidationTest(unittest.TestCase):
    def test_default_values_are_valid(self) -> None:
        make_settings()._validate()

    def test_invalid_values_are_refused(self) -> None:
        cases = {
            "password di esempio": {"mqtt_password": "cambia_la_password_della_telecamera"},
            "soglia fuori intervallo": {"recognition_threshold": 1.2},
            "apprendimento sotto il riconoscimento": {"continuous_learning_threshold": 0.30},
            "pochi campioni": {"enrollment_samples": 4},
            "troppi campioni": {"enrollment_samples": 11},
            "porta non valida": {"mqtt_port": 70000},
            "dimensione non multipla di 32": {"insightface_det_size": 300},
            "batch nullo": {"continuous_batch_size": 0},
            "attesa negativa": {"unknown_confirm_seconds": -1.0},
            "assenza negativa": {"welcome_back_absence_seconds": -1.0},
            "attesa del saluto con la mano negativa": {"wave_cooldown_seconds": -1.0},
        }
        for label, overrides in cases.items():
            with self.subTest(label):
                with self.assertRaises(ConfigurationError):
                    make_settings(**overrides)._validate()


class EnvFileTest(unittest.TestCase):
    def load(self, content: str) -> Settings:
        """Carica le impostazioni da un file .env temporaneo, senza toccare l'ambiente reale."""
        with tempfile.TemporaryDirectory() as folder:
            env_file = Path(folder) / ".env"
            env_file.write_text(content, encoding="utf-8")
            with mock.patch.object(config, "ENV_FILE", env_file), mock.patch.dict(os.environ, {}, clear=True):
                return Settings.from_env()

    def test_values_and_defaults(self) -> None:
        settings = self.load("MQTT_PASSWORD=una-password-di-prova\nCAMERA_INDEX=2\nRECOGNITION_THRESHOLD=0.45\n")
        self.assertEqual(settings.mqtt_password, "una-password-di-prova")
        self.assertEqual(settings.camera_index, 2)
        self.assertEqual(settings.recognition_threshold, 0.45)
        self.assertEqual(settings.mqtt_username, "camera")
        self.assertEqual(settings.unknown_confirm_seconds, 1.5)
        self.assertTrue(settings.yolo_model_path.is_absolute())

    def test_missing_password_is_refused(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.load("DB_USER=face_app\n")

    def test_wrong_number_is_refused(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.load("MQTT_PASSWORD=una-password-di-prova\nMQTT_PORT=cinquemila\n")

    def test_missing_file_is_refused(self) -> None:
        with mock.patch.object(config, "ENV_FILE", Path("questo_file_non_esiste.env")):
            with self.assertRaises(ConfigurationError):
                Settings.from_env()

    def test_example_file_only_needs_a_password(self) -> None:
        example = (config.PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            self.load(example)  # la password di esempio non e accettata
        settings = self.load(example.replace("cambia_la_password_della_telecamera", "una-password-di-prova"))
        self.assertEqual(settings.enrollment_samples, 8)
        self.assertEqual(settings.unknown_confirm_seconds, 1.5)
        self.assertEqual(settings.welcome_back_absence_seconds, 600.0)
        self.assertFalse(settings.wave_greeting)  # il saluto con la mano si attiva telecamera per telecamera
        self.assertEqual(settings.wave_cooldown_seconds, 30.0)
        self.assertTrue(settings.pose_model_path.is_absolute())

    def test_wave_greeting_can_be_enabled(self) -> None:
        settings = self.load("MQTT_PASSWORD=una-password-di-prova\nWAVE_GREETING=1\nWAVE_COOLDOWN_SECONDS=10\n")
        self.assertTrue(settings.wave_greeting)
        self.assertEqual(settings.wave_cooldown_seconds, 10.0)

    def test_old_greeting_cooldown_is_ignored(self) -> None:
        # Un .env precedente puo contenere ancora la vecchia variabile: non deve accorciare l'assenza richiesta
        settings = self.load("MQTT_PASSWORD=una-password-di-prova\nGREETING_COOLDOWN_SECONDS=300\n")
        self.assertEqual(settings.welcome_back_absence_seconds, 600.0)


if __name__ == "__main__":
    unittest.main()
