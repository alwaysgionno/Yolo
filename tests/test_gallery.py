"""Test della galleria locale: messaggi del server, ricerca del volto piu simile, sincronizzazione."""

from __future__ import annotations

import unittest

import numpy as np

from app import contract
from app.contract import ContractError
from app.gallery import Gallery
from tests.helpers import FakeClock, face_vector, other_person_vector

MARIO, ANNA, CARLA = "uid-mario", "uid-anna", "uid-carla"


def person_message(full_name: str, embeddings: list[np.ndarray], last_access_at: str | None = None) -> bytes:
    """Messaggio `gallery/{uid}` come lo pubblica il server."""
    return contract.encode_json(
        {
            "full_name": full_name,
            "active": bool(embeddings),
            "last_access_at": last_access_at,
            "embeddings": [contract.encode_embedding(vector) for vector in embeddings],
        }
    )


def info_message(*person_uids: str) -> bytes:
    return contract.encode_json({"person_uids": list(person_uids)})


class GalleryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock(start=1_800_000_000.0)  # orologio di sistema, in secondi Unix
        self.gallery = Gallery(wall_clock=self.clock)


class SearchTest(GalleryTestCase):
    def test_empty_gallery_has_no_matches(self) -> None:
        self.assertEqual(self.gallery.find_nearest(face_vector(1), 5), [])

    def test_nearest_person_comes_first_with_cosine_similarity(self) -> None:
        self.gallery.apply_person_message(MARIO, person_message("Mario", [face_vector(1), face_vector(2)]))
        self.gallery.apply_person_message(ANNA, person_message("Anna", [other_person_vector()]))

        matches = self.gallery.find_nearest(face_vector(1), 5)
        self.assertEqual([match.person_id for match in matches], [MARIO, MARIO, ANNA])
        self.assertAlmostEqual(matches[0].similarity, 1.0, places=5)
        self.assertAlmostEqual(matches[1].similarity, float(np.dot(face_vector(1), face_vector(2))), places=5)
        self.assertAlmostEqual(matches[2].similarity, 0.0, places=5)
        self.assertEqual(len(self.gallery.find_nearest(face_vector(1), 1)), 1)

    def test_similarity_does_not_depend_on_vector_length(self) -> None:
        self.gallery.apply_person_message(MARIO, person_message("Mario", [face_vector(1) * 3.0]))
        self.assertAlmostEqual(self.gallery.find_nearest(face_vector(1) * 0.5, 1)[0].similarity, 1.0, places=5)

    def test_inactive_person_is_never_matched_but_keeps_the_name(self) -> None:
        self.gallery.apply_person_message(MARIO, person_message("Mario", []))
        self.assertEqual(self.gallery.find_nearest(face_vector(1), 5), [])
        self.assertTrue(self.gallery.name_exists("MARIO"))

    def test_age_of_the_last_access(self) -> None:
        saved = contract.format_timestamp(self.clock.now - 900.0)
        self.gallery.apply_person_message(MARIO, person_message("Mario", [face_vector(1)], saved))
        self.gallery.apply_person_message(ANNA, person_message("Anna", [other_person_vector()]))

        self.assertAlmostEqual(self.gallery.find_nearest(face_vector(1), 1)[0].last_access_age_s, 900.0, places=3)
        self.assertIsNone(self.gallery.find_nearest(other_person_vector(), 1)[0].last_access_age_s)

        self.gallery.mark_greeted(ANNA)
        self.clock.advance(5.0)
        self.assertAlmostEqual(self.gallery.find_nearest(other_person_vector(), 1)[0].last_access_age_s, 5.0, places=3)


class UpdateTest(GalleryTestCase):
    def test_new_message_replaces_the_person(self) -> None:
        self.gallery.apply_person_message(MARIO, person_message("Mario", [face_vector(1)]))
        self.gallery.apply_person_message(MARIO, person_message("Mario Rossi", [face_vector(1), face_vector(2)]))
        matches = self.gallery.find_nearest(face_vector(1), 5)
        self.assertEqual([match.full_name for match in matches], ["Mario Rossi", "Mario Rossi"])
        self.assertFalse(self.gallery.name_exists("Mario"))

    def test_empty_message_removes_the_person(self) -> None:
        self.gallery.apply_person_message(MARIO, person_message("Mario", [face_vector(1)]))
        self.gallery.apply_person_message(MARIO, b"")
        self.assertEqual(self.gallery.find_nearest(face_vector(1), 5), [])
        self.assertFalse(self.gallery.name_exists("Mario"))
        self.gallery.apply_person_message(ANNA, b"")  # persona mai vista: nessun effetto

    def test_invalid_message_leaves_the_gallery_unchanged(self) -> None:
        self.gallery.apply_person_message(MARIO, person_message("Mario", [face_vector(1)]))
        broken = [
            b"non e json",
            contract.encode_json({"full_name": "Mario"}),
            contract.encode_json({"full_name": "Mario", "embeddings": ["abc"]}),
            contract.encode_json({"full_name": "Mario", "embeddings": [], "last_access_at": "ieri"}),
        ]
        for payload in broken:
            with self.subTest(payload=payload[:30]):
                with self.assertRaises(ContractError):
                    self.gallery.apply_person_message(MARIO, payload)
        self.assertEqual(len(self.gallery.find_nearest(face_vector(1), 5)), 1)

    def test_person_added_locally_is_recognized_at_once(self) -> None:
        self.gallery.add_local(CARLA, "Carla", [face_vector(1), face_vector(2)])
        match = self.gallery.find_nearest(face_vector(1), 1)[0]
        self.assertEqual((match.person_id, match.full_name), (CARLA, "Carla"))
        self.assertEqual(match.last_access_age_s, 0.0)  # la registrazione vale come primo accesso
        self.assertTrue(self.gallery.name_exists("carla"))


class SyncTest(GalleryTestCase):
    def test_ready_only_when_every_announced_person_has_arrived(self) -> None:
        self.assertFalse(self.gallery.ready)
        self.gallery.apply_person_message(MARIO, person_message("Mario", [face_vector(1)]))
        self.assertFalse(self.gallery.ready)  # senza elenco non si sa se la galleria e completa
        self.gallery.apply_info_message(info_message(MARIO, ANNA))
        self.assertFalse(self.gallery.ready)
        self.gallery.apply_person_message(ANNA, person_message("Anna", [other_person_vector()]))
        self.assertTrue(self.gallery.ready)

    def test_empty_server_is_ready_at_once(self) -> None:
        self.gallery.apply_info_message(info_message())
        self.assertTrue(self.gallery.ready)

    def test_ready_is_kept_while_later_updates_arrive(self) -> None:
        self.gallery.apply_info_message(info_message())
        self.gallery.apply_info_message(info_message(MARIO))  # annunciata, non ancora arrivata
        self.assertTrue(self.gallery.ready)

    def test_list_removes_persons_deleted_on_the_server(self) -> None:
        self.gallery.apply_person_message(MARIO, person_message("Mario", [face_vector(1)]))
        self.gallery.apply_person_message(ANNA, person_message("Anna", [other_person_vector()]))
        self.gallery.apply_info_message(info_message(ANNA))
        self.assertEqual(len(self.gallery), 1)
        self.assertEqual(self.gallery.find_nearest(face_vector(1), 5)[0].person_id, ANNA)

    def test_list_keeps_a_local_person_until_the_server_answers(self) -> None:
        self.gallery.apply_info_message(info_message())
        self.gallery.add_local(CARLA, "Carla", [face_vector(1)])
        self.gallery.apply_info_message(info_message())  # elenco pubblicato prima che il server la salvasse
        self.assertTrue(self.gallery.name_exists("Carla"))

        # Il server conferma: da qui in poi vale il suo elenco
        self.gallery.apply_person_message(CARLA, person_message("Carla", [face_vector(1)]))
        self.gallery.apply_info_message(info_message())
        self.assertFalse(self.gallery.name_exists("Carla"))

    def test_server_can_reject_a_local_person(self) -> None:
        self.gallery.add_local(CARLA, "Carla", [face_vector(1)])
        self.gallery.apply_person_message(CARLA, b"")
        self.assertEqual(self.gallery.find_nearest(face_vector(1), 5), [])

    def test_invalid_list_is_refused(self) -> None:
        for payload in (b"[]", contract.encode_json({"person_uids": "tutti"}), contract.encode_json({})):
            with self.assertRaises(ContractError):
                self.gallery.apply_info_message(payload)
        self.assertFalse(self.gallery.ready)


if __name__ == "__main__":
    unittest.main()
