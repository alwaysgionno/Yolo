"""Test di DatabaseManager con un pool simulato: gestione di errori e connessioni cadute."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import psycopg2

from app.database_manager import DatabaseError, DatabaseManager, to_pgvector
from tests.helpers import EMBEDDING_SIZE, face_vector, make_settings


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, query: str, params: object = None) -> None:
        self._connection.queries.append(" ".join(query.split()))
        if self._connection.fail_with is not None:
            raise self._connection.fail_with

    def executemany(self, query: str, rows: object) -> None:
        self.execute(query)

    def fetchone(self) -> tuple | None:
        return self._connection.row

    def fetchall(self) -> list[tuple]:
        return [(1, "Mario", 0.75)]


class FakeConnection:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.row: tuple | None = (1,)
        self.fail_with: Exception | None = None
        self.rollback_fails = False
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        if self.rollback_fails:
            self.closed = 1
            raise psycopg2.InterfaceError("connection already closed")
        self.rollbacks += 1


class FakePool:
    """Sostituisce ThreadedConnectionPool: consegna sempre la stessa connessione simulata."""

    row: tuple | None = (1,)

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.connection = FakeConnection()
        self.connection.row = self.row
        self.getconn_error: Exception | None = None
        self.returned: list[bool] = []  # valore di `close` a ogni restituzione
        self.closed = False

    def getconn(self) -> FakeConnection:
        if self.getconn_error is not None:
            raise self.getconn_error
        return self.connection

    def putconn(self, conn: FakeConnection, close: bool = False) -> None:
        self.returned.append(close)

    def closeall(self) -> None:
        self.closed = True


class EmptySchemaPool(FakePool):
    row = None  # ogni controllo dello schema risponde "non trovato"


def patch_pool(pool_class: type) -> mock._patch:
    return mock.patch("app.database_manager.pool.ThreadedConnectionPool", pool_class)


class ConnectionHandlingTest(unittest.TestCase):
    def setUp(self) -> None:
        with patch_pool(FakePool):
            self.manager = DatabaseManager(make_settings())
        self.pool = self.manager._pool
        self.connection = self.pool.connection

    def test_successful_query_is_committed_and_connection_returned(self) -> None:
        commits = self.connection.commits
        matches = self.manager.find_nearest(face_vector(1), limit=5)
        self.assertEqual(matches[0].full_name, "Mario")
        self.assertEqual(self.connection.commits, commits + 1)
        self.assertEqual(self.pool.returned[-1], False)

    def test_sql_error_becomes_database_error(self) -> None:
        self.connection.fail_with = psycopg2.OperationalError("errore simulato")
        with self.assertRaises(DatabaseError):
            self.manager.person_exists("Mario")
        self.assertEqual(self.connection.rollbacks, 1)
        self.assertEqual(self.pool.returned[-1], False)

    def test_dropped_connection_becomes_database_error_and_is_discarded(self) -> None:
        self.connection.fail_with = psycopg2.OperationalError("server closed the connection")
        self.connection.rollback_fails = True
        with self.assertRaises(DatabaseError):
            self.manager.record_greeting(1, 0.8)
        self.assertEqual(self.pool.returned[-1], True)

    def test_unreachable_database_becomes_database_error(self) -> None:
        self.pool.getconn_error = psycopg2.OperationalError("connection refused")
        with self.assertRaises(DatabaseError):
            self.manager.find_nearest(face_vector(1), limit=5)

    def test_continuous_embedding_prunes_and_recounts(self) -> None:
        self.connection.queries.clear()
        self.manager.add_continuous_embedding(1, face_vector(2), quality=88.0, max_stored=20)
        statements = [query.split()[0] for query in self.connection.queries]
        self.assertEqual(statements, ["INSERT", "DELETE", "UPDATE"])
        self.assertIn("SELECT COUNT(*) FROM face_embeddings", self.connection.queries[-1])

    def test_create_person_needs_one_quality_per_embedding(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.create_person("Mario", [face_vector(1), face_vector(2)], [90.0])
        with self.assertRaises(ValueError):
            self.manager.create_person("Mario", [], [])

    def test_close_can_be_called_twice(self) -> None:
        self.manager.close()
        self.manager.close()
        self.assertTrue(self.pool.closed)


class StartupTest(unittest.TestCase):
    def test_missing_schema_stops_startup_and_closes_the_pool(self) -> None:
        created: list[FakePool] = []

        class RecordingPool(EmptySchemaPool):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                created.append(self)

        with patch_pool(RecordingPool), self.assertRaises(DatabaseError):
            DatabaseManager(make_settings())
        self.assertTrue(created[0].closed)

    def test_connection_failure_at_startup_becomes_database_error(self) -> None:
        def refuse(*args: object, **kwargs: object) -> None:
            raise psycopg2.OperationalError("connection refused")

        with patch_pool(refuse), self.assertRaises(DatabaseError):  # type: ignore[arg-type]
            DatabaseManager(make_settings())


class VectorFormatTest(unittest.TestCase):
    def test_vector_is_written_in_pgvector_format(self) -> None:
        vector = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
        vector[0] = 0.5
        vector[1] = -1.0
        text = to_pgvector(vector)
        self.assertTrue(text.startswith("[0.5000000,-1.0000000,0.0000000,"))
        self.assertTrue(text.endswith("]"))
        self.assertEqual(text.count(","), EMBEDDING_SIZE - 1)

    def test_wrong_dimension_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            to_pgvector(np.zeros(128, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
