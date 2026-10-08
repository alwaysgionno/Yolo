"""Accesso a PostgreSQL con estensione pgvector: anagrafiche, embedding e log degli accessi.

Il pool di connessioni e thread-safe: viene usato sia dal thread di inferenza
(riconoscimento e saluti) sia dal thread della UI (registrazione di nuovi volti).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection as PgConnection

from .config import Settings
from .models import Match

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 512   # deve coincidere con VECTOR(512) definito in sql/init.sql
CONNECT_TIMEOUT_S = 5
STATEMENT_TIMEOUT_MS = 5000  # una query bloccata non deve fermare il thread di inferenza


class DatabaseError(RuntimeError):
    """Errore di accesso al database: connessione, query o schema incompleto."""


def to_pgvector(vector: np.ndarray) -> str:
    """Converte un vettore NumPy nel formato testuale di pgvector, ad esempio '[0.1,0.2,0.3]'."""
    if vector.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(f"Embedding con forma {vector.shape}, atteso ({EMBEDDING_DIMENSION},)")
    return "[" + ",".join(f"{value:.7f}" for value in vector.astype(np.float32).tolist()) + "]"


def _safe_rollback(conn: PgConnection) -> bool:
    """Annulla la transazione. Restituisce False se la connessione non e piu utilizzabile."""
    try:
        conn.rollback()
        return True
    except psycopg2.Error:
        return False


class DatabaseManager:
    """Incapsula tutte le operazioni SQL dell'applicazione."""

    REQUIRED_TABLES = ("persons", "face_embeddings", "access_logs")

    def __init__(self, settings: Settings, min_connections: int = 2, max_connections: int = 5) -> None:
        # Due connessioni sempre aperte, una per il thread di inferenza e una per la UI:
        # cosi nessuno dei due deve aprirne una nuova mentre l'altro lavora
        try:
            self._pool = pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                host=settings.db_host,
                port=settings.db_port,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                connect_timeout=CONNECT_TIMEOUT_S,
                options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
            )
        except psycopg2.Error as exc:
            raise DatabaseError(
                f"Connessione a PostgreSQL non riuscita "
                f"({settings.db_host}:{settings.db_port}/{settings.db_name}): {exc}"
            ) from exc
        try:
            self._verify_schema()
        except Exception:
            self.close()
            raise
        logger.info("Connesso a PostgreSQL (%s/%s)", settings.db_host, settings.db_name)

    @contextmanager
    def _connection(self) -> Iterator[PgConnection]:
        """Fornisce una connessione dal pool: commit automatico, oppure rollback in caso di errore."""
        try:
            conn = self._pool.getconn()
        except (pool.PoolError, psycopg2.Error) as exc:
            raise DatabaseError(f"Nessuna connessione disponibile al database: {exc}") from exc
        usable = True
        try:
            yield conn
            conn.commit()
        except psycopg2.Error as exc:
            usable = _safe_rollback(conn)
            raise DatabaseError(f"Errore SQL: {exc}") from exc
        except Exception:
            usable = _safe_rollback(conn)
            raise
        finally:
            # Una connessione caduta (ad esempio dopo un riavvio del database) viene chiusa:
            # alla richiesta successiva il pool ne apre una nuova
            try:
                self._pool.putconn(conn, close=not usable or bool(conn.closed))
            except (pool.PoolError, psycopg2.Error) as exc:
                logger.debug("Connessione non restituita al pool: %s", exc)

    def _verify_schema(self) -> None:
        """Controlla che pgvector e le tabelle siano presenti, altrimenti interrompe con un messaggio chiaro."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone() is None:
                raise DatabaseError(
                    "Estensione pgvector non attiva. Esegui 'docker compose down -v' "
                    "e 'docker compose up -d' per ricreare il database."
                )
            for table in self.REQUIRED_TABLES:
                cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                if cur.fetchone()[0] is None:
                    raise DatabaseError(f"Tabella mancante: {table}. Verifica che sql/init.sql sia stato eseguito.")

    def person_exists(self, full_name: str) -> bool:
        """Verifica se esiste gia una persona con lo stesso nome (senza distinguere maiuscole)."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM persons WHERE LOWER(full_name) = LOWER(%s)", (full_name,))
            return cur.fetchone() is not None

    def find_nearest(self, embedding: np.ndarray, limit: int) -> list[Match]:
        """Restituisce le `limit` corrispondenze piu vicine, ordinate per similarita decrescente.

        La similarita coseno e calcolata come 1 - distanza coseno (operatore <=> di pgvector).
        """
        vector = to_pgvector(embedding)
        query = """
            SELECT p.id, p.full_name, 1 - (fe.embedding <=> %(vec)s::vector) AS similarity
            FROM face_embeddings AS fe
            JOIN persons AS p ON p.id = fe.person_id
            WHERE p.is_active
            ORDER BY fe.embedding <=> %(vec)s::vector
            LIMIT %(limit)s
        """
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(query, {"vec": vector, "limit": limit})
            return [
                Match(person_id=int(row[0]), full_name=str(row[1]), similarity=float(row[2]))
                for row in cur.fetchall()
            ]

    def create_person(
        self,
        full_name: str,
        embeddings: Sequence[np.ndarray],
        qualities: Sequence[float],
    ) -> int:
        """Crea la persona, salva i suoi embedding iniziali e registra l'evento, in un'unica transazione."""
        if not embeddings:
            raise ValueError("Serve almeno un embedding per registrare una persona")
        if len(embeddings) != len(qualities):
            raise ValueError("Ogni embedding deve avere il proprio punteggio di qualita")
        rows = [(to_pgvector(emb), float(quality)) for emb, quality in zip(embeddings, qualities)]
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (full_name, sample_count) VALUES (%s, %s) RETURNING id",
                (full_name, len(rows)),
            )
            person_id = int(cur.fetchone()[0])
            cur.executemany(
                """
                INSERT INTO face_embeddings (person_id, embedding, quality_score, source)
                VALUES (%s, %s::vector, %s, 'enrollment')
                """,
                [(person_id, vector, quality) for vector, quality in rows],
            )
            cur.execute(
                "INSERT INTO access_logs (person_id, event_type) VALUES (%s, 'ENROLLED')",
                (person_id,),
            )
        logger.info("Nuova persona registrata: %s (id=%d, campioni=%d)", full_name, person_id, len(rows))
        return person_id

    def record_greeting(self, person_id: int, similarity: float) -> None:
        """Aggiorna il timestamp dell'ultimo accesso e registra l'evento di saluto."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE persons SET last_access_at = NOW() WHERE id = %s", (person_id,))
            cur.execute(
                "INSERT INTO access_logs (person_id, event_type, similarity) VALUES (%s, 'GREETED', %s)",
                (person_id, float(similarity)),
            )

    def add_continuous_embedding(
        self,
        person_id: int,
        embedding: np.ndarray,
        quality: float,
        max_stored: int,
    ) -> None:
        """Aggiunge un embedding di apprendimento continuo e rimuove i piu vecchi oltre il limite."""
        vector = to_pgvector(embedding)
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_embeddings (person_id, embedding, quality_score, source)
                VALUES (%s, %s::vector, %s, 'continuous')
                """,
                (person_id, vector, float(quality)),
            )
            cur.execute(
                """
                DELETE FROM face_embeddings
                WHERE id IN (
                    SELECT id FROM face_embeddings
                    WHERE person_id = %s AND source = 'continuous'
                    ORDER BY created_at DESC, id DESC
                    OFFSET %s
                )
                """,
                (person_id, max_stored),
            )
            # Il contatore viene ricalcolato, cosi resta corretto anche dopo la rimozione dei piu vecchi
            cur.execute(
                """
                UPDATE persons
                SET sample_count = (SELECT COUNT(*) FROM face_embeddings WHERE person_id = %s)
                WHERE id = %s
                """,
                (person_id, person_id),
            )
        logger.info("Profilo aggiornato con apprendimento continuo (id=%d)", person_id)

    def close(self) -> None:
        """Chiude tutte le connessioni del pool. Puo essere chiamato piu volte."""
        if not self._pool.closed:
            self._pool.closeall()
