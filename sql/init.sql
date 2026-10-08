-- =============================================================================
--  Schema del sistema di riconoscimento facciale (PostgreSQL + pgvector)
--  Eseguito automaticamente dal container Docker la prima volta che viene
--  creato il volume dei dati. Per rieseguirlo da zero: docker compose down -v
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- Anagrafica delle persone registrate
CREATE TABLE IF NOT EXISTS persons (
    id              SERIAL PRIMARY KEY,
    full_name       VARCHAR(120) NOT NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    sample_count    INTEGER      NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_access_at  TIMESTAMPTZ
);

-- Nomi univoci, senza distinzione tra maiuscole e minuscole
CREATE UNIQUE INDEX IF NOT EXISTS uq_persons_full_name_ci ON persons (LOWER(full_name));

-- Vettori biometrici (ArcFace, 512 dimensioni, normalizzati L2)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id             BIGSERIAL PRIMARY KEY,
    person_id      INTEGER     NOT NULL REFERENCES persons (id) ON DELETE CASCADE,
    embedding      VECTOR(512) NOT NULL,
    quality_score  REAL,
    source         VARCHAR(20) NOT NULL DEFAULT 'enrollment'
                   CHECK (source IN ('enrollment', 'continuous')),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_person ON face_embeddings (person_id);

-- Indice HNSW per la ricerca approssimata del vicino piu simile (distanza coseno)
CREATE INDEX IF NOT EXISTS idx_face_embeddings_hnsw
    ON face_embeddings USING hnsw (embedding vector_cosine_ops);

-- Log degli eventi (i saluti e le registrazioni).
-- Se una persona viene cancellata, i suoi log restano ma diventano anonimi (person_id NULL).
CREATE TABLE IF NOT EXISTS access_logs (
    id           BIGSERIAL PRIMARY KEY,
    person_id    INTEGER REFERENCES persons (id) ON DELETE SET NULL,
    event_type   VARCHAR(30) NOT NULL CHECK (event_type IN ('GREETED', 'ENROLLED')),
    similarity   REAL,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_access_logs_person_time
    ON access_logs (person_id, occurred_at DESC);
