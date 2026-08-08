-- Setup script for weather_embeddings table
-- Run this AFTER sql/05_setup_weather_documents_table.sql.
--
-- Stores chunk-level embeddings for weather_documents.narrative_text, computed
-- by notebooks/ingest_weather_embeddings.py using
-- sentence-transformers/all-MiniLM-L6-v2 (384-dim) -- the SAME model used by
-- the ticker-news pipeline, so both stay compatible with the same pgvector
-- distance operator conventions. If you swap models, update the VECTOR(384)
-- dimension below to match.

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES weather_documents(id),
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
ON weather_embeddings (document_id);

-- Verify the table was created
SELECT
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
