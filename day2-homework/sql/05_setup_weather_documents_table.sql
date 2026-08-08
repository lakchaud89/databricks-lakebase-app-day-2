-- Setup script for weather_documents table
-- Run this manually in your Lakebase Postgres database before syncing weather data.
--
-- This is the RAW document store for unstructured weather text harvested from
-- the National Weather Service API (see weather_client.py / POST /weather/sync):
-- active alerts (source_type='alert') and multi-day forecast periods
-- (source_type='forecast'). notebooks/ingest_weather_embeddings.py reads
-- narrative_text from here to compute vector embeddings into weather_embeddings.

CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
    headline TEXT,
    narrative_text TEXT,
    issued_at TIMESTAMPTZ,
    effective_at TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location ON weather_documents (location);
CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type ON weather_documents (source_type);

-- Verify the table was created
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
