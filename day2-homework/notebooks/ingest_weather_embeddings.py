"""
Ingest Weather Documents -> Vector Embeddings (Lakebase)

Plain Python script (NOT a Databricks notebook, NOT Spark) that:
  1. Reads weather_documents rows that don't have embeddings yet.
  2. Chunks narrative_text with a sliding window (CHUNK_SIZE/CHUNK_OVERLAP).
  3. Embeds each chunk with sentence-transformers/all-MiniLM-L6-v2 (384-dim) --
     the same model used by the ticker-news pipeline, so both pipelines stay
     queryable with the same pgvector distance-operator conventions.
  4. Writes the chunk embeddings into weather_embeddings via psycopg2
     (execute_values), casting to vector(384).

Run directly:
    python notebooks/ingest_weather_embeddings.py

Or as a Databricks Job "Python script" task pointed at this file -- it needs
no dbutils/widgets machinery, since Spark JDBC writes don't work reliably
against this Lakebase instance (per the assignment). It reuses lakebase.py's
get_connection() for the Lakebase URL rather than re-implementing a secret
lookup here -- that duplication is exactly what let the class reference
app's Lakebase secret scope drift out of sync in an earlier version of this
project, so every script in this repo goes through the one shared helper.
"""


import os
from datetime import datetime, timezone

import psycopg2.extras

import lakebase

# Table names without schema prefix - let Postgres use search_path to find them
# (The tables were created without schema qualification, so they live in whatever
# schema was active at creation time, which Postgres will find via search_path)
WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
WEATHER_EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 32))


def ensure_weather_embeddings_table() -> None:
    """Belt-and-suspenders: makes this script runnable even if
    sql/06_setup_weather_embeddings_table.sql hasn't been run manually yet."""
    lakebase.run_write("CREATE EXTENSION IF NOT EXISTS vector")
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_EMBEDDINGS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES {WEATHER_TABLE_NAME}(id),
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding "
        f"ON {WEATHER_EMBEDDINGS_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id "
        f"ON {WEATHER_EMBEDDINGS_TABLE_NAME} (document_id)"
    )


def fetch_unembedded_documents() -> list[dict]:
    """Documents with narrative text that don't have any rows in
    weather_embeddings yet (anti-join, so re-running this script is cheap
    and idempotent instead of re-embedding everything every time)."""
    return lakebase.run_query(
        f"""
        SELECT wd.id, wd.location, wd.headline, wd.narrative_text
        FROM {WEATHER_TABLE_NAME} wd
        WHERE wd.narrative_text IS NOT NULL AND wd.narrative_text != ''
          AND NOT EXISTS (
              SELECT 1 FROM {WEATHER_EMBEDDINGS_TABLE_NAME} we
              WHERE we.document_id = wd.id
          )
        """
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunking, identical scheme to the ticker-news pipeline
    (notebooks/ingest_ticker_news_embeddings-v2.py). Most NWS narrative text
    is short enough to fit in a single chunk; this only matters for the
    longer combined alert description+instruction text."""
    chunks = []
    for start in range(0, len(text), chunk_size - chunk_overlap):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def main() -> None:
    ensure_weather_embeddings_table()

    documents = fetch_unembedded_documents()
    print(f"Found {len(documents)} weather documents without embeddings.")
    if not documents:
        print("Nothing to embed. Run POST /weather/sync first if this is unexpected.")
        return

    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunk_rows: list[dict] = []
    for doc in documents:
        for chunk_index, piece in enumerate(chunk_text(doc["narrative_text"])):
            chunk_rows.append(
                {
                    "id": f"{doc['id']}_{chunk_index}",
                    "document_id": doc["id"],
                    "chunk_index": chunk_index,
                    "chunk_text": piece,
                }
            )

    if not chunk_rows:
        print("No non-empty chunks produced from the fetched documents.")
        return

    print(f"Embedding {len(chunk_rows)} chunks from {len(documents)} documents...")
    all_texts = [row["chunk_text"] for row in chunk_rows]
    all_embeddings = []
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[i:i + BATCH_SIZE]
        vectors = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(vectors.tolist())
        if (i + BATCH_SIZE) % (BATCH_SIZE * 4) == 0:
            print(f"  Embedded {min(i + BATCH_SIZE, len(all_texts))}/{len(all_texts)} chunks")

    now = datetime.now(timezone.utc)
    insert_data = [
        (
            row["id"],
            row["document_id"],
            row["chunk_index"],
            row["chunk_text"],
            "{" + ",".join(str(float(x)) for x in embedding) + "}",
            EMBEDDING_MODEL_NAME,
            now,
        )
        for row, embedding in zip(chunk_rows, all_embeddings)
    ]

    insert_sql = f"""
        INSERT INTO {WEATHER_EMBEDDINGS_TABLE_NAME} (
            id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
        ) VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    # Cast the string array literal to double precision[] -- pgvector's
    # ASSIGNMENT cast into vector(384) fires automatically on insert. This is
    # the demonstrably-working write path against this Lakebase instance
    # (see sql/04_cast_arrays_to_vectors.sql for the belt-and-suspenders
    # re-cast this script also runs below).
    template = "(%s, %s, %s, %s, %s::double precision[], %s, %s)"

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, insert_sql, insert_data, template=template, page_size=100)
            inserted = cur.rowcount
        conn.commit()

    print(f"Inserted {inserted} chunk embeddings from {len(documents)} documents")

    # Defensive re-cast, no-op if the implicit ASSIGNMENT cast above already
    # worked -- kept for parity with the known-working reference pipeline.
    lakebase.run_write(
        f"UPDATE {WEATHER_EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector "
        f"WHERE embedding IS NOT NULL"
    )
    print("Vector search is ready.")


if __name__ == "__main__":
    main()
