"""
Databricks App: Weather Intelligence (Day 2 homework)
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls unstructured weather text from the NWS API via weather_client.py,
  embeds it, and serves a semantic search endpoint over it

This app has no dependency on the Massive API -- it's a standalone weather
harvest -> vectorize -> retrieve pipeline.

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os

import requests
from flask import Flask, jsonify, render_template, request

import lakebase
from weather_client import WeatherClient, resolve_location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

# Lazy-load the embedding model (only when vector search is used)
_embedding_model = None


def get_embedding_model():
    """Lazy-load the sentence transformer model for embedding queries.

    Loaded once at first use (not per-request), then reused for every
    /weather/search call.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except ImportError:
            logger.error("sentence-transformers not installed. Vector search will not work.")
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install it with: pip install sentence-transformers"
            )
    return _embedding_model


WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
WEATHER_EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Locations to sync weather for by default (comma-separated "City, ST"), e.g.
# "Chicago, IL,Austin, TX" -- NOTE: since locations themselves contain a comma,
# use a semicolon to separate multiple locations in the env var.
DEFAULT_WEATHER_LOCATIONS = [
    loc.strip()
    for loc in os.environ.get("WEATHER_LOCATIONS", "Sacramento, CA;Chicago, IL;New York, NY").split(";")
    if loc.strip()
]


def ensure_weather_documents_table():
    """
    Create the raw weather documents table in Lakebase if it doesn't exist
    yet. This is the RAW document store (alerts + forecast periods from the
    NWS API) that notebooks/ingest_weather_embeddings.py reads from to
    compute vector embeddings into WEATHER_EMBEDDINGS_TABLE_NAME.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
            headline TEXT,
            narrative_text TEXT,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_location "
        f"ON {WEATHER_TABLE_NAME} (location)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_source_type "
        f"ON {WEATHER_TABLE_NAME} (source_type)"
    )


def ensure_weather_embeddings_table():
    """
    Create the weather chunk-embeddings table in Lakebase if it doesn't
    exist yet. Called defensively from /weather/search so that endpoint
    degrades to an empty result set instead of a 500 when nothing has been
    embedded yet. See sql/06_setup_weather_embeddings_table.sql for the
    canonical, commented version of this DDL.
    """
    lakebase.run_write("CREATE EXTENSION IF NOT EXISTS vector")
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_EMBEDDINGS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES {WEATHER_TABLE_NAME}(id),
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_EMBEDDINGS_TABLE_NAME}_embedding "
        f"ON {WEATHER_EMBEDDINGS_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_EMBEDDINGS_TABLE_NAME}_document_id "
        f"ON {WEATHER_EMBEDDINGS_TABLE_NAME} (document_id)"
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI: sync weather data for a set of locations, then semantic
    search over the synced alerts + forecasts."""
    return render_template("index.html")


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Resolve each requested location to an NWS grid point, fetch active
    alerts + forecast periods for it (via WeatherClient.get_documents_for_point),
    and upsert the normalized documents into weather_documents.

    Body (optional JSON): {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    Defaults to DEFAULT_WEATHER_LOCATIONS when no locations are supplied.

    A single bad location (unresolvable, or an NWS API error for that point)
    is skipped rather than aborting the whole batch.
    """
    ensure_weather_documents_table()
    client = WeatherClient()

    body = request.json if request.is_json else {}
    locations = body.get("locations") or DEFAULT_WEATHER_LOCATIONS
    limit = int(body.get("limit", 50))

    total = 0
    synced_locations = []
    for location in locations:
        try:
            lat, lon, label = resolve_location(location)
            docs = client.get_documents_for_point(lat, lon, label=label, limit=limit)
        except ValueError as e:
            logger.warning(f"Skipping unresolvable weather location {location!r}: {e}")
            continue
        except requests.HTTPError as e:
            logger.warning(f"Skipping weather location {location!r}: NWS API error: {e}")
            continue

        total += _upsert_weather_batch(docs)
        synced_locations.append(label)

    return jsonify({"synced": total, "locations": synced_locations})


@app.route("/weather/search", methods=["POST"])
def weather_search():
    """
    Semantic search over synced weather documents (alerts + forecast
    narrative text): embeds the query with the same model used at ingestion
    time and returns the top K most similar chunks via pgvector's cosine
    similarity (<=>).

    Body: {"query": "flash flood risk this weekend", "top_k": 5}
    Returns: {"query": "...", "top_k": 5, "results": [
        {"location": "Chicago, IL", "headline": "Flash Flood Warning",
         "source_type": "alert", "chunk_text": "...", "similarity": 0.87}
    ]}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    query_text = request.json.get("query", "").strip()
    top_k = int(request.json.get("top_k", 5))

    if not query_text:
        return jsonify({"error": "query parameter is required"}), 400

    # Homework spec clamps weather search to 1-20.
    if top_k < 1 or top_k > 20:
        return jsonify({"error": "top_k must be between 1 and 20"}), 400

    try:
        model = get_embedding_model()
        query_embedding = model.encode([query_text])[0].tolist()
    except ImportError as e:
        logger.error("sentence-transformers not installed")
        return jsonify({
            "error": "Vector search is not available. The sentence-transformers package is not installed.",
            "details": str(e)
        }), 503
    except Exception as e:
        logger.exception("Failed to embed query")
        return jsonify({"error": f"Failed to embed query: {str(e)}"}), 500

    query_vector_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'

    # Defensive: makes an empty/not-yet-created table degrade to results=[]
    # instead of a 500 if /weather/sync + the ingestion script haven't run yet.
    ensure_weather_embeddings_table()

    try:
        results = lakebase.run_query(
            f"""
            SELECT
                we.document_id,
                we.chunk_index,
                we.chunk_text,
                wd.location,
                wd.headline,
                wd.source_type,
                wd.effective_at,
                1 - (we.embedding <=> %s::vector) AS similarity
            FROM {WEATHER_EMBEDDINGS_TABLE_NAME} we
            JOIN {WEATHER_TABLE_NAME} wd ON we.document_id = wd.id
            ORDER BY we.embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (query_vector_str, query_vector_str, top_k)
        )
    except Exception as e:
        logger.exception("Weather vector search query failed")
        return jsonify({"error": f"Vector search failed: {str(e)}"}), 500

    return jsonify({
        "query": query_text,
        "top_k": top_k,
        "results": results
    })


def _upsert_weather_batch(docs: list[dict]) -> int:
    """Upsert a batch of normalized weather documents into weather_documents.

    One INSERT ... ON CONFLICT per row, single commit at the end.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in docs:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_TABLE_NAME} (
                        id, location, source_type, headline, narrative_text,
                        issued_at, effective_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET location = EXCLUDED.location,
                            source_type = EXCLUDED.source_type,
                            headline = EXCLUDED.headline,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        doc["id"],
                        doc["location"],
                        doc["source_type"],
                        doc.get("headline"),
                        doc.get("narrative_text"),
                        doc.get("issued_at"),
                        doc.get("effective_at"),
                        _json.dumps(doc.get("payload", {})),
                    ),
                )
                count += 1
            conn.commit()
    return count


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")
