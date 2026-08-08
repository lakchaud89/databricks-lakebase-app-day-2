"""
Databricks App boilerplate:
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls data from the Massive API via massive_client.py and syncs it into Lakebase
- Pulls unstructured weather text from the NWS API via weather_client.py, embeds it,
  and serves a semantic search endpoint over it (Day 2 homework)

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os
import re

import requests
from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request

import lakebase
from massive_client import MassiveClient
from weather_client import WeatherClient, resolve_location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("massive-app")

app = Flask(__name__)
_w = WorkspaceClient()

# Lazy-load the embedding model (only when vector search is used)
_embedding_model = None


def get_embedding_model():
    """Lazy-load the sentence transformer model for embedding queries.

    Loaded once at first use (not per-request) and reused for both the news
    and weather search endpoints, since both pipelines use the same model.
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


TABLE_NAME = os.environ.get("MASSIVE_TABLE_NAME", "massive_records")
WATCHLIST_TABLE_NAME = os.environ.get("WATCHLIST_TABLE_NAME", "watchlist")
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
CHUNK_EMBEDDINGS_TABLE_NAME = os.environ.get("CHUNK_EMBEDDINGS_TABLE_NAME", "ticker_news_chunk_embeddings")
WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
WEATHER_EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Tickers to fetch news for by default (comma-separated), e.g. "AAPL,MSFT,GOOGL"
DEFAULT_NEWS_TICKERS = [
    t.strip().upper()
    for t in os.environ.get("NEWS_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
    if t.strip()
]

# Locations to sync weather for by default (comma-separated "City, ST"), e.g.
# "Chicago, IL,Austin, TX" -- NOTE: since locations themselves contain a comma,
# use a semicolon to separate multiple locations in the env var.
DEFAULT_WEATHER_LOCATIONS = [
    loc.strip()
    for loc in os.environ.get("WEATHER_LOCATIONS", "Sacramento, CA;Chicago, IL;New York, NY").split(";")
    if loc.strip()
]

# Basic stock ticker shape check: 1-10 uppercase letters, with an optional
# ".X" or ".XX" share-class suffix (e.g. "BRK.B"). This rejects obviously
# malformed input before we even call the Massive API.
_TICKER_RE = re.compile(r"^[A-Z]{1,10}(\.[A-Z]{1,2})?$")


def ensure_table():
    """Create the destination table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def ensure_watchlist_table():
    """Create the watchlist table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WATCHLIST_TABLE_NAME} (
            symbol TEXT NOT NULL,
            email TEXT NOT NULL,
            latest_price NUMERIC,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, email)
        )
        """
    )


def ensure_news_table():
    """
    Create the raw ticker-news documents table in Lakebase if it doesn't
    exist yet. This is the RAW document store the ingestion notebook
    (notebooks/ingest_ticker_news_embeddings.py) reads from to compute
    vector embeddings into a separate `<NEWS_TABLE_NAME>_embeddings` table.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {NEWS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            author TEXT,
            article_url TEXT,
            publisher_name TEXT,
            keywords JSONB,
            sentiment TEXT,
            sentiment_reasoning TEXT,
            published_utc TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{NEWS_TABLE_NAME}_ticker "
        f"ON {NEWS_TABLE_NAME} (ticker)"
    )


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


def _current_user_email() -> str:
    """
    Resolve the current user's email so the watchlist can be personalized.

    Databricks Apps inject the logged-in user's identity via the
    X-Forwarded-Email header on every request. Fall back to the Databricks
    SDK's current_user API for local development where that header isn't set.
    """
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    return _w.current_user.me().user_name


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
    """Simple UI to submit a list of stock symbols to sync from Massive,
    plus semantic search over synced news and weather documents."""
    return render_template("index.html")


@app.route("/records")
def list_records():
    """Read records already synced into Lakebase."""
    limit = int(request.args.get("limit", 100))
    rows = lakebase.run_query(
        f"SELECT id, payload, synced_at FROM {TABLE_NAME} ORDER BY synced_at DESC LIMIT %s",
        (limit,),
    )
    return jsonify(rows)


@app.route("/sync", methods=["POST"])
def sync_from_massive():
    """
    Pull data from the Massive API (paginated, potentially huge dataset) and
    upsert it into Lakebase in batches.
    """
    ensure_table()
    client = MassiveClient()

    path = request.json.get("path", "/records") if request.is_json else "/records"
    batch_size = int(request.args.get("batch_size", 500))

    batch = []
    total = 0
    for item in client.paginated_get(path):
        batch.append(item)
        if len(batch) >= batch_size:
            total += _upsert_batch(batch)
            batch = []

    if batch:
        total += _upsert_batch(batch)

    return jsonify({"synced": total})


@app.route("/news/sync", methods=["POST"])
def sync_news_from_massive():
    """
    Pull recent news articles for a set of tickers from Massive (ONE API
    call per ticker, via MassiveClient.get_news) and upsert them into the
    ticker_news_documents table in Lakebase.

    Body (optional JSON): {"tickers": ["AAPL", "MSFT"], "limit": 50}
    Defaults to DEFAULT_NEWS_TICKERS when no tickers are supplied.
    """
    ensure_news_table()
    client = MassiveClient()

    body = request.json if request.is_json else {}
    tickers = body.get("tickers") or DEFAULT_NEWS_TICKERS
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    limit = int(body.get("limit", 50))

    total = 0
    for ticker in tickers:
        if not _TICKER_RE.match(ticker):
            continue
        articles = client.get_news(ticker, limit=limit)
        total += _upsert_news_batch(ticker, articles)

    return jsonify({"synced": total, "tickers": tickers})


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Resolve each requested location to an NWS grid point, fetch active
    alerts + forecast periods for it (via WeatherClient.get_documents_for_point),
    and upsert the normalized documents into weather_documents.

    Body (optional JSON): {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    Defaults to DEFAULT_WEATHER_LOCATIONS when no locations are supplied.

    Mirrors /news/sync: a single bad location (unresolvable, or an NWS API
    error for that point) is skipped rather than aborting the whole batch.
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


@app.route("/watchlist", methods=["GET"])
def get_watchlist():
    """Return the current user's watchlist symbols, with their last known price."""
    ensure_watchlist_table()
    email = _current_user_email()
    rows = lakebase.run_query(
        f"SELECT symbol, email, latest_price, updated_at FROM {WATCHLIST_TABLE_NAME} "
        f"WHERE email = %s ORDER BY symbol ASC",
        (email,),
    )
    return jsonify(rows)


@app.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    """
    Fetch the latest price for a single stock symbol from Massive using
    exactly ONE API call (see MassiveClient.get_latest_price), then add/
    update that symbol on the watchlist in Lakebase.
    """
    ensure_watchlist_table()

    if request.is_json:
        symbol = request.json.get("symbol", "")
    else:
        symbol = request.form.get("symbol", "")

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""

    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    client = MassiveClient()
    try:
        data = client.get_latest_price(symbol)  # <-- single API call, latest price only
    except requests.HTTPError:
        # Massive returns a 404/4xx for tickers it doesn't recognize.
        return jsonify({"error": f"Unknown ticker symbol: {symbol}"}), 400

    price = _extract_latest_price(data)
    if price is None:
        # No usable price in the response (e.g. delisted/invalid ticker
        # that still 200s with an empty result set) - don't add it.
        return jsonify({"error": f"No price data available for ticker: {symbol}"}), 400

    email = _current_user_email()

    lakebase.run_write(
        f"""
        INSERT INTO {WATCHLIST_TABLE_NAME} (symbol, email, latest_price, updated_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (symbol, email) DO UPDATE
            SET latest_price = EXCLUDED.latest_price,
                updated_at = EXCLUDED.updated_at
        """,
        (symbol, email, price),
    )

    return jsonify({"symbol": symbol, "email": email, "latest_price": price})


@app.route("/watchlist/<symbol>", methods=["DELETE"])
def delete_from_watchlist(symbol: str):
    """Remove a single symbol from the current user's watchlist."""
    ensure_watchlist_table()

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    email = _current_user_email()
    deleted = lakebase.run_write(
        f"DELETE FROM {WATCHLIST_TABLE_NAME} WHERE symbol = %s AND email = %s",
        (symbol, email),
    )

    if not deleted:
        return jsonify({"error": f"{symbol} is not on your watchlist"}), 404

    return jsonify({"symbol": symbol, "email": email, "deleted": True})


@app.route("/search/vector", methods=["POST"])
def vector_search():
    """
    Semantic search endpoint: takes a natural language query, embeds it using
    the same model that created the chunk embeddings, and returns the top K
    most similar news chunks via pgvector's cosine similarity (<=>).

    Body: {"query": "What are the latest trends in AI?", "top_k": 5}
    Returns: [{"article_id": "...", "ticker": "AAPL", "chunk_index": 0,
              "chunk_text": "...", "title": "...", "similarity": 0.87, ...}]
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    query_text = request.json.get("query", "").strip()
    top_k = int(request.json.get("top_k", 5))

    if not query_text:
        return jsonify({"error": "query parameter is required"}), 400

    if top_k < 1 or top_k > 50:
        return jsonify({"error": "top_k must be between 1 and 50"}), 400

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

    try:
        results = lakebase.run_query(
            f"""
            SELECT
                ce.article_id,
                ce.ticker,
                ce.chunk_index,
                ce.chunk_text,
                n.title,
                n.article_url,
                n.published_utc,
                n.sentiment,
                1 - (ce.embedding <=> %s::vector) AS similarity
            FROM {CHUNK_EMBEDDINGS_TABLE_NAME} ce
            LEFT JOIN {NEWS_TABLE_NAME} n ON ce.article_id = n.id
            ORDER BY ce.embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (query_vector_str, query_vector_str, top_k)
        )
    except Exception as e:
        logger.exception("Vector search query failed")
        return jsonify({"error": f"Vector search failed: {str(e)}"}), 500

    return jsonify({
        "query": query_text,
        "top_k": top_k,
        "results": results
    })


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

    # Homework spec clamps weather search to 1-20 (narrower than the
    # existing /search/vector's 1-50) -- intentional, don't "fix" this back.
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


def _extract_latest_price(data: dict) -> float | None:
    """Pull the trade price out of the Massive 'previous close' response shape.

    The /v2/aggs/ticker/{symbol}/prev endpoint returns "results" as a LIST
    containing a single aggregate bar (not a dict), e.g.:
        {"status": "OK", "resultsCount": 1, "results": [{"c": 148.845, ...}]}
    Previously this code treated "results" as a dict, so isinstance(results, dict)
    was always False for this endpoint's real shape and the price silently
    resolved to None. Unwrap the list here, and check "status"/"resultsCount"
    so invalid tickers (empty results) are detected instead of "succeeding"
    with a null price.

    Adjust the key lookup here if the real Massive API returns a different
    field name for the traded/close price.
    """
    if not isinstance(data, dict):
        return None
    if data.get("status") not in (None, "OK") or data.get("resultsCount") == 0:
        return None
    results = data.get("results", data)
    if isinstance(results, list):
        results = results[0] if results else None
    if isinstance(results, dict):
        for key in ("c", "p", "price", "last_price", "vw"):
            if key in results:
                return results[key]
    return None


def _upsert_batch(items: list[dict]) -> int:
    """Upsert a batch of Massive API items into Lakebase, one statement per row.

    For very large batches, consider psycopg2.extras.execute_values for
    higher throughput instead of per-row execute calls.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (id, payload, synced_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (str(item.get("id")), _json.dumps(item)),
                )
                count += 1
            conn.commit()
    return count


def _upsert_news_batch(ticker: str, articles: list[dict]) -> int:
    """Upsert news articles for a single ticker into the news documents table.

    Flattens the top-level "insights" sentiment entry that matches this
    ticker (if present) into its own columns so the ingestion script can read
    plain text columns instead of parsing JSONB for the common case.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for article in articles:
                sentiment = None
                sentiment_reasoning = None
                for insight in article.get("insights", []) or []:
                    if insight.get("ticker") == ticker:
                        sentiment = insight.get("sentiment")
                        sentiment_reasoning = insight.get("sentiment_reasoning")
                        break

                publisher = article.get("publisher") or {}
                cur.execute(
                    f"""
                    INSERT INTO {NEWS_TABLE_NAME} (
                        id, ticker, title, description, author, article_url,
                        publisher_name, keywords, sentiment, sentiment_reasoning,
                        published_utc, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET ticker = EXCLUDED.ticker,
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            author = EXCLUDED.author,
                            article_url = EXCLUDED.article_url,
                            publisher_name = EXCLUDED.publisher_name,
                            keywords = EXCLUDED.keywords,
                            sentiment = EXCLUDED.sentiment,
                            sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                            published_utc = EXCLUDED.published_utc,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        str(article.get("id")),
                        ticker,
                        article.get("title", ""),
                        article.get("description"),
                        article.get("author"),
                        article.get("article_url"),
                        publisher.get("name"),
                        _json.dumps(article.get("keywords", [])),
                        sentiment,
                        sentiment_reasoning,
                        article.get("published_utc"),
                        _json.dumps(article),
                    ),
                )
                count += 1
            conn.commit()
    return count


def _upsert_weather_batch(docs: list[dict]) -> int:
    """Upsert a batch of normalized weather documents into weather_documents.

    Same shape as _upsert_news_batch: one INSERT ... ON CONFLICT per row,
    single commit at the end.
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
