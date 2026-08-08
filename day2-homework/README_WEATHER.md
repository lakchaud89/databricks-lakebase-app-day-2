# Day 2 Homework: Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

This folder started as a self-contained copy of the class reference app, but the Massive-API
-dependent pieces (`/sync`, `/news/sync`, `/watchlist`, `/search/vector`, `massive_client.py`)
have been stripped out -- this app has no Massive API dependency at all. What's left is a
standalone weather harvest → vectorize → retrieve pipeline (`app.py`, `lakebase.py`,
`weather_client.py`, `templates/index.html`), deployed as its **own Databricks App**, independent
of the class reference app at the repo root. `sql/01`-`04` and
`notebooks/ingest_ticker_news_embeddings.py` are left in place only as reference material from the
original copy -- they're unused by this app and safe to delete.

## Data source: National Weather Service API (`api.weather.gov`)

Chosen per the assignment's recommendation:
- Free, no API key required.
- Generous rate limits for a class project.
- Returns rich unstructured narrative text well-suited to embedding: `description`/`instruction`
  free-text fields on active alerts, and `detailedForecast` narrative strings on forecast periods.

NWS does require every client to send a descriptive `User-Agent` header identifying the
application and a contact method — generic/default `User-Agent`s get throttled or blocked. Set
`WEATHER_USER_AGENT` in `app.yaml` (already set) or your local `.env`.

## Schema

### `weather_documents` (raw harvested documents — `sql/05_setup_weather_documents_table.sql`)

| column          | type        | notes                                                             |
|-----------------|-------------|--------------------------------------------------------------------|
| `id`            | TEXT PK     | `alert:<NWS alert URN>` or `forecast:<office>:<x>,<y>:<period#>`   |
| `location`      | TEXT        | resolved label, e.g. `"Chicago, IL"`                              |
| `source_type`   | TEXT        | `'alert'` or `'forecast'`                                          |
| `headline`      | TEXT        | e.g. `"Flash Flood Warning"` or `"Tonight"`                        |
| `narrative_text`| TEXT        | the free-text body that gets embedded                              |
| `issued_at`     | TIMESTAMPTZ | alert `sent` time, or forecast `properties.updated`                |
| `effective_at`  | TIMESTAMPTZ | alert `effective` time, or forecast period `startTime`             |
| `payload`       | JSONB       | raw NWS JSON, for provenance                                       |
| `synced_at`     | TIMESTAMPTZ | last upsert time                                                    |

### `weather_embeddings` (chunk-level vectors — `sql/06_setup_weather_embeddings_table.sql`)

| column        | type          | notes                                             |
|---------------|---------------|----------------------------------------------------|
| `id`          | TEXT PK       | `<document_id>_<chunk_index>`                      |
| `document_id` | TEXT FK       | → `weather_documents.id`                           |
| `chunk_index` | INT           |                                                      |
| `chunk_text`  | TEXT          | the embedded chunk                                  |
| `embedding`   | VECTOR(384)   | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim   |
| `model_name`  | TEXT          |                                                      |
| `created_at`  | TIMESTAMPTZ   |                                                      |

Uses the **same embedding model** as the ticker-news pipeline (384-dim), so both pipelines stay
queryable with the same pgvector distance-operator conventions (`<=>`, cosine).

## Chunking parameters

`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` — same sliding-window values as the news pipeline, for
consistency. Most NWS narrative text (a single forecast period, or a short alert) fits well under
800 characters and never actually gets split; chunking mostly matters for longer alerts where
`description` + `instruction` are concatenated.

## Location resolution — known limitation

There's no geocoding API in scope for this assignment, so `weather_client.resolve_location()`
accepts:
- a small built-in city lookup (`CITY_COORDS` in `weather_client.py` — Sacramento, New York,
  Chicago, Miami, Seattle, Austin, Denver),
- a `"lat,lon"` string,
- or a `{"lat": .., "lon": ..}` dict.

Anything else raises `ValueError` and that location is skipped (not silently dropped — a warning
is logged and the location is excluded from the `/weather/sync` response's `locations` list).
Extending `CITY_COORDS` or wiring in a real geocoder would remove this limitation.

## Running the pipeline end-to-end

1. **Secrets** (once): run `python setup_secrets.py` from a Databricks notebook to store your
   Lakebase URL under secret scope `database_weather` (kept separate from the class reference
   app's `database`/`database1` scope on purpose, per this assignment).
2. **Schema** (once, in the Lakebase SQL editor):
   ```sql
   -- sql/05_setup_weather_documents_table.sql, then:
   -- sql/06_setup_weather_embeddings_table.sql
   ```
3. **Harvest**:
   ```bash
   curl -X POST <APP_URL>/weather/sync \
     -H "Content-Type: application/json" \
     -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'
   ```
   Returns `{"synced": N, "locations": [...]}`. Defaults to `WEATHER_LOCATIONS` (semicolon
   separated in `app.yaml`/`.env`, since individual locations contain a comma) if no body is sent.
4. **Vectorize**:
   ```bash
   python notebooks/ingest_weather_embeddings.py
   ```
   Idempotent — only embeds documents that don't already have rows in `weather_embeddings`, so
   it's safe to re-run after every sync.
5. **Retrieve**:
   ```bash
   curl -X POST <APP_URL>/weather/search \
     -H "Content-Type: application/json" \
     -d '{"query": "risk of flooding near rivers", "top_k": 5}'
   ```
   Returns `{"query": "...", "top_k": 5, "results": [{"location", "headline", "source_type",
   "chunk_text", "similarity", ...}]}`. `top_k` is clamped to 1–20. Also usable from the UI
   at `<APP_URL>/` (the "🌦️ Search Weather" card).

## Known limitations / things to improve given more time

- **Forecast documents overwrite in place.** NWS doesn't give forecast periods a stable id of
  their own, so each location's ~14 rolling forecast documents (`forecast:<office>:<x>,<y>:<n>`)
  get upserted on every sync — only the latest issuance is retained, no historical snapshots.
- **US-only.** NWS alerts/forecasts only cover the United States and territories.
- **No enforced NWS rate limiting.** The client doesn't throttle requests client-side; for a
  larger `/weather/sync` location list, add the same rate-limit-and-sleep pattern the ticker-news
  ingestion notebook uses for the Massive API.
- **Location resolution is a hardcoded lookup**, not a real geocoder (see above).
- **Stretch goals not implemented**: `GET /weather/search` with an LLM-generated RAG summary,
  `source_type` filtering, a scheduled Databricks Job for periodic re-sync, and an HNSW
  benchmark comparing indexed vs. unindexed query latency.
