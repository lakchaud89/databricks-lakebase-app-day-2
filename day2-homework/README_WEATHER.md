# Day 2 Homework: Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

This folder started as a self-contained copy of the class reference app, but the Massive-API
-dependent pieces (`/sync`, `/news/sync`, `/watchlist`, `/search/vector`, `massive_client.py`)
have been stripped out -- this app has no Massive API dependency at all. What's left is a
standalone weather harvest → vectorize → retrieve pipeline (`app.py`, `lakebase.py`,
`weather_client.py`, `templates/index.html`), deployed as its **own Databricks App**, independent
of the class reference app at the repo root. `sql/01`-`04` and
`notebooks/ingest_ticker_news_embeddings.py` are left in place only as reference material from the
original copy -- they're unused by this app and safe to delete.

## Assignment deliverables checklist

| Deliverable | Where |
|---|---|
| `weather_client.py` — NWS API client | [`weather_client.py`](./weather_client.py) |
| `app.py` with `POST /weather/sync` + `POST /weather/search` | [`app.py`](./app.py) |
| DDL/migration for `weather_documents` + `weather_embeddings` | [`sql/05_setup_weather_documents_table.sql`](./sql/05_setup_weather_documents_table.sql), [`sql/06_setup_weather_embeddings_table.sql`](./sql/06_setup_weather_embeddings_table.sql) — mirrored by `ensure_weather_documents_table()` / `ensure_weather_embeddings_table()` in `app.py` (not in `lakebase.py` itself, which stays a generic, table-agnostic connection helper) |
| psycopg2-based embedding ingestion script | [`notebooks/ingest_weather_embeddings.py`](./notebooks/ingest_weather_embeddings.py) |
| This README | you're reading it |

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

**Note on the insert cast**: the assignment suggests casting embeddings with `%s::vector` directly
on `INSERT`. `notebooks/ingest_weather_embeddings.py` instead casts to `%s::double precision[]`
and relies on pgvector's `ASSIGNMENT` cast into the `vector(384)` column (with a defensive
`UPDATE ... SET embedding = embedding::vector` immediately after, as a no-op safety net) — this is
the same pattern the class reference ticker-news notebook uses, and it's the one demonstrably
proven to work against this specific Lakebase instance (see `sql/04_cast_arrays_to_vectors.sql`,
which exists in the original project *because* a direct inline `%s::vector` cast wasn't reliable
here). Functionally equivalent end state (a proper `vector` column), different path to get there.

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

**Status: verified working end-to-end** against a live `weather-db` Lakebase instance — synced 4
locations, embedded the resulting documents, and confirmed `/weather/search` returns real,
semantically-ranked results.

1. **Secrets** (once): run `python setup_secrets.py` from a Databricks notebook to store your
   Lakebase URL under secret scope `database_weather` (kept separate from the class reference
   app's `database`/`database1` scope on purpose, per this assignment). Also grant the app itself
   READ access: in the Databricks Apps UI, add an **App resource** of type Secret pointing at
   scope `database_weather`, key `lakebase-url` — the scope-level ACL from `setup_secrets.py`
   covers interactive/notebook use, but the deployed app runs under its own service principal and
   needs this separate grant.
   - If your Lakebase instance's default connection string points at a database name you don't
     want (e.g. you created a dedicated `weather-db` database inside the instance rather than
     using `databricks_postgres`), set the `LAKEBASE_DATABASE` env var (in `app.yaml` or your
     notebook) to override just the database name in the stored URL, instead of re-storing a new
     secret — see `lakebase.py`'s `_lakebase_url()`.
2. **Schema** (once, in the Lakebase SQL editor):
   ```sql
   -- sql/05_setup_weather_documents_table.sql, then:
   -- sql/06_setup_weather_embeddings_table.sql
   ```
   Run these authenticated as the *same* role your `database_weather` secret uses, not your
   personal Databricks identity — see Troubleshooting below for why that matters. Simplest: skip
   this step entirely and let `/weather/sync` / `/weather/search` create both tables on first call
   (`app.py`'s `ensure_weather_documents_table()` / `ensure_weather_embeddings_table()` run the
   same DDL as these files).
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

## Troubleshooting notes (issues actually hit while building this)

- **`Error: must be owner of table weather_documents`** — happens when the tables were created
  under one Postgres role (e.g. your personal Databricks identity, if you ran `sql/05`/`sql/06`
  from a browser SQL editor that authenticates as you) but the app connects as a *different* role
  (whatever's embedded in the `database_weather` secret's connection URL). Postgres requires you
  to be the table owner to run `CREATE INDEX`/`ALTER TABLE`, which `ensure_weather_documents_table()`
  does on every request. Fix: either `DROP TABLE` and let the app recreate them under its own
  connection (simplest), or `ALTER TABLE weather_documents OWNER TO <app_role>;` /
  `ALTER TABLE weather_embeddings OWNER TO <app_role>;`.
- **`ModuleNotFoundError: No module named 'sqlalchemy'`** (and similarly for
  `sentence-transformers`) when running `notebooks/ingest_weather_embeddings.py` directly from a
  notebook/cluster — `requirements.txt` is only auto-installed by the *deployed Databricks App*,
  not by whatever compute you attach an ad-hoc notebook to. `lakebase.py` now imports `sqlalchemy`
  lazily inside `get_engine()` (nothing in this pipeline calls it) so a bare `import lakebase`
  doesn't need it installed at all; `sentence-transformers` is a genuine runtime dependency of the
  ingestion script and the search endpoint, so it must actually be installed wherever you run
  either one (`%pip install -r requirements.txt` in the notebook if it's missing).

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
