# Athena

A self-contained, open-source document ingestion and retrieval engine. It chunks and embeds
UTF-8 text into PostgreSQL/pgvector with LlamaIndex, and answers hybrid
(vector + full-text) search queries with ranked chunks, document metadata, and
citations.

Chunking uses a bundled regex sentence/line splitter and needs no downloaded
NLTK corpora, which keeps container and offline startup deterministic.

The engine has no dependency on any other repository: it needs only the
environment variables below and a PostgreSQL database with the `vector`
extension available. The directory can be moved into its own repository as-is.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull request
guidance. Athena is released under the [MIT License](LICENSE).

## Contract

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health/live` | none | Process liveness |
| `GET` | `/health/ready` | none | Database reachable; `503` when not |
| `POST` | `/v1/documents/text` | bearer | Idempotently ingest or update one text document |
| `POST` | `/v1/documents/file` | bearer | Upload one bounded file; normalizes and indexes it |
| `GET` | `/v1/documents?collection_id=…` | bearer | List the sources in one collection, with status |
| `GET` | `/v1/documents/source?collection_id=…&external_id=…` | bearer | Metadata for one retained original |
| `GET` | `/v1/documents/source/content?collection_id=…&external_id=…[&variant=preview]` | bearer | Stream those bytes; supports `Range` and `ETag` |
| `GET` | `/v1/jobs/{job_id}` | bearer | `accepted` \| `processing` \| `completed` \| `failed` |
| `POST` | `/v1/search` | bearer | Search one or more named collections |

Every `/v1/*` request needs `Authorization: Bearer $KNOWLEDGE_API_TOKEN`. Health
probes stay unauthenticated so orchestrators can use them.

### Collections are the isolation boundary

`collection_id` is mandatory on ingest and on search. A search only ever returns
chunks whose `collection_id` is in the requested list: the service both sends a
metadata filter to the vector backend and re-checks every retrieved chunk before
building the response, so isolation does not depend on the backend honouring
filters. Ingesting or re-indexing one document never clears, re-embeds, or
otherwise touches another collection.

### Idempotency

A document is identified by `(collection_id, external_id)`. Its `document_id` is
a deterministic UUIDv5 of that pair. Ingest computes a SHA-256 checksum of the
text unless the caller supplies `checksum` (useful when an upstream parser
already knows the source version). If the stored checksum matches, the job
completes with `unchanged: true` and nothing is re-embedded. Send
`force_reindex: true` to re-embed anyway.

When the text has changed, the document is first marked as needing re-indexing,
then its previous chunks are deleted by `document_id` before the new ones are
written. If deletion or embedding fails, the zero-chunk marker prevents a later
retry from incorrectly treating the empty index as unchanged.

### Ingest example

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/documents/text \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "collection_id": "example-collection",
        "external_id": "maintenance-notes",
        "title": "Maintenance notes",
        "source_type": "note",
        "source_uri": "https://example.invalid/maintenance-notes",
        "version": "rev-1",
        "text": "Inspect the equipment before service ..."
      }'
# {"job_id":"…","status":"accepted","collection_id":"example-collection","external_id":"maintenance-notes"}

curl -sS http://127.0.0.1:8080/v1/jobs/$JOB_ID \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN"
# {"job_id":"…","status":"completed","chunk_count":612,"unchanged":false,…}
```

Optional ingest fields: `page`, `section`, `version`, `checksum`, `force_reindex`,
and a free-form `metadata` object. `metadata` may not use the service-owned keys
(`collection_id`, `document_id`, `external_id`, `title`, `source_type`,
`source_uri`, `checksum`, `version`, `page`, `section`, `updated_at`,
`updated_at_ts`, `embedding_model`, `embedding_dim`); the request is rejected
with `422` if it does.

### File upload

`POST /v1/documents/file` takes one `multipart/form-data` request with a `file`
part plus `collection_id` and an optional `title`. It answers `202` as soon as
the job is durable; normalization, conversion, and embedding then run in the
background, so a slow PDF never holds the connection open.

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/documents/file \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  -F 'collection_id=example-collection' \
  -F 'title=Field notes' \
  -F 'file=@notes.md;type=text/markdown'
# {"job_id":"…","status":"accepted","collection_id":"example-collection","external_id":"file-9f2c…"}
```

| Format | Extension | Normalized by |
| --- | --- | --- |
| Plain text | `.txt` | this service, no converter needed |
| Markdown | `.md`, `.markdown` | this service, no converter needed |
| PDF | `.pdf` | Docling |
| Word, PowerPoint, Excel | `.docx`, `.pptx`, `.xlsx` | Docling |
| HTML | `.html` | Docling |

The extension decides the format. A browser-declared content type is checked
against that extension but a generic `application/octet-stream` is accepted,
because browsers disagree about Office types.

`external_id` is derived from the file's content and extension, so re-uploading
the same file completes with `unchanged: true` and re-embeds nothing. Uploading
the same bytes under a different title keeps the original title, because the
document is recognized as unchanged.

Everything a caller controls is bounded: the collection id, the file name
(`KNOWLEDGE_MAX_FILENAME_CHARACTERS`), the byte count
(`KNOWLEDGE_MAX_UPLOAD_BYTES`, read incrementally so a lying `Content-Length`
cannot exhaust memory), the converted text (`KNOWLEDGE_MAX_DOCUMENT_BYTES`), and
the job `detail`. The file name is stored as display metadata only and is never
used to build a path.

Rejections are synchronous: `415` for an unsupported or contradictory type,
`422` for an empty file or an invalid `collection_id`, `413` for an oversized
upload. Anything discovered after acceptance — a non-UTF-8 text file, a scanned
PDF with no text layer, an unreachable converter — fails that one job with a
readable `detail` and leaves every other source indexed and searchable.

### Document conversion

Text and Markdown are decoded in-process. Everything else goes through a
converter behind an adapter (`app/parsing/normalize.py`), so an Azure or Mistral
OCR client can replace Docling later without the ingest path changing.

The bundled adapter prefers docling-serve's synchronous chunking endpoint:
multipart `POST {DOCLING_BASE_URL}/v1/chunk/hierarchical/file` with the file
under `files`. That reply carries one entry per chunk with `text`, `page_numbers`,
and `headings`, which is why it is used instead of Markdown alone — Markdown
flattens away the page a passage came from, and a citation that cannot name a
page cannot open one. The first page a chunk appears on becomes its `page`, and
the most specific heading above it becomes its `section`; both travel onto the
indexed chunks and therefore onto search citations.

When that route is absent (an older or trimmed deployment answers `404`, `405`,
or `501`) the adapter latches onto `POST {DOCLING_BASE_URL}/v1/convert/file` with
`to_formats=md` and reads `document.md_content`. That path has no provenance, so
nothing claims a page. A format with no stable pagination — Markdown, plain text —
never reports one either.

Both requests have a bounded timeout and are size-capped before parsing. Docling
is reached from this service only and is never exposed to a browser.

`DOCLING_BASE_URL` is optional. When it is unset the service starts normally,
logs that conversion is disabled, and fails only the uploads that need it:

```json
{ "status": "failed", "detail": "PDF uploads need a document converter, which is not configured on this deployment. Plain text and Markdown uploads still work." }
```

### Listing sources

`GET /v1/documents?collection_id=example-collection` returns every source in a collection,
needs, including uploads that have no document row yet:

```json
{
  "collection_id": "example-collection",
  "items": [
    {
      "external_id": "file-9f2c…",
      "title": "Field notes",
      "source_type": "markdown",
      "status": "ready",
      "chunk_count": 12,
      "detail": null,
      "filename": "notes.md",
      "media_type": "text/markdown",
      "viewable": true,
      "byte_size": 4096,
      "page_count": null,
      "preview_available": false,
      "created_at": "2026-08-16T12:00:00Z",
      "updated_at": "2026-08-16T12:00:04Z"
    }
  ],
  "truncated": false
}
```

`status` collapses the document row and its latest job into the three states a
person can act on. A document with chunks is `ready` — including while a
re-index runs, because its existing chunks stay searchable until the new ones
replace them. An accepted or processing job with no searchable chunks is
`processing`. Everything else is `failed`, carrying the job's `detail`. The list
is bounded at 200 entries, most recently updated first, and `truncated` says
when that bound was hit.

`viewable` says whether the original upload was retained and can therefore be
fetched back. It is `false` for a document ingested through
`POST /v1/documents/text`, which has no file, and for every upload indexed before
originals were kept. `page_count` is the highest page that produced indexed text,
not a claim about the document's physical length, and is `null` for a format
without pagination.

### Retained originals

`POST /v1/documents/file` writes the uploaded bytes to durable storage and
records them in `source_objects` **before** the background job starts, so an
accepted upload can always be fetched back — including one whose conversion later
fails. Bytes are written first and the row second: a crash between the two leaves
an unreferenced file that the next upload of the same content overwrites, whereas
the other order would leave a row pointing at nothing. If storage is unavailable
the upload is refused with `503` and no job is created.

```bash
curl -sS -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  "http://127.0.0.1:8080/v1/documents/source?collection_id=example-collection&external_id=file-9f2c…"
```

```json
{
  "collection_id": "example-collection",
  "external_id": "file-9f2c…",
  "title": "Field notes",
  "source_type": "pdf",
  "status": "ready",
  "filename": "service manual.pdf",
  "media_type": "application/pdf",
  "byte_size": 2411008,
  "checksum": "…",
  "page_count": 212,
  "preview_available": true,
  "preview_bytes": 481203,
  "chunk_count": 640
}
```

`GET /v1/documents/source/content` streams the object. `variant=original` is the
uploaded file; `variant=preview` is the normalized text that was indexed, which
exists for formats that needed conversion and is what a viewer shows when the
original cannot be rendered safely. Both identifiers are bounded and validated,
and the lookup is keyed by the pair, so a source in another collection is a `404`
rather than a leak.

Response rules, all enforced here rather than by the caller:

- `Content-Type` is downgraded to `application/octet-stream` unless the stored
  type is on the inline allowlist (`application/pdf`, `text/plain`,
  `text/markdown`). **`text/html` is deliberately absent**: uploaded HTML always
  downloads, and its extracted text is what a viewer shows instead.
- `Content-Disposition` is `inline` only for that allowlist, `attachment`
  otherwise. The quoted form keeps printable ASCII; the exact name travels in the
  percent-encoded RFC 5987 `filename*` form, which has no delimiter to escape.
- `X-Content-Type-Options: nosniff` on every response.
- `Accept-Ranges: bytes`, with `206` and `Content-Range` for a single satisfiable
  range and `416` for one outside the object. A multi-range request is answered
  in full rather than parsed.
- `Cache-Control: private, no-cache` and an `ETag` over **that representation's**
  bytes. A browser's PDF viewer only moves to another page on a real navigation, so
  a viewer reloads the frame per page jump; the validator turns those reloads into
  `304`s instead of re-sending the document. `private` keeps it out of shared
  caches. The preview has its own stored checksum rather than borrowing the
  original's: converting one file again can produce different extracted text, and a
  browser holding the old preview must be sent the new text, not a `304`. Rows
  written before that checksum existed fall back to the preview's byte count and
  gain a real hash the next time they are reprocessed.

A row whose file has gone missing answers `404` before the response body starts,
rather than streaming a truncated document.

### Search example

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/search \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "maintenance procedure", "collection_ids": ["example-collection"], "top_k": 4}'
```

```json
{
  "items": [
    {
      "text": "…",
      "score": 0.81,
      "collection_id": "example-collection",
      "document_id": "…",
      "external_id": "file-9f2c…",
      "title": "Service manual",
      "source_type": "pdf",
      "source_uri": "upload://service-manual.pdf",
      "page": null,
      "section": null,
      "chunk_id": "…",
      "updated_at": "2026-08-16T12:00:00Z",
      "metadata": {},
      "citations": [{ "label": "…", "source_uri": "…", "locator": "chunk:…" }]
    }
  ],
  "warnings": [],
  "stats": { "retrieved": 8, "returned": 4 }
}
```

Optional `filters`: `source_type` (list), `external_id` (inclusion list),
`exclude_external_id` (exclusion list), and `updated_after` (ISO-8601).
`warnings` contains `no_results` when nothing matched.

## Configuration

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `KNOWLEDGE_DATABASE_URL` | yes | – | `postgres://` or `postgresql://` DSN |
| `KNOWLEDGE_API_TOKEN` | yes | – | Bearer token for `/v1/*`; at least 16 characters |
| `KNOWLEDGE_EMBEDDING_BASE_URL` | yes | – | OpenAI-compatible base, e.g. `https://<resource>.cognitiveservices.azure.com/openai/v1` |
| `KNOWLEDGE_EMBEDDING_API_KEY` | yes | – | Key for that endpoint |
| `KNOWLEDGE_EMBEDDING_MODEL` | yes | – | Model name, or the Azure embedding **deployment** name |
| `KNOWLEDGE_EMBEDDING_DIMENSION` | yes | – | Must match the model (1536 or 3072 for `text-embedding-3-*`) |
| `KNOWLEDGE_EMBEDDING_BATCH_SIZE` | no | `64` | Texts per embedding request |
| `KNOWLEDGE_CHUNK_SIZE` | no | `800` | Sentence-splitter chunk size |
| `KNOWLEDGE_CHUNK_OVERLAP` | no | `120` | Must be smaller than the chunk size |
| `KNOWLEDGE_DEFAULT_TOP_K` | no | `8` | Used when a search omits `top_k` |
| `KNOWLEDGE_MAX_TOP_K` | no | `50` | Upper bound applied to any `top_k` |
| `KNOWLEDGE_RETRIEVAL_MODE` | no | `hybrid` | `hybrid` or `vector` |
| `KNOWLEDGE_DB_SCHEMA` | no | `knowledge` | Lowercase SQL identifier |
| `KNOWLEDGE_MAX_DOCUMENT_BYTES` | no | `8000000` | Per-document UTF-8 size limit, also bounds converted text |
| `KNOWLEDGE_MAX_UPLOAD_BYTES` | no | `20971520` | Largest accepted upload; configurable from 1 KiB through the 20 MiB deployment cap |
| `KNOWLEDGE_MAX_FILENAME_CHARACTERS` | no | `255` | Longest stored file name |
| `KNOWLEDGE_SOURCE_STORAGE_DIR` | no | `/var/lib/athena/sources` | Absolute path for retained originals; must be a writable durable mount |
| `DOCLING_BASE_URL` | no | – | docling-serve base, e.g. `http://docling:5001`. Unset disables PDF/Office uploads |
| `DOCLING_TIMEOUT_SECONDS` | no | `660` | Per-conversion client timeout; keep above docling-serve's `DOCLING_SERVE_MAX_SYNC_WAIT` |
| `KNOWLEDGE_LOG_LEVEL` | no | `INFO` | Standard logging level |

Configuration is validated at startup. Errors name the variable and never echo
its value.

## Storage and startup

Startup creates the schema if needed: `collections`, `documents`, `ingest_jobs`,
`source_objects`, `embedding_state`, plus the LlamaIndex vector table
(`data_knowledge_vectors`). Schema evolution is idempotent DDL — `CREATE TABLE IF
NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS` — so an
existing database gains `source_objects` on the next start with no migration step
and no downtime. Existing documents simply have no row there and report
`viewable: false`.

Startup also creates and probes `KNOWLEDGE_SOURCE_STORAGE_DIR` and touches the
vector store, so a read-only volume, a missing `vector` extension, or a
permissions problem fails startup instead of the first ingest.

### Source-object storage

Retained originals sit behind a small storage boundary (`app/storage/base.py`):
put, stat, read a byte range, delete. The bundled backend writes one file per
object under `KNOWLEDGE_SOURCE_STORAGE_DIR` (`app/storage/local.py`), atomically —
`mkstemp` in the target's own directory, `fsync`, then `os.replace` — so a reader
never sees a partial object and a crash mid-write leaves either the old object or
none. The temporary name comes from `mkstemp` rather than from the key, so
concurrent writers of one key each rename their own file instead of racing over a
shared one. Blocking file work runs in a worker thread, so a large upload or a
range read never stalls the event loop.

Keys are built by the service from the document UUID, which is itself derived
from `(collection_id, external_id)`: `ab/abcdef…/original` and `…/preview`. **No
caller-supplied name, path, or file name ever reaches a backend.** Because the
key is stable, re-uploading identical content replaces one object rather than
accumulating copies. Swapping in Azure Blob or S3 later means a new class
implementing that contract; ingestion and both HTTP contracts stay as they are.

Job and document status live in PostgreSQL, so a restart never loses them. Jobs
still marked `accepted` or `processing` when the process died are reconciled to
`failed` with an explicit "interrupted by a service restart" detail.

`embedding_state` records the model and dimension that built the index. If the
configured model or dimension later differs, the service refuses to start rather
than silently mixing vector spaces or wiping every collection. Point
`KNOWLEDGE_DB_SCHEMA` at a fresh schema, or drop the old one and re-ingest.

Above 2000 dimensions pgvector cannot build an HNSW index; the table is then
created without an ANN index and searches fall back to an exact scan.

## CI/CD

GitHub Actions runs the full test suite and builds the Docker image for pull
requests and pushes to `main`. Pushes to `main` and version tags (`v*`) publish
the image to `ghcr.io/thehapyone/athena`; the `latest` tag follows `main`.

## Run locally

```bash
docker run --rm -d -p 55432:5432 -e POSTGRES_PASSWORD=postgres \
  --name knowledge-db pgvector/pgvector:pg17

python -m venv .venv && .venv/bin/pip install -e ".[dev]"

KNOWLEDGE_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres \
KNOWLEDGE_API_TOKEN=$(openssl rand -hex 24) \
KNOWLEDGE_EMBEDDING_BASE_URL=https://<resource>.cognitiveservices.azure.com/openai/v1 \
KNOWLEDGE_EMBEDDING_API_KEY=<key> \
KNOWLEDGE_EMBEDDING_MODEL=text-embedding-3-large \
KNOWLEDGE_EMBEDDING_DIMENSION=1536 \
.venv/bin/uvicorn app.main:create_app_from_env --factory --port 8080
```

## Tests

```bash
.venv/bin/python -m pytest tests/unit -q

# Integration tests need a real database; they are skipped without this variable.
KNOWLEDGE_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres \
  .venv/bin/python -m pytest tests -q
```

Unit tests run against an in-memory repository and a deterministic keyword
embedding, so they need neither PostgreSQL nor an embedding endpoint. The vector
store double deliberately ignores metadata filters so the isolation tests prove
the service's own guard.

## Resource expectations

The chunk count depends on document size and structure; a 300 KB document produces
about 110 chunks at the default 800-token chunk
size, which is a couple of batched embedding requests. Ingestion cost is
dominated by the embedding endpoint's latency, not by this service. Steady state
is small: a few hundred MB of memory for the container and a few MB of database
storage per manual-sized document at 1536 dimensions. Repeat ingests of
unchanged content cost one checksum comparison and no embedding calls.

Docling is the expensive part and is deliberately not a dependency of this
service. The official `docling-serve-cpu` image is around **4.4 GB** and wants
well over a gigabyte of memory while converting, so it does not belong on a 1 GB
evaluation VM. Run it elsewhere, or accept that PDF and Office uploads are
unavailable there; text and Markdown uploads need nothing extra.

## Not in this phase

No external object storage: originals are retained, but on a local volume behind
the storage boundary above rather than in Azure Blob or S3. No durable worker
queue:
conversion and embedding run as in-process background tasks, so a restart
reconciles interrupted jobs to `failed` with an explicit detail and the tester
re-uploads. No document deletion or reindex API, no per-user access control, and
no Azure or Mistral OCR adapters — Docling is the only converter wired up, behind
an adapter that is meant to take those later. Scanned documents with no text
layer are not OCR'd; they fail with a message saying so.
