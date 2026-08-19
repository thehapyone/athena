# Architecture

This document covers Athena's deployment model, runtime boundaries, and
configuration. Start with the [README](../README.md) to run and use the service,
or see the [API Reference](api.md) for supported endpoints.

## Deployment

The default [`compose.yaml`](../compose.yaml) starts three containers:

| Container | Responsibility |
| --- | --- |
| Athena | Authenticated ingest, storage, indexing, and search API. |
| PostgreSQL with pgvector | Document metadata, jobs, and vector search. |
| Docling | PDF, Office, and HTML conversion. |

Athena keeps original uploads in a Docker volume and exposes only its API on
`127.0.0.1:8080`. PostgreSQL and Docling have no host ports. Docling is a CPU
image pinned to `v1.30.0`; it is sizeable and should run separately when its
resource requirements do not fit the deployment.

Docling runs conversion, including OCR, on Athena's own CPU, which is slow and
resource-heavy for large or scanned PDFs. Setting `DOCUMENT_CONVERTER=azure`
switches conversion to Azure AI Document Intelligence instead: OCR runs as a
managed cloud service, and Athena only submits documents and polls for the
result. The `docling` container becomes unused in that mode and can be removed
from `compose.yaml` along with the `depends_on` entry that names it.

## Data Flow

1. A client submits text or a supported file to a named collection.
2. Athena writes an uploaded original before background processing begins.
3. Text and Markdown are normalized in Athena. Other supported files are sent
   to the configured document converter -- Docling by default, or Azure AI
   Document Intelligence when `DOCUMENT_CONVERTER=azure` -- which returns text
   and, when available, page, heading, and table context.
4. Athena splits anything the converter did not already size, prefixes each chunk
   with its document title, heading, and page, requests embeddings, and stores
   vectors in pgvector.
5. A client searches explicit collection IDs and receives ranked passages with
   source metadata and citations.

`collection_id` is the data-isolation boundary. Athena applies the collection
filter during retrieval and checks every returned chunk before responding.

## Asynchronous Conversion

Conversion never happens inside the upload request. `POST /v1/documents/file`
answers `202 Accepted` with an Athena `job_id`, and the client polls
`GET /v1/jobs/{job_id}`. Both converters are then driven by submit-and-poll, but
only Docling's task id is persisted, which is what makes a Docling conversion
survive an Athena restart.

For Docling, Athena submits the file to
`POST /v1/chunk/hierarchical/file/async`, records the returned `task_id` on the
job row before polling starts, then follows `GET /v1/status/poll/{task_id}` until
the task settles and fetches `GET /v1/result/{task_id}`.

`/v1/convert/file` is not used instead. Its Markdown holds the same text, and
page boundaries can be recovered from it with `md_page_break_placeholder`, so the
difference is not page provenance. It is that the chunk route *reports* structure
instead of leaving it to be re-derived: a chunk's `headings` are its real
ancestors and its `doc_items` state that it is a table, whereas Markdown alone
would have a table identified by guessing from punctuation.

The converter is asked for document structure, not for chunk sizes. It returns
chunks that follow the document's own hierarchy, each carrying `page_numbers`,
`headings` and `doc_items`; sizing them for the embedding model is Athena's job
and happens centrally, for every source alike. `chunking_use_markdown_tables`
returns tables as Markdown rows rather than triplets, which both preserves a
tabular document's text -- measured at 42% of one service parts list -- and lets
an oversized table be split on row boundaries with its header repeated. Each
chunk's `doc_items` self-references (`#/tables/0`) are what identify a table at
all, so table structure is read from the converter, never guessed from the text.

This route loads no model of its own, so Docling serves it without reaching a
model host at conversion time.

Docling's synchronous routes are deliberately unused. They answer `504` once
`DOCLING_SERVE_MAX_SYNC_WAIT` elapses while the Docling worker keeps running,
which loses conversions that in fact succeeded. Submitting a task instead means
the only conversion bound is Athena's own `DOCLING_CONVERSION_DEADLINE_SECONDS`,
measured from submission, while each individual HTTP request stays bounded by
`DOCLING_TIMEOUT_SECONDS`.

Because the task id is durable:

- Re-uploading a file whose job is still `accepted` or `processing` returns that
  job. Uploads are identified by content, so the second request is the same work
  and no second conversion is submitted.
- On restart, jobs holding a converter task are resumed against that task rather
  than failed. Their deadline still runs from the original submission, so a
  restart grants no extra budget.
- A task the converter no longer holds -- expired, or lost to a Docling restart
  -- fails the job with a message that says to upload the file again. So does a
  task belonging to a converter this deployment is no longer configured with.

Because Docling mints the task id, submitting and recording it cannot be one
atomic step. If Athena dies in the moment between the two, the task id is lost:
the job is failed on restart and a retry converts the file again, while the
orphaned Docling task runs to completion unread. The window is one database write
wide and the outcome is a wasted conversion rather than incorrect state, but it
is not zero. Closing it would need an idempotency key on Docling's submit route,
which its API does not offer.

While a conversion is in flight, a Docling answer that means "not now" -- a 5xx,
a throttling or auth response, or a transport failure -- is retried rather than
treated as a failure, for up to five continuous minutes and never past the
conversion deadline. That window covers a converter restart or a gateway blip
without discarding an hour of work; a converter that has genuinely gone away
still fails the job promptly. Result retrieval is retried on the same terms,
since by then the conversion has already succeeded.

Job details distinguish converter unavailable, refused submission, remote
conversion failure, a lost task, an exceeded deadline, and a result that could
not be retrieved. None of them carry a URL or a credential.

## Sources and Connectors

Athena exposes two ingestion boundaries: normalized text and file upload. It
does not fetch remote systems itself. Source-specific connectors belong in the
calling application, where credentials, permissions, web crawling, incremental
sync, and deletion policy are already owned.

Files use the extension to select a format. Athena directly handles `.txt`,
`.md`, and `.markdown`; the configured document converter handles `.pdf`,
`.docx`, `.pptx`, `.xlsx`, and `.html`. Docling does not OCR a scanned PDF that
has no extractable text; Azure Document Intelligence does, since OCR is part
of its layout analysis.

## Configuration

The Compose deployment reads required credentials and common tuning settings
from `.env`. The application validates all configuration at startup and does not
write secret values to logs.

| Setting | Default | Purpose |
| --- | --- | --- |
| `ATHENA_EMBEDDING_BASE_URL` | required | OpenAI-compatible embedding endpoint. |
| `ATHENA_EMBEDDING_API_KEY` | required | Credential for that endpoint. |
| `ATHENA_EMBEDDING_MODEL` | required | Model or provider deployment name. |
| `ATHENA_EMBEDDING_DIMENSION` | required | Selected embedding dimension. |
| `ATHENA_MAX_UPLOAD_BYTES` | `52428800` | Upload limit; maximum is 50 MiB. |
| `ATHENA_RETRIEVAL_MODE` | `hybrid` | `hybrid` or `vector`. |
| `DOCUMENT_CONVERTER` | `docling` | `docling` or `azure`. Selects which service handles PDF and Office uploads. |
| `DOCLING_BASE_URL` | `http://docling:5001` | Override for a separately managed Docling service. |
| `DOCLING_TIMEOUT_SECONDS` | `120` | Timeout for one HTTP request to Docling (submit, poll, or result fetch). |
| `DOCLING_CONVERSION_DEADLINE_SECONDS` | `3600` | Total time one document's conversion may take, measured from submission and across restarts. Must be at least `DOCLING_TIMEOUT_SECONDS`. |
| `DOCLING_POLL_INTERVAL_SECONDS` | `5` | Delay between Docling task status polls. |
| `AZURE_OCR_ENDPOINT` | unset | Azure Document Intelligence resource URL. Required when `DOCUMENT_CONVERTER=azure`. |
| `AZURE_OCR_API_KEY` | unset | Credential for that resource. Required when `DOCUMENT_CONVERTER=azure`; never logged. |
| `AZURE_OCR_MODEL_ID` | `prebuilt-layout` | Analysis model used for conversion. |
| `AZURE_OCR_TIMEOUT_SECONDS` | `300` | Total time allowed for one document's analysis, including polling. |

The binding size limit on an upload is normally `ATHENA_MAX_UPLOAD_BYTES`
(50 MiB max), well under Azure Document Intelligence's own per-tier ceiling
(500 MB / 2,000 pages on Standard; 4 MB / 2 pages on the Free tier). The
extracted text is separately bounded by `ATHENA_MAX_DOCUMENT_BYTES`. A large
increase to `ATHENA_MAX_UPLOAD_BYTES` for Azure conversion should stay well
under Azure's own per-tier limits.

For deployments outside Compose, Athena also supports `ATHENA_DATABASE_URL`,
`ATHENA_API_TOKEN`, `ATHENA_DB_SCHEMA`, `ATHENA_SOURCE_STORAGE_DIR`, chunking
controls, search-result limits, and
filename/document-size bounds. Their defaults and accepted ranges are defined in
[`app/config.py`](../app/config.py).

## Lifecycle and Limits

Text sources are identified by `(collection_id, external_id)`. A matching
checksum avoids re-embedding unchanged content. File uploads derive an external
ID from their content, so re-uploading the same file is idempotent.

Ingestion and conversion run as in-process background work. If Athena restarts,
a job waiting on a converter task is resumed; any other interrupted job is marked
as failed and needs to be submitted again. Athena has no built-in deletion API,
remote object-storage adapter, per-user authorization, or remote-system
connector.

The Compose deployment sets `DOCLING_SERVE_RESULT_REMOVAL_DELAY=3600` so a
completed conversion stays fetchable long enough for a restarting Athena to
collect it. Docling keeps its task state in memory, so restarting the `docling`
container does lose in-flight tasks; the affected jobs fail with a retryable
message rather than hanging.

At more than 2,000 embedding dimensions, pgvector cannot build Athena's HNSW
index, so searches use an exact scan instead.
