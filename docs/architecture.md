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
   and, when available, page and heading context.
4. Athena chunks the text, requests embeddings, and stores vectors in pgvector.
5. A client searches explicit collection IDs and receives ranked passages with
   source metadata and citations.

`collection_id` is the data-isolation boundary. Athena applies the collection
filter during retrieval and checks every returned chunk before responding.

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
| `DOCLING_TIMEOUT_SECONDS` | `660` | Per-conversion client timeout. |
| `AZURE_OCR_ENDPOINT` | unset | Azure Document Intelligence resource URL. Required when `DOCUMENT_CONVERTER=azure`. |
| `AZURE_OCR_API_KEY` | unset | Credential for that resource. Required when `DOCUMENT_CONVERTER=azure`; never logged. |
| `AZURE_OCR_MODEL_ID` | `prebuilt-layout` | Analysis model used for conversion. |
| `AZURE_OCR_TIMEOUT_SECONDS` | `300` | Total time allowed for one document's analysis, including polling. |

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
interrupted jobs are marked as failed and need to be submitted again. Athena has
no built-in deletion API, remote object-storage adapter, per-user authorization,
or remote-system connector.

At more than 2,000 embedding dimensions, pgvector cannot build Athena's HNSW
index, so searches use an exact scan instead.
