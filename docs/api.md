# API Reference

Athena exposes an HTTP API at `/v1`. Every `/v1/*` request requires:

```http
Authorization: Bearer <ATHENA_API_TOKEN>
```

The running service also exposes an interactive OpenAPI reference at `/docs`.

## Health

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Process liveness. Does not require a database connection. |
| `GET` | `/health/ready` | Database readiness. Returns `503` until Athena is ready. |

## Ingestion

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/documents/text` | Create or update a text source. |
| `POST` | `/v1/documents/file` | Upload and index one supported file. |
| `GET` | `/v1/jobs/{job_id}` | Check an asynchronous ingestion job. |

`POST /v1/documents/text` accepts JSON with `collection_id`, `external_id`, and
`text`. Optional fields include `title`, `source_type`, `source_uri`, `version`,
`page`, `section`, `metadata`, and `force_reindex`.

`POST /v1/documents/file` accepts `multipart/form-data` with `file`,
`collection_id`, and an optional `title`. Supported extensions are `.txt`,
`.md`, `.markdown`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, and `.html`.

Both ingestion endpoints return `202 Accepted` with a `job_id`. Poll the job
endpoint until `status` is `completed` or `failed`.

## Sources

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/documents?collection_id=…` | List up to 200 sources in a collection. |
| `GET` | `/v1/documents/source?collection_id=…&external_id=…` | Get metadata for a retained upload. |
| `GET` | `/v1/documents/source/content?collection_id=…&external_id=…` | Download or stream a retained upload. |

Source content supports `Range` requests and `ETag` validation. Set
`variant=preview` to retrieve extracted text when it is available; the default
`variant=original` retrieves the uploaded file.

## Retrieval

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/search` | Retrieve ranked passages from explicit collections. |

The JSON request requires `query` and `collection_ids`. Optional `top_k` limits
the results. Optional `filters` can include `source_type`, `external_id`,
`exclude_external_id`, and `updated_after`.

Each result includes the retrieved text, score, source metadata, and citations.
Searches never cross a collection boundary unless the caller explicitly includes
that collection in `collection_ids`.
