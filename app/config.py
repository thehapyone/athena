"""Environment-driven configuration for the standalone knowledge service.

Every setting comes from an explicit environment variable so the service can run
without any repository-specific configuration file. Validation failures name the
variable but never echo its value, so secrets stay out of logs and tracebacks.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

RetrievalMode = Literal["hybrid", "vector"]

VECTOR_TABLE_NAME = "knowledge_vectors"
MINIMUM_API_TOKEN_LENGTH = 16
MAXIMUM_SEARCH_COLLECTIONS = 10
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAXIMUM_LISTED_SOURCES = 200
DEFAULT_SOURCE_STORAGE_DIR = "/var/lib/athena/sources"

_SCHEMA_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_POSTGRES_SCHEMES = ("postgres://", "postgresql://")


class ConfigurationError(ValueError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable service settings."""

    database_url: str
    db_schema: str
    api_token: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimension: int
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    max_top_k: int
    retrieval_mode: RetrievalMode
    max_document_bytes: int
    max_upload_bytes: int
    max_filename_characters: int
    source_storage_dir: str
    docling_base_url: str
    docling_timeout_seconds: int
    log_level: str

    @property
    def conversion_configured(self) -> bool:
        """Whether uploads that need a converter (PDF, Office) can be processed."""
        return bool(self.docling_base_url)

    @property
    def vector_table(self) -> str:
        return VECTOR_TABLE_NAME

    @property
    def async_database_url(self) -> str:
        """SQLAlchemy asyncpg URL used by the vector store."""
        return _replace_scheme(self.database_url, "postgresql+asyncpg://")

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy psycopg2 URL used by the vector store for DDL."""
        return _replace_scheme(self.database_url, "postgresql://")

    def redacted(self) -> dict[str, object]:
        """Return a log-safe view of the settings."""
        return {
            "db_schema": self.db_schema,
            "embedding_base_url": self.embedding_base_url,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "embedding_batch_size": self.embedding_batch_size,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "default_top_k": self.default_top_k,
            "max_top_k": self.max_top_k,
            "retrieval_mode": self.retrieval_mode,
            "max_document_bytes": self.max_document_bytes,
            "max_upload_bytes": self.max_upload_bytes,
            "source_storage_dir": self.source_storage_dir,
            "docling_base_url": self.docling_base_url or "(unset)",
            "docling_timeout_seconds": self.docling_timeout_seconds,
            "log_level": self.log_level,
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env

        database_url = _required(source, "KNOWLEDGE_DATABASE_URL")
        if not database_url.startswith(_POSTGRES_SCHEMES):
            raise ConfigurationError(
                "KNOWLEDGE_DATABASE_URL must be a postgres:// or postgresql:// URL."
            )

        db_schema = _text(source, "KNOWLEDGE_DB_SCHEMA", "knowledge")
        if not _SCHEMA_PATTERN.match(db_schema):
            raise ConfigurationError(
                "KNOWLEDGE_DB_SCHEMA must be a lowercase SQL identifier "
                "(letters, digits and underscores, not starting with a digit)."
            )

        api_token = _required(source, "KNOWLEDGE_API_TOKEN")
        if len(api_token) < MINIMUM_API_TOKEN_LENGTH:
            raise ConfigurationError(
                f"KNOWLEDGE_API_TOKEN must be at least {MINIMUM_API_TOKEN_LENGTH} characters."
            )

        embedding_base_url = _required(source, "KNOWLEDGE_EMBEDDING_BASE_URL").rstrip("/")
        parsed = urlsplit(embedding_base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ConfigurationError(
                "KNOWLEDGE_EMBEDDING_BASE_URL must be an absolute http(s) URL pointing at an "
                "OpenAI-compatible base endpoint (for example https://<resource>/openai/v1)."
            )

        embedding_dimension = _integer(source, "KNOWLEDGE_EMBEDDING_DIMENSION", None, 8, 4096)
        chunk_size = _integer(source, "KNOWLEDGE_CHUNK_SIZE", 800, 64, 8192)
        chunk_overlap = _integer(source, "KNOWLEDGE_CHUNK_OVERLAP", 120, 0, 4096)
        if chunk_overlap >= chunk_size:
            raise ConfigurationError(
                "KNOWLEDGE_CHUNK_OVERLAP must be smaller than KNOWLEDGE_CHUNK_SIZE."
            )

        default_top_k = _integer(source, "KNOWLEDGE_DEFAULT_TOP_K", 8, 1, 200)
        max_top_k = _integer(source, "KNOWLEDGE_MAX_TOP_K", 50, 1, 200)
        if default_top_k > max_top_k:
            raise ConfigurationError(
                "KNOWLEDGE_DEFAULT_TOP_K must not exceed KNOWLEDGE_MAX_TOP_K."
            )

        retrieval_mode = _text(source, "KNOWLEDGE_RETRIEVAL_MODE", "hybrid").lower()
        if retrieval_mode not in ("hybrid", "vector"):
            raise ConfigurationError("KNOWLEDGE_RETRIEVAL_MODE must be 'hybrid' or 'vector'.")

        # Original upload bytes are kept here so a source can be reopened and
        # cross-checked after a restart. It must be an absolute path on a durable
        # mount; a relative path would follow the process's working directory.
        source_storage_dir = _text(
            source, "KNOWLEDGE_SOURCE_STORAGE_DIR", DEFAULT_SOURCE_STORAGE_DIR
        )
        if not source_storage_dir.startswith("/"):
            raise ConfigurationError(
                "KNOWLEDGE_SOURCE_STORAGE_DIR must be an absolute path."
            )

        # Docling is optional: without it, text and Markdown uploads still work and
        # formats that need conversion fail per job with an explicit message.
        docling_base_url = _text(source, "DOCLING_BASE_URL", "").rstrip("/")
        if docling_base_url:
            docling = urlsplit(docling_base_url)
            if docling.scheme not in ("http", "https") or not docling.netloc:
                raise ConfigurationError(
                    "DOCLING_BASE_URL must be an absolute http(s) URL pointing at a docling-serve "
                    "instance (for example http://docling:5001)."
                )

        return cls(
            database_url=database_url,
            db_schema=db_schema,
            api_token=api_token,
            embedding_base_url=embedding_base_url,
            embedding_api_key=_required(source, "KNOWLEDGE_EMBEDDING_API_KEY"),
            embedding_model=_required(source, "KNOWLEDGE_EMBEDDING_MODEL"),
            embedding_dimension=embedding_dimension,
            embedding_batch_size=_integer(source, "KNOWLEDGE_EMBEDDING_BATCH_SIZE", 64, 1, 2048),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            default_top_k=default_top_k,
            max_top_k=max_top_k,
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            max_document_bytes=_integer(
                source, "KNOWLEDGE_MAX_DOCUMENT_BYTES", 8_000_000, 1_024, 64_000_000
            ),
            max_upload_bytes=_integer(
                source,
                "KNOWLEDGE_MAX_UPLOAD_BYTES",
                DEFAULT_MAX_UPLOAD_BYTES,
                1_024,
                DEFAULT_MAX_UPLOAD_BYTES,
            ),
            max_filename_characters=_integer(
                source, "KNOWLEDGE_MAX_FILENAME_CHARACTERS", 255, 16, 1_024
            ),
            source_storage_dir=source_storage_dir,
            docling_base_url=docling_base_url,
            docling_timeout_seconds=_integer(source, "DOCLING_TIMEOUT_SECONDS", 660, 5, 900),
            log_level=_text(source, "KNOWLEDGE_LOG_LEVEL", "INFO").upper(),
        )


def _replace_scheme(dsn: str, scheme: str) -> str:
    for candidate in _POSTGRES_SCHEMES:
        if dsn.startswith(candidate):
            return f"{scheme}{dsn[len(candidate):]}"
    return dsn


def _text(source: Mapping[str, str], name: str, default: str) -> str:
    value = source.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _required(source: Mapping[str, str], name: str) -> str:
    value = (source.get(name) or "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required but was empty or unset.")
    return value


def _integer(
    source: Mapping[str, str],
    name: str,
    default: int | None,
    minimum: int,
    maximum: int,
) -> int:
    raw = (source.get(name) or "").strip()
    if not raw:
        if default is None:
            raise ConfigurationError(f"{name} is required but was empty or unset.")
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer.") from None
    if value < minimum or value > maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value
