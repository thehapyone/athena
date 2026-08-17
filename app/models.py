"""Versioned request/response contract for Athena."""

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.config import MAXIMUM_SEARCH_COLLECTIONS

JobStatus = Literal["accepted", "processing", "completed", "failed"]
# What a source looks like to a UI: an ingest job's four states collapsed into
# the three a person can act on.
SourceStatus = Literal["processing", "ready", "failed"]

CollectionId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,62}$"),
]
ExternalId = Annotated[str, StringConstraints(min_length=1, max_length=256, strip_whitespace=True)]

# Metadata keys the service owns. User-supplied metadata may not overwrite them,
# because search filtering and collection isolation read these keys back.
RESERVED_METADATA_KEYS = frozenset(
    {
        "collection_id",
        "document_id",
        "external_id",
        "title",
        "source_type",
        "source_uri",
        "checksum",
        "version",
        "page",
        "section",
        "updated_at",
        "updated_at_ts",
        "embedding_model",
        "embedding_dim",
        "filename",
        "media_type",
    }
)


class TextDocumentRequest(BaseModel):
    """Normalized text document submitted by a caller or a future parser worker."""

    model_config = ConfigDict(extra="forbid")

    collection_id: CollectionId
    external_id: ExternalId
    text: str = Field(min_length=1)
    title: str = Field(default="", max_length=512)
    source_type: str = Field(default="text", max_length=64)
    source_uri: str = Field(default="", max_length=2048)
    version: str = Field(default="", max_length=128)
    checksum: str | None = Field(default=None, max_length=128)
    page: int | None = Field(default=None, ge=0)
    section: str = Field(default="", max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)
    force_reindex: bool = False


class IngestAcceptedResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    collection_id: str
    external_id: str


class JobResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    collection_id: str
    external_id: str
    document_id: UUID | None = None
    chunk_count: int = 0
    unchanged: bool = False
    detail: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SourceItem(BaseModel):
    """One selectable document source."""

    external_id: str
    title: str
    source_type: str
    status: SourceStatus
    chunk_count: int = Field(ge=0)
    detail: str | None = None
    filename: str | None = None
    media_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Whether the original upload was retained and can therefore be opened.
    # Sources ingested before originals were kept report false.
    viewable: bool = False
    byte_size: int | None = None
    page_count: int | None = None
    preview_available: bool = False


class SourceListResponse(BaseModel):
    collection_id: str
    items: list[SourceItem]
    truncated: bool = False


class SourceContentResponse(BaseModel):
    """What a viewer needs before it decides how to render a source."""

    collection_id: str
    external_id: str
    title: str
    source_type: str
    status: SourceStatus
    filename: str
    media_type: str
    byte_size: int
    checksum: str
    page_count: int | None = None
    preview_available: bool = False
    preview_bytes: int | None = None
    chunk_count: int = Field(default=0, ge=0)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: list[str] | None = None
    external_id: list[str] | None = None
    exclude_external_id: list[str] | None = None
    updated_after: datetime | None = None


class SearchRequest(BaseModel):
    """Search is always scoped to an explicit, non-empty set of collections."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2048)
    collection_ids: list[CollectionId] = Field(
        min_length=1, max_length=MAXIMUM_SEARCH_COLLECTIONS
    )
    top_k: int | None = Field(default=None, ge=1, le=200)
    filters: SearchFilters | None = None


class Citation(BaseModel):
    label: str
    source_uri: str | None = None
    locator: str | None = None
    page: int | None = None
    section: str | None = None


class SearchResultItem(BaseModel):
    text: str
    score: float
    retrieval_score: float | None = None
    collection_id: str
    document_id: str | None = None
    external_id: str | None = None
    title: str | None = None
    source_type: str | None = None
    source_uri: str | None = None
    version: str | None = None
    checksum: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_id: str
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)


class SearchStats(BaseModel):
    retrieved: int = Field(ge=0)
    returned: int = Field(ge=0)


class SearchResponse(BaseModel):
    items: list[SearchResultItem]
    warnings: list[str] = Field(default_factory=list)
    stats: SearchStats


class HealthResponse(BaseModel):
    status: str
    database: str | None = None
    detail: str | None = None
