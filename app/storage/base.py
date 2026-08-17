"""The source-object storage contract.

Original upload bytes are durable state, so they live behind the same kind of
explicit boundary as PostgreSQL rather than being written wherever the ingest
path happens to run. The contract is deliberately small — put, stat, read a
byte range, delete — so a later Azure Blob or S3 backend is a new class here and
nothing else: ingestion, the HTTP contract, and the browser all keep working.

Keys are always built by :func:`storage_key` from a service-owned document UUID.
No caller-supplied name, path, or filename ever reaches a backend.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID

# Which representation of one source a key addresses. "original" is the bytes the
# tester uploaded; "preview" is the normalized text the service indexed, kept so
# a format that cannot be shown inline still has something safe to display.
Variant = str
ORIGINAL_VARIANT: Variant = "original"
PREVIEW_VARIANT: Variant = "preview"
VARIANTS: tuple[Variant, ...] = (ORIGINAL_VARIANT, PREVIEW_VARIANT)


class StorageError(RuntimeError):
    """Raised when durable source storage cannot serve a request."""


class StoredObjectMissingError(StorageError):
    """Raised when a key that the database references is absent from storage."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    """What a backend guarantees about one stored object."""

    key: str
    byte_size: int
    checksum: str


def storage_key(document_id: UUID, variant: Variant = ORIGINAL_VARIANT) -> str:
    """Build the storage key for one document representation.

    The key is derived only from the service-owned document UUID — which is
    itself derived from ``(collection_id, external_id)`` — so it is stable across
    restarts and re-uploads, and cannot be influenced by a file name. The first
    two hex characters are a directory shard so one collection cannot grow a
    single directory with hundreds of thousands of entries.
    """
    if variant not in VARIANTS:  # pragma: no cover - defensive
        raise ValueError(f"Unknown storage variant: {variant!r}")
    identity = document_id.hex
    return f"{identity[:2]}/{identity}/{variant}"


class SourceObjectStore(ABC):
    """Durable, content-addressed-by-document storage for source bytes."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Short backend name, persisted alongside the key for diagnostics."""

    @abstractmethod
    async def put(self, key: str, content: bytes) -> StoredObject:
        """Store *content* at *key* atomically, replacing any previous object.

        A reader must never observe a partially written object, and a crash
        mid-write must leave either the old object or no object at all.
        """

    @abstractmethod
    async def stat(self, key: str) -> StoredObject | None:
        """Return what is stored at *key*, or ``None`` when it is absent."""

    @abstractmethod
    def read(self, key: str, *, offset: int = 0, length: int | None = None) -> AsyncIterator[bytes]:
        """Stream at most *length* bytes from *offset*.

        Range reads are a backend concern because a PDF viewer asks for them and
        buffering a whole document per request would not survive concurrent
        testers on a small VM.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove *key* if it exists. Absence is not an error."""
