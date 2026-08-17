"""Hybrid retrieval with mandatory collection isolation and citations."""

import hashlib
import inspect
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)

from app.config import Settings
from app.log import logger
from app.models import (
    Citation,
    RESERVED_METADATA_KEYS,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchStats,
)


async def search_documents(
    vector_store: BasePydanticVectorStore,
    embed_model: BaseEmbedding,
    request: SearchRequest,
    settings: Settings,
) -> SearchResponse:
    """Retrieve ranked chunks from the requested collections.

    Collection scoping is applied twice: once as a backend metadata filter and
    once as a post-retrieval guard. The guard is what makes isolation a service
    guarantee rather than a property of whichever vector backend is configured.
    """
    allowed = list(dict.fromkeys(request.collection_ids))
    top_k = min(request.top_k or settings.default_top_k, settings.max_top_k)

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = _build_retriever(
        index,
        similarity_top_k=top_k,
        filters=_build_metadata_filters(allowed, request),
        retrieval_mode=settings.retrieval_mode,
    )

    nodes = await retriever.aretrieve(request.query)
    retrieved = len(nodes)

    nodes, leaked = _enforce_collection_isolation(nodes, set(allowed))
    if leaked:
        logger.error(
            "Vector backend returned %d chunk(s) outside the requested collections; "
            "they were dropped before the response was built",
            leaked,
        )
    nodes = _apply_post_filters(nodes, request)
    nodes = _dedupe(nodes)
    nodes.sort(key=lambda nws: nws.score or 0.0, reverse=True)
    nodes = nodes[:top_k]

    items = [_build_item(nws) for nws in nodes]
    warnings: list[str] = []
    if not items:
        warnings.append("no_results")

    logger.info(
        "Search over %s returned %d of %d retrieved chunk(s)",
        ",".join(allowed),
        len(items),
        retrieved,
    )
    return SearchResponse(
        items=items,
        warnings=warnings,
        stats=SearchStats(retrieved=retrieved, returned=len(items)),
    )


def _build_metadata_filters(
    collection_ids: list[str],
    request: SearchRequest,
) -> MetadataFilters:
    filters = [
        MetadataFilter(
            key="collection_id",
            value=collection_ids,
            operator=FilterOperator.IN,
        )
    ]
    extra = request.filters
    if extra is not None:
        if extra.source_type:
            filters.append(
                MetadataFilter(
                    key="source_type", value=list(extra.source_type), operator=FilterOperator.IN
                )
            )
        if extra.external_id:
            filters.append(
                MetadataFilter(
                    key="external_id", value=list(extra.external_id), operator=FilterOperator.IN
                )
            )
        if extra.exclude_external_id:
            filters.append(
                MetadataFilter(
                    key="external_id",
                    value=list(extra.exclude_external_id),
                    operator=FilterOperator.NIN,
                )
            )
        if extra.updated_after:
            filters.append(
                MetadataFilter(
                    key="updated_at_ts",
                    value=int(_ensure_tz(extra.updated_after).timestamp()),
                    operator=FilterOperator.GTE,
                )
            )
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)


def _build_retriever(
    index: VectorStoreIndex,
    *,
    similarity_top_k: int,
    filters: MetadataFilters,
    retrieval_mode: str,
) -> Any:
    kwargs: dict[str, Any] = {"similarity_top_k": similarity_top_k, "filters": filters}
    if retrieval_mode == "hybrid":
        kwargs["vector_store_query_mode"] = VectorStoreQueryMode.HYBRID
    return index.as_retriever(**_supported_kwargs(index.as_retriever, kwargs))


def _supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the retriever factory does not accept."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _enforce_collection_isolation(
    nodes: Iterable[Any],
    allowed: set[str],
) -> tuple[list[Any], int]:
    kept: list[Any] = []
    dropped = 0
    for nws in nodes:
        if nws.node.metadata.get("collection_id") in allowed:
            kept.append(nws)
        else:
            dropped += 1
    return kept, dropped


def _apply_post_filters(nodes: Iterable[Any], request: SearchRequest) -> list[Any]:
    """Re-apply the optional filters that a backend may not support."""
    extra = request.filters
    if extra is None:
        return list(nodes)

    source_types = set(extra.source_type or ())
    external_ids = set(extra.external_id or ())
    excluded_external_ids = set(extra.exclude_external_id or ())
    cutoff = _ensure_tz(extra.updated_after) if extra.updated_after else None

    kept: list[Any] = []
    for nws in nodes:
        metadata = nws.node.metadata
        if source_types and metadata.get("source_type") not in source_types:
            continue
        if external_ids and metadata.get("external_id") not in external_ids:
            continue
        if excluded_external_ids and metadata.get("external_id") in excluded_external_ids:
            continue
        if cutoff is not None:
            updated_at = _parse_updated_at(metadata)
            if updated_at is None or updated_at < cutoff:
                continue
        kept.append(nws)
    return kept


def _dedupe(nodes: Iterable[Any]) -> list[Any]:
    """Keep the highest-scoring node per (document, chunk text)."""
    best: dict[tuple[str, str], Any] = {}
    for nws in nodes:
        node = nws.node
        document_key = (
            node.metadata.get("document_id") or getattr(node, "ref_doc_id", None) or node.node_id
        )
        text_hash = hashlib.sha256(node.get_content().encode("utf-8")).hexdigest()
        key = (str(document_key), text_hash)
        previous = best.get(key)
        if previous is None or (nws.score or 0.0) > (previous.score or 0.0):
            best[key] = nws
    return list(best.values())


def _build_item(nws: Any) -> SearchResultItem:
    metadata = dict(nws.node.metadata)
    score = float(nws.score or 0.0)
    page = metadata.get("page")
    section = metadata.get("section") or None
    return SearchResultItem(
        text=nws.node.get_content(),
        score=score,
        retrieval_score=score,
        collection_id=str(metadata.get("collection_id", "")),
        document_id=_optional_str(metadata.get("document_id")),
        external_id=_optional_str(metadata.get("external_id")),
        title=_optional_str(metadata.get("title")),
        source_type=_optional_str(metadata.get("source_type")),
        source_uri=_optional_str(metadata.get("source_uri")),
        version=_optional_str(metadata.get("version")),
        checksum=_optional_str(metadata.get("checksum")),
        page=page if isinstance(page, int) else None,
        section=section,
        chunk_id=str(nws.node.node_id),
        updated_at=_parse_updated_at(metadata),
        metadata={
            key: value for key, value in metadata.items() if key not in RESERVED_METADATA_KEYS
        },
        citations=[_build_citation(metadata, str(nws.node.node_id))],
    )


def _build_citation(metadata: dict[str, Any], chunk_id: str) -> Citation:
    page = metadata.get("page")
    section = metadata.get("section") or None
    label = (
        _optional_str(metadata.get("title"))
        or _optional_str(metadata.get("external_id"))
        or _optional_str(metadata.get("collection_id"))
        or "document"
    )
    if section:
        locator = f"section:{section}"
    elif isinstance(page, int):
        locator = f"page:{page}"
    else:
        locator = f"chunk:{chunk_id}"
    return Citation(
        label=label,
        source_uri=_optional_str(metadata.get("source_uri")),
        locator=locator,
        page=page if isinstance(page, int) else None,
        section=section,
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_updated_at(metadata: dict[str, Any]) -> datetime | None:
    value = metadata.get("updated_at")
    if value is None:
        value = metadata.get("updated_at_ts")
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_tz(value)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            return _ensure_tz(datetime.fromisoformat(text))
        except ValueError:
            return None
    return None


def _ensure_tz(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)
