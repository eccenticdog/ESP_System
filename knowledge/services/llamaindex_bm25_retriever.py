from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
from pathlib import Path
from typing import Any

import jieba
from langchain_core.documents import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from config.settings import settings
from utils.vector_store import get_vector_store_collection_name


class LlamaIndexBM25DependencyError(RuntimeError):
    """Raised when the BM25 retriever dependencies or backend are unavailable."""


def _get_elasticsearch_client_class():
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise LlamaIndexBM25DependencyError(
            "The Elasticsearch Python client is not installed. Install 'elasticsearch'."
        ) from exc

    return Elasticsearch


def _get_elasticsearch_helpers_module():
    try:
        from elasticsearch import helpers
    except ImportError as exc:
        raise LlamaIndexBM25DependencyError(
            "The Elasticsearch helpers are not installed. Install 'elasticsearch'."
        ) from exc

    return helpers


def get_bm25_storage_dir(storage_dir: str | None = None) -> Path:
    base_dir = Path(storage_dir or settings.BM25_STORAGE_DIR)
    return base_dir / get_vector_store_collection_name()


def get_bm25_retriever_persist_dir(storage_dir: str | None = None) -> Path:
    return get_bm25_storage_dir(storage_dir) / "retriever"


def get_bm25_docstore_persist_path(storage_dir: str | None = None) -> Path:
    return get_bm25_storage_dir(storage_dir) / "docstore.json"


def get_bm25_index_name(index_name: str | None = None) -> str:
    candidate = (index_name or settings.BM25_ELASTICSEARCH_INDEX or "").strip()
    if not candidate:
        candidate = f"{get_vector_store_collection_name()}-bm25"

    candidate = candidate.lower()
    candidate = re.sub(r"[^a-z0-9._-]+", "-", candidate).strip("-_.")
    candidate = candidate or "its-knowledge-bm25"
    return candidate[:255]


def build_nodes_from_documents(documents: list[Document]) -> list[TextNode]:
    nodes: list[TextNode] = []

    for index, document in enumerate(documents):
        text = str(document.page_content or "").strip()
        if not text:
            continue

        metadata = dict(document.metadata or {})
        path = str(
            metadata.get("path")
            or metadata.get("source")
            or metadata.get("file_path")
            or ""
        ).strip()
        title = str(metadata.get("title") or metadata.get("file_name") or "").strip()

        if path:
            metadata.setdefault("path", path)
            metadata.setdefault("source", path)

        if path and not title:
            title = Path(path).stem
        if title:
            metadata.setdefault("title", title)

        node = TextNode(
            id_=_build_node_id(path=path, chunk_index=index, text=text),
            text=text,
            metadata=metadata,
        )
        nodes.append(node)

    return nodes


class ElasticSearchBM25Retriever(BaseRetriever):
    """ElasticSearch-backed BM25 retriever compatible with LlamaIndex fusion."""

    def __init__(
        self,
        similarity_top_k: int = 5,
        index_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__()
        self._client = client or _build_elasticsearch_client()
        self._index_name = get_bm25_index_name(index_name=index_name)
        self.similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = str(query_bundle.query_str or "").strip()
        if not query:
            return []

        try:
            response = self._client.search(
                index=self._index_name,
                body=_build_search_body(
                    query=query,
                    similarity_top_k=self.similarity_top_k,
                ),
            )
        except Exception as exc:
            raise LlamaIndexBM25DependencyError(
                f"Failed to query Elasticsearch BM25 index '{self._index_name}'."
            ) from exc

        hits = (((response or {}).get("hits") or {}).get("hits") or [])
        nodes: list[NodeWithScore] = []
        for hit in hits:
            source = dict(hit.get("_source") or {})
            text = str(source.get("text") or "").strip()
            if not text:
                continue

            node = TextNode(
                id_=str(hit.get("_id") or _build_node_id("", 0, text)),
                text=text,
                metadata=_extract_metadata_from_source(source),
            )
            nodes.append(
                NodeWithScore(
                    node=node,
                    score=float(hit.get("_score") or 0.0),
                )
            )

        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return await asyncio.to_thread(self._retrieve, query_bundle)


def persist_bm25_index(
    documents: list[Document],
    similarity_top_k: int = 5,
    storage_dir: str | None = None,
    index_name: str | None = None,
) -> Any:
    store_documents_for_bm25(
        documents,
        storage_dir=storage_dir,
        index_name=index_name,
        refresh=False,
    )
    return rebuild_bm25_index(
        similarity_top_k=similarity_top_k,
        storage_dir=storage_dir,
        index_name=index_name,
    )


def store_documents_for_bm25(
    documents: list[Document],
    storage_dir: str | None = None,
    index_name: str | None = None,
    refresh: bool = False,
) -> None:
    del storage_dir

    nodes = build_nodes_from_documents(documents)
    if not nodes:
        return

    client = _build_elasticsearch_client()
    resolved_index_name = get_bm25_index_name(index_name=index_name)
    _ensure_bm25_index(client, resolved_index_name)

    source_paths = sorted(
        {
            str(node.metadata.get("path") or node.metadata.get("source") or "").strip()
            for node in nodes
            if str(node.metadata.get("path") or node.metadata.get("source") or "").strip()
        }
    )
    if source_paths:
        try:
            client.delete_by_query(
                index=resolved_index_name,
                body={"query": {"terms": {"path": source_paths}}},
                conflicts="proceed",
                refresh=False,
            )
        except Exception as exc:
            raise LlamaIndexBM25DependencyError(
                f"Failed to replace Elasticsearch BM25 documents in index '{resolved_index_name}'."
            ) from exc

    actions = [
        {
            "_op_type": "index",
            "_index": resolved_index_name,
            "_id": node.node_id,
            "_source": _serialize_node_for_elasticsearch(node),
        }
        for node in nodes
    ]

    try:
        _get_elasticsearch_helpers_module().bulk(
            client,
            actions,
            refresh=refresh,
        )
    except Exception as exc:
        raise LlamaIndexBM25DependencyError(
            f"Failed to persist Elasticsearch BM25 documents into index '{resolved_index_name}'."
        ) from exc


def rebuild_bm25_index(
    similarity_top_k: int = 5,
    storage_dir: str | None = None,
    index_name: str | None = None,
) -> Any:
    del storage_dir

    client = _build_elasticsearch_client()
    resolved_index_name = get_bm25_index_name(index_name=index_name)
    _ensure_bm25_index(client, resolved_index_name)

    try:
        client.indices.refresh(index=resolved_index_name)
    except Exception as exc:
        raise LlamaIndexBM25DependencyError(
            f"Failed to refresh Elasticsearch BM25 index '{resolved_index_name}'."
        ) from exc

    return ElasticSearchBM25Retriever(
        similarity_top_k=similarity_top_k,
        index_name=resolved_index_name,
        client=client,
    )


def load_bm25_retriever(
    similarity_top_k: int = 5,
    storage_dir: str | None = None,
    index_name: str | None = None,
) -> Any:
    del storage_dir

    client = _build_elasticsearch_client()
    resolved_index_name = get_bm25_index_name(index_name=index_name)

    try:
        index_exists = bool(client.indices.exists(index=resolved_index_name))
    except Exception as exc:
        raise LlamaIndexBM25DependencyError(
            f"Failed to access Elasticsearch BM25 index '{resolved_index_name}'."
        ) from exc

    if not index_exists:
        raise FileNotFoundError(
            f"BM25 retriever index not found in Elasticsearch: {resolved_index_name}"
        )

    return ElasticSearchBM25Retriever(
        similarity_top_k=similarity_top_k,
        index_name=resolved_index_name,
        client=client,
    )


def load_bm25_docstore(storage_dir: str | None = None):
    del storage_dir
    raise NotImplementedError(
        "BM25 docstore persistence now uses Elasticsearch and no longer exposes a local docstore."
    )


def persist_bm25_docstore(docstore: Any, storage_dir: str | None = None) -> None:
    del docstore, storage_dir
    raise NotImplementedError(
        "BM25 docstore persistence now uses Elasticsearch and no longer writes local files."
    )


def _build_node_id(path: str, chunk_index: int, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{path or 'document'}::{chunk_index}::{digest}"


def _build_elasticsearch_client() -> Any:
    client_class = _get_elasticsearch_client_class()
    signature = inspect.signature(client_class.__init__)
    parameters = signature.parameters

    urls = [
        item.strip()
        for item in settings.BM25_ELASTICSEARCH_URL.split(",")
        if item.strip()
    ] or ["http://localhost:9200"]

    kwargs: dict[str, Any] = {}
    if "hosts" in parameters:
        kwargs["hosts"] = urls

    auth = None
    if settings.BM25_ELASTICSEARCH_USERNAME and settings.BM25_ELASTICSEARCH_PASSWORD:
        auth = (
            settings.BM25_ELASTICSEARCH_USERNAME,
            settings.BM25_ELASTICSEARCH_PASSWORD,
        )
    if auth:
        if "basic_auth" in parameters:
            kwargs["basic_auth"] = auth
        elif "http_auth" in parameters:
            kwargs["http_auth"] = auth

    if settings.BM25_ELASTICSEARCH_API_KEY and "api_key" in parameters:
        kwargs["api_key"] = settings.BM25_ELASTICSEARCH_API_KEY
    if settings.BM25_ELASTICSEARCH_CA_CERTS and "ca_certs" in parameters:
        kwargs["ca_certs"] = settings.BM25_ELASTICSEARCH_CA_CERTS
    if "verify_certs" in parameters:
        kwargs["verify_certs"] = settings.BM25_ELASTICSEARCH_VERIFY_CERTS
    if "request_timeout" in parameters:
        kwargs["request_timeout"] = settings.BM25_ELASTICSEARCH_TIMEOUT

    return client_class(**kwargs)


def _ensure_bm25_index(client: Any, index_name: str) -> None:
    try:
        if client.indices.exists(index=index_name):
            return

        client.indices.create(
            index=index_name,
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "bm25_whitespace_analyzer": {
                                "type": "custom",
                                "tokenizer": "whitespace",
                                "filter": ["lowercase"],
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "node_id": {"type": "keyword"},
                        "text": {"type": "text"},
                        "text_bm25": {
                            "type": "text",
                            "analyzer": "bm25_whitespace_analyzer",
                        },
                        "title": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "title_bm25": {
                            "type": "text",
                            "analyzer": "bm25_whitespace_analyzer",
                        },
                        "path": {"type": "keyword", "ignore_above": 2048},
                        "source": {"type": "keyword", "ignore_above": 2048},
                        "metadata": {"type": "object", "dynamic": True},
                    }
                },
            },
        )
    except Exception as exc:
        raise LlamaIndexBM25DependencyError(
            f"Failed to prepare Elasticsearch BM25 index '{index_name}'."
        ) from exc


def _serialize_node_for_elasticsearch(node: TextNode) -> dict[str, Any]:
    metadata = dict(node.metadata or {})
    path = str(
        metadata.get("path")
        or metadata.get("source")
        or metadata.get("file_path")
        or ""
    ).strip()
    title = str(metadata.get("title") or metadata.get("file_name") or "").strip()

    if path:
        metadata.setdefault("path", path)
        metadata.setdefault("source", path)
    if path and not title:
        title = Path(path).stem
    if title:
        metadata.setdefault("title", title)

    text = str(node.text or "").strip()
    return {
        "node_id": node.node_id,
        "text": text,
        "text_bm25": _tokenize_text_for_bm25(text),
        "title": title,
        "title_bm25": _tokenize_text_for_bm25(title),
        "path": path,
        "source": str(metadata.get("source") or path).strip(),
        "metadata": metadata,
    }


def _extract_metadata_from_source(source: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(source.get("metadata") or {})

    path = str(
        source.get("path")
        or metadata.get("path")
        or metadata.get("source")
        or metadata.get("file_path")
        or ""
    ).strip()
    title = str(
        source.get("title")
        or metadata.get("title")
        or metadata.get("file_name")
        or ""
    ).strip()

    if path:
        metadata.setdefault("path", path)
        metadata.setdefault("source", path)
    if path and not title:
        title = Path(path).stem
    if title:
        metadata.setdefault("title", title)

    return metadata


def _tokenize_text_for_bm25(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""

    return " ".join(
        token.strip()
        for token in jieba.lcut_for_search(normalized)
        if token.strip()
    )


def _build_search_body(query: str, similarity_top_k: int) -> dict[str, Any]:
    tokenized_query = _tokenize_text_for_bm25(query)
    should_clauses: list[dict[str, Any]] = []

    if tokenized_query:
        should_clauses.extend(
            [
                {"match": {"text_bm25": {"query": tokenized_query, "boost": 1.0}}},
                {"match": {"title_bm25": {"query": tokenized_query, "boost": 1.5}}},
            ]
        )

    raw_query = str(query or "").strip()
    if raw_query:
        should_clauses.extend(
            [
                {"match": {"text": {"query": raw_query, "boost": 0.15}}},
                {"match": {"title": {"query": raw_query, "boost": 0.3}}},
            ]
        )

    return {
        "size": similarity_top_k,
        "track_total_hits": False,
        "query": {
            "bool": {
                "should": should_clauses or [{"match_none": {}}],
                "minimum_should_match": 1,
            }
        },
    }
