from __future__ import annotations

import logging

import requests
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from config.settings import settings


logger = logging.getLogger(__name__)


class BGERerankerPostprocessor(BaseNodePostprocessor):
    """Rerank retrieved nodes with a BGE reranker API."""

    model_name: str = settings.BGE_RERANKER_MODEL
    top_n: int = settings.BGE_RERANKER_TOP_N
    api_url: str = settings.BGE_RERANKER_API_URL
    api_key: str | None = settings.API_KEY

    def __init__(
        self,
        model_name: str | None = None,
        top_n: int | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name or settings.BGE_RERANKER_MODEL
        self.top_n = top_n or settings.BGE_RERANKER_TOP_N
        self.api_url = (
            api_url
            or settings.BGE_RERANKER_API_URL
            or (
                f"{settings.BASE_URL.rstrip('/')}/rerank"
                if settings.BASE_URL
                else ""
            )
        )
        self.api_key = settings.API_KEY if api_key is None else api_key

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None or not query_bundle.query_str.strip():
            return nodes[: self.top_n]

        try:
            reranked_nodes = self._rerank_nodes(nodes, query_bundle.query_str.strip())
        except Exception:
            logger.exception(
                "Failed to call the BGE reranker API. Falling back to fused retrieval order."
            )
            return nodes[: self.top_n]

        return reranked_nodes[: self.top_n]

    def _rerank_nodes(
        self,
        nodes: list[NodeWithScore],
        query: str,
    ) -> list[NodeWithScore]:
        documents = [self._node_text(node) for node in nodes]
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(self.top_n, len(documents)),
            "return_documents": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        response_payload = response.json()
        results = response_payload.get("results") or []

        reranked_nodes: list[NodeWithScore] = []
        for item in results:
            index = int(item["index"])
            score = float(item.get("relevance_score") or item.get("score") or 0.0)
            reranked_nodes.append(NodeWithScore(node=nodes[index].node, score=score))

        if not reranked_nodes:
            return nodes[: self.top_n]

        reranked_nodes.sort(key=lambda item: item.score or 0.0, reverse=True)
        return reranked_nodes

    @staticmethod
    def _node_text(node_with_score: NodeWithScore) -> str:
        node = node_with_score.node

        for attr_name in ("text", "page_content"):
            value = getattr(node, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        get_content = getattr(node, "get_content", None)
        if callable(get_content):
            for kwargs in ({}, {"metadata_mode": "none"}):
                try:
                    value = get_content(**kwargs)
                except TypeError:
                    continue
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""
