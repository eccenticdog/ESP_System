from __future__ import annotations

import re
from pathlib import Path

import jieba
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from config.settings import settings
from repositories.file_repository import FileRepository
from utils.markdown_utils import MarkDownUtils


def _normalize_text(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _tokenize(value: str) -> list[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return []

    parts = re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", normalized)
    tokens: list[str] = []

    for part in parts:
        if not part:
            continue

        if re.search(r"[\u4e00-\u9fff]", part):
            tokens.extend(token for token in jieba.lcut(part) if token.strip())
        else:
            tokens.append(part)

    # Keep token order stable while deduplicating.
    return list(dict.fromkeys(tokens))


class TitleKeywordRetriever(BaseRetriever):
    """Title-oriented keyword retriever for LlamaIndex hybrid search."""

    def __init__(
        self,
        nodes: list[TextNode],
        similarity_top_k: int = 5,
    ) -> None:
        super().__init__()
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k

    @classmethod
    def from_crawl_directory(
        cls,
        crawl_directory: str | None = None,
        similarity_top_k: int = 5,
    ) -> "TitleKeywordRetriever":
        file_repository = FileRepository()
        target_directory = crawl_directory or settings.CRAWL_OUTPUT_DIR
        file_paths = file_repository.list_files(target_directory, extension=".md")
        unique_file_paths = file_repository.remove_duplicate_files(file_paths)

        nodes: list[TextNode] = []
        for file_path in unique_file_paths:
            content = file_repository.read_file_content(file_path)
            if not content.strip():
                continue

            title = MarkDownUtils.extract_title(file_path)
            nodes.append(
                TextNode(
                    text=f"文档标题:{title}\n{content}",
                    metadata={
                        "title": title,
                        "path": file_path,
                        "source": file_path,
                    },
                )
            )

        return cls(nodes=nodes, similarity_top_k=similarity_top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        query_terms = set(_tokenize(query))
        scored_nodes: list[NodeWithScore] = []

        for node in self._nodes:
            title = str(node.metadata.get("title") or Path(node.metadata.get("path", "")).stem)
            score = self._score_title(query=query, query_terms=query_terms, title=title)
            if score <= 0:
                continue

            scored_nodes.append(NodeWithScore(node=node, score=score))

        scored_nodes.sort(key=lambda item: item.score or 0.0, reverse=True)
        return scored_nodes[: self._similarity_top_k]

    @staticmethod
    def _score_title(query: str, query_terms: set[str], title: str) -> float:
        normalized_query = _normalize_text(query)
        normalized_title = _normalize_text(title)
        if not normalized_query or not normalized_title:
            return 0.0

        score = 0.0

        if normalized_query == normalized_title:
            score += 8.0
        if normalized_query in normalized_title:
            score += 4.0
        if normalized_title in normalized_query:
            score += 2.0

        title_terms = set(_tokenize(normalized_title))
        if query_terms and title_terms:
            overlap = query_terms & title_terms
            if overlap:
                score += 3.0 * (len(overlap) / len(query_terms))
                score += 2.0 * (len(overlap) / len(title_terms))

        return score
