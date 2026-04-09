from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config.settings import settings
from services.llamaindex_bge_reranker_postprocessor import BGERerankerPostprocessor
from services.llamaindex_bm25_retriever import (
    LlamaIndexBM25DependencyError,
    load_bm25_retriever,
)
from utils.vector_store import (
    get_vector_store_collection_name,
    get_vector_store_connection_args,
    get_vector_store_dimension,
    get_vector_store_embedding_model_name,
)


logger = logging.getLogger(__name__)


class LlamaIndexDependencyError(RuntimeError):
    """Raised when the LlamaIndex runtime dependencies are unavailable."""


@dataclass(slots=True)
class QueryEngineResult:
    answer: str
    documents: list[Document]


class LlamaIndexQueryEngineService:
    """Adapter around an ElasticSearch BM25 + BGE vector hybrid QueryEngine."""

    EMPTY_ANSWER = "当前知识库中暂时没有找到该问题的解决方案。"

    def __init__(self):
        self._milvus_client = None
        self._vector_store = None
        self._query_engine = None
        self._bm25_retriever = None
        self._reranker_postprocessor = None

    def query(self, question: str) -> QueryEngineResult:
        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        if not self._collection_exists():
            return QueryEngineResult(answer=self.EMPTY_ANSWER, documents=[])

        query_engine = self._get_query_engine()
        response = query_engine.query(normalized_question)
        documents = self._source_nodes_to_documents(getattr(response, "source_nodes", []))

        if not documents:
            return QueryEngineResult(answer=self.EMPTY_ANSWER, documents=[])

        answer = str(getattr(response, "response", "") or response).strip()
        if not answer:
            answer = self.EMPTY_ANSWER

        return QueryEngineResult(answer=answer, documents=documents)

    def retrieve(self, question: str) -> list[Document]:
        return self.query(question).documents

    def _get_query_engine(self):
        if self._query_engine is None:
            self._query_engine = self._build_query_engine()
        return self._query_engine

    def _get_milvus_client(self):
        if self._milvus_client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise LlamaIndexDependencyError(
                    "The Milvus client is not installed."
                ) from exc

            self._milvus_client = MilvusClient(**get_vector_store_connection_args())

        return self._milvus_client

    def _collection_exists(self) -> bool:
        collection_name = get_vector_store_collection_name()
        return bool(
            self._get_milvus_client().has_collection(collection_name=collection_name)
        )

    def _get_vector_store(self):
        if self._vector_store is None:
            try:
                from llama_index.vector_stores.milvus import MilvusVectorStore
            except ImportError as exc:
                raise LlamaIndexDependencyError(
                    "The Milvus LlamaIndex integration is not installed."
                ) from exc

            vector_store_kwargs = {
                **get_vector_store_connection_args(),
                "collection_name": get_vector_store_collection_name(),
                "dim": get_vector_store_dimension(),
                "similarity_metric": settings.VECTOR_STORE_SIMILARITY_METRIC,
            }
            self._vector_store = MilvusVectorStore(**vector_store_kwargs)

        return self._vector_store

    def _build_query_engine(self):
        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import QueryFusionRetriever
            from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
        except ImportError as exc:
            raise LlamaIndexDependencyError(
                "LlamaIndex dependencies are missing. Install "
                "'llama-index-core', 'llama-index-llms-openai', "
                "'llama-index-embeddings-openai', and "
                "'llama-index-vector-stores-milvus'."
            ) from exc

        index = VectorStoreIndex.from_vector_store(
            vector_store=self._get_vector_store(),
            embed_model=self._create_embedding_model(),
        )

        vector_retriever = index.as_retriever(
            similarity_top_k=max(settings.TOP_FINAL, min(settings.TOP_ROUGH, 10)),
        )
        bm25_retriever = self._get_bm25_retriever()
        retriever = vector_retriever

        if bm25_retriever is not None:
            retriever = QueryFusionRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                llm=None,
                mode=FUSION_MODES.RECIPROCAL_RANK,
                similarity_top_k=settings.TOP_FINAL,
                num_queries=1,
                use_async=False,
                retriever_weights=[0.35, 0.65],
            )

        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=self._create_llm(),
            node_postprocessors=self._get_node_postprocessors(),
        )

    def _get_bm25_retriever(self):
        if self._bm25_retriever is None:
            try:
                self._bm25_retriever = load_bm25_retriever(
                    similarity_top_k=max(settings.TOP_FINAL, min(settings.TOP_ROUGH, 10)),
                )
            except (FileNotFoundError, LlamaIndexBM25DependencyError):
                logger.warning(
                    "Persisted BM25 retriever is unavailable. Falling back to vector-only retrieval.",
                    exc_info=True,
                )
                self._bm25_retriever = False

        if self._bm25_retriever is False:
            return None

        return self._bm25_retriever

    def _get_node_postprocessors(self) -> list[Any] | None:
        if not settings.ENABLE_BGE_RERANKER:
            return None

        if self._reranker_postprocessor is None:
            self._reranker_postprocessor = BGERerankerPostprocessor(
                model_name=settings.BGE_RERANKER_MODEL,
                top_n=settings.BGE_RERANKER_TOP_N,
                api_url=settings.BGE_RERANKER_API_URL,
                api_key=settings.API_KEY,
            )

        return [self._reranker_postprocessor]

    def _create_llm(self):
        try:
            from langchain_openai import ChatOpenAI
            from llama_index.llms.langchain import LangChainLLM
        except ImportError as exc:
            raise LlamaIndexDependencyError(
                "The LlamaIndex LangChain LLM bridge is not installed."
            ) from exc

        llm = ChatOpenAI(
            model_name=settings.MODEL,
            openai_api_key=settings.API_KEY,
            openai_api_base=settings.BASE_URL,
            temperature=0,
        )
        return LangChainLLM(llm=llm)

    def _create_embedding_model(self):
        try:
            from langchain_openai import OpenAIEmbeddings
            from llama_index.embeddings.langchain import LangchainEmbedding
        except ImportError as exc:
            raise LlamaIndexDependencyError(
                "The LlamaIndex LangChain embedding bridge is not installed."
            ) from exc

        embedding = OpenAIEmbeddings(
            model=get_vector_store_embedding_model_name(),
            openai_api_key=settings.API_KEY,
            openai_api_base=settings.BASE_URL,
        )
        return LangchainEmbedding(langchain_embeddings=embedding)

    @staticmethod
    def _build_openai_kwargs(factory: type[Any], **base_kwargs: Any) -> dict[str, Any]:
        signature = inspect.signature(factory.__init__)
        parameters = signature.parameters

        kwargs = {
            key: value
            for key, value in base_kwargs.items()
            if value is not None and key in parameters
        }

        if settings.API_KEY:
            if "api_key" in parameters:
                kwargs["api_key"] = settings.API_KEY
            elif "openai_api_key" in parameters:
                kwargs["openai_api_key"] = settings.API_KEY

        if settings.BASE_URL:
            if "api_base" in parameters:
                kwargs["api_base"] = settings.BASE_URL
            elif "base_url" in parameters:
                kwargs["base_url"] = settings.BASE_URL
            elif "openai_api_base" in parameters:
                kwargs["openai_api_base"] = settings.BASE_URL

        return kwargs

    @staticmethod
    def _extract_node_text(node: Any) -> str:
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

    @classmethod
    def _source_nodes_to_documents(cls, source_nodes: list[Any]) -> list[Document]:
        documents: list[Document] = []
        seen: set[tuple[str, str, str]] = set()

        for source_node in source_nodes or []:
            node = getattr(source_node, "node", source_node)
            metadata = dict(getattr(node, "metadata", {}) or {})

            path = str(
                metadata.get("path")
                or metadata.get("source")
                or metadata.get("file_path")
                or ""
            ).strip()
            title = str(metadata.get("title") or metadata.get("file_name") or "").strip()

            if path and not title:
                title = Path(path).stem

            if path:
                metadata.setdefault("path", path)
            if title:
                metadata.setdefault("title", title)

            text = cls._extract_node_text(node)
            if not text:
                continue

            dedupe_key = (
                str(metadata.get("title", "")),
                str(metadata.get("path", "")),
                text[:100],
            )
            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            documents.append(Document(page_content=text, metadata=metadata))

        return documents[: settings.TOP_FINAL]
