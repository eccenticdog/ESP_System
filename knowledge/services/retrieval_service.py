from langchain_core.documents import Document

from services.llamaindex_query_engine_service import LlamaIndexQueryEngineService


class RetrievalService:
    """Knowledge retrieval service backed by LlamaIndex QueryEngine."""

    def __init__(self, query_engine_service: LlamaIndexQueryEngineService | None = None):
        self.query_engine_service = query_engine_service or LlamaIndexQueryEngineService()

    def retrieval(self, user_question: str) -> list[Document]:
        return self.query_engine_service.retrieve(user_question)
