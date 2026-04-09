from typing import List

from langchain_core.documents import Document

from services.llamaindex_query_engine_service import (
    LlamaIndexQueryEngineService,
    QueryEngineResult,
)


class QueryService:
    """Knowledge query service backed by LlamaIndex QueryEngine."""

    def __init__(self, query_engine_service: LlamaIndexQueryEngineService | None = None):
        self.query_engine_service = query_engine_service or LlamaIndexQueryEngineService()

    def query(self, user_question: str) -> QueryEngineResult:
        return self.query_engine_service.query(user_question)

    def generate_answer(
        self,
        user_question: str,
        retrival_context: List[Document] | None = None,
    ) -> str:
        return self.query(user_question).answer
