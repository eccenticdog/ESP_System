import logging
from typing import Any, List

from langchain_core.documents import Document

from config.settings import settings


logger = logging.getLogger(__name__)


class PromptCompressionService:
    """使用 LLMLingua 对 RAG 检索上下文做提示压缩。"""

    def __init__(self):
        self.enabled = settings.ENABLE_RAG_PROMPT_COMPRESSION
        self._compressor = None
        self._compressor_initialized = False

    def build_context(self, user_question: str, retrieval_context: List[Document]) -> str:
        formatted_contexts = [
            self._format_document(index=index, document=document)
            for index, document in enumerate(retrieval_context)
        ]
        original_context = "\n\n".join(formatted_contexts)

        if not self.enabled or not formatted_contexts:
            return original_context

        compressor = self._get_compressor()
        if compressor is None:
            return original_context

        try:
            compression_result = compressor.compress_prompt(
                formatted_contexts,
                question=user_question,
                rate=settings.LLMLINGUA_RATE,
                condition_in_question=settings.LLMLINGUA_CONDITION_IN_QUESTION,
                reorder_context=settings.LLMLINGUA_REORDER_CONTEXT,
                dynamic_context_compression_ratio=settings.LLMLINGUA_DYNAMIC_CONTEXT_COMPRESSION_RATIO,
                condition_compare=settings.LLMLINGUA_CONDITION_COMPARE,
                context_budget=settings.LLMLINGUA_CONTEXT_BUDGET,
                rank_method=settings.LLMLINGUA_RANK_METHOD,
            )
        except Exception:
            logger.exception(
                "LLMLingua prompt compression failed, fallback to original retrieval context."
            )
            return original_context

        compressed_context = str(compression_result.get("compressed_prompt", "")).strip()
        if not compressed_context:
            logger.warning(
                "LLMLingua returned an empty compressed prompt, fallback to original retrieval context."
            )
            return original_context

        self._log_compression_result(compression_result)
        return compressed_context

    def _get_compressor(self):
        if self._compressor_initialized:
            return self._compressor

        self._compressor_initialized = True

        try:
            from llmlingua import PromptCompressor

            if settings.LLMLINGUA_MODEL_NAME:
                self._compressor = PromptCompressor(
                    model_name=settings.LLMLINGUA_MODEL_NAME
                )
            else:
                self._compressor = PromptCompressor()
        except ImportError:
            logger.warning("LLMLingua is not installed, prompt compression is disabled.")
            self._compressor = None
        except Exception:
            logger.exception(
                "Failed to initialize LLMLingua compressor, prompt compression is disabled."
            )
            self._compressor = None

        return self._compressor

    @staticmethod
    def _format_document(index: int, document: Document) -> str:
        title = str(document.metadata.get("title", "")).strip()
        path = str(document.metadata.get("path", "")).strip()

        header = f"资料{index + 1}"
        if title:
            header = f"{header} | 标题: {title}"
        if path:
            header = f"{header} | 路径: {path}"

        return f"{header}\n{document.page_content.strip()}"

    @staticmethod
    def _log_compression_result(compression_result: dict[str, Any]) -> None:
        logger.info(
            "LLMLingua compressed retrieval context: origin_tokens=%s, compressed_tokens=%s, ratio=%s",
            compression_result.get("origin_tokens"),
            compression_result.get("compressed_tokens"),
            compression_result.get("ratio"),
        )
