import logging
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from repositories.vector_store_repository import VectorStoreRepository
from services.llamaindex_bm25_retriever import rebuild_bm25_index, store_documents_for_bm25
from utils.markdown_utils import MarkDownUtils


logger = logging.getLogger(__name__)


class IngestionProcessor:
    """Split markdown documents, store vectors, and refresh BM25 artifacts."""

    def __init__(self):
        self.vector_store = VectorStoreRepository()
        self.document_spliter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n##", "\n**", "\n\n", "\n", " ", ""],
        )

    def ingest_file(self, md_path: str, refresh_bm25: bool = True) -> int:
        documents = self._load_documents(md_path)

        for document in documents:
            document.metadata["title"] = MarkDownUtils.extract_title(md_path)

        final_document_chunks = []
        for document in documents:
            if len(document.page_content) < 3000:
                final_document_chunks.append(document)
                continue

            document_chunks = self.document_spliter.split_documents([document])
            for chunk in document_chunks:
                chunk_path = chunk.metadata.get("source", md_path)
                title = os.path.basename(chunk_path)
                chunk.page_content = f"文档标题:{title}\n{chunk.page_content}"

            final_document_chunks.extend(document_chunks)

        clean_document_chunks = filter_complex_metadata(final_document_chunks)
        valid_document_chunks = [
            document
            for document in clean_document_chunks
            if str(document.page_content or "").strip()
        ]

        if not valid_document_chunks:
            logger.error("No valid document chunks were produced for %s", md_path)
            return 0

        total_document_chunks = self.vector_store.add_documents(valid_document_chunks)
        store_documents_for_bm25(valid_document_chunks)

        if refresh_bm25:
            self.rebuild_bm25_index()

        return total_document_chunks

    def rebuild_bm25_index(self) -> None:
        rebuild_bm25_index(
            similarity_top_k=max(settings.TOP_FINAL, min(settings.TOP_ROUGH, 10)),
        )

    @staticmethod
    def _load_documents(md_path: str):
        try:
            text_loader = TextLoader(file_path=md_path, encoding="utf-8")
            return text_loader.load()
        except Exception as exc:
            raise Exception(f"Failed to load markdown file {md_path}: {exc}") from exc


if __name__ == "__main__":
    ingest_processor = IngestionProcessor()
    result = ingest_processor.ingest_file(
        "G:\\BaiduNetdiskDownload\\初始化项目\\初始化项目\\代码\\bj250716\\its_multi_agent\\backend\\knowledge\\data\\crawl\\0031-如何分图层修改 Visio 流程图？.md"
    )
    print(result)
