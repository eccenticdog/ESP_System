import unittest
from unittest.mock import MagicMock, patch

from services.llamaindex_query_engine_service import LlamaIndexQueryEngineService


class _FakeNode:
    def __init__(self, text: str, metadata: dict | None = None):
        self.text = text
        self.metadata = metadata or {}


class _FakeSourceNode:
    def __init__(self, node):
        self.node = node


class LlamaIndexQueryEngineServiceTest(unittest.TestCase):
    def test_source_nodes_to_documents_maps_metadata_and_deduplicates(self):
        source_nodes = [
            _FakeSourceNode(
                _FakeNode(
                    text="文档内容 A",
                    metadata={"source": "data/a.md", "title": "标题 A"},
                )
            ),
            _FakeSourceNode(
                _FakeNode(
                    text="文档内容 A",
                    metadata={"source": "data/a.md", "title": "标题 A"},
                )
            ),
            _FakeSourceNode(
                _FakeNode(
                    text="文档内容 B",
                    metadata={"file_path": "data/b.md"},
                )
            ),
        ]

        documents = LlamaIndexQueryEngineService._source_nodes_to_documents(source_nodes)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["path"], "data/a.md")
        self.assertEqual(documents[0].metadata["title"], "标题 A")
        self.assertEqual(documents[1].metadata["path"], "data/b.md")
        self.assertEqual(documents[1].metadata["title"], "b")

    @patch("services.llamaindex_query_engine_service.BGERerankerPostprocessor")
    @patch("services.llamaindex_query_engine_service.load_bm25_retriever")
    @patch("llama_index.core.retrievers.QueryFusionRetriever")
    @patch("llama_index.core.query_engine.RetrieverQueryEngine.from_args")
    @patch("llama_index.core.VectorStoreIndex.from_vector_store")
    def test_build_query_engine_uses_hybrid_retrieval(
        self,
        mock_index_from_vector_store,
        mock_query_engine_from_args,
        mock_query_fusion_retriever,
        mock_load_bm25_retriever,
        mock_bge_reranker,
    ):
        service = LlamaIndexQueryEngineService()
        service._create_embedding_model = MagicMock(return_value="embedding")
        service._create_llm = MagicMock(return_value="llm")

        vector_retriever = MagicMock()
        index = MagicMock()
        index.as_retriever.return_value = vector_retriever
        mock_index_from_vector_store.return_value = index
        vector_store = MagicMock()
        service._get_vector_store = MagicMock(return_value=vector_store)

        bm25_retriever = MagicMock()
        mock_load_bm25_retriever.return_value = bm25_retriever
        reranker = MagicMock()
        mock_bge_reranker.return_value = reranker
        hybrid_retriever = MagicMock()
        mock_query_fusion_retriever.return_value = hybrid_retriever

        service._build_query_engine()

        service._get_vector_store.assert_called_once()
        mock_query_fusion_retriever.assert_called_once()
        hybrid_kwargs = mock_query_fusion_retriever.call_args.kwargs
        self.assertEqual(hybrid_kwargs["retrievers"], [bm25_retriever, vector_retriever])
        self.assertEqual(hybrid_kwargs["retriever_weights"], [0.35, 0.65])
        mock_query_engine_from_args.assert_called_once_with(
            retriever=hybrid_retriever,
            llm="llm",
            node_postprocessors=[reranker],
        )

    @patch("services.llamaindex_query_engine_service.BGERerankerPostprocessor")
    @patch("services.llamaindex_query_engine_service.load_bm25_retriever", side_effect=FileNotFoundError)
    @patch("llama_index.core.query_engine.RetrieverQueryEngine.from_args")
    @patch("llama_index.core.VectorStoreIndex.from_vector_store")
    def test_build_query_engine_falls_back_to_vector_only_when_bm25_missing(
        self,
        mock_index_from_vector_store,
        mock_query_engine_from_args,
        _mock_load_bm25_retriever,
        mock_bge_reranker,
    ):
        service = LlamaIndexQueryEngineService()
        service._create_embedding_model = MagicMock(return_value="embedding")
        service._create_llm = MagicMock(return_value="llm")

        vector_retriever = MagicMock()
        index = MagicMock()
        index.as_retriever.return_value = vector_retriever
        mock_index_from_vector_store.return_value = index
        service._get_vector_store = MagicMock(return_value=MagicMock())

        reranker = MagicMock()
        mock_bge_reranker.return_value = reranker

        service._build_query_engine()

        mock_query_engine_from_args.assert_called_once_with(
            retriever=vector_retriever,
            llm="llm",
            node_postprocessors=[reranker],
        )


if __name__ == "__main__":
    unittest.main()
