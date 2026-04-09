import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from llama_index.core.schema import QueryBundle

from services.llamaindex_bm25_retriever import (
    ElasticSearchBM25Retriever,
    build_nodes_from_documents,
    load_bm25_retriever,
    store_documents_for_bm25,
)


class BM25RetrieverHelpersTest(unittest.TestCase):
    def test_build_nodes_from_documents_preserves_metadata(self):
        nodes = build_nodes_from_documents(
            [
                Document(
                    page_content="document content A",
                    metadata={"source": "data/a.md", "title": "title-a"},
                ),
                Document(
                    page_content="document content B",
                    metadata={"file_path": "data/b.md"},
                ),
            ]
        )

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].metadata["path"], "data/a.md")
        self.assertEqual(nodes[0].metadata["source"], "data/a.md")
        self.assertEqual(nodes[0].metadata["title"], "title-a")
        self.assertEqual(nodes[1].metadata["title"], "b")
        self.assertTrue(nodes[0].node_id)
        self.assertTrue(nodes[1].node_id)

    @patch("services.llamaindex_bm25_retriever._get_elasticsearch_helpers_module")
    @patch("services.llamaindex_bm25_retriever._build_elasticsearch_client")
    def test_store_documents_for_bm25_indexes_documents_in_elasticsearch(
        self,
        mock_build_client,
        mock_helpers_module,
    ):
        client = MagicMock()
        client.indices.exists.return_value = True
        mock_build_client.return_value = client
        helpers_module = MagicMock()
        mock_helpers_module.return_value = helpers_module

        store_documents_for_bm25(
            [
                Document(
                    page_content="document content A",
                    metadata={"source": "data/a.md", "title": "title-a"},
                )
            ]
        )

        client.delete_by_query.assert_called_once()
        helpers_module.bulk.assert_called_once()
        actions = helpers_module.bulk.call_args.args[1]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["_source"]["path"], "data/a.md")
        self.assertEqual(actions[0]["_source"]["title"], "title-a")
        self.assertIn("document", actions[0]["_source"]["text_bm25"])

    @patch("services.llamaindex_bm25_retriever._build_elasticsearch_client")
    def test_load_bm25_retriever_raises_when_index_missing(self, mock_build_client):
        client = MagicMock()
        client.indices.exists.return_value = False
        mock_build_client.return_value = client

        with self.assertRaises(FileNotFoundError):
            load_bm25_retriever()

    def test_elasticsearch_bm25_retriever_maps_hits_to_nodes(self):
        client = MagicMock()
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "doc-1",
                        "_score": 12.5,
                        "_source": {
                            "text": "document content A",
                            "path": "data/a.md",
                            "title": "title-a",
                            "metadata": {"source": "data/a.md"},
                        },
                    }
                ]
            }
        }

        retriever = ElasticSearchBM25Retriever(
            client=client,
            index_name="its-knowledge-bm25",
        )
        results = retriever._retrieve(QueryBundle("query"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].node.text, "document content A")
        self.assertEqual(results[0].node.metadata["path"], "data/a.md")
        self.assertEqual(results[0].score, 12.5)


if __name__ == "__main__":
    unittest.main()
