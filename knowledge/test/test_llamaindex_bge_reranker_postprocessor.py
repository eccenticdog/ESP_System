import unittest
from unittest.mock import patch

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from services.llamaindex_bge_reranker_postprocessor import BGERerankerPostprocessor


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.12},
            ]
        }


class BGERerankerPostprocessorTest(unittest.TestCase):
    @patch("services.llamaindex_bge_reranker_postprocessor.requests.post", return_value=_FakeResponse())
    def test_postprocess_nodes_reranks_by_api_score(self, _mock_post):
        postprocessor = BGERerankerPostprocessor(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=2,
            api_url="https://example.test/rerank",
            api_key="test-key",
        )
        nodes = [
            NodeWithScore(node=TextNode(text="Outlook 文档"), score=0.8),
            NodeWithScore(node=TextNode(text="Word 默认样式文档"), score=0.2),
        ]

        reranked = postprocessor.postprocess_nodes(
            nodes,
            query_bundle=QueryBundle("如何修改 Microsoft Word 的默认样式？"),
        )

        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].node.text, "Word 默认样式文档")
        self.assertGreater(reranked[0].score, reranked[1].score)


if __name__ == "__main__":
    unittest.main()
