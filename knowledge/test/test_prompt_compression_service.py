import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from services.prompt_compression_service import PromptCompressionService


class FakeCompressor:
    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def compress_prompt(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.result


class PromptCompressionServiceTest(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                page_content="第一段上下文内容",
                metadata={"title": "文档A", "path": "data/a.md"},
            ),
            Document(
                page_content="第二段上下文内容",
                metadata={"title": "文档B", "path": "data/b.md"},
            ),
        ]

    def test_returns_compressed_prompt_when_llmlingua_succeeds(self):
        service = PromptCompressionService()
        fake_compressor = FakeCompressor(
            result={
                "compressed_prompt": "压缩后的资料1\n压缩后的资料2",
                "origin_tokens": 200,
                "compressed_tokens": 80,
                "ratio": "2.5x",
            }
        )

        with patch.object(service, "_get_compressor", return_value=fake_compressor):
            compressed_context = service.build_context("怎么处理蓝屏？", self.documents)

        self.assertEqual(compressed_context, "压缩后的资料1\n压缩后的资料2")

    def test_falls_back_to_original_context_when_llmlingua_fails(self):
        service = PromptCompressionService()
        fake_compressor = FakeCompressor(error=RuntimeError("compression failed"))

        with patch.object(service, "_get_compressor", return_value=fake_compressor):
            compressed_context = service.build_context("怎么处理蓝屏？", self.documents)

        self.assertIn("资料1 | 标题: 文档A | 路径: data/a.md", compressed_context)
        self.assertIn("第一段上下文内容", compressed_context)
        self.assertIn("资料2 | 标题: 文档B | 路径: data/b.md", compressed_context)

    def test_skips_compression_when_disabled(self):
        service = PromptCompressionService()

        with patch.object(service, "enabled", False):
            compressed_context = service.build_context("怎么处理蓝屏？", self.documents)

        self.assertIn("资料1 | 标题: 文档A | 路径: data/a.md", compressed_context)
        self.assertIn("第二段上下文内容", compressed_context)


if __name__ == "__main__":
    unittest.main()
