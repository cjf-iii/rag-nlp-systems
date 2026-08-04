from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_nlp_systems.rag import KnowledgeBase, RAGAgent


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "inference.md").write_text(
            "Inference latency measures the time required to return a model response. "
            "Batching can improve throughput when several requests share compute.",
            encoding="utf-8",
        )
        (self.root / "evaluation.md").write_text(
            "A held-out test set estimates generalization on unseen examples. "
            "Cross-validation measures stability across several train-test partitions.",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_retrieval_returns_the_relevant_source(self) -> None:
        knowledge_base = KnowledgeBase.from_directory(self.root)
        result = knowledge_base.retrieve("How does batching affect inference throughput?", top_k=1)[0]
        self.assertEqual(result.source, "inference.md")
        self.assertGreater(result.score, 0)

    def test_index_round_trip_preserves_results(self) -> None:
        knowledge_base = KnowledgeBase.from_directory(self.root)
        index_path = self.root / "index.joblib"
        knowledge_base.save(index_path)
        loaded = KnowledgeBase.load(index_path)
        self.assertEqual(
            loaded.retrieve("held-out test set", top_k=1)[0].source,
            "evaluation.md",
        )

    def test_agent_returns_citations(self) -> None:
        answer = RAGAgent(KnowledgeBase.from_directory(self.root)).answer(
            "What estimates generalization?",
            top_k=2,
        )
        self.assertIn("evaluation.md", answer.citations)
        self.assertIn("held-out", answer.answer.lower())


if __name__ == "__main__":
    unittest.main()
