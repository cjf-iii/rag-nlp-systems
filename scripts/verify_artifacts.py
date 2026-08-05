#!/usr/bin/env python3
"""Verify artifacts produced locally by the documented CLI workflow."""

from __future__ import annotations

import json
from pathlib import Path

from rag_nlp_systems.rag import KnowledgeBase, RAGAgent
from rag_nlp_systems.sentiment import SentimentPipeline


def main() -> None:
    artifacts = Path("artifacts")
    index_path = artifacts / "knowledge.joblib"
    sentiment_path = artifacts / "sentiment.joblib"
    if not index_path.is_file() or not sentiment_path.is_file():
        raise RuntimeError("the index and sentiment artifacts must both exist")

    # These joblib files were created immediately beforehand from this checkout.
    knowledge_base = KnowledgeBase.load(index_path)
    rag_answer = RAGAgent(knowledge_base).answer(
        "How does batching affect inference capacity?",
        top_k=2,
    )
    if "batch" not in rag_answer.answer.lower() or "inference.md" not in rag_answer.citations:
        raise RuntimeError("the persisted index did not produce the documented grounded answer")

    sentiment = SentimentPipeline.load(sentiment_path).predict(
        ["The benchmark was clear and reproducible"]
    )[0]
    if sentiment["label"] != "positive":
        raise RuntimeError("the persisted sentiment model failed the documented positive example")

    print(
        json.dumps(
            {
                "chunks": len(knowledge_base.chunks),
                "citations": list(rag_answer.citations),
                "sentiment_label": sentiment["label"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
