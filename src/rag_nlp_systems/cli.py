"""Command-line interface for indexing, grounded Q&A, and sentiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rag import KnowledgeBase, OllamaGenerator, RAGAgent
from .sentiment import SentimentPipeline, load_labeled_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="build and persist a retrieval index")
    index_parser.add_argument("--corpus", type=Path, required=True)
    index_parser.add_argument("--output", type=Path, required=True)

    ask_parser = commands.add_parser("ask", help="answer a question from indexed sources")
    source = ask_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=Path)
    source.add_argument("--corpus", type=Path)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument("--provider", choices=("extractive", "ollama"), default="extractive")
    ask_parser.add_argument("--model", default="qwen2.5:1.5b")

    train_parser = commands.add_parser("sentiment-train", help="train and evaluate sentiment NLP")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)

    predict_parser = commands.add_parser("sentiment-predict", help="classify one text input")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--text", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "index":
        knowledge_base = KnowledgeBase.from_directory(args.corpus)
        output = knowledge_base.save(args.output)
        print(json.dumps({"chunks": len(knowledge_base.chunks), "index": str(output)}, indent=2))
        return 0

    if args.command == "ask":
        knowledge_base = (
            KnowledgeBase.load(args.index)
            if args.index
            else KnowledgeBase.from_directory(args.corpus)
        )
        generator = OllamaGenerator(model=args.model) if args.provider == "ollama" else None
        result = RAGAgent(knowledge_base, generator=generator).answer(args.question, top_k=args.top_k)
        print(
            json.dumps(
                {
                    "question": result.question,
                    "answer": result.answer,
                    "citations": list(result.citations),
                    "retrieval": [
                        {
                            "source": item.source,
                            "chunk_id": item.chunk_id,
                            "score": round(item.score, 4),
                        }
                        for item in result.results
                    ],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "sentiment-train":
        texts, labels = load_labeled_csv(args.data)
        pipeline = SentimentPipeline()
        evaluation = pipeline.evaluate(texts, labels)
        pipeline.fit(texts, labels).save(args.output)
        print(
            json.dumps(
                {
                    "accuracy": round(evaluation.accuracy, 4),
                    "train_rows": evaluation.train_rows,
                    "test_rows": evaluation.test_rows,
                    "model": str(args.output.expanduser().resolve()),
                },
                indent=2,
            )
        )
        return 0

    pipeline = SentimentPipeline.load(args.model)
    print(json.dumps(pipeline.predict([args.text])[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
