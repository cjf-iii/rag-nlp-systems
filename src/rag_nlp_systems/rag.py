"""Retrieval, grounding, persistence, and optional local generation."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SUPPORTED_SUFFIXES = {".md", ".txt"}
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]+")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage with source provenance."""

    chunk_id: str
    source: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    """A passage and its cosine-similarity score."""

    chunk_id: str
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class RAGAnswer:
    """A grounded answer with the exact sources used to produce it."""

    question: str
    answer: str
    citations: tuple[str, ...]
    results: tuple[RetrievalResult, ...]


class Generator(Protocol):
    """Interface shared by deterministic and local-LLM generators."""

    def generate(self, question: str, context: str) -> str:
        """Generate an answer from a question and retrieved context."""


def _split_document(text: str, max_chars: int = 850) -> list[str]:
    """Split a document into paragraph-aware chunks with bounded size."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        if current and current_size + len(paragraph) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0

        if len(paragraph) <= max_chars:
            current.append(paragraph)
            current_size += len(paragraph) + 2
            continue

        sentences = SENTENCE_PATTERN.split(paragraph)
        for sentence in sentences:
            if current and current_size + len(sentence) + 1 > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_size = 0
            current.append(sentence)
            current_size += len(sentence) + 1

    if current:
        chunks.append("\n\n".join(current))

    return chunks


class KnowledgeBase:
    """A persisted TF-IDF retrieval index over Markdown and text files."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectorizer: TfidfVectorizer,
        matrix: object,
    ) -> None:
        if not chunks:
            raise ValueError("knowledge base requires at least one chunk")
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def from_directory(cls, corpus_dir: str | Path) -> "KnowledgeBase":
        """Index supported files below *corpus_dir* in stable path order."""

        root = Path(corpus_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"corpus directory not found: {root}")

        chunks: list[Chunk] = []
        files = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        for path in files:
            relative = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            for index, passage in enumerate(_split_document(text), start=1):
                chunks.append(Chunk(f"{relative}#{index}", relative, passage))

        if not chunks:
            raise ValueError(f"no Markdown or text content found under {root}")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=20_000,
        )
        matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
        return cls(chunks, vectorizer, matrix)

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        """Return the highest-scoring chunks for *query*."""

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query cannot be blank")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self.vectorizer.transform([clean_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = np.argsort(scores)[::-1][: min(top_k, len(self.chunks))]
        return [
            RetrievalResult(
                chunk_id=self.chunks[index].chunk_id,
                source=self.chunks[index].source,
                text=self.chunks[index].text,
                score=float(scores[index]),
            )
            for index in ranked
        ]

    def save(self, output_path: str | Path) -> Path:
        """Persist the fitted vectorizer, sparse matrix, and source chunks."""

        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "chunks": self.chunks,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, input_path: str | Path) -> "KnowledgeBase":
        """Load an index created by :meth:`save`."""

        payload = joblib.load(Path(input_path).expanduser().resolve())
        return cls(payload["chunks"], payload["vectorizer"], payload["matrix"])


class ExtractiveGenerator:
    """Offline answer generator that selects relevant source sentences."""

    def generate(self, question: str, context: str) -> str:
        context = re.sub(r"(?m)^\[[^\]]+\]\s*$", "", context)
        context = re.sub(r"(?m)^#{1,6}\s+.*$", "", context)
        query_terms = set(WORD_PATTERN.findall(question.lower()))
        sentences = [part.strip() for part in SENTENCE_PATTERN.split(context) if part.strip()]
        ranked = sorted(
            sentences,
            key=lambda sentence: (
                len(query_terms.intersection(WORD_PATTERN.findall(sentence.lower()))),
                -len(sentence),
            ),
            reverse=True,
        )
        selected = [sentence for sentence in ranked if sentence][:3]
        if not selected:
            return "The retrieved sources did not contain a usable answer."
        return " ".join(selected)


class OllamaGenerator:
    """Optional local generator using Ollama's loopback HTTP API."""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def generate(self, question: str, context: str) -> str:
        prompt = (
            "Answer only from the supplied context. If the context is insufficient, say so. "
            "Keep the answer concise and preserve factual qualifiers.\n\n"
            f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"
        )
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        answer = str(payload.get("response", "")).strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response")
        return answer


class RAGAgent:
    """Retrieve grounded context, generate an answer, and return citations."""

    def __init__(self, knowledge_base: KnowledgeBase, generator: Generator | None = None) -> None:
        self.knowledge_base = knowledge_base
        self.generator = generator or ExtractiveGenerator()

    def answer(self, question: str, top_k: int = 4) -> RAGAnswer:
        results = self.knowledge_base.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            f"[{result.chunk_id}]\n{result.text}" for result in results if result.score > 0
        )
        if not context:
            return RAGAnswer(
                question=question,
                answer="The indexed sources do not contain enough information to answer that question.",
                citations=(),
                results=tuple(results),
            )

        answer = self.generator.generate(question, context)
        citations = tuple(dict.fromkeys(result.source for result in results if result.score > 0))
        return RAGAnswer(question, answer, citations, tuple(results))
