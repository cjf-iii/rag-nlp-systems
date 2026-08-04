"""Local-first retrieval-augmented generation and NLP pipelines."""

from .rag import KnowledgeBase, RAGAgent
from .sentiment import SentimentPipeline

__all__ = ["KnowledgeBase", "RAGAgent", "SentimentPipeline"]
