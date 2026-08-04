"""TF-IDF sentiment classification with evaluation and persistence."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


@dataclass(frozen=True)
class SentimentEvaluation:
    """Held-out classification results."""

    accuracy: float
    train_rows: int
    test_rows: int
    report: dict[str, object]


def load_labeled_csv(path: str | Path) -> tuple[list[str], list[str]]:
    """Load a UTF-8 CSV containing ``text`` and ``label`` columns."""

    texts: list[str] = []
    labels: list[str] = []
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("sentiment CSV must contain text and label columns")
        for row in reader:
            text = str(row["text"]).strip()
            label = str(row["label"]).strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    if len(texts) < 4:
        raise ValueError("sentiment training requires at least four labeled rows")
    return texts, labels


class SentimentPipeline:
    """Train, evaluate, save, and run a lightweight NLP classifier."""

    def __init__(self, model: Pipeline | None = None) -> None:
        self.model = model or Pipeline(
            [
                (
                    "tfidf",
                    FeatureUnion(
                        [
                            (
                                "words",
                                TfidfVectorizer(
                                    lowercase=True,
                                    ngram_range=(1, 2),
                                    min_df=1,
                                    sublinear_tf=True,
                                ),
                            ),
                            (
                                "characters",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    lowercase=True,
                                    ngram_range=(3, 5),
                                    min_df=1,
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42),
                ),
            ]
        )

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "SentimentPipeline":
        text_list = list(texts)
        label_list = list(labels)
        if len(text_list) != len(label_list):
            raise ValueError("texts and labels must have the same length")
        self.model.fit(text_list, label_list)
        return self

    def evaluate(
        self,
        texts: Iterable[str],
        labels: Iterable[str],
        test_size: float = 0.25,
    ) -> SentimentEvaluation:
        text_list = list(texts)
        label_list = list(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            text_list,
            label_list,
            test_size=test_size,
            random_state=42,
            stratify=label_list,
        )
        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        return SentimentEvaluation(
            accuracy=float(accuracy_score(y_test, predictions)),
            train_rows=len(x_train),
            test_rows=len(x_test),
            report=report,
        )

    def predict(self, texts: Iterable[str]) -> list[dict[str, object]]:
        text_list = list(texts)
        labels = self.model.predict(text_list)
        probabilities = self.model.predict_proba(text_list)
        classes = list(self.model.classes_)
        output: list[dict[str, object]] = []
        for text, label, row in zip(text_list, labels, probabilities, strict=True):
            confidence = float(np.max(row))
            output.append(
                {
                    "text": text,
                    "label": str(label),
                    "confidence": confidence,
                    "probabilities": {name: float(value) for name, value in zip(classes, row, strict=True)},
                }
            )
        return output

    def save(self, output_path: str | Path) -> Path:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, destination)
        return destination

    @classmethod
    def load(cls, input_path: str | Path) -> "SentimentPipeline":
        return cls(joblib.load(Path(input_path).expanduser().resolve()))
