from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_nlp_systems.sentiment import SentimentPipeline, load_labeled_csv

TEXTS = [
    "The implementation was clear and the result was excellent",
    "The model produced reliable answers with useful citations",
    "The evaluation was thorough and easy to understand",
    "The product worked well during the technical review",
    "The documentation made integration straightforward",
    "The response was accurate and supported by evidence",
    "The implementation failed and the output was unusable",
    "The model returned inaccurate answers without citations",
    "The evaluation missed obvious errors in the results",
    "The product crashed repeatedly during the technical review",
    "The documentation was confusing and incomplete",
    "The response was unsupported and contradicted the source",
]
LABELS = ["positive"] * 6 + ["negative"] * 6


class SentimentPipelineTests(unittest.TestCase):
    def test_fit_predict_and_persist(self) -> None:
        pipeline = SentimentPipeline().fit(TEXTS, LABELS)
        prediction = pipeline.predict(["The result was accurate and useful"])[0]
        self.assertIn(prediction["label"], {"positive", "negative"})
        self.assertGreaterEqual(prediction["confidence"], 0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sentiment.joblib"
            pipeline.save(model_path)
            loaded = SentimentPipeline.load(model_path)
            self.assertEqual(
                loaded.predict(["The result was accurate and useful"])[0]["label"],
                prediction["label"],
            )

    def test_held_out_evaluation_reports_all_rows(self) -> None:
        evaluation = SentimentPipeline().evaluate(TEXTS, LABELS, test_size=1 / 3)
        self.assertEqual(evaluation.train_rows + evaluation.test_rows, len(TEXTS))
        self.assertGreaterEqual(evaluation.accuracy, 0)
        self.assertLessEqual(evaluation.accuracy, 1)

    def test_sample_dataset_has_a_stable_held_out_signal(self) -> None:
        data_path = Path(__file__).parents[1] / "data" / "sentiment.csv"
        texts, labels = load_labeled_csv(data_path)
        evaluation = SentimentPipeline().evaluate(texts, labels)
        self.assertGreaterEqual(evaluation.accuracy, 0.7)

    def test_documented_positive_example_is_classified_positive(self) -> None:
        data_path = Path(__file__).parents[1] / "data" / "sentiment.csv"
        texts, labels = load_labeled_csv(data_path)
        prediction = (
            SentimentPipeline()
            .fit(texts, labels)
            .predict(["The benchmark was clear and reproducible"])[0]
        )
        self.assertEqual(prediction["label"], "positive")


if __name__ == "__main__":
    unittest.main()
