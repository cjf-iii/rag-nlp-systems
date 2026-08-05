# Verification Record

Recorded August 4, 2026.

## Release gate

```bash
uv sync --locked --group dev --group notebook
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
uv run --group notebook jupyter nbconvert --to notebook --execute notebooks/01_data_to_validation.ipynb --stdout > /dev/null
uv run rag-nlp index --corpus data/knowledge --output artifacts/knowledge.joblib
uv run rag-nlp ask --index artifacts/knowledge.joblib --question "How does batching affect inference capacity?"
uv run rag-nlp sentiment-train --data data/sentiment.csv --output artifacts/sentiment.joblib
uv run rag-nlp sentiment-predict --model artifacts/sentiment.joblib --text "The benchmark was clear and reproducible"
uv run python scripts/verify_artifacts.py
```

## Acceptance criteria

- one locked `uv` environment on Python 3.12;
- public CPU-based GitHub Actions workflow;
- retrieval, citation, chunk-boundary, index round-trip, and insufficient-context tests;
- held-out sentiment accounting, persistence, and documented-example tests;
- every reviewer-notebook cell executes in the locked notebook environment;
- both documented CLI workflows complete offline;
- saved index and sentiment artifacts reload and reproduce the expected outputs.

## Recorded result

All 10 tests and every reviewer-notebook cell passed. The CLI indexed 3 source chunks, answered the documented batching question with `inference.md` as its citation, and correctly returned an explicit boundary for an unrelated question.

The sentiment CLI evaluated 60 labeled examples with a seeded 45/15 split and observed 0.80 held-out accuracy. After full-data fitting and persistence, the documented sentence—“The benchmark was clear and reproducible”—reloaded as `positive` with 0.5723 confidence. The artifact verifier independently reloaded both locally generated joblib files and reproduced the expected outputs.
