# RAG and NLP Systems

Local-first Python systems for document retrieval, grounded answers, sentiment classification, evaluation, and model persistence.

This repository demonstrates the résumé claim:

> Built RAG agents and NLP pipelines from first principles for document retrieval, text analysis, and sentiment classification.

## What is implemented

- recursive Markdown and text ingestion with paragraph-aware chunking;
- TF-IDF retrieval with unigram and bigram features plus cosine ranking;
- citation-preserving RAG orchestration and insufficient-context handling;
- deterministic offline answers or optional generation through local Ollama;
- TF-IDF and logistic-regression sentiment classification;
- held-out evaluation, confidence scores, joblib persistence, CLI commands, and tests.

## Architecture

```text
Markdown or text files
        |
        v
paragraph-aware chunks -> TF-IDF index -> cosine retrieval
                                           |
                                           v
                               grounded context + citations
                                           |
                         +-----------------+-----------------+
                         |                                   |
                  extractive answer                   local Ollama

labeled text -> TF-IDF bigrams -> logistic regression -> label + confidence
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

rag-nlp index --corpus data/knowledge --output artifacts/knowledge.joblib
rag-nlp ask \
  --index artifacts/knowledge.joblib \
  --question "How does batching affect inference capacity?"

rag-nlp sentiment-train \
  --data data/sentiment.csv \
  --output artifacts/sentiment.joblib
rag-nlp sentiment-predict \
  --model artifacts/sentiment.joblib \
  --text "The benchmark was clear and reproducible"
```

The default commands run offline. To use a local generative model, install Ollama, pull a model that fits the machine, and select the provider:

```bash
ollama pull qwen2.5:1.5b
rag-nlp ask \
  --index artifacts/knowledge.joblib \
  --question "What is the latency-throughput tradeoff?" \
  --provider ollama \
  --model qwen2.5:1.5b
```

The default TF-IDF systems run comfortably on low-memory Apple-silicon Macs. The optional generator determines the additional memory requirement.

## Test

```bash
./scripts/verify_macos_metal.sh
python -m unittest discover -s tests -v
```

The preflight fails unless the host is an ARM64 Mac with Metal support. The tests then prove source ranking, citation output, index round trips, sentiment prediction, held-out evaluation, and model persistence without contacting a network service.

TF-IDF and logistic regression are CPU algorithms; they are tested on the declared Mac hardware but are not mislabeled as GPU-accelerated. Optional local neural generation can use an Apple-Silicon-native backend such as Ollama/Metal.

See [VERIFICATION.md](VERIFICATION.md) for the recorded host and completed checks.

## Claim boundary

This is a working local system, not a claim of customer deployment or foundation-model training. The included corpus and sentiment examples are original demonstration data. See [MODEL_CARD.md](MODEL_CARD.md) for intended use and limitations.

## License

MIT
