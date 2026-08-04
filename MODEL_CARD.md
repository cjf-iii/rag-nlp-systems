# Model card

## System

This repository contains two local-first systems:

- a TF-IDF retrieval index with citation-preserving RAG orchestration;
- a TF-IDF and logistic-regression sentiment classifier.

The default RAG generator is deterministic and extractive. Ollama support is optional and disabled unless the caller selects it.

## Intended use

- learning and demonstrating retrieval, grounding, persistence, and classification;
- indexing small private Markdown or text corpora on a local computer;
- testing how retrieval inputs affect an answer before introducing a larger model.

## Inputs and outputs

The retrieval index accepts UTF-8 Markdown and text files. It returns ranked passages, cosine-similarity scores, an answer, and source filenames. The sentiment pipeline accepts labeled text and returns a class, confidence, and per-class probabilities.

## Limitations

- TF-IDF matches vocabulary rather than semantic meaning.
- The sample sentiment data is intentionally small and cannot support production claims.
- The extractive generator can quote irrelevant sentences when retrieval is weak.
- Optional Ollama output inherits the limitations of the selected local model.

## Privacy

The default path performs no network requests. Ollama calls only the loopback endpoint configured by the user.
