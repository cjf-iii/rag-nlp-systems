# Model Card

## System

This repository contains two local-first systems:

- a TF-IDF retrieval index with citation-preserving RAG orchestration;
- a word- and character-TF-IDF sentiment classifier using logistic regression.

The default answer generator is deterministic and extractive. Ollama support is optional and disabled unless the caller selects it.

## Intended use

- demonstrate retrieval, grounding, citation, persistence, and classification;
- index small, trusted Markdown or text corpora on a local computer;
- inspect how retrieved evidence affects an answer before introducing a generative model.

## Inputs and outputs

The retrieval index accepts UTF-8 Markdown and text files. It returns ranked passages, cosine-similarity scores, an answer, and source filenames. The sentiment pipeline accepts labeled text and returns a class, confidence, per-class probabilities, and a held-out report.

## Evaluation

The release gate checks retrieval relevance, citation output, bounded chunking, index round trips, insufficient-context behavior, sentiment persistence, held-out row accounting, a minimum demonstration-data signal, and the complete CLI workflow. These operational checks do not establish production accuracy.

## Limitations

- TF-IDF matches lexical patterns rather than general semantic meaning.
- The sample sentiment dataset is intentionally small and synthetic in purpose.
- The extractive generator can select a locally relevant sentence that does not fully resolve a complex question.
- Optional Ollama output inherits the behavior and limitations of the selected model.
- No prompt-injection filtering or document access-control layer is implemented; only trusted corpora should be indexed.

## Privacy and artifact safety

The default path performs no network request. The Ollama adapter defaults to a loopback endpoint but can be configured by the caller. Saved indexes and classifiers use joblib/pickle-compatible serialization; load only artifacts created locally or obtained from a trusted source.
