# Retrieval-augmented generation

A retrieval-augmented system searches an indexed corpus before producing an answer. The retrieval stage selects relevant passages, and the generation stage receives those passages as bounded context.

Source identifiers let the application return citations with each answer. A grounded system should state when the retrieved context is insufficient instead of filling the gap with unsupported text.
