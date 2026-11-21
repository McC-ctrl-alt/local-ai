"""Helpers shared by curator.py and policy_rag.py."""

import math


def cosine_similarity(vector_a, vector_b):
    """Return the cosine similarity of two embedding vectors.

    A zero-length vector (no components, or every component zero) is an
    embedding failure, not a similarity of 0.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            "cosine_similarity requires two vectors of the same length, "
            f"got {len(vector_a)} and {len(vector_b)}."
        )
    length_a = math.hypot(*vector_a)
    length_b = math.hypot(*vector_b)
    if length_a == 0.0 or length_b == 0.0:
        raise ValueError("cosine_similarity cannot score a zero-length vector.")
    if not math.isfinite(length_a) or not math.isfinite(length_b):
        raise ValueError("cosine_similarity cannot score a non-finite vector.")
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    if not math.isfinite(dot_product):
        raise ValueError("cosine_similarity cannot score a non-finite vector.")
    return dot_product / (length_a * length_b)


def embedding_vector(payload, model):
    """Return the first embedding from an Ollama /api/embed response."""
    try:
        vector = payload["embeddings"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Ollama returned no embedding from {model}.") from error
    if isinstance(vector, (str, bytes)) or len(vector) == 0:
        raise ValueError(f"Ollama returned an empty embedding from {model}.")
    return vector
