"""Helpers shared by engine.py, curator.py, and policy_rag.py."""

from typing import Any

import numpy as np


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Return the cosine similarity of two embedding vectors.

    A zero-length vector (no components, or every component zero) is an
    embedding failure, not a similarity of 0. The result is a plain float.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            "cosine_similarity requires two vectors of the same length, "
            f"got {len(vector_a)} and {len(vector_b)}."
        )
    array_a = np.asarray(vector_a, dtype=np.float64)
    array_b = np.asarray(vector_b, dtype=np.float64)
    length_a = float(np.linalg.norm(array_a))
    length_b = float(np.linalg.norm(array_b))
    if length_a == 0.0 or length_b == 0.0:
        raise ValueError("cosine_similarity cannot score a zero-length vector.")
    if not np.isfinite(length_a) or not np.isfinite(length_b):
        raise ValueError("cosine_similarity cannot score a non-finite vector.")
    dot_product = float(np.dot(array_a, array_b))
    if not np.isfinite(dot_product):
        raise ValueError("cosine_similarity cannot score a non-finite vector.")
    return dot_product / (length_a * length_b)


def embedding_vector(payload: dict[str, Any], model: str) -> list[float]:
    """Return the first embedding from an Ollama /api/embed response."""
    try:
        vector = payload["embeddings"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Ollama returned no embedding from {model}.") from error
    if isinstance(vector, (str, bytes)) or len(vector) == 0:
        raise ValueError(f"Ollama returned an empty embedding from {model}.")
    return vector
