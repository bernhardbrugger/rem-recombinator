"""Test doubles that keep the suite offline."""

from __future__ import annotations

import numpy as np


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32)


class MapEmbedder:
    """Returns a predeclared vector for each exact string."""

    name = "map"

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = mapping

    def embed_text(self, text: str, input_type: str = "document") -> np.ndarray:
        del input_type
        if text not in self.mapping:
            raise KeyError(text)
        return np.asarray(self.mapping[text], dtype=np.float32)

    def embed_batch(self, texts: list[str], input_type: str = "document") -> list[np.ndarray]:
        return [self.embed_text(text, input_type=input_type) for text in texts]
