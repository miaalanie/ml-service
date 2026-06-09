from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)
        self._cache: dict = {}

    def encode(self, text: str) -> np.ndarray:
        # Encode teks menjadi vektor 384 dimensi. Gunakan cache untuk hindari encode ulang teks yang sama.
        if not text or not text.strip():
            return np.zeros(384)

        if text not in self._cache:
            self._cache[text] = self.model.encode(
                text,
                convert_to_numpy=True
            )
        return self._cache[text]

    def encode_batch(self, texts: list) -> list:
        results = []
        to_encode = []
        indices = []

        # Pisahkan yang sudah di-cache
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append((i, self._cache[text]))
            else:
                to_encode.append(text)
                indices.append(i)

        # Encode yang belum di-cache dalam 1 batch
        if to_encode:
            vecs = self.model.encode(
                to_encode,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=False
            )
            for idx, vec, text in zip(indices, vecs, to_encode):
                self._cache[text] = vec
                results.append((idx, vec))

        # Urutkan kembali sesuai index original
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def clear_cache(self):
        self._cache = {}
