from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )


    def encode(self, text: str):
        return self.model.encode(text)