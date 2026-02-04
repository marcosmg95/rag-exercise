from typing import List
import torch
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class LocalPyTorchEmbeddings(Embeddings):
    """Local embeddings using PyTorch and SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        with torch.no_grad():
            embedding = self.model.encode([text], convert_to_tensor=True)
            return embedding.tolist()[0]
