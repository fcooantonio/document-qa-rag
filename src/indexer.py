import os
import pickle
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("A lista de chunks está vazia.")

        self.chunks_metadata = chunks
        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Normaliza L2 para que o Inner Product funcione como similaridade de cosseno
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Índice FAISS não inicializado.")

        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item = self.chunks_metadata[idx].copy()
                item["similarity_score"] = float(score)
                results.append(item)
        return results

    def save(self, output_dir: str = "data/processed") -> None:
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(output_dir, "index.faiss"))
        with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks_metadata, f)

    def load(self, input_dir: str = "data/processed") -> None:
        index_path = os.path.join(input_dir, "index.faiss")
        meta_path = os.path.join(input_dir, "chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Índice não encontrado em {input_dir}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.chunks_metadata = pickle.load(f)