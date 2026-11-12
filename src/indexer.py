import os
import re
import pickle
import unicodedata
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


class VectorIndexer:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
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

    def _lexical_score(self, query: str, text: str) -> float:
        stopwords = {"quais", "sao", "como", "para", "com", "dos", "das", "nos", "nas", "que", "uma", "uns", "umas", "por"}
        q_clean = _strip_accents(query.lower())
        q_tokens = [w for w in re.findall(r"\w+", q_clean) if len(w) > 2 and w not in stopwords]
        if not q_tokens:
            return 0.0

        t_clean = _strip_accents(text.lower())
        matches = sum(1 for token in q_tokens if token in t_clean)
        return matches / len(q_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Índice FAISS não inicializado.")

        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Busca híbrida: combina similaridade vetorial (75%) + match lexical (25%)
        total_chunks = len(self.chunks_metadata)
        scores, indices = self.index.search(query_vector, total_chunks)

        candidates = []
        for sem_score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item = self.chunks_metadata[idx].copy()
                lex_score = self._lexical_score(query, item["text"])
                item["similarity_score"] = 0.75 * float(sem_score) + 0.25 * lex_score
                candidates.append(item)

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]

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