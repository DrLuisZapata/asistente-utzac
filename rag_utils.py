"""
rag_utils.py
Carga los documentos del temario, los divide en fragmentos, genera
embeddings y construye un índice de búsqueda (FAISS) en memoria.

No necesita API keys: sentence-transformers corre localmente (CPU).
"""

import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500       # caracteres por fragmento
CHUNK_OVERLAP = 80     # solape entre fragmentos consecutivos
# Umbral de similitud (distancia L2, menor = más parecido).
# Si el mejor resultado supera este valor, se considera "no encontrado".
NOT_FOUND_THRESHOLD = 1.15


def load_documents(data_dir: str) -> list[dict]:
    """Lee todos los .txt y .md de data_dir. Devuelve [{"source": ..., "text": ...}]."""
    docs = []
    paths = glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.md"))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide un texto largo en fragmentos con solape, respetando saltos de párrafo cuando puede."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


class RAGIndex:
    """Índice de búsqueda semántica sobre los documentos del curso."""

    def __init__(self, data_dir: str):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chunks: list[dict] = []  # [{"text": ..., "source": ...}]
        self._build(data_dir)

    def _build(self, data_dir: str):
        documents = load_documents(data_dir)
        for doc in documents:
            for chunk in chunk_text(doc["text"]):
                self.chunks.append({"text": chunk, "source": doc["source"]})

        if not self.chunks:
            raise RuntimeError(
                f"No se encontraron documentos .txt/.md en '{data_dir}'. "
                "Agrega tus notas de curso antes de iniciar el servicio."
            )

        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 3) -> dict:
        """Busca los fragmentos más relevantes. Devuelve found=False si nada supera el umbral."""
        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k)

        best_distance = float(distances[0][0])
        if best_distance > NOT_FOUND_THRESHOLD:
            return {"found": False, "context": "", "sources": []}

        results = [self.chunks[i] for i in indices[0] if i != -1]
        context = "\n\n---\n\n".join(r["text"] for r in results)
        sources = sorted(set(r["source"] for r in results))
        return {"found": True, "context": context, "sources": sources}
