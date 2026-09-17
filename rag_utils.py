"""
rag_utils.py — versión ligera (TF-IDF), sin dependencia de torch.

Antes usaba sentence-transformers (embeddings neuronales), pero eso
requiere cargar PyTorch en memoria, lo cual excede los 512 MB del plan
gratis de Render. TF-IDF logra un resultado similar para búsqueda de
palabras clave/frases en documentos de texto, con una fracción de la
memoria y sin descargar ningún modelo.
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 500       # caracteres por fragmento
CHUNK_OVERLAP = 80     # solape entre fragmentos consecutivos
# Umbral de similitud coseno (0 a 1, mayor = más parecido).
# Si el mejor resultado NO supera este valor, se considera "no encontrado".
NOT_FOUND_THRESHOLD = 0.08

# Lista breve de stopwords en español para mejorar la calidad de la búsqueda.
SPANISH_STOPWORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
    "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
    "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes",
    "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas",
    "algo", "nosotros", "es", "son", "ser", "está", "están",
]


def load_documents(data_dir: str) -> list[dict]:
    """Lee todos los .txt y .md de data_dir. Devuelve [{"source": ..., "text": ...}]."""
    docs = []
    paths = glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.md"))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide un texto largo en fragmentos con solape."""
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
    """Índice de búsqueda por palabras clave (TF-IDF) sobre los documentos del curso."""

    def __init__(self, data_dir: str):
        self.chunks: list[dict] = []  # [{"text": ..., "source": ...}]
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
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
        self.vectorizer = TfidfVectorizer(
            stop_words=SPANISH_STOPWORDS,
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> dict:
        """Busca los fragmentos más relevantes. Devuelve found=False si nada supera el umbral."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]

        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        if best_score < NOT_FOUND_THRESHOLD:
            return {"found": False, "context": "", "sources": []}

        top_indices = similarities.argsort()[::-1][:top_k]
        top_indices = [i for i in top_indices if similarities[i] >= NOT_FOUND_THRESHOLD]

        results = [self.chunks[i] for i in top_indices]
        context = "\n\n---\n\n".join(r["text"] for r in results)
        sources = sorted(set(r["source"] for r in results))
        return {"found": True, "context": context, "sources": sources}
