"""
rag_utils.py — versión ligera (TF-IDF), sin dependencia de torch.

Antes usaba sentence-transformers (embeddings neuronales), pero eso
requiere cargar PyTorch en memoria, lo cual excede los 512 MB del plan
gratis de Render. TF-IDF logra un resultado similar para búsqueda de
palabras clave/frases en documentos de texto, con una fracción de la
memoria y sin descargar ningún modelo.
"""

import os
import re
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 350       # caracteres máximos por fragmento antes de subdividir una sección
CHUNK_OVERLAP = 60     # solape entre sub-fragmentos consecutivos
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

SECTION_HEADER_RE = re.compile(r"^##\s+.*$", re.MULTILINE)


def load_documents(data_dir: str) -> list[dict]:
    """Lee todos los .txt y .md de data_dir. Devuelve [{"source": ..., "text": ...}]."""
    docs = []
    paths = glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.md"))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide un texto largo en fragmentos con solape (respaldo cuando una sección es muy larga)."""
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


def split_into_sections(text: str) -> list[str]:
    """
    Divide el texto por encabezados de sección (##), manteniendo cada
    encabezado junto con su contenido. Esto evita que el buscador separe,
    por ejemplo, "## Metodología de evaluación" del párrafo que sí contiene
    la respuesta, lo cual causaba que se recuperara la sección equivocada
    para preguntas cuyo tema coincide con el título de otra sección.

    Si una sección resulta demasiado larga, se subdivide en fragmentos más
    pequeños, repitiendo el encabezado "## ..." en cada uno — de lo
    contrario, un párrafo específico queda "diluido" entre otros párrafos
    de la misma sección sin palabras clave en común con la pregunta.
    """
    text = text.strip()
    matches = list(SECTION_HEADER_RE.finditer(text))

    if not matches:
        return chunk_text(text)

    # Título del documento (línea(s) antes del primer "##"), si existe.
    doc_title = text[: matches[0].start()].strip()

    final_chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        header = match.group().strip()  # p. ej. "## Metodología de evaluación"
        body = section[len(header):].strip()

        prefix = f"{doc_title}\n\n{header}" if (i == 0 and doc_title) else header

        if len(section) <= CHUNK_SIZE:
            final_chunks.append(f"{prefix}\n\n{body}" if body else prefix)
            continue

        for piece in chunk_text(body, chunk_size=max(CHUNK_SIZE - len(prefix) - 2, 100)):
            final_chunks.append(f"{prefix}\n\n{piece}")

    return final_chunks


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
            for chunk in split_into_sections(doc["text"]):
                self.chunks.append({"text": chunk, "source": doc["source"]})

        if not self.chunks:
            raise RuntimeError(
                f"No se encontraron documentos .txt/.md en '{data_dir}'. "
                "Agrega tus notas de curso antes de iniciar el servicio."
            )

        texts = [c["text"] for c in self.chunks]
        # Se combinan dos vectorizadores: uno por PALABRAS (bueno para
        # distinguir temas relevantes de irrelevantes) y uno por CARACTERES
        # (reconoce que "evalúa", "evaluación" y "evaluar" comparten raíz,
        # sin necesitar un lematizador de español). Se promedian ambas
        # similitudes en la búsqueda.
        self.word_vectorizer = TfidfVectorizer(
            stop_words=SPANISH_STOPWORDS,
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(4, 6),
            lowercase=True,
            min_df=1,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> dict:
        """Busca los fragmentos más relevantes. Devuelve found=False si nada supera el umbral."""
        word_sims = cosine_similarity(self.word_vectorizer.transform([query]), self.word_matrix)[0]
        char_sims = cosine_similarity(self.char_vectorizer.transform([query]), self.char_matrix)[0]
        # Más peso a palabras completas (mejor para distinguir tema relevante
        # de irrelevante); los caracteres solo ayudan a "rescatar" coincidencias
        # cuando la pregunta usa una forma distinta de la misma palabra raíz.
        similarities = 0.65 * word_sims + 0.35 * char_sims

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
