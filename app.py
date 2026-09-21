"""
app.py — Endpoint FastAPI del asistente de curso (RAG + Llama).

Flujo de una petición:
1. Recibe una pregunta en POST /ask
2. Busca el fragmento más relevante del temario (rag_utils.RAGIndex)
3. Si no hay nada relevante -> responde con un mensaje claro (no inventa)
4. Si hay contexto -> arma un prompt y llama a Llama vía Hugging Face Inference
5. Devuelve la respuesta en un formato JSON estándar

Nota de arquitectura: se usa Hugging Face Inference (Inference Providers) en
vez de Groq, porque Groq restringió sus modelos Llama de propósito general a
cuentas Enterprise. Con Hugging Face usamos el mismo modelo base
(Llama-3.1-8B-Instruct) que se ajustó con LoRA en el notebook de Colab, cuya
licencia el usuario ya aceptó ahí.
"""

import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from rag_utils import RAGIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("course-assistant")

DATA_DIR = os.environ.get("DATA_DIR", "data")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Si el usuario fija LLAMA_MODEL explícitamente, esa variable tiene prioridad.
# Si no, probamos esta lista de candidatos Llama (de mayor a menor tamaño) y
# usamos el primero que responda vía Hugging Face Inference Providers.
LLAMA_MODEL_OVERRIDE = os.environ.get("LLAMA_MODEL")
LLAMA_MODEL_CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]

app = FastAPI(
    title="Asistente de Curso UTZAC (Llama + RAG)",
    description="Responde preguntas de alumnos usando el temario del curso.",
    version="1.0.0",
)

rag_index: RAGIndex | None = None
hf_client: InferenceClient | None = None
LLAMA_MODEL: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Pregunta del alumno")


class AskResponse(BaseModel):
    answer: str
    found_context: bool
    sources: list[str]
    model: str
    latency_seconds: float


def pick_working_model(client: InferenceClient) -> str:
    """Prueba cada modelo candidato con una petición mínima y usa el primero que funcione."""
    candidates = [LLAMA_MODEL_OVERRIDE] if LLAMA_MODEL_OVERRIDE else LLAMA_MODEL_CANDIDATES
    last_error = None
    for candidate in candidates:
        try:
            client.chat.completions.create(
                model=candidate,
                messages=[{"role": "user", "content": "hola"}],
                max_tokens=5,
            )
            logger.info(f"Modelo Llama seleccionado (Hugging Face): {candidate}")
            return candidate
        except Exception as exc:
            logger.warning(f"Modelo '{candidate}' no disponible: {exc}")
            last_error = exc
    raise RuntimeError(
        f"Ninguno de los modelos Llama candidatos está disponible vía Hugging Face "
        f"Inference para esta cuenta. Último error: {last_error}"
    )


@app.on_event("startup")
def startup():
    global rag_index, hf_client, LLAMA_MODEL
    if not HF_TOKEN:
        logger.warning("HF_TOKEN no está configurada. El endpoint /ask fallará hasta configurarla.")
    else:
        hf_client = InferenceClient(token=HF_TOKEN)
        LLAMA_MODEL = pick_working_model(hf_client)
    rag_index = RAGIndex(DATA_DIR)
    logger.info(f"RAG listo con {len(rag_index.chunks)} fragmentos de '{DATA_DIR}'.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(rag_index.chunks) if rag_index else 0,
        "hf_configured": hf_client is not None,
        "model": LLAMA_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if rag_index is None:
        raise HTTPException(status_code=503, detail="El índice RAG aún no está listo.")
    if hf_client is None:
        raise HTTPException(status_code=500, detail="HF_TOKEN no configurada en el servidor.")

    start = time.time()
    question = payload.question.strip()

    result = rag_index.search(question)

    if not result["found"]:
        # Caso borde clave del caso práctico: no fallar en silencio ni inventar.
        return AskResponse(
            answer=(
                "No encontré información sobre eso en el material del curso que tengo "
                "cargado. Intenta reformular tu pregunta o consúltalo directamente con "
                "el profesor."
            ),
            found_context=False,
            sources=[],
            model=LLAMA_MODEL or "n/a",
            latency_seconds=round(time.time() - start, 3),
        )

    system_prompt = (
        "Eres el asistente del curso de un profesor de la Universidad Tecnológica de "
        "Zacatecas (UTZAC). Responde SOLO con base en el CONTEXTO proporcionado. "
        "Si el contexto no alcanza para responder con certeza, dilo explícitamente "
        "en vez de inventar información. Responde en español, de forma breve y clara, "
        "como si le explicaras a un estudiante."
    )
    user_prompt = f"CONTEXTO DEL TEMARIO:\n{result['context']}\n\nPREGUNTA DEL ALUMNO:\n{question}"

    try:
        completion = hf_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        answer = completion.choices[0].message.content
    except (HfHubHTTPError, Exception) as exc:
        logger.exception("Error llamando a Hugging Face Inference")
        raise HTTPException(status_code=502, detail=f"Error al generar la respuesta: {exc}")

    return AskResponse(
        answer=answer,
        found_context=True,
        sources=result["sources"],
        model=LLAMA_MODEL,
        latency_seconds=round(time.time() - start, 3),
    )


@app.get("/")
def root():
    return {
        "message": "Asistente de curso UTZAC. Usa POST /ask con {'question': '...'}",
        "docs": "/docs",
    }
