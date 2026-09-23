"""
app.py — Endpoint FastAPI del asistente de curso (RAG + Llama).

Flujo de una petición:
1. Recibe una pregunta en POST /ask
2. Busca el fragmento más relevante del temario (rag_utils.RAGIndex)
3. Si no hay nada relevante -> responde con un mensaje claro (no inventa)
4. Si hay contexto -> arma un prompt y llama a Llama vía GitHub Models
5. Devuelve la respuesta en un formato JSON estándar

Nota de arquitectura: se llama a GitHub Models (https://github.com/marketplace/models)
directamente con `requests` en vez del SDK `openai`, porque el SDK tuvo un
comportamiento inconsistente con este endpoint (a veces regresaba texto plano
en vez del objeto JSON esperado). Llamar a la API REST directamente da control
total y facilita ver el cuerpo exacto de la respuesta si algo falla.
"""

import os
import time
import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_utils import RAGIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("course-assistant")

DATA_DIR = os.environ.get("DATA_DIR", "data")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"

# Si el usuario fija LLAMA_MODEL explícitamente, esa variable tiene prioridad.
# Si no, probamos esta lista de candidatos (el catálogo de GitHub Models ha
# usado distintas convenciones de nombre) y usamos el primero que responda.
LLAMA_MODEL_OVERRIDE = os.environ.get("LLAMA_MODEL")
LLAMA_MODEL_CANDIDATES = [
    "meta/Llama-3.3-70B-Instruct",
    "meta/Meta-Llama-3.1-8B-Instruct",
    "meta/Llama-3.2-11B-Vision-Instruct",
    "meta/Meta-Llama-3.1-70B-Instruct",
]

app = FastAPI(
    title="Asistente de Curso UTZAC (Llama + RAG)",
    description="Responde preguntas de alumnos usando el temario del curso.",
    version="1.0.0",
)

rag_index: RAGIndex | None = None
LLAMA_MODEL: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Pregunta del alumno")


class AskResponse(BaseModel):
    answer: str
    found_context: bool
    sources: list[str]
    model: str
    latency_seconds: float


def call_github_models(model: str, messages: list[dict], max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Llama directamente a la API REST de GitHub Models y regresa el texto de la respuesta."""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "utzac-course-assistant/1.0",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(GITHUB_MODELS_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} del modelo '{model}': {response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"Respuesta no-JSON del modelo '{model}' (status {response.status_code}, "
            f"content-type: {response.headers.get('content-type')}): {response.text[:300]}"
        )

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Formato de respuesta inesperado del modelo '{model}': {data}")


def pick_working_model() -> str:
    """Prueba cada modelo candidato con una petición mínima y usa el primero que funcione."""
    candidates = [LLAMA_MODEL_OVERRIDE] if LLAMA_MODEL_OVERRIDE else LLAMA_MODEL_CANDIDATES
    last_error = None
    for candidate in candidates:
        try:
            call_github_models(candidate, [{"role": "user", "content": "hola"}], max_tokens=5)
            logger.info(f"Modelo Llama seleccionado (GitHub Models): {candidate}")
            return candidate
        except Exception as exc:
            logger.warning(f"Modelo '{candidate}' no disponible: {exc}")
            last_error = exc
    raise RuntimeError(
        f"Ninguno de los modelos Llama candidatos está disponible vía GitHub Models "
        f"para esta cuenta. Último error: {last_error}"
    )


@app.on_event("startup")
def startup():
    global rag_index, LLAMA_MODEL
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN no está configurada. El endpoint /ask fallará hasta configurarla.")
    else:
        LLAMA_MODEL = pick_working_model()
    rag_index = RAGIndex(DATA_DIR)
    logger.info(f"RAG listo con {len(rag_index.chunks)} fragmentos de '{DATA_DIR}'.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(rag_index.chunks) if rag_index else 0,
        "github_models_configured": GITHUB_TOKEN is not None,
        "model": LLAMA_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if rag_index is None:
        raise HTTPException(status_code=503, detail="El índice RAG aún no está listo.")
    if not GITHUB_TOKEN or not LLAMA_MODEL:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN no configurada o ningún modelo disponible.")

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
            model=LLAMA_MODEL,
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
        answer = call_github_models(
            LLAMA_MODEL,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        logger.exception(f"Error llamando a GitHub Models (tipo: {type(exc).__name__})")
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
