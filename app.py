"""
app.py — Endpoint FastAPI del asistente de curso (RAG + Llama).

Flujo de una petición:
1. Recibe una pregunta en POST /ask
2. Busca el fragmento más relevante del temario (rag_utils.RAGIndex)
3. Si no hay nada relevante -> responde con un mensaje claro (no inventa)
4. Si hay contexto -> arma un prompt y llama a Llama 3 en Groq
5. Devuelve la respuesta en un formato JSON estándar
"""

import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq

from rag_utils import RAGIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("course-assistant")

DATA_DIR = os.environ.get("DATA_DIR", "data")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI(
    title="Asistente de Curso UTZAC (Llama + RAG)",
    description="Responde preguntas de alumnos usando el temario del curso.",
    version="1.0.0",
)

rag_index: RAGIndex | None = None
groq_client: Groq | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Pregunta del alumno")


class AskResponse(BaseModel):
    answer: str
    found_context: bool
    sources: list[str]
    model: str
    latency_seconds: float


@app.on_event("startup")
def startup():
    global rag_index, groq_client
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY no está configurada. El endpoint /ask fallará hasta configurarla.")
    else:
        groq_client = Groq(api_key=GROQ_API_KEY)
    rag_index = RAGIndex(DATA_DIR)
    logger.info(f"RAG listo con {len(rag_index.chunks)} fragmentos de '{DATA_DIR}'.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(rag_index.chunks) if rag_index else 0,
        "groq_configured": groq_client is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if rag_index is None:
        raise HTTPException(status_code=503, detail="El índice RAG aún no está listo.")
    if groq_client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY no configurada en el servidor.")

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
            model=GROQ_MODEL,
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
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        answer = completion.choices[0].message.content
    except Exception as exc:
        logger.exception("Error llamando a Groq")
        raise HTTPException(status_code=502, detail=f"Error al generar la respuesta: {exc}")

    return AskResponse(
        answer=answer,
        found_context=True,
        sources=result["sources"],
        model=GROQ_MODEL,
        latency_seconds=round(time.time() - start, 3),
    )


@app.get("/")
def root():
    return {
        "message": "Asistente de curso UTZAC. Usa POST /ask con {'question': '...'}",
        "docs": "/docs",
    }
