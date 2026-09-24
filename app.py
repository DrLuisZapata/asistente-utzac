"""
app.py — Endpoint FastAPI del asistente de curso (RAG + Llama).

Flujo de una petición:
1. Recibe una pregunta en POST /ask
2. Busca el fragmento más relevante del temario (rag_utils.RAGIndex)
3. Si no hay nada relevante -> responde con un mensaje claro (no inventa)
4. Si hay contexto -> arma un prompt y llama a Llama vía SambaNova Cloud
5. Devuelve la respuesta en un formato JSON estándar

Nota de arquitectura — historial de proveedores probados:
- Groq: sus modelos Llama de propósito general quedaron restringidos a
  cuentas Enterprise (verificado contra su endpoint de modelos).
- Hugging Face Inference Providers: el plan gratis da menos de $0.10/mes en
  créditos, insuficiente para uso real.
- GitHub Models: fue RETIRADO POR COMPLETO el 30 de julio de 2026 (confirmado
  en docs.github.com/en/github-models); cualquier llamada a su API regresa
  ahora un texto genérico "OK" sin generar nada.
- OpenRouter: capa gratuita confirmada vigente, pero su registro presentó
  fallos técnicos persistentes (error genérico incluso en modo incógnito).
- Cerebras Cloud: capa gratuita vigente, pero retiró los modelos Llama de
  su catálogo (ahora solo ofrece GPT-OSS y Qwen).
- SambaNova Cloud (actual): capa gratuita sin tarjeta (solo correo), con
  Llama 3.3 70B disponible de forma confirmada.
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
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
SAMBANOVA_CHAT_URL = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_MODELS_URL = "https://api.sambanova.ai/v1/models"

# Si el usuario fija LLAMA_MODEL explícitamente, esa variable tiene prioridad
# y se usa directamente sin consultar el catálogo.
LLAMA_MODEL_OVERRIDE = os.environ.get("LLAMA_MODEL")

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


def call_sambanova(model: str, messages: list[dict], max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Llama a la API REST de SambaNova Cloud y regresa el texto de la respuesta."""
    headers = {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(SAMBANOVA_CHAT_URL, headers=headers, json=payload, timeout=60)

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

    if "error" in data:
        raise RuntimeError(f"SambaNova regresó un error para '{model}': {data['error']}")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Formato de respuesta inesperado del modelo '{model}': {data}")


def discover_llama_model() -> str:
    """
    Consulta GET /v1/models de SambaNova y elige automáticamente un modelo
    Llama disponible en la cuenta. Esto evita tener que fijar un nombre de
    modelo a mano y que se rompa cuando el proveedor cambia su catálogo
    (ya pasó con Groq, GitHub Models y Cerebras en este mismo proyecto).
    """
    if LLAMA_MODEL_OVERRIDE:
        logger.info(f"Usando modelo fijado manualmente: {LLAMA_MODEL_OVERRIDE}")
        return LLAMA_MODEL_OVERRIDE

    headers = {"Authorization": f"Bearer {SAMBANOVA_API_KEY}"}
    response = requests.get(SAMBANOVA_MODELS_URL, headers=headers, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"No se pudo consultar el catálogo de modelos de SambaNova "
            f"(HTTP {response.status_code}): {response.text[:500]}"
        )

    try:
        data = response.json()
        model_ids = [m["id"] for m in data.get("data", [])]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"No se pudo interpretar el catálogo de modelos: {exc}")

    if not model_ids:
        raise RuntimeError("El catálogo de SambaNova regresó una lista de modelos vacía.")

    logger.info(f"Modelos disponibles en esta cuenta de SambaNova: {model_ids}")

    llama_models = [m for m in model_ids if "llama" in m.lower()]
    if not llama_models:
        raise RuntimeError(
            f"Esta cuenta de SambaNova no tiene ningún modelo Llama disponible. "
            f"Modelos que sí tiene: {model_ids}. Puedes fijar uno manualmente "
            f"con la variable de entorno LLAMA_MODEL."
        )

    # Preferir el modelo más grande (70b) si hay varios, por mejor calidad de respuesta.
    llama_models.sort(key=lambda m: "70b" not in m.lower())
    chosen = llama_models[0]

    # Verificación rápida de que el modelo elegido realmente responde.
    call_sambanova(chosen, [{"role": "user", "content": "hola"}], max_tokens=5)
    logger.info(f"Modelo Llama seleccionado (SambaNova): {chosen}")
    return chosen


@app.on_event("startup")
def startup():
    global rag_index, LLAMA_MODEL
    if not SAMBANOVA_API_KEY:
        logger.warning("SAMBANOVA_API_KEY no está configurada. El endpoint /ask fallará hasta configurarla.")
    else:
        LLAMA_MODEL = discover_llama_model()
    rag_index = RAGIndex(DATA_DIR)
    logger.info(f"RAG listo con {len(rag_index.chunks)} fragmentos de '{DATA_DIR}'.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(rag_index.chunks) if rag_index else 0,
        "sambanova_configured": SAMBANOVA_API_KEY is not None,
        "model": LLAMA_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if rag_index is None:
        raise HTTPException(status_code=503, detail="El índice RAG aún no está listo.")
    if not SAMBANOVA_API_KEY or not LLAMA_MODEL:
        raise HTTPException(status_code=500, detail="SAMBANOVA_API_KEY no configurada o ningún modelo disponible.")

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
        answer = call_sambanova(
            LLAMA_MODEL,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        logger.exception(f"Error llamando a SambaNova (tipo: {type(exc).__name__})")
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
