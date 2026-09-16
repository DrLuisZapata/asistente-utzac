"""
test_e2e.py — 15 pruebas end-to-end del pipeline completo (RAG + Llama vía FastAPI).

Uso:
    export API_URL="https://tu-espacio.hf.space"   # o http://localhost:7860 en local
    python test_e2e.py

No requiere pytest: imprime un resumen con aciertos/fallos para incluir
como evidencia en tu documentación.
"""

import os
import sys
import requests

API_URL = os.environ.get("API_URL", "http://localhost:7860")

# Cada caso: (descripción, pregunta, validación)
# La validación es una función que recibe la respuesta JSON y regresa (ok: bool, motivo: str)

def contains_any(answer: str, keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(k.lower() in answer_lower for k in keywords)


CASES = [
    # --- Preguntas con respuesta real en el temario ---
    ("Pregunta directa cubierta por el temario",
     "¿Qué es una red neuronal artificial?",
     lambda r: (r["found_context"] is True, "Debía encontrar contexto")),

    ("Pregunta sobre evaluación del curso",
     "¿Cómo se calcula la calificación final del curso?",
     lambda r: (r["found_context"] is True, "Debía encontrar contexto")),

    ("Pregunta sobre función de activación",
     "¿Para qué sirve la función ReLU?",
     lambda r: (r["found_context"] is True, "Debía encontrar contexto")),

    ("Pregunta reformulada / sinónimos",
     "¿Cómo se llama el algoritmo que ajusta los pesos en una red neuronal?",
     lambda r: (r["found_context"] is True, "Debía encontrar contexto por similitud semántica")),

    ("Pregunta sobre métricas de evaluación de modelos",
     "¿Qué métricas se usan para evaluar un modelo de clasificación?",
     lambda r: (r["found_context"] is True, "Debía encontrar contexto")),

    # --- Casos borde: documento/tema que NO existe ---
    ("Tema totalmente fuera del temario",
     "¿Cuál es la capital de Australia?",
     lambda r: (r["found_context"] is False, "No debía encontrar contexto")),

    ("Pregunta sobre un tema de IA no cubierto en las notas",
     "Explícame cómo funciona un transformer con atención multi-cabeza",
     lambda r: (r["found_context"] is False, "No debía encontrar contexto (no está en el temario de ejemplo)")),

    ("Pregunta sobre otra materia",
     "¿Cuáles son las leyes de Newton?",
     lambda r: (r["found_context"] is False, "No debía encontrar contexto")),

    ("Documento/curso inexistente mencionado explícitamente",
     "¿Qué dice el capítulo 9 sobre redes generativas adversarias?",
     lambda r: (r["found_context"] is False, "No debía encontrar contexto")),

    # --- Robustez / entradas raras ---
    ("Pregunta vacía de contenido (solo saludo)",
     "hola",
     lambda r: (isinstance(r["answer"], str) and len(r["answer"]) > 0, "Debe responder algo, no crashear")),

    ("Pregunta muy larga",
     "Explícame con todo detalle, paso a paso, absolutamente todo lo que "
     "necesito saber sobre redes neuronales, backpropagation, funciones de "
     "activación, overfitting, y cómo se relaciona todo esto con el proyecto "
     "final del curso, dame ejemplos de cada concepto por favor",
     lambda r: (isinstance(r["answer"], str) and len(r["answer"]) > 0, "No debe fallar con preguntas largas")),

    ("Pregunta con errores ortográficos",
     "q es el obrefiting en redes neuronalez",
     lambda r: (isinstance(r["answer"], str) and len(r["answer"]) > 0, "Debe tolerar errores de ortografía")),

    ("Pregunta ambigua / poco específica",
     "¿me explicas el tema 2?",
     lambda r: (isinstance(r["answer"], str) and len(r["answer"]) > 0, "No debe fallar en silencio")),

    ("Verificar que no alucina cuando falta contexto",
     "¿Cuántos exámenes parciales de estadística hay en el curso de física?",
     lambda r: (r["found_context"] is False and contains_any(r["answer"], ["no encontré", "no encontre", "consúltalo", "profesor"]),
                "Debe usar el mensaje de fallback, no inventar")),

    ("Endpoint de salud responde correctamente",
     None,  # caso especial: prueba /health, no /ask
     lambda r: (r.get("status") == "ok" and r.get("chunks_loaded", 0) > 0, "Health check debe reportar chunks cargados")),
]


def run():
    passed, failed = 0, 0
    print(f"Probando endpoint: {API_URL}\n" + "=" * 60)

    for i, (description, question, validator) in enumerate(CASES, start=1):
        try:
            if question is None:  # caso de /health
                resp = requests.get(f"{API_URL}/health", timeout=30)
                resp.raise_for_status()
                data = resp.json()
            else:
                resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=30)
                resp.raise_for_status()
                data = resp.json()

            ok, reason = validator(data)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1

            print(f"[{i:02d}] {status} — {description}")
            if question:
                print(f"      Pregunta: {question[:70]}")
                print(f"      Respuesta: {str(data.get('answer', ''))[:100]}")
            if not ok:
                print(f"      Motivo del fallo: {reason}")

        except Exception as exc:
            failed += 1
            print(f"[{i:02d}] ERROR — {description}: {exc}")

    print("=" * 60)
    print(f"Resultado: {passed}/{len(CASES)} pruebas exitosas, {failed} fallidas")
    return failed == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
