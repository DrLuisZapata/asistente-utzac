# Asistente de Curso UTZAC — Llama + RAG + Fine-tuning (LoRA)

Asistente que responde preguntas de alumnos usando el temario del curso,
combinando búsqueda por palabras clave (RAG) sobre las notas del profesor
con Llama 3.1 como modelo generativo.

## Etapas del pipeline y dónde vive cada una

| Etapa | Dónde | Archivo |
|---|---|---|
| Preparación de datos | Colab | `colab_finetune_lora.ipynb` (sección 3) |
| Fine-tuning (LoRA) | Colab (GPU T4 gratis) | `colab_finetune_lora.ipynb` (secciones 4-6) |
| Evaluación de salidas | Colab | `colab_finetune_lora.ipynb` (sección 7) |
| RAG (búsqueda TF-IDF) | Servidor | `rag_utils.py` |
| Endpoint (API) | Servidor | `app.py` |
| Despliegue | Render (Docker) | `Dockerfile` |
| Testing end-to-end | Local | `test_e2e.py` |

## Decisiones de arquitectura (y por qué cambiaron sobre la marcha)

**Fine-tuning vs. modelo servido en producción.** El fine-tuning con LoRA
(notebook de Colab) especializa a `meta-llama/Llama-3.1-8B-Instruct` en el
tono y formato de respuesta del curso, y el adaptador queda publicado en
Hugging Face Hub como evidencia de esa etapa. El **endpoint de producción**
usa el mismo modelo base, pero sin el adaptador aplicado, servido vía
**Hugging Face Inference** (no hay GPU propia disponible 24/7 para cargar
el adaptador). El componente que realmente aporta el conocimiento
específico del curso es el **RAG**, no el fine-tuning — el LoRA ajusta
estilo, el RAG aporta los datos actualizados.

**Por qué Hugging Face Inference y no Groq.** Groq fue la primera opción,
pero sus modelos Llama de propósito general (`llama-3.1-8b-instant`,
`llama-3.3-70b-versatile`, Llama 4 Scout/Maverick) están restringidos a
cuentas Enterprise — una cuenta gratis nueva no tiene acceso a ninguno
(verificado contra `GET /openai/v1/models` de Groq). Como ya se había
aceptado la licencia de Meta para `Llama-3.1-8B-Instruct` en Hugging Face
(paso previo al fine-tuning), se usa ese mismo canal para producción.

**Por qué TF-IDF y no embeddings neuronales.** La primera versión de
`rag_utils.py` usaba `sentence-transformers` (embeddings + FAISS), pero esa
librería carga PyTorch en memoria, lo cual excede los 512 MB del plan
gratis de Render. Se reemplazó por TF-IDF (scikit-learn) + similitud
coseno: mismo propósito (encontrar el fragmento más relevante), una
fracción de la memoria (~160 MB en total), sin descargar ningún modelo.

**Por qué Render y no Hugging Face Spaces.** Hugging Face cambió su
política y ahora requiere cuenta PRO de pago para crear Spaces que
ejecutan código (Docker/Gradio); solo los Spaces estáticos son gratis.
Render sí permite desplegar un contenedor Docker gratis con una URL
pública permanente.

## Estructura del proyecto

```
utzac-rag-llama/
├── app.py                       # Endpoint FastAPI (/ask, /health)
├── rag_utils.py                 # Carga de documentos, chunking, TF-IDF
├── colab_finetune_lora.ipynb    # Notebook de fine-tuning con LoRA
├── test_e2e.py                  # 15 pruebas end-to-end
├── requirements.txt
├── Dockerfile                   # Para desplegar en Render (o similar)
└── data/
    ├── reglamento_curso.txt     # Reglamento y metodología del Dr. Zapata
    └── ...                      # Un .txt/.md por cada materia
```

## Cómo correrlo en local

```bash
pip install -r requirements.txt
export HF_TOKEN="hf_tu_token_aqui"
uvicorn app:app --reload --port 7860
```

Prueba en el navegador: `http://localhost:7860/docs` (documentación
interactiva automática de FastAPI).

## Cómo desplegarlo en Render (para obtener tu URL pública)

1. Sube el proyecto a un repositorio de GitHub (todos los archivos, incluida
   la carpeta `data/` con tus notas reales).
2. Ve a render.com → New + → Web Service → conecta ese repositorio.
   Render detecta el `Dockerfile` automáticamente.
3. En "Instance Type" elige **Free**.
4. En "Environment Variables" agrega `HF_TOKEN` con tu token de Hugging
   Face (Settings → Access Tokens, con permiso de lectura basta).
5. Crea el servicio y espera 3-5 minutos a que compile. Tu URL será algo
   como `https://tu-servicio.onrender.com`.
6. Esa es la URL que le mandas a tu sensei. Puede probarla en
   `https://tu-servicio.onrender.com/docs` o hacer POST a `/ask`.

**Nota:** en el plan gratis, Render "duerme" el servicio tras 15 minutos
sin uso; la primera petición después de eso puede tardar 30-50 segundos.

## Cómo correr las pruebas end-to-end

```bash
export API_URL="https://tu-servicio.onrender.com"
python test_e2e.py
```

Cubre 15 casos: preguntas con respuesta en el reglamento del curso,
preguntas reformuladas, temas fuera de los documentos cargados (para
verificar que no alucina y responde con un mensaje claro en vez de fallar
en silencio), y entradas raras (saludos, texto muy largo, errores
ortográficos, preguntas ambiguas).

## Cómo agregar tu propio temario

Agrega uno o varios archivos `.txt`/`.md` en `data/`, uno por materia o
unidad. El sistema los divide automáticamente en fragmentos y no requiere
ningún cambio de código.

## Ejemplo de uso del endpoint

```bash
curl -X POST "https://tu-servicio.onrender.com/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cómo se compone la calificación final del curso?"}'
```

Respuesta:
```json
{
  "answer": "La calificación se compone de 20% por cada avance del proyecto (son 3) y 40% por la exposición final.",
  "found_context": true,
  "sources": ["reglamento_curso.txt"],
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "latency_seconds": 1.842
}
```
