# Asistente de Curso UTZAC — Llama + RAG + Fine-tuning (LoRA)

Asistente que responde preguntas de alumnos usando el temario del curso,
combinando búsqueda semántica (RAG) sobre las notas del profesor con
Llama 3 como modelo generativo.

## Etapas del pipeline y dónde vive cada una

| Etapa | Dónde | Archivo |
|---|---|---|
| Preparación de datos | Colab | `colab_finetune_lora.ipynb` (sección 3) |
| Fine-tuning (LoRA) | Colab (GPU T4 gratis) | `colab_finetune_lora.ipynb` (secciones 4-6) |
| Evaluación de salidas | Colab | `colab_finetune_lora.ipynb` (sección 7) |
| RAG (búsqueda semántica) | Servidor | `rag_utils.py` |
| Endpoint (API) | Servidor | `app.py` |
| Despliegue | Hugging Face Spaces | `Dockerfile` |
| Testing end-to-end | Local / CI | `test_e2e.py` |

## Decisión de arquitectura: ¿por qué el endpoint no usa el modelo fine-tuneado directamente?

El fine-tuning con LoRA (notebook de Colab) demuestra y ejercita esa etapa
del pipeline, especializando a Llama-3.1-8B-Instruct en el tono y formato
de respuesta del curso, y el adaptador queda publicado en Hugging Face
Hub como evidencia.

Sin embargo, el **endpoint de producción** (el que prueba tu sensei) llama a
Llama 3 alojado en **Groq**, por tres razones prácticas:

1. No hay GPU disponible 24/7 fuera de Colab para servir el modelo propio.
2. Groq es gratis, muy rápido, y no se cae cuando cierras tu laptop.
3. El componente que realmente diferencia la respuesta (qué información usa
   el modelo) es el **RAG**, no el fine-tuning — el LoRA mejora estilo, el
   RAG aporta el conocimiento específico y actualizado del curso.

Esto es una decisión de arquitectura común en proyectos reales: se entrena
y evalúa un modelo propio, pero se sirve con la infraestructura más
confiable disponible.

## Estructura del proyecto

```
utzac-rag-llama/
├── app.py                     # Endpoint FastAPI (/ask, /health)
├── rag_utils.py                # Carga de documentos, chunking, FAISS
├── colab_finetune_lora.ipynb   # Notebook de fine-tuning con LoRA
├── test_e2e.py                  # 15 pruebas end-to-end
├── requirements.txt
├── Dockerfile                   # Para desplegar en Hugging Face Spaces
└── data/
    └── temario_ejemplo.txt      # REEMPLAZA con tus notas reales (.txt o .md)
```

## Cómo correrlo en local

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_tu_clave_aqui"
uvicorn app:app --reload --port 7860
```

Prueba en el navegador: `http://localhost:7860/docs` (documentación
interactiva automática de FastAPI).

## Cómo desplegarlo en Hugging Face Spaces (para obtener tu URL pública)

1. Ve a huggingface.co → New Space.
2. Elige un nombre, licencia, y en "Space SDK" selecciona **Docker**.
3. Visibilidad: puede ser pública para que tu sensei la abra sin cuenta.
4. Sube TODOS los archivos de esta carpeta (incluyendo tu carpeta `data/`
   con tus notas reales) al repositorio del Space, ya sea arrastrando los
   archivos en la web o con `git push` (Spaces son repos git).
5. En el Space, ve a Settings → Variables and secrets → agrega
   `GROQ_API_KEY` como **secret** (no como variable pública).
6. Espera a que el Space compile (unos 2-3 minutos). Tu URL será algo como:
   `https://tu-usuario-tu-espacio.hf.space`
7. Esa es la URL que le mandas a tu sensei. Puede probarla directamente en
   `https://tu-usuario-tu-espacio.hf.space/docs` o hacer POST a `/ask`.

## Cómo correr las pruebas end-to-end

```bash
export API_URL="https://tu-usuario-tu-espacio.hf.space"
python test_e2e.py
```

Cubre 15 casos: preguntas con respuesta en el temario, preguntas
reformuladas, temas fuera del temario (para verificar que no alucina y
responde con un mensaje claro en vez de fallar en silencio), y entradas
raras (saludos, texto muy largo, errores ortográficos, preguntas ambiguas).

## Cómo agregar tu propio temario

Reemplaza `data/temario_ejemplo.txt` con uno o varios archivos `.txt`/`.md`
con tus notas reales, uno por unidad o tema. El sistema los divide
automáticamente en fragmentos y no requiere ningún cambio de código.

## Ejemplo de uso del endpoint

```bash
curl -X POST "https://tu-espacio.hf.space/ask" \\
  -H "Content-Type: application/json" \\
  -d '{"question": "¿Qué es una red neuronal artificial?"}'
```

Respuesta:
```json
{
  "answer": "Es un modelo computacional inspirado en el cerebro humano...",
  "found_context": true,
  "sources": ["temario_ejemplo.txt"],
  "model": "llama-3.1-8b-instant",
  "latency_seconds": 0.842
}
```
