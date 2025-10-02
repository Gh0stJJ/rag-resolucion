#multiquery.py

from openai import OpenAI
import json
from settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

def decompose_query_into_subqueries(query: str, n: int =3) -> list[str]:
    """
    Use an LLM to decompose a complex query into n simpler sub-queries.
    This subquestions are designed to retrieve complementary information from documents
    """

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    prompt = f"""
    Tu tarea es actuar como un motor de búsqueda experto. Descompón la siguiente "pregunta principal" en {n} "sub-preguntas" más pequeñas y específicas.

    El objetivo es que cada sub-pregunta busque una pieza de información concreta y fáctica que, en conjunto, ayude a responder la pregunta principal. Piensa en términos de "quién", "qué", "cuándo", "por qué", "cuál fue el resultado".

    Pregunta principal: "{query}"

    ### Ejemplo:
    Pregunta principal: "¿Cuál es el proceso y resultado de la anulación de matrícula de un estudiante por error administrativo?"
    Sub-preguntas:
    - "¿Cuál es el error administrativo que motivó la anulación de la matrícula?"
    - "¿Qué normativa o artículo se utilizó para justificar la anulación?"
    - "¿Qué órgano universitario tomó la decisión final de anular la matrícula?"
    - "¿Qué se decidió sobre el reembolso de los valores pagados por el estudiante?"

    Devuelve tu respuesta únicamente en formato JSON, con una sola clave "subqueries" que contenga la lista de strings de las sub-preguntas. No incluyas texto adicional ni explicaciones.
    """

    try: 
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en recuperación de información y análisis de documentos. Tu respuesta debe ser siempre un objeto JSON válido."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.5,
        )

        text = resp.choices[0].message.content.strip()

        data = json.loads(text)
        subqueries = data.get("subqueries", [])

        if isinstance(subqueries, list) and all(isinstance(sq, str) for sq in subqueries):
            return subqueries
        
        return [query]
    
    except Exception as e:
        print(f"[Decomposition error] {e}")
        return [query]
 