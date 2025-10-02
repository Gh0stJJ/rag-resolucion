# hyde.py

from openai import OpenAI
from settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

def generate_hypothetical_document(query: str) -> str:
    """
    Use the LLM to generate a hypothetical document based on the query (HyDE).
    This document is quite similar to a legal fragment that would answer the query.
    """

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    prompt = f"""
    Actúa como un Secretario del Consejo Universitario de una universidad pública ecuatoriana.
    Tu tarea es generar un extracto del un documento hipotético que sea una resolución del Consejo Universitario, en respuesta a la siguiente pregunta.

    Pregunta: "{query}"

    El documento debe ser extremandamente realista y seguir estrictamente la estructura y el estilo del derecho administrativo ecuatoriano. Sigue estas reglas al pie de la letra:

    1. **Sección "CONSIDERANDO"**
        * Cada párrafo debe empezar obligatoriamente con la palabra "Que, ".
        * Los primeros párrafos deben citar fundamentos legales (puedes inventar los números de artículo, pero sé consistente). Cita, por ejemplo, la Ley Orgánica de Educación Superior, el Reglamento de Régimen Académico o el Estatuto de la Universidad.
        * Los párrafos siguientes deben describir los hechos que llevan a la resolución. Menciona documentos ficticios con códigos (ej. "Que, mediante Memorando Nro. UC-VICAC-2025-0123-M..."), fechas específicas y cargos (ej. "suscrito por el Decano de la Facultad...").

    3.  **Sección "RESUELVE:":**
        * Esta sección debe empezar exactamente con la palabra "RESUELVE:".
        * Enumera las decisiones tomadas de forma clara y directa, usando numerales (1., 2., 3.). Las resoluciones deben ser acciones concretas, como "DECLARAR NULA...", "DISPONER...", "NOTIFICAR...".

    4.  **Tono y Estilo:**
        * Utiliza un lenguaje formal, técnico y legalista en todo momento.
        * Sé impersonal y objetivo.
        * No uses un lenguaje conversacional.

    5.  **Contenido:** El contenido de los "CONSIDERANDO" y "RESUELVE" debe abordar directamente la pregunta del usuario.

    **Ejemplo de formato:**
    CONSIDERANDO:
    Que, el artículo 18 de la Ley Orgánica de Educación Superior dispone...
    Que, mediante oficio Nro. [código ficticio], de fecha [fecha ficticia], el [cargo ficticio] informa sobre...
    Que, es necesario atender la solicitud presentada por...

    RESUELVE:
    1.  Aprobar el informe técnico presentado por la Dirección de...
    2.  Disponer que el Departamento Financiero realice el ajuste presupuestario correspondiente.
    3.  Notificar con el contenido de la presente resolución a las partes interesadas.

    Genera ahora el extracto de la resolución basándote en la pregunta. Mantén una longitud aproximada de 150 a 250 palabras para asegurar el detalle necesario.
    """

    try: 
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[HyDE] Error generating hypothetical document: {e}")
        return query  # Fallback to returning the original query