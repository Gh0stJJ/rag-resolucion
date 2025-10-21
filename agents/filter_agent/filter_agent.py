#filter_agent.py
import json
import re
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict
from filter_agent.system_role import get_system_role_prompt



SYSTEM_ROLE = get_system_role_prompt()

# --- Modelo de Datos de Entrada ---
class PromtRequest(BaseModel):
    """Define la estructura de la solicitud que espera el endpoint."""
    promt: str
    max_tokens: int = 1000


async def get_structured_filters(request: PromtRequest, client: OpenAI) -> Dict:
    """
    Llama al LLM para obtener los filtros y procesa la respuesta.
    Esta función contiene la lógica principal del agente.
    """
    try:
        completion = client.chat.completions.create(
            model="google/gemma-3-4b",
            messages=[
                {"role": "system", "content": f"{SYSTEM_ROLE}"},
                {"role": "user", "content": request.promt}
            ],
            temperature=0.1,
            max_tokens=request.max_tokens
        )
        raw_response = completion.choices[0].message.content.strip()
        
        match = re.search(r"\{[\s\S]*\}", raw_response)
                
        if match:
            #Si encuentra un bloque, intenta parsearlo.
            try:
                # Extrae solo el contenido del JSON encontrado
                json_string = match.group(0)
                return json.loads(json_string)
            except json.JSONDecodeError:
                # El bloque encontrado no era un JSON válido.
                return {"error": "Se encontró un bloque JSON pero no es válido.", "raw_response": raw_response}
        else:
            # 3. Si no encuentra ningún bloque que parezca JSON.
            return {"error": "No se encontró ningún objeto JSON en la respuesta del modelo.", "raw_response": raw_response}
    except Exception as e:
        # Captura errores de conexión u otros problemas con la API
        return {"error": f"No se pudo conectar con el servicio del LLM. Detalle: {str(e)}"}

