import re 

from typing import List, Dict, Any, Iterable
import dateparser


MONTHS_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}

def normalize_tipo(tipo: str) -> str:
    if not tipo: return ""
    t = re.sub(r"\s+", " ", tipo).strip()
    t = t.replace("Ordinaría", "Ordinaria")
    t = t.capitalize()
    MAP = {"ordinaria": "Ordinaria", "extraordinaria": "Extraordinaria"}
    return MAP.get(t.lower(), t)


def parse_fecha_es(fecha_txt: str):
    if not fecha_txt:
        return None, None, None
    dt = dateparser.parse(fecha_txt, languages=["es"])
    if not dt:
        return None, None, None
    fecha_iso = dt.strftime("%Y-%m-%d")
    legible = f"{dt.day} de {MONTHS_ES[dt.month]} de {dt.year}"
    return dt.date(), fecha_iso, legible

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def to_chunks_from_arrays(doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Genera chunks desde arrays de texto 'Considerando' y 'Resuelve' con metadatos.
    """
    id_reso = doc.get("id_reso")

    fecha_txt = doc.get("fecha")
    acta = doc.get("acta")
    tipo = normalize_tipo(doc.get("tipo", ""))

    _, fecha_iso, fecha_legible = parse_fecha_es(fecha_txt)

    anio = int(fecha_iso.split("-")[0]) if fecha_iso else None

    for seccion in ("considerando", "resuelve"):
        arr = doc.get(seccion) or []
        for idx, parrafo in enumerate(arr):
            if not parrafo or not isinstance(parrafo, str):
                continue
            clean = re.sub(r"SECRETARÍA GENERAL.*?(Aprobado por:.*)?", "", parrafo, flags=re.IGNORECASE|re.DOTALL)
            clean = re.sub(r"\s+", " ", clean).strip()

            for cidx, ch in enumerate(chunk_text(clean, max_chars=900, overlap=140)):
                yield {
                    "id_reso": id_reso,
                    "anio": anio,
                    "fecha_texto": fecha_txt,
                    "fecha_iso": fecha_iso,
                    "fecha_legible": fecha_legible,
                    "acta": acta,
                    "tipo": tipo,
                    "seccion": seccion,
                    "parrafo_index": idx,
                    "chunk_index": cidx,
                    "texto": ch,
                }
