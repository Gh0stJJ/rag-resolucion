# utils.py
import re 

from typing import List, Dict, Any, Iterable
import dateparser
from datetime import datetime
import tiktoken
from settings import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_USE_SPACY, SPACY_MODEL, SPACY_MAX_LENGTH
import spacy
from functools import lru_cache

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
    match = re.match(r"^\d{4}-\d{2}-\d{2}$", fecha_txt)
    if match:
        try:
            dt = datetime.strptime(fecha_txt, "%Y-%m-%d")
        except Exception:
            return None, None, None
    else:
        dt = dateparser.parse(fecha_txt, languages=["es"])
        if not dt:
            return None, None, None
    fecha_iso = dt.strftime("%Y-%m-%d")
    legible = f"{dt.day} de {MONTHS_ES[dt.month]} de {dt.year}"
    return dt.date(), fecha_iso, legible



def _get_token_len_fn():
    """
    Return a function that estimates the number of tokens in a text.
    Fallback: 1 word = 1 token
    """
    try: 
        encod = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encod.encode(text))
    except Exception:
        pass
    # Using spaCy
    if CHUNK_USE_SPACY:
        try: 
            nlp = _get_spacy()
            return lambda s: sum(1 for t in nlp.make_doc(s) if not t.is_space)
        except Exception:
            pass
    # Fallback: 1 word = 1 token
    return lambda s: len(s.split())

        
_TOKEN_LEN = None


@lru_cache(maxsize=1)
def _get_spacy():
    try:
        nlp = spacy.load(
            SPACY_MODEL, 
            exclude=["tagger", "lemmatizer", "ner", "attribute_ruler", "morphologizer"]
        )

        # Use senter 
        try: 
            nlp.add_pipe("senter")
        except Exception:
            nlp.add_pipe("sentencizer")
    except Exception:
        # Fallback
        nlp = spacy.blank("es")
        nlp.add_pipe("sentencizer")
    nlp.max_length = max(nlp.max_length, SPACY_MAX_LENGTH)

    return nlp


_ABBR_LIST = {
    "Art.", "Arts.", "etc.", "Sr.", "Sra.", "Sres.", "Dr.", "Dra.", "Ing.", "Lic.",
    "No.", "N°.", "Nro.", "pág.", "pp.", "Cap.", "Sec."
}

def _split_sentences_spacy(text: str) -> List[str]:
    nlp = _get_spacy()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    # match if the previous one ends with a known abbreviation
    out: List[str] = []
    for s in sents:
        if out and any(out[-1].endswith(a) for a in _ABBR_LIST):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out

   

# Fallback sentence splitter (simple regex)
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


# Chunker sentence-aware
def _slice_by_tokens_words(words: List[str], max_tokens: int) -> List[str]:
    chunks= []
    cur, cur_tokens = [], 0
    for w in words:
        t = 1 #aprox
        if cur and cur_tokens + t > max_tokens:
            chunks.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(w)
        cur_tokens += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _build_overlap_from_history(history: List[str], overlap_tokens: int, token_len_fn) -> List[str]:
    if overlap_tokens <= 0 or not history:
        return []
    acc, total = [], 0
    for s in reversed(history):
        tlen = token_len_fn(s)
        if total + tlen > overlap_tokens and acc:
            break
        acc.append(s)
        total += tlen
    return list(reversed(acc))


def chunk_text_sentence_tokens(
    text: str,
    max_tokens: int = None,
    overlap_tokens: int = None
) -> List[str]:
    global _TOKEN_LEN
    if _TOKEN_LEN is None:
        _TOKEN_LEN = _get_token_len_fn()
    if max_tokens is None:
        max_tokens = CHUNK_MAX_TOKENS
    if overlap_tokens is None:
        overlap_tokens = CHUNK_OVERLAP_TOKENS

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    if CHUNK_USE_SPACY:
        try:
            sentences = _split_sentences_spacy(text)
        except Exception:
            sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    else:
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)

    sentences = [s for s in sentences if s]

    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    sent_history: List[str] = []

    for sent in sentences:
        t = _TOKEN_LEN(sent)
        if t > max_tokens:
            # split by words
            long_parts = _slice_by_tokens_words(sent.split(), max_tokens)
            for lp in long_parts:
                lp_t = _TOKEN_LEN(lp)
                if cur and cur_tokens + lp_t > max_tokens:
                    chunks.append(" ".join(cur))
                    overlap = _build_overlap_from_history(sent_history, overlap_tokens, _TOKEN_LEN)
                    cur = overlap[:]
                    cur_tokens = _TOKEN_LEN(" ".join(cur)) if cur else 0
                    sent_history = cur[:]
                cur.append(lp)
                cur_tokens += lp_t
            continue

        if cur and cur_tokens + t > max_tokens:
            chunks.append(" ".join(cur))
            overlap = _build_overlap_from_history(sent_history, overlap_tokens, _TOKEN_LEN)
            cur = overlap[:]
            cur_tokens = _TOKEN_LEN(" ".join(cur)) if cur else 0
            sent_history = cur[:]

        cur.append(sent)
        cur_tokens += t
        sent_history = (sent_history + [sent])[-20:]

    if cur:
        chunks.append(" ".join(cur))
    return chunks

def to_chunks_from_arrays(doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Gen chunks from "considerando" and "resuelve" sections of the document using segmentation by sentences and token count.
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
            if not clean:
                continue

            header = f"[{id_reso}; {seccion}; {fecha_legible or fecha_iso or ''}; {tipo}] "
            for cidx, ch in enumerate(chunk_text_sentence_tokens(clean, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)):
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
                    "texto": header + ch,
                }

ID_RE = re.compile(r"\bUC-CU-RES-\d{3}-\d{4}\b", re.IGNORECASE)


def extract_id_reso(query: str) -> str | None:
    m = ID_RE.search(query or "")
    return m.group(0).upper() if m else None
