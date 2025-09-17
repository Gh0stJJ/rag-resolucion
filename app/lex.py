#lex.py

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC, KEYWORD
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.scoring import BM25F
from whoosh.query import And, Term
def _schema():
    # Stemming
    analyzer = StemmingAnalyzer("spanish")
    return Schema(
        chunk_id=ID(stored=True, unique=True),
        id_reso=ID(stored=True),
        seccion=ID(stored=True),
        anio=NUMERIC(stored=True, numtype=int),
        tipo=KEYWORD(stored=True, lowercase=True, commas=False),
        fecha=ID(stored=True),
        texto=TEXT(stored=True, analyzer=analyzer)
    )

def open_or_create(index_dir: str):
    p = Path(index_dir); p.mkdir(parents=True, exist_ok=True)
    if index.exists_in(str(p)):
        return index.open_dir(str(p))
    return index.create_in(str(p), _schema())

def reset(index_dir: str):
    p = Path(index_dir)
    if p.exists():
        for f in p.glob("*"): 
            try: f.unlink()
            except: pass
    p.mkdir(parents=True, exist_ok=True)
    return index.create_in(str(p), _schema())

def build_chunk_id(d: Dict) -> str:
    return f"{d['id_reso']}|{d['seccion']}|p{int(d['parrafo_index'])}|c{int(d['chunk_index'])}"

def add_chunks(ix, docs: List[Dict]):
    writer = ix.writer(limitmb=256, procs=1, multisegment=True)

    for d in docs:
        chunk_id = build_chunk_id(d)
        base_text = str(d.get("texto") or "")
        injected = f"[{d.get('id_reso')}; {d.get('seccion')}; {d.get('fecha_legible') or d.get('fecha_iso') or ''}; {d.get('tipo')}] "
        
        writer.update_document(
            chunk_id=chunk_id,
            id_reso=str(d.get("id_reso") or ""),
            seccion=str(d.get("seccion") or ""),
            anio=int(d.get("anio") or 0),
            tipo=str(d.get("tipo") or ""),
            fecha=str(d.get("fecha_legible") or d.get("fecha_iso") or ""),
            texto=(injected + base_text)
        )
    writer.commit()

def search(ix, query: str, filtros: Optional[Dict]=None, limit: int=200) -> List[Tuple[str, float]]:
    """
    Devuelve [(chunk_id, score_bm25), ...] ordenado desc.
    Aplica filtros simples post-búsqueda (rápido y suficiente).
    """
    filtros = filtros or {}
    with ix.searcher(weighting=BM25F()) as s:
        parser = MultifieldParser(
            ["texto", "id_reso", "seccion", "tipo"],
            schema=ix.schema,
            group=OrGroup
        )
        
        q_text = parser.parse(query)
        if filtros.get("id_reso"):
            q = And([Term("id_reso", filtros["id_reso"]), q_text])
        else:
            q = q_text

        results = s.search(q, limit=limit)
        out: List[Tuple[str, float]] = []
        for r in results:
            # Filtros opcionales
            if "anio" in filtros and filtros["anio"]:
                try:
                    if int(r["anio"]) != int(filtros["anio"]): 
                        continue
                except: 
                    continue
            if "tipo" in filtros and filtros["tipo"]:
                if (r["tipo"] or "").lower() != str(filtros["tipo"]).lower():
                    continue
            if "id_reso" in filtros and filtros["id_reso"]:
                if r["id_reso"] != filtros["id_reso"]:
                    continue
            out.append((r["chunk_id"], float(r.score)))
        return out