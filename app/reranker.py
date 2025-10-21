#reranker.py

from openai import OpenAI

from typing import List, Tuple
import numpy as np
from settings import RERANKER_BASE_URL, RERANKER_MODEL, LLM_API_KEY

client = OpenAI(api_key=LLM_API_KEY, base_url=RERANKER_BASE_URL)

def rerank(query: str, contexts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Re-rank the provided contexts based on their relevance to the query using a pre-trained model.

    Args:
        query (str): The user's query.
        contexts (List[Tuple[str, str]]): A list of tuples where each tuple contains (document_id, document_text).

    Returns:
        List[Tuple[str, str]]: The re-ranked list of contexts.
    """
    if not contexts:
        return []

    inputs = [
        { "role": "user", "content": f"Query: {query}\nDocument: {text}" }
        for _, text in contexts
    ]

    scores = []

    for i, inp in enumerate(inputs):
        try:
            response = client.chat.completions.create(
                model=RERANKER_MODEL,
                messages=[inp],
                max_tokens=1,
                temperature=0.0,
                n=1,
                stop=None,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text) if score_text.replace('.', '', 1).isdigit() else 0.0
            scores.append(score)
        except Exception as e:
            print(f"[Reranker error] Error processing input {i}: {e}")
            scores.append(0.0)

    ranked = sorted(zip([id for id, _ in contexts], scores), key=lambda x: x[1], reverse=True)
    return ranked

