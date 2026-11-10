"""Backward-compatible facade for ChromaDB indexing and RAG queries.

Import-time side effects have been removed. Prefer importing directly from
`services.embeddings` and `services.llm` in new code.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import CHROMA_COLLECTION_NAME, DEFAULT_RAG_RESULTS, load_settings
from services.embeddings import (
    create_gemini_embedding_function,
    get_chroma_client,
)
from services.embeddings import (
    index_nutrition_data as _index_nutrition_data,
)
from services.llm import answer_nutrition_question


def _resolve_embedding_function(
    embedding_function: Any | None = None,
    api_key: str | None = None,
) -> Any:
    """Resolve an embedding function lazily without crashing on module import."""
    if embedding_function is not None:
        return embedding_function
    resolved_key = api_key or load_settings(require_api_key=True).gemini_api_key
    assert resolved_key is not None
    return create_gemini_embedding_function(resolved_key)


def index_nutrition_data(
    df: pd.DataFrame,
    client: Any,
    embedding_function: Any | None = None,
    api_key: str | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> int:
    """Index nutrition descriptions into ChromaDB using lazy embedding initialization."""
    ef = _resolve_embedding_function(embedding_function=embedding_function, api_key=api_key)
    return _index_nutrition_data(
        df=df,
        client=client,
        embedding_function=ef,
        collection_name=collection_name,
    )


def rag_query(
    gemini_model: Any,
    query: str,
    client: Any,
    n_results: int = DEFAULT_RAG_RESULTS,
    embedding_function: Any | None = None,
    api_key: str | None = None,
) -> str:
    """Execute a RAG query against ChromaDB and return the generated answer string."""
    ef = _resolve_embedding_function(embedding_function=embedding_function, api_key=api_key)
    answer, _ = answer_nutrition_question(
        model=gemini_model,
        query=query,
        chroma_client=client,
        embedding_function=ef,
        n_results=n_results,
    )
    return answer


__all__ = [
    "get_chroma_client",
    "index_nutrition_data",
    "rag_query",
]
